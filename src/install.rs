use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use a3s_use_core::{UseError, UseResult};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

mod download;

use crate::assets::{
    managed_model_dir, managed_root, ocr_status, validate_assets, OcrInstallSource,
    OcrRuntimeStatus, RECEIPT_FILE,
};
use crate::config::MODEL_FAMILY;

const INSTALL_LOCK: &str = ".install.lock";
const STAGE_PREFIX: &str = ".stage-";
const BACKUP_PREFIX: &str = ".backup-";

const NATIVE_ARCHIVE: PinnedArchive = PinnedArchive {
    url: "https://github.com/A3S-Lab/OCR/releases/download/ppocr-v6-paddlex-paddle3.0.0-native-v1/ppocr-v6-small-native-v1.tar",
    bytes: 31_074_816,
    sha256: "c5b040d7abe67ef8c144c493bcb6b38e79f902d1841224146fc5d0d3800921de",
};

#[derive(Debug, Clone, Copy)]
struct PinnedArchive {
    url: &'static str,
    bytes: u64,
    sha256: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InstallReceipt {
    schema_version: u32,
    provider: String,
    model: String,
    #[serde(default)]
    bundle_url: String,
    #[serde(default)]
    bundle_sha256: String,
    #[serde(default)]
    detection_weights_sha256: String,
    #[serde(default)]
    recognition_weights_sha256: String,
}

struct InstallLock {
    _file: std::fs::File,
}

pub async fn install_ppocr_v6(force: bool) -> UseResult<OcrRuntimeStatus> {
    let current = ocr_status();
    if !force && current.available {
        return Ok(current);
    }

    let root = managed_root()?;
    let _lock = acquire_lock(&root).await?;
    cleanup_stale(&root).await?;

    let current = ocr_status();
    if !force && current.available {
        return Ok(current);
    }

    let stage = create_stage(&root).await?;
    let install_result = install_into(&stage).await;
    if install_result.is_err() {
        let _ = tokio::fs::remove_dir_all(&stage).await;
    }
    install_result?;

    let target = managed_model_dir()?;
    activate(&stage, &target).await?;
    validate_assets(&target, OcrInstallSource::Managed)?;

    let status = ocr_status();
    if status.available {
        Ok(status)
    } else {
        Err(ocr_error(
            "use.ocr.install_failed",
            "PP-OCRv6 installation completed without a usable model bundle.",
        ))
    }
}

pub async fn repair_ppocr_v6() -> UseResult<OcrRuntimeStatus> {
    let status = ocr_status();
    if status.available {
        Ok(status)
    } else {
        install_ppocr_v6(true).await
    }
}

pub async fn uninstall_managed_ppocr_v6() -> UseResult<bool> {
    let root = managed_root()?;
    let _lock = acquire_lock(&root).await?;
    let target = managed_model_dir()?;
    if !owned_install(&target) {
        return Ok(false);
    }
    tokio::fs::remove_dir_all(&target).await.map_err(|error| {
        ocr_error(
            "use.ocr.uninstall_failed",
            format!(
                "Failed to remove managed PP-OCRv6 bundle '{}': {error}",
                target.display()
            ),
        )
    })?;
    Ok(true)
}

async fn install_into(stage: &Path) -> UseResult<()> {
    let client = download::client()?;
    let archive_path = stage.join("ppocr-v6-small-native-v1.tar");
    let downloaded = download::pinned(
        &client,
        NATIVE_ARCHIVE.url,
        &archive_path,
        NATIVE_ARCHIVE.bytes,
    )
    .await?;
    if downloaded.bytes != NATIVE_ARCHIVE.bytes || downloaded.sha256 != NATIVE_ARCHIVE.sha256 {
        return Err(ocr_error(
            "use.ocr.integrity_mismatch",
            format!(
                "Native PP-OCRv6 bundle integrity mismatch: expected {} bytes and {}, got {} bytes and {}.",
                NATIVE_ARCHIVE.bytes,
                NATIVE_ARCHIVE.sha256,
                downloaded.bytes,
                downloaded.sha256
            ),
        ));
    }
    let archive_path_for_task = archive_path.clone();
    let destination = stage.to_path_buf();
    tokio::task::spawn_blocking(move || extract_archive(&archive_path_for_task, &destination))
        .await
        .map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!("PP-OCRv6 archive extraction task failed: {error}"),
            )
        })??;
    tokio::fs::remove_file(&archive_path)
        .await
        .map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Failed to remove staged archive '{}': {error}",
                    archive_path.display()
                ),
            )
        })?;
    write_receipt(stage).await?;
    validate_assets(stage, OcrInstallSource::Managed)?;
    Ok(())
}

fn extract_archive(archive_path: &Path, destination: &Path) -> UseResult<()> {
    let file = std::fs::File::open(archive_path).map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to open PP-OCRv6 archive '{}': {error}",
                archive_path.display()
            ),
        )
    })?;
    let mut archive = tar::Archive::new(file);
    let mut extracted = [false; 4];
    for entry in archive.entries().map_err(archive_error)? {
        let entry = entry.map_err(archive_error)?;
        let path = entry.path().map_err(archive_error)?.into_owned();
        let components = path.components().collect::<Vec<_>>();
        if components.len() == 1 && entry.header().entry_type().is_dir() {
            let role = match components[0] {
                Component::Normal(value) if value == "det" => "det",
                Component::Normal(value) if value == "rec" => "rec",
                _ => return Err(unexpected_archive_entry(&path)),
            };
            let directory = destination.join(role);
            match std::fs::create_dir(&directory) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(ocr_error(
                        "use.ocr.install_failed",
                        format!(
                            "Failed to create PP-OCRv6 model directory '{}': {error}",
                            directory.display()
                        ),
                    ))
                }
            }
            continue;
        }
        if components.len() != 2 || !entry.header().entry_type().is_file() {
            return Err(unexpected_archive_entry(&path));
        }
        let role = match components[0] {
            Component::Normal(value) if value == "det" => "det",
            Component::Normal(value) if value == "rec" => "rec",
            _ => return Err(unexpected_archive_entry(&path)),
        };
        let (name, index, max) = match (role, components[1]) {
            ("det", Component::Normal(value)) if value == "model.safetensors" => {
                ("model.safetensors", 0, 64 * 1024 * 1024)
            }
            ("det", Component::Normal(value)) if value == "inference.yml" => {
                ("inference.yml", 1, 2 * 1024 * 1024)
            }
            ("rec", Component::Normal(value)) if value == "model.safetensors" => {
                ("model.safetensors", 2, 64 * 1024 * 1024)
            }
            ("rec", Component::Normal(value)) if value == "inference.yml" => {
                ("inference.yml", 3, 2 * 1024 * 1024)
            }
            _ => return Err(unexpected_archive_entry(&path)),
        };
        if extracted[index] {
            return Err(ocr_error(
                "use.ocr.archive_invalid",
                format!("PP-OCRv6 archive repeats entry '{}'.", path.display()),
            ));
        }
        extracted[index] = true;
        if entry.size() == 0 || entry.size() > max {
            return Err(ocr_error(
                "use.ocr.archive_invalid",
                format!("PP-OCRv6 archive entry '{name}' has an invalid size."),
            ));
        }
        let expected_size = entry.size();
        let role_directory = destination.join(role);
        std::fs::create_dir_all(&role_directory).map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Failed to create PP-OCRv6 model directory '{}': {error}",
                    role_directory.display()
                ),
            )
        })?;
        let output_path = role_directory.join(name);
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&output_path)
            .map_err(|error| {
                ocr_error(
                    "use.ocr.install_failed",
                    format!(
                        "Failed to create PP-OCRv6 asset '{}': {error}",
                        output_path.display()
                    ),
                )
            })?;
        let copied = std::io::copy(&mut entry.take(max + 1), &mut output).map_err(archive_error)?;
        if copied != expected_size {
            return Err(ocr_error(
                "use.ocr.archive_invalid",
                format!("PP-OCRv6 archive entry '{name}' was truncated."),
            ));
        }
        output.flush().map_err(archive_error)?;
        output.sync_all().map_err(archive_error)?;
    }
    if !extracted.into_iter().all(|present| present) {
        return Err(ocr_error(
            "use.ocr.archive_invalid",
            "PP-OCRv6 archive is missing native SafeTensors weights or an inference config.",
        ));
    }
    Ok(())
}

fn unexpected_archive_entry(path: &Path) -> UseError {
    ocr_error(
        "use.ocr.archive_invalid",
        format!(
            "PP-OCRv6 archive contains an unexpected entry '{}'.",
            path.display()
        ),
    )
}

async fn write_receipt(stage: &Path) -> UseResult<()> {
    let receipt = InstallReceipt {
        schema_version: 2,
        provider: "pp-ocr-v6".to_string(),
        model: MODEL_FAMILY.to_string(),
        bundle_url: NATIVE_ARCHIVE.url.to_string(),
        bundle_sha256: NATIVE_ARCHIVE.sha256.to_string(),
        detection_weights_sha256: crate::ppocr_v6::native::DETECTION_WEIGHTS_SHA256.to_string(),
        recognition_weights_sha256: crate::ppocr_v6::native::RECOGNITION_WEIGHTS_SHA256.to_string(),
    };
    let bytes = serde_json::to_vec_pretty(&receipt).map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!("Failed to encode PP-OCRv6 install receipt: {error}"),
        )
    })?;
    let path = stage.join(RECEIPT_FILE);
    let mut file = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .await
        .map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Failed to create PP-OCRv6 receipt '{}': {error}",
                    path.display()
                ),
            )
        })?;
    file.write_all(&bytes).await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to write PP-OCRv6 receipt '{}': {error}",
                path.display()
            ),
        )
    })?;
    file.sync_all().await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to sync PP-OCRv6 receipt '{}': {error}",
                path.display()
            ),
        )
    })
}

async fn acquire_lock(root: &Path) -> UseResult<InstallLock> {
    tokio::fs::create_dir_all(root).await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to create OCR data root '{}': {error}",
                root.display()
            ),
        )
    })?;
    let path = root.join(INSTALL_LOCK);
    tokio::task::spawn_blocking(move || {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .map_err(|error| {
                ocr_error(
                    "use.ocr.install_failed",
                    format!(
                        "Failed to open OCR install lock '{}': {error}",
                        path.display()
                    ),
                )
            })?;
        file.lock_exclusive().map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Failed to acquire OCR install lock '{}': {error}",
                    path.display()
                ),
            )
        })?;
        Ok(InstallLock { _file: file })
    })
    .await
    .map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!("OCR install lock task failed: {error}"),
        )
    })?
}

async fn create_stage(root: &Path) -> UseResult<PathBuf> {
    static NEXT_STAGE: AtomicU64 = AtomicU64::new(1);
    for _ in 0..32 {
        let path = root.join(format!(
            "{STAGE_PREFIX}{}-{}",
            std::process::id(),
            NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
        ));
        match tokio::fs::create_dir(&path).await {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(ocr_error(
                    "use.ocr.install_failed",
                    format!(
                        "Failed to create OCR staging directory '{}': {error}",
                        path.display()
                    ),
                ))
            }
        }
    }
    Err(ocr_error(
        "use.ocr.install_failed",
        "Failed to allocate a unique OCR staging directory.",
    ))
}

async fn cleanup_stale(root: &Path) -> UseResult<()> {
    let mut entries = tokio::fs::read_dir(root).await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to inspect OCR data root '{}': {error}",
                root.display()
            ),
        )
    })?;
    while let Some(entry) = entries.next_entry().await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to inspect OCR data root '{}': {error}",
                root.display()
            ),
        )
    })? {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let is_owned_backup = name.starts_with(BACKUP_PREFIX) && owned_install(&entry.path());
        if name.starts_with(STAGE_PREFIX) || is_owned_backup {
            tokio::fs::remove_dir_all(entry.path())
                .await
                .map_err(|error| {
                    ocr_error(
                        "use.ocr.install_failed",
                        format!("Failed to remove stale OCR staging directory: {error}"),
                    )
                })?;
        }
    }
    Ok(())
}

async fn activate(stage: &Path, target: &Path) -> UseResult<()> {
    static NEXT_BACKUP: AtomicU64 = AtomicU64::new(1);
    let parent = target.parent().ok_or_else(|| {
        ocr_error(
            "use.ocr.install_failed",
            "OCR install target has no parent directory.",
        )
    })?;
    let backup = parent.join(format!(
        "{BACKUP_PREFIX}{}-{}",
        std::process::id(),
        NEXT_BACKUP.fetch_add(1, Ordering::Relaxed)
    ));
    let had_target = tokio::fs::try_exists(target).await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to inspect OCR install target '{}': {error}",
                target.display()
            ),
        )
    })?;
    if had_target {
        if !owned_install(target) {
            return Err(ocr_error(
                "use.ocr.install_target_unowned",
                format!(
                    "Refusing to replace unowned OCR model directory '{}'.",
                    target.display()
                ),
            ));
        }
        tokio::fs::rename(target, &backup).await.map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Failed to stage existing OCR install '{}': {error}",
                    target.display()
                ),
            )
        })?;
    }
    if let Err(error) = tokio::fs::rename(stage, target).await {
        if had_target {
            let _ = tokio::fs::rename(&backup, target).await;
        }
        return Err(ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to activate OCR install '{}': {error}",
                target.display()
            ),
        ));
    }
    if had_target {
        tokio::fs::remove_dir_all(&backup).await.map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Activated OCR but failed to remove backup '{}': {error}",
                    backup.display()
                ),
            )
        })?;
    }
    Ok(())
}

fn owned_install(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path.join(RECEIPT_FILE)) else {
        return false;
    };
    serde_json::from_slice::<InstallReceipt>(&bytes).is_ok_and(|receipt| {
        if receipt.provider != "pp-ocr-v6" || receipt.model != MODEL_FAMILY {
            return false;
        }
        match receipt.schema_version {
            // Legacy ONNX installs are still recognized as owned so a forced
            // repair can transactionally replace them with native assets.
            1 => true,
            2 => {
                receipt.bundle_url == NATIVE_ARCHIVE.url
                    && receipt.bundle_sha256 == NATIVE_ARCHIVE.sha256
                    && receipt.detection_weights_sha256
                        == crate::ppocr_v6::native::DETECTION_WEIGHTS_SHA256
                    && receipt.recognition_weights_sha256
                        == crate::ppocr_v6::native::RECOGNITION_WEIGHTS_SHA256
            }
            _ => false,
        }
    })
}

fn archive_error(error: impl std::fmt::Display) -> UseError {
    ocr_error(
        "use.ocr.archive_invalid",
        format!("Failed to extract PP-OCRv6 archive: {error}"),
    )
}

fn ocr_error(code: &str, message: impl Into<String>) -> UseError {
    UseError::new(code, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::TempDir;

    const REQUIRED_ENTRIES: [(&str, &[u8]); 4] = [
        ("det/model.safetensors", b"det-weights"),
        ("det/inference.yml", b"det-config"),
        ("rec/model.safetensors", b"rec-weights"),
        ("rec/inference.yml", b"rec-config"),
    ];

    fn append_file(builder: &mut tar::Builder<std::fs::File>, name: &str, bytes: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o600);
        header.set_size(bytes.len() as u64);
        header.set_cksum();
        builder
            .append_data(&mut header, name, Cursor::new(bytes))
            .unwrap();
    }

    fn archive_with(entries: &[(&str, &[u8])]) -> (TempDir, PathBuf) {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("bundle.tar");
        let file = std::fs::File::create(&path).unwrap();
        let mut builder = tar::Builder::new(file);
        for (name, bytes) in entries {
            append_file(&mut builder, name, bytes);
        }
        builder.finish().unwrap();
        (temporary, path)
    }

    #[test]
    fn extracts_only_the_four_native_assets() {
        let (temporary, archive) = archive_with(&REQUIRED_ENTRIES);
        let destination = temporary.path().join("destination");
        std::fs::create_dir(&destination).unwrap();

        extract_archive(&archive, &destination).unwrap();

        let mut installed = Vec::new();
        for role in ["det", "rec"] {
            for entry in std::fs::read_dir(destination.join(role)).unwrap() {
                installed.push(
                    entry
                        .unwrap()
                        .path()
                        .strip_prefix(&destination)
                        .unwrap()
                        .to_path_buf(),
                );
            }
        }
        installed.sort();
        assert_eq!(
            installed,
            [
                PathBuf::from("det/inference.yml"),
                PathBuf::from("det/model.safetensors"),
                PathBuf::from("rec/inference.yml"),
                PathBuf::from("rec/model.safetensors"),
            ]
        );
    }

    #[test]
    fn rejects_unexpected_missing_and_duplicate_entries() {
        let mut unexpected = REQUIRED_ENTRIES.to_vec();
        unexpected.push(("det/inference.onnx", b"legacy-model"));
        let (temporary, archive) = archive_with(&unexpected);
        let destination = temporary.path().join("unexpected");
        std::fs::create_dir(&destination).unwrap();
        assert!(extract_archive(&archive, &destination).is_err());

        let (temporary, archive) = archive_with(&REQUIRED_ENTRIES[..3]);
        let destination = temporary.path().join("missing");
        std::fs::create_dir(&destination).unwrap();
        assert!(extract_archive(&archive, &destination).is_err());

        let mut duplicate = REQUIRED_ENTRIES.to_vec();
        duplicate.push(REQUIRED_ENTRIES[0]);
        let (temporary, archive) = archive_with(&duplicate);
        let destination = temporary.path().join("duplicate");
        std::fs::create_dir(&destination).unwrap();
        assert!(extract_archive(&archive, &destination).is_err());
    }

    #[test]
    fn rejects_non_regular_archive_entries() {
        let temporary = tempfile::tempdir().unwrap();
        let archive = temporary.path().join("symlink.tar");
        let file = std::fs::File::create(&archive).unwrap();
        let mut builder = tar::Builder::new(file);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_mode(0o777);
        header.set_size(0);
        header.set_link_name("/tmp/not-a-model").unwrap();
        header.set_cksum();
        builder
            .append_data(&mut header, "det/model.safetensors", Cursor::new([]))
            .unwrap();
        builder.finish().unwrap();
        let destination = temporary.path().join("destination");
        std::fs::create_dir(&destination).unwrap();

        assert!(extract_archive(&archive, &destination).is_err());
        assert!(!destination.join("det/model.safetensors").exists());
    }

    #[test]
    fn download_redirect_hosts_are_closed() {
        assert!(download::approved_host("github.com"));
        assert!(download::approved_host(
            "release-assets.githubusercontent.com"
        ));
        assert!(!download::approved_host("github.com.evil.example"));
        assert!(!download::approved_host("objects.githubusercontent.com"));
    }

    #[test]
    fn ownership_receipts_are_revision_bound_but_legacy_migration_is_allowed() {
        let temporary = tempfile::tempdir().unwrap();
        let receipt_path = temporary.path().join(RECEIPT_FILE);
        let current = InstallReceipt {
            schema_version: 2,
            provider: "pp-ocr-v6".to_string(),
            model: MODEL_FAMILY.to_string(),
            bundle_url: NATIVE_ARCHIVE.url.to_string(),
            bundle_sha256: NATIVE_ARCHIVE.sha256.to_string(),
            detection_weights_sha256: crate::ppocr_v6::native::DETECTION_WEIGHTS_SHA256.to_string(),
            recognition_weights_sha256: crate::ppocr_v6::native::RECOGNITION_WEIGHTS_SHA256
                .to_string(),
        };
        std::fs::write(&receipt_path, serde_json::to_vec(&current).unwrap()).unwrap();
        assert!(owned_install(temporary.path()));

        let mut tampered = current.clone();
        tampered.bundle_sha256 = "0".repeat(64);
        std::fs::write(&receipt_path, serde_json::to_vec(&tampered).unwrap()).unwrap();
        assert!(!owned_install(temporary.path()));

        let legacy = InstallReceipt {
            schema_version: 1,
            provider: "pp-ocr-v6".to_string(),
            model: MODEL_FAMILY.to_string(),
            bundle_url: String::new(),
            bundle_sha256: String::new(),
            detection_weights_sha256: String::new(),
            recognition_weights_sha256: String::new(),
        };
        std::fs::write(&receipt_path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert!(owned_install(temporary.path()));
    }
}
