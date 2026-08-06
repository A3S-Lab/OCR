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

const MODEL_BUNDLE: PinnedBundle = PinnedBundle {
    url: "https://github.com/A3S-Lab/OCR/releases/download/v0.3.0/a3s-use-ocr-assets-0.3.0.tar.gz",
    bytes: 26_105_899,
    sha256: "3376f84f400590c3c9c06ccef11494aac9877d7c19b8ffa38254e64db55c6d75",
};

#[derive(Debug, Clone, Copy)]
struct PinnedBundle {
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
    detection_url: String,
    #[serde(default)]
    detection_sha256: String,
    #[serde(default)]
    recognition_url: String,
    #[serde(default)]
    recognition_sha256: String,
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
    let bundle_path = stage.join("a3s-use-ocr-assets-0.3.0.tar.gz");
    let downloaded =
        download::pinned(&client, MODEL_BUNDLE.url, &bundle_path, MODEL_BUNDLE.bytes).await?;
    if downloaded.bytes != MODEL_BUNDLE.bytes || downloaded.sha256 != MODEL_BUNDLE.sha256 {
        return Err(ocr_error(
            "use.ocr.integrity_mismatch",
            format!(
                "PP-OCRv6 release bundle integrity mismatch: expected {} bytes and {}, got {} bytes and {}.",
                MODEL_BUNDLE.bytes,
                MODEL_BUNDLE.sha256,
                downloaded.bytes,
                downloaded.sha256
            ),
        ));
    }
    let bundle_path_for_task = bundle_path.clone();
    let destination = stage.to_path_buf();
    tokio::task::spawn_blocking(move || extract_bundle(&bundle_path_for_task, &destination))
        .await
        .map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!("PP-OCRv6 bundle extraction task failed: {error}"),
            )
        })??;
    tokio::fs::remove_file(&bundle_path)
        .await
        .map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Failed to remove staged bundle '{}': {error}",
                    bundle_path.display()
                ),
            )
        })?;
    write_receipt(stage).await?;
    validate_assets(stage, OcrInstallSource::Managed)?;
    Ok(())
}

fn extract_bundle(bundle_path: &Path, destination: &Path) -> UseResult<()> {
    let file = std::fs::File::open(bundle_path).map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to open PP-OCRv6 release bundle '{}': {error}",
                bundle_path.display()
            ),
        )
    })?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    let mut extracted = [false; 4];
    for entry in archive.entries().map_err(archive_error)? {
        let entry = entry.map_err(archive_error)?;
        let path = entry.path().map_err(archive_error)?.into_owned();
        let mut components = Vec::new();
        for component in path.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(value) => components.push(value),
                _ => return Err(unexpected_bundle_entry(&path)),
            }
        }
        if entry.header().entry_type().is_dir() && allowed_bundle_directory(&components) {
            continue;
        }
        if !entry.header().entry_type().is_file() {
            return Err(unexpected_bundle_entry(&path));
        }
        if ignored_bundle_file(&components) {
            if entry.size() == 0 || entry.size() > 2 * 1024 * 1024 {
                return Err(ocr_error(
                    "use.ocr.archive_invalid",
                    format!(
                        "PP-OCRv6 metadata entry '{}' has an invalid size.",
                        path.display()
                    ),
                ));
            }
            continue;
        }
        let Some((role, name, index, max)) = model_bundle_entry(&components) else {
            return Err(unexpected_bundle_entry(&path));
        };
        if extracted[index] {
            return Err(ocr_error(
                "use.ocr.archive_invalid",
                format!(
                    "PP-OCRv6 release bundle repeats entry '{}'.",
                    path.display()
                ),
            ));
        }
        extracted[index] = true;
        if entry.size() == 0 || entry.size() > max {
            return Err(ocr_error(
                "use.ocr.archive_invalid",
                format!("PP-OCRv6 release bundle entry '{name}' has an invalid size."),
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
                format!("PP-OCRv6 release bundle entry '{name}' was truncated."),
            ));
        }
        output.flush().map_err(archive_error)?;
        output.sync_all().map_err(archive_error)?;
    }
    if !extracted.into_iter().all(|present| present) {
        return Err(ocr_error(
            "use.ocr.archive_invalid",
            "PP-OCRv6 release bundle is missing a detection or recognition asset.",
        ));
    }
    Ok(())
}

fn allowed_bundle_directory(components: &[&std::ffi::OsStr]) -> bool {
    match components {
        [] => true,
        [root] => *root == "ocr-models" || *root == "ocr-skills",
        [root, model] => {
            (*root == "ocr-models" && *model == MODEL_FAMILY)
                || (*root == "ocr-skills" && *model == "a3s-use-ocr")
        }
        [root, model, role] => {
            *root == "ocr-models" && *model == MODEL_FAMILY && (*role == "det" || *role == "rec")
        }
        _ => false,
    }
}

fn ignored_bundle_file(components: &[&std::ffi::OsStr]) -> bool {
    match components {
        [name] => *name == "LICENSE" || *name == "THIRD_PARTY_NOTICES.md",
        [root, scope, name] => {
            (*root == "ocr-skills" && *scope == "a3s-use-ocr" && *name == "SKILL.md")
                || (*root == "ocr-models" && *scope == MODEL_FAMILY && *name == RECEIPT_FILE)
        }
        _ => false,
    }
}

fn model_bundle_entry(
    components: &[&std::ffi::OsStr],
) -> Option<(&'static str, &'static str, usize, u64)> {
    match components {
        [root, model, role, name] if *root == "ocr-models" && *model == MODEL_FAMILY => {
            match (*role, *name) {
                (role, name) if role == "det" && name == "inference.onnx" => {
                    Some(("det", "inference.onnx", 0, 256 * 1024 * 1024))
                }
                (role, name) if role == "det" && name == "inference.yml" => {
                    Some(("det", "inference.yml", 1, 2 * 1024 * 1024))
                }
                (role, name) if role == "rec" && name == "inference.onnx" => {
                    Some(("rec", "inference.onnx", 2, 256 * 1024 * 1024))
                }
                (role, name) if role == "rec" && name == "inference.yml" => {
                    Some(("rec", "inference.yml", 3, 2 * 1024 * 1024))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn unexpected_bundle_entry(path: &Path) -> UseError {
    ocr_error(
        "use.ocr.archive_invalid",
        format!(
            "PP-OCRv6 release bundle contains an unexpected entry '{}'.",
            path.display()
        ),
    )
}

async fn write_receipt(stage: &Path) -> UseResult<()> {
    let receipt = InstallReceipt {
        schema_version: 2,
        provider: "pp-ocr-v6".to_string(),
        model: MODEL_FAMILY.to_string(),
        bundle_url: MODEL_BUNDLE.url.to_string(),
        bundle_sha256: MODEL_BUNDLE.sha256.to_string(),
        detection_url: String::new(),
        detection_sha256: String::new(),
        recognition_url: String::new(),
        recognition_sha256: String::new(),
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
        receipt.provider == "pp-ocr-v6"
            && receipt.model == MODEL_FAMILY
            && match receipt.schema_version {
                1 => true,
                2 => {
                    receipt.bundle_url == MODEL_BUNDLE.url
                        && receipt.bundle_sha256 == MODEL_BUNDLE.sha256
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
mod download_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::net::{Shutdown, TcpListener};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    fn response(
        status: &str,
        declared_length: usize,
        extra_headers: &[(&str, String)],
        body: &[u8],
    ) -> Vec<u8> {
        let mut value = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {declared_length}\r\nConnection: close\r\n"
        );
        for (name, header_value) in extra_headers {
            value.push_str(&format!("{name}: {header_value}\r\n"));
        }
        value.push_str("\r\n");
        let mut value = value.into_bytes();
        value.extend_from_slice(body);
        value
    }

    fn scripted_server(
        responses: Vec<Vec<u8>>,
    ) -> (
        reqwest::Url,
        Arc<Mutex<Vec<String>>>,
        thread::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&requests);
        let server = thread::spawn(move || {
            for response in responses {
                let deadline = Instant::now() + Duration::from_secs(5);
                let (mut stream, _) = loop {
                    match listener.accept() {
                        Ok(connection) => break connection,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(Instant::now() < deadline, "timed out waiting for request");
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("failed to accept request: {error}"),
                    }
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 1024];
                while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                    let read = stream.read(&mut buffer).unwrap();
                    assert_ne!(read, 0, "request ended before its headers");
                    request.extend_from_slice(&buffer[..read]);
                }
                observed
                    .lock()
                    .unwrap()
                    .push(String::from_utf8(request).unwrap());
                stream.write_all(&response).unwrap();
                stream.shutdown(Shutdown::Both).unwrap();
            }
        });
        (
            reqwest::Url::parse(&format!("http://{address}/bundle.tar")).unwrap(),
            requests,
            server,
        )
    }

    #[tokio::test]
    async fn interrupted_download_resumes_from_the_verified_prefix() {
        let archive = b"immutable-model-archive";
        let split = 7;
        let content_range = format!("bytes {split}-{}/{}", archive.len() - 1, archive.len());
        let (url, requests, server) = scripted_server(vec![
            response("200 OK", archive.len(), &[], &archive[..split]),
            response(
                "206 Partial Content",
                archive.len() - split,
                &[("Content-Range", content_range)],
                &archive[split..],
            ),
        ]);
        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join("bundle.tar");
        let client = reqwest::Client::builder().build().unwrap();

        let downloaded = download::validated(
            &client,
            url,
            &destination,
            archive.len() as u64,
            Duration::ZERO,
        )
        .await
        .unwrap();

        server.join().unwrap();
        assert_eq!(std::fs::read(destination).unwrap(), archive);
        assert_eq!(downloaded.bytes, archive.len() as u64);
        assert_eq!(downloaded.sha256, format!("{:x}", Sha256::digest(archive)));
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(!requests[0].to_ascii_lowercase().contains("\r\nrange:"));
        assert!(requests[1]
            .to_ascii_lowercase()
            .contains(&format!("\r\nrange: bytes={split}-\r\n")));
    }

    #[tokio::test]
    async fn range_ignoring_retry_restarts_without_appending_duplicate_bytes() {
        let archive = b"immutable-model-archive";
        let split = 5;
        let (url, requests, server) = scripted_server(vec![
            response("200 OK", archive.len(), &[], &archive[..split]),
            response("200 OK", archive.len(), &[], archive),
        ]);
        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join("bundle.tar");
        let client = reqwest::Client::builder().build().unwrap();

        download::validated(
            &client,
            url,
            &destination,
            archive.len() as u64,
            Duration::ZERO,
        )
        .await
        .unwrap();

        server.join().unwrap();
        assert_eq!(std::fs::read(destination).unwrap(), archive);
        let requests = requests.lock().unwrap();
        assert!(requests[1]
            .to_ascii_lowercase()
            .contains(&format!("\r\nrange: bytes={split}-\r\n")));
    }

    #[tokio::test]
    async fn resumed_download_rejects_a_substituted_content_range() {
        let archive = b"immutable-model-archive";
        let split = 5;
        let substituted_range = format!("bytes 4-{}/{}", archive.len() - 1, archive.len());
        let (url, _requests, server) = scripted_server(vec![
            response("200 OK", archive.len(), &[], &archive[..split]),
            response(
                "206 Partial Content",
                archive.len() - split,
                &[("Content-Range", substituted_range)],
                &archive[split..],
            ),
        ]);
        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join("bundle.tar");
        let client = reqwest::Client::builder().build().unwrap();

        let result = download::validated(
            &client,
            url,
            &destination,
            archive.len() as u64,
            Duration::ZERO,
        )
        .await;

        server.join().unwrap();
        assert!(result.is_err());
        assert_eq!(std::fs::read(destination).unwrap(), &archive[..split]);
    }

    #[tokio::test]
    async fn retryable_origin_failures_recover_after_a_short_outage() {
        let archive = b"immutable-model-archive";
        let mut responses = (0..5)
            .map(|_| response("503 Service Unavailable", 0, &[], &[]))
            .collect::<Vec<_>>();
        responses.push(response("200 OK", archive.len(), &[], archive));
        let (url, requests, server) = scripted_server(responses);
        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join("bundle.tar");
        let client = reqwest::Client::builder().build().unwrap();

        download::validated(
            &client,
            url,
            &destination,
            archive.len() as u64,
            Duration::ZERO,
        )
        .await
        .unwrap();

        server.join().unwrap();
        assert_eq!(std::fs::read(destination).unwrap(), archive);
        assert_eq!(requests.lock().unwrap().len(), 6);
    }
}

#[cfg(test)]
mod bundle_tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Cursor;

    const MODEL_ENTRIES: [(&str, &[u8]); 4] = [
        ("ocr-models/PP-OCRv6_small/det/inference.onnx", b"det-model"),
        ("ocr-models/PP-OCRv6_small/det/inference.yml", b"det-config"),
        ("ocr-models/PP-OCRv6_small/rec/inference.onnx", b"rec-model"),
        ("ocr-models/PP-OCRv6_small/rec/inference.yml", b"rec-config"),
    ];

    fn append_file(builder: &mut tar::Builder<GzEncoder<std::fs::File>>, name: &str, bytes: &[u8]) {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Regular);
        header.set_mode(0o600);
        header.set_size(bytes.len() as u64);
        header.set_cksum();
        builder
            .append_data(&mut header, name, Cursor::new(bytes))
            .unwrap();
    }

    fn bundle_with(entries: &[(&str, &[u8])]) -> (tempfile::TempDir, PathBuf) {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("bundle.tar.gz");
        let encoder = GzEncoder::new(
            std::fs::File::create(&path).unwrap(),
            Compression::default(),
        );
        let mut builder = tar::Builder::new(encoder);
        append_file(&mut builder, "LICENSE", b"license");
        for (name, bytes) in entries {
            append_file(&mut builder, name, bytes);
        }
        builder.into_inner().unwrap().finish().unwrap();
        (temporary, path)
    }

    #[test]
    fn release_bundle_extracts_only_the_four_model_assets() {
        let (temporary, bundle) = bundle_with(&MODEL_ENTRIES);
        let destination = temporary.path().join("models");
        std::fs::create_dir(&destination).unwrap();

        extract_bundle(&bundle, &destination).unwrap();

        for (source, bytes) in MODEL_ENTRIES {
            let relative = source.strip_prefix("ocr-models/PP-OCRv6_small/").unwrap();
            assert_eq!(std::fs::read(destination.join(relative)).unwrap(), bytes);
        }
        assert!(!destination.join("LICENSE").exists());
    }

    #[test]
    fn release_bundle_rejects_missing_and_unexpected_entries() {
        let (temporary, bundle) = bundle_with(&MODEL_ENTRIES[..3]);
        let destination = temporary.path().join("missing");
        std::fs::create_dir(&destination).unwrap();
        assert!(extract_bundle(&bundle, &destination).is_err());

        let mut unexpected = MODEL_ENTRIES.to_vec();
        unexpected.push(("unexpected", b"content"));
        let (temporary, bundle) = bundle_with(&unexpected);
        let destination = temporary.path().join("unexpected");
        std::fs::create_dir(&destination).unwrap();
        assert!(extract_bundle(&bundle, &destination).is_err());
    }

    #[test]
    fn release_bundle_receipt_is_revision_bound_with_legacy_migration() {
        let temporary = tempfile::tempdir().unwrap();
        let receipt_path = temporary.path().join(RECEIPT_FILE);
        let mut receipt = InstallReceipt {
            schema_version: 2,
            provider: "pp-ocr-v6".to_string(),
            model: MODEL_FAMILY.to_string(),
            bundle_url: MODEL_BUNDLE.url.to_string(),
            bundle_sha256: MODEL_BUNDLE.sha256.to_string(),
            detection_url: String::new(),
            detection_sha256: String::new(),
            recognition_url: String::new(),
            recognition_sha256: String::new(),
        };
        std::fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(owned_install(temporary.path()));

        receipt.bundle_sha256 = "0".repeat(64);
        std::fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(!owned_install(temporary.path()));

        receipt.schema_version = 1;
        receipt.bundle_url.clear();
        receipt.bundle_sha256.clear();
        std::fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(owned_install(temporary.path()));
    }
}
