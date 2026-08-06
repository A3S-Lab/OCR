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

const DETECTION_ARCHIVE: PinnedArchive = PinnedArchive {
    role: "det",
    directory: "PP-OCRv6_small_det_onnx_infer",
    url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_det_onnx_infer.tar",
    bytes: 9_891_840,
    sha256: "d218f6fbf0f1c23d2161bd6ac7f5eaa6104fa89955c09290497e31008e2618e4",
};
const RECOGNITION_ARCHIVE: PinnedArchive = PinnedArchive {
    role: "rec",
    directory: "PP-OCRv6_small_rec_onnx_infer",
    url: "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_onnx_infer.tar",
    bytes: 21_319_680,
    sha256: "d267ab077a44a0eedb1ea8f8c542d263f211de8e9d7a029bf9fcfff7e5a88fb1",
};

#[derive(Debug, Clone, Copy)]
struct PinnedArchive {
    role: &'static str,
    directory: &'static str,
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
    detection_url: String,
    detection_sha256: String,
    recognition_url: String,
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
    for archive in [DETECTION_ARCHIVE, RECOGNITION_ARCHIVE] {
        let archive_path = stage.join(format!("{}.tar", archive.role));
        let downloaded =
            download::pinned(&client, archive.url, &archive_path, archive.bytes).await?;
        if downloaded.bytes != archive.bytes || downloaded.sha256 != archive.sha256 {
            return Err(ocr_error(
                "use.ocr.integrity_mismatch",
                format!(
                    "{} archive integrity mismatch: expected {} bytes and {}, got {} bytes and {}.",
                    archive.directory,
                    archive.bytes,
                    archive.sha256,
                    downloaded.bytes,
                    downloaded.sha256
                ),
            ));
        }
        let archive_path_for_task = archive_path.clone();
        let destination = stage.join(archive.role);
        tokio::task::spawn_blocking(move || {
            extract_archive(&archive_path_for_task, &destination, archive)
        })
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
    }
    write_receipt(stage).await?;
    validate_assets(stage, OcrInstallSource::Managed)?;
    Ok(())
}

fn extract_archive(archive_path: &Path, destination: &Path, spec: PinnedArchive) -> UseResult<()> {
    std::fs::create_dir(destination).map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to create PP-OCRv6 model directory '{}': {error}",
                destination.display()
            ),
        )
    })?;
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
    let mut extracted = [false; 2];
    for entry in archive.entries().map_err(archive_error)? {
        let entry = entry.map_err(archive_error)?;
        let path = entry.path().map_err(archive_error)?;
        let components = path.components().collect::<Vec<_>>();
        if components.len() == 1
            && matches!(components[0], Component::Normal(value) if value == spec.directory)
            && entry.header().entry_type().is_dir()
        {
            continue;
        }
        if components.len() != 2
            || !matches!(components[0], Component::Normal(value) if value == spec.directory)
            || !entry.header().entry_type().is_file()
        {
            return Err(ocr_error(
                "use.ocr.archive_invalid",
                format!(
                    "PP-OCRv6 archive contains an unexpected entry '{}'.",
                    path.display()
                ),
            ));
        }
        let name = match components[1] {
            Component::Normal(name) if name == "inference.onnx" => {
                extracted[0] = true;
                "inference.onnx"
            }
            Component::Normal(name) if name == "inference.yml" => {
                extracted[1] = true;
                "inference.yml"
            }
            _ => {
                return Err(ocr_error(
                    "use.ocr.archive_invalid",
                    format!(
                        "PP-OCRv6 archive contains an unexpected entry '{}'.",
                        path.display()
                    ),
                ))
            }
        };
        let max = if name.ends_with(".onnx") {
            256 * 1024 * 1024
        } else {
            2 * 1024 * 1024
        };
        if entry.size() == 0 || entry.size() > max {
            return Err(ocr_error(
                "use.ocr.archive_invalid",
                format!("PP-OCRv6 archive entry '{name}' has an invalid size."),
            ));
        }
        let expected_size = entry.size();
        let output_path = destination.join(name);
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
            "PP-OCRv6 archive is missing inference.onnx or inference.yml.",
        ));
    }
    Ok(())
}

async fn write_receipt(stage: &Path) -> UseResult<()> {
    let receipt = InstallReceipt {
        schema_version: 1,
        provider: "pp-ocr-v6".to_string(),
        model: MODEL_FAMILY.to_string(),
        detection_url: DETECTION_ARCHIVE.url.to_string(),
        detection_sha256: DETECTION_ARCHIVE.sha256.to_string(),
        recognition_url: RECOGNITION_ARCHIVE.url.to_string(),
        recognition_sha256: RECOGNITION_ARCHIVE.sha256.to_string(),
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
        receipt.schema_version == 1
            && receipt.provider == "pp-ocr-v6"
            && receipt.model == MODEL_FAMILY
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
