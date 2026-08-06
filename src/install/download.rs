use std::path::Path;
use std::time::Duration;

use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use super::ocr_error;

const DOWNLOAD_HOST: &str = "paddle-model-ecology.bj.bcebos.com";
const DOWNLOAD_ATTEMPTS: usize = 8;
const DOWNLOAD_RETRY_DELAY: Duration = Duration::from_secs(2);
const DOWNLOAD_MAX_RETRY_DELAY: Duration = Duration::from_secs(30);
const MAX_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;

pub(super) struct Downloaded {
    pub(super) bytes: u64,
    pub(super) sha256: String,
}

pub(super) fn client() -> UseResult<reqwest::Client> {
    let redirects = reqwest::redirect::Policy::custom(|attempt| {
        let approved = attempt.previous().len() < 5
            && attempt.url().scheme() == "https"
            && attempt.url().host_str() == Some(DOWNLOAD_HOST);
        if approved {
            attempt.follow()
        } else {
            attempt.error("PP-OCRv6 download redirected to an unapproved host")
        }
    });
    reqwest::Client::builder()
        .user_agent(concat!("a3s-use-ocr/", env!("CARGO_PKG_VERSION")))
        .redirect(redirects)
        .connect_timeout(Duration::from_secs(30))
        // Bound stalled reads, not the complete transfer. The pinned model
        // archives can take more than five minutes on a healthy slow link.
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|error| {
            ocr_error(
                "use.ocr.download_failed",
                format!("Failed to create PP-OCRv6 download client: {error}"),
            )
        })
}

pub(super) async fn pinned(
    client: &reqwest::Client,
    value: &str,
    destination: &Path,
    expected_bytes: u64,
) -> UseResult<Downloaded> {
    let url = reqwest::Url::parse(value).map_err(|error| {
        ocr_error(
            "use.ocr.download_source_invalid",
            format!("Invalid PP-OCRv6 download URL: {error}"),
        )
    })?;
    if url.scheme() != "https" || url.host_str() != Some(DOWNLOAD_HOST) {
        return Err(ocr_error(
            "use.ocr.download_source_invalid",
            "PP-OCRv6 download source is not the pinned official HTTPS host.",
        ));
    }
    if expected_bytes == 0 || expected_bytes > MAX_ARCHIVE_BYTES {
        return Err(ocr_error(
            "use.ocr.download_too_large",
            "Pinned PP-OCRv6 archive length is outside the 256 MiB limit.",
        ));
    }
    validated(
        client,
        url,
        destination,
        expected_bytes,
        DOWNLOAD_RETRY_DELAY,
    )
    .await
}

pub(super) async fn validated(
    client: &reqwest::Client,
    url: reqwest::Url,
    destination: &Path,
    expected_bytes: u64,
    retry_delay: Duration,
) -> UseResult<Downloaded> {
    let mut file = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination)
        .await
        .map_err(|error| {
            ocr_error(
                "use.ocr.install_failed",
                format!(
                    "Failed to create PP-OCRv6 download '{}': {error}",
                    destination.display()
                ),
            )
        })?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut completed = false;

    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        if attempt > 1 && !retry_delay.is_zero() {
            let multiplier = 1_u32 << (attempt - 2);
            tokio::time::sleep(
                retry_delay
                    .saturating_mul(multiplier)
                    .min(DOWNLOAD_MAX_RETRY_DELAY),
            )
            .await;
        }

        let mut request = client
            .get(url.clone())
            .header(reqwest::header::ACCEPT_ENCODING, "identity");
        if total > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={total}-"));
        }
        let mut response = match request.send().await {
            Ok(response) => response,
            Err(_error) if attempt < DOWNLOAD_ATTEMPTS => continue,
            Err(error) => {
                return Err(attempts_exhausted(
                    attempt,
                    total,
                    describe_reqwest_error(&error),
                ));
            }
        };

        let status = response.status();
        if status == reqwest::StatusCode::OK {
            if total > 0 {
                file = tokio::fs::OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(destination)
                    .await
                    .map_err(|error| {
                        ocr_error(
                            "use.ocr.install_failed",
                            format!(
                                "Failed to restart PP-OCRv6 download '{}': {error}",
                                destination.display()
                            ),
                        )
                    })?;
                total = 0;
                hasher = Sha256::new();
            }
        } else if status == reqwest::StatusCode::PARTIAL_CONTENT {
            validate_content_range(response.headers(), total, expected_bytes)?;
        } else if retryable_status(status) && attempt < DOWNLOAD_ATTEMPTS {
            continue;
        } else {
            return Err(ocr_error(
                "use.ocr.download_failed",
                format!("PP-OCRv6 download failed with HTTP status {status}."),
            ));
        }

        if response.content_length().is_some_and(|length| {
            total
                .checked_add(length)
                .is_none_or(|response_end| response_end > expected_bytes)
        }) {
            return Err(ocr_error(
                "use.ocr.integrity_mismatch",
                "PP-OCRv6 response exceeds the pinned archive length.",
            ));
        }

        let mut interrupted = None;
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    total = total.checked_add(chunk.len() as u64).ok_or_else(|| {
                        ocr_error("use.ocr.download_too_large", "Download size overflowed.")
                    })?;
                    if total > expected_bytes || total > MAX_ARCHIVE_BYTES {
                        return Err(ocr_error(
                            "use.ocr.integrity_mismatch",
                            "PP-OCRv6 response exceeds the pinned archive length.",
                        ));
                    }
                    hasher.update(&chunk);
                    file.write_all(&chunk).await.map_err(|error| {
                        ocr_error(
                            "use.ocr.install_failed",
                            format!(
                                "Failed to write PP-OCRv6 download '{}': {error}",
                                destination.display()
                            ),
                        )
                    })?;
                    if total == expected_bytes {
                        completed = true;
                        break;
                    }
                }
                Ok(None) => {
                    interrupted = Some(format!(
                        "response ended at byte {total} before the pinned {expected_bytes}-byte length"
                    ));
                    break;
                }
                Err(error) => {
                    interrupted = Some(describe_reqwest_error(&error));
                    break;
                }
            }
        }
        if completed {
            break;
        }
        if attempt == DOWNLOAD_ATTEMPTS {
            return Err(attempts_exhausted(
                attempt,
                total,
                interrupted.unwrap_or_else(|| "response was interrupted".to_string()),
            ));
        }
    }

    if !completed {
        return Err(attempts_exhausted(
            DOWNLOAD_ATTEMPTS,
            total,
            "response was interrupted".to_string(),
        ));
    }
    file.flush().await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to flush PP-OCRv6 download '{}': {error}",
                destination.display()
            ),
        )
    })?;
    file.sync_all().await.map_err(|error| {
        ocr_error(
            "use.ocr.install_failed",
            format!(
                "Failed to sync PP-OCRv6 download '{}': {error}",
                destination.display()
            ),
        )
    })?;
    Ok(Downloaded {
        bytes: total,
        sha256: format!("{:x}", hasher.finalize()),
    })
}

fn retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn validate_content_range(
    headers: &reqwest::header::HeaderMap,
    expected_start: u64,
    expected_total: u64,
) -> UseResult<()> {
    let value = headers
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            ocr_error(
                "use.ocr.download_failed",
                "PP-OCRv6 range response omitted Content-Range.",
            )
        })?;
    let (range, total) = value
        .strip_prefix("bytes ")
        .and_then(|value| value.split_once('/'))
        .ok_or_else(|| {
            ocr_error(
                "use.ocr.download_failed",
                "PP-OCRv6 range response has an invalid Content-Range.",
            )
        })?;
    let (start, end) = range.split_once('-').ok_or_else(|| {
        ocr_error(
            "use.ocr.download_failed",
            "PP-OCRv6 range response has an invalid byte range.",
        )
    })?;
    let start = start.parse::<u64>().map_err(|_| {
        ocr_error(
            "use.ocr.download_failed",
            "PP-OCRv6 range response has an invalid start offset.",
        )
    })?;
    let end = end.parse::<u64>().map_err(|_| {
        ocr_error(
            "use.ocr.download_failed",
            "PP-OCRv6 range response has an invalid end offset.",
        )
    })?;
    let total = total.parse::<u64>().map_err(|_| {
        ocr_error(
            "use.ocr.download_failed",
            "PP-OCRv6 range response has an invalid total length.",
        )
    })?;
    if start != expected_start || end < start || end >= total || total != expected_total {
        return Err(ocr_error(
            "use.ocr.download_failed",
            format!(
                "PP-OCRv6 range response '{value}' does not continue byte {expected_start} of the pinned {expected_total}-byte archive."
            ),
        ));
    }
    Ok(())
}

fn describe_reqwest_error(error: &reqwest::Error) -> String {
    let mut message = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        let cause_message = cause.to_string();
        if !message.contains(&cause_message) {
            message.push_str(": ");
            message.push_str(&cause_message);
        }
        source = std::error::Error::source(cause);
    }
    message
}

fn attempts_exhausted(attempts: usize, offset: u64, cause: String) -> UseError {
    ocr_error(
        "use.ocr.download_failed",
        format!(
            "Failed to read PP-OCRv6 download after {attempts} attempts at byte {offset}: {cause}"
        ),
    )
}
