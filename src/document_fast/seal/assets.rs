use std::io::Read;
use std::path::{Path, PathBuf};

use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};

pub(super) const MODEL_FAMILY: &str = "picodet-l-layout-3cls";
pub(super) const MODEL_REVISION: &str = "paddleocr-paddle3-reviewed-v1";
pub(super) const SOURCE_GRAPH_SHA256: &str =
    "9df09659ed993444d068cc41b8b3e69306890b79c2af6f674d4111ab86e845da";
pub(super) const GRAPH_SHA256: &str =
    "6903f703d263e965d82bd0327f51dceb3f787ffff0b9411960a551f1f8119bd5";
pub(super) const WEIGHTS_FILE_SHA256: &str =
    "88c2d62f5ad48ff0487d0dc86e347f45ca369746cb8e0c8693ed9ecf1cb7fc9e";
pub(super) const WEIGHTS_COLLECTION_SHA256: &str =
    "361452be560223a2a4799026f92bf6ed2612f7a6d9abf8db4679a806e5eab965";

const WEIGHTS_BYTES: u64 = 23_361_700;
const MODEL_ENV: &str = "A3S_OCR_PICODET_LAYOUT_MODEL_DIR";

#[derive(Debug, Clone)]
pub(super) struct PicodetLayoutAssets {
    pub(super) root: PathBuf,
    pub(super) weights: PathBuf,
}

impl PicodetLayoutAssets {
    pub(super) fn from_env_optional() -> UseResult<Option<Self>> {
        let Some(root) = std::env::var_os(MODEL_ENV).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        Self::from_root(Path::new(&root)).map(Some)
    }

    #[cfg(test)]
    pub(super) fn from_env() -> UseResult<Self> {
        Self::from_env_optional()?.ok_or_else(|| {
            UseError::new(
                "use.ocr.seal_model_missing",
                "The pinned PicoDet layout model directory is not configured.",
            )
            .with_suggestion(format!(
                "Set {MODEL_ENV} to the reviewed local model bundle."
            ))
        })
    }

    pub(super) fn from_root(root: &Path) -> UseResult<Self> {
        let root = std::fs::canonicalize(root).map_err(|error| {
            model_error(format!(
                "Failed to resolve the PicoDet layout model directory '{}': {error}",
                root.display()
            ))
        })?;
        let weights = checked_asset(
            &root,
            "model.safetensors",
            WEIGHTS_BYTES,
            WEIGHTS_FILE_SHA256,
        )?;
        Ok(Self { root, weights })
    }
}

fn checked_asset(
    root: &Path,
    relative: &str,
    expected_bytes: u64,
    expected_sha256: &str,
) -> UseResult<PathBuf> {
    let requested = root.join(relative);
    let canonical = std::fs::canonicalize(&requested).map_err(|error| {
        model_error(format!(
            "Required PicoDet layout asset '{}' is unreadable: {error}",
            requested.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(model_error(format!(
            "Required PicoDet layout asset '{}' escapes its model directory.",
            requested.display()
        )));
    }
    let metadata = std::fs::metadata(&canonical).map_err(|error| {
        model_error(format!(
            "Failed to inspect PicoDet layout asset '{}': {error}",
            canonical.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() != expected_bytes {
        return Err(model_error(format!(
            "PicoDet layout asset '{}' must be a regular file of exactly {expected_bytes} bytes.",
            canonical.display()
        )));
    }
    let actual_sha256 = file_sha256(&canonical)?;
    if actual_sha256 != expected_sha256 {
        return Err(model_error(format!(
            "PicoDet layout asset '{}' failed its pinned SHA-256 check.",
            canonical.display()
        ))
        .with_detail("expectedSha256", expected_sha256)
        .with_detail("actualSha256", actual_sha256));
    }
    Ok(canonical)
}

fn file_sha256(path: &Path) -> UseResult<String> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        model_error(format!(
            "Failed to open PicoDet layout asset '{}': {error}",
            path.display()
        ))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            model_error(format!(
                "Failed to hash PicoDet layout asset '{}': {error}",
                path.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(super) fn model_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_model_invalid", message)
        .with_suggestion("Restore the exact reviewed PicoDet layout model bundle.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_reviewed_bundle_is_accepted_when_available() {
        let Some(root) = std::env::var_os(MODEL_ENV) else {
            return;
        };
        let assets = PicodetLayoutAssets::from_root(Path::new(&root)).unwrap();
        assert!(assets.weights.ends_with("model.safetensors"));
    }

    #[test]
    fn incomplete_bundle_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let error = PicodetLayoutAssets::from_root(directory.path()).unwrap_err();
        assert_eq!(error.code, "use.ocr.seal_model_invalid");
    }
}
