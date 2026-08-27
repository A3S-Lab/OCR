use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};

use super::profile::PicodetLayoutProfile;

const MODEL_ENV: &str = "A3S_OCR_PICODET_LAYOUT_MODEL_DIR";

#[derive(Debug, Clone)]
pub(super) struct PicodetLayoutAssets {
    pub(super) profile: PicodetLayoutProfile,
    pub(super) root: PathBuf,
    pub(super) weights: PathBuf,
    pub(super) graph: Arc<str>,
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
        let weights = resolved_asset(&root, "model.safetensors")?;
        let metadata = std::fs::metadata(&weights).map_err(|error| {
            model_error(format!(
                "Failed to inspect PicoDet layout asset '{}': {error}",
                weights.display()
            ))
        })?;
        let actual_sha256 = file_sha256(&weights)?;
        let profile = PicodetLayoutProfile::from_weight_identity(metadata.len(), &actual_sha256)
            .ok_or_else(|| {
                model_error(format!(
                    "PicoDet layout asset '{}' is not an exact reviewed model artifact.",
                    weights.display()
                ))
                .with_detail("actualBytes", metadata.len())
                .with_detail("actualSha256", actual_sha256)
            })?;
        let graph = Arc::<str>::from(profile.embedded_graph());
        Ok(Self {
            profile,
            root,
            weights,
            graph,
        })
    }
}

fn resolved_asset(root: &Path, relative: &str) -> UseResult<PathBuf> {
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
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(model_error(format!(
            "PicoDet layout asset '{}' must be a non-empty regular file.",
            canonical.display()
        )));
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
        assert!(PicodetLayoutProfile::ALL.contains(&assets.profile));
    }

    #[test]
    fn incomplete_bundle_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let error = PicodetLayoutAssets::from_root(directory.path()).unwrap_err();
        assert_eq!(error.code, "use.ocr.seal_model_invalid");
    }

    #[test]
    fn unreviewed_weight_identity_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("model.safetensors"), b"unreviewed").unwrap();
        let error = PicodetLayoutAssets::from_root(directory.path()).unwrap_err();
        assert_eq!(error.code, "use.ocr.seal_model_invalid");
    }
}
