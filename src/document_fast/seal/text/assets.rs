use std::io::Read;
use std::path::{Path, PathBuf};

use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};

use super::profile::{GRAPH_SHA256, WEIGHTS_BYTES, WEIGHTS_FILE_SHA256};

const MODEL_ENV: &str = "A3S_OCR_SEAL_TEXT_MODEL_DIR";

#[derive(Debug, Clone)]
pub(super) struct SealTextAssets {
    pub(super) root: PathBuf,
    pub(super) graph: String,
    pub(super) weights: PathBuf,
}

impl SealTextAssets {
    pub(super) fn from_env_optional() -> UseResult<Option<Self>> {
        let Some(root) = std::env::var_os(MODEL_ENV).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        Self::from_root(Path::new(&root)).map(Some)
    }

    pub(super) fn from_root(root: &Path) -> UseResult<Self> {
        let root = std::fs::canonicalize(root).map_err(|error| {
            model_error(format!(
                "Failed to resolve the seal-text model directory '{}': {error}",
                root.display()
            ))
        })?;
        let graph_path = resolved_asset(&root, "graph.json")?;
        let weights = resolved_asset(&root, "model.safetensors")?;
        let graph_hash = file_sha256(&graph_path)?;
        if graph_hash != GRAPH_SHA256 {
            return Err(model_error(format!(
                "Seal-text graph digest is {graph_hash}, expected {GRAPH_SHA256}."
            )));
        }
        let metadata = std::fs::metadata(&weights).map_err(|error| {
            model_error(format!(
                "Failed to inspect seal-text weights '{}': {error}",
                weights.display()
            ))
        })?;
        let weights_hash = file_sha256(&weights)?;
        if metadata.len() != WEIGHTS_BYTES || weights_hash != WEIGHTS_FILE_SHA256 {
            return Err(model_error(
                "Seal-text weights do not match the exact reviewed artifact.",
            ));
        }
        let graph = std::fs::read_to_string(&graph_path).map_err(|error| {
            model_error(format!(
                "Failed to read seal-text graph '{}': {error}",
                graph_path.display()
            ))
        })?;
        Ok(Self {
            root,
            graph,
            weights,
        })
    }
}

fn resolved_asset(root: &Path, relative: &str) -> UseResult<PathBuf> {
    let requested = root.join(relative);
    let canonical = std::fs::canonicalize(&requested).map_err(|error| {
        model_error(format!(
            "Required seal-text asset '{}' is unreadable: {error}",
            requested.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(model_error(format!(
            "Required seal-text asset '{}' escapes its model directory.",
            requested.display()
        )));
    }
    let metadata = std::fs::metadata(&canonical).map_err(|error| {
        model_error(format!(
            "Failed to inspect seal-text asset '{}': {error}",
            canonical.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(model_error(format!(
            "Required seal-text asset '{}' must be a non-empty regular file.",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn file_sha256(path: &Path) -> UseResult<String> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        model_error(format!(
            "Failed to open seal-text asset '{}': {error}",
            path.display()
        ))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            model_error(format!(
                "Failed to hash seal-text asset '{}': {error}",
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
    UseError::new("use.ocr.seal_text_model_invalid", message)
        .with_suggestion("Restore the exact reviewed PP-OCRv4 seal-text model bundle.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document_fast::seal::text::profile::{
        FAMILY, ROLE, SOURCE_GRAPH_SHA256, WEIGHTS_COLLECTION_SHA256,
    };

    #[test]
    fn exact_reviewed_bundle_is_accepted_when_available() {
        let Some(root) = std::env::var_os(MODEL_ENV) else {
            return;
        };
        let assets = SealTextAssets::from_root(Path::new(&root)).unwrap();
        let graph: serde_json::Value = serde_json::from_str(&assets.graph).unwrap();
        assert_eq!(graph["family"], FAMILY);
        assert_eq!(graph["role"], ROLE);
        assert_eq!(graph["source"]["sha256"], SOURCE_GRAPH_SHA256);
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 526);
        assert_eq!(graph["initializers"].as_array().unwrap().len(), 246);
        assert!(!WEIGHTS_COLLECTION_SHA256.is_empty());
    }

    #[test]
    fn incomplete_bundle_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let error = SealTextAssets::from_root(directory.path()).unwrap_err();
        assert_eq!(error.code, "use.ocr.seal_text_model_invalid");
    }
}
