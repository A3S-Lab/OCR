use std::io::Read;
use std::path::{Path, PathBuf};

use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};

use super::profile::{GRAPH_SHA256, WEIGHTS_BYTES, WEIGHTS_FILE_SHA256};

const MODEL_ENV: &str = "A3S_OCR_DOCUMENT_LAYOUT_MODEL_DIR";

#[derive(Debug, Clone)]
pub(super) struct DocumentLayoutAssets {
    pub(super) root: PathBuf,
    pub(super) graph: String,
}

impl DocumentLayoutAssets {
    pub(super) fn from_env_optional() -> UseResult<Option<Self>> {
        let Some(root) = std::env::var_os(MODEL_ENV).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        Self::from_root(Path::new(&root)).map(Some)
    }

    pub(super) fn from_root(root: &Path) -> UseResult<Self> {
        let root = std::fs::canonicalize(root).map_err(|error| {
            model_error(format!(
                "Failed to resolve the document-layout model directory '{}': {error}",
                root.display()
            ))
        })?;
        if !root.is_dir() {
            return Err(model_error(
                "The document-layout model root must be a directory.",
            ));
        }
        let graph_path = checked_asset(&root, "graph.json", None, GRAPH_SHA256)?;
        checked_asset(
            &root,
            "model.safetensors",
            Some(WEIGHTS_BYTES),
            WEIGHTS_FILE_SHA256,
        )?;
        let graph = std::fs::read_to_string(&graph_path).map_err(|error| {
            model_error(format!(
                "Failed to read document-layout graph '{}': {error}",
                graph_path.display()
            ))
        })?;
        Ok(Self { root, graph })
    }

    pub(super) fn model_root(&self) -> &Path {
        &self.root
    }
}

fn checked_asset(
    root: &Path,
    relative: &str,
    expected_bytes: Option<u64>,
    expected_sha256: &str,
) -> UseResult<PathBuf> {
    let requested = root.join(relative);
    let canonical = std::fs::canonicalize(&requested).map_err(|error| {
        model_error(format!(
            "Required document-layout asset '{}' is unreadable: {error}",
            requested.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(model_error(format!(
            "Required document-layout asset '{}' escapes its model directory.",
            requested.display()
        )));
    }
    let metadata = std::fs::metadata(&canonical).map_err(|error| {
        model_error(format!(
            "Failed to inspect document-layout asset '{}': {error}",
            canonical.display()
        ))
    })?;
    if !metadata.is_file()
        || metadata.len() == 0
        || expected_bytes.is_some_and(|expected| metadata.len() != expected)
    {
        return Err(model_error(format!(
            "Document-layout asset '{}' has an invalid file identity.",
            canonical.display()
        )));
    }
    let actual_sha256 = file_sha256(&canonical)?;
    if actual_sha256 != expected_sha256 {
        return Err(model_error(format!(
            "Document-layout asset '{}' is not the exact reviewed artifact.",
            canonical.display()
        ))
        .with_detail("actualSha256", actual_sha256)
        .with_detail("expectedSha256", expected_sha256));
    }
    Ok(canonical)
}

fn file_sha256(path: &Path) -> UseResult<String> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        model_error(format!(
            "Failed to open document-layout asset '{}': {error}",
            path.display()
        ))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            model_error(format!(
                "Failed to hash document-layout asset '{}': {error}",
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
    UseError::new("use.ocr.document_layout_model_invalid", message)
        .with_suggestion("Restore the exact reviewed PP-DocLayout-S model bundle.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_and_unreviewed_bundles_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            DocumentLayoutAssets::from_root(directory.path())
                .unwrap_err()
                .code,
            "use.ocr.document_layout_model_invalid"
        );
        std::fs::write(directory.path().join("graph.json"), b"{}").unwrap();
        std::fs::write(directory.path().join("model.safetensors"), b"unreviewed").unwrap();
        assert_eq!(
            DocumentLayoutAssets::from_root(directory.path())
                .unwrap_err()
                .code,
            "use.ocr.document_layout_model_invalid"
        );
    }
}
