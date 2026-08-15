use std::io::Read;
use std::path::{Path, PathBuf};

use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};

pub(super) const MODEL_FAMILY: &str = "slanet-plus-wired";
pub(super) const MODEL_REVISION: &str = "turboocr-models-v3.0.0-ppocrv6";
pub(super) const ENCODER_SOURCE_SHA256: &str =
    "dbd5431b4051b0f3037e3f8650dba4297cdf38a6a132ac9ccf57886184f4b66e";
pub(super) const ENCODER_FILE_SHA256: &str =
    "fe9a2eec8b6fe5303e7a263b51210d069806bd4caef5dc02b33bbcb9ce5b5098";
pub(super) const ENCODER_WEIGHTS_SHA256: &str =
    "1f032d592d5c710586888820f69f798ebeb65aa94288b726683eaff0ac5ca638";
pub(super) const DECODER_SHA256: &str =
    "f4b9f9b2d3fd5e85aebc839ebcd6bb5b27863077cc2b77c8e0a7ca01c6c0542d";
pub(super) const DICTIONARY_SHA256: &str =
    "68d344a84b726e043f390122240ff2b2ced2949b2a80ce9b61ae955054d190ef";

const ENCODER_BYTES: u64 = 5_502_136;
const DECODER_BYTES: u64 = 2_189_544;
const DICTIONARY_BYTES: u64 = 578;

#[derive(Debug, Clone)]
pub(super) struct SlanetPlusAssets {
    pub(super) root: PathBuf,
    pub(super) encoder_weights: PathBuf,
    pub(super) decoder_weights: PathBuf,
    pub(super) dictionary: PathBuf,
}

impl SlanetPlusAssets {
    pub(super) fn from_env() -> UseResult<Self> {
        let root = std::env::var_os("A3S_OCR_SLANET_PLUS_MODEL_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| {
                UseError::new(
                    "use.ocr.table_model_missing",
                    "The pinned SLANet-Plus wired-table model directory is not configured.",
                )
                .with_suggestion(
                    "Set A3S_OCR_SLANET_PLUS_MODEL_DIR to the reviewed local model bundle.",
                )
            })?;
        Self::from_root(&root)
    }

    pub(super) fn from_root(root: &Path) -> UseResult<Self> {
        let root = std::fs::canonicalize(root).map_err(|error| {
            model_error(format!(
                "Failed to resolve the SLANet-Plus model directory '{}': {error}",
                root.display()
            ))
        })?;
        let encoder_weights = checked_asset(
            &root,
            "encoder/model.safetensors",
            ENCODER_BYTES,
            ENCODER_FILE_SHA256,
        )?;
        let decoder_weights = checked_asset(
            &root,
            "slanext_wired_decoder.bin",
            DECODER_BYTES,
            DECODER_SHA256,
        )?;
        let dictionary = checked_asset(
            &root,
            "slanext_dict_infer.txt",
            DICTIONARY_BYTES,
            DICTIONARY_SHA256,
        )?;
        Ok(Self {
            root,
            encoder_weights,
            decoder_weights,
            dictionary,
        })
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
            "Required SLANet-Plus asset '{}' is unreadable: {error}",
            requested.display()
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(model_error(format!(
            "Required SLANet-Plus asset '{}' escapes its model directory.",
            requested.display()
        )));
    }
    let metadata = std::fs::metadata(&canonical).map_err(|error| {
        model_error(format!(
            "Failed to inspect SLANet-Plus asset '{}': {error}",
            canonical.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() != expected_bytes {
        return Err(model_error(format!(
            "SLANet-Plus asset '{}' must be a regular file of exactly {expected_bytes} bytes.",
            canonical.display()
        )));
    }
    let actual_sha256 = file_sha256(&canonical)?;
    if actual_sha256 != expected_sha256 {
        return Err(model_error(format!(
            "SLANet-Plus asset '{}' failed its pinned SHA-256 check.",
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
            "Failed to open SLANet-Plus asset '{}': {error}",
            path.display()
        ))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            model_error(format!(
                "Failed to hash SLANet-Plus asset '{}': {error}",
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
    UseError::new("use.ocr.table_model_invalid", message)
        .with_suggestion("Restore the exact reviewed SLANet-Plus wired-table model bundle.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_reviewed_bundle_is_accepted_when_available() {
        let Some(root) = std::env::var_os("A3S_OCR_SLANET_PLUS_MODEL_DIR") else {
            return;
        };
        let assets = SlanetPlusAssets::from_root(Path::new(&root)).unwrap();
        assert!(assets
            .encoder_weights
            .ends_with("encoder/model.safetensors"));
        assert!(assets
            .decoder_weights
            .ends_with("slanext_wired_decoder.bin"));
        assert!(assets.dictionary.ends_with("slanext_dict_infer.txt"));
    }

    #[test]
    fn symlink_escape_and_wrong_inventory_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let error = SlanetPlusAssets::from_root(directory.path()).unwrap_err();
        assert_eq!(error.code, "use.ocr.table_model_invalid");
    }
}
