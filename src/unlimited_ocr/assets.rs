use std::path::{Path, PathBuf};

use a3s_power::inference::WeightStore;
use a3s_use_core::{UseError, UseResult};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(crate) const MODEL_REVISION: &str = "07dea832e22aefee32ad281d4b80551282e1c168";
pub(crate) const MODEL_WEIGHT_FILE: &str = "model-00001-of-000001.safetensors";
pub(crate) const MODEL_WEIGHT_BYTES: u64 = 6_672_547_120;
pub(crate) const MODEL_WEIGHT_SHA256: &str =
    "2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6";
const MODEL_TENSOR_COUNT: usize = 2_710;

const REVIEWED_FILES: [ReviewedFile; 5] = [
    ReviewedFile {
        relative: "config.json",
        max_bytes: 128 * 1024,
        sha256: "27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9",
    },
    ReviewedFile {
        relative: "tokenizer.json",
        max_bytes: 64 * 1024 * 1024,
        sha256: "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4",
    },
    ReviewedFile {
        relative: "tokenizer_config.json",
        max_bytes: 1024 * 1024,
        sha256: "a0cbe8464049da1f891b7a12676de06af4cb54c130995d42f71adc1c30c6e9f3",
    },
    ReviewedFile {
        relative: "special_tokens_map.json",
        max_bytes: 1024 * 1024,
        sha256: "ab4bd57ce17d62e39e0a39e739de1e407484f090f0b2c7e391312bca7a5b061a",
    },
    ReviewedFile {
        relative: "processor_config.json",
        max_bytes: 1024 * 1024,
        sha256: "92588cffb1d7032ec83d0a06c3a5171b41df5cbf432d68765441139a57899328",
    },
];

#[derive(Debug, Clone)]
pub(crate) struct ModelAssets {
    pub(crate) root: PathBuf,
    pub(crate) tokenizer: PathBuf,
}

#[derive(Clone, Copy)]
struct ReviewedFile {
    relative: &'static str,
    max_bytes: u64,
    sha256: &'static str,
}

pub(crate) fn inspect_assets(root: &Path) -> UseResult<ModelAssets> {
    let root = std::fs::canonicalize(root).map_err(|error| {
        model_missing(format!(
            "Failed to resolve the Unlimited-OCR model directory '{}': {error}",
            root.display()
        ))
    })?;
    if !std::fs::metadata(&root)
        .map(|metadata| metadata.is_dir())
        .unwrap_or(false)
    {
        return Err(model_invalid(format!(
            "Unlimited-OCR model root '{}' is not a directory.",
            root.display()
        )));
    }

    let weight = checked_file(&root, MODEL_WEIGHT_FILE, MODEL_WEIGHT_BYTES)?;
    let weight_bytes = std::fs::metadata(&weight)
        .map_err(|error| model_invalid(format!("Failed to inspect model weights: {error}")))?
        .len();
    if weight_bytes != MODEL_WEIGHT_BYTES {
        return Err(model_invalid(format!(
            "Unlimited-OCR weights must contain exactly {MODEL_WEIGHT_BYTES} bytes; found {weight_bytes}."
        )));
    }

    for reviewed in REVIEWED_FILES {
        let path = checked_file(&root, reviewed.relative, reviewed.max_bytes)?;
        verify_small_digest(&path, reviewed.sha256)?;
    }
    validate_reviewed_config(&root.join("config.json"))?;
    validate_reviewed_processor(&root.join("processor_config.json"))?;

    Ok(ModelAssets {
        tokenizer: root.join("tokenizer.json"),
        root,
    })
}

/// Bind Power's canonical store to the reviewed upstream checkpoint.
///
/// Power owns all full checkpoint hashing, including replica verification.
/// OCR compares the primary inventory with the model-owned revision pin
/// instead of hashing 6.7 GiB a second time.
pub(crate) fn verify_weight_store(store: &WeightStore, assets: &ModelAssets) -> UseResult<()> {
    if store.root() != assets.root {
        return Err(model_invalid(
            "A3S Power opened a different Unlimited-OCR primary weight root.",
        ));
    }
    let files = store.files();
    if files.len() != 1
        || files[0].relative_path != MODEL_WEIGHT_FILE
        || files[0].bytes != MODEL_WEIGHT_BYTES
        || files[0].sha256 != MODEL_WEIGHT_SHA256
    {
        return Err(UseError::new(
            "use.ocr.integrity_mismatch",
            "Unlimited-OCR weights do not match the reviewed upstream checkpoint.",
        )
        .with_detail("revision", MODEL_REVISION));
    }
    if store.inventory().len() != MODEL_TENSOR_COUNT {
        return Err(model_invalid(format!(
            "Unlimited-OCR checkpoint must contain {MODEL_TENSOR_COUNT} tensors; found {}.",
            store.inventory().len()
        )));
    }
    Ok(())
}

fn checked_file(root: &Path, relative: &str, max_bytes: u64) -> UseResult<PathBuf> {
    let requested = root.join(relative);
    let path = std::fs::canonicalize(&requested).map_err(|error| {
        model_missing(format!(
            "Required Unlimited-OCR asset '{}' is unreadable: {error}",
            requested.display()
        ))
    })?;
    if !path.starts_with(root) {
        return Err(model_invalid(format!(
            "Required Unlimited-OCR asset '{}' escapes its model directory.",
            requested.display()
        )));
    }
    let metadata = std::fs::metadata(&path).map_err(|error| {
        model_invalid(format!(
            "Failed to inspect Unlimited-OCR asset '{}': {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > max_bytes {
        return Err(model_invalid(format!(
            "Unlimited-OCR asset '{}' must be a non-empty regular file no larger than {max_bytes} bytes.",
            path.display()
        )));
    }
    Ok(path)
}

fn verify_small_digest(path: &Path, expected: &str) -> UseResult<()> {
    let bytes = std::fs::read(path).map_err(|error| {
        model_invalid(format!(
            "Failed to read Unlimited-OCR asset '{}': {error}",
            path.display()
        ))
    })?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(UseError::new(
            "use.ocr.integrity_mismatch",
            format!(
                "Unlimited-OCR asset '{}' failed its reviewed SHA-256 check.",
                path.display()
            ),
        )
        .with_detail("expected", expected)
        .with_detail("actual", actual));
    }
    Ok(())
}

fn validate_reviewed_config(path: &Path) -> UseResult<()> {
    let config: Value = serde_json::from_slice(&std::fs::read(path).map_err(|error| {
        model_invalid(format!(
            "Failed to read Unlimited-OCR config '{}': {error}",
            path.display()
        ))
    })?)
    .map_err(|error| model_invalid(format!("Invalid Unlimited-OCR config JSON: {error}")))?;
    let expected = [
        ("/model_type", Value::String("unlimited-ocr".to_string())),
        ("/hidden_size", Value::from(1_280)),
        ("/num_hidden_layers", Value::from(12)),
        ("/num_attention_heads", Value::from(10)),
        ("/n_routed_experts", Value::from(64)),
        ("/num_experts_per_tok", Value::from(6)),
        ("/n_shared_experts", Value::from(2)),
        ("/vocab_size", Value::from(129_280)),
        ("/max_position_embeddings", Value::from(32_768)),
        ("/sliding_window_size", Value::from(128)),
        ("/vision_config/image_size", Value::from(1_024)),
        ("/projector_config/input_dim", Value::from(2_048)),
        ("/projector_config/n_embed", Value::from(1_280)),
    ];
    for (pointer, value) in expected {
        if config.pointer(pointer) != Some(&value) {
            return Err(model_invalid(format!(
                "Unlimited-OCR config field '{pointer}' does not match the reviewed architecture."
            )));
        }
    }
    Ok(())
}

fn validate_reviewed_processor(path: &Path) -> UseResult<()> {
    let config: Value = serde_json::from_slice(&std::fs::read(path).map_err(|error| {
        model_invalid(format!(
            "Failed to read Unlimited-OCR processor config '{}': {error}",
            path.display()
        ))
    })?)
    .map_err(|error| {
        model_invalid(format!(
            "Invalid Unlimited-OCR processor config JSON: {error}"
        ))
    })?;
    let expected = [
        ("/patch_size", Value::from(16)),
        ("/downsample_ratio", Value::from(4)),
        ("/image_token", Value::String("<image>".to_string())),
        ("/normalize", Value::Bool(true)),
    ];
    for (pointer, value) in expected {
        if config.pointer(pointer) != Some(&value) {
            return Err(model_invalid(format!(
                "Unlimited-OCR processor field '{pointer}' does not match the reviewed preprocessing contract."
            )));
        }
    }
    Ok(())
}

fn model_missing(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.model_missing", message).with_suggestion(
        "Set A3S_UNLIMITED_OCR_MODEL_DIR to the reviewed local baidu/Unlimited-OCR checkpoint.",
    )
}

fn model_invalid(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.model_invalid", message)
        .with_detail("revision", MODEL_REVISION)
        .with_suggestion("Restore the exact reviewed baidu/Unlimited-OCR model assets.")
}
