use std::collections::{BTreeMap, BTreeSet};

use super::*;

const MODEL_HEADER_BYTES: u64 = 334_632;
const MODEL_TENSOR_DATA_BYTES: u64 = 6_672_212_480;
const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";
const MODEL_INDEX_SHA256: &str = "354be1f2dcfb72ebb385e25465522ce5413a77c36f3b35fec088a3162a11af99";
const OFFICIAL_FIXTURE_ENV: &str = "A3S_UNLIMITED_OCR_OFFICIAL_FIXTURE";

#[test]
fn inventory_digest_contract_is_stable() {
    let mut descriptors = vec![
        ReviewedTensorDescriptor {
            name: "z.weight".to_string(),
            dtype: "bf16".to_string(),
            shape: vec![2, 3],
            bytes: 12,
        },
        ReviewedTensorDescriptor {
            name: "a.bias".to_string(),
            dtype: "f32".to_string(),
            shape: vec![4],
            bytes: 16,
        },
    ];
    assert_eq!(
        reviewed_inventory_sha256(&mut descriptors).unwrap(),
        "985e9dde56f192a7cc47c0334fd835e26cfcbd1d859577868a59e59ac4d02c99"
    );

    let baseline = reviewed_inventory_sha256(&mut descriptors.clone()).unwrap();
    let mutations: [fn(&mut ReviewedTensorDescriptor); 4] = [
        |descriptor: &mut ReviewedTensorDescriptor| descriptor.name.push_str(".changed"),
        |descriptor: &mut ReviewedTensorDescriptor| descriptor.dtype = "f16".to_string(),
        |descriptor: &mut ReviewedTensorDescriptor| descriptor.shape.push(1),
        |descriptor: &mut ReviewedTensorDescriptor| descriptor.bytes += 1,
    ];
    for mutation in mutations {
        let mut changed = descriptors.clone();
        mutation(&mut changed[0]);
        assert_ne!(reviewed_inventory_sha256(&mut changed).unwrap(), baseline);
    }
}

#[test]
#[ignore = "requires the complete SHA-256-pinned 6.7 GiB Unlimited-OCR checkpoint"]
fn complete_local_checkpoint_matches_reviewed_inventory() {
    let root = PathBuf::from(
        std::env::var_os("A3S_UNLIMITED_OCR_MODEL_DIR")
            .expect("A3S_UNLIMITED_OCR_MODEL_DIR must point to the complete checkpoint"),
    );
    let assets = inspect_assets(&root).unwrap();
    let store = WeightStore::open(
        &assets.root,
        &a3s_power::inference::InferenceLimits::default(),
    )
    .unwrap();
    verify_weight_store(&store, &assets).unwrap();
}

#[test]
#[ignore = "requires the SHA-256-pinned official Unlimited-OCR metadata and SafeTensors header"]
fn official_checkpoint_header_matches_reviewed_inventory() {
    let root = PathBuf::from(
        std::env::var_os(OFFICIAL_FIXTURE_ENV)
            .unwrap_or_else(|| panic!("{OFFICIAL_FIXTURE_ENV} must be set by the official gate")),
    );
    for reviewed in REVIEWED_FILES {
        let path = checked_file(&root, reviewed.relative, reviewed.max_bytes).unwrap();
        verify_small_digest(&path, reviewed.sha256).unwrap();
    }
    validate_reviewed_config(&root.join("config.json")).unwrap();
    validate_reviewed_processor(&root.join("processor_config.json")).unwrap();

    let header = std::fs::read(root.join("model.safetensors.header")).unwrap();
    let descriptors = parse_official_header(&header).unwrap();
    verify_reviewed_inventory(descriptors.clone()).unwrap();
    verify_official_index(&root.join(MODEL_INDEX_FILE), &descriptors).unwrap();
}

fn parse_official_header(bytes: &[u8]) -> UseResult<Vec<ReviewedTensorDescriptor>> {
    let prefix = bytes
        .get(..8)
        .ok_or_else(|| model_invalid("Unlimited-OCR SafeTensors header is truncated."))?;
    let header_bytes = u64::from_le_bytes(
        prefix
            .try_into()
            .map_err(|_| model_invalid("Unlimited-OCR SafeTensors header prefix is malformed."))?,
    );
    if header_bytes != MODEL_HEADER_BYTES {
        return Err(model_invalid(format!(
            "Unlimited-OCR SafeTensors header must contain {MODEL_HEADER_BYTES} JSON bytes; found {header_bytes}."
        )));
    }
    let expected_fixture_bytes = MODEL_HEADER_BYTES
        .checked_add(8)
        .ok_or_else(|| model_invalid("Unlimited-OCR header length overflowed."))?;
    if u64::try_from(bytes.len()).ok() != Some(expected_fixture_bytes) {
        return Err(model_invalid(format!(
            "Unlimited-OCR header fixture must contain {expected_fixture_bytes} bytes; found {}.",
            bytes.len()
        )));
    }
    let value: Value = serde_json::from_slice(&bytes[8..])
        .map_err(|error| model_invalid(format!("Invalid SafeTensors header JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| model_invalid("Unlimited-OCR SafeTensors header must be an object."))?;
    let metadata = object
        .get("__metadata__")
        .and_then(Value::as_object)
        .ok_or_else(|| model_invalid("Unlimited-OCR SafeTensors metadata is missing."))?;
    if metadata.len() != 1 || metadata.get("format").and_then(Value::as_str) != Some("pt") {
        return Err(model_invalid(
            "Unlimited-OCR SafeTensors metadata does not match the reviewed PyTorch container.",
        ));
    }

    let mut descriptors = Vec::with_capacity(object.len().saturating_sub(1));
    let mut spans = Vec::with_capacity(object.len().saturating_sub(1));
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor = value.as_object().ok_or_else(|| {
            model_invalid(format!("Unlimited-OCR tensor '{name}' is not an object."))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_invalid(format!("Tensor '{name}' has no dtype.")))?;
        if dtype != "BF16" {
            return Err(model_invalid(format!(
                "Tensor '{name}' uses unexpected dtype '{dtype}'."
            )));
        }
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_invalid(format!("Tensor '{name}' has no shape.")))?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .and_then(|dimension| usize::try_from(dimension).ok())
                    .ok_or_else(|| {
                        model_invalid(format!("Tensor '{name}' has an invalid dimension."))
                    })
            })
            .collect::<UseResult<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| model_invalid(format!("Tensor '{name}' has invalid offsets.")))?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| model_invalid(format!("Tensor '{name}' has invalid start.")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| model_invalid(format!("Tensor '{name}' has invalid end.")))?;
        let tensor_bytes = end
            .checked_sub(start)
            .ok_or_else(|| model_invalid(format!("Tensor '{name}' has reversed byte offsets.")))?;
        let elements = shape.iter().try_fold(1_u64, |product, dimension| {
            let dimension = u64::try_from(*dimension)
                .map_err(|_| model_invalid(format!("Tensor '{name}' is too large.")))?;
            product
                .checked_mul(dimension)
                .ok_or_else(|| model_invalid(format!("Tensor '{name}' size overflowed.")))
        })?;
        if elements.checked_mul(2) != Some(tensor_bytes) {
            return Err(model_invalid(format!(
                "Tensor '{name}' byte range does not match its BF16 shape."
            )));
        }
        spans.push((start, end, name.as_str()));
        descriptors.push(ReviewedTensorDescriptor {
            name: name.clone(),
            dtype: dtype.to_ascii_lowercase(),
            shape,
            bytes: tensor_bytes,
        });
    }
    spans.sort_by_key(|(start, _, _)| *start);
    let mut cursor = 0_u64;
    for (start, end, name) in spans {
        if start != cursor {
            return Err(model_invalid(format!(
                "Tensor '{name}' does not begin at the next canonical SafeTensors offset."
            )));
        }
        cursor = end;
    }
    if cursor != MODEL_TENSOR_DATA_BYTES
        || MODEL_HEADER_BYTES
            .checked_add(8)
            .and_then(|header| header.checked_add(cursor))
            != Some(MODEL_WEIGHT_BYTES)
    {
        return Err(model_invalid(
            "Unlimited-OCR tensor ranges do not cover the reviewed checkpoint exactly.",
        ));
    }
    Ok(descriptors)
}

fn verify_official_index(path: &Path, descriptors: &[ReviewedTensorDescriptor]) -> UseResult<()> {
    verify_small_digest(path, MODEL_INDEX_SHA256)?;
    let value: Value = serde_json::from_slice(&std::fs::read(path).map_err(|error| {
        model_invalid(format!("Failed to read the official model index: {error}"))
    })?)
    .map_err(|error| model_invalid(format!("Invalid official model index: {error}")))?;
    if value
        .pointer("/metadata/total_size")
        .and_then(Value::as_u64)
        != Some(MODEL_TENSOR_DATA_BYTES)
    {
        return Err(model_invalid(
            "Unlimited-OCR model index has an unexpected total tensor size.",
        ));
    }
    let weight_map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| model_invalid("Unlimited-OCR model index has no weight map."))?;
    if weight_map
        .values()
        .any(|file| file.as_str() != Some(MODEL_WEIGHT_FILE))
    {
        return Err(model_invalid(
            "Unlimited-OCR model index points outside the reviewed weight file.",
        ));
    }
    let indexed = weight_map
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let reviewed = descriptors
        .iter()
        .map(|descriptor| descriptor.name.as_str())
        .collect::<BTreeSet<_>>();
    if indexed != reviewed {
        return Err(model_invalid(
            "Unlimited-OCR model index and SafeTensors header inventories differ.",
        ));
    }

    let mut counts = BTreeMap::<&str, usize>::new();
    for descriptor in descriptors {
        *counts.entry(descriptor.dtype.as_str()).or_default() += 1;
    }
    if counts != BTreeMap::from([("bf16", MODEL_TENSOR_COUNT)]) {
        return Err(model_invalid(
            "Unlimited-OCR official tensor dtype inventory changed.",
        ));
    }
    Ok(())
}
