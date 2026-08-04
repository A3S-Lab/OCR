use a3s_use_core::{UseError, UseResult};

use crate::{OcrBlock, OcrBoundingBox, OcrExecutionReceipt, OcrProviderOutput};

const MAX_BLOCK_CATEGORY_LABEL_BYTES: usize = 128;
pub(crate) const MAX_COMPONENT_BOXES_PER_BLOCK: usize = 128;

pub(crate) fn validate_provider_output(output: &OcrProviderOutput) -> UseResult<()> {
    if output.model.as_ref().is_some_and(|model| {
        model.is_empty() || model.len() > 256 || model.chars().any(char::is_control)
    }) {
        return Err(provider_output_error(
            "OCR provider model names must contain 1 through 256 control-free characters.",
        ));
    }
    for receipt in &output.execution_receipts {
        validate_execution_receipt(receipt)?;
    }
    for block in &output.blocks {
        if block.page == 0 {
            return Err(provider_output_error(
                "OCR providers must number pages starting at 1.",
            ));
        }
        for (name, value) in [
            ("confidence", block.confidence),
            ("detection confidence", block.detection_confidence),
        ] {
            if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
                return Err(provider_output_error(format!(
                    "OCR provider {name} must be between 0 and 1."
                )));
            }
        }
        validate_block_category(block)?;
        validate_block_geometry(block)?;
    }
    Ok(())
}

pub(crate) fn validate_execution_receipt(receipt: &OcrExecutionReceipt) -> UseResult<()> {
    for (label, value, maximum) in [
        ("receipt schema", receipt.schema.as_str(), 128),
        ("model family", receipt.model.family.as_str(), 256),
        ("model revision", receipt.model.revision.as_str(), 256),
        ("runtime name", receipt.runtime.name.as_str(), 128),
        ("runtime version", receipt.runtime.version.as_str(), 128),
        ("runtime device", receipt.runtime.device.as_str(), 128),
    ] {
        if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
            return Err(provider_output_error(format!(
                "OCR provider {label} must contain 1 through {maximum} control-free characters."
            )));
        }
    }
    for (label, value) in [
        ("model weights", receipt.model.weights_sha256.as_str()),
        ("input", receipt.input.sha256.as_str()),
        ("output", receipt.output.sha256.as_str()),
    ] {
        validate_sha256(value, label)?;
    }
    for representation in [
        receipt.input.representation.as_str(),
        receipt.output.representation.as_str(),
    ] {
        if !matches!(
            representation,
            "f32-tensor" | "image-request" | "token-ids" | "utf8-text"
        ) {
            return Err(provider_output_error(
                "OCR execution receipt representations must use a supported typed domain.",
            ));
        }
    }
    match &receipt.microbatch {
        Some(evidence) => {
            if receipt.schema != "a3s.power.embedded-execution-receipt.v4"
                || evidence.schema != "a3s.power.microbatch-execution.v1"
                || evidence.batch_count == 0
                || evidence.batch_index >= evidence.batch_count
                || evidence.slot_count == 0
            {
                return Err(provider_output_error(
                    "OCR microbatch receipt evidence has an invalid schema or shape.",
                ));
            }
            validate_sha256(&evidence.plan_sha256, "microbatch plan")?;
            if let Some(session) = &evidence.session_declaration_sha256 {
                validate_sha256(session, "microbatch session")?;
            }
        }
        None if receipt.schema == "a3s.power.embedded-execution-receipt.v4" => {
            return Err(provider_output_error(
                "OCR receipt v4 requires microbatch execution evidence.",
            ));
        }
        None => {}
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> UseResult<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(provider_output_error(format!(
            "OCR provider {label} must be a SHA-256 value."
        )));
    }
    Ok(())
}

fn validate_block_category(block: &OcrBlock) -> UseResult<()> {
    let Some(category) = &block.category else {
        return Ok(());
    };
    let label = category.raw_label.as_bytes();
    if label.is_empty()
        || label.len() > MAX_BLOCK_CATEGORY_LABEL_BYTES
        || !label
            .first()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        || !label
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(provider_output_error(
            "OCR provider block labels must contain 1 through 128 ASCII letters, digits, underscores, or hyphens and start with a letter or underscore.",
        ));
    }
    Ok(())
}

fn validate_block_geometry(block: &OcrBlock) -> UseResult<()> {
    if block.bounding_boxes.len() > MAX_COMPONENT_BOXES_PER_BLOCK {
        return Err(provider_output_error(
            "OCR provider blocks must not contain more than 128 component boxes.",
        ));
    }
    if let Some(bounds) = block.bounding_box {
        validate_box(bounds)?;
    }
    for bounds in &block.bounding_boxes {
        validate_box(*bounds)?;
    }
    if !block.bounding_boxes.is_empty()
        && block.bounding_box != component_envelope(&block.bounding_boxes)?
    {
        return Err(provider_output_error(
            "An OCR provider block bounding box must equal the envelope of its component boxes.",
        ));
    }
    Ok(())
}

fn validate_box(bounds: OcrBoundingBox) -> UseResult<()> {
    if bounds.width == 0
        || bounds.height == 0
        || bounds.x.checked_add(bounds.width).is_none()
        || bounds.y.checked_add(bounds.height).is_none()
    {
        return Err(provider_output_error(
            "OCR provider bounding boxes must cover a positive representable area.",
        ));
    }
    Ok(())
}

fn component_envelope(boxes: &[OcrBoundingBox]) -> UseResult<Option<OcrBoundingBox>> {
    let mut left = u32::MAX;
    let mut top = u32::MAX;
    let mut right = 0_u32;
    let mut bottom = 0_u32;
    for bounds in boxes {
        left = left.min(bounds.x);
        top = top.min(bounds.y);
        right =
            right.max(bounds.x.checked_add(bounds.width).ok_or_else(|| {
                provider_output_error("An OCR provider bounding box overflowed.")
            })?);
        bottom =
            bottom.max(bounds.y.checked_add(bounds.height).ok_or_else(|| {
                provider_output_error("An OCR provider bounding box overflowed.")
            })?);
    }
    if boxes.is_empty() {
        return Ok(None);
    }
    Ok(Some(OcrBoundingBox {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    }))
}

fn provider_output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_output_invalid", message)
}
