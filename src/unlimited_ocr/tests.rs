use a3s_use_core::Artifact;
use image::ImageEncoder as _;

use super::grounding::{GroundingGeometry, MAX_GROUNDING_BLOCK_TEXT_BYTES, MAX_GROUNDING_MARKERS};
use super::*;

#[test]
fn provider_configuration_is_local_lazy_and_model_typed() {
    let temporary = tempfile::tempdir().unwrap();
    let model_dir = temporary.path().join("reviewed-model");
    let config = UnlimitedOcrConfig::new(&model_dir)
        .unwrap()
        .with_max_generated_tokens(512)
        .unwrap();
    let provider = UnlimitedOcrProvider::new(config).unwrap();
    let descriptor = provider.descriptor();
    assert_eq!(descriptor.id, UNLIMITED_OCR_PROVIDER_ID);
    assert_eq!(descriptor.engine, ENGINE_NAME);
    assert!(!descriptor.sends_source_off_device);
    assert_eq!(provider.config().model(), UNLIMITED_OCR_MODEL);
    assert_eq!(provider.config().max_generated_tokens(), 512);
    assert_eq!(provider.diagnostic().readiness, Readiness::Missing);
    assert!(provider.loaded.lock().unwrap().is_none());
}

#[test]
fn embedded_provider_and_session_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<UnlimitedOcrProvider>();
    assert_send_sync::<UnlimitedOcrSession>();
}

#[test]
fn automatic_cpu_residency_uses_the_shared_power_runtime_limit() {
    let limits = a3s_power::inference::InferenceLimits {
        max_resident_weight_bytes: 2_048,
        ..a3s_power::inference::InferenceLimits::default()
    };
    let budget = a3s_power::inference::ResidencyBudgetPolicy::new(10_000, 0)
        .unwrap()
        .with_max_host_cache_bytes(1_024)
        .unwrap();
    let config = UnlimitedOcrConfig::new("fixture-model")
        .unwrap()
        .with_device(a3s_power::inference::DevicePreference::Cpu)
        .with_limits(limits)
        .unwrap()
        .with_residency_policy(a3s_power::inference::ResidencyPolicy {
            max_entries_per_layer: 7,
            ..a3s_power::inference::ResidencyPolicy::default()
        })
        .unwrap()
        .with_residency_budget_policy(budget)
        .unwrap();
    let runtime = EmbeddedRuntime::new(config.device, config.limits.clone()).unwrap();

    let residency = resolve_residency(&config, &runtime).unwrap();

    assert_eq!(residency.host_cache_bytes, 1_024);
    assert_eq!(residency.device_cache_bytes, 0);
    assert_eq!(residency.max_entries_per_layer, 7);
}

#[cfg(all(target_os = "macos", feature = "unlimited-ocr-metal"))]
#[test]
fn automatic_metal_residency_reuses_powers_unified_memory_plan() {
    let limits = a3s_power::inference::InferenceLimits {
        max_resident_weight_bytes: 2 * 1024 * 1024,
        ..a3s_power::inference::InferenceLimits::default()
    };
    let budget = a3s_power::inference::ResidencyBudgetPolicy::new(10_000, 10_000).unwrap();
    let config = UnlimitedOcrConfig::new("fixture-model")
        .unwrap()
        .with_device(a3s_power::inference::DevicePreference::Metal { ordinal: 0 })
        .with_limits(limits)
        .unwrap()
        .with_residency_budget_policy(budget.clone())
        .unwrap();
    let runtime = EmbeddedRuntime::new(config.device, config.limits.clone()).unwrap();
    let snapshot = runtime.memory_snapshot().unwrap();
    let expected = runtime
        .plan_residency_budget(&budget)
        .unwrap()
        .apply_to(&config.residency)
        .unwrap();

    let residency = resolve_residency(&config, &runtime).unwrap();

    assert!(snapshot.device.as_ref().unwrap().unified_with_host);
    assert_eq!(residency, expected);
    assert!(
        residency
            .host_cache_bytes
            .checked_add(residency.device_cache_bytes)
            .unwrap()
            <= config.limits.max_resident_weight_bytes
    );
}

#[test]
fn parses_reviewed_grounding_forms_into_source_pixel_blocks() {
    let raw = concat!(
        "<|ref|>title<|/ref|><|det|>[[0, 0, 999, 100]]<|/det|>",
        "# Heading\ncontinued\n",
        "<|det|>text [[100, 200, 400, 300], [500, 250, 900, 350]]<|/det|>",
        "Body \\coloneqq value",
        "<｜end▁of▁sentence｜>"
    );
    let parsed = parse_model_output(
        raw,
        Some(GroundingGeometry {
            width: 1_000,
            height: 500,
        }),
    )
    .unwrap();

    assert_eq!(parsed.text, "# Heading\ncontinued\nBody := value");
    assert!(parsed.warnings.is_empty());
    assert_eq!(parsed.blocks.len(), 2);
    assert_eq!(parsed.blocks[0].page, 1);
    assert_eq!(parsed.blocks[0].text, "# Heading\ncontinued");
    assert_eq!(
        parsed.blocks[0]
            .category
            .as_ref()
            .map(|category| (category.raw_label.as_str(), category.role)),
        Some(("title", crate::OcrBlockRole::Title))
    );
    assert_eq!(
        parsed.blocks[0].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 0,
            y: 0,
            width: 1_000,
            height: 50,
        })
    );
    assert_eq!(
        parsed.blocks[1].bounding_box,
        Some(crate::OcrBoundingBox {
            x: 100,
            y: 100,
            width: 800,
            height: 75,
        })
    );
    assert_eq!(
        parsed.blocks[1].bounding_boxes,
        [
            crate::OcrBoundingBox {
                x: 100,
                y: 100,
                width: 300,
                height: 50,
            },
            crate::OcrBoundingBox {
                x: 500,
                y: 125,
                width: 400,
                height: 50,
            },
        ]
    );
}

#[test]
fn preserves_open_labels_with_conservative_canonical_roles() {
    let raw = concat!(
        "<|det|>equation_isolated [0, 0, 100, 100]<|/det|>E = mc^2\n",
        "<|det|>figure_caption [0, 200, 100, 300]<|/det|>Figure 1\n",
        "<|det|>vendor-special [0, 400, 100, 500]<|/det|>Opaque"
    );
    let parsed = parse_model_output(
        raw,
        Some(GroundingGeometry {
            width: 999,
            height: 999,
        }),
    )
    .unwrap();

    assert!(parsed.warnings.is_empty());
    assert_eq!(
        parsed
            .blocks
            .iter()
            .map(|block| {
                let category = block.category.as_ref().unwrap();
                (category.raw_label.as_str(), category.role)
            })
            .collect::<Vec<_>>(),
        [
            ("equation_isolated", crate::OcrBlockRole::EquationBlock),
            ("figure_caption", crate::OcrBlockRole::Caption),
            ("vendor-special", crate::OcrBlockRole::Unknown),
        ]
    );
}

#[test]
fn malformed_grounding_never_fabricates_boxes() {
    let raw = concat!(
        "<|det|>image [0, 0, 999, 999]<|/det|>![figure](figure.jpg)\n",
        "<|det|>text [0, 0, 1000, 10]<|/det|>Out of range\n",
        "<|det|>text [10, 20, 10, 30]<|/det|>Collapsed\n",
        "Plain text"
    );
    let parsed = parse_model_output(
        raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap();
    assert!(parsed.blocks.is_empty());
    assert_eq!(parsed.warnings.len(), 1);

    let unclosed = parse_model_output(
        "text <|det|>unfinished",
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap();
    assert_eq!(unclosed.text, "text <|det|>unfinished");
    assert!(unclosed.blocks.is_empty());
}

#[test]
fn grounding_work_and_output_are_bounded() {
    let raw = "<|det|>text [0, 0, 1, 1]<|/det|>x".repeat(MAX_GROUNDING_MARKERS + 1);
    let error = parse_model_output(
        &raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap_err();
    assert_eq!(error.code, "use.ocr.provider_output_invalid");

    let boxes = (0..129)
        .map(|_| "[0, 0, 1, 1]")
        .collect::<Vec<_>>()
        .join(",");
    let raw = format!("<|det|>text [{boxes}]<|/det|>bounded");
    assert!(parse_model_output(
        &raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .is_err());

    let text = "x".repeat(MAX_GROUNDING_BLOCK_TEXT_BYTES + 1);
    let raw = format!("<|det|>text [0, 0, 999, 999]<|/det|>{text}");
    let parsed = parse_model_output(
        &raw,
        Some(GroundingGeometry {
            width: 100,
            height: 50,
        }),
    )
    .unwrap();
    assert_eq!(parsed.text, text);
    assert!(parsed.blocks.is_empty());
    assert_eq!(parsed.warnings.len(), 1);
}

#[test]
fn missing_or_orientation_transformed_geometry_is_explicitly_degraded() {
    let plain = parse_model_output(
        "plain text",
        Some(GroundingGeometry {
            width: 3,
            height: 2,
        }),
    )
    .unwrap();
    assert!(plain.blocks.is_empty());
    assert_eq!(plain.warnings.len(), 1);

    let raw = "<|det|>text [0, 0, 999, 999]<|/det|>grounded";
    let transformed = parse_model_output(raw, None).unwrap();
    assert_eq!(transformed.text, "grounded");
    assert!(transformed.blocks.is_empty());
    assert_eq!(transformed.warnings.len(), 1);

    let oriented_jpeg = fixture_oriented_jpeg(3, 2);
    assert_eq!(source_grounding_geometry(&oriented_jpeg).unwrap(), None);
    assert!(source_grounding_geometry(b"\x89PNG\r\n\x1a\ninvalid").is_err());
}

#[tokio::test]
async fn reviewed_checkpoint_runs_locally_when_configured() {
    let Some(root) = std::env::var_os("A3S_UNLIMITED_OCR_MODEL_DIR") else {
        return;
    };
    let config = UnlimitedOcrConfig::new(root)
        .unwrap()
        .with_max_generated_tokens(8)
        .unwrap();
    let provider = UnlimitedOcrProvider::new(config).unwrap();
    let image = fixture_png(64, 64);
    let input = OcrInput::new(
        Artifact {
            path: "/tmp/unlimited-ocr-fixture.png".into(),
            media_type: "image/png".to_string(),
            size: image.len() as u64,
            sha256: "test-source-digest".to_string(),
        },
        image,
    );
    let output = provider.recognize(input).await.unwrap();
    assert_eq!(output.model.as_deref(), Some(UNLIMITED_OCR_MODEL));
    assert_eq!(output.execution_receipts.len(), 1);
    let receipt = &output.execution_receipts[0];
    assert_eq!(receipt.model.family, UNLIMITED_OCR_MODEL);
    assert_eq!(receipt.model.revision, MODEL_REVISION);
    assert_eq!(receipt.input.representation, "image-request");
    assert_eq!(receipt.output.representation, "utf8-text");
}

fn fixture_png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(
            &vec![255; (width * height) as usize],
            width,
            height,
            image::ExtendedColorType::L8,
        )
        .unwrap();
    bytes
}

fn fixture_oriented_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
        .encode(
            &vec![255; (width * height * 3) as usize],
            width,
            height,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    let exif = [
        b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0,
        0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
    ];
    let segment_length = u16::try_from(exif.len() + 2).unwrap().to_be_bytes();
    let mut oriented = Vec::with_capacity(jpeg.len() + exif.len() + 4);
    oriented.extend_from_slice(&jpeg[..2]);
    oriented.extend_from_slice(&[0xff, 0xe1]);
    oriented.extend_from_slice(&segment_length);
    oriented.extend_from_slice(&exif);
    oriented.extend_from_slice(&jpeg[2..]);
    oriented
}
