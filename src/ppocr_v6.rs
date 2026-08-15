mod batch;
pub(crate) mod native;

use std::sync::Mutex;

use a3s_power::inference::{DevicePreference, ModelSessionPool, ModelSessionPoolPolicy};
use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;

use crate::assets::{ocr_status, OcrInstallSource};
use crate::config::MODEL_FAMILY;
use crate::engine::{EngineExtraction, PpOcrV6Engine};
use crate::models::{OcrBlock, OcrBoundingBox, OcrPoint};
use crate::provider::{
    OcrInput, OcrProvider, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
};
use crate::receipt::project_receipt;
use crate::{OcrProviderBatchOutput, OcrProviderBatchRequest, OcrStage};

pub const PP_OCR_V6_PROVIDER_ID: &str = "pp-ocr-v6";
const ENGINE_NAME: &str = "a3s-power-native";

/// Local PP-OCRv6 provider shipped as the default A3S Use integration.
#[derive(Clone)]
pub struct PpOcrV6Provider {
    descriptor: OcrProviderDescriptor,
    sessions: ModelSessionPool<PpOcrV6Session>,
}

pub(super) struct PpOcrV6Session {
    engine: Mutex<PpOcrV6Engine>,
}

impl PpOcrV6Provider {
    pub fn from_env() -> UseResult<Self> {
        let policy = ModelSessionPoolPolicy::new(2, 1024 * 1024 * 1024, 1, 32)
            .map_err(|error| pool_error("configure", error))?;
        Ok(Self {
            descriptor: OcrProviderDescriptor::new(PP_OCR_V6_PROVIDER_ID, ENGINE_NAME, false)?
                .with_stages(vec![OcrStage::Preprocessing, OcrStage::Text])?,
            sessions: ModelSessionPool::new(DevicePreference::Auto, policy)
                .map_err(|error| pool_error("initialize", error))?,
        })
    }
}

#[async_trait]
impl OcrProvider for PpOcrV6Provider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        self.descriptor.clone()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        let status = ocr_status();
        let (readiness, suggestions) = if status.available {
            (Readiness::Ready, Vec::new())
        } else if status.source == OcrInstallSource::Missing {
            (
                Readiness::Missing,
                vec![
                    "Run 'a3s install use/ocr' to install the pinned local model bundle."
                        .to_string(),
                ],
            )
        } else {
            (
                Readiness::Broken,
                vec![
                    "Run 'a3s install use/ocr --force' to restore the pinned local model bundle."
                        .to_string(),
                ],
            )
        };
        OcrProviderStatus {
            readiness,
            model: Some(status.model),
            model_dir: status.model_dir,
            message: if status.available {
                "Local PP-OCRv6 detection and recognition models are ready.".to_string()
            } else {
                status.detail
            },
            suggestions,
        }
    }

    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
        batch::recognize_one(self, input).await
    }

    async fn recognize_batch(
        &self,
        request: OcrProviderBatchRequest,
    ) -> UseResult<OcrProviderBatchOutput> {
        batch::recognize_batch(self, request).await
    }
}

pub(super) fn build_output(extraction: EngineExtraction) -> UseResult<OcrProviderOutput> {
    let EngineExtraction { blocks, receipts } = extraction;
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
        let blank = blocks
            .iter()
            .filter(|block| block.text.trim().is_empty())
            .collect::<Vec<_>>();
        if !blank.is_empty() {
            let blank_detection_max = blank
                .iter()
                .map(|block| block.detection_confidence)
                .max_by(f32::total_cmp)
                .unwrap_or_default();
            let nonblank_detection_min = blocks
                .iter()
                .filter(|block| !block.text.trim().is_empty())
                .map(|block| block.detection_confidence)
                .min_by(f32::total_cmp);
            eprintln!(
                "A3S_OCR_BLANK_RECOGNITION blocks={} blank={} blank_detection_max={blank_detection_max:.6} nonblank_detection_min={nonblank_detection_min:?}",
                blocks.len(),
                blank.len(),
            );
        }
    }
    let blocks = blocks
        .into_iter()
        .filter(|block| !block.text.trim().is_empty())
        .map(|block| {
            let [first, second, third, fourth] = block.polygon;
            let polygon = [
                ocr_point(first)?,
                ocr_point(second)?,
                ocr_point(third)?,
                ocr_point(fourth)?,
            ];
            let min_x = polygon.iter().map(|point| point.x).min().unwrap_or(0);
            let max_x = polygon.iter().map(|point| point.x).max().unwrap_or(0);
            let min_y = polygon.iter().map(|point| point.y).min().unwrap_or(0);
            let max_y = polygon.iter().map(|point| point.y).max().unwrap_or(0);
            Ok(OcrBlock {
                page: 1,
                text: block.text,
                category: None,
                confidence: Some(block.confidence),
                detection_confidence: Some(block.detection_confidence),
                polygon: Some(polygon),
                bounding_box: Some(OcrBoundingBox {
                    x: min_x,
                    y: min_y,
                    width: max_x.saturating_sub(min_x),
                    height: max_y.saturating_sub(min_y),
                }),
                bounding_boxes: Vec::new(),
            })
        })
        .collect::<UseResult<Vec<_>>>()?;
    let text = blocks
        .iter()
        .filter(|block| !block.text.trim().is_empty())
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(OcrProviderOutput {
        model: Some(MODEL_FAMILY.to_string()),
        text,
        blocks,
        execution_receipts: receipts.into_iter().map(project_receipt).collect(),
        warnings: Vec::new(),
    })
}

fn pool_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} the PP-OCRv6 session pool: {error}"),
    )
}

fn ocr_point(point: imageproc::point::Point<f32>) -> UseResult<OcrPoint> {
    Ok(OcrPoint {
        x: finite_coordinate(point.x)?,
        y: finite_coordinate(point.y)?,
    })
}

fn finite_coordinate(value: f32) -> UseResult<u32> {
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f32 {
        return Err(UseError::new(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 returned an invalid polygon coordinate.",
        ));
    }
    Ok(value.round() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_output_omits_blank_recognition_blocks() {
        let polygon = [
            imageproc::point::Point::new(10.0, 20.0),
            imageproc::point::Point::new(40.0, 20.0),
            imageproc::point::Point::new(40.0, 50.0),
            imageproc::point::Point::new(10.0, 50.0),
        ];
        let block = |text: &str| crate::engine::EngineBlock {
            polygon,
            detection_confidence: 0.9,
            text: text.to_string(),
            confidence: 0.8,
        };
        let output = build_output(EngineExtraction {
            blocks: vec![block(""), block(" \t\r\n"), block("preserved text")],
            receipts: Vec::new(),
        })
        .unwrap();

        assert_eq!(output.text, "preserved text");
        assert_eq!(output.blocks.len(), 1);
        assert_eq!(output.blocks[0].text, "preserved text");
    }

    #[test]
    fn default_provider_is_local_and_explicit() {
        let provider = PpOcrV6Provider::from_env().unwrap();
        assert_eq!(provider.descriptor().id, PP_OCR_V6_PROVIDER_ID);
        assert_eq!(provider.descriptor().engine, ENGINE_NAME);
        assert!(!provider.descriptor().sends_source_off_device);
        assert_eq!(
            provider.descriptor().supported_stages,
            vec![OcrStage::Preprocessing, OcrStage::Text]
        );
    }

    #[tokio::test]
    async fn corrupt_batch_input_isolated_before_model_session_loading() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("corrupt.bmp");
        std::fs::write(&source, b"BM-not-a-decodable-bitmap").unwrap();
        let client = crate::OcrClient::with_provider(PpOcrV6Provider::from_env().unwrap()).unwrap();
        let result = client
            .extract_batch(
                crate::OcrBatchRequest::new(
                    vec![OcrStage::Preprocessing, OcrStage::Text],
                    vec![crate::OcrBatchSlotRequest::new(
                        crate::OcrBatchSlotId::new("target-a").unwrap(),
                        source,
                    )],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.slots[0].status, crate::OcrBatchSlotStatus::Failed);
        assert_eq!(
            result.slots[0].stages[0].status,
            crate::OcrStageStatus::Failed
        );
        assert_eq!(
            result.slots[0].stages[1].status,
            crate::OcrStageStatus::Skipped
        );
        assert!(result.execution_receipts.is_empty());
    }

    #[tokio::test]
    #[ignore = "requires the pinned official PP-OCRv6 bundle and real-image fixture"]
    async fn real_mixed_shape_batch_matches_scalar_and_emits_power_v4_receipts() {
        let source_path = std::path::PathBuf::from(
            std::env::var_os("A3S_PPOCR_V6_REAL_IMAGE")
                .expect("A3S_PPOCR_V6_REAL_IMAGE must name the pinned official image"),
        );
        let source_bytes = std::fs::read(&source_path).unwrap();
        let source = crate::preprocess::decode_image(&source_bytes).unwrap();
        let fixtures = tempfile::tempdir().unwrap();
        let wide_path = fixtures.path().join("wide.png");
        let square_path = fixtures.path().join("square.png");
        let tall_path = fixtures.path().join("tall.png");
        let mut wide = image::RgbImage::from_pixel(320, 288, image::Rgb([0, 0, 0]));
        let wide_content =
            image::imageops::resize(&source, 288, 170, image::imageops::FilterType::Triangle);
        image::imageops::replace(&mut wide, &wide_content, 16, 59);
        wide.save(&wide_path).unwrap();
        let mut square = image::RgbImage::from_pixel(320, 320, image::Rgb([0, 0, 0]));
        image::imageops::replace(&mut square, &wide_content, 16, 75);
        square.save(&square_path).unwrap();
        let tall_source = image::imageops::crop_imm(&source, 248, 0, 400, 528).to_image();
        let mut tall = image::RgbImage::from_pixel(256, 320, image::Rgb([0, 0, 0]));
        let tall_content = image::imageops::resize(
            &tall_source,
            224,
            296,
            image::imageops::FilterType::Triangle,
        );
        image::imageops::replace(&mut tall, &tall_content, 16, 12);
        tall.save(&tall_path).unwrap();

        let client = crate::OcrClient::with_provider(PpOcrV6Provider::from_env().unwrap()).unwrap();
        let wide_scalar = client
            .extract(crate::OcrRequest {
                path: wide_path.clone(),
            })
            .await
            .unwrap();
        let tall_scalar = client
            .extract(crate::OcrRequest {
                path: tall_path.clone(),
            })
            .await
            .unwrap();
        let square_scalar = client
            .extract(crate::OcrRequest {
                path: square_path.clone(),
            })
            .await
            .unwrap();
        let batch = client
            .extract_batch(
                crate::OcrBatchRequest::new(
                    vec![OcrStage::Preprocessing, OcrStage::Text],
                    vec![
                        crate::OcrBatchSlotRequest::new(
                            crate::OcrBatchSlotId::new("target-a").unwrap(),
                            wide_path,
                        ),
                        crate::OcrBatchSlotRequest::new(
                            crate::OcrBatchSlotId::new("target-b").unwrap(),
                            square_path,
                        ),
                        crate::OcrBatchSlotRequest::new(
                            crate::OcrBatchSlotId::new("target-c").unwrap(),
                            tall_path,
                        ),
                    ],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(batch.slots.len(), 3);
        assert_eq!(batch.slots[0].status, crate::OcrBatchSlotStatus::Completed);
        assert_eq!(batch.slots[1].status, crate::OcrBatchSlotStatus::Completed);
        assert_eq!(batch.slots[2].status, crate::OcrBatchSlotStatus::Completed);
        let wide_batch = batch.slots[0].result.as_ref().unwrap();
        let square_batch = batch.slots[1].result.as_ref().unwrap();
        let tall_batch = batch.slots[2].result.as_ref().unwrap();
        assert!(!wide_scalar.blocks.is_empty());
        assert!(!square_scalar.blocks.is_empty());
        assert!(!tall_scalar.blocks.is_empty());
        assert_token_f1("wide slot", &wide_scalar.text, &wide_batch.text);
        assert_token_f1("square slot", &square_scalar.text, &square_batch.text);
        assert_token_f1("tall slot", &tall_scalar.text, &tall_batch.text);
        assert_source_bounds(wide_batch, 320, 288);
        assert_source_bounds(square_batch, 320, 320);
        assert_source_bounds(tall_batch, 256, 320);
        assert_eq!(
            wide_batch.execution_receipts[0].input.item_count,
            square_batch.execution_receipts[0].input.item_count
        );
        assert!(
            wide_batch.execution_receipts[0].input.item_count
                > wide_scalar.execution_receipts[0].input.item_count
        );
        assert_eq!(
            square_batch.execution_receipts[0].input.item_count,
            square_scalar.execution_receipts[0].input.item_count * 2
        );
        assert_eq!(
            tall_batch.execution_receipts[0].input.item_count,
            tall_scalar.execution_receipts[0].input.item_count
        );
        let mut cohort_slot_counts = batch
            .execution_receipts
            .iter()
            .map(|receipt| receipt.microbatch.as_ref().unwrap().slot_count)
            .collect::<Vec<_>>();
        cohort_slot_counts.sort_unstable();
        assert_eq!(cohort_slot_counts, vec![1, 2]);
        assert_eq!(
            batch
                .execution_receipts
                .iter()
                .map(|receipt| receipt.microbatch.as_ref().unwrap().slot_count)
                .sum::<usize>(),
            3
        );
        for receipt in &batch.execution_receipts {
            assert_eq!(receipt.schema, "a3s.power.embedded-execution-receipt.v4");
            let evidence = receipt.microbatch.as_ref().unwrap();
            assert!(evidence.session_declaration_sha256.is_some());
            assert_eq!(evidence.batch_count, 1);
            assert_eq!(evidence.batch_index, 0);
        }
        assert_ne!(
            batch.execution_receipts[0]
                .microbatch
                .as_ref()
                .unwrap()
                .plan_sha256,
            batch.execution_receipts[1]
                .microbatch
                .as_ref()
                .unwrap()
                .plan_sha256
        );
    }

    fn assert_token_f1(label: &str, scalar: &str, batch: &str) {
        let expected = ascii_tokens(scalar);
        let actual = ascii_tokens(batch);
        let mut unmatched = actual.clone();
        let mut matches = 0_usize;
        for token in &expected {
            if let Some(index) = unmatched.iter().position(|candidate| candidate == token) {
                unmatched.swap_remove(index);
                matches += 1;
            }
        }
        let precision = matches as f64 / actual.len().max(1) as f64;
        let recall = matches as f64 / expected.len().max(1) as f64;
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        assert!(
            f1 >= 0.95,
            "{label} batch/scalar ASCII-token F1 {f1:.3} is below 0.950; scalar={expected:?}; batch={actual:?}"
        );
    }

    fn ascii_tokens(text: &str) -> Vec<String> {
        text.to_ascii_lowercase()
            .split(|character: char| !character.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }

    fn assert_source_bounds(result: &crate::OcrResult, width: u32, height: u32) {
        for block in &result.blocks {
            if let Some(polygon) = block.polygon {
                assert!(
                    polygon
                        .iter()
                        .all(|point| point.x < width && point.y < height),
                    "batch polygon escaped the source image"
                );
            }
            if let Some(bounds) = block.bounding_box {
                assert!(
                    bounds.x.saturating_add(bounds.width) <= width
                        && bounds.y.saturating_add(bounds.height) <= height,
                    "batch bounding box escaped the source image"
                );
            }
            assert!(block.bounding_boxes.iter().all(|bounds| {
                bounds.x.saturating_add(bounds.width) <= width
                    && bounds.y.saturating_add(bounds.height) <= height
            }));
        }
    }
}
