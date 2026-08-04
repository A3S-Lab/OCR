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
    let blocks = blocks
        .into_iter()
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
    async fn real_staged_batch_is_deterministic_and_emits_power_v4_receipts() {
        let image = std::path::PathBuf::from(
            std::env::var_os("A3S_PPOCR_V6_REAL_IMAGE")
                .expect("A3S_PPOCR_V6_REAL_IMAGE must name the pinned official image"),
        );
        let client = crate::OcrClient::with_provider(PpOcrV6Provider::from_env().unwrap()).unwrap();
        let batch = client
            .extract_batch(
                crate::OcrBatchRequest::new(
                    vec![OcrStage::Preprocessing, OcrStage::Text],
                    vec![
                        crate::OcrBatchSlotRequest::new(
                            crate::OcrBatchSlotId::new("target-a").unwrap(),
                            image.clone(),
                        ),
                        crate::OcrBatchSlotRequest::new(
                            crate::OcrBatchSlotId::new("target-b").unwrap(),
                            image.clone(),
                        ),
                    ],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(batch.slots.len(), 2);
        assert_eq!(batch.slots[0].status, crate::OcrBatchSlotStatus::Completed);
        assert_eq!(batch.slots[1].status, crate::OcrBatchSlotStatus::Completed);
        assert_eq!(batch.slots[0].result, batch.slots[1].result);
        assert!(!batch.execution_receipts.is_empty());
        assert_eq!(
            batch
                .execution_receipts
                .iter()
                .map(|receipt| receipt.microbatch.as_ref().unwrap().slot_count)
                .sum::<usize>(),
            2
        );
        for receipt in &batch.execution_receipts {
            assert_eq!(receipt.schema, "a3s.power.embedded-execution-receipt.v4");
            let evidence = receipt.microbatch.as_ref().unwrap();
            assert!(evidence.session_declaration_sha256.is_some());
            assert_eq!(evidence.batch_count, batch.execution_receipts.len());
        }
    }
}
