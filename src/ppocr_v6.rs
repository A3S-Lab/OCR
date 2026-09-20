mod batch;
pub(crate) mod native;

use std::sync::Arc;

use a3s_power::inference::{
    DevicePreference, HardwareMemorySnapshot, ModelSessionPool, ModelSessionPoolPolicy,
    ModelSessionSpec, RuntimeDeviceKind,
};
use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use crate::assets::{ocr_status, resolve_model_assets, ModelAssets, OcrInstallSource};
use crate::config::ModelProfile;
#[cfg(test)]
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
const MAX_EXECUTION_REPLICAS: usize = 3;

/// Local PP-OCRv6 provider shipped as the default A3S Use integration.
#[derive(Clone)]
pub struct PpOcrV6Provider {
    descriptor: OcrProviderDescriptor,
    sessions: ModelSessionPool<PpOcrV6Session>,
    execution_replica: usize,
    bound_model: Option<Arc<BoundPpOcrV6Model>>,
}

pub(super) struct PpOcrV6Session {
    engine: PpOcrV6Engine,
}

#[derive(Clone)]
struct BoundPpOcrV6Model {
    assets: ModelAssets,
    session_spec: ModelSessionSpec,
}

impl PpOcrV6Provider {
    pub fn from_env() -> UseResult<Self> {
        Self::new(None)
    }

    /// Resolve and freeze the exact local model/configuration used by a
    /// composite provider. A later environment-variable change cannot switch
    /// the already-admitted document pipeline to another PP-OCRv6 profile.
    pub(crate) fn from_env_bound() -> UseResult<Self> {
        let assets = resolve_model_assets()?;
        let session_spec = native::session_spec(&assets)?;
        Self::new(Some(BoundPpOcrV6Model {
            assets,
            session_spec,
        }))
    }

    fn new(bound_model: Option<BoundPpOcrV6Model>) -> UseResult<Self> {
        let policy = ModelSessionPoolPolicy::new(
            MAX_EXECUTION_REPLICAS,
            1024 * 1024 * 1024,
            MAX_EXECUTION_REPLICAS,
            32,
        )
        .map_err(|error| pool_error("configure", error))?;
        Ok(Self {
            descriptor: OcrProviderDescriptor::new(PP_OCR_V6_PROVIDER_ID, ENGINE_NAME, false)?
                .with_stages(vec![OcrStage::Preprocessing, OcrStage::Text])?
                .with_text_windows(true)?,
            sessions: ModelSessionPool::new(DevicePreference::Auto, policy)
                .map_err(|error| pool_error("initialize", error))?,
            execution_replica: 0,
            bound_model: bound_model.map(Arc::new),
        })
    }

    pub(crate) fn configured_model_profile(&self) -> UseResult<ModelProfile> {
        self.bound_model
            .as_ref()
            .map(|model| Ok(model.assets.profile))
            .unwrap_or_else(|| resolve_model_assets().map(|assets| assets.profile))
    }

    pub(super) fn resolved_session_assets(&self) -> UseResult<(ModelAssets, ModelSessionSpec)> {
        let Some(bound) = self.bound_model.as_ref() else {
            let assets = resolve_model_assets()?;
            let spec = native::session_spec(&assets)?;
            return Ok((assets, spec));
        };
        let observed = native::session_spec(&bound.assets)?;
        if observed != bound.session_spec {
            return Err(UseError::new(
                "use.ocr.model_configuration_changed",
                "The bound PP-OCRv6 model configuration changed after provider initialization.",
            )
            .with_suggestion(
                "Create a new OCR provider after restoring or intentionally replacing the model bundle.",
            ));
        }
        Ok((bound.assets.clone(), bound.session_spec.clone()))
    }

    pub(crate) fn execution_replica(&self, execution_replica: usize) -> UseResult<Self> {
        if execution_replica >= MAX_EXECUTION_REPLICAS {
            return Err(pool_error(
                "select",
                format!(
                    "execution replica {execution_replica} exceeds the PP-OCRv6 bound of {MAX_EXECUTION_REPLICAS}"
                ),
            ));
        }
        if execution_replica > 0 && self.runtime_device_kind() != RuntimeDeviceKind::Cuda {
            return Err(pool_error(
                "select",
                "additional PP-OCRv6 execution replicas require CUDA",
            ));
        }
        Ok(Self {
            descriptor: self.descriptor.clone(),
            sessions: self.sessions.clone(),
            execution_replica,
            bound_model: self.bound_model.clone(),
        })
    }

    pub(crate) async fn recognize_batch_decoded_with_helpers(
        &self,
        helpers: &[Self],
        request: OcrProviderBatchRequest,
        images: Vec<Result<std::sync::Arc<RgbImage>, UseError>>,
        cancellation: &CancellationToken,
    ) -> UseResult<OcrProviderBatchOutput> {
        batch::recognize_batch_decoded_with_helpers(self, helpers, request, images, cancellation)
            .await
    }

    pub(crate) fn runtime_memory_snapshot(&self) -> UseResult<HardwareMemorySnapshot> {
        self.sessions
            .memory_snapshot()
            .map_err(|error| pool_error("inspect device memory for", error))
    }

    pub(crate) fn runtime_device_kind(&self) -> RuntimeDeviceKind {
        self.sessions.snapshot().device.kind
    }
}

#[async_trait]
impl OcrProvider for PpOcrV6Provider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        self.descriptor.clone()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        if let Some(bound) = self.bound_model.as_ref() {
            let (readiness, message, suggestions) = match self.resolved_session_assets() {
                Ok(_) => (
                    Readiness::Ready,
                    "The bound local PP-OCRv6 detection and recognition models are ready."
                        .to_string(),
                    Vec::new(),
                ),
                Err(error) => (
                    Readiness::Broken,
                    error.message,
                    vec![
                        "Create a new OCR provider after restoring or intentionally replacing the model bundle."
                            .to_string(),
                    ],
                ),
            };
            return OcrProviderStatus {
                readiness,
                model: Some(bound.assets.profile.family().to_string()),
                model_dir: Some(bound.assets.root.clone()),
                message,
                suggestions,
            };
        }
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
    let EngineExtraction {
        model,
        blocks,
        receipts,
    } = extraction;
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some()
        || std::env::var_os("A3S_OCR_TRACE_RECOGNITION_SUMMARY").is_some()
    {
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
                text_rotation_millidegrees: Some(block.text_rotation_millidegrees),
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
        model: Some(model.to_string()),
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
            text_rotation_millidegrees: 0,
            detection_confidence: 0.9,
            text: text.to_string(),
            confidence: 0.8,
        };
        let output = build_output(EngineExtraction {
            model: MODEL_FAMILY,
            blocks: vec![block(""), block(" \t\r\n"), block("preserved text")],
            receipts: Vec::new(),
        })
        .unwrap();

        assert_eq!(output.text, "preserved text");
        assert_eq!(output.blocks.len(), 1);
        assert_eq!(output.blocks[0].text, "preserved text");
        assert_eq!(output.blocks[0].text_rotation_millidegrees, Some(0));
    }

    #[test]
    fn default_provider_is_local_and_explicit() {
        let provider = PpOcrV6Provider::from_env().unwrap();
        assert_eq!(provider.descriptor().id, PP_OCR_V6_PROVIDER_ID);
        assert_eq!(provider.descriptor().engine, ENGINE_NAME);
        assert!(!provider.descriptor().sends_source_off_device);
        assert!(provider.descriptor().supports_text_windows);
        assert_eq!(
            provider.descriptor().supported_stages,
            vec![OcrStage::Preprocessing, OcrStage::Text]
        );
    }

    #[test]
    fn bound_provider_rejects_a_post_admission_configuration_change() {
        let directory = tempfile::tempdir().unwrap();
        let detection_root = directory.path().join("det");
        let recognition_root = directory.path().join("rec");
        std::fs::create_dir_all(&detection_root).unwrap();
        std::fs::create_dir_all(&recognition_root).unwrap();
        std::fs::write(detection_root.join("model.safetensors"), b"detection").unwrap();
        std::fs::write(recognition_root.join("model.safetensors"), b"recognition").unwrap();
        std::fs::write(
            detection_root.join("inference.yml"),
            "Global:\n  model_name: PP-OCRv6_small_det\nPreProcess:\n  transform_ops:\n    - NormalizeImage:\n        mean: [0.485, 0.456, 0.406]\n        std: [0.229, 0.224, 0.225]\nPostProcess:\n  name: DBPostProcess\n",
        )
        .unwrap();
        let recognition_config = recognition_root.join("inference.yml");
        std::fs::write(
            &recognition_config,
            "Global:\n  model_name: PP-OCRv6_small_rec\nPreProcess:\n  transform_ops:\n    - RecResizeImg:\n        image_shape: [3, 48, 320]\nPostProcess:\n  name: CTCLabelDecode\n  character_dict: [a]\n",
        )
        .unwrap();
        let assets = crate::assets::validate_assets(
            directory.path(),
            crate::assets::OcrInstallSource::Environment,
        )
        .unwrap();
        let session_spec = native::session_spec(&assets).unwrap();
        let provider = PpOcrV6Provider::new(Some(BoundPpOcrV6Model {
            assets,
            session_spec,
        }))
        .unwrap();
        assert_eq!(provider.diagnostic().readiness, Readiness::Ready);

        std::fs::write(
            recognition_config,
            "Global:\n  model_name: PP-OCRv6_small_rec\nPreProcess:\n  transform_ops:\n    - RecResizeImg:\n        image_shape: [3, 48, 320]\nPostProcess:\n  name: CTCLabelDecode\n  character_dict: [a, b]\n",
        )
        .unwrap();

        let error = provider.resolved_session_assets().unwrap_err();
        assert_eq!(error.code, "use.ocr.model_configuration_changed");
        assert_eq!(provider.diagnostic().readiness, Readiness::Broken);
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
        assert_eq!(batch.execution_receipts.len(), 1);
        for receipt in &batch.execution_receipts {
            assert_eq!(receipt.schema, "a3s.power.embedded-execution-receipt.v4");
            let evidence = receipt.microbatch.as_ref().unwrap();
            assert!(evidence.session_declaration_sha256.is_some());
            assert_eq!(evidence.slot_count, 3);
            assert_eq!(evidence.batch_count, 1);
            assert_eq!(evidence.batch_index, 0);
        }
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
