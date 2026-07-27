use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;

use crate::assets::{ocr_status, resolve_model_assets, OcrInstallSource};
use crate::config::MODEL_FAMILY;
use crate::engine::{EngineBlock, PpOcrV6Engine};
use crate::models::{OcrBlock, OcrBoundingBox, OcrPoint};
use crate::preprocess::decode_image;
use crate::provider::{
    OcrInput, OcrProvider, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
};

pub const PP_OCR_V6_PROVIDER_ID: &str = "pp-ocr-v6";
const ENGINE_NAME: &str = "onnx-runtime";

/// Local PP-OCRv6 provider shipped as the default A3S Use integration.
#[derive(Clone)]
pub struct PpOcrV6Provider {
    descriptor: OcrProviderDescriptor,
    loaded: Arc<Mutex<Option<LoadedEngine>>>,
}

struct LoadedEngine {
    model_dir: PathBuf,
    engine: PpOcrV6Engine,
}

impl PpOcrV6Provider {
    pub fn from_env() -> UseResult<Self> {
        Ok(Self {
            descriptor: OcrProviderDescriptor::new(PP_OCR_V6_PROVIDER_ID, ENGINE_NAME, false)?,
            loaded: Arc::new(Mutex::new(None)),
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
        let loaded = Arc::clone(&self.loaded);
        tokio::task::spawn_blocking(move || {
            let image = decode_image(input.bytes())?;
            let assets = resolve_model_assets()?;
            let mut loaded = loaded.lock().map_err(|_| {
                UseError::new(
                    "use.ocr.runtime_failed",
                    "The local PP-OCRv6 engine lock is poisoned.",
                )
            })?;
            let should_load = loaded
                .as_ref()
                .map(|loaded| loaded.model_dir != assets.root)
                .unwrap_or(true);
            if should_load {
                *loaded = Some(LoadedEngine {
                    model_dir: assets.root.clone(),
                    engine: PpOcrV6Engine::load(&assets)?,
                });
            }
            let engine = loaded.as_mut().ok_or_else(|| {
                UseError::new(
                    "use.ocr.runtime_failed",
                    "The local PP-OCRv6 engine failed to initialize.",
                )
            })?;
            build_output(engine.engine.extract(&image)?)
        })
        .await
        .map_err(|error| {
            UseError::new(
                "use.ocr.runtime_failed",
                format!("The local PP-OCRv6 inference task failed: {error}"),
            )
        })?
    }
}

fn build_output(blocks: Vec<EngineBlock>) -> UseResult<OcrProviderOutput> {
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
                confidence: Some(block.confidence),
                detection_confidence: Some(block.detection_confidence),
                polygon: Some(polygon),
                bounding_box: Some(OcrBoundingBox {
                    x: min_x,
                    y: min_y,
                    width: max_x.saturating_sub(min_x),
                    height: max_y.saturating_sub(min_y),
                }),
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
        warnings: Vec::new(),
    })
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
    }
}
