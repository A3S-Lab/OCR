use a3s_power::inference::ExecutionReceipt;
use a3s_use_core::{UseError, UseResult};
use image::{imageops, ImageBuffer, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use imageproc::point::Point;
use tokio_util::sync::CancellationToken;

use crate::assets::ModelAssets;
use crate::config::{load_detection, load_recognition, DetectionConfig, RecognitionConfig};
use crate::postprocess::{decode_ctc, detection_boxes, Detection};
use crate::ppocr_v6::native::NativePpOcrV6;
use crate::preprocess::{detection_input, recognition_input};

const RECOGNITION_BATCH_SIZE: usize = 8;
const MAX_CROP_PIXELS: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct EngineBlock {
    pub(crate) polygon: [Point<f32>; 4],
    pub(crate) detection_confidence: f32,
    pub(crate) text: String,
    pub(crate) confidence: f32,
}

pub(crate) struct EngineExtraction {
    pub(crate) blocks: Vec<EngineBlock>,
    pub(crate) receipts: Vec<ExecutionReceipt>,
}

pub(crate) struct PpOcrV6Engine {
    native: NativePpOcrV6,
    detection_config: DetectionConfig,
    recognition_config: RecognitionConfig,
}

impl PpOcrV6Engine {
    pub(crate) fn load(assets: &ModelAssets) -> UseResult<Self> {
        let detection_config = load_detection(&assets.detection_config)?;
        let recognition_config = load_recognition(&assets.recognition_config)?;
        let native = NativePpOcrV6::load(assets)?;
        Ok(Self {
            native,
            detection_config,
            recognition_config,
        })
    }

    pub(crate) fn extract(&mut self, image: &RgbImage) -> UseResult<EngineExtraction> {
        let cancellation = CancellationToken::new();
        let permit = self.native.begin(&cancellation)?;
        let input = detection_input(image, &self.detection_config)?;
        let detection = self
            .native
            .detect(input.data, input.shape, &permit, &cancellation)?;
        let shape = detection.tensor.shape;
        let output = detection.tensor.values;
        let detections = detection_boxes(
            &output,
            &shape,
            input.original_width,
            input.original_height,
            &self.detection_config,
        )?;
        if detections.is_empty() {
            return Ok(EngineExtraction {
                blocks: Vec::new(),
                receipts: vec![detection.receipt],
            });
        }

        let crops = detections
            .iter()
            .map(|detection| perspective_crop(image, detection))
            .collect::<UseResult<Vec<_>>>()?;
        let mut blocks = Vec::with_capacity(detections.len());
        let mut receipts = vec![detection.receipt];
        for (detection_batch, crop_batch) in detections
            .chunks(RECOGNITION_BATCH_SIZE)
            .zip(crops.chunks(RECOGNITION_BATCH_SIZE))
        {
            let input = recognition_input(crop_batch, &self.recognition_config)?;
            let recognition =
                self.native
                    .recognize(input.data, input.shape, &permit, &cancellation)?;
            let shape = recognition.tensor.shape;
            let output = recognition.tensor.values;
            receipts.push(recognition.receipt);
            if shape.len() != 3 || shape[0] != detection_batch.len() {
                return Err(engine_error(
                    "use.ocr.provider_output_invalid",
                    format!(
                        "PP-OCRv6 recognition output shape must be [N, T, C] for N={}, found {shape:?}.",
                        detection_batch.len()
                    ),
                ));
            }
            let item_len = shape[1].checked_mul(shape[2]).ok_or_else(|| {
                engine_error(
                    "use.ocr.provider_output_invalid",
                    "PP-OCRv6 recognition output dimensions overflowed.",
                )
            })?;
            if output.len() != detection_batch.len().saturating_mul(item_len) {
                return Err(engine_error(
                    "use.ocr.provider_output_invalid",
                    "PP-OCRv6 recognition output length does not match its batch shape.",
                ));
            }
            for (index, detection) in detection_batch.iter().enumerate() {
                let start = index * item_len;
                let recognition = decode_ctc(
                    &output[start..start + item_len],
                    &[1, shape[1], shape[2]],
                    &self.recognition_config,
                )?;
                blocks.push(EngineBlock {
                    polygon: detection.polygon,
                    detection_confidence: detection.confidence,
                    text: recognition.text,
                    confidence: recognition.confidence,
                });
            }
        }
        Ok(EngineExtraction { blocks, receipts })
    }
}

fn perspective_crop(image: &RgbImage, detection: &Detection) -> UseResult<RgbImage> {
    let polygon = detection.polygon;
    let width = distance(polygon[0], polygon[1])
        .max(distance(polygon[2], polygon[3]))
        .round()
        .max(1.0) as u32;
    let height = distance(polygon[0], polygon[3])
        .max(distance(polygon[1], polygon[2]))
        .round()
        .max(1.0) as u32;
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| crop_error("PP-OCRv6 text crop dimensions overflowed."))?;
    if pixels > MAX_CROP_PIXELS {
        return Err(crop_error(
            "PP-OCRv6 text crop exceeds the 64 megapixel safety limit.",
        ));
    }

    let source = polygon.map(|point| (point.x, point.y));
    let destination = [
        (0.0, 0.0),
        (width.saturating_sub(1) as f32, 0.0),
        (
            width.saturating_sub(1) as f32,
            height.saturating_sub(1) as f32,
        ),
        (0.0, height.saturating_sub(1) as f32),
    ];
    let projection = Projection::from_control_points(source, destination).ok_or_else(|| {
        crop_error("PP-OCRv6 detected a degenerate text polygon that cannot be rectified.")
    })?;
    let mut crop = ImageBuffer::new(width, height);
    warp_into(
        image,
        &projection,
        Interpolation::Bicubic,
        Rgb([255, 255, 255]),
        &mut crop,
    );
    if f64::from(height) / f64::from(width) >= 1.5 {
        Ok(imageops::rotate270(&crop))
    } else {
        Ok(crop)
    }
}

fn distance(left: Point<f32>, right: Point<f32>) -> f32 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn crop_error(message: impl Into<String>) -> UseError {
    engine_error("use.ocr.crop_invalid", message)
}

fn engine_error(code: &str, message: impl Into<String>) -> UseError {
    UseError::new(code, message)
}
