use a3s_power::inference::{ExecutionPermit, ExecutionReceipt, TensorOutput};
use a3s_use_core::UseResult;
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use super::{engine_error, PpOcrV6Engine};
use crate::cancellation::check_cancelled;
use crate::config::DetectionConfig;
use crate::postprocess::{detection_boxes_in_content, Detection};
use crate::preprocess::{
    detection_input_with_max_side, DetectionGeometry, DETECTION_QUALITY_MAX_SIDE,
};

const QUALITY_RETRY_MIN_CHANNEL_RANGE: u8 = 32;

pub(super) fn postprocess_batch(
    inputs: Vec<DetectionGeometry>,
    tensor: TensorOutput,
    config: &DetectionConfig,
) -> UseResult<Vec<UseResult<Vec<Detection>>>> {
    if inputs.is_empty()
        || tensor.shape.len() != 4
        || tensor.shape[0] != inputs.len()
        || tensor.shape[1] != 1
    {
        return Err(engine_error(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 detection postprocessing received invalid batch cardinality.",
        ));
    }
    let slot_elements = tensor.shape[1..]
        .iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection output dimensions overflowed.",
            )
        })?;
    if slot_elements == 0
        || slot_elements
            .checked_mul(inputs.len())
            .is_none_or(|expected| expected != tensor.values.len())
    {
        return Err(engine_error(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 detection output length does not match its batch shape.",
        ));
    }
    let slot_shape = [1, tensor.shape[1], tensor.shape[2], tensor.shape[3]];
    if inputs.len() == 1 {
        let input = inputs.into_iter().next().ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection postprocessing lost its scalar input.",
            )
        })?;
        return Ok(vec![postprocess_one(
            input,
            &tensor.values,
            &slot_shape,
            config,
        )]);
    }
    std::thread::scope(|scope| {
        let workers = inputs
            .into_iter()
            .zip(tensor.values.chunks_exact(slot_elements))
            .map(|(input, values)| {
                scope.spawn(move || postprocess_one(input, values, &slot_shape, config))
            })
            .collect::<Vec<_>>();
        let completed = workers
            .into_iter()
            .map(|worker| worker.join())
            .collect::<Vec<_>>();
        Ok(completed
            .into_iter()
            .map(|result| match result {
                Ok(detections) => detections,
                Err(_) => Err(engine_error(
                    "use.ocr.runtime_failed",
                    "PP-OCRv6 detection postprocessing worker failed.",
                )),
            })
            .collect())
    })
}

fn postprocess_one(
    input: DetectionGeometry,
    values: &[f32],
    shape: &[usize],
    config: &DetectionConfig,
) -> UseResult<Vec<Detection>> {
    detection_boxes_in_content(
        values,
        shape,
        input.content_width,
        input.content_height,
        input.original_width,
        input.original_height,
        config,
    )
}

impl PpOcrV6Engine {
    pub(super) fn retry_empty_detections(
        &self,
        images: &[&RgbImage],
        detections: &mut [UseResult<Vec<Detection>>],
        receipts: &mut [Vec<ExecutionReceipt>],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<()> {
        if detections.len() != images.len() || receipts.len() != images.len() {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 quality retry received mismatched batch cardinality.",
            ));
        }
        for index in 0..images.len() {
            check_cancelled(cancellation)?;
            if !should_retry_for_quality(images[index], &detections[index]) {
                continue;
            }
            match self.detect_one_for_quality(images[index], permit, cancellation) {
                Ok((quality_detections, receipt)) => {
                    detections[index] = Ok(quality_detections);
                    receipts[index].push(receipt);
                }
                Err(error) => detections[index] = Err(error),
            }
        }
        Ok(())
    }

    fn detect_one_for_quality(
        &self,
        image: &RgbImage,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<(Vec<Detection>, ExecutionReceipt)> {
        let input = detection_input_with_max_side(
            image,
            &self.detection_config,
            DETECTION_QUALITY_MAX_SIDE,
        )?;
        let output = self
            .native
            .detect_batch(input.data, input.shape, permit, cancellation)?;
        let detections = detection_boxes_in_content(
            &output.tensor.values,
            &output.tensor.shape,
            input.geometry.content_width,
            input.geometry.content_height,
            input.geometry.original_width,
            input.geometry.original_height,
            &self.detection_config,
        )?;
        Ok((detections, output.receipt))
    }
}

fn should_retry_for_quality(image: &RgbImage, detections: &UseResult<Vec<Detection>>) -> bool {
    detections.as_ref().is_ok_and(Vec::is_empty) && image_has_visual_variation(image)
}

fn image_has_visual_variation(image: &RgbImage) -> bool {
    let mut minimum = [u8::MAX; 3];
    let mut maximum = [u8::MIN; 3];
    for pixel in image.as_raw().chunks_exact(3) {
        for channel in 0..3 {
            minimum[channel] = minimum[channel].min(pixel[channel]);
            maximum[channel] = maximum[channel].max(pixel[channel]);
        }
        if (0..3).any(|channel| {
            maximum[channel].saturating_sub(minimum[channel]) >= QUALITY_RETRY_MIN_CHANNEL_RANGE
        }) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DetectionConfig {
        DetectionConfig {
            scale: 1.0 / 255.0,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            threshold: 0.3,
            box_threshold: 0.6,
            max_candidates: 1_000,
            unclip_ratio: 1.5,
        }
    }

    fn detection_geometry(side: usize) -> DetectionGeometry {
        DetectionGeometry {
            original_width: side as u32,
            original_height: side as u32,
            content_width: side as u32,
            content_height: side as u32,
        }
    }

    #[test]
    fn visual_variation_distinguishes_blank_and_marked_pages() {
        let blank = RgbImage::from_pixel(64, 64, image::Rgb([248, 248, 248]));
        let mut marked = blank.clone();
        marked.put_pixel(31, 17, image::Rgb([0, 0, 0]));
        let mut slight_noise = blank.clone();
        slight_noise.put_pixel(31, 17, image::Rgb([225, 225, 225]));

        assert!(!image_has_visual_variation(&blank));
        assert!(image_has_visual_variation(&marked));
        assert!(!image_has_visual_variation(&slight_noise));
    }

    #[test]
    fn quality_retry_requires_an_empty_successful_detection_on_a_marked_page() {
        let mut marked = RgbImage::from_pixel(64, 64, image::Rgb([248, 248, 248]));
        marked.put_pixel(31, 17, image::Rgb([0, 0, 0]));
        let blank = RgbImage::from_pixel(64, 64, image::Rgb([248, 248, 248]));
        let detection = Detection {
            polygon: [imageproc::point::Point::new(0.0, 0.0); 4],
            confidence: 1.0,
        };

        assert!(should_retry_for_quality(&marked, &Ok(Vec::new())));
        assert!(!should_retry_for_quality(&blank, &Ok(Vec::new())));
        assert!(!should_retry_for_quality(&marked, &Ok(vec![detection])));
        assert!(!should_retry_for_quality(
            &marked,
            &Err(engine_error("use.ocr.fixture", "failed detection")),
        ));
    }

    #[test]
    fn batch_postprocessing_preserves_cardinality_and_isolates_slots() {
        let tensor = TensorOutput {
            shape: vec![2, 1, 32, 32],
            values: vec![0.0; 2 * 32 * 32],
        };

        let detections = postprocess_batch(
            vec![detection_geometry(32), detection_geometry(64)],
            tensor,
            &config(),
        )
        .unwrap();

        assert_eq!(detections.len(), 2);
        assert!(detections[0].as_ref().unwrap().is_empty());
        assert!(detections[1].is_err());
        assert!(postprocess_batch(
            Vec::new(),
            TensorOutput {
                shape: vec![0, 1, 32, 32],
                values: Vec::new(),
            },
            &config()
        )
        .is_err());
    }
}
