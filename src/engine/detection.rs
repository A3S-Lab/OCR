use a3s_power::inference::{ExecutionPermit, ExecutionReceipt, TensorOutput};
use a3s_use_core::UseResult;
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use super::{engine_error, PpOcrV6Engine};
use crate::cancellation::check_cancelled;
use crate::config::DetectionConfig;
use crate::postprocess::{detection_boxes_in_content, Detection};
use crate::preprocess::{
    detection_batch_inputs_with_max_side, DetectionInput, DETECTION_QUALITY_MAX_SIDE,
};

const QUALITY_RETRY_MIN_CHANNEL_RANGE: u8 = 32;

pub(super) fn postprocess_batch(
    inputs: Vec<DetectionInput>,
    tensors: Vec<TensorOutput>,
    config: &DetectionConfig,
) -> UseResult<Vec<UseResult<Vec<Detection>>>> {
    if inputs.is_empty() || inputs.len() != tensors.len() {
        return Err(engine_error(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 detection postprocessing received invalid batch cardinality.",
        ));
    }
    if inputs.len() == 1 {
        let input = inputs.into_iter().next().ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection postprocessing lost its scalar input.",
            )
        })?;
        let tensor = tensors.into_iter().next().ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection postprocessing lost its scalar output.",
            )
        })?;
        return Ok(vec![postprocess_one(input, tensor, config)]);
    }
    std::thread::scope(|scope| {
        let workers = inputs
            .into_iter()
            .zip(tensors)
            .map(|(input, tensor)| scope.spawn(move || postprocess_one(input, tensor, config)))
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
    input: DetectionInput,
    tensor: TensorOutput,
    config: &DetectionConfig,
) -> UseResult<Vec<Detection>> {
    detection_boxes_in_content(
        &tensor.values,
        &tensor.shape,
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
        let mut input = detection_batch_inputs_with_max_side(
            &[image],
            &self.detection_config,
            DETECTION_QUALITY_MAX_SIDE,
        )?
        .pop()
        .ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 quality retry produced no detection input.",
            )
        })?;
        let output = self.native.detect_batch(
            vec![(std::mem::take(&mut input.data), input.shape)],
            permit,
            cancellation,
        )?;
        if output.tensors.len() != 1 {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 quality retry changed scalar detection cardinality.",
            ));
        }
        let tensor = output.tensors.into_iter().next().ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 quality retry returned no detection tensor.",
            )
        })?;
        let detections = detection_boxes_in_content(
            &tensor.values,
            &tensor.shape,
            input.content_width,
            input.content_height,
            input.original_width,
            input.original_height,
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

    fn empty_detection_pair(side: usize) -> (DetectionInput, TensorOutput) {
        (
            DetectionInput {
                data: Vec::new(),
                shape: [1, 3, side, side],
                original_width: side as u32,
                original_height: side as u32,
                content_width: side as u32,
                content_height: side as u32,
            },
            TensorOutput {
                shape: vec![1, 1, side, side],
                values: vec![0.0; side * side],
            },
        )
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
        let (first_input, first_tensor) = empty_detection_pair(32);
        let (second_input, second_tensor) = empty_detection_pair(64);

        let detections = postprocess_batch(
            vec![first_input, second_input],
            vec![first_tensor, second_tensor],
            &config(),
        )
        .unwrap();

        assert_eq!(detections.len(), 2);
        assert!(detections.into_iter().all(|slot| slot.unwrap().is_empty()));
        assert!(postprocess_batch(Vec::new(), Vec::new(), &config()).is_err());
    }
}
