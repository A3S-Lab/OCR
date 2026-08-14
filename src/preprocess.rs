use std::io::Cursor;

use a3s_use_core::{UseError, UseResult};
use image::imageops::FilterType;
use image::{DynamicImage, ImageReader, Limits, RgbImage};

use crate::config::{DetectionConfig, RecognitionConfig};

const MAX_IMAGE_SIDE: u32 = 16_384;
const MAX_DECODED_BYTES: u64 = 256 * 1024 * 1024;
// PaddleOCR's reviewed PP-OCRv6 small pipeline uses `limit_type = min` with
// `limit_side_len = 64`. The model's HPI metadata also names 736 as an
// optimization profile, but that is not the pipeline resize policy.
const DETECTION_MIN_SIDE: u32 = 64;
// Whole-page detection is bounded independently from recognition. Detected
// polygons are mapped back to the immutable source image, and recognition
// crops that higher-resolution source rather than this detector raster.
const DETECTION_FAST_MAX_SIDE: u32 = 896;
pub(crate) const DETECTION_QUALITY_MAX_SIDE: u32 = 4_000;
const DETECTION_MAX_BATCH_SIZE: usize = 16;
const RECOGNITION_MAX_BATCH_SIZE: usize = 8;
const RECOGNITION_MAX_WIDTH: u32 = 3_200;

pub(crate) struct DetectionInput {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: [usize; 4],
    pub(crate) original_width: u32,
    pub(crate) original_height: u32,
    pub(crate) content_width: u32,
    pub(crate) content_height: u32,
}

pub(crate) struct RecognitionInput {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: [usize; 4],
}

pub(crate) fn decode_image(bytes: &[u8]) -> UseResult<RgbImage> {
    let cursor = Cursor::new(bytes);
    let mut reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|error| image_error(format!("Failed to detect OCR image format: {error}")))?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE);
    limits.max_image_height = Some(MAX_IMAGE_SIDE);
    limits.max_alloc = Some(MAX_DECODED_BYTES);
    reader.limits(limits);
    let image = reader
        .decode()
        .map_err(|error| image_error(format!("Failed to decode OCR image: {error}")))?;
    let width = image.width();
    let height = image.height();
    if width == 0
        || height == 0
        || u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .is_none_or(|bytes| bytes > MAX_DECODED_BYTES)
    {
        return Err(image_error(
            "Decoded OCR image dimensions exceed the 256 MiB pixel limit.",
        ));
    }
    Ok(image.to_rgb8())
}

pub(crate) fn detection_batch_inputs(
    images: &[&RgbImage],
    config: &DetectionConfig,
) -> UseResult<Vec<DetectionInput>> {
    detection_batch_inputs_with_max_side(images, config, DETECTION_FAST_MAX_SIDE)
}

pub(crate) fn detection_batch_inputs_with_max_side(
    images: &[&RgbImage],
    config: &DetectionConfig,
    max_side: u32,
) -> UseResult<Vec<DetectionInput>> {
    if images.is_empty() || images.len() > DETECTION_MAX_BATCH_SIZE {
        return Err(image_error(format!(
            "PP-OCRv6 detection batches must contain from 1 through {DETECTION_MAX_BATCH_SIZE} images."
        )));
    }
    let dimensions = images
        .iter()
        .map(|image| detection_dimensions_with_max_side(image.width(), image.height(), max_side))
        .collect::<UseResult<Vec<_>>>()?;
    let (canvas_width, canvas_height) = canvas_dimensions(&dimensions)?;
    let plane = usize::try_from(u64::from(canvas_width) * u64::from(canvas_height))
        .map_err(|_| image_error("Detection batch tensor dimensions overflowed."))?;
    let tensor_elements = plane
        .checked_mul(3)
        .ok_or_else(|| image_error("Detection batch tensor dimensions overflowed."))?;
    if images.len() == 1 {
        return detection_input(
            images[0],
            dimensions[0],
            (canvas_width, canvas_height),
            plane,
            tensor_elements,
            config,
        )
        .map(|input| vec![input]);
    }

    std::thread::scope(|scope| {
        let workers = images
            .iter()
            .copied()
            .zip(dimensions.iter().copied())
            .map(|(image, dimensions)| {
                scope.spawn(move || {
                    detection_input(
                        image,
                        dimensions,
                        (canvas_width, canvas_height),
                        plane,
                        tensor_elements,
                        config,
                    )
                })
            })
            .collect::<Vec<_>>();
        let completed = workers
            .into_iter()
            .map(|worker| worker.join())
            .collect::<Vec<_>>();
        completed
            .into_iter()
            .map(|result| match result {
                Ok(input) => input,
                Err(_) => Err(image_error(
                    "PP-OCRv6 detection preprocessing worker failed.",
                )),
            })
            .collect()
    })
}

fn detection_input(
    image: &RgbImage,
    (content_width, content_height): (u32, u32),
    (canvas_width, canvas_height): (u32, u32),
    plane: usize,
    tensor_elements: usize,
    config: &DetectionConfig,
) -> UseResult<DetectionInput> {
    let original_width = image.width();
    let original_height = image.height();
    let resized = if content_width == original_width && content_height == original_height {
        image.clone()
    } else {
        DynamicImage::ImageRgb8(image.clone())
            .resize_exact(content_width, content_height, FilterType::Triangle)
            .to_rgb8()
    };
    let mut data = vec![0.0_f32; tensor_elements];
    for channel in 0..3 {
        data[channel * plane..(channel + 1) * plane]
            .fill(-config.mean[channel] / config.std[channel]);
    }
    for y in 0..content_height {
        for x in 0..content_width {
            let pixel = resized.get_pixel(x, y);
            let target = y as usize * canvas_width as usize + x as usize;
            let channels = [pixel[2], pixel[1], pixel[0]];
            for channel in 0..3 {
                data[channel * plane + target] = (f32::from(channels[channel]) * config.scale
                    - config.mean[channel])
                    / config.std[channel];
            }
        }
    }
    Ok(DetectionInput {
        data,
        shape: [1, 3, canvas_height as usize, canvas_width as usize],
        original_width,
        original_height,
        content_width,
        content_height,
    })
}

pub(crate) fn detection_canvas_dimensions(images: &[&RgbImage]) -> UseResult<(u32, u32)> {
    if images.is_empty() {
        return Err(image_error(
            "PP-OCRv6 detection canvas requires at least one image.",
        ));
    }
    let dimensions = images
        .iter()
        .map(|image| detection_dimensions(image.width(), image.height()))
        .collect::<UseResult<Vec<_>>>()?;
    canvas_dimensions(&dimensions)
}

fn canvas_dimensions(dimensions: &[(u32, u32)]) -> UseResult<(u32, u32)> {
    dimensions
        .iter()
        .copied()
        .reduce(|left, right| (left.0.max(right.0), left.1.max(right.1)))
        .ok_or_else(|| image_error("PP-OCRv6 detection batch has no canvas dimensions."))
}

pub(crate) fn recognition_input(
    images: &[&RgbImage],
    config: &RecognitionConfig,
) -> UseResult<RecognitionInput> {
    if images.is_empty() || images.len() > RECOGNITION_MAX_BATCH_SIZE {
        return Err(image_error(format!(
            "PP-OCRv6 recognition batches must contain from 1 through {RECOGNITION_MAX_BATCH_SIZE} text crops."
        )));
    }
    if images
        .iter()
        .any(|image| image.width() == 0 || image.height() == 0)
    {
        return Err(image_error("PP-OCRv6 text crop has zero width or height."));
    }
    let model_height = recognition_model_height(config)?;
    let resized_widths = images
        .iter()
        .map(|image| recognition_resized_width(image.width(), image.height(), model_height))
        .collect::<UseResult<Vec<_>>>()?;
    let widest = resized_widths.iter().copied().max().unwrap_or(1);
    let canvas_width = recognition_default_width(config)?
        .max(widest)
        .min(RECOGNITION_MAX_WIDTH);
    let target_plane = usize::try_from(u64::from(canvas_width) * u64::from(model_height))
        .map_err(|_| image_error("Recognition tensor dimensions overflowed."))?;
    let batch_stride = config
        .channels
        .checked_mul(target_plane)
        .ok_or_else(|| image_error("Recognition tensor dimensions overflowed."))?;
    let mut data = vec![0.0_f32; images.len() * batch_stride];
    for (batch, (image, resized_width)) in images.iter().zip(resized_widths).enumerate() {
        let resized = DynamicImage::ImageRgb8((*image).clone())
            .resize_exact(resized_width, model_height, FilterType::Triangle)
            .to_rgb8();
        for y in 0..model_height {
            for x in 0..resized_width {
                let pixel = resized.get_pixel(x, y);
                let target = y as usize * canvas_width as usize + x as usize;
                let channels = [pixel[2], pixel[1], pixel[0]];
                for channel in 0..config.channels {
                    data[batch * batch_stride + channel * target_plane + target] =
                        f32::from(channels[channel]) / 127.5 - 1.0;
                }
            }
        }
    }
    Ok(RecognitionInput {
        data,
        shape: [
            images.len(),
            config.channels,
            config.height,
            canvas_width as usize,
        ],
    })
}

pub(crate) fn recognition_canvas_width(
    width: u32,
    height: u32,
    config: &RecognitionConfig,
) -> UseResult<u32> {
    let resized_width =
        recognition_resized_width(width, height, recognition_model_height(config)?)?;
    Ok(recognition_default_width(config)?
        .max(resized_width)
        .min(RECOGNITION_MAX_WIDTH))
}

fn recognition_resized_width(width: u32, height: u32, model_height: u32) -> UseResult<u32> {
    if width == 0 || height == 0 {
        return Err(image_error("PP-OCRv6 text crop has zero width or height."));
    }
    Ok(
        ((f64::from(model_height) * f64::from(width) / f64::from(height)).ceil() as u32)
            .clamp(1, RECOGNITION_MAX_WIDTH),
    )
}

fn recognition_model_height(config: &RecognitionConfig) -> UseResult<u32> {
    u32::try_from(config.height)
        .ok()
        .filter(|height| *height > 0)
        .ok_or_else(|| image_error("Recognition model height is invalid."))
}

fn recognition_default_width(config: &RecognitionConfig) -> UseResult<u32> {
    u32::try_from(config.default_width)
        .ok()
        .filter(|width| *width > 0)
        .ok_or_else(|| image_error("Recognition model width is invalid."))
}

pub(crate) fn detection_dimensions(width: u32, height: u32) -> UseResult<(u32, u32)> {
    detection_dimensions_with_max_side(width, height, DETECTION_FAST_MAX_SIDE)
}

fn detection_dimensions_with_max_side(
    width: u32,
    height: u32,
    max_side_limit: u32,
) -> UseResult<(u32, u32)> {
    if width == 0 || height == 0 {
        return Err(image_error("OCR image has zero width or height."));
    }
    if !(DETECTION_MIN_SIDE..=DETECTION_QUALITY_MAX_SIDE).contains(&max_side_limit) {
        return Err(image_error(format!(
            "OCR detection maximum side must contain {DETECTION_MIN_SIDE} through {DETECTION_QUALITY_MAX_SIDE} pixels."
        )));
    }
    let mut ratio = 1.0_f64;
    let min_side = width.min(height);
    let max_side = width.max(height);
    if min_side < DETECTION_MIN_SIDE {
        ratio = f64::from(DETECTION_MIN_SIDE) / f64::from(min_side);
    }
    if f64::from(max_side) * ratio > f64::from(max_side_limit) {
        ratio = f64::from(max_side_limit) / f64::from(max_side);
    }
    let resized_width = round_stride(f64::from(width) * ratio, 32);
    let resized_height = round_stride(f64::from(height) * ratio, 32);
    Ok((resized_width, resized_height))
}

fn round_stride(value: f64, stride: u32) -> u32 {
    let rounded = (value / f64::from(stride)).round_ties_even() as u32 * stride;
    rounded.max(stride)
}

fn image_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.image_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection_config() -> DetectionConfig {
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

    fn recognition_config() -> RecognitionConfig {
        RecognitionConfig {
            channels: 3,
            height: 48,
            default_width: 320,
            characters: vec!["blank".to_string()],
        }
    }

    #[test]
    fn detection_dimensions_are_bounded_stride_multiples() {
        assert_eq!(detection_dimensions(10, 20).unwrap(), (64, 128));
        assert_eq!(detection_dimensions(4_000, 1_000).unwrap(), (896, 224));
        assert_eq!(detection_dimensions(20_000, 10_000).unwrap(), (896, 448));
        assert_eq!(
            detection_dimensions_with_max_side(4_000, 1_000, DETECTION_QUALITY_MAX_SIDE).unwrap(),
            (4_000, 992)
        );
        assert!(detection_dimensions_with_max_side(320, 320, 32).is_err());
    }

    #[test]
    fn detection_batch_letterboxes_mixed_shapes_on_one_canvas() {
        let wide = RgbImage::from_pixel(40, 20, image::Rgb([255, 0, 0]));
        let tall = RgbImage::from_pixel(20, 40, image::Rgb([0, 255, 0]));

        let inputs = detection_batch_inputs(&[&wide, &tall], &detection_config()).unwrap();

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].shape, [1, 3, 128, 128]);
        assert_eq!(inputs[1].shape, [1, 3, 128, 128]);
        assert_eq!(
            (inputs[0].content_width, inputs[0].content_height),
            (128, 64)
        );
        assert_eq!(
            (inputs[1].content_width, inputs[1].content_height),
            (64, 128)
        );
        let plane = 128 * 128;
        for channel in 0..3 {
            let config = detection_config();
            let normalized_black = -config.mean[channel] / config.std[channel];
            assert_eq!(
                inputs[0].data[channel * plane + 100 * 128],
                normalized_black
            );
            assert_eq!(inputs[1].data[channel * plane + 100], normalized_black);
        }
        assert_ne!(inputs[0].data[0], 0.0);
        assert_ne!(inputs[1].data[0], 0.0);
    }

    #[test]
    fn detection_batch_rejects_empty_and_oversized_groups() {
        let image = RgbImage::new(32, 32);
        assert!(detection_batch_inputs(&[], &detection_config()).is_err());
        let images = std::iter::repeat_n(&image, 17).collect::<Vec<_>>();
        assert!(detection_batch_inputs(&images, &detection_config()).is_err());
    }

    #[test]
    fn detection_batch_preserves_input_order_and_single_input_values() {
        let red = RgbImage::from_pixel(64, 64, image::Rgb([255, 0, 0]));
        let green = RgbImage::from_pixel(64, 64, image::Rgb([0, 255, 0]));
        let blue = RgbImage::from_pixel(64, 64, image::Rgb([0, 0, 255]));
        let config = detection_config();

        let batch = detection_batch_inputs(&[&red, &green, &blue], &config).unwrap();
        for (batch_input, image) in batch.iter().zip([&red, &green, &blue]) {
            let single = detection_batch_inputs(&[image], &config).unwrap().remove(0);
            assert_eq!(batch_input.shape, single.shape);
            assert_eq!(batch_input.data, single.data);
        }
    }

    #[test]
    fn recognition_canvas_width_matches_the_materialized_tensor() {
        let narrow = RgbImage::new(100, 20);
        let wide = RgbImage::new(400, 20);
        let config = recognition_config();

        assert_eq!(recognition_canvas_width(100, 20, &config).unwrap(), 320);
        assert_eq!(recognition_canvas_width(400, 20, &config).unwrap(), 960);
        assert_eq!(
            recognition_input(&[&narrow, &wide], &config).unwrap().shape,
            [2, 3, 48, 960]
        );
        assert_eq!(
            recognition_canvas_width(10_000, 1, &config).unwrap(),
            RECOGNITION_MAX_WIDTH
        );
        assert!(recognition_canvas_width(0, 20, &config).is_err());
        let oversized =
            std::iter::repeat_n(&narrow, RECOGNITION_MAX_BATCH_SIZE + 1).collect::<Vec<_>>();
        assert!(recognition_input(&oversized, &config).is_err());
    }
}
