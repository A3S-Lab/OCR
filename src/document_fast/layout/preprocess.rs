use a3s_use_core::{UseError, UseResult};
use image::RgbImage;

use super::profile::{INPUT_ELEMENTS_PER_IMAGE, INPUT_SIDE};
use crate::document_fast::opencv_cubic::{resize_rgb_crop_cubic_normalized_chw, Crop};

const CHANNEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const CHANNEL_STANDARD_DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];

/// Exact PP-DocLayout-S preprocessing: full-canvas, non-aspect-preserving
/// 480x480 bicubic resize, RGB scale, ImageNet normalization, then CHW.
pub(super) fn image_tensor_into(image: &RgbImage, tensor: &mut [f32]) -> UseResult<()> {
    if image.width() == 0 || image.height() == 0 {
        return Err(input_error(
            "A document-layout image must have positive dimensions.",
        ));
    }
    if tensor.len() != INPUT_ELEMENTS_PER_IMAGE {
        return Err(input_error(format!(
            "A document-layout tensor requires exactly {INPUT_ELEMENTS_PER_IMAGE} f32 values."
        )));
    }
    resize_rgb_crop_cubic_normalized_chw(
        image,
        Crop {
            x: 0,
            y: 0,
            width: image.width(),
            height: image.height(),
        },
        INPUT_SIDE,
        INPUT_SIDE,
        CHANNEL_MEAN,
        CHANNEL_STANDARD_DEVIATION,
        tensor,
    )
    .map_err(input_error)
}

fn input_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.document_layout_input_invalid", message)
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    #[test]
    fn tensor_uses_the_reviewed_full_canvas_rgb_contract() {
        let image = RgbImage::from_pixel(2, 4, Rgb([255, 0, 0]));
        let mut tensor = vec![0.0; INPUT_ELEMENTS_PER_IMAGE];
        image_tensor_into(&image, &mut tensor).unwrap();
        let plane = INPUT_SIDE * INPUT_SIDE;
        assert!((tensor[0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((tensor[plane] - (0.0 - 0.456) / 0.224).abs() < 1e-5);
        assert!((tensor[2 * plane] - (0.0 - 0.406) / 0.225).abs() < 1e-5);
    }
}
