use a3s_use_core::{UseError, UseResult};
use image::RgbImage;

use super::super::opencv_cubic::{resize_rgb_crop_cubic_normalized_chw, Crop};
use super::super::wired::PixelRect;
use super::profile::PicodetLayoutProfile;

const CHANNEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const CHANNEL_STANDARD_DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SealView {
    pub(super) region: PixelRect,
}

/// Returns the exact full-canvas view from the official model contract.
pub(super) fn full_page_view(image: &RgbImage) -> SealView {
    SealView {
        region: PixelRect {
            x: 0,
            y: 0,
            width: image.width(),
            height: image.height(),
        },
    }
}

/// Returns the official full-page model view for an immutable source canvas.
///
/// Admission depends only on immutable canvas dimensions and model topology.
/// It never reads source pixels, text, filenames, page numbers, or detections.
pub(super) fn model_contract_views(image: &RgbImage) -> Vec<SealView> {
    vec![full_page_view(image)]
}

#[cfg(test)]
pub(super) fn view_tensor(
    image: &RgbImage,
    view: SealView,
    profile: PicodetLayoutProfile,
) -> UseResult<Vec<f32>> {
    let mut tensor = vec![0.0_f32; profile.tensor_elements_per_view()];
    view_tensor_into(image, view, profile, &mut tensor)?;
    Ok(tensor)
}

pub(super) fn view_tensor_into(
    image: &RgbImage,
    view: SealView,
    profile: PicodetLayoutProfile,
    tensor: &mut [f32],
) -> UseResult<()> {
    validate_view(image, view)?;
    let tensor_elements = profile.tensor_elements_per_view();
    if tensor.len() != tensor_elements {
        return Err(UseError::new(
            "use.ocr.seal_view_invalid",
            format!("A PicoDet seal tensor requires exactly {tensor_elements} f32 values."),
        ));
    }
    let input_side = profile.input_side();
    resize_rgb_crop_cubic_normalized_chw(
        image,
        Crop {
            x: view.region.x,
            y: view.region.y,
            width: view.region.width,
            height: view.region.height,
        },
        input_side,
        input_side,
        CHANNEL_MEAN,
        CHANNEL_STANDARD_DEVIATION,
        tensor,
    )
    .map_err(|message| UseError::new("use.ocr.seal_view_invalid", message))
}

fn validate_view(image: &RgbImage, view: SealView) -> UseResult<()> {
    let right = view.region.x.checked_add(view.region.width);
    let bottom = view.region.y.checked_add(view.region.height);
    if view.region.width == 0
        || view.region.height == 0
        || right.is_none_or(|right| right > image.width())
        || bottom.is_none_or(|bottom| bottom > image.height())
    {
        return Err(UseError::new(
            "use.ocr.seal_view_invalid",
            "A PicoDet seal view must cover positive area inside its exact source canvas.",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn model_admission_is_exactly_the_immutable_source_canvas() {
        let image = RgbImage::new(1_200, 1_600);
        assert_eq!(
            full_page_view(&image).region,
            PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            }
        );
    }

    #[test]
    fn canvas_admission_is_the_exact_full_page_model_contract() {
        let portrait = RgbImage::new(1_190, 1_684);
        let views = model_contract_views(&portrait);
        assert_eq!(views, vec![full_page_view(&portrait)]);
    }

    #[test]
    fn tensor_uses_official_rgb_imagenet_normalization() {
        let image = RgbImage::from_pixel(2, 2, Rgb([255, 0, 0]));
        let view = full_page_view(&image);
        let profile = PicodetLayoutProfile::Large;
        let tensor = view_tensor(&image, view, profile).unwrap();
        let plane = profile.input_side() * profile.input_side();
        assert!((tensor[0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((tensor[plane] - (0.0 - 0.456) / 0.224).abs() < 1e-5);
        assert!((tensor[2 * plane] - (0.0 - 0.406) / 0.225).abs() < 1e-5);
    }
}
