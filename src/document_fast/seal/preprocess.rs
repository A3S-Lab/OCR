use a3s_use_core::{UseError, UseResult};
use image::imageops::FilterType;
use image::RgbImage;

use crate::OcrCanvasEdge;

use super::super::wired::PixelRect;

pub(super) const INPUT_SIDE: usize = 640;
const CHANNEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const CHANNEL_STD: [f32; 3] = [0.229, 0.224, 0.225];
const MIN_EDGE_STRIP_WIDTH: u32 = 64;
const MAX_EDGE_STRIP_WIDTH: u32 = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SealViewKind {
    FullPage,
    Boundary(OcrCanvasEdge),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SealView {
    pub(super) region: PixelRect,
    pub(super) kind: SealViewKind,
}

pub(super) fn page_views(image: &RgbImage) -> Vec<SealView> {
    let full = SealView {
        region: PixelRect {
            x: 0,
            y: 0,
            width: image.width(),
            height: image.height(),
        },
        kind: SealViewKind::FullPage,
    };
    if image.width() < MIN_EDGE_STRIP_WIDTH.saturating_mul(2) {
        return vec![full];
    }
    let strip_width = image
        .width()
        .div_ceil(12)
        .clamp(MIN_EDGE_STRIP_WIDTH, MAX_EDGE_STRIP_WIDTH)
        .min(image.width());
    vec![
        full,
        SealView {
            region: PixelRect {
                x: 0,
                y: 0,
                width: strip_width,
                height: image.height(),
            },
            kind: SealViewKind::Boundary(OcrCanvasEdge::Left),
        },
        SealView {
            region: PixelRect {
                x: image.width() - strip_width,
                y: 0,
                width: strip_width,
                height: image.height(),
            },
            kind: SealViewKind::Boundary(OcrCanvasEdge::Right),
        },
    ]
}

pub(super) fn adjacent_boundary_view(
    image: &RgbImage,
    edge: OcrCanvasEdge,
    predecessor_region: PixelRect,
    predecessor_height: u32,
) -> Option<SealView> {
    if predecessor_height == 0
        || predecessor_region.height == 0
        || predecessor_region.height > predecessor_height / 2
        || !matches!(edge, OcrCanvasEdge::Left | OcrCanvasEdge::Right)
    {
        return None;
    }
    let strip_width = MIN_EDGE_STRIP_WIDTH.min(image.width());
    let scaled_center = (u64::from(predecessor_region.y) * 2
        + u64::from(predecessor_region.height))
    .saturating_mul(u64::from(image.height()))
        / (u64::from(predecessor_height) * 2);
    let scaled_height = u64::from(predecessor_region.height)
        .saturating_mul(2)
        .saturating_mul(u64::from(image.height()))
        / u64::from(predecessor_height);
    let window_height = u32::try_from(scaled_height)
        .unwrap_or(u32::MAX)
        .clamp(320, 512)
        .min(image.height());
    let center = u32::try_from(scaled_center)
        .unwrap_or(u32::MAX)
        .min(image.height());
    let y = center
        .saturating_sub(window_height / 2)
        .min(image.height().saturating_sub(window_height));
    let x = match edge {
        OcrCanvasEdge::Left => 0,
        OcrCanvasEdge::Right => image.width().saturating_sub(strip_width),
        OcrCanvasEdge::Top | OcrCanvasEdge::Bottom => return None,
    };
    Some(SealView {
        region: PixelRect {
            x,
            y,
            width: strip_width,
            height: window_height,
        },
        kind: SealViewKind::Boundary(edge),
    })
}

pub(super) fn view_tensor(image: &RgbImage, view: SealView) -> UseResult<Vec<f32>> {
    validate_view(image, view)?;
    let crop = image::imageops::crop_imm(
        image,
        view.region.x,
        view.region.y,
        view.region.width,
        view.region.height,
    )
    .to_image();
    let resized = image::imageops::resize(
        &crop,
        INPUT_SIDE as u32,
        INPUT_SIDE as u32,
        FilterType::CatmullRom,
    );
    let plane = INPUT_SIDE * INPUT_SIDE;
    let mut tensor = vec![0.0_f32; 3 * plane];
    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            tensor[channel * plane + index] =
                (value - CHANNEL_MEAN[channel]) / CHANNEL_STD[channel];
        }
    }
    Ok(tensor)
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
    fn page_views_are_full_left_and_right_in_canonical_order() {
        let image = RgbImage::new(1_200, 1_600);
        let views = page_views(&image);
        assert_eq!(views.len(), 3);
        assert_eq!(views[0].kind, SealViewKind::FullPage);
        assert_eq!(views[1].kind, SealViewKind::Boundary(OcrCanvasEdge::Left));
        assert_eq!(views[2].kind, SealViewKind::Boundary(OcrCanvasEdge::Right));
        assert_eq!(views[1].region.width, 100);
        assert_eq!(views[2].region.x, 1_100);
    }

    #[test]
    fn tensor_uses_rgb_imagenet_normalization() {
        let image = RgbImage::from_pixel(2, 2, Rgb([255, 0, 0]));
        let view = page_views(&image)[0];
        let tensor = view_tensor(&image, view).unwrap();
        let plane = INPUT_SIDE * INPUT_SIDE;
        assert!((tensor[0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((tensor[plane] - (0.0 - 0.456) / 0.224).abs() < 1e-5);
        assert!((tensor[2 * plane] - (0.0 - 0.406) / 0.225).abs() < 1e-5);
    }

    #[test]
    fn adjacent_view_maps_the_predecessor_band_without_scanning_the_page() {
        let image = RgbImage::new(1_190, 1_684);
        let view = adjacent_boundary_view(
            &image,
            OcrCanvasEdge::Right,
            PixelRect {
                x: 1_130,
                y: 790,
                width: 60,
                height: 206,
            },
            1_684,
        )
        .unwrap();
        assert_eq!(view.region.x, 1_126);
        assert_eq!(view.region.width, 64);
        assert_eq!(view.region.height, 412);
        assert!(view.region.y <= 790);
        assert!(view.region.y + view.region.height >= 996);
    }
}
