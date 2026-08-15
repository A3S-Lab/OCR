use a3s_use_core::{UseError, UseResult};
use image::RgbImage;

use super::wired::PixelRect;

pub(super) const INPUT_SIDE: usize = 488;
const CHANNEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const CHANNEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(super) fn crop_tensor(image: &RgbImage, region: PixelRect) -> UseResult<Vec<f32>> {
    validate_region(image, region)?;
    let mut tensor = vec![0.0_f32; 3 * INPUT_SIDE * INPUT_SIDE];
    let long_side = region.width.max(region.height) as f32;
    let scale = INPUT_SIDE as f32 / long_side;
    let resized_width = ((region.width as f32 * scale).round() as usize).clamp(1, INPUT_SIDE);
    let resized_height = ((region.height as f32 * scale).round() as usize).clamp(1, INPUT_SIDE);
    let inverse_scale = 1.0_f32 / scale;
    let plane = INPUT_SIDE * INPUT_SIDE;

    for destination_y in 0..resized_height {
        let source_y = (destination_y as f32 + 0.5) * inverse_scale - 0.5;
        let y_floor = source_y.floor();
        let y_fraction = source_y - y_floor;
        let y0 = clamp_sample(y_floor as i64, region.height);
        let y1 = clamp_sample(y_floor as i64 + 1, region.height);
        for destination_x in 0..resized_width {
            let source_x = (destination_x as f32 + 0.5) * inverse_scale - 0.5;
            let x_floor = source_x.floor();
            let x_fraction = source_x - x_floor;
            let x0 = clamp_sample(x_floor as i64, region.width);
            let x1 = clamp_sample(x_floor as i64 + 1, region.width);
            let samples = [
                image.get_pixel(region.x + x0, region.y + y0).0,
                image.get_pixel(region.x + x1, region.y + y0).0,
                image.get_pixel(region.x + x0, region.y + y1).0,
                image.get_pixel(region.x + x1, region.y + y1).0,
            ];
            let weights = [
                (1.0 - x_fraction) * (1.0 - y_fraction),
                x_fraction * (1.0 - y_fraction),
                (1.0 - x_fraction) * y_fraction,
                x_fraction * y_fraction,
            ];
            let destination = destination_y * INPUT_SIDE + destination_x;
            for channel in 0..3 {
                let value = samples
                    .iter()
                    .zip(weights)
                    .map(|(pixel, weight)| f32::from(pixel[channel]) * weight)
                    .sum::<f32>()
                    / 255.0;
                tensor[channel * plane + destination] =
                    (value - CHANNEL_MEAN[channel]) / CHANNEL_STD[channel];
            }
        }
    }
    Ok(tensor)
}

fn clamp_sample(value: i64, extent: u32) -> u32 {
    value.clamp(0, i64::from(extent.saturating_sub(1))) as u32
}

fn validate_region(image: &RgbImage, region: PixelRect) -> UseResult<()> {
    let right = region.x.checked_add(region.width);
    let bottom = region.y.checked_add(region.height);
    if region.width == 0
        || region.height == 0
        || right.is_none_or(|right| right > image.width())
        || bottom.is_none_or(|bottom| bottom > image.height())
    {
        return Err(UseError::new(
            "use.ocr.table_candidate_invalid",
            "A wired-table candidate must cover positive area inside its exact source canvas.",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn square_crop_uses_rgb_imagenet_normalization() {
        let image = RgbImage::from_pixel(2, 2, Rgb([255, 0, 0]));
        let tensor = crop_tensor(
            &image,
            PixelRect {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
        )
        .unwrap();
        let plane = INPUT_SIDE * INPUT_SIDE;
        assert!((tensor[0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((tensor[plane] - (0.0 - 0.456) / 0.224).abs() < 1e-5);
        assert!((tensor[2 * plane] - (0.0 - 0.406) / 0.225).abs() < 1e-5);
    }

    #[test]
    fn aspect_padding_is_zero_in_normalized_space() {
        let image = RgbImage::from_pixel(4, 2, Rgb([255, 255, 255]));
        let tensor = crop_tensor(
            &image,
            PixelRect {
                x: 0,
                y: 0,
                width: 4,
                height: 2,
            },
        )
        .unwrap();
        let padded = 400 * INPUT_SIDE;
        assert_eq!(tensor[padded], 0.0);
        assert_eq!(tensor[INPUT_SIDE * INPUT_SIDE + padded], 0.0);
        assert_eq!(tensor[2 * INPUT_SIDE * INPUT_SIDE + padded], 0.0);
    }

    #[test]
    fn crop_must_remain_inside_canvas() {
        let image = RgbImage::new(10, 10);
        let error = crop_tensor(
            &image,
            PixelRect {
                x: 9,
                y: 9,
                width: 2,
                height: 2,
            },
        )
        .unwrap_err();
        assert_eq!(error.code, "use.ocr.table_candidate_invalid");
    }
}
