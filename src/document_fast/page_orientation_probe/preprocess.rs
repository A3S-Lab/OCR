use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use rayon::prelude::*;

pub(super) const INPUT_SIDE: usize = 224;
pub(super) const INPUT_ELEMENTS_PER_IMAGE: usize = 3 * INPUT_SIDE * INPUT_SIDE;
const RESIZE_SHORT: u32 = 256;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STANDARD_DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];
const SCALE: f32 = 1.0 / 255.0;

#[cfg(test)]
pub(super) fn batch(images: &[&RgbImage], maximum_batch_size: usize) -> UseResult<Vec<f32>> {
    let oriented = images
        .iter()
        .map(|image| (*image, 0_u8))
        .collect::<Vec<_>>();
    batch_oriented(&oriented, maximum_batch_size)
}

pub(super) fn batch_oriented(
    images: &[(&RgbImage, u8)],
    maximum_batch_size: usize,
) -> UseResult<Vec<f32>> {
    if images.is_empty() || images.len() > maximum_batch_size {
        return Err(preprocess_error(format!(
            "Document-orientation batches must contain from 1 through {maximum_batch_size} images."
        )));
    }
    if images
        .iter()
        .any(|(image, turns)| image.width() == 0 || image.height() == 0 || *turns > 3)
    {
        return Err(preprocess_error(
            "Document-orientation input images must have positive dimensions and at most three quarter turns.",
        ));
    }
    let slot_elements = INPUT_ELEMENTS_PER_IMAGE;
    let mut output = vec![0.0_f32; images.len() * slot_elements];
    images
        .par_iter()
        .zip(output.par_chunks_mut(slot_elements))
        .try_for_each(|((image, turns), slot)| write_image(image, *turns, slot))?;
    Ok(output)
}

fn write_image(image: &RgbImage, quarter_turns: u8, output: &mut [f32]) -> UseResult<()> {
    let raw_width = image.width() as usize;
    let raw_height = image.height() as usize;
    match quarter_turns {
        0 => write_oriented_image(image, raw_width, raw_height, output, |x, y| {
            (y * raw_width + x) * 3
        }),
        1 => write_oriented_image(image, raw_height, raw_width, output, |x, y| {
            ((raw_height - x - 1) * raw_width + y) * 3
        }),
        2 => write_oriented_image(image, raw_width, raw_height, output, |x, y| {
            ((raw_height - y - 1) * raw_width + (raw_width - x - 1)) * 3
        }),
        3 => write_oriented_image(image, raw_height, raw_width, output, |x, y| {
            ((raw_width - y - 1) + x * raw_width) * 3
        }),
        _ => Err(preprocess_error(
            "Document-orientation preprocessing exceeded three quarter turns.",
        )),
    }
}

fn write_oriented_image(
    image: &RgbImage,
    source_width: usize,
    source_height: usize,
    output: &mut [f32],
    pixel_offset: impl Fn(usize, usize) -> usize,
) -> UseResult<()> {
    let source_width_u32 = u32::try_from(source_width)
        .map_err(|_| preprocess_error("Document-orientation width cannot be represented."))?;
    let source_height_u32 = u32::try_from(source_height)
        .map_err(|_| preprocess_error("Document-orientation height cannot be represented."))?;
    let resized_width = round_ratio_ties_even(
        u64::from(source_width_u32) * u64::from(RESIZE_SHORT),
        u64::from(source_width_u32.min(source_height_u32)),
    )?;
    let resized_height = round_ratio_ties_even(
        u64::from(source_height_u32) * u64::from(RESIZE_SHORT),
        u64::from(source_width_u32.min(source_height_u32)),
    )?;
    if resized_width < INPUT_SIDE as u32 || resized_height < INPUT_SIDE as u32 {
        return Err(preprocess_error(
            "Document-orientation resize cannot contain its center crop.",
        ));
    }
    let crop_left = (resized_width - INPUT_SIDE as u32) / 2;
    let crop_top = (resized_height - INPUT_SIDE as u32) / 2;
    let horizontal = sampling_axis(source_width_u32, resized_width, crop_left)?;
    let vertical = sampling_axis(source_height_u32, resized_height, crop_top)?;
    let plane = INPUT_SIDE * INPUT_SIDE;
    let (red, remaining) = output.split_at_mut(plane);
    let (green, blue) = remaining.split_at_mut(plane);
    let source = image.as_raw();

    for (target_y, y) in vertical.iter().enumerate() {
        for (target_x, x) in horizontal.iter().enumerate() {
            let target = target_y * INPUT_SIDE + target_x;
            let upper_left = pixel_offset(x.lower, y.lower);
            let upper_right = pixel_offset(x.upper, y.lower);
            let lower_left = pixel_offset(x.lower, y.upper);
            let lower_right = pixel_offset(x.upper, y.upper);
            for (channel, plane) in [&mut *red, &mut *green, &mut *blue].into_iter().enumerate() {
                let top = i64::from(source[upper_left + channel]) * i64::from(x.lower_weight)
                    + i64::from(source[upper_right + channel]) * i64::from(x.upper_weight);
                let bottom = i64::from(source[lower_left + channel]) * i64::from(x.lower_weight)
                    + i64::from(source[lower_right + channel]) * i64::from(x.upper_weight);
                // This is OpenCV's reviewed 8-bit vertical path: discard four
                // low horizontal bits, multiply by the 11-bit vertical
                // coefficients, discard sixteen more bits, then round the
                // remaining two fractional bits. Keeping the staged shifts is
                // required for byte identity with INTER_LINEAR.
                let upper_term = (i64::from(y.lower_weight) * (top >> 4)) >> 16;
                let lower_term = (i64::from(y.upper_weight) * (bottom >> 4)) >> 16;
                let sample = ((upper_term + lower_term + 2) >> 2).clamp(0, 255) as f32;
                plane[target] = (sample * SCALE - MEAN[channel]) / STANDARD_DEVIATION[channel];
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Sample {
    lower: usize,
    upper: usize,
    lower_weight: i32,
    upper_weight: i32,
}

fn sampling_axis(source: u32, resized: u32, crop_start: u32) -> UseResult<Vec<Sample>> {
    if source == 0 || resized == 0 {
        return Err(preprocess_error(
            "Document-orientation sampling dimensions must be positive.",
        ));
    }
    let source_max = i64::from(source - 1);
    (0..INPUT_SIDE)
        .map(|offset| {
            let destination = u64::from(crop_start)
                .checked_add(offset as u64)
                .ok_or_else(|| preprocess_error("Document-orientation crop offset overflowed."))?;
            // Match OpenCV's contract: calculate in double precision, cast
            // the coordinate to f32, and only then take its floor.
            let coordinate = (((destination as f64) + 0.5) * f64::from(source) / f64::from(resized)
                - 0.5) as f32;
            let floor = coordinate.floor() as i64;
            let (lower, upper, fraction) = if floor < 0 {
                (0, 0, 0.0_f32)
            } else if floor >= source_max {
                (source_max, source_max, 0.0_f32)
            } else {
                (floor, floor + 1, coordinate - floor as f32)
            };
            const COEFFICIENT_SCALE: i32 = 1 << 11;
            let upper_weight =
                cv_round(fraction * COEFFICIENT_SCALE as f32).clamp(0, COEFFICIENT_SCALE);
            let lower_weight =
                cv_round((1.0 - fraction) * COEFFICIENT_SCALE as f32).clamp(0, COEFFICIENT_SCALE);
            Ok(Sample {
                lower: lower as usize,
                upper: upper as usize,
                lower_weight,
                upper_weight,
            })
        })
        .collect()
}

fn round_ratio_ties_even(numerator: u64, denominator: u64) -> UseResult<u32> {
    if denominator == 0 {
        return Err(preprocess_error(
            "Document-orientation resize denominator must be positive.",
        ));
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled = remainder
        .checked_mul(2)
        .ok_or_else(|| preprocess_error("Document-orientation resize rounding overflowed."))?;
    let rounded = quotient
        + u64::from(doubled > denominator || (doubled == denominator && quotient % 2 == 1));
    u32::try_from(rounded)
        .map_err(|_| preprocess_error("Document-orientation resize dimensions overflowed."))
}

fn cv_round(value: f32) -> i32 {
    let floor = value.floor();
    let fraction = value - floor;
    let floor = floor as i32;
    if fraction < 0.5 || (fraction == 0.5 && floor % 2 == 0) {
        floor
    } else {
        floor + 1
    }
}

fn preprocess_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.orientation_preprocess_failed", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_rounding_uses_the_declared_python_ties_to_even_contract() {
        assert_eq!(round_ratio_ties_even(10, 4).unwrap(), 2);
        assert_eq!(round_ratio_ties_even(14, 4).unwrap(), 4);
        assert_eq!(round_ratio_ties_even(11, 4).unwrap(), 3);
        assert_eq!(round_ratio_ties_even(9, 4).unwrap(), 2);
    }

    #[test]
    fn fused_crop_preserves_constant_rgb_channels() {
        let image = RgbImage::from_pixel(503, 701, image::Rgb([17, 91, 233]));
        let output = batch(&[&image], 1).unwrap();
        let plane = INPUT_SIDE * INPUT_SIDE;
        for (channel, source) in [17.0_f32, 91.0, 233.0].into_iter().enumerate() {
            let expected = (source * SCALE - MEAN[channel]) / STANDARD_DEVIATION[channel];
            assert!(output[channel * plane..][..plane]
                .iter()
                .all(|actual| actual.to_bits() == expected.to_bits()));
        }
    }

    #[test]
    fn virtual_quarter_turns_are_bit_exact_with_materialized_rotations() {
        for (width, height) in [(257, 383), (383, 257), (503, 701), (704, 511)] {
            let image = RgbImage::from_fn(width, height, |x, y| {
                image::Rgb([
                    (x.wrapping_mul(17).wrapping_add(y.wrapping_mul(3)) % 251) as u8,
                    (x.wrapping_mul(5).wrapping_add(y.wrapping_mul(19)) % 253) as u8,
                    (x.wrapping_mul(23).wrapping_add(y.wrapping_mul(7)) % 255) as u8,
                ])
            });
            let materialized = [
                image.clone(),
                image::imageops::rotate90(&image),
                image::imageops::rotate180(&image),
                image::imageops::rotate270(&image),
            ];
            for (turns, expected_image) in materialized.iter().enumerate() {
                let expected = batch(&[expected_image], 1).unwrap();
                let actual = batch_oriented(&[(&image, turns as u8)], 1).unwrap();
                assert_eq!(
                    actual, expected,
                    "{width}x{height} at {turns} quarter turns"
                );
            }
        }
    }
}
