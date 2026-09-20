//! OpenCV-compatible cubic resize for model preprocessing.
//!
//! PaddleDetection exports `interp: 2`, which is OpenCV `INTER_CUBIC`.
//! OpenCV applies a half-pixel transform, an `A = -0.75` cubic kernel,
//! 11-bit fixed-point coefficients for `u8` images, replicated borders, and
//! a 22-bit rounded vertical reduction. Keeping that contract here avoids a
//! native OpenCV runtime dependency while preserving the model's training and
//! reference-inference input distribution.

use image::RgbImage;

const COEFFICIENT_BITS: u32 = 11;
const COEFFICIENT_SCALE: f32 = (1_u32 << COEFFICIENT_BITS) as f32;
const REDUCTION_ROUNDING: i64 = 1_i64 << (COEFFICIENT_BITS * 2 - 1);

#[derive(Clone, Copy)]
struct AxisTaps {
    indices: [usize; 4],
    coefficients: [i16; 4],
}

#[derive(Clone, Copy)]
pub(super) struct Crop {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) fn resize_rgb_crop_cubic_normalized_chw(
    image: &RgbImage,
    crop: Crop,
    output_width: usize,
    output_height: usize,
    mean: [f32; 3],
    standard_deviation: [f32; 3],
    output: &mut [f32],
) -> Result<(), &'static str> {
    let plane = output_width
        .checked_mul(output_height)
        .ok_or("cubic resize output area overflowed")?;
    let expected = plane
        .checked_mul(3)
        .ok_or("cubic resize output cardinality overflowed")?;
    if output.len() != expected {
        return Err("cubic resize output has the wrong cardinality");
    }
    if mean.iter().any(|value| !value.is_finite())
        || standard_deviation
            .iter()
            .any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err("cubic resize normalization is invalid");
    }
    resize_rgb_crop_cubic(image, crop, output_width, output_height, |index, pixel| {
        for channel in 0..3 {
            let scaled = f32::from(pixel[channel]) * (1.0 / 255.0);
            output[channel * plane + index] =
                (scaled - mean[channel]) / standard_deviation[channel];
        }
    })
}

fn resize_rgb_crop_cubic(
    image: &RgbImage,
    crop: Crop,
    output_width: usize,
    output_height: usize,
    mut write: impl FnMut(usize, [u8; 3]),
) -> Result<(), &'static str> {
    validate_crop(image, crop)?;
    if output_width == 0 || output_height == 0 {
        return Err("cubic resize output dimensions must be positive");
    }

    let source_width = usize::try_from(crop.width).map_err(|_| "crop width is unsupported")?;
    let source_height = usize::try_from(crop.height).map_err(|_| "crop height is unsupported")?;
    let crop_x = usize::try_from(crop.x).map_err(|_| "crop x is unsupported")?;
    let crop_y = usize::try_from(crop.y).map_err(|_| "crop y is unsupported")?;
    let image_width = usize::try_from(image.width()).map_err(|_| "image width is unsupported")?;
    let x_taps = axis_taps(source_width, output_width);
    let y_taps = axis_taps(source_height, output_height);
    let row_values = output_width
        .checked_mul(3)
        .ok_or("cubic resize row cardinality overflowed")?;
    let mut horizontal: [Vec<i32>; 4] = std::array::from_fn(|_| vec![0_i32; row_values]);
    let mut cached_source_rows = [None; 4];
    let source = image.as_raw();

    for (destination_y, vertical) in y_taps.iter().enumerate() {
        let mut horizontal_slots = [0_usize; 4];
        for (tap, &source_y) in vertical.indices.iter().enumerate() {
            if let Some(slot) = cached_source_rows
                .iter()
                .position(|cached| *cached == Some(source_y))
            {
                horizontal_slots[tap] = slot;
                continue;
            }
            let slot = cached_source_rows
                .iter()
                .position(|cached| cached.is_none_or(|cached| !vertical.indices.contains(&cached)))
                .ok_or("cubic resize horizontal row cache is inconsistent")?;
            let absolute_y = crop_y + source_y;
            let source_row = absolute_y
                .checked_mul(image_width)
                .and_then(|value| value.checked_mul(3))
                .ok_or("cubic resize source row overflowed")?;
            let destination = &mut horizontal[slot];
            for (destination_x, horizontal_taps) in x_taps.iter().enumerate() {
                for channel in 0..3 {
                    let mut value = 0_i32;
                    for (&source_x, &coefficient) in horizontal_taps
                        .indices
                        .iter()
                        .zip(&horizontal_taps.coefficients)
                    {
                        let absolute_x = crop_x + source_x;
                        let offset = source_row + absolute_x * 3 + channel;
                        value += i32::from(source[offset]) * i32::from(coefficient);
                    }
                    destination[destination_x * 3 + channel] = value;
                }
            }
            cached_source_rows[slot] = Some(source_y);
            horizontal_slots[tap] = slot;
        }

        for destination_x in 0..output_width {
            let mut pixel = [0_u8; 3];
            for (channel, output_channel) in pixel.iter_mut().enumerate() {
                let offset = destination_x * 3 + channel;
                let mut value = 0_i64;
                for (&slot, &coefficient) in horizontal_slots.iter().zip(&vertical.coefficients) {
                    value += i64::from(horizontal[slot][offset]) * i64::from(coefficient);
                }
                let rounded = (value + REDUCTION_ROUNDING) >> (COEFFICIENT_BITS * 2);
                *output_channel = rounded.clamp(0, 255) as u8;
            }
            write(destination_y * output_width + destination_x, pixel);
        }
    }
    Ok(())
}

fn validate_crop(image: &RgbImage, crop: Crop) -> Result<(), &'static str> {
    let right = crop.x.checked_add(crop.width);
    let bottom = crop.y.checked_add(crop.height);
    if crop.width == 0
        || crop.height == 0
        || right.is_none_or(|value| value > image.width())
        || bottom.is_none_or(|value| value > image.height())
    {
        return Err("cubic resize crop is outside the source image");
    }
    Ok(())
}

fn axis_taps(source_length: usize, output_length: usize) -> Vec<AxisTaps> {
    let scale = source_length as f64 / output_length as f64;
    (0..output_length)
        .map(|destination| {
            let coordinate = (((destination as f64 + 0.5) * scale) - 0.5) as f32;
            let base = coordinate.floor() as isize;
            let fraction = coordinate - base as f32;
            let coefficients = cubic_coefficients(fraction);
            let indices = std::array::from_fn(|tap| {
                (base + tap as isize - 1).clamp(0, source_length as isize - 1) as usize
            });
            AxisTaps {
                indices,
                coefficients,
            }
        })
        .collect()
}

fn cubic_coefficients(fraction: f32) -> [i16; 4] {
    const A: f32 = -0.75;
    let shifted = fraction + 1.0;
    let complement = 1.0 - fraction;
    let first = ((A * shifted - 5.0 * A) * shifted + 8.0 * A) * shifted - 4.0 * A;
    let second = ((A + 2.0) * fraction - (A + 3.0)) * fraction * fraction + 1.0;
    let third = ((A + 2.0) * complement - (A + 3.0)) * complement * complement + 1.0;
    let fourth = 1.0 - first - second - third;
    [first, second, third, fourth].map(fixed_coefficient)
}

fn fixed_coefficient(value: f32) -> i16 {
    (value * COEFFICIENT_SCALE)
        .round_ties_even()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    fn resize_rgb_crop_cubic_uncached(
        image: &RgbImage,
        crop: Crop,
        output_width: usize,
        output_height: usize,
    ) -> Result<Vec<[u8; 3]>, &'static str> {
        validate_crop(image, crop)?;
        if output_width == 0 || output_height == 0 {
            return Err("cubic resize output dimensions must be positive");
        }

        let source_width = usize::try_from(crop.width).map_err(|_| "crop width is unsupported")?;
        let source_height =
            usize::try_from(crop.height).map_err(|_| "crop height is unsupported")?;
        let crop_x = usize::try_from(crop.x).map_err(|_| "crop x is unsupported")?;
        let crop_y = usize::try_from(crop.y).map_err(|_| "crop y is unsupported")?;
        let image_width =
            usize::try_from(image.width()).map_err(|_| "image width is unsupported")?;
        let x_taps = axis_taps(source_width, output_width);
        let y_taps = axis_taps(source_height, output_height);
        let row_values = output_width
            .checked_mul(3)
            .ok_or("cubic resize row cardinality overflowed")?;
        let source = image.as_raw();
        let mut result = Vec::with_capacity(output_width * output_height);

        for vertical in &y_taps {
            let mut horizontal: [Vec<i32>; 4] = std::array::from_fn(|_| vec![0_i32; row_values]);
            for (destination, &source_y) in horizontal.iter_mut().zip(&vertical.indices) {
                let absolute_y = crop_y + source_y;
                let source_row = absolute_y
                    .checked_mul(image_width)
                    .and_then(|value| value.checked_mul(3))
                    .ok_or("cubic resize source row overflowed")?;
                for (destination_x, horizontal_taps) in x_taps.iter().enumerate() {
                    for channel in 0..3 {
                        let mut value = 0_i32;
                        for (&source_x, &coefficient) in horizontal_taps
                            .indices
                            .iter()
                            .zip(&horizontal_taps.coefficients)
                        {
                            let absolute_x = crop_x + source_x;
                            let offset = source_row + absolute_x * 3 + channel;
                            value += i32::from(source[offset]) * i32::from(coefficient);
                        }
                        destination[destination_x * 3 + channel] = value;
                    }
                }
            }

            for destination_x in 0..output_width {
                let mut pixel = [0_u8; 3];
                for (channel, output_channel) in pixel.iter_mut().enumerate() {
                    let offset = destination_x * 3 + channel;
                    let mut value = 0_i64;
                    for (horizontal, &coefficient) in horizontal.iter().zip(&vertical.coefficients)
                    {
                        value += i64::from(horizontal[offset]) * i64::from(coefficient);
                    }
                    let rounded = (value + REDUCTION_ROUNDING) >> (COEFFICIENT_BITS * 2);
                    *output_channel = rounded.clamp(0, 255) as u8;
                }
                result.push(pixel);
            }
        }
        Ok(result)
    }

    #[test]
    fn cubic_u8_path_matches_opencv_scalar_reference_on_an_aspect_change() {
        let source = [
            0, 109, 218, 37, 146, 255, 74, 183, 36, 111, 220, 73, 148, 1, 110, 71, 180, 33, 121,
            230, 83, 171, 24, 133, 221, 74, 183, 15, 124, 233, 142, 251, 104, 205, 58, 167, 12,
            121, 230, 75, 184, 37, 138, 247, 100, 213, 66, 175, 33, 142, 251, 109, 218, 71, 185,
            38, 147, 5, 114, 223,
        ];
        let image = RgbImage::from_raw(5, 4, source.to_vec()).unwrap();
        let expected = [
            0, 100, 231, 11, 117, 255, 41, 156, 208, 66, 197, 28, 87, 252, 39, 130, 135, 82, 164,
            0, 102, 26, 133, 116, 46, 188, 152, 91, 181, 140, 137, 94, 75, 183, 130, 116, 138, 105,
            166, 61, 44, 187, 82, 199, 22, 118, 226, 48, 150, 160, 105, 155, 22, 158, 214, 48, 164,
            136, 117, 193, 9, 160, 227, 120, 255, 80, 194, 151, 110, 168, 58, 182, 28, 96, 232, 60,
            150, 95, 112, 214, 56, 133, 248, 115, 189, 172, 141, 159, 108, 193, 96, 102, 213, 46,
            179, 160, 98, 133, 89, 110, 145, 99, 80, 198, 155, 235, 42, 175, 92, 99, 247, 19, 189,
            209, 117, 226, 57, 200, 73, 113, 118, 37, 199, 0, 108, 240,
        ];
        let mut actual = Vec::new();
        resize_rgb_crop_cubic(
            &image,
            Crop {
                x: 0,
                y: 0,
                width: 5,
                height: 4,
            },
            7,
            6,
            |_, pixel| actual.extend_from_slice(&pixel),
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn horizontal_row_cache_matches_uncached_reduction_across_geometries() {
        let image = RgbImage::from_fn(19, 17, |x, y| {
            Rgb([
                ((x * 37 + y * 17 + 3) % 256) as u8,
                ((x * 11 + y * 53 + 101) % 256) as u8,
                ((x * 71 + y * 7 + 211) % 256) as u8,
            ])
        });
        let cases = [
            (
                Crop {
                    x: 0,
                    y: 0,
                    width: 19,
                    height: 17,
                },
                1,
                1,
            ),
            (
                Crop {
                    x: 1,
                    y: 2,
                    width: 13,
                    height: 11,
                },
                3,
                5,
            ),
            (
                Crop {
                    x: 2,
                    y: 1,
                    width: 9,
                    height: 14,
                },
                23,
                7,
            ),
            (
                Crop {
                    x: 5,
                    y: 4,
                    width: 1,
                    height: 1,
                },
                8,
                9,
            ),
            (
                Crop {
                    x: 0,
                    y: 0,
                    width: 19,
                    height: 17,
                },
                31,
                29,
            ),
            (
                Crop {
                    x: 3,
                    y: 3,
                    width: 11,
                    height: 9,
                },
                4,
                21,
            ),
        ];

        for (crop, output_width, output_height) in cases {
            let expected =
                resize_rgb_crop_cubic_uncached(&image, crop, output_width, output_height).unwrap();
            let mut actual = Vec::new();
            resize_rgb_crop_cubic(&image, crop, output_width, output_height, |_, pixel| {
                actual.push(pixel)
            })
            .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn normalized_chw_is_channel_planar_without_intermediate_rgb_storage() {
        let image = RgbImage::from_pixel(2, 2, Rgb([255, 0, 127]));
        let mut output = vec![0.0; 3 * 3 * 3];
        resize_rgb_crop_cubic_normalized_chw(
            &image,
            Crop {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            3,
            3,
            [0.0; 3],
            [1.0; 3],
            &mut output,
        )
        .unwrap();
        assert!(output[..9].iter().all(|value| *value == 1.0));
        assert!(output[9..18].iter().all(|value| *value == 0.0));
        assert!(output[18..]
            .iter()
            .all(|value| (*value - 127.0 / 255.0).abs() <= f32::EPSILON));
    }

    #[test]
    fn invalid_crop_and_output_contracts_fail_closed() {
        let image = RgbImage::new(4, 4);
        let mut output = vec![0.0; 3];
        assert!(resize_rgb_crop_cubic_normalized_chw(
            &image,
            Crop {
                x: 3,
                y: 0,
                width: 2,
                height: 1,
            },
            1,
            1,
            [0.0; 3],
            [1.0; 3],
            &mut output,
        )
        .is_err());
        assert!(resize_rgb_crop_cubic_normalized_chw(
            &image,
            Crop {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            2,
            2,
            [0.0; 3],
            [1.0; 3],
            &mut output,
        )
        .is_err());
    }
}
