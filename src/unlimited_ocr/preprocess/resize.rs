use a3s_use_core::UseResult;
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use crate::cancellation::check_cancelled;

use super::image_error;

const PRECISION_BITS: u32 = 22;
const PRECISION_SCALE: i64 = 1_i64 << PRECISION_BITS;
const ROUNDING_BIAS: i64 = 1_i64 << (PRECISION_BITS - 1);

#[derive(Debug)]
struct Kernel {
    start: usize,
    coefficients: Vec<i32>,
}

/// Resize an RGB image with the same separable, fixed-point bicubic path used
/// by Pillow for 8-bit images.
pub(super) fn pillow_bicubic_rgb(
    image: &RgbImage,
    target_width: u32,
    target_height: u32,
    work_pixel_limit: u64,
    cancellation: &CancellationToken,
) -> UseResult<RgbImage> {
    check_cancelled(cancellation)?;
    if image.width() == 0 || image.height() == 0 || target_width == 0 || target_height == 0 {
        return Err(image_error(
            "Unlimited-OCR bicubic resize dimensions must be non-zero.",
        ));
    }
    if image.width() == target_width && image.height() == target_height {
        return Ok(image.clone());
    }

    let target_pixels = u64::from(target_width)
        .checked_mul(u64::from(target_height))
        .ok_or_else(|| image_error("Unlimited-OCR resized image dimensions overflowed."))?;
    if target_pixels > work_pixel_limit {
        return Err(work_limit_error("output", target_pixels, work_pixel_limit));
    }

    let horizontal = if image.width() != target_width {
        Some(precompute_axis(
            image.width(),
            target_width,
            work_pixel_limit,
            cancellation,
        )?)
    } else {
        None
    };
    let vertical = if image.height() != target_height {
        Some(precompute_axis(
            image.height(),
            target_height,
            work_pixel_limit,
            cancellation,
        )?)
    } else {
        None
    };

    if let Some(horizontal) = horizontal.as_deref() {
        let (first_row, last_row) = vertical
            .as_deref()
            .map(required_rows)
            .unwrap_or((0, image.height() as usize));
        let intermediate_height = last_row.saturating_sub(first_row);
        let intermediate_pixels = u64::from(target_width)
            .checked_mul(intermediate_height as u64)
            .ok_or_else(|| {
                image_error("Unlimited-OCR bicubic intermediate dimensions overflowed.")
            })?;
        if intermediate_pixels > work_pixel_limit {
            return Err(work_limit_error(
                "intermediate",
                intermediate_pixels,
                work_pixel_limit,
            ));
        }

        let intermediate = horizontal_pass(
            image.as_raw(),
            image.width() as usize,
            first_row,
            last_row,
            horizontal,
            target_width as usize,
            cancellation,
        )?;
        if let Some(vertical) = vertical.as_deref() {
            vertical_pass(
                &intermediate,
                target_width as usize,
                target_width,
                target_height,
                vertical,
                first_row,
                cancellation,
            )
        } else {
            image_from_raw(target_width, target_height, intermediate)
        }
    } else if let Some(vertical) = vertical.as_deref() {
        vertical_pass(
            image.as_raw(),
            image.width() as usize,
            target_width,
            target_height,
            vertical,
            0,
            cancellation,
        )
    } else {
        Ok(image.clone())
    }
}

fn precompute_axis(
    input_size: u32,
    output_size: u32,
    coefficient_limit: u64,
    cancellation: &CancellationToken,
) -> UseResult<Vec<Kernel>> {
    // Pillow passes the crop box through a C float before its coefficient
    // builder promotes the extent back to double precision.
    let input_extent = f64::from(input_size as f32);
    let scale = input_extent / f64::from(output_size);
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let mut kernels = Vec::new();
    kernels
        .try_reserve_exact(output_size as usize)
        .map_err(|error| {
            image_error(format!(
                "Failed to allocate Unlimited-OCR bicubic kernels: {error}"
            ))
        })?;
    let mut coefficient_count = 0_u64;

    for output_index in 0..output_size {
        check_cancelled(cancellation)?;
        let center = (f64::from(output_index) + 0.5) * scale;
        let start = ((center - support + 0.5) as i64).clamp(0, i64::from(input_size)) as usize;
        let end = ((center + support + 0.5) as i64).clamp(0, i64::from(input_size)) as usize;
        let coefficient_len = end.saturating_sub(start);
        coefficient_count = coefficient_count
            .checked_add(coefficient_len as u64)
            .ok_or_else(|| image_error("Unlimited-OCR bicubic kernel count overflowed."))?;
        if coefficient_count > coefficient_limit {
            return Err(work_limit_error(
                "coefficient",
                coefficient_count,
                coefficient_limit,
            ));
        }

        let mut weights = Vec::new();
        weights
            .try_reserve_exact(coefficient_len)
            .map_err(|error| {
                image_error(format!(
                    "Failed to allocate an Unlimited-OCR bicubic kernel: {error}"
                ))
            })?;
        let mut weight_sum = 0.0_f64;
        for source_index in start..end {
            let distance = (source_index as f64 - center + 0.5) / filter_scale;
            let weight = cubic_filter(distance);
            weights.push(weight);
            weight_sum += weight;
        }

        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(coefficient_len)
            .map_err(|error| {
                image_error(format!(
                    "Failed to quantize an Unlimited-OCR bicubic kernel: {error}"
                ))
            })?;
        for weight in weights {
            let normalized = if weight_sum == 0.0 {
                weight
            } else {
                weight / weight_sum
            };
            let scaled = normalized * PRECISION_SCALE as f64;
            let rounded = if normalized < 0.0 {
                scaled - 0.5
            } else {
                scaled + 0.5
            };
            coefficients.push(rounded as i32);
        }
        kernels.push(Kernel {
            start,
            coefficients,
        });
    }
    Ok(kernels)
}

fn cubic_filter(distance: f64) -> f64 {
    let distance = distance.abs();
    if distance < 1.0 {
        ((1.5 * distance - 2.5) * distance * distance) + 1.0
    } else if distance < 2.0 {
        -0.5 * (((distance - 5.0) * distance + 8.0) * distance - 4.0)
    } else {
        0.0
    }
}

fn required_rows(kernels: &[Kernel]) -> (usize, usize) {
    let first = kernels.first().map_or(0, |kernel| kernel.start);
    let last = kernels.last().map_or(first, |kernel| {
        kernel.start.saturating_add(kernel.coefficients.len())
    });
    (first, last)
}

fn horizontal_pass(
    source: &[u8],
    source_width: usize,
    first_row: usize,
    last_row: usize,
    kernels: &[Kernel],
    target_width: usize,
    cancellation: &CancellationToken,
) -> UseResult<Vec<u8>> {
    let row_count = last_row.saturating_sub(first_row);
    let mut output = rgb_buffer(target_width, row_count, "bicubic horizontal buffer")?;
    for source_row in first_row..last_row {
        check_cancelled(cancellation)?;
        let output_row = source_row - first_row;
        for (target_x, kernel) in kernels.iter().enumerate() {
            let mut sums = [ROUNDING_BIAS; 3];
            for (offset, coefficient) in kernel.coefficients.iter().enumerate() {
                let source_index = ((source_row * source_width) + kernel.start + offset) * 3;
                for channel in 0..3 {
                    sums[channel] +=
                        i64::from(source[source_index + channel]) * i64::from(*coefficient);
                }
            }
            let output_index = ((output_row * target_width) + target_x) * 3;
            for channel in 0..3 {
                output[output_index + channel] = clip_fixed(sums[channel]);
            }
        }
    }
    Ok(output)
}

fn vertical_pass(
    source: &[u8],
    source_width: usize,
    target_width: u32,
    target_height: u32,
    kernels: &[Kernel],
    source_row_offset: usize,
    cancellation: &CancellationToken,
) -> UseResult<RgbImage> {
    let mut output = rgb_buffer(
        target_width as usize,
        target_height as usize,
        "bicubic output buffer",
    )?;
    for (target_y, kernel) in kernels.iter().enumerate() {
        check_cancelled(cancellation)?;
        let kernel_start = kernel
            .start
            .checked_sub(source_row_offset)
            .ok_or_else(|| image_error("Unlimited-OCR bicubic row bounds became inconsistent."))?;
        for target_x in 0..target_width as usize {
            let mut sums = [ROUNDING_BIAS; 3];
            for (offset, coefficient) in kernel.coefficients.iter().enumerate() {
                let source_index = (((kernel_start + offset) * source_width) + target_x) * 3;
                for channel in 0..3 {
                    sums[channel] +=
                        i64::from(source[source_index + channel]) * i64::from(*coefficient);
                }
            }
            let output_index = ((target_y * target_width as usize) + target_x) * 3;
            for channel in 0..3 {
                output[output_index + channel] = clip_fixed(sums[channel]);
            }
        }
    }
    image_from_raw(target_width, target_height, output)
}

fn rgb_buffer(width: usize, height: usize, purpose: &str) -> UseResult<Vec<u8>> {
    let bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| image_error(format!("Unlimited-OCR {purpose} dimensions overflowed.")))?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(bytes).map_err(|error| {
        image_error(format!(
            "Failed to allocate the Unlimited-OCR {purpose}: {error}"
        ))
    })?;
    buffer.resize(bytes, 0);
    Ok(buffer)
}

fn image_from_raw(width: u32, height: u32, bytes: Vec<u8>) -> UseResult<RgbImage> {
    RgbImage::from_raw(width, height, bytes).ok_or_else(|| {
        image_error("Unlimited-OCR bicubic output did not match its declared dimensions.")
    })
}

fn clip_fixed(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

fn work_limit_error(kind: &str, pixels: u64, limit: u64) -> a3s_use_core::UseError {
    image_error(format!(
        "Unlimited-OCR bicubic {kind} work requires {pixels} entries, exceeding the {limit} entry preprocessing limit."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cancellation() -> CancellationToken {
        CancellationToken::new()
    }

    #[test]
    fn matches_pillow_bicubic_upscale_pixels() {
        let input = RgbImage::from_raw(
            2,
            2,
            vec![0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255],
        )
        .unwrap();
        let resized = pillow_bicubic_rgb(&input, 3, 3, 9, &cancellation()).unwrap();
        let values = resized.pixels().map(|pixel| pixel[0]).collect::<Vec<_>>();
        assert_eq!(values, [0, 21, 56, 60, 112, 162, 128, 203, 255]);
    }

    #[test]
    fn matches_pillow_bicubic_downscale_pixels() {
        let values = (0_u16..20)
            .map(|index| ((index * 37 + (index / 5) * 19) % 256) as u8)
            .flat_map(|value| [value, (u16::from(value) * 3 % 256) as u8, 255 - value])
            .collect::<Vec<_>>();
        let input = RgbImage::from_raw(5, 4, values).unwrap();
        let resized = pillow_bicubic_rgb(&input, 3, 2, 20, &cancellation()).unwrap();
        assert_eq!(
            resized.into_raw(),
            [
                126, 102, 129, 83, 146, 172, 90, 110, 165, 158, 122, 97, 165, 86, 90, 122, 130,
                133,
            ]
        );
    }

    #[test]
    fn cancellation_stops_resize_before_work() {
        let cancellation = cancellation();
        cancellation.cancel();
        let input = RgbImage::new(2, 2);
        let error = pillow_bicubic_rgb(&input, 3, 3, 9, &cancellation).unwrap_err();
        assert_eq!(error.code, "use.ocr.runtime_failed");
    }
}
