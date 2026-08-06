use std::collections::BTreeSet;
use std::io::Cursor;

use a3s_power::inference::InferenceLimits;
use a3s_use_core::{UseError, UseResult};
use image::imageops;
use image::{DynamicImage, GenericImage, ImageDecoder as _, ImageReader, Limits, Rgb, RgbImage};
use tokio_util::sync::CancellationToken;

use crate::cancellation::check_cancelled;

mod resize;

use self::resize::pillow_bicubic_rgb;

pub(crate) const GLOBAL_IMAGE_SIDE: u32 = 1_024;
pub(crate) const TILE_IMAGE_SIDE: u32 = 640;
pub(crate) const PATCH_SIZE: usize = 16;
pub(crate) const DOWNSAMPLE_RATIO: usize = 4;
const MIN_TILES: u32 = 2;
const MAX_TILES: u32 = 32;
const NORMALIZATION_PAD: u8 = 127;

#[derive(Debug, Clone)]
pub(crate) struct PreprocessedImage {
    pub(crate) global: Vec<f32>,
    pub(crate) tiles: Vec<Vec<f32>>,
    pub(crate) tile_columns: u32,
    pub(crate) tile_rows: u32,
}

impl PreprocessedImage {
    pub(crate) fn image_token_count(&self) -> usize {
        let global_queries = query_side(GLOBAL_IMAGE_SIDE);
        let mut count = base_view_tokens(global_queries);
        if !self.tiles.is_empty() {
            count = count.saturating_add(tile_view_tokens(
                query_side(TILE_IMAGE_SIDE),
                self.tile_columns,
                self.tile_rows,
            ));
        }
        count
    }

    pub(crate) fn image_token_ids(&self, image_token_id: u32) -> Vec<u32> {
        vec![image_token_id; self.image_token_count()]
    }
}

pub(crate) fn preprocess(
    bytes: &[u8],
    limits: &InferenceLimits,
    cancellation: &CancellationToken,
) -> UseResult<PreprocessedImage> {
    check_cancelled(cancellation)?;
    if bytes.len() > limits.max_input_bytes {
        return Err(image_error(format!(
            "Unlimited-OCR input contains {} bytes, exceeding the {} byte embedded runtime limit.",
            bytes.len(),
            limits.max_input_bytes
        )));
    }
    let image = decode_oriented(bytes, limits)?.to_rgb8();
    check_cancelled(cancellation)?;
    let global = normalize_chw(&pad_to_square(
        &image,
        GLOBAL_IMAGE_SIDE,
        limits.max_image_pixels,
        cancellation,
    )?);
    let (tiles, tile_columns, tile_rows) = if image.width() <= TILE_IMAGE_SIDE
        && image.height() <= TILE_IMAGE_SIDE
    {
        (Vec::new(), 1, 1)
    } else {
        let (tiles, columns, rows) = dynamic_tiles(&image, limits.max_image_pixels, cancellation)?;
        (
            tiles.into_iter().map(|tile| normalize_chw(&tile)).collect(),
            columns,
            rows,
        )
    };
    Ok(PreprocessedImage {
        global,
        tiles,
        tile_columns,
        tile_rows,
    })
}

fn decode_oriented(bytes: &[u8], limits: &InferenceLimits) -> UseResult<DynamicImage> {
    let cursor = Cursor::new(bytes);
    let mut reader = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|error| image_error(format!("Failed to detect OCR image format: {error}")))?;
    let max_side = limits
        .max_image_pixels
        .clamp(1, u64::from(u32::MAX))
        .try_into()
        .unwrap_or(u32::MAX);
    let mut image_limits = Limits::default();
    image_limits.max_image_width = Some(max_side);
    image_limits.max_image_height = Some(max_side);
    image_limits.max_alloc = Some(limits.max_image_pixels.saturating_mul(4));
    reader.limits(image_limits);
    let mut decoder = reader
        .into_decoder()
        .map_err(|error| image_error(format!("Failed to initialize the image decoder: {error}")))?;
    let orientation = decoder.orientation().map_err(|error| {
        image_error(format!("Failed to read the OCR image orientation: {error}"))
    })?;
    let mut image = DynamicImage::from_decoder(decoder)
        .map_err(|error| image_error(format!("Failed to decode the OCR image: {error}")))?;
    image.apply_orientation(orientation);
    let pixels = u64::from(image.width())
        .checked_mul(u64::from(image.height()))
        .ok_or_else(|| image_error("Decoded Unlimited-OCR image dimensions overflowed."))?;
    if image.width() == 0 || image.height() == 0 || pixels > limits.max_image_pixels {
        return Err(image_error(format!(
            "Decoded Unlimited-OCR image exceeds the {} pixel embedded runtime limit.",
            limits.max_image_pixels
        )));
    }
    Ok(image)
}

fn pad_to_square(
    image: &RgbImage,
    side: u32,
    max_image_pixels: u64,
    cancellation: &CancellationToken,
) -> UseResult<RgbImage> {
    let (target_width, target_height) = contain_size(image.width(), image.height(), side)?;
    let work_pixel_limit = resize_work_pixel_limit(max_image_pixels, target_width, target_height)?;
    let resized = pillow_bicubic_rgb(
        image,
        target_width,
        target_height,
        work_pixel_limit,
        cancellation,
    )?;
    let mut canvas = RgbImage::from_pixel(
        side,
        side,
        Rgb([NORMALIZATION_PAD, NORMALIZATION_PAD, NORMALIZATION_PAD]),
    );
    let x = round_half_to_even(side.saturating_sub(resized.width()));
    let y = round_half_to_even(side.saturating_sub(resized.height()));
    canvas.copy_from(&resized, x, y).map_err(|error| {
        image_error(format!(
            "Failed to place the Unlimited-OCR global image view: {error}"
        ))
    })?;
    Ok(canvas)
}

fn dynamic_tiles(
    image: &RgbImage,
    max_image_pixels: u64,
    cancellation: &CancellationToken,
) -> UseResult<(Vec<RgbImage>, u32, u32)> {
    let (columns, rows) = closest_grid(image.width(), image.height(), TILE_IMAGE_SIDE)?;
    let target_width = TILE_IMAGE_SIDE
        .checked_mul(columns)
        .ok_or_else(|| image_error("Unlimited-OCR tile-grid width overflowed."))?;
    let target_height = TILE_IMAGE_SIDE
        .checked_mul(rows)
        .ok_or_else(|| image_error("Unlimited-OCR tile-grid height overflowed."))?;
    let work_pixel_limit = resize_work_pixel_limit(max_image_pixels, target_width, target_height)?;
    let resized = pillow_bicubic_rgb(
        image,
        target_width,
        target_height,
        work_pixel_limit,
        cancellation,
    )?;
    let count = columns
        .checked_mul(rows)
        .ok_or_else(|| image_error("Unlimited-OCR tile count overflowed."))?;
    let mut tiles = Vec::with_capacity(count as usize);
    for index in 0..count {
        check_cancelled(cancellation)?;
        let column = index % columns;
        let row = index / columns;
        tiles.push(
            imageops::crop_imm(
                &resized,
                column * TILE_IMAGE_SIDE,
                row * TILE_IMAGE_SIDE,
                TILE_IMAGE_SIDE,
                TILE_IMAGE_SIDE,
            )
            .to_image(),
        );
    }
    Ok((tiles, columns, rows))
}

fn contain_size(width: u32, height: u32, side: u32) -> UseResult<(u32, u32)> {
    if width == 0 || height == 0 || side == 0 {
        return Err(image_error(
            "Unlimited-OCR global image dimensions must be non-zero.",
        ));
    }
    if width == height {
        return Ok((side, side));
    }
    if width > height {
        let scaled_height =
            round_ties_to_even(f64::from(height) / f64::from(width) * f64::from(side));
        if scaled_height == 0 {
            return Err(image_error(
                "Unlimited-OCR global image aspect ratio produced a zero height.",
            ));
        }
        Ok((side, scaled_height))
    } else {
        let scaled_width =
            round_ties_to_even(f64::from(width) / f64::from(height) * f64::from(side));
        if scaled_width == 0 {
            return Err(image_error(
                "Unlimited-OCR global image aspect ratio produced a zero width.",
            ));
        }
        Ok((scaled_width, side))
    }
}

fn round_ties_to_even(value: f64) -> u32 {
    let lower = value.floor();
    let fraction = value - lower;
    if fraction < 0.5 || (fraction == 0.5 && lower as u64 % 2 == 0) {
        lower as u32
    } else {
        (lower + 1.0) as u32
    }
}

fn round_half_to_even(value: u32) -> u32 {
    let lower = value / 2;
    if value % 2 == 0 || lower % 2 == 0 {
        lower
    } else {
        lower + 1
    }
}

fn resize_work_pixel_limit(
    max_image_pixels: u64,
    target_width: u32,
    target_height: u32,
) -> UseResult<u64> {
    let target_pixels = u64::from(target_width)
        .checked_mul(u64::from(target_height))
        .ok_or_else(|| image_error("Unlimited-OCR resize target dimensions overflowed."))?;
    Ok(max_image_pixels.max(target_pixels))
}

fn closest_grid(width: u32, height: u32, image_side: u32) -> UseResult<(u32, u32)> {
    if width == 0 || height == 0 {
        return Err(image_error(
            "Unlimited-OCR cannot tile an image with a zero dimension.",
        ));
    }
    let aspect = f64::from(width) / f64::from(height);
    let area = u64::from(width) * u64::from(height);
    let mut ratios = BTreeSet::new();
    for product_limit in MIN_TILES..=MAX_TILES {
        for columns in 1..=product_limit {
            for rows in 1..=product_limit {
                let product = columns.saturating_mul(rows);
                if (MIN_TILES..=MAX_TILES).contains(&product) {
                    ratios.insert((columns, rows));
                }
            }
        }
    }
    let mut ratios = ratios.into_iter().collect::<Vec<_>>();
    ratios.sort_by_key(|(columns, rows)| (columns * rows, *columns, *rows));

    let mut best = (1, 1);
    let mut best_difference = f64::INFINITY;
    for (columns, rows) in ratios {
        let difference = (aspect - f64::from(columns) / f64::from(rows)).abs();
        if difference < best_difference {
            best_difference = difference;
            best = (columns, rows);
        } else if difference == best_difference {
            let candidate_area = u64::from(image_side)
                .saturating_mul(u64::from(image_side))
                .saturating_mul(u64::from(columns))
                .saturating_mul(u64::from(rows));
            if area.saturating_mul(2) > candidate_area {
                best = (columns, rows);
            }
        }
    }
    Ok(best)
}

fn normalize_chw(image: &RgbImage) -> Vec<f32> {
    let plane = image.width() as usize * image.height() as usize;
    let mut output = vec![0.0_f32; plane * 3];
    for (index, pixel) in image.pixels().enumerate() {
        for channel in 0..3 {
            output[channel * plane + index] = f32::from(pixel[channel]) / 127.5 - 1.0;
        }
    }
    output
}

const fn query_side(image_side: u32) -> usize {
    let patches = image_side as usize / PATCH_SIZE;
    patches.div_ceil(DOWNSAMPLE_RATIO)
}

const fn base_view_tokens(query_side: usize) -> usize {
    query_side
        .saturating_mul(query_side.saturating_add(1))
        .saturating_add(1)
}

const fn tile_view_tokens(query_side: usize, columns: u32, rows: u32) -> usize {
    let columns = columns as usize;
    let rows = rows as usize;
    query_side
        .saturating_mul(rows)
        .saturating_mul(query_side.saturating_mul(columns).saturating_add(1))
}

fn image_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.image_invalid", message)
}

#[cfg(test)]
mod tests {
    use image::ImageEncoder as _;

    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes)
            .write_image(
                &vec![255; (width * height * 3) as usize],
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        bytes
    }

    fn preprocess_fixture(bytes: &[u8], limits: &InferenceLimits) -> UseResult<PreprocessedImage> {
        preprocess(bytes, limits, &CancellationToken::new())
    }

    #[test]
    fn small_page_matches_the_reviewed_273_token_global_layout() {
        let image = preprocess_fixture(&png(320, 200), &InferenceLimits::default()).unwrap();
        assert_eq!(image.global.len(), 3 * 1_024 * 1_024);
        assert!(image.tiles.is_empty());
        assert_eq!(image.image_token_count(), 273);
        assert_eq!(image.image_token_ids(128_815).len(), 273);
    }

    #[test]
    fn large_page_adds_a_bounded_640_pixel_tile_grid() {
        let image = preprocess_fixture(&png(1_200, 800), &InferenceLimits::default()).unwrap();
        assert!(!image.tiles.is_empty());
        assert_eq!(
            image.tiles.len(),
            (image.tile_columns * image.tile_rows) as usize
        );
        assert!(image.tiles.len() <= MAX_TILES as usize);
        assert!(image.image_token_count() > 273);
    }

    #[test]
    fn wide_and_tall_pages_select_matching_grid_directions() {
        let wide = closest_grid(2_000, 500, TILE_IMAGE_SIDE).unwrap();
        let tall = closest_grid(500, 2_000, TILE_IMAGE_SIDE).unwrap();
        assert!(wide.0 > wide.1);
        assert!(tall.1 > tall.0);
    }

    #[test]
    fn pixel_limits_fail_before_unbounded_preprocessing() {
        let limits = InferenceLimits {
            max_image_pixels: 100,
            ..InferenceLimits::default()
        };
        assert!(preprocess_fixture(&png(20, 20), &limits).is_err());
    }

    #[test]
    fn cancelled_preprocessing_fails_before_image_work() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error =
            preprocess(&png(20, 20), &InferenceLimits::default(), &cancellation).unwrap_err();
        assert_eq!(error.code, "use.ocr.runtime_failed");
    }

    #[test]
    fn pillow_contain_uses_ties_to_even_for_size_and_centering() {
        assert_eq!(contain_size(4, 1, 10).unwrap(), (10, 2));
        assert_eq!(contain_size(4, 1, 14).unwrap(), (14, 4));
        assert!(contain_size(2_048, 1, 1_024).is_err());
        assert_eq!(round_half_to_even(1), 0);
        assert_eq!(round_half_to_even(3), 2);
        assert_eq!(round_half_to_even(5), 2);
    }

    #[test]
    fn global_view_matches_the_upstream_pillow_pad_fixture() {
        let values = (0_u16..12)
            .flat_map(|index| {
                [
                    (index * 31 % 256) as u8,
                    (index * 67 % 256) as u8,
                    (255 - index * 17) as u8,
                ]
            })
            .collect::<Vec<_>>();
        let input = RgbImage::from_raw(4, 3, values).unwrap();
        let output = pad_to_square(&input, 5, 20, &CancellationToken::new()).unwrap();
        assert_eq!(
            output.into_raw(),
            [
                0, 0, 255, 14, 42, 248, 41, 100, 234, 67, 157, 219, 88, 203, 207, 68, 5, 218, 102,
                50, 206, 135, 108, 192, 160, 165, 177, 181, 211, 165, 181, 14, 158, 130, 60, 146,
                124, 118, 132, 156, 175, 117, 177, 221, 105, 255, 22, 116, 81, 68, 104, 15, 126,
                90, 58, 183, 75, 79, 229, 63, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
                127, 127, 127, 127, 127,
            ]
        );
    }
}
