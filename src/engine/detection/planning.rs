use std::ops::Range;

use a3s_use_core::UseResult;
use image::RgbImage;

use super::super::engine_error;
use crate::preprocess::{detection_canvas_dimensions, detection_dimensions};

const MAX_DETECTION_COHORT_ITEMS: usize = 16;
// The reviewed detection graph reaches its largest activation at Concat.0:
// 12 elements per pixel in the shared detection canvas for every batch slot.
// Cohorts must fit this intermediate tensor, not just their input tensor.
const PEAK_TENSOR_ELEMENTS_PER_CANVAS_PIXEL: usize = 12;

pub(crate) fn detection_cohort_ranges(
    images: &[&RgbImage],
    max_tensor_elements: usize,
) -> UseResult<Vec<Range<usize>>> {
    if images.is_empty() {
        return Err(engine_error(
            "use.ocr.provider_input_invalid",
            "PP-OCRv6 detection cohort planning requires at least one image.",
        ));
    }
    if max_tensor_elements == 0 {
        return Err(engine_error(
            "use.ocr.provider_input_invalid",
            "PP-OCRv6 detection cohort planning requires a positive tensor element limit.",
        ));
    }
    let dimensions = images
        .iter()
        .map(|image| detection_dimensions(image.width(), image.height()))
        .collect::<UseResult<Vec<_>>>()?;
    let mut ranges = Vec::new();
    let mut start = 0_usize;
    let mut cohort = Vec::with_capacity(MAX_DETECTION_COHORT_ITEMS);
    for (index, dimensions) in dimensions.into_iter().enumerate() {
        if !cohort.is_empty()
            && (cohort.len() == MAX_DETECTION_COHORT_ITEMS
                || !combined_canvas_does_not_add_work(&cohort, dimensions)
                || !peak_tensor_fits(&cohort, dimensions, max_tensor_elements))
        {
            ranges.push(start..index);
            start = index;
            cohort.clear();
        }
        cohort.push(dimensions);
    }
    ranges.push(start..images.len());
    Ok(ranges)
}

pub(super) fn detection_cohort_peak_elements(images: &[&RgbImage]) -> UseResult<usize> {
    let (canvas_width, canvas_height) = detection_canvas_dimensions(images)?;
    usize::try_from(u64::from(canvas_width) * u64::from(canvas_height))
        .ok()
        .and_then(|pixels| pixels.checked_mul(PEAK_TENSOR_ELEMENTS_PER_CANVAS_PIXEL))
        .and_then(|slot_elements| slot_elements.checked_mul(images.len()))
        .ok_or_else(|| {
            engine_error(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 detection cohort reservation overflowed.",
            )
        })
}

fn peak_tensor_fits(
    cohort: &[(u32, u32)],
    candidate: (u32, u32),
    max_tensor_elements: usize,
) -> bool {
    let canvas_width = cohort
        .iter()
        .map(|dimensions| dimensions.0)
        .chain(std::iter::once(candidate.0))
        .max()
        .unwrap_or(candidate.0);
    let canvas_height = cohort
        .iter()
        .map(|dimensions| dimensions.1)
        .chain(std::iter::once(candidate.1))
        .max()
        .unwrap_or(candidate.1);
    usize::try_from(u64::from(canvas_width) * u64::from(canvas_height))
        .ok()
        .and_then(|pixels| pixels.checked_mul(PEAK_TENSOR_ELEMENTS_PER_CANVAS_PIXEL))
        .and_then(|slot_elements| slot_elements.checked_mul(cohort.len() + 1))
        .is_some_and(|elements| elements <= max_tensor_elements)
}

fn combined_canvas_does_not_add_work(cohort: &[(u32, u32)], candidate: (u32, u32)) -> bool {
    let current_canvas_width = cohort
        .iter()
        .map(|dimensions| dimensions.0)
        .max()
        .unwrap_or(candidate.0);
    let current_canvas_height = cohort
        .iter()
        .map(|dimensions| dimensions.1)
        .max()
        .unwrap_or(candidate.1);
    let combined_canvas_width = cohort
        .iter()
        .map(|dimensions| dimensions.0)
        .chain(std::iter::once(candidate.0))
        .max()
        .unwrap_or(candidate.0);
    let combined_canvas_height = cohort
        .iter()
        .map(|dimensions| dimensions.1)
        .chain(std::iter::once(candidate.1))
        .max()
        .unwrap_or(candidate.1);
    let current_canvas_area = u128::from(current_canvas_width) * u128::from(current_canvas_height);
    let candidate_area = u128::from(candidate.0) * u128::from(candidate.1);
    let combined_canvas_area =
        u128::from(combined_canvas_width) * u128::from(combined_canvas_height);
    let separate_work = current_canvas_area * cohort.len() as u128 + candidate_area;
    let combined_work = combined_canvas_area * (cohort.len() + 1) as u128;

    combined_work <= separate_work
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohorts_only_fuse_shapes_without_additional_canvas_work() {
        let wide = RgbImage::new(320, 288);
        let square = RgbImage::new(320, 320);
        let second_square = RgbImage::new(320, 320);
        let tall = RgbImage::new(256, 320);

        let ranges =
            detection_cohort_ranges(&[&wide, &square, &second_square, &tall], usize::MAX).unwrap();

        assert_eq!(ranges, vec![0..1, 1..3, 3..4]);
    }

    #[test]
    fn cohorts_never_exceed_the_reviewed_graph_batch_limit() {
        let image = RgbImage::new(320, 320);
        let images = std::iter::repeat_n(&image, 17).collect::<Vec<_>>();

        let ranges = detection_cohort_ranges(&images, usize::MAX).unwrap();

        assert_eq!(ranges, vec![0..16, 16..17]);
    }

    #[test]
    fn cohorts_respect_the_intermediate_tensor_limit() {
        let image = RgbImage::new(1_224, 1_584);
        let images = std::iter::repeat_n(&image, 16).collect::<Vec<_>>();
        let (width, height) = detection_dimensions(image.width(), image.height()).unwrap();
        let eleven_slot_limit = usize::try_from(u64::from(width) * u64::from(height)).unwrap()
            * PEAK_TENSOR_ELEMENTS_PER_CANVAS_PIXEL
            * 11;

        let ranges = detection_cohort_ranges(&images, eleven_slot_limit).unwrap();

        assert_eq!(ranges, vec![0..11, 11..16]);
    }

    #[test]
    fn smaller_canvases_retain_the_full_detection_batch() {
        let image = RgbImage::new(816, 1_056);
        let images = std::iter::repeat_n(&image, 16).collect::<Vec<_>>();

        let ranges = detection_cohort_ranges(&images, 256 * 1024 * 1024).unwrap();

        assert_eq!(ranges, vec![0..16]);
    }

    #[test]
    fn peak_reservation_matches_the_reviewed_graph_activation() {
        let first = RgbImage::new(320, 288);
        let second = RgbImage::new(320, 320);

        assert_eq!(
            detection_cohort_peak_elements(&[&first, &second]).unwrap(),
            2 * 320 * 320 * PEAK_TENSOR_ELEMENTS_PER_CANVAS_PIXEL
        );
    }
}
