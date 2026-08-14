use std::ops::Range;

use a3s_power::inference::{
    MicrobatchCandidate, MicrobatchPolicy, ModelSession, RuntimeDeviceKind,
};
use a3s_use_core::UseResult;
use image::RgbImage;
use sha2::{Digest, Sha256};

use super::{power_error, runtime_error, update_text, PreparedSlot};
use crate::ppocr_v6::PpOcrV6Session;
use crate::preprocess::{detection_canvas_dimensions, detection_dimensions};

const MAX_MICROBATCH_ITEMS: usize = 16;
// The reviewed detection graph reaches its largest activation at Concat.0:
// 12 elements per pixel in the shared detection canvas for every batch slot.
// Cohorts must fit this intermediate tensor, not just their input tensor.
const PEAK_TENSOR_ELEMENTS_PER_CANVAS_PIXEL: usize = 12;
const MIN_DETECTION_CANVAS_FILL_BPS: u64 = 9_000;
const HOST_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const DEVICE_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const CPU_SLOT_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
const ACCELERATOR_HOST_SLOT_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const ACCELERATOR_DEVICE_SLOT_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

pub(super) fn detection_cohort_ranges(
    images: &[&RgbImage],
    max_tensor_elements: usize,
) -> UseResult<Vec<Range<usize>>> {
    if images.is_empty() {
        return Err(runtime_error(
            "PP-OCRv6 detection cohort planning requires at least one image.",
        ));
    }
    if max_tensor_elements == 0 {
        return Err(runtime_error(
            "PP-OCRv6 detection cohort planning requires a positive tensor element limit.",
        ));
    }
    let dimensions = images
        .iter()
        .map(|image| detection_dimensions(image.width(), image.height()))
        .collect::<UseResult<Vec<_>>>()?;
    let mut ranges = Vec::new();
    let mut start = 0_usize;
    let mut cohort = Vec::with_capacity(MAX_MICROBATCH_ITEMS);
    for (index, dimensions) in dimensions.into_iter().enumerate() {
        if !cohort.is_empty()
            && (cohort.len() == MAX_MICROBATCH_ITEMS
                || !canvas_fill_is_compatible(&cohort, dimensions)
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

fn canvas_fill_is_compatible(cohort: &[(u32, u32)], candidate: (u32, u32)) -> bool {
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
    let canvas_area = u64::from(canvas_width) * u64::from(canvas_height);
    cohort
        .iter()
        .copied()
        .chain(std::iter::once(candidate))
        .all(|(width, height)| {
            u64::from(width) * u64::from(height) * 10_000
                >= canvas_area * MIN_DETECTION_CANVAS_FILL_BPS
        })
}

pub(super) fn microbatch_candidates(
    session: &ModelSession<PpOcrV6Session>,
    slots: &[PreparedSlot],
) -> UseResult<Vec<MicrobatchCandidate>> {
    let images = slots.iter().map(|slot| &slot.image).collect::<Vec<_>>();
    let (canvas_width, canvas_height) = detection_canvas_dimensions(&images)?;
    let canvas_tensor_bytes = u64::from(canvas_width)
        .checked_mul(u64::from(canvas_height))
        .and_then(|pixels| pixels.checked_mul(3))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| runtime_error("PP-OCRv6 detection canvas bytes overflowed."))?;
    slots
        .iter()
        .map(|slot| microbatch_candidate(session, slot, canvas_tensor_bytes))
        .collect()
}

fn microbatch_candidate(
    session: &ModelSession<PpOcrV6Session>,
    slot: &PreparedSlot,
    canvas_tensor_bytes: u64,
) -> UseResult<MicrobatchCandidate> {
    let raw_bytes = slot.input.source().size;
    let decoded_bytes = u64::from(slot.image.width())
        .checked_mul(u64::from(slot.image.height()))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| runtime_error("PP-OCRv6 decoded image bytes overflowed."))?;
    let (host_scratch, device_scratch) = match session.runtime().device().identity().kind {
        RuntimeDeviceKind::Cpu => (CPU_SLOT_SCRATCH_BYTES, 0),
        RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal => (
            ACCELERATOR_HOST_SLOT_SCRATCH_BYTES,
            ACCELERATOR_DEVICE_SLOT_SCRATCH_BYTES,
        ),
    };
    let host_peak_bytes = raw_bytes
        .checked_add(decoded_bytes)
        .and_then(|bytes| bytes.checked_add(host_scratch))
        .and_then(|bytes| bytes.checked_add(canvas_tensor_bytes))
        .ok_or_else(|| runtime_error("PP-OCRv6 slot memory declaration overflowed."))?;
    let device_peak_bytes = if device_scratch == 0 {
        0
    } else {
        device_scratch
            .checked_add(canvas_tensor_bytes)
            .ok_or_else(|| runtime_error("PP-OCRv6 device memory declaration overflowed."))?
    };
    let input_bytes = usize::try_from(raw_bytes)
        .map_err(|_| runtime_error("PP-OCRv6 input byte count cannot be represented."))?;
    MicrobatchCandidate::new(
        slot_sha256(slot)?,
        input_bytes,
        1,
        0,
        host_peak_bytes,
        device_peak_bytes,
    )
    .map_err(|error| power_error("declare a PP-OCRv6 microbatch slot", error))
}

pub(super) fn microbatch_policy(
    session: &ModelSession<PpOcrV6Session>,
    resident_bytes: u64,
) -> UseResult<MicrobatchPolicy> {
    let accelerator = session.runtime().device().identity().kind != RuntimeDeviceKind::Cpu;
    let policy = MicrobatchPolicy::new(
        MAX_MICROBATCH_ITEMS,
        7_500,
        if accelerator { 7_500 } else { 0 },
    )
    .map_err(|error| power_error("configure PP-OCRv6 microbatch memory", error))?
    .with_host_reserve_bytes(HOST_RESERVE_BYTES)
    .with_device_reserve_bytes(if accelerator { DEVICE_RESERVE_BYTES } else { 0 })
    .with_base_memory(resident_bytes, 0);
    policy
        .validate()
        .map_err(|error| power_error("validate PP-OCRv6 microbatch memory", error))?;
    Ok(policy)
}

fn slot_sha256(slot: &PreparedSlot) -> UseResult<String> {
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-ppocr-v6-batch-slot-v1\0");
    update_text(&mut digest, slot.slot_id.as_str())?;
    update_text(&mut digest, &slot.input.source().sha256)?;
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_cohorts_fuse_compatible_shapes_and_split_quality_outliers() {
        let wide = RgbImage::new(320, 288);
        let square = RgbImage::new(320, 320);
        let tall = RgbImage::new(256, 320);

        let ranges = detection_cohort_ranges(&[&wide, &square, &tall], usize::MAX).unwrap();

        assert_eq!(ranges, vec![0..2, 2..3]);
    }

    #[test]
    fn detection_cohorts_never_exceed_the_reviewed_graph_batch_limit() {
        let image = RgbImage::new(320, 320);
        let images = std::iter::repeat_n(&image, 17).collect::<Vec<_>>();

        let ranges = detection_cohort_ranges(&images, usize::MAX).unwrap();

        assert_eq!(ranges, vec![0..16, 16..17]);
    }

    #[test]
    fn detection_cohorts_respect_the_intermediate_tensor_limit() {
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
}
