use a3s_power::inference::{
    MicrobatchCandidate, MicrobatchPolicy, ModelSession, RuntimeDeviceKind,
};
use a3s_use_core::UseResult;
use sha2::{Digest, Sha256};

use super::{power_error, runtime_error, update_text, PreparedSlot};
use crate::batch::MAX_BATCH_SLOTS;
use crate::engine::detection_cohort_ranges;
use crate::ppocr_v6::PpOcrV6Session;
use crate::preprocess::detection_canvas_dimensions;

const HOST_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const DEVICE_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const CPU_SLOT_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
const ACCELERATOR_HOST_SLOT_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const ACCELERATOR_DEVICE_SLOT_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

pub(super) fn microbatch_candidates(
    session: &ModelSession<PpOcrV6Session>,
    slots: &[PreparedSlot],
) -> UseResult<Vec<MicrobatchCandidate>> {
    let images = slots
        .iter()
        .map(|slot| slot.image.as_ref())
        .collect::<Vec<_>>();
    let ranges = detection_cohort_ranges(&images, session.runtime().limits().max_tensor_elements)?;
    let mut candidates = Vec::with_capacity(slots.len());
    for range in ranges {
        let (canvas_width, canvas_height) = detection_canvas_dimensions(&images[range.clone()])?;
        let canvas_tensor_bytes = u64::from(canvas_width)
            .checked_mul(u64::from(canvas_height))
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| runtime_error("PP-OCRv6 detection canvas bytes overflowed."))?;
        for slot in &slots[range] {
            candidates.push(microbatch_candidate(session, slot, canvas_tensor_bytes)?);
        }
    }
    Ok(candidates)
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
    let device_kind = session.runtime().device().identity().kind;
    let accelerator = device_kind != RuntimeDeviceKind::Cpu;
    let policy = MicrobatchPolicy::new(
        maximum_microbatch_items(device_kind),
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

fn maximum_microbatch_items(_device_kind: RuntimeDeviceKind) -> usize {
    #[cfg(test)]
    if let Some(items) = std::env::var_os("A3S_OCR_TEST_MAX_MICROBATCH_ITEMS")
        .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
    {
        return items.clamp(1, MAX_BATCH_SLOTS);
    }
    // Slot count is only the protocol ceiling. Power derives the actual batch
    // from current host/device memory plus each slot's conservative scratch
    // and canvas declarations; a second empirical item cap would fragment
    // otherwise compatible recognition work before resource admission.
    MAX_BATCH_SLOTS
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
    fn outer_microbatch_limit_follows_the_runtime_device() {
        assert_eq!(
            maximum_microbatch_items(RuntimeDeviceKind::Cpu),
            MAX_BATCH_SLOTS
        );
        assert_eq!(
            maximum_microbatch_items(RuntimeDeviceKind::Cuda),
            MAX_BATCH_SLOTS
        );
        assert_eq!(
            maximum_microbatch_items(RuntimeDeviceKind::Metal),
            MAX_BATCH_SLOTS
        );
    }
}
