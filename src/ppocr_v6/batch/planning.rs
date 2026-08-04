use a3s_power::inference::{
    MicrobatchCandidate, MicrobatchPolicy, ModelSession, RuntimeDeviceKind,
};
use a3s_use_core::UseResult;
use sha2::{Digest, Sha256};

use super::{power_error, runtime_error, update_text, PreparedSlot};
use crate::ppocr_v6::PpOcrV6Session;

const MAX_MICROBATCH_ITEMS: usize = 8;
const HOST_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const DEVICE_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const CPU_SLOT_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
const ACCELERATOR_HOST_SLOT_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const ACCELERATOR_DEVICE_SLOT_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

pub(super) fn microbatch_candidate(
    session: &ModelSession<PpOcrV6Session>,
    slot: &PreparedSlot,
) -> UseResult<MicrobatchCandidate> {
    let raw_bytes = slot.input.source().size;
    let decoded_bytes = u64::from(slot.image.width())
        .checked_mul(u64::from(slot.image.height()))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| runtime_error("PP-OCRv6 decoded image bytes overflowed."))?;
    let (host_scratch, device_peak_bytes) = match session.runtime().device().identity().kind {
        RuntimeDeviceKind::Cpu => (CPU_SLOT_SCRATCH_BYTES, 0),
        RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal => (
            ACCELERATOR_HOST_SLOT_SCRATCH_BYTES,
            ACCELERATOR_DEVICE_SLOT_SCRATCH_BYTES,
        ),
    };
    let host_peak_bytes = raw_bytes
        .checked_add(decoded_bytes)
        .and_then(|bytes| bytes.checked_add(host_scratch))
        .ok_or_else(|| runtime_error("PP-OCRv6 slot memory declaration overflowed."))?;
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
