use std::ops::Range;

use a3s_power::inference::RuntimeDeviceKind;
use a3s_use_core::{Readiness, UseError, UseResult};
use tokio_util::sync::CancellationToken;

use super::shared_decode::{validate_decoded_cardinality, SharedDecodedImage};
use crate::batch::MAX_BATCH_SLOTS;
use crate::config::ModelProfile;
use crate::{
    OcrInput, OcrProvider, OcrProviderBatchOutput, OcrProviderBatchRequest, OcrProviderOutput,
    OcrProviderStatus, PpOcrV6Provider,
};

const MAX_CUDA_TEXT_LANES: usize = 3;
const CUDA_DEVICE_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CUDA_TEXT_LANE_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone)]
pub(super) struct TextLanes {
    lanes: Vec<PpOcrV6Provider>,
    slots_per_lane: usize,
    profile: ModelProfile,
}

impl TextLanes {
    pub(super) fn from_env() -> UseResult<Self> {
        let first = PpOcrV6Provider::from_env_bound()?;
        let profile = first.configured_model_profile()?;
        let device_kind = first.runtime_device_kind();
        let lane_count = recommended_lane_count(&first)?;
        let mut lanes = Vec::with_capacity(lane_count);
        for execution_replica in 0..lane_count {
            lanes.push(first.execution_replica(execution_replica)?);
        }
        Ok(Self {
            lanes,
            slots_per_lane: slots_per_lane(device_kind),
            profile,
        })
    }

    pub(super) fn primary(&self) -> &PpOcrV6Provider {
        &self.lanes[0]
    }

    pub(super) const fn model_profile(&self) -> ModelProfile {
        self.profile
    }

    pub(super) fn diagnostic(&self) -> OcrProviderStatus {
        let mut status = self.primary().diagnostic();
        let expected = self.profile.family();
        if status.readiness == Readiness::Ready && status.model.as_deref() != Some(expected) {
            let observed = status.model.as_deref().unwrap_or("undeclared");
            status.readiness = Readiness::Broken;
            status.message = format!(
                "The bound PP-OCRv6 profile is {expected}, but provider diagnostics reported {observed}."
            );
            status.suggestions = vec![
                "Create a new document-fast provider after restoring or intentionally replacing the OCR model bundle."
                    .to_string(),
            ];
        }
        status.model = Some(expected.to_string());
        status
    }

    pub(super) async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
        let output = self.primary().recognize(input).await?;
        self.validate_output_model(&output)?;
        Ok(output)
    }

    pub(super) async fn recognize_batch_decoded(
        &self,
        request: OcrProviderBatchRequest,
        decoded: Vec<SharedDecodedImage>,
        cancellation: &CancellationToken,
    ) -> UseResult<OcrProviderBatchOutput> {
        validate_decoded_cardinality(&request.slots, &decoded)?;
        let mut slots = Vec::with_capacity(request.slots.len());
        let mut execution_receipts = Vec::new();
        let work = lane_ranges(request.slots.len(), self.slots_per_lane);

        for window in work.chunks(self.lanes.len()) {
            let helpers = &self.lanes[window.len()..];
            let first = self.run_lane(0, window.first(), helpers, &request, &decoded, cancellation);
            let second = self.run_lane(1, window.get(1), helpers, &request, &decoded, cancellation);
            let third = self.run_lane(2, window.get(2), helpers, &request, &decoded, cancellation);
            let completed = tokio::join!(first, second, third);
            for output in [completed.0?, completed.1?, completed.2?]
                .into_iter()
                .flatten()
            {
                slots.extend(output.slots);
                execution_receipts.extend(output.execution_receipts);
            }
        }

        if slots.len() != request.slots.len() {
            return Err(text_lane_error(
                "Bounded text lanes changed OCR batch cardinality.",
            ));
        }
        let output = OcrProviderBatchOutput {
            slots,
            execution_receipts,
        };
        self.validate_batch_model(&output)?;
        Ok(output)
    }

    async fn run_lane(
        &self,
        lane_index: usize,
        work: Option<&Range<usize>>,
        helpers: &[PpOcrV6Provider],
        request: &OcrProviderBatchRequest,
        decoded: &[SharedDecodedImage],
        cancellation: &CancellationToken,
    ) -> UseResult<Option<OcrProviderBatchOutput>> {
        let Some(provider) = self.lanes.get(lane_index) else {
            return Ok(None);
        };
        let Some(work) = work else {
            return Ok(None);
        };
        let range = work.clone();
        let mut lane_slots = request.slots[range.clone()].to_vec();
        // Text extraction is page-local. Adjacency authorizes the separate
        // boundary-evidence stage; it does not require the preceding page to
        // be detected and recognized again merely to form a standalone text
        // lane request.
        if let Some(first) = lane_slots.first_mut() {
            first.adjacent_predecessor_slot_id = None;
        }
        let lane_request = OcrProviderBatchRequest {
            stages: request.stages.clone(),
            slots: lane_slots,
        };
        provider
            .recognize_batch_decoded_with_helpers(
                helpers,
                lane_request,
                decoded[range].to_vec(),
                cancellation,
            )
            .await
            .map(Some)
    }

    fn validate_batch_model(&self, output: &OcrProviderBatchOutput) -> UseResult<()> {
        for slot in &output.slots {
            if let Some(output) = &slot.output {
                self.validate_output_model(output)?;
            }
        }
        let expected_bundle = format!("{}-bundle", self.profile.execution_family());
        if output
            .execution_receipts
            .iter()
            .any(|receipt| receipt.model.family != expected_bundle)
        {
            return Err(text_lane_error(
                "A PP-OCRv6 text lane returned a receipt for a different bound model profile.",
            ));
        }
        Ok(())
    }

    fn validate_output_model(&self, output: &OcrProviderOutput) -> UseResult<()> {
        if output.model.as_deref() != Some(self.profile.family()) {
            return Err(text_lane_error(
                "A PP-OCRv6 text lane returned output for a different bound model profile.",
            ));
        }
        Ok(())
    }
}

fn recommended_lane_count(provider: &PpOcrV6Provider) -> UseResult<usize> {
    let kind = provider.runtime_device_kind();
    let memory = provider.runtime_memory_snapshot()?;
    let available_device_bytes = memory.device.as_ref().map(|device| device.available_bytes);
    Ok(lane_count_for(kind, available_device_bytes))
}

fn lane_count_for(kind: RuntimeDeviceKind, available_device_bytes: Option<u64>) -> usize {
    if kind != RuntimeDeviceKind::Cuda {
        return 1;
    }
    let available = available_device_bytes.unwrap_or_default();
    let funded = available
        .saturating_sub(CUDA_DEVICE_RESERVE_BYTES)
        .checked_div(CUDA_TEXT_LANE_BUDGET_BYTES)
        .unwrap_or_default();
    usize::try_from(funded)
        .unwrap_or(MAX_CUDA_TEXT_LANES)
        .clamp(1, MAX_CUDA_TEXT_LANES)
}

fn slots_per_lane(kind: RuntimeDeviceKind) -> usize {
    #[cfg(test)]
    if let Some(slots) = std::env::var_os("A3S_OCR_TEST_TEXT_SLOTS_PER_LANE")
        .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
    {
        return slots.clamp(1, MAX_BATCH_SLOTS);
    }
    match kind {
        RuntimeDeviceKind::Cpu | RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal => {
            MAX_BATCH_SLOTS
        }
    }
}

fn lane_ranges(slot_count: usize, slots_per_lane: usize) -> Vec<Range<usize>> {
    let mut work = Vec::new();
    let mut start = 0_usize;
    while start < slot_count {
        let end = start.saturating_add(slots_per_lane).min(slot_count);
        work.push(start..end);
        start = end;
    }
    work
}

fn text_lane_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_batch_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_lane_count_retains_a_fixed_device_reserve() {
        assert_eq!(lane_count_for(RuntimeDeviceKind::Cpu, None), 1);
        assert_eq!(lane_count_for(RuntimeDeviceKind::Cuda, Some(6 << 30)), 1);
        assert_eq!(lane_count_for(RuntimeDeviceKind::Cuda, Some(10 << 30)), 2);
        assert_eq!(lane_count_for(RuntimeDeviceKind::Cuda, Some(14 << 30)), 3);
        assert_eq!(lane_count_for(RuntimeDeviceKind::Cuda, Some(24 << 30)), 3);
    }

    #[test]
    fn cpu_lane_span_uses_the_bounded_batch_contract() {
        assert_eq!(slots_per_lane(RuntimeDeviceKind::Cpu), MAX_BATCH_SLOTS);
        assert_eq!(slots_per_lane(RuntimeDeviceKind::Cuda), MAX_BATCH_SLOTS);
        assert_eq!(slots_per_lane(RuntimeDeviceKind::Metal), MAX_BATCH_SLOTS);
    }

    #[test]
    fn lane_ranges_are_bounded_non_overlapping_and_canonical() {
        assert_eq!(lane_ranges(48, 16), vec![0..16, 16..32, 32..48]);
        assert_eq!(lane_ranges(46, 16), vec![0..16, 16..32, 32..46]);
        for slot_count in 0..=100 {
            let ranges = lane_ranges(slot_count, 16);
            assert!(ranges.iter().all(|range| range.len() <= 16));
            assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
            assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), slot_count);
        }
    }
}
