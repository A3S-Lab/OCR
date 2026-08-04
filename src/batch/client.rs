use a3s_use_core::{Artifact, UseError, UseResult};

use super::types::{
    provider_batch_error, slot_status, OcrBatchRequest, OcrBatchResult, OcrBatchSlotId,
    OcrBatchSlotResult, OcrModelFingerprint, OcrProviderBatchOutput, OcrProviderBatchRequest,
    OcrProviderBatchSlot, OcrProviderBatchSlotOutput, OcrProviderFingerprint, OcrStage,
    OcrStageOutcome, OcrStageStatus, MAX_BATCH_INPUT_BYTES,
};
use crate::client::read_source;
use crate::output_validation::{validate_execution_receipt, validate_provider_output};
use crate::OcrClient;

impl OcrClient {
    /// Executes a bounded staged batch while preserving exact caller slot order.
    ///
    /// Request-shape errors fail the call. Source, provider, and stage failures
    /// remain isolated to their exact slots.
    pub async fn extract_batch(&self, request: OcrBatchRequest) -> UseResult<OcrBatchResult> {
        request.validate()?;
        let stages = request.canonical_stages();
        let provider_fingerprint = OcrProviderFingerprint::from_descriptor(&self.descriptor)?;
        let mut results = (0..request.slots.len())
            .map(|_| None)
            .collect::<Vec<Option<OcrBatchSlotResult>>>();
        let mut provider_slots = Vec::new();
        let mut valid = Vec::new();
        let mut admitted_bytes = 0_u64;

        for (index, slot) in request.slots.into_iter().enumerate() {
            match read_source(&slot.path).await {
                Ok(input) => {
                    let next_bytes = admitted_bytes.checked_add(input.source().size);
                    if next_bytes.is_none_or(|bytes| bytes > MAX_BATCH_INPUT_BYTES) {
                        let error = UseError::new(
                            "use.ocr.batch_too_large",
                            format!(
                                "Validated OCR batch inputs must not exceed {MAX_BATCH_INPUT_BYTES} bytes."
                            ),
                        )
                        .with_detail("maximumBytes", MAX_BATCH_INPUT_BYTES);
                        results[index] = Some(failed_slot(
                            slot.slot_id,
                            Some(input.source().clone()),
                            &stages,
                            &self.descriptor,
                            error,
                        ));
                        continue;
                    }
                    admitted_bytes = next_bytes.unwrap_or(admitted_bytes);
                    valid.push((index, slot.slot_id.clone(), input.source().clone()));
                    provider_slots.push(OcrProviderBatchSlot {
                        slot_id: slot.slot_id,
                        input,
                    });
                }
                Err(error) => {
                    results[index] = Some(failed_slot(
                        slot.slot_id,
                        None,
                        &stages,
                        &self.descriptor,
                        error,
                    ));
                }
            }
        }

        let mut execution_receipts = Vec::new();
        if !provider_slots.is_empty() {
            let expected_ids = provider_slots
                .iter()
                .map(|slot| slot.slot_id.clone())
                .collect::<Vec<_>>();
            let provider_request = OcrProviderBatchRequest {
                stages: stages.clone(),
                slots: provider_slots,
            };
            match self.provider.recognize_batch(provider_request).await {
                Ok(output) => {
                    validate_provider_batch(&output, &expected_ids, &stages, &self.descriptor)?;
                    execution_receipts = output.execution_receipts;
                    for ((index, slot_id, source), output) in valid.into_iter().zip(output.slots) {
                        results[index] = Some(self.finish_batch_slot(slot_id, source, output)?);
                    }
                }
                Err(error) => {
                    for (index, slot_id, source) in valid {
                        results[index] = Some(failed_slot(
                            slot_id,
                            Some(source),
                            &stages,
                            &self.descriptor,
                            error.clone(),
                        ));
                    }
                }
            }
        }

        let slots = results
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                provider_batch_error("OCR batch result assembly left an unresolved slot.")
            })?;
        Ok(OcrBatchResult {
            schema: OcrBatchResult::SCHEMA.to_string(),
            provider: provider_fingerprint,
            requested_stages: stages,
            slots,
            execution_receipts,
        })
    }

    fn finish_batch_slot(
        &self,
        slot_id: OcrBatchSlotId,
        source: Artifact,
        output: OcrProviderBatchSlotOutput,
    ) -> UseResult<OcrBatchSlotResult> {
        let model_fingerprint = output
            .output
            .as_ref()
            .map(OcrModelFingerprint::from_output)
            .transpose()?
            .flatten();
        let result = output
            .output
            .map(|output| self.finish_output(source.clone(), output))
            .transpose()?;
        Ok(OcrBatchSlotResult {
            slot_id,
            status: slot_status(&output.stages),
            source: Some(source),
            stages: output.stages,
            model_fingerprint,
            result,
        })
    }
}

fn validate_provider_batch(
    output: &OcrProviderBatchOutput,
    expected_ids: &[OcrBatchSlotId],
    stages: &[OcrStage],
    descriptor: &crate::OcrProviderDescriptor,
) -> UseResult<()> {
    if output.slots.len() != expected_ids.len() {
        return Err(provider_batch_error(
            "An OCR provider batch must return exactly one slot for every validated input.",
        ));
    }
    for receipt in &output.execution_receipts {
        validate_execution_receipt(receipt)?;
    }
    for ((slot, expected_id), expected_stages) in output
        .slots
        .iter()
        .zip(expected_ids)
        .zip(std::iter::repeat(stages))
    {
        if &slot.slot_id != expected_id {
            return Err(provider_batch_error(
                "An OCR provider batch must preserve exact slot identity and input order.",
            ));
        }
        if slot.stages.len() != expected_stages.len()
            || slot
                .stages
                .iter()
                .map(|outcome| outcome.stage)
                .ne(expected_stages.iter().copied())
        {
            return Err(provider_batch_error(
                "An OCR provider batch must return every requested stage once in canonical order.",
            ));
        }
        for outcome in &slot.stages {
            outcome.validate()?;
            if outcome.status == OcrStageStatus::Completed
                && !descriptor.supports_stage(outcome.stage)
            {
                return Err(provider_batch_error(
                    "An OCR provider completed a stage absent from its descriptor.",
                ));
            }
        }
        if slot.stages.iter().any(|outcome| {
            outcome.stage == OcrStage::Text && outcome.status == OcrStageStatus::Completed
        }) && slot.output.is_none()
        {
            return Err(provider_batch_error(
                "A completed OCR text stage requires provider output.",
            ));
        }
        if let Some(provider_output) = &slot.output {
            validate_provider_output(provider_output)?;
        }
    }
    Ok(())
}

fn failed_slot(
    slot_id: OcrBatchSlotId,
    source: Option<Artifact>,
    stages: &[OcrStage],
    descriptor: &crate::OcrProviderDescriptor,
    error: UseError,
) -> OcrBatchSlotResult {
    let mut failed = false;
    let outcomes = stages
        .iter()
        .map(|stage| {
            if descriptor.supports_stage(*stage) && !failed {
                failed = true;
                OcrStageOutcome::failed(*stage, error.clone())
            } else if descriptor.supports_stage(*stage) {
                OcrStageOutcome::skipped(*stage, error.clone())
            } else {
                OcrStageOutcome::unsupported(*stage)
            }
        })
        .collect::<Vec<_>>();
    OcrBatchSlotResult {
        slot_id,
        status: slot_status(&outcomes),
        source,
        stages: outcomes,
        model_fingerprint: None,
        result: None,
    }
}
