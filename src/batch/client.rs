use std::collections::{BTreeMap, BTreeSet};

use a3s_use_core::{Artifact, UseError, UseResult};
use tokio::task::JoinSet;

use super::types::{
    provider_batch_error, slot_status, OcrBatchRequest, OcrBatchResult, OcrBatchSlotId,
    OcrBatchSlotRequest, OcrBatchSlotResult, OcrModelFingerprint, OcrProviderBatchOutput,
    OcrProviderBatchRequest, OcrProviderBatchSlot, OcrProviderBatchSlotOutput,
    OcrProviderFingerprint, OcrStage, OcrStageOutcome, OcrStageStatus, MAX_BATCH_INPUT_BYTES,
};
use crate::client::read_source;
use crate::output_validation::{validate_execution_receipt, validate_provider_output};
use crate::{OcrClient, OcrStageEvidence};

const MAX_CONCURRENT_SOURCE_READS: usize = 8;

impl OcrClient {
    /// Executes a bounded staged batch while preserving exact caller slot order.
    ///
    /// Request-shape errors fail the call. Source, provider, and stage failures
    /// remain isolated to their exact slots.
    pub async fn extract_batch(&self, request: OcrBatchRequest) -> UseResult<OcrBatchResult> {
        let trace = std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some();
        let started = std::time::Instant::now();
        request.validate()?;
        if request.slots.iter().any(|slot| slot.text_window.is_some())
            && !self.descriptor.supports_text_windows
        {
            return Err(provider_batch_error(
                "The selected OCR provider does not declare Text-stage source-window support.",
            ));
        }
        let stages = request.canonical_stages();
        let provider_fingerprint = OcrProviderFingerprint::from_descriptor(&self.descriptor)?;
        let mut results = (0..request.slots.len())
            .map(|_| None)
            .collect::<Vec<Option<OcrBatchSlotResult>>>();
        let mut provider_slots = Vec::new();
        let mut valid = Vec::new();
        let mut admitted_bytes = 0_u64;

        read_sources_bounded(request.slots, |index, slot, source| {
            match source {
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
                        return;
                    }
                    admitted_bytes = next_bytes.unwrap_or(admitted_bytes);
                    valid.push((index, slot.slot_id.clone(), input.source().clone()));
                    provider_slots.push(OcrProviderBatchSlot {
                        slot_id: slot.slot_id,
                        input,
                        adjacent_predecessor_slot_id: slot.adjacent_predecessor_slot_id,
                        text_window: slot.text_window,
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
        })
        .await?;
        let admitted_at = started.elapsed();

        let mut execution_receipts = Vec::new();
        let mut provider_at = admitted_at;
        let mut validated_at = admitted_at;
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
                    provider_at = started.elapsed();
                    validate_provider_batch(&output, &expected_ids, &stages, &self.descriptor)?;
                    validated_at = started.elapsed();
                    execution_receipts = output.execution_receipts;
                    for ((index, slot_id, source), output) in valid.into_iter().zip(output.slots) {
                        results[index] = Some(self.finish_batch_slot(slot_id, source, output)?);
                    }
                }
                Err(error) => {
                    provider_at = started.elapsed();
                    validated_at = provider_at;
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
        if trace {
            let completed = started.elapsed();
            eprintln!(
                "A3S_OCR_BATCH_CLIENT_TIMING slots={} read_ms={:.3} provider_ms={:.3} validate_ms={:.3} finish_ms={:.3} total_ms={:.3}",
                slots.len(),
                admitted_at.as_secs_f64() * 1_000.0,
                (provider_at - admitted_at).as_secs_f64() * 1_000.0,
                (validated_at - provider_at).as_secs_f64() * 1_000.0,
                (completed - validated_at).as_secs_f64() * 1_000.0,
                completed.as_secs_f64() * 1_000.0,
            );
        }
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

type SourceRead = (usize, OcrBatchSlotRequest, UseResult<crate::OcrInput>);

async fn read_sources_bounded(
    slots: Vec<OcrBatchSlotRequest>,
    mut consume: impl FnMut(usize, OcrBatchSlotRequest, UseResult<crate::OcrInput>),
) -> UseResult<()> {
    let total = slots.len();
    let mut waiting = slots.into_iter().enumerate();
    let mut tasks = JoinSet::<SourceRead>::new();
    for _ in 0..MAX_CONCURRENT_SOURCE_READS {
        let Some((index, slot)) = waiting.next() else {
            break;
        };
        spawn_source_read(&mut tasks, index, slot);
    }

    let mut pending = BTreeMap::new();
    let mut next_index = 0_usize;
    while let Some(completed) = tasks.join_next().await {
        let (index, slot, source) = completed.map_err(|error| {
            UseError::new(
                "use.ocr.source_unreadable",
                format!("A bounded OCR source-read task did not complete: {error}"),
            )
        })?;
        if pending.insert(index, (slot, source)).is_some() {
            return Err(provider_batch_error(
                "Bounded OCR source reading produced a duplicate slot index.",
            ));
        }
        while let Some((slot, source)) = pending.remove(&next_index) {
            consume(next_index, slot, source);
            next_index += 1;
            if let Some((index, slot)) = waiting.next() {
                spawn_source_read(&mut tasks, index, slot);
            }
        }
    }
    if next_index != total || !pending.is_empty() {
        return Err(provider_batch_error(
            "Bounded OCR source reading changed slot cardinality.",
        ));
    }
    Ok(())
}

fn spawn_source_read(tasks: &mut JoinSet<SourceRead>, index: usize, slot: OcrBatchSlotRequest) {
    tasks.spawn(async move {
        let source = read_source(&slot.path).await;
        (index, slot, source)
    });
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
        validate_layout_text_block_references(slot)?;
        validate_table_text_block_references(slot)?;
    }
    Ok(())
}

pub(super) fn validate_layout_text_block_references(
    slot: &OcrProviderBatchSlotOutput,
) -> UseResult<()> {
    let mut claimed = BTreeSet::new();
    for layout in slot.stages.iter().filter_map(|outcome| {
        let OcrStageEvidence::Layout(layout) = outcome.evidence.as_ref()? else {
            return None;
        };
        Some(layout)
    }) {
        for region in &layout.regions {
            for index in &region.source_text_block_indices {
                let index_usize = usize::try_from(*index).map_err(|_| {
                    provider_batch_error(
                        "A layout region referenced a text block outside the platform index range.",
                    )
                })?;
                if !claimed.insert(*index) {
                    return Err(provider_batch_error(
                        "A source text block must belong to at most one layout region.",
                    ));
                }
                let bounds = slot
                    .output
                    .as_ref()
                    .and_then(|output| output.blocks.get(index_usize))
                    .and_then(|block| block.bounding_box)
                    .ok_or_else(|| {
                        provider_batch_error(
                            "A layout-region source text block requires exact source-image bounds.",
                        )
                    })?;
                let right = bounds.x.checked_add(bounds.width).ok_or_else(|| {
                    provider_batch_error("A layout source text-block width overflowed.")
                })?;
                let bottom = bounds.y.checked_add(bounds.height).ok_or_else(|| {
                    provider_batch_error("A layout source text-block height overflowed.")
                })?;
                if right > layout.canvas.width || bottom > layout.canvas.height {
                    return Err(provider_batch_error(
                        "A layout source text block must remain inside the exact stage canvas.",
                    ));
                }
                let center_x_twice = u64::from(bounds.x) * 2 + u64::from(bounds.width);
                let center_y_twice = u64::from(bounds.y) * 2 + u64::from(bounds.height);
                let region_bounds = region.region.bounding_box;
                let region_right_twice = u64::from(
                    region_bounds
                        .x
                        .checked_add(region_bounds.width)
                        .ok_or_else(|| provider_batch_error("A layout-region width overflowed."))?,
                ) * 2;
                let region_bottom_twice = u64::from(
                    region_bounds
                        .y
                        .checked_add(region_bounds.height)
                        .ok_or_else(|| {
                            provider_batch_error("A layout-region height overflowed.")
                        })?,
                ) * 2;
                if center_x_twice < u64::from(region_bounds.x) * 2
                    || center_x_twice > region_right_twice
                    || center_y_twice < u64::from(region_bounds.y) * 2
                    || center_y_twice > region_bottom_twice
                {
                    return Err(provider_batch_error(
                        "A layout region may reference only text blocks whose source-box center it contains.",
                    ));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn validate_table_text_block_references(
    slot: &OcrProviderBatchSlotOutput,
) -> UseResult<()> {
    let mut claimed = BTreeSet::new();
    for table in slot.stages.iter().filter_map(|outcome| {
        let OcrStageEvidence::Table(table) = outcome.evidence.as_ref()? else {
            return None;
        };
        Some(table)
    }) {
        for cell in table.tables.iter().flat_map(|table| &table.cells) {
            let mut source_texts = Vec::with_capacity(cell.source_text_block_indices.len());
            for index in &cell.source_text_block_indices {
                let index_usize = usize::try_from(*index).map_err(|_| {
                    provider_batch_error(
                        "A table cell referenced a text block outside the platform index range.",
                    )
                })?;
                if !claimed.insert(*index) {
                    return Err(provider_batch_error(
                        "Table-cell text-block references must be in range and owned by one cell.",
                    ));
                }
                let source_text = slot
                    .output
                    .as_ref()
                    .and_then(|output| output.blocks.get(index_usize))
                    .ok_or_else(|| {
                        provider_batch_error(
                            "Table-cell text-block references must be in range and owned by one cell.",
                        )
                    })?;
                if source_text.polygon.is_none()
                    && source_text.bounding_box.is_none()
                    && source_text.bounding_boxes.is_empty()
                {
                    return Err(provider_batch_error(
                        "A table-cell source text block requires exact source-image geometry.",
                    ));
                }
                source_texts.push(source_text.text.as_str());
            }
            if !cell.source_text_block_indices.is_empty() {
                // This is an exact contract check over provider-declared
                // identities. It does not search or rank by document text.
                let expected = source_texts.join(" ");
                if cell.text.as_deref() != Some(expected.as_str()) {
                    return Err(provider_batch_error(
                        "A table cell's text must equal its ordered source text blocks.",
                    ));
                }
            }
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
