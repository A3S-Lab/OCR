use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;

#[path = "provider/model.rs"]
mod model;

use super::assets::SlanetPlusAssets;
use super::initialization::DocumentFastInitializationError;
use super::layout::{
    project_evidence as layout_evidence, DetectedLayoutPage, DocumentLayoutBatch,
    DocumentLayoutRunner,
};
use super::page_orientation::{PageOrientationAssets, PageOrientationRunner};
use super::projection::table_evidence;
use super::seal::{seal_evidence, DetectedSealPage, SealStageBatch, SealStageRunner};
use super::shared_decode::decode_slots_once;
use super::stage::{DetectedPage, TableStageBatch, TableStageRunner};
use super::text_lanes::TextLanes;
use crate::cancellation::CancellationScope;
use crate::{
    OcrInput, OcrProvider, OcrProviderBatchOutput, OcrProviderBatchRequest,
    OcrProviderBatchSlotOutput, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
    OcrStage, OcrStageOutcome,
};
use model::document_fast_model_name;

pub const DOCUMENT_FAST_PROVIDER_ID: &str = "document-fast-v1";
const ENGINE_NAME: &str = "a3s-power-native";

/// Local fast-document composition with PP-OCRv6 text and model-backed wired
/// table structure. Cross-page reconciliation remains a Parser concern.
#[derive(Clone)]
pub struct DocumentFastOcrProvider {
    descriptor: OcrProviderDescriptor,
    model_name: String,
    orientation: Option<PageOrientationRunner>,
    layout: Option<DocumentLayoutRunner>,
    text: TextLanes,
    table: TableStageRunner,
    seal: Option<SealStageRunner>,
}

impl DocumentFastOcrProvider {
    pub fn from_env() -> UseResult<Self> {
        Self::from_env_typed().map_err(DocumentFastInitializationError::into_use_error)
    }

    pub fn from_env_typed() -> Result<Self, DocumentFastInitializationError> {
        let table_assets = SlanetPlusAssets::from_env_typed()?;
        let orientation = PageOrientationAssets::from_env_optional()
            .map_err(DocumentFastInitializationError::configuration_invalid)?
            .map(PageOrientationRunner::new)
            .transpose()
            .map_err(DocumentFastInitializationError::configuration_invalid)?;
        let layout = DocumentLayoutRunner::from_env_optional()
            .map_err(DocumentFastInitializationError::configuration_invalid)?;
        let seal = SealStageRunner::from_env_optional()
            .map_err(DocumentFastInitializationError::configuration_invalid)?;
        let mut stages = vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Table];
        if orientation.is_some() {
            stages.insert(0, OcrStage::Orientation);
        }
        if layout.is_some() {
            let text_index = stages
                .iter()
                .position(|stage| *stage == OcrStage::Text)
                .unwrap_or(stages.len());
            stages.insert(text_index, OcrStage::Layout);
        }
        if seal.is_some() {
            stages.push(OcrStage::Seal);
        }
        let text = TextLanes::from_env()
            .map_err(DocumentFastInitializationError::configuration_invalid)?;
        let model_name = document_fast_model_name(
            text.model_profile(),
            orientation.is_some(),
            layout.is_some(),
            seal.as_ref()
                .map(|runner| (runner.layout_model_family(), runner.has_text_fusion())),
        );
        let descriptor = OcrProviderDescriptor::new(DOCUMENT_FAST_PROVIDER_ID, ENGINE_NAME, false)
            .and_then(|descriptor| descriptor.with_stages(stages))
            .and_then(|descriptor| descriptor.with_text_windows(true))
            .map_err(DocumentFastInitializationError::configuration_invalid)?;
        let table = TableStageRunner::new(table_assets)
            .map_err(DocumentFastInitializationError::configuration_invalid)?;
        Ok(Self {
            descriptor,
            model_name,
            orientation,
            layout,
            text,
            table,
            seal,
        })
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[async_trait]
impl OcrProvider for DocumentFastOcrProvider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        self.descriptor.clone()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        let text = self.text.diagnostic();
        let model = self.model_name();
        if text.readiness == Readiness::Ready {
            OcrProviderStatus {
                readiness: Readiness::Ready,
                model: Some(model.to_string()),
                model_dir: Some(
                    self.layout
                        .as_ref()
                        .map(DocumentLayoutRunner::model_root)
                        .or_else(|| self.seal.as_ref().map(SealStageRunner::model_root))
                        .unwrap_or_else(|| self.table.model_root())
                        .to_path_buf(),
                ),
                message: if self.layout.is_some()
                    && self
                        .seal
                        .as_ref()
                        .is_some_and(SealStageRunner::has_text_fusion)
                {
                    "Local PP-OCRv6 text, PP-DocLayout-S document layout, deterministic wired-grid, SLANet-Plus fallback, PicoDet seal layout, and orthogonally verified seal-text fusion stages are ready."
                        .to_string()
                } else if self.layout.is_some() && self.seal.is_some() {
                    "Local PP-OCRv6 text, PP-DocLayout-S document layout, deterministic wired-grid, SLANet-Plus fallback, and PicoDet seal stages are ready; the optional seal-text verifier is not configured."
                        .to_string()
                } else if self.layout.is_some() {
                    "Local PP-OCRv6 text, PP-DocLayout-S document layout, deterministic wired-grid, and SLANet-Plus fallback stages are ready; the optional PicoDet seal model is not configured."
                        .to_string()
                } else if self.seal.is_some() {
                    "Local PP-OCRv6 text, deterministic wired-grid, SLANet-Plus fallback, and PicoDet seal stages are ready; the optional seal-text verifier is not configured."
                        .to_string()
                } else {
                    "Local PP-OCRv6 text, deterministic wired-grid, and SLANet-Plus fallback stages are ready; the optional PicoDet seal model is not configured."
                        .to_string()
                },
                suggestions: Vec::new(),
            }
        } else {
            OcrProviderStatus {
                readiness: text.readiness,
                model: Some(model.to_string()),
                model_dir: text.model_dir,
                message: format!(
                    "The wired-table model is ready, but the PP-OCRv6 text model is not: {}",
                    text.message
                ),
                suggestions: text.suggestions,
            }
        }
    }

    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
        self.text.recognize(input).await
    }

    async fn recognize_batch(
        &self,
        request: OcrProviderBatchRequest,
    ) -> UseResult<OcrProviderBatchOutput> {
        let trace = std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some();
        let started = std::time::Instant::now();
        let stages = request.stages;
        let mut stage_slots = request.slots;
        let cancellation = CancellationScope::new();
        let token = cancellation.token();
        let source_decoded = decode_slots_once(&stage_slots, token.clone()).await?;
        let decoded_at = started.elapsed();
        let slot_ids = stage_slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>();
        let orientation_requested = stages.contains(&OcrStage::Orientation);
        let (text_decoded, orientation_slots, orientation_receipts) =
            match (orientation_requested, self.orientation.as_ref()) {
                (true, Some(orientation)) => {
                    let batch = orientation
                        .normalize_decoded(source_decoded.clone(), token.clone())
                        .await?;
                    if batch.slots.len() != stage_slots.len()
                        || batch.images.len() != stage_slots.len()
                    {
                        return Err(composition_error(
                            "The orientation stage changed document-fast slot cardinality.",
                        ));
                    }
                    for (slot, orientation) in stage_slots.iter_mut().zip(&batch.slots) {
                        if let (Some(transform), Some(window)) =
                            (orientation.transform, slot.text_window)
                        {
                            slot.text_window = Some(transform.orient_window(window)?);
                        }
                    }
                    (
                        batch.images,
                        batch
                            .slots
                            .into_iter()
                            .map(|slot| (Some(slot.outcome), slot.transform, slot.receipts))
                            .collect::<Vec<_>>(),
                        batch.receipts,
                    )
                }
                (true, None) => (
                    source_decoded.clone(),
                    (0..stage_slots.len())
                        .map(|_| {
                            (
                                Some(OcrStageOutcome::unsupported(OcrStage::Orientation)),
                                None,
                                Vec::new(),
                            )
                        })
                        .collect(),
                    Vec::new(),
                ),
                (false, _) => (
                    source_decoded.clone(),
                    (0..stage_slots.len())
                        .map(|_| (None, None, Vec::new()))
                        .collect(),
                    Vec::new(),
                ),
            };
        let oriented_at = started.elapsed();
        let text_stages = stages
            .iter()
            .copied()
            .filter(|stage| matches!(stage, OcrStage::Preprocessing | OcrStage::Text))
            .collect::<Vec<_>>();
        let text_request = (!text_stages.is_empty()).then(|| OcrProviderBatchRequest {
            stages: text_stages,
            slots: stage_slots.clone(),
        });
        let table_slots = stages
            .contains(&OcrStage::Table)
            .then(|| stage_slots.clone());
        let layout_slots = stages
            .contains(&OcrStage::Layout)
            .then(|| stage_slots.clone());
        let seal_slots = stages.contains(&OcrStage::Seal).then_some(stage_slots);
        let seal_source_decoded = source_decoded.clone();
        let seal_oriented_decoded = text_decoded.clone();
        let seal_transforms = orientation_slots
            .iter()
            .map(|(_, transform, _)| *transform)
            .collect::<Vec<_>>();
        let layout_transforms = seal_transforms.clone();
        let text_token = token.clone();
        let layout_token = token.clone();
        let table_token = token.clone();
        let seal_token = token;
        let text_future = async {
            match text_request {
                Some(request) => Some(
                    self.text
                        .recognize_batch_decoded(request, text_decoded.clone(), &text_token)
                        .await,
                ),
                None => None,
            }
        };
        let table_future = async {
            match table_slots {
                Some(slots) => Some(
                    self.table
                        .run_decoded(slots, source_decoded.clone(), table_token)
                        .await,
                ),
                None => None,
            }
        };
        let layout_future = async {
            match (layout_slots, self.layout.as_ref()) {
                (Some(slots), Some(layout)) => Some(
                    layout
                        .run_decoded(slots, text_decoded.clone(), layout_transforms, layout_token)
                        .await,
                ),
                _ => None,
            }
        };
        let seal_future = async {
            match (seal_slots, self.seal.as_ref()) {
                (Some(slots), Some(seal)) => Some(
                    seal.run_source_with_oriented_layout(
                        slots,
                        seal_source_decoded,
                        seal_oriented_decoded,
                        seal_transforms,
                        seal_token,
                    )
                    .await,
                ),
                _ => None,
            }
        };
        let (text_result, layout_result, table_result, seal_result) =
            tokio::join!(text_future, layout_future, table_future, seal_future);
        let joined_at = started.elapsed();
        cancellation.disarm();

        let (text_slots, mut execution_receipts) = normalize_text_slots(text_result, &slot_ids)?;
        let (layout_slots, layout_receipts) = normalize_layout_slots(layout_result, &slot_ids)?;
        let (table_slots, table_receipts) = normalize_table_slots(table_result, &slot_ids)?;
        let (seal_slots, seal_receipts) = normalize_seal_slots(seal_result, &slot_ids)?;
        execution_receipts.extend(orientation_receipts);
        execution_receipts.extend(layout_receipts);
        execution_receipts.extend(table_receipts);
        execution_receipts.extend(seal_receipts);
        let model_name = self.model_name();
        let mut outputs = Vec::with_capacity(slot_ids.len());
        for (((((slot_id, text_slot), layout_slot), table_slot), seal_slot), orientation_slot) in
            slot_ids
                .into_iter()
                .zip(text_slots)
                .zip(layout_slots)
                .zip(table_slots)
                .zip(seal_slots)
                .zip(orientation_slots)
        {
            let (mut orientation_outcome, transform, orientation_slot_receipts) = orientation_slot;
            let (mut output, mut text_outcomes) = text_slot
                .map(|slot| (slot.output, slot.stages))
                .unwrap_or_default();
            if let (Some(transform), Some(output)) = (transform, output.as_mut()) {
                transform.restore_output(output)?;
            }
            let mut layout_resolution = match layout_slot {
                None => None,
                Some(Err(error)) => Some(Err(error)),
                Some(Ok(page)) => Some(layout_evidence(page, output.as_ref())),
            };
            let mut table_resolution = match table_slot {
                None => None,
                Some(Err(error)) => Some(Err(error)),
                Some(Ok(page)) => Some(table_evidence(page, output.as_ref()).map(
                    |(evidence, receipts)| {
                        let output = output.get_or_insert_with(OcrProviderOutput::default);
                        output.model = Some(model_name.to_string());
                        output.execution_receipts.extend(receipts);
                        evidence
                    },
                )),
            };
            let mut seal_resolution = match seal_slot {
                None => None,
                Some(Err(error)) => Some(Err(error)),
                Some(Ok(page)) => Some(seal_evidence(page).map(|(evidence, receipts)| {
                    let output = output.get_or_insert_with(OcrProviderOutput::default);
                    output.model = Some(model_name.to_string());
                    output.execution_receipts.extend(receipts);
                    evidence
                })),
            };
            let outcomes = stages
                .iter()
                .map(|stage| match stage {
                    OcrStage::Orientation => orientation_outcome.take().unwrap_or_else(|| {
                        OcrStageOutcome::failed(
                            *stage,
                            composition_error(
                                "The orientation sub-provider omitted a requested stage outcome.",
                            ),
                        )
                    }),
                    OcrStage::Preprocessing | OcrStage::Text => {
                        take_stage_outcome(&mut text_outcomes, *stage).unwrap_or_else(|| {
                            OcrStageOutcome::failed(
                                *stage,
                                composition_error(
                                    "The PP-OCRv6 sub-provider omitted a requested stage outcome.",
                                ),
                            )
                        })
                    }
                    OcrStage::Layout => match layout_resolution.take() {
                        Some(Ok(evidence)) => OcrStageOutcome::completed_with_evidence(evidence),
                        Some(Err(error)) => OcrStageOutcome::failed(*stage, error),
                        None => OcrStageOutcome::unsupported(*stage),
                    },
                    OcrStage::Table => match table_resolution.take() {
                        Some(Ok(evidence)) => OcrStageOutcome::completed_with_evidence(evidence),
                        Some(Err(error)) => OcrStageOutcome::failed(*stage, error),
                        None => OcrStageOutcome::failed(
                            *stage,
                            composition_error(
                                "The table sub-provider omitted a requested stage outcome.",
                            ),
                        ),
                    },
                    OcrStage::Seal => match seal_resolution.take() {
                        Some(Ok(evidence)) => OcrStageOutcome::completed_with_evidence(evidence),
                        Some(Err(error)) => OcrStageOutcome::failed(*stage, error),
                        None => OcrStageOutcome::unsupported(*stage),
                    },
                    OcrStage::Formula => OcrStageOutcome::unsupported(*stage),
                })
                .collect::<Vec<_>>();
            if !orientation_slot_receipts.is_empty() {
                let output = output.get_or_insert_with(OcrProviderOutput::default);
                output.execution_receipts.extend(orientation_slot_receipts);
            }
            if let Some(output) = output.as_mut() {
                output.model = Some(model_name.to_string());
            }
            outputs.push(OcrProviderBatchSlotOutput {
                slot_id,
                stages: outcomes,
                output,
            });
        }
        if trace {
            let completed = started.elapsed();
            eprintln!(
                "A3S_OCR_DOCUMENT_FAST_TIMING slots={} decode_ms={:.3} orientation_ms={:.3} parallel_stages_ms={:.3} compose_ms={:.3} total_ms={:.3}",
                outputs.len(),
                decoded_at.as_secs_f64() * 1_000.0,
                (oriented_at - decoded_at).as_secs_f64() * 1_000.0,
                (joined_at - oriented_at).as_secs_f64() * 1_000.0,
                (completed - joined_at).as_secs_f64() * 1_000.0,
                completed.as_secs_f64() * 1_000.0,
            );
        }
        Ok(OcrProviderBatchOutput {
            slots: outputs,
            execution_receipts,
        })
    }
}

fn take_stage_outcome(
    outcomes: &mut Vec<OcrStageOutcome>,
    stage: OcrStage,
) -> Option<OcrStageOutcome> {
    outcomes
        .iter()
        .position(|outcome| outcome.stage == stage)
        .map(|index| outcomes.remove(index))
}

fn normalize_text_slots(
    result: Option<UseResult<OcrProviderBatchOutput>>,
    expected: &[crate::OcrBatchSlotId],
) -> UseResult<(
    Vec<Option<OcrProviderBatchSlotOutput>>,
    Vec<crate::OcrExecutionReceipt>,
)> {
    let Some(result) = result else {
        return Ok(((0..expected.len()).map(|_| None).collect(), Vec::new()));
    };
    let output = result?;
    if output.slots.len() != expected.len()
        || output
            .slots
            .iter()
            .zip(expected)
            .any(|(slot, expected)| slot.slot_id != *expected)
    {
        return Err(composition_error(
            "The PP-OCRv6 sub-provider changed document-fast slot identity or cardinality.",
        ));
    }
    Ok((
        output.slots.into_iter().map(Some).collect(),
        output.execution_receipts,
    ))
}

type LayoutSlot = Result<DetectedLayoutPage, UseError>;

fn normalize_layout_slots(
    result: Option<UseResult<DocumentLayoutBatch>>,
    expected: &[crate::OcrBatchSlotId],
) -> UseResult<(Vec<Option<LayoutSlot>>, Vec<crate::OcrExecutionReceipt>)> {
    let Some(result) = result else {
        return Ok(((0..expected.len()).map(|_| None).collect(), Vec::new()));
    };
    match result {
        Ok(output) => {
            if output.slots.len() != expected.len()
                || output
                    .slots
                    .iter()
                    .zip(expected)
                    .any(|(slot, expected)| slot.slot_id != *expected)
            {
                return Err(composition_error(
                    "The document-layout sub-provider changed slot identity or cardinality.",
                ));
            }
            Ok((
                output
                    .slots
                    .into_iter()
                    .map(|slot| Some(slot.page))
                    .collect(),
                output.receipts,
            ))
        }
        Err(error) => Ok((
            expected.iter().map(|_| Some(Err(error.clone()))).collect(),
            Vec::new(),
        )),
    }
}

type TableSlot = Result<DetectedPage, UseError>;

fn normalize_table_slots(
    result: Option<UseResult<TableStageBatch>>,
    expected: &[crate::OcrBatchSlotId],
) -> UseResult<(Vec<Option<TableSlot>>, Vec<crate::OcrExecutionReceipt>)> {
    let Some(result) = result else {
        return Ok(((0..expected.len()).map(|_| None).collect(), Vec::new()));
    };
    match result {
        Ok(output) => {
            if output.slots.len() != expected.len()
                || output
                    .slots
                    .iter()
                    .zip(expected)
                    .any(|(slot, expected)| slot.slot_id != *expected)
            {
                return Err(composition_error(
                    "The table sub-provider changed document-fast slot identity or cardinality.",
                ));
            }
            Ok((
                output
                    .slots
                    .into_iter()
                    .map(|slot| Some(slot.page))
                    .collect(),
                output.receipts,
            ))
        }
        Err(error) => Ok((
            expected.iter().map(|_| Some(Err(error.clone()))).collect(),
            Vec::new(),
        )),
    }
}

type SealSlot = Result<DetectedSealPage, UseError>;

fn normalize_seal_slots(
    result: Option<UseResult<SealStageBatch>>,
    expected: &[crate::OcrBatchSlotId],
) -> UseResult<(Vec<Option<SealSlot>>, Vec<crate::OcrExecutionReceipt>)> {
    let Some(result) = result else {
        return Ok(((0..expected.len()).map(|_| None).collect(), Vec::new()));
    };
    match result {
        Ok(output) => {
            if output.slots.len() != expected.len()
                || output
                    .slots
                    .iter()
                    .zip(expected)
                    .any(|(slot, expected)| slot.slot_id != *expected)
            {
                return Err(composition_error(
                    "The seal sub-provider changed document-fast slot identity or cardinality.",
                ));
            }
            Ok((
                output
                    .slots
                    .into_iter()
                    .map(|slot| Some(slot.page))
                    .collect(),
                output.receipts,
            ))
        }
        Err(error) => Ok((
            expected.iter().map(|_| Some(Err(error.clone()))).collect(),
            Vec::new(),
        )),
    }
}

fn composition_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_batch_invalid", message)
}

#[cfg(test)]
#[path = "provider/tests.rs"]
mod tests;
