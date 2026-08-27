use std::collections::BTreeSet;
use std::sync::Arc;

use a3s_power::error::PowerError;
use a3s_power::inference::{ExecutionDigest, MicrobatchExecution, ModelSession, ModelSessionSpec};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use super::native::batch_binding;
use super::{build_output, PpOcrV6Engine, PpOcrV6Provider, PpOcrV6Session};
use crate::assets::ModelAssets;
use crate::cancellation::{check_cancelled, run_blocking_with, CancellationScope};
use crate::preprocess::decode_image;
use crate::receipt::project_receipt;
use crate::{
    OcrBatchSlotId, OcrInput, OcrNormalizedWindow, OcrProviderBatchOutput, OcrProviderBatchRequest,
    OcrProviderBatchSlot, OcrProviderBatchSlotOutput, OcrProviderOutput, OcrStage, OcrStageOutcome,
};

mod planning;

use planning::{microbatch_candidates, microbatch_policy};

pub(super) async fn recognize_one(
    provider: &PpOcrV6Provider,
    input: OcrInput,
) -> UseResult<OcrProviderOutput> {
    let slot_id = OcrBatchSlotId::new(format!("single-{}", input.source().sha256))?;
    let output = recognize_batch(
        provider,
        OcrProviderBatchRequest {
            stages: vec![OcrStage::Preprocessing, OcrStage::Text],
            slots: vec![OcrProviderBatchSlot {
                slot_id,
                input,
                adjacent_predecessor_slot_id: None,
                text_window: None,
            }],
        },
    )
    .await?;
    let slot = output.slots.into_iter().next().ok_or_else(|| {
        runtime_error("PP-OCRv6 returned no slot for a single recognition request.")
    })?;
    if let Some(output) = slot.output {
        return Ok(output);
    }
    Err(slot
        .stages
        .into_iter()
        .find(|stage| stage.stage == OcrStage::Text)
        .and_then(|stage| stage.error)
        .unwrap_or_else(|| runtime_error("PP-OCRv6 recognition produced no text output.")))
}

pub(super) async fn recognize_batch(
    provider: &PpOcrV6Provider,
    request: OcrProviderBatchRequest,
) -> UseResult<OcrProviderBatchOutput> {
    let cancellation = CancellationScope::new();
    let token = cancellation.token();
    let stages = request.stages;
    if !stages.contains(&OcrStage::Preprocessing) && !stages.contains(&OcrStage::Text) {
        let slots = request
            .slots
            .into_iter()
            .map(|slot| OcrProviderBatchSlotOutput {
                slot_id: slot.slot_id,
                stages: stages
                    .iter()
                    .map(|stage| OcrStageOutcome::unsupported(*stage))
                    .collect(),
                output: None,
            })
            .collect();
        cancellation.disarm();
        return Ok(OcrProviderBatchOutput {
            slots,
            execution_receipts: Vec::new(),
        });
    }
    let decoded = decode_slots(request.slots, token.clone()).await?;
    let output = recognize_decoded(provider, &[], stages, decoded, &token).await;
    cancellation.disarm();
    output
}

pub(super) async fn recognize_batch_decoded_with_helpers(
    provider: &PpOcrV6Provider,
    helpers: &[PpOcrV6Provider],
    request: OcrProviderBatchRequest,
    images: Vec<Result<Arc<RgbImage>, UseError>>,
    cancellation: &CancellationToken,
) -> UseResult<OcrProviderBatchOutput> {
    if request.slots.len() != images.len() {
        return Err(runtime_error(
            "PP-OCRv6 shared decoding changed staged batch cardinality.",
        ));
    }
    let decoded = request
        .slots
        .into_iter()
        .zip(images)
        .enumerate()
        .map(|(index, (slot, image))| DecodedSlot {
            index,
            slot_id: slot.slot_id,
            input: slot.input,
            text_window: slot.text_window,
            image,
        })
        .collect();
    recognize_decoded(provider, helpers, request.stages, decoded, cancellation).await
}

async fn recognize_decoded(
    provider: &PpOcrV6Provider,
    helpers: &[PpOcrV6Provider],
    stages: Vec<OcrStage>,
    decoded: Vec<DecodedSlot>,
    token: &CancellationToken,
) -> UseResult<OcrProviderBatchOutput> {
    let mut outputs = (0..decoded.len())
        .map(|_| None)
        .collect::<Vec<Option<OcrProviderBatchSlotOutput>>>();
    let mut prepared = Vec::new();
    let text_requested = stages.contains(&OcrStage::Text);

    for slot in decoded {
        match slot.image {
            Ok(image) if text_requested => prepared.push(PreparedSlot {
                original_index: slot.index,
                slot_id: slot.slot_id,
                input: slot.input,
                text_window: slot.text_window,
                image,
            }),
            Ok(_) => {
                outputs[slot.index] = Some(OcrProviderBatchSlotOutput {
                    slot_id: slot.slot_id,
                    stages: stage_outcomes(
                        &stages,
                        StageResolution::Completed,
                        StageResolution::Unsupported,
                    ),
                    output: None,
                });
            }
            Err(error) => {
                let text = if stages.contains(&OcrStage::Preprocessing) {
                    StageResolution::Skipped(error.clone())
                } else {
                    StageResolution::Failed(error.clone())
                };
                outputs[slot.index] = Some(OcrProviderBatchSlotOutput {
                    slot_id: slot.slot_id,
                    stages: stage_outcomes(&stages, StageResolution::Failed(error), text),
                    output: None,
                });
            }
        }
    }

    let mut receipts = Vec::new();
    if !prepared.is_empty() {
        match prepare_sessions(provider, helpers, token.clone()).await {
            Ok((session, helper_sessions, spec)) => {
                execute_microbatches(
                    MicrobatchContext {
                        session: &session,
                        helper_sessions: &helper_sessions,
                        spec: &spec,
                        stages: &stages,
                        cancellation: token,
                    },
                    prepared,
                    &mut outputs,
                    &mut receipts,
                )
                .await;
            }
            Err(error) => fail_prepared(&stages, prepared, error, &mut outputs),
        }
    }

    let slots = outputs
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| runtime_error("PP-OCRv6 left an unresolved staged batch slot."))?;
    Ok(OcrProviderBatchOutput {
        slots,
        execution_receipts: receipts,
    })
}

async fn decode_slots(
    slots: Vec<OcrProviderBatchSlot>,
    cancellation: CancellationToken,
) -> UseResult<Vec<DecodedSlot>> {
    run_blocking_with(
        "PP-OCRv6 staged image decoding",
        cancellation,
        move |cancellation| {
            slots
                .into_par_iter()
                .enumerate()
                .map(|(index, slot)| {
                    check_cancelled(&cancellation)?;
                    let image = decode_image(slot.input.bytes()).map(Arc::new);
                    Ok(DecodedSlot {
                        index,
                        slot_id: slot.slot_id,
                        input: slot.input,
                        text_window: slot.text_window,
                        image,
                    })
                })
                .collect()
        },
    )
    .await
}

async fn prepare_sessions(
    provider: &PpOcrV6Provider,
    helpers: &[PpOcrV6Provider],
    cancellation: CancellationToken,
) -> UseResult<(
    ModelSession<PpOcrV6Session>,
    Vec<ModelSession<PpOcrV6Session>>,
    ModelSessionSpec,
)> {
    let bound_provider = provider.clone();
    let (assets, spec) = run_blocking_with(
        "PP-OCRv6 session declaration",
        cancellation.clone(),
        move |cancellation| {
            check_cancelled(&cancellation)?;
            bound_provider.resolved_session_assets()
        },
    )
    .await?;
    let mut providers = Vec::with_capacity(helpers.len() + 1);
    providers.push(provider.clone());
    let mut seen = BTreeSet::from([provider.execution_replica]);
    providers.extend(
        helpers
            .iter()
            .filter(|helper| seen.insert(helper.execution_replica))
            .cloned(),
    );

    let mut loads = tokio::task::JoinSet::new();
    for (position, provider) in providers.into_iter().enumerate() {
        let assets = assets.clone();
        let spec = spec.clone();
        let cancellation = cancellation.clone();
        loads.spawn(async move {
            let result = provider
                .sessions
                .get_or_load_replica(
                    spec,
                    provider.execution_replica,
                    &cancellation,
                    move |runtime, loader_cancellation| {
                        load_session(assets, runtime, loader_cancellation)
                    },
                )
                .await
                .map_err(|error| power_error("load the exact PP-OCRv6 session", error));
            (position, result)
        });
    }

    let mut loaded = vec![None; helpers.len() + 1];
    while let Some(joined) = loads.join_next().await {
        let (position, result) = joined.map_err(|error| {
            runtime_error(format!("PP-OCRv6 session loader task failed: {error}"))
        })?;
        if position == 0 {
            loaded[0] = Some(result?);
        } else if let Ok(session) = result {
            loaded[position] = Some(session);
        }
    }
    check_cancelled(&cancellation)?;
    let session = loaded[0]
        .take()
        .ok_or_else(|| runtime_error("PP-OCRv6 primary session was not loaded."))?;
    let helper_sessions = loaded.into_iter().skip(1).flatten().collect();
    Ok((session, helper_sessions, spec))
}

async fn load_session(
    assets: ModelAssets,
    runtime: a3s_power::inference::EmbeddedRuntime,
    cancellation: CancellationToken,
) -> a3s_power::error::Result<PpOcrV6Session> {
    tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        PpOcrV6Engine::load_with_runtime(&assets, runtime)
            .map(|engine| PpOcrV6Session { engine })
            .map_err(|error| PowerError::InferenceFailed(error.message))
    })
    .await
    .map_err(|error| PowerError::InferenceFailed(format!("PP-OCRv6 loader task failed: {error}")))?
}

struct MicrobatchContext<'a> {
    session: &'a ModelSession<PpOcrV6Session>,
    helper_sessions: &'a [ModelSession<PpOcrV6Session>],
    spec: &'a ModelSessionSpec,
    stages: &'a [OcrStage],
    cancellation: &'a CancellationToken,
}

async fn execute_microbatches(
    context: MicrobatchContext<'_>,
    prepared: Vec<PreparedSlot>,
    outputs: &mut [Option<OcrProviderBatchSlotOutput>],
    receipts: &mut Vec<crate::OcrExecutionReceipt>,
) {
    let MicrobatchContext {
        session,
        helper_sessions,
        spec,
        stages,
        cancellation,
    } = context;
    let candidates = microbatch_candidates(session, &prepared);
    let candidates = match candidates {
        Ok(candidates) => candidates,
        Err(error) => {
            fail_prepared(stages, prepared, error, outputs);
            return;
        }
    };
    let policy = match microbatch_policy(session, spec.resident_bytes()) {
        Ok(policy) => policy,
        Err(error) => {
            fail_prepared(stages, prepared, error, outputs);
            return;
        }
    };
    let binding = match batch_binding(&session.binding().model.weights_sha256) {
        Ok(binding) => binding,
        Err(error) => {
            fail_prepared(stages, prepared, error, outputs);
            return;
        }
    };
    let plan = match session.plan_microbatches(binding, policy, candidates) {
        Ok(plan) => plan,
        Err(error) => {
            fail_prepared(
                stages,
                prepared,
                power_error("plan the PP-OCRv6 microbatches", error),
                outputs,
            );
            return;
        }
    };

    for batch in &plan.batches {
        let batch_slots = batch
            .slots
            .iter()
            .map(|slot| prepared[slot.source_index].clone())
            .collect::<Vec<_>>();
        let execution = match session
            .begin_microbatch(&plan, batch.index, cancellation)
            .await
        {
            Ok(execution) => execution,
            Err(error) => {
                fail_prepared(
                    stages,
                    batch_slots,
                    power_error("admit the PP-OCRv6 microbatch", error),
                    outputs,
                );
                continue;
            }
        };
        match run_engine_batch(
            session.clone(),
            helper_sessions.to_vec(),
            execution,
            batch_slots.clone(),
            cancellation.clone(),
        )
        .await
        {
            Ok(run) => {
                receipts.push(run.receipt);
                for (slot, result) in batch_slots.into_iter().zip(run.outputs) {
                    let (text, output) = match result {
                        Ok(output) => (StageResolution::Completed, Some(output)),
                        Err(error) => (StageResolution::Failed(error), None),
                    };
                    outputs[slot.original_index] = Some(OcrProviderBatchSlotOutput {
                        slot_id: slot.slot_id,
                        stages: stage_outcomes(stages, StageResolution::Completed, text),
                        output,
                    });
                }
            }
            Err(error) => fail_prepared(stages, batch_slots, error, outputs),
        }
    }
}

async fn run_engine_batch(
    session: ModelSession<PpOcrV6Session>,
    helper_sessions: Vec<ModelSession<PpOcrV6Session>>,
    execution: MicrobatchExecution,
    slots: Vec<PreparedSlot>,
    cancellation: CancellationToken,
) -> UseResult<BatchRun> {
    run_blocking_with(
        "PP-OCRv6 admitted microbatch",
        cancellation,
        move |cancellation| {
            let mut input_bytes = Vec::with_capacity(execution.batch().input_bytes);
            for slot in &slots {
                input_bytes.extend_from_slice(slot.input.bytes());
            }
            let images = slots
                .iter()
                .map(|slot| slot.image.as_ref())
                .collect::<Vec<_>>();
            let text_windows = slots
                .iter()
                .map(|slot| slot.text_window)
                .collect::<Vec<_>>();
            let helpers = helper_sessions
                .iter()
                .map(|helper| &helper.value().engine)
                .collect::<Vec<_>>();
            let outputs = session
                .value()
                .engine
                .extract_batch_admitted_with_windows_and_helpers(
                    &images,
                    &text_windows,
                    session.runtime().limits().max_tensor_elements,
                    execution.permit(),
                    &cancellation,
                    &helpers,
                )?
                .into_iter()
                .map(|result| result.and_then(build_output))
                .collect::<Vec<_>>();
            if outputs.len() != slots.len() {
                return Err(runtime_error(
                    "PP-OCRv6 changed admitted microbatch output cardinality.",
                ));
            }
            let mut output_declaration = Sha256::new();
            output_declaration.update(b"a3s-ocr-ppocr-v6-batch-output-v1\0");
            for (slot, result) in slots.iter().zip(&outputs) {
                update_text(&mut output_declaration, slot.slot_id.as_str())?;
                match result {
                    Ok(output) => {
                        output_declaration.update([1]);
                        for receipt in &output.execution_receipts {
                            update_text(&mut output_declaration, &receipt.output.sha256)?;
                        }
                    }
                    Err(error) => {
                        output_declaration.update([0]);
                        update_text(&mut output_declaration, &error.code)?;
                    }
                }
            }
            let input = ExecutionDigest::image_request(&input_bytes, slots.len());
            let output_marker = format!("{:x}", output_declaration.finalize());
            let output = ExecutionDigest::utf8_text(&output_marker);
            let receipt = execution
                .receipt(session.binding().model.clone(), input, output)
                .map(project_receipt)
                .map_err(|error| power_error("issue the PP-OCRv6 microbatch receipt", error))?;
            Ok(BatchRun { outputs, receipt })
        },
    )
    .await
}

fn fail_prepared(
    stages: &[OcrStage],
    slots: Vec<PreparedSlot>,
    error: UseError,
    outputs: &mut [Option<OcrProviderBatchSlotOutput>],
) {
    for slot in slots {
        outputs[slot.original_index] = Some(OcrProviderBatchSlotOutput {
            slot_id: slot.slot_id,
            stages: stage_outcomes(
                stages,
                StageResolution::Completed,
                StageResolution::Failed(error.clone()),
            ),
            output: None,
        });
    }
}

fn stage_outcomes(
    stages: &[OcrStage],
    preprocessing: StageResolution,
    text: StageResolution,
) -> Vec<OcrStageOutcome> {
    stages
        .iter()
        .map(|stage| match stage {
            OcrStage::Preprocessing => preprocessing.outcome(*stage),
            OcrStage::Text => text.outcome(*stage),
            OcrStage::Orientation
            | OcrStage::Layout
            | OcrStage::Table
            | OcrStage::Formula
            | OcrStage::Seal => OcrStageOutcome::unsupported(*stage),
        })
        .collect()
}

fn update_text(digest: &mut Sha256, value: &str) -> UseResult<()> {
    let length = u64::try_from(value.len())
        .map_err(|_| runtime_error("PP-OCRv6 batch evidence length cannot be represented."))?;
    digest.update(length.to_le_bytes());
    digest.update(value.as_bytes());
    Ok(())
}

fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} through a3s-power: {error}"),
    )
}

fn runtime_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.runtime_failed", message)
}

struct DecodedSlot {
    index: usize,
    slot_id: OcrBatchSlotId,
    input: OcrInput,
    text_window: Option<OcrNormalizedWindow>,
    image: UseResult<Arc<RgbImage>>,
}

#[derive(Clone)]
struct PreparedSlot {
    original_index: usize,
    slot_id: OcrBatchSlotId,
    input: OcrInput,
    text_window: Option<OcrNormalizedWindow>,
    // Scheduling, admission, and the blocking inference hand-off all retain
    // the same immutable decoded surface. Cloning a prepared slot must never
    // clone page pixels.
    image: Arc<RgbImage>,
}

struct BatchRun {
    outputs: Vec<UseResult<OcrProviderOutput>>,
    receipt: crate::OcrExecutionReceipt,
}

#[derive(Clone)]
enum StageResolution {
    Completed,
    Failed(UseError),
    Skipped(UseError),
    Unsupported,
}

impl StageResolution {
    fn outcome(&self, stage: OcrStage) -> OcrStageOutcome {
        match self {
            Self::Completed => OcrStageOutcome::completed(stage),
            Self::Failed(error) => OcrStageOutcome::failed(stage, error.clone()),
            Self::Skipped(error) => OcrStageOutcome::skipped(stage, error.clone()),
            Self::Unsupported => OcrStageOutcome::unsupported(stage),
        }
    }
}
