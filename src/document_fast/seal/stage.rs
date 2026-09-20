use std::collections::BTreeSet;
#[cfg(test)]
use std::sync::Arc;
use std::sync::Mutex;

use a3s_power::error::PowerError;
use a3s_power::inference::{
    DevicePreference, ModelSession, ModelSessionPool, ModelSessionPoolPolicy, RuntimeDeviceKind,
};
use a3s_use_core::{UseError, UseResult};
#[cfg(test)]
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use super::super::page_orientation::SourceCanvasTransform;
use super::super::shared_decode::{validate_decoded_cardinality, SharedDecodedImage};
use super::super::wired::PixelRect;
use super::assets::PicodetLayoutAssets;
use super::decoder::DecodedSeal;
use super::fusion::{fuse_seal_evidence, has_text_fusion_consumer};
use super::native::{session_spec, NativePicodetLayout, MAX_BATCH_SIZE};
use super::refinement::MAX_REFINEMENT_VIEWS_PER_PAGE;
use super::text::{SealTextPageReference, SealTextStageBatch, SealTextStageRunner};
use crate::cancellation::{check_cancelled, run_blocking_with};
#[cfg(test)]
use crate::preprocess::decode_image;
use crate::{OcrBatchSlotId, OcrExecutionReceipt, OcrImageCanvas, OcrProviderBatchSlot};

mod admission;
mod batching;
mod branches;
mod decode;
mod state;

use state::{DecodedPage, PageAccumulator, ViewReference};

const MAX_LAYOUT_SESSIONS: usize = 4;
const CUDA_DEVICE_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CUDA_LAYOUT_SESSION_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone)]
pub(in crate::document_fast) struct SealStageRunner {
    assets: PicodetLayoutAssets,
    sessions: ModelSessionPool<SealSession>,
    text: Option<SealTextStageRunner>,
}

impl SealStageRunner {
    pub(in crate::document_fast) fn from_env_optional() -> UseResult<Option<Self>> {
        let layout = PicodetLayoutAssets::from_env_optional()?;
        let text = SealTextStageRunner::from_env_optional()?;
        match (layout, text) {
            (Some(layout), text) => Self::new(layout, text).map(Some),
            (None, None) => Ok(None),
            (None, Some(_)) => Err(UseError::new(
                "use.ocr.seal_text_model_invalid",
                "The seal-text verifier requires the reviewed PicoDet layout model.",
            )
            .with_suggestion(
                "Configure A3S_OCR_PICODET_LAYOUT_MODEL_DIR or remove A3S_OCR_SEAL_TEXT_MODEL_DIR.",
            )),
        }
    }

    fn new(assets: PicodetLayoutAssets, text: Option<SealTextStageRunner>) -> UseResult<Self> {
        let policy = ModelSessionPoolPolicy::new(
            MAX_LAYOUT_SESSIONS,
            512 * 1024 * 1024,
            MAX_LAYOUT_SESSIONS,
            32,
        )
        .map_err(|error| pool_error("configure", error))?;
        Ok(Self {
            assets,
            sessions: ModelSessionPool::new(DevicePreference::Auto, policy)
                .map_err(|error| pool_error("initialize", error))?,
            text,
        })
    }

    pub(in crate::document_fast) fn model_root(&self) -> &std::path::Path {
        &self.assets.root
    }

    pub(in crate::document_fast) fn layout_model_family(&self) -> &'static str {
        self.assets.profile.family()
    }

    pub(in crate::document_fast) fn has_text_fusion(&self) -> bool {
        self.text.is_some()
    }

    fn active_layout_session_count(&self) -> UseResult<usize> {
        #[cfg(test)]
        if let Some(count) = std::env::var_os("A3S_OCR_TEST_PICODET_LAYOUT_SESSION_COUNT")
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
            .filter(|count| (1..=MAX_LAYOUT_SESSIONS).contains(count))
        {
            return Ok(count);
        }

        let memory = self
            .sessions
            .memory_snapshot()
            .map_err(|error| pool_error("discover layout replica memory", error))?;
        let available = memory.device.as_ref().map(|device| device.available_bytes);
        Ok(layout_session_count_for(
            self.sessions.snapshot().device.kind,
            available,
            MAX_LAYOUT_SESSIONS,
        ))
    }

    #[cfg(test)]
    pub(in crate::document_fast) async fn run(
        &self,
        slots: Vec<OcrProviderBatchSlot>,
        cancellation: CancellationToken,
    ) -> UseResult<SealStageBatch> {
        let decoded = decode_pages(slots, cancellation.clone()).await?;
        self.run_decoded_pages(decoded, cancellation).await
    }

    pub(in crate::document_fast) async fn run_source_with_oriented_layout(
        &self,
        slots: Vec<OcrProviderBatchSlot>,
        source_images: Vec<SharedDecodedImage>,
        oriented_images: Vec<SharedDecodedImage>,
        transforms: Vec<Option<SourceCanvasTransform>>,
        cancellation: CancellationToken,
    ) -> UseResult<SealStageBatch> {
        let stage_started = std::time::Instant::now();
        validate_decoded_cardinality(&slots, &source_images)?;
        if oriented_images.len() != slots.len() {
            return Err(runtime_error(
                "Page orientation changed seal image cardinality.",
            ));
        }
        if transforms.len() != slots.len() {
            return Err(runtime_error(
                "Page orientation changed seal transform cardinality.",
            ));
        }

        let mut pages = slots
            .into_iter()
            .zip(source_images)
            .map(|(slot, image)| {
                PageAccumulator::from_decoded(DecodedPage {
                    slot_id: slot.slot_id,
                    adjacent_predecessor_slot_id: slot.adjacent_predecessor_slot_id,
                    image,
                })
            })
            .collect::<Vec<_>>();
        let mut supplemental_pages = Vec::new();
        let mut supplemental_bindings = Vec::new();
        for (page_index, ((oriented, transform), source)) in oriented_images
            .into_iter()
            .zip(transforms)
            .zip(&pages)
            .enumerate()
        {
            let Some(transform) = transform.filter(|transform| !transform.is_identity()) else {
                continue;
            };
            let (Some(source_image), Ok(oriented_image)) = (&source.image, &oriented) else {
                return Err(runtime_error(
                    "A non-identity orientation transform was bound to a failed decoded image.",
                ));
            };
            let source_canvas = OcrImageCanvas::new(source_image.width(), source_image.height())?;
            let oriented_canvas =
                OcrImageCanvas::new(oriented_image.width(), oriented_image.height())?;
            if source_canvas != transform.source_canvas()?
                || oriented_canvas != transform.oriented_canvas()?
            {
                return Err(runtime_error(
                    "A non-identity orientation transform changed an admitted seal canvas.",
                ));
            }
            supplemental_pages.push(PageAccumulator::from_decoded(DecodedPage {
                slot_id: source.slot_id.clone(),
                adjacent_predecessor_slot_id: source.adjacent_predecessor_slot_id.clone(),
                image: oriented,
            }));
            supplemental_bindings.push((page_index, transform));
        }

        let source_page_count = pages.iter().filter(|page| page.image.is_some()).count();
        let supplemental_page_count = supplemental_pages
            .iter()
            .filter(|page| page.image.is_some())
            .count();
        let requested_layout_sessions =
            self.active_layout_session_count()?
                .min(layout_session_demand(
                    source_page_count,
                    supplemental_page_count,
                    MAX_LAYOUT_SESSIONS,
                ));
        let text_prepare_future = async {
            match self.text.as_ref() {
                Some(text) => Some(text.prepare(cancellation.clone()).await),
                None => None,
            }
        };
        let layout_future = async {
            if !pages.iter().any(|page| page.image.is_some()) {
                return Ok::<_, UseError>((Vec::new(), Vec::new()));
            }
            let layout_sessions =
                prepare_layout_sessions(self, requested_layout_sessions, cancellation.clone())
                    .await?;
            trace_layout_session_count(requested_layout_sessions, layout_sessions.len());
            if layout_sessions.is_empty() {
                return Err(runtime_error(
                    "PicoDet layout session preparation omitted its primary session.",
                ));
            }
            if supplemental_pages.is_empty() {
                let source_receipts = run_layout_pages_with_sessions(
                    self,
                    &mut pages,
                    &layout_sessions,
                    cancellation.clone(),
                )
                .await?;
                return Ok((source_receipts, Vec::new()));
            }
            if layout_sessions.len() == 1 {
                let source_receipts = run_layout_pages_with_sessions(
                    self,
                    &mut pages,
                    &layout_sessions,
                    cancellation.clone(),
                )
                .await?;
                let supplemental_receipts = run_layout_pages_with_sessions(
                    self,
                    &mut supplemental_pages,
                    &layout_sessions,
                    cancellation.clone(),
                )
                .await?;
                return Ok((source_receipts, supplemental_receipts));
            }
            let (source_session_count, supplemental_session_count) =
                layout_branch_session_counts(layout_sessions.len());
            branches::run_paired(
                self,
                &mut pages,
                &mut supplemental_pages,
                &layout_sessions,
                source_session_count,
                supplemental_session_count,
                cancellation.clone(),
            )
            .await
        };
        let (layout_result, text_preparation) = tokio::join!(layout_future, text_prepare_future);
        let (mut receipts, supplemental_receipts) = layout_result?;
        receipts.extend(supplemental_receipts);
        if supplemental_pages.len() != supplemental_bindings.len() {
            return Err(runtime_error(
                "Oriented seal supplements changed source binding cardinality.",
            ));
        }
        for (supplemental, (page_index, transform)) in
            supplemental_pages.into_iter().zip(supplemental_bindings)
        {
            pages[page_index].merge_oriented_supplemental(supplemental, transform)?;
        }
        let text_references = text_page_references(&pages);
        trace_text_page_admission(&pages, &text_references);
        let layout_elapsed = stage_started.elapsed();
        let text_started = std::time::Instant::now();
        if let Some(text_preparation) = text_preparation.filter(|_| !text_references.is_empty()) {
            let expected_pages = text_references
                .iter()
                .map(|reference| reference.page_index)
                .collect::<BTreeSet<_>>();
            let text = text_preparation?
                .run(text_references, cancellation.clone())
                .await?;
            receipts.extend(text.receipts.iter().cloned());
            apply_text_fusion(&mut pages, text, &expected_pages)?;
        }
        let text_elapsed = text_started.elapsed();
        let batch = SealStageBatch {
            slots: pages.into_iter().map(PageAccumulator::finish).collect(),
            receipts,
        };
        if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
            eprintln!(
                "A3S_OCR_SEAL_STAGE_TIMING slots={} layout_ms={:.3} text_ms={:.3} total_ms={:.3}",
                batch.slots.len(),
                layout_elapsed.as_secs_f64() * 1_000.0,
                text_elapsed.as_secs_f64() * 1_000.0,
                stage_started.elapsed().as_secs_f64() * 1_000.0,
            );
        }
        Ok(batch)
    }

    #[cfg(test)]
    async fn run_decoded_pages(
        &self,
        decoded: Vec<DecodedPage>,
        cancellation: CancellationToken,
    ) -> UseResult<SealStageBatch> {
        let mut pages = decoded
            .into_iter()
            .map(PageAccumulator::from_decoded)
            .collect::<Vec<_>>();
        let text_prepare_future = async {
            match self.text.as_ref() {
                Some(text) => Some(text.prepare(cancellation.clone()).await),
                None => None,
            }
        };
        let layout_future = run_layout_pages(self, &mut pages, cancellation.clone());
        let (layout_result, text_preparation) = tokio::join!(layout_future, text_prepare_future);
        let mut receipts = layout_result?;
        let text_references = text_page_references(&pages);
        trace_text_page_admission(&pages, &text_references);
        if let Some(text_preparation) = text_preparation.filter(|_| !text_references.is_empty()) {
            let expected_pages = text_references
                .iter()
                .map(|reference| reference.page_index)
                .collect::<BTreeSet<_>>();
            let text = text_preparation?
                .run(text_references, cancellation.clone())
                .await?;
            receipts.extend(text.receipts.iter().cloned());
            apply_text_fusion(&mut pages, text, &expected_pages)?;
        }
        Ok(SealStageBatch {
            slots: pages.into_iter().map(PageAccumulator::finish).collect(),
            receipts,
        })
    }
}

#[cfg(test)]
async fn run_layout_pages(
    runner: &SealStageRunner,
    pages: &mut [PageAccumulator],
    cancellation: CancellationToken,
) -> UseResult<Vec<OcrExecutionReceipt>> {
    if !pages.iter().any(|page| page.image.is_some()) {
        return Ok(Vec::new());
    }
    let session = prepare_session(runner, 0, cancellation.clone()).await?;
    run_layout_pages_with_sessions(runner, pages, &[session], cancellation).await
}

async fn run_layout_pages_with_sessions(
    runner: &SealStageRunner,
    pages: &mut [PageAccumulator],
    sessions: &[ModelSession<SealSession>],
    cancellation: CancellationToken,
) -> UseResult<Vec<OcrExecutionReceipt>> {
    branches::run_single(runner, pages, sessions, cancellation).await
}

fn text_page_references(pages: &[PageAccumulator]) -> Vec<SealTextPageReference> {
    text_page_references_for(pages, seal_text_dead_work_elimination_enabled())
}

fn text_page_references_for(
    pages: &[PageAccumulator],
    eliminate_dead_work: bool,
) -> Vec<SealTextPageReference> {
    pages
        .iter()
        .enumerate()
        .filter_map(|(page_index, page)| {
            page.image.as_ref().and_then(|image| {
                (!eliminate_dead_work || has_text_fusion_consumer(&page.layout_images, &page.seals))
                    .then(|| SealTextPageReference {
                        page_index,
                        image: image.clone(),
                    })
            })
        })
        .collect()
}

fn seal_text_dead_work_elimination_enabled() -> bool {
    #[cfg(test)]
    if std::env::var_os("A3S_OCR_TEST_EAGER_ALL_PAGE_SEAL_TEXT").is_some() {
        return false;
    }
    true
}

fn trace_text_page_admission(pages: &[PageAccumulator], references: &[SealTextPageReference]) {
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_none() {
        return;
    }
    let valid_pages = pages.iter().filter(|page| page.image.is_some()).count();
    let consumer_pages = pages
        .iter()
        .filter(|page| has_text_fusion_consumer(&page.layout_images, &page.seals))
        .count();
    eprintln!(
        "A3S_OCR_SEAL_TEXT_ADMISSION valid_pages={valid_pages} consumer_pages={consumer_pages} admitted_pages={} eliminated_pages={}",
        references.len(),
        valid_pages.saturating_sub(references.len()),
    );
}

fn apply_text_fusion(
    pages: &mut [PageAccumulator],
    text: SealTextStageBatch,
    expected_pages: &BTreeSet<usize>,
) -> UseResult<()> {
    if text.pages.len() != expected_pages.len() {
        return Err(runtime_error(
            "The seal-text verifier changed admitted-page cardinality.",
        ));
    }
    let mut seen = BTreeSet::new();
    for result in text.pages {
        if result.page_index >= pages.len()
            || !expected_pages.contains(&result.page_index)
            || !seen.insert(result.page_index)
        {
            return Err(runtime_error(
                "The seal-text verifier changed page identity or ordering.",
            ));
        }
        let page = &mut pages[result.page_index];
        match result.evidence {
            Ok(evidence) => {
                let canvas = page.canvas.ok_or_else(|| {
                    runtime_error(
                        "Seal-text fusion lost the immutable source canvas for a valid page.",
                    )
                })?;
                let fused = fuse_seal_evidence(
                    &evidence,
                    &page.layout_images,
                    &page.seals,
                    PixelRect {
                        x: 0,
                        y: 0,
                        width: canvas.width,
                        height: canvas.height,
                    },
                );
                if std::env::var_os("A3S_OCR_TRACE_SEAL_DETECTIONS").is_some() {
                    eprintln!(
                        "A3S_OCR_SEAL_TEXT_EVIDENCE slot_id={:?} direct={:?} clockwise90={:?} clockwise180={:?} clockwise270={:?} fused={:?}",
                        page.slot_id,
                        evidence.direct,
                        evidence.clockwise90,
                        evidence.clockwise180,
                        evidence.clockwise270,
                        fused,
                    );
                }
                page.receipts.extend(evidence.receipts.iter().cloned());
                page.add_seals(fused);
            }
            Err(error) => page.fail(error),
        }
    }
    if &seen != expected_pages {
        return Err(runtime_error(
            "The seal-text verifier omitted one or more admitted pages.",
        ));
    }
    Ok(())
}

fn trace_view_count(kind: &str, views: usize) {
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
        eprintln!("A3S_OCR_SEAL_VIEWS kind={kind} views={views}");
    }
}

fn trace_layout_session_count(requested: usize, active: usize) {
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
        eprintln!("A3S_OCR_SEAL_LAYOUT_SESSIONS requested={requested} active={active}");
    }
}

fn layout_session_count_for(
    kind: RuntimeDeviceKind,
    available_device_bytes: Option<u64>,
    maximum_sessions: usize,
) -> usize {
    if kind != RuntimeDeviceKind::Cuda {
        return 1;
    }
    let funded = available_device_bytes
        .unwrap_or_default()
        .saturating_sub(CUDA_DEVICE_RESERVE_BYTES)
        .checked_div(CUDA_LAYOUT_SESSION_BUDGET_BYTES)
        .unwrap_or_default();
    usize::try_from(funded)
        .unwrap_or(maximum_sessions)
        .clamp(1, maximum_sessions.max(1))
}

fn layout_session_demand(
    source_pages: usize,
    supplemental_pages: usize,
    maximum_sessions: usize,
) -> usize {
    let source_batches = maximum_refinement_batch_count(source_pages);
    let supplemental_batches = maximum_refinement_batch_count(supplemental_pages);
    let independent_branches = usize::from(source_pages > 0) + usize::from(supplemental_pages > 0);
    source_batches
        .saturating_add(supplemental_batches)
        .max(independent_branches)
        .clamp(1, maximum_sessions.max(1))
}

fn layout_branch_session_counts(available_sessions: usize) -> (usize, usize) {
    let supplemental_sessions = available_sessions / 2;
    (
        available_sessions.saturating_sub(supplemental_sessions),
        supplemental_sessions,
    )
}

fn maximum_refinement_batch_count(page_count: usize) -> usize {
    page_count
        .saturating_mul(MAX_REFINEMENT_VIEWS_PER_PAGE)
        .div_ceil(MAX_BATCH_SIZE)
}

pub(in crate::document_fast) struct SealStageBatch {
    pub(in crate::document_fast) slots: Vec<SealSlotResult>,
    pub(in crate::document_fast) receipts: Vec<OcrExecutionReceipt>,
}

pub(in crate::document_fast) struct SealSlotResult {
    pub(in crate::document_fast) slot_id: OcrBatchSlotId,
    pub(in crate::document_fast) page: UseResult<DetectedSealPage>,
}

#[derive(Clone)]
pub(in crate::document_fast) struct DetectedSealPage {
    pub(super) canvas: OcrImageCanvas,
    pub(super) seals: Vec<DecodedSeal>,
    pub(super) receipts: Vec<OcrExecutionReceipt>,
}

struct SealSession {
    engine: Mutex<NativePicodetLayout>,
}

async fn prepare_layout_sessions(
    runner: &SealStageRunner,
    requested: usize,
    cancellation: CancellationToken,
) -> UseResult<Vec<ModelSession<SealSession>>> {
    if !(1..=MAX_LAYOUT_SESSIONS).contains(&requested) {
        return Err(runtime_error(
            "PicoDet layout session request is outside its bounded replica count.",
        ));
    }
    if requested == 1 {
        return prepare_session(runner, 0, cancellation)
            .await
            .map(|session| vec![session]);
    }

    let primary = prepare_session(runner, 0, cancellation.clone());
    let secondary = prepare_optional_session(runner, 1, requested, cancellation.clone());
    let tertiary = prepare_optional_session(runner, 2, requested, cancellation.clone());
    let quaternary = prepare_optional_session(runner, 3, requested, cancellation.clone());
    let (primary, secondary, tertiary, quaternary) =
        tokio::join!(primary, secondary, tertiary, quaternary);
    let primary = primary?;
    check_cancelled(&cancellation)?;
    let mut sessions = vec![primary];
    for session in [secondary, tertiary, quaternary]
        .into_iter()
        .flatten()
        .flatten()
    {
        sessions.push(session);
    }
    Ok(sessions)
}

async fn prepare_optional_session(
    runner: &SealStageRunner,
    execution_replica: usize,
    requested: usize,
    cancellation: CancellationToken,
) -> Option<UseResult<ModelSession<SealSession>>> {
    if execution_replica >= requested {
        return None;
    }
    Some(prepare_session(runner, execution_replica, cancellation).await)
}

async fn prepare_session(
    runner: &SealStageRunner,
    execution_replica: usize,
    cancellation: CancellationToken,
) -> UseResult<ModelSession<SealSession>> {
    let assets = runner.assets.clone();
    let spec_assets = assets.clone();
    let spec = run_blocking_with(
        "PicoDet layout session declaration",
        cancellation.clone(),
        move |cancellation| {
            check_cancelled(&cancellation)?;
            session_spec(&spec_assets)
        },
    )
    .await?;
    runner
        .sessions
        .get_or_load_replica(
            spec,
            execution_replica,
            &cancellation,
            move |runtime, loader_cancellation| load_session(assets, runtime, loader_cancellation),
        )
        .await
        .map_err(|error| pool_error("load", error))
}

async fn load_session(
    assets: PicodetLayoutAssets,
    runtime: a3s_power::inference::EmbeddedRuntime,
    cancellation: CancellationToken,
) -> a3s_power::error::Result<SealSession> {
    tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        let engine = NativePicodetLayout::load_with_runtime(&assets, runtime)
            .map_err(|error| PowerError::InferenceFailed(error.message))?;
        Ok(SealSession {
            engine: Mutex::new(engine),
        })
    })
    .await
    .map_err(|error| PowerError::InferenceFailed(format!("PicoDet loader task failed: {error}")))?
}

#[cfg(test)]
async fn decode_pages(
    slots: Vec<OcrProviderBatchSlot>,
    cancellation: CancellationToken,
) -> UseResult<Vec<DecodedPage>> {
    let page_count = slots.len();
    run_blocking_with(
        "document-fast seal image decoding",
        cancellation,
        move |cancellation| {
            let started = std::time::Instant::now();
            let decoded = slots
                .into_par_iter()
                .map(|slot| {
                    check_cancelled(&cancellation)?;
                    Ok(DecodedPage {
                        slot_id: slot.slot_id,
                        adjacent_predecessor_slot_id: slot.adjacent_predecessor_slot_id,
                        image: decode_image(slot.input.bytes()).map(Arc::new),
                    })
                })
                .collect::<UseResult<Vec<_>>>();
            trace_phase_timing("decode", page_count, started.elapsed());
            decoded
        },
    )
    .await
}

fn trace_phase_timing(phase: &str, items: usize, elapsed: std::time::Duration) {
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
        eprintln!(
            "A3S_OCR_SEAL_TIMING phase={phase} items={items} total_ms={:.3}",
            elapsed.as_secs_f64() * 1_000.0,
        );
    }
}

fn pool_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} the PicoDet layout session pool: {error}"),
    )
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

#[cfg(test)]
#[path = "stage/tests.rs"]
mod tests;
