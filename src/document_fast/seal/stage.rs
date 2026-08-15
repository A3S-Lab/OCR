use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use a3s_power::error::PowerError;
use a3s_power::inference::{
    DevicePreference, ModelSession, ModelSessionPool, ModelSessionPoolPolicy,
};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use super::assets::PicodetLayoutAssets;
use super::decoder::{decode_page_views, merge_page_detections, DecodedSeal};
use super::native::{session_spec, NativePicodetLayout, LOCATION_COUNT, OUTPUT_WIDTH};
use super::preprocess::{adjacent_boundary_view, page_views, view_tensor, SealView};
use crate::cancellation::{check_cancelled, run_blocking_with};
use crate::preprocess::decode_image;
use crate::receipt::project_receipt;
use crate::{
    OcrBatchSlotId, OcrCanvasEdge, OcrExecutionReceipt, OcrImageCanvas, OcrProviderBatchSlot,
    OcrSealDetectionStatus,
};

const MAX_LAYOUT_BATCH: usize = 8;

#[derive(Clone)]
pub(in crate::document_fast) struct SealStageRunner {
    assets: PicodetLayoutAssets,
    sessions: ModelSessionPool<SealSession>,
}

impl SealStageRunner {
    pub(in crate::document_fast) fn from_env_optional() -> UseResult<Option<Self>> {
        PicodetLayoutAssets::from_env_optional()?
            .map(Self::new)
            .transpose()
    }

    fn new(assets: PicodetLayoutAssets) -> UseResult<Self> {
        let policy = ModelSessionPoolPolicy::new(2, 512 * 1024 * 1024, 1, 32)
            .map_err(|error| pool_error("configure", error))?;
        Ok(Self {
            assets,
            sessions: ModelSessionPool::new(DevicePreference::Auto, policy)
                .map_err(|error| pool_error("initialize", error))?,
        })
    }

    pub(in crate::document_fast) fn model_root(&self) -> &std::path::Path {
        &self.assets.root
    }

    pub(in crate::document_fast) async fn run(
        &self,
        slots: Vec<OcrProviderBatchSlot>,
        cancellation: CancellationToken,
    ) -> UseResult<SealStageBatch> {
        let decoded = decode_pages(slots, cancellation.clone()).await?;
        let mut pages = decoded
            .into_iter()
            .map(PageAccumulator::from_decoded)
            .collect::<Vec<_>>();
        let baseline = baseline_views(&pages);
        let mut receipts = Vec::new();
        if !baseline.is_empty() {
            let session = prepare_session(self, cancellation.clone()).await?;
            run_view_batches(
                session.clone(),
                baseline,
                &mut pages,
                &mut receipts,
                cancellation.clone(),
            )
            .await?;
            let adjacent = adjacent_views(&pages);
            if !adjacent.is_empty() {
                run_view_batches(session, adjacent, &mut pages, &mut receipts, cancellation)
                    .await?;
            }
        }
        Ok(SealStageBatch {
            slots: pages.into_iter().map(PageAccumulator::finish).collect(),
            receipts,
        })
    }
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

async fn prepare_session(
    runner: &SealStageRunner,
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
        .get_or_load(spec, &cancellation, move |runtime, loader_cancellation| {
            load_session(assets, runtime, loader_cancellation)
        })
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

async fn decode_pages(
    slots: Vec<OcrProviderBatchSlot>,
    cancellation: CancellationToken,
) -> UseResult<Vec<DecodedPage>> {
    run_blocking_with(
        "document-fast seal image decoding",
        cancellation,
        move |cancellation| {
            slots
                .into_iter()
                .map(|slot| {
                    check_cancelled(&cancellation)?;
                    Ok(DecodedPage {
                        slot_id: slot.slot_id,
                        adjacent_predecessor_slot_id: slot.adjacent_predecessor_slot_id,
                        image: decode_image(slot.input.bytes()),
                    })
                })
                .collect()
        },
    )
    .await
}

async fn run_view_batches(
    session: ModelSession<SealSession>,
    views: Vec<ViewReference>,
    pages: &mut [PageAccumulator],
    receipts: &mut Vec<OcrExecutionReceipt>,
    cancellation: CancellationToken,
) -> UseResult<()> {
    for chunk in views.chunks(MAX_LAYOUT_BATCH) {
        let prepared = prepare_views(chunk.to_vec(), cancellation.clone()).await;
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                fail_view_pages(chunk, pages, error);
                continue;
            }
        };
        let permit = match session.runtime().begin_wait(&cancellation).await {
            Ok(permit) => permit,
            Err(error) => {
                fail_view_pages(
                    chunk,
                    pages,
                    power_error("admit a PicoDet layout batch", error),
                );
                continue;
            }
        };
        match execute_batch(session.clone(), prepared, permit, cancellation.clone()).await {
            Ok(run) => {
                let receipt = project_receipt(run.receipt);
                receipts.push(receipt.clone());
                let mut receipt_pages = BTreeSet::new();
                for result in run.results {
                    let page = &mut pages[result.page_index];
                    if receipt_pages.insert(result.page_index) {
                        page.receipts.push(receipt.clone());
                    }
                    match result.seals {
                        Ok(seals) => page.add_seals(seals),
                        Err(error) => page.fail(error),
                    }
                }
            }
            Err(error) => fail_view_pages(chunk, pages, error),
        }
    }
    Ok(())
}

async fn prepare_views(
    views: Vec<ViewReference>,
    cancellation: CancellationToken,
) -> UseResult<PreparedBatch> {
    run_blocking_with(
        "PicoDet layout view preprocessing",
        cancellation,
        move |cancellation| {
            let mut tensor = Vec::with_capacity(
                views.len() * 3 * super::preprocess::INPUT_SIDE * super::preprocess::INPUT_SIDE,
            );
            for view in &views {
                check_cancelled(&cancellation)?;
                tensor.extend(view_tensor(&view.image, view.view)?);
            }
            Ok(PreparedBatch { views, tensor })
        },
    )
    .await
}

async fn execute_batch(
    session: ModelSession<SealSession>,
    prepared: PreparedBatch,
    permit: a3s_power::inference::ExecutionPermit,
    cancellation: CancellationToken,
) -> UseResult<BatchRun> {
    run_blocking_with(
        "PicoDet admitted layout batch",
        cancellation,
        move |cancellation| {
            let started = std::time::Instant::now();
            let engine = session
                .value()
                .engine
                .lock()
                .map_err(|_| runtime_error("The PicoDet layout engine lock is poisoned."))?;
            let inferred = engine.infer_batch(
                prepared.tensor,
                prepared.views.len(),
                &permit,
                &cancellation,
            )?;
            let sample_elements = LOCATION_COUNT * OUTPUT_WIDTH;
            let mut results = Vec::with_capacity(prepared.views.len());
            for (sample, reference) in prepared.views.into_iter().enumerate() {
                let start = sample * sample_elements;
                let end = start + sample_elements;
                let seals = decode_page_views(
                    &[(&reference.view, &inferred.tensor.values[start..end])],
                    reference.image.width(),
                    reference.image.height(),
                );
                results.push(ViewResult {
                    page_index: reference.page_index,
                    seals,
                });
            }
            if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
                eprintln!(
                    "A3S_OCR_SEAL_TIMING views={} device={} total_ms={:.3}",
                    results.len(),
                    inferred.receipt.runtime.device,
                    started.elapsed().as_secs_f64() * 1_000.0,
                );
            }
            Ok(BatchRun {
                results,
                receipt: inferred.receipt,
            })
        },
    )
    .await
}

fn baseline_views(pages: &[PageAccumulator]) -> Vec<ViewReference> {
    pages
        .iter()
        .enumerate()
        .flat_map(|(page_index, page)| {
            page.image.iter().flat_map(move |image| {
                page_views(image)
                    .into_iter()
                    .map(move |view| ViewReference {
                        page_index,
                        image: Arc::clone(image),
                        view,
                    })
            })
        })
        .collect()
}

fn adjacent_views(pages: &[PageAccumulator]) -> Vec<ViewReference> {
    let mut views = Vec::new();
    for page_index in 1..pages.len() {
        let predecessor = &pages[page_index - 1];
        let page = &pages[page_index];
        if page.adjacent_predecessor_slot_id.as_ref() != Some(&predecessor.slot_id) {
            continue;
        }
        let (Some(image), Some(predecessor_canvas)) = (&page.image, predecessor.canvas) else {
            continue;
        };
        for edge in [OcrCanvasEdge::Left, OcrCanvasEdge::Right] {
            let candidate = predecessor
                .seals
                .iter()
                .filter(|seal| {
                    seal.status == OcrSealDetectionStatus::BoundaryCandidate
                        && seal.clipped_edge == Some(edge)
                        && seal.region.width <= predecessor_canvas.width / 4
                        && seal.region.height <= predecessor_canvas.height / 2
                })
                .max_by(|left, right| left.confidence.total_cmp(&right.confidence));
            let Some(candidate) = candidate else {
                continue;
            };
            if let Some(view) =
                adjacent_boundary_view(image, edge, candidate.region, predecessor_canvas.height)
            {
                views.push(ViewReference {
                    page_index,
                    image: Arc::clone(image),
                    view,
                });
            }
        }
    }
    views
}

fn fail_view_pages(views: &[ViewReference], pages: &mut [PageAccumulator], error: UseError) {
    let mut failed = BTreeSet::new();
    for view in views {
        if failed.insert(view.page_index) {
            pages[view.page_index].fail(error.clone());
        }
    }
}

struct DecodedPage {
    slot_id: OcrBatchSlotId,
    adjacent_predecessor_slot_id: Option<OcrBatchSlotId>,
    image: UseResult<RgbImage>,
}

struct PageAccumulator {
    slot_id: OcrBatchSlotId,
    adjacent_predecessor_slot_id: Option<OcrBatchSlotId>,
    image: Option<Arc<RgbImage>>,
    canvas: Option<OcrImageCanvas>,
    seals: Vec<DecodedSeal>,
    receipts: Vec<OcrExecutionReceipt>,
    error: Option<UseError>,
}

impl PageAccumulator {
    fn from_decoded(decoded: DecodedPage) -> Self {
        match decoded.image {
            Ok(image) => match OcrImageCanvas::new(image.width(), image.height()) {
                Ok(canvas) => Self {
                    slot_id: decoded.slot_id,
                    adjacent_predecessor_slot_id: decoded.adjacent_predecessor_slot_id,
                    image: Some(Arc::new(image)),
                    canvas: Some(canvas),
                    seals: Vec::new(),
                    receipts: Vec::new(),
                    error: None,
                },
                Err(error) => {
                    Self::failed(decoded.slot_id, decoded.adjacent_predecessor_slot_id, error)
                }
            },
            Err(error) => {
                Self::failed(decoded.slot_id, decoded.adjacent_predecessor_slot_id, error)
            }
        }
    }

    fn failed(
        slot_id: OcrBatchSlotId,
        adjacent_predecessor_slot_id: Option<OcrBatchSlotId>,
        error: UseError,
    ) -> Self {
        Self {
            slot_id,
            adjacent_predecessor_slot_id,
            image: None,
            canvas: None,
            seals: Vec::new(),
            receipts: Vec::new(),
            error: Some(error),
        }
    }

    fn add_seals(&mut self, seals: Vec<DecodedSeal>) {
        self.seals = merge_page_detections(std::mem::take(&mut self.seals), seals);
    }

    fn fail(&mut self, error: UseError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn finish(self) -> SealSlotResult {
        let page = match (self.error, self.canvas) {
            (Some(error), _) => Err(error),
            (None, Some(canvas)) => Ok(DetectedSealPage {
                canvas,
                seals: self.seals,
                receipts: self.receipts,
            }),
            (None, None) => Err(runtime_error(
                "A seal slot lost its exact source canvas during execution.",
            )),
        };
        SealSlotResult {
            slot_id: self.slot_id,
            page,
        }
    }
}

#[derive(Clone)]
struct ViewReference {
    page_index: usize,
    image: Arc<RgbImage>,
    view: SealView,
}

struct PreparedBatch {
    views: Vec<ViewReference>,
    tensor: Vec<f32>,
}

struct ViewResult {
    page_index: usize,
    seals: UseResult<Vec<DecodedSeal>>,
}

struct BatchRun {
    results: Vec<ViewResult>,
    receipt: a3s_power::inference::ExecutionReceipt,
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
