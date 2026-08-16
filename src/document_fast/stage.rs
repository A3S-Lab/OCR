use std::sync::{Arc, Mutex};

use a3s_power::error::PowerError;
use a3s_power::inference::{
    DevicePreference, ModelSession, ModelSessionPool, ModelSessionPoolPolicy,
};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use super::assets::SlanetPlusAssets;
use super::decoder::{SlanetPlusDecoder, StructureGrid};
use super::native::{session_spec, NativeSlanetPlus};
use super::orientation::TableCropOrientation;
use super::preprocess::crop_tensor;
use super::wired::{candidates, PixelRect, WiredCandidate};
use crate::cancellation::{check_cancelled, run_blocking_with};
use crate::preprocess::decode_image;
use crate::receipt::project_receipt;
use crate::{OcrExecutionReceipt, OcrImageCanvas, OcrProviderBatchSlot};

#[cfg(test)]
mod tests;

const MAX_CANDIDATES_PER_PAGE: usize = 8;
const MAX_ENCODER_BATCH: usize = 16;
const MIN_STRUCTURE_CONFIDENCE: f32 = 0.5;

#[derive(Clone)]
pub(super) struct TableStageRunner {
    assets: SlanetPlusAssets,
    sessions: ModelSessionPool<TableSession>,
}

impl TableStageRunner {
    pub(super) fn new(assets: SlanetPlusAssets) -> UseResult<Self> {
        let policy = ModelSessionPoolPolicy::new(2, 512 * 1024 * 1024, 1, 32)
            .map_err(|error| pool_error("configure", error))?;
        Ok(Self {
            assets,
            sessions: ModelSessionPool::new(DevicePreference::Auto, policy)
                .map_err(|error| pool_error("initialize", error))?,
        })
    }

    pub(super) fn model_root(&self) -> &std::path::Path {
        &self.assets.root
    }

    pub(super) async fn run(
        &self,
        slots: Vec<OcrProviderBatchSlot>,
        cancellation: CancellationToken,
    ) -> UseResult<TableStageBatch> {
        let decoded = decode_pages(slots, cancellation.clone()).await?;
        let mut pages = Vec::with_capacity(decoded.len());
        let mut crops = Vec::new();
        for decoded in decoded {
            match decoded.page {
                Ok(DecodedTablePage { image, candidates }) => {
                    let canvas = OcrImageCanvas::new(image.width(), image.height())?;
                    if candidates.len() > MAX_CANDIDATES_PER_PAGE {
                        pages.push(PageAccumulator::failed(
                            decoded.slot_id,
                            runtime_error(format!(
                                "A page produced more than {MAX_CANDIDATES_PER_PAGE} wired-table candidates."
                            )),
                        ));
                        continue;
                    }
                    let page_index = pages.len();
                    let image = Arc::new(image);
                    let page = PageAccumulator::ready(decoded.slot_id, canvas, candidates.len());
                    for (table_index, candidate) in candidates.into_iter().enumerate() {
                        crops.push(CropReference {
                            page_index,
                            table_index,
                            image: Arc::clone(&image),
                            region: candidate.inference_region,
                            table_region: candidate.region,
                            orientation: candidate.orientation,
                        });
                    }
                    pages.push(page);
                }
                Err(error) => pages.push(PageAccumulator::failed(decoded.slot_id, error)),
            }
        }

        let mut receipts = Vec::new();
        if !crops.is_empty() {
            let session = prepare_session(self, cancellation.clone()).await?;
            for chunk in crops.chunks(MAX_ENCODER_BATCH) {
                let prepared = prepare_crops(chunk.to_vec(), cancellation.clone()).await;
                let prepared = match prepared {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        fail_crop_pages(chunk, &mut pages, error);
                        continue;
                    }
                };
                let permit = match session.runtime().begin_wait(&cancellation).await {
                    Ok(permit) => permit,
                    Err(error) => {
                        fail_crop_pages(
                            chunk,
                            &mut pages,
                            power_error("admit a SLANet-Plus encoder batch", error),
                        );
                        continue;
                    }
                };
                let run =
                    execute_batch(session.clone(), prepared, permit, cancellation.clone()).await;
                match run {
                    Ok(run) => {
                        let receipt = project_receipt(run.receipt);
                        receipts.push(receipt.clone());
                        for result in run.results {
                            let page = &mut pages[result.page_index];
                            page.receipts.push(receipt.clone());
                            match result.grid {
                                Ok(grid) => {
                                    page.tables[result.table_index] = Some(DetectedTable {
                                        region: result.region,
                                        grid,
                                    })
                                }
                                Err(error) => page.fail(error),
                            }
                        }
                    }
                    Err(error) => fail_crop_pages(chunk, &mut pages, error),
                }
            }
        }

        let slots = pages.into_iter().map(PageAccumulator::finish).collect();
        Ok(TableStageBatch { slots, receipts })
    }
}

pub(super) struct TableStageBatch {
    pub(super) slots: Vec<TableSlotResult>,
    pub(super) receipts: Vec<OcrExecutionReceipt>,
}

pub(super) struct TableSlotResult {
    pub(super) slot_id: crate::OcrBatchSlotId,
    pub(super) page: UseResult<DetectedPage>,
}

pub(super) struct DetectedPage {
    pub(super) canvas: OcrImageCanvas,
    pub(super) tables: Vec<DetectedTable>,
    pub(super) receipts: Vec<OcrExecutionReceipt>,
}

pub(super) struct DetectedTable {
    pub(super) region: PixelRect,
    pub(super) grid: StructureGrid,
}

struct TableSession {
    engine: Mutex<TableEngine>,
}

struct TableEngine {
    encoder: NativeSlanetPlus,
    decoder: SlanetPlusDecoder,
}

async fn prepare_session(
    runner: &TableStageRunner,
    cancellation: CancellationToken,
) -> UseResult<ModelSession<TableSession>> {
    let assets = runner.assets.clone();
    let spec_assets = assets.clone();
    let spec = run_blocking_with(
        "SLANet-Plus session declaration",
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
    assets: SlanetPlusAssets,
    runtime: a3s_power::inference::EmbeddedRuntime,
    cancellation: CancellationToken,
) -> a3s_power::error::Result<TableSession> {
    tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        let encoder = NativeSlanetPlus::load_with_runtime(&assets, runtime)
            .map_err(|error| PowerError::InferenceFailed(error.message))?;
        let decoder = SlanetPlusDecoder::load(&assets.decoder_weights, &assets.dictionary)
            .map_err(|error| PowerError::InferenceFailed(error.message))?;
        Ok(TableSession {
            engine: Mutex::new(TableEngine { encoder, decoder }),
        })
    })
    .await
    .map_err(|error| {
        PowerError::InferenceFailed(format!("SLANet-Plus loader task failed: {error}"))
    })?
}

async fn decode_pages(
    slots: Vec<OcrProviderBatchSlot>,
    cancellation: CancellationToken,
) -> UseResult<Vec<DecodedPage>> {
    run_blocking_with(
        "document-fast table image decoding",
        cancellation,
        move |cancellation| {
            slots
                .into_iter()
                .map(|slot| {
                    check_cancelled(&cancellation)?;
                    let page = decode_image(slot.input.bytes()).and_then(|image| {
                        let candidates = candidates(&image, &cancellation)?;
                        Ok(DecodedTablePage { image, candidates })
                    });
                    Ok(DecodedPage {
                        slot_id: slot.slot_id,
                        page,
                    })
                })
                .collect()
        },
    )
    .await
}

async fn prepare_crops(
    crops: Vec<CropReference>,
    cancellation: CancellationToken,
) -> UseResult<PreparedBatch> {
    run_blocking_with(
        "SLANet-Plus crop preprocessing",
        cancellation,
        move |cancellation| {
            let mut tensor = Vec::with_capacity(
                crops.len() * 3 * super::preprocess::INPUT_SIDE * super::preprocess::INPUT_SIDE,
            );
            for crop in &crops {
                check_cancelled(&cancellation)?;
                tensor.extend(crop_tensor(&crop.image, crop.region, crop.orientation)?);
            }
            Ok(PreparedBatch { crops, tensor })
        },
    )
    .await
}

async fn execute_batch(
    session: ModelSession<TableSession>,
    prepared: PreparedBatch,
    permit: a3s_power::inference::ExecutionPermit,
    cancellation: CancellationToken,
) -> UseResult<BatchRun> {
    run_blocking_with(
        "SLANet-Plus admitted encoder batch",
        cancellation,
        move |cancellation| {
            let trace = std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some();
            let batch_started = std::time::Instant::now();
            let engine = session
                .value()
                .engine
                .lock()
                .map_err(|_| runtime_error("The SLANet-Plus engine lock is poisoned."))?;
            let encoded = engine.encoder.encode_batch(
                prepared.tensor,
                prepared.crops.len(),
                &permit,
                &cancellation,
            )?;
            let encoded_at = batch_started.elapsed();
            let sample_elements = 256 * 96;
            let mut results = Vec::with_capacity(prepared.crops.len());
            for (sample, crop) in prepared.crops.into_iter().enumerate() {
                let start = sample * sample_elements;
                let end = start + sample_elements;
                let grid = engine
                    .decoder
                    .decode(
                        &encoded.tensor.values[start..end],
                        crop.region,
                        crop.orientation,
                        &cancellation,
                    )
                    .and_then(|decoded| decoded.into_grid())
                    .and_then(validate_grid);
                let region = grid
                    .as_ref()
                    .map(|grid| table_evidence_region(crop.table_region, grid))
                    .unwrap_or(crop.table_region);
                results.push(CropResult {
                    page_index: crop.page_index,
                    table_index: crop.table_index,
                    region,
                    grid,
                });
            }
            if trace {
                let completed = batch_started.elapsed();
                eprintln!(
                    "A3S_OCR_TABLE_TIMING slots={} device={} encoder_ms={:.3} decoder_ms={:.3} total_ms={:.3}",
                    results.len(),
                    encoded.receipt.runtime.device,
                    encoded_at.as_secs_f64() * 1_000.0,
                    (completed - encoded_at).as_secs_f64() * 1_000.0,
                    completed.as_secs_f64() * 1_000.0,
                );
            }
            Ok(BatchRun {
                results,
                receipt: encoded.receipt,
            })
        },
    )
    .await
}

fn validate_grid(grid: StructureGrid) -> UseResult<StructureGrid> {
    if grid.confidence < MIN_STRUCTURE_CONFIDENCE {
        return Err(runtime_error(format!(
            "SLANet-Plus structure confidence {:.3} is below the reviewed {:.3} floor.",
            grid.confidence, MIN_STRUCTURE_CONFIDENCE
        )));
    }
    Ok(grid)
}

fn table_evidence_region(wire_region: PixelRect, grid: &StructureGrid) -> PixelRect {
    let mut left = wire_region.x;
    let mut top = wire_region.y;
    let mut right = wire_region.x.saturating_add(wire_region.width);
    let mut bottom = wire_region.y.saturating_add(wire_region.height);
    for quad in grid.cells.iter().filter_map(|cell| cell.quad) {
        for point in quad.chunks_exact(2) {
            left = left.min(point[0]);
            top = top.min(point[1]);
            right = right.max(point[0]);
            bottom = bottom.max(point[1]);
        }
    }
    PixelRect {
        x: left,
        y: top,
        width: right.saturating_sub(left),
        height: bottom.saturating_sub(top),
    }
}

fn fail_crop_pages(crops: &[CropReference], pages: &mut [PageAccumulator], error: UseError) {
    let mut failed = std::collections::BTreeSet::new();
    for crop in crops {
        if failed.insert(crop.page_index) {
            pages[crop.page_index].fail(error.clone());
        }
    }
}

struct DecodedPage {
    slot_id: crate::OcrBatchSlotId,
    page: UseResult<DecodedTablePage>,
}

struct DecodedTablePage {
    image: RgbImage,
    candidates: Vec<WiredCandidate>,
}

struct PageAccumulator {
    slot_id: crate::OcrBatchSlotId,
    canvas: Option<OcrImageCanvas>,
    tables: Vec<Option<DetectedTable>>,
    receipts: Vec<OcrExecutionReceipt>,
    error: Option<UseError>,
}

impl PageAccumulator {
    fn ready(slot_id: crate::OcrBatchSlotId, canvas: OcrImageCanvas, table_count: usize) -> Self {
        Self {
            slot_id,
            canvas: Some(canvas),
            tables: (0..table_count).map(|_| None).collect(),
            receipts: Vec::new(),
            error: None,
        }
    }

    fn failed(slot_id: crate::OcrBatchSlotId, error: UseError) -> Self {
        Self {
            slot_id,
            canvas: None,
            tables: Vec::new(),
            receipts: Vec::new(),
            error: Some(error),
        }
    }

    fn fail(&mut self, error: UseError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn finish(self) -> TableSlotResult {
        let page = if let Some(error) = self.error {
            Err(error)
        } else {
            let canvas = self.canvas.ok_or_else(|| {
                runtime_error("A table slot lost its exact source canvas during execution.")
            });
            let tables = self
                .tables
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    runtime_error("A table slot left an unresolved model-backed candidate.")
                });
            canvas.and_then(|canvas| {
                tables.map(|tables| DetectedPage {
                    canvas,
                    tables,
                    receipts: self.receipts,
                })
            })
        };
        TableSlotResult {
            slot_id: self.slot_id,
            page,
        }
    }
}

#[derive(Clone)]
struct CropReference {
    page_index: usize,
    table_index: usize,
    image: Arc<RgbImage>,
    region: PixelRect,
    table_region: PixelRect,
    orientation: TableCropOrientation,
}

struct PreparedBatch {
    crops: Vec<CropReference>,
    tensor: Vec<f32>,
}

struct CropResult {
    page_index: usize,
    table_index: usize,
    region: PixelRect,
    grid: UseResult<StructureGrid>,
}

struct BatchRun {
    results: Vec<CropResult>,
    receipt: a3s_power::inference::ExecutionReceipt,
}

fn pool_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} the SLANet-Plus session pool: {error}"),
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
