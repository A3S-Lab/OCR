use std::sync::Arc;

use a3s_power::error::PowerError;
use a3s_power::inference::{
    DevicePreference, ModelSession, ModelSessionPool, ModelSessionPoolPolicy, RuntimeDeviceKind,
};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use super::assets::SealTextAssets;
use super::batching::execute_views_batched;
use super::decoder::{
    clockwise180_observations, clockwise270_observations, direct_observations,
    orthogonal_observations, SealTextObservation,
};
use super::native::{session_spec, NativeSealText, MAX_CONCURRENCY};
use super::profile::{detection_config, RESIZE_LONG, RESIZE_STRIDE};
use crate::cancellation::{check_cancelled, run_blocking_with};
use crate::postprocess::detection_boxes_in_content;
use crate::preprocess::detection_input_with_resize_long;
use crate::receipt::project_receipt;
use crate::OcrExecutionReceipt;

#[derive(Clone)]
pub(in crate::document_fast::seal) struct SealTextStageRunner {
    assets: SealTextAssets,
    sessions: Vec<ModelSessionPool<SealTextSession>>,
}

const CUDA_DEVICE_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CUDA_SESSION_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

impl SealTextStageRunner {
    pub(in crate::document_fast::seal) fn from_env_optional() -> UseResult<Option<Self>> {
        SealTextAssets::from_env_optional()?
            .map(Self::new)
            .transpose()
    }

    fn new(assets: SealTextAssets) -> UseResult<Self> {
        let policy = ModelSessionPoolPolicy::new(1, 128 * 1024 * 1024, MAX_CONCURRENCY, 64)
            .map_err(|error| pool_error("configure", error))?;
        let first = ModelSessionPool::new(DevicePreference::Auto, policy.clone())
            .map_err(|error| pool_error("initialize", error))?;
        let identity = first.snapshot().device;
        let mut sessions = Vec::with_capacity(if identity.kind == RuntimeDeviceKind::Cuda {
            MAX_CONCURRENCY
        } else {
            1
        });
        sessions.push(first);
        if let (RuntimeDeviceKind::Cuda, Some(ordinal)) = (identity.kind, identity.ordinal) {
            for _ in 1..MAX_CONCURRENCY {
                sessions.push(
                    ModelSessionPool::new(DevicePreference::Cuda { ordinal }, policy.clone())
                        .map_err(|error| pool_error("initialize replica", error))?,
                );
            }
        }
        Ok(Self { assets, sessions })
    }

    /// Loads the bounded model-session lanes without starting page inference.
    /// The seal stage uses this to overlap immutable model loading with the
    /// prerequisite layout pass while retaining layout-driven view admission.
    pub(in crate::document_fast::seal) async fn prepare(
        &self,
        cancellation: CancellationToken,
    ) -> UseResult<PreparedSealTextStage> {
        Ok(PreparedSealTextStage {
            sessions: prepare_sessions(self, cancellation).await?,
        })
    }

    fn active_session_count(&self) -> UseResult<usize> {
        #[cfg(test)]
        if let Some(count) = std::env::var_os("A3S_OCR_TEST_SEAL_TEXT_SESSION_COUNT")
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
            .filter(|count| (1..=self.sessions.len()).contains(count))
        {
            return Ok(count);
        }

        let memory = self.sessions[0]
            .memory_snapshot()
            .map_err(|error| pool_error("discover replica memory", error))?;
        let available = memory.device.as_ref().map(|device| device.available_bytes);
        Ok(session_count_for(
            self.sessions[0].snapshot().device.kind,
            available,
            self.sessions.len(),
        ))
    }
}

pub(in crate::document_fast::seal) struct PreparedSealTextStage {
    sessions: Vec<ModelSession<SealTextSession>>,
}

impl PreparedSealTextStage {
    pub(in crate::document_fast::seal) async fn run(
        self,
        pages: Vec<SealTextPageReference>,
        cancellation: CancellationToken,
    ) -> UseResult<SealTextStageBatch> {
        if pages.is_empty() {
            return Ok(SealTextStageBatch {
                pages: Vec::new(),
                receipts: Vec::new(),
            });
        }
        run_views(self.sessions, pages, cancellation).await
    }
}

#[derive(Clone)]
pub(in crate::document_fast::seal) struct SealTextPageReference {
    pub(in crate::document_fast::seal) page_index: usize,
    pub(in crate::document_fast::seal) image: Arc<RgbImage>,
}

pub(in crate::document_fast::seal) struct SealTextStageBatch {
    pub(in crate::document_fast::seal) pages: Vec<SealTextPageResult>,
    pub(in crate::document_fast::seal) receipts: Vec<OcrExecutionReceipt>,
}

pub(in crate::document_fast::seal) struct SealTextPageResult {
    pub(in crate::document_fast::seal) page_index: usize,
    pub(in crate::document_fast::seal) evidence: UseResult<SealTextPageEvidence>,
}

pub(in crate::document_fast::seal) struct SealTextPageEvidence {
    pub(in crate::document_fast::seal) direct: Vec<SealTextObservation>,
    pub(in crate::document_fast::seal) clockwise90: Vec<SealTextObservation>,
    pub(in crate::document_fast::seal) clockwise180: Vec<SealTextObservation>,
    pub(in crate::document_fast::seal) clockwise270: Vec<SealTextObservation>,
    pub(in crate::document_fast::seal) receipts: Vec<OcrExecutionReceipt>,
}

pub(super) struct SealTextSession {
    pub(super) engine: NativeSealText,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ViewOrientation {
    Direct,
    Clockwise90,
    Clockwise180,
    Clockwise270,
}

#[derive(Clone)]
pub(super) struct ViewReference {
    pub(super) page_index: usize,
    pub(super) image: Arc<RgbImage>,
    pub(super) orientation: ViewOrientation,
}

impl ViewReference {
    pub(super) fn oriented_dimensions(&self) -> (u32, u32) {
        match self.orientation {
            ViewOrientation::Direct | ViewOrientation::Clockwise180 => {
                (self.image.width(), self.image.height())
            }
            ViewOrientation::Clockwise90 | ViewOrientation::Clockwise270 => {
                (self.image.height(), self.image.width())
            }
        }
    }
}

pub(super) struct ViewEvidence {
    pub(super) observations: Vec<SealTextObservation>,
    pub(super) receipt: OcrExecutionReceipt,
}

pub(super) struct ViewResult {
    pub(super) page_index: usize,
    pub(super) orientation: ViewOrientation,
    pub(super) evidence: UseResult<ViewEvidence>,
}

impl ViewResult {
    pub(super) fn new(reference: ViewReference, evidence: UseResult<ViewEvidence>) -> Self {
        Self {
            page_index: reference.page_index,
            orientation: reference.orientation,
            evidence,
        }
    }

    pub(super) fn failed(reference: ViewReference, error: UseError) -> Self {
        Self::new(reference, Err(error))
    }
}

async fn prepare_sessions(
    runner: &SealTextStageRunner,
    cancellation: CancellationToken,
) -> UseResult<Vec<ModelSession<SealTextSession>>> {
    let assets = runner.assets.clone();
    let spec_assets = assets.clone();
    let spec = run_blocking_with(
        "seal-text session declaration",
        cancellation.clone(),
        move |cancellation| {
            check_cancelled(&cancellation)?;
            session_spec(&spec_assets)
        },
    )
    .await?;
    let session_count = runner.active_session_count()?;
    let mut sessions = Vec::with_capacity(session_count);
    for pool in runner.sessions.iter().take(session_count) {
        let assets = assets.clone();
        let session = pool
            .get_or_load(
                spec.clone(),
                &cancellation,
                move |runtime, loader_cancellation| {
                    load_session(assets, runtime, loader_cancellation)
                },
            )
            .await
            .map_err(|error| pool_error("load", error))?;
        sessions.push(session);
    }
    Ok(sessions)
}

async fn load_session(
    assets: SealTextAssets,
    runtime: a3s_power::inference::EmbeddedRuntime,
    cancellation: CancellationToken,
) -> a3s_power::error::Result<SealTextSession> {
    tokio::task::spawn_blocking(move || {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceCancelled);
        }
        let engine = NativeSealText::load_with_runtime(&assets, runtime)
            .map_err(|error| PowerError::InferenceFailed(error.message))?;
        Ok(SealTextSession { engine })
    })
    .await
    .map_err(|error| {
        PowerError::InferenceFailed(format!("Seal-text loader task failed: {error}"))
    })?
}

async fn run_views(
    sessions: Vec<ModelSession<SealTextSession>>,
    pages: Vec<SealTextPageReference>,
    cancellation: CancellationToken,
) -> UseResult<SealTextStageBatch> {
    run_blocking_with(
        "parallel seal-text views",
        cancellation,
        move |cancellation| {
            let started = std::time::Instant::now();
            let mut views = Vec::with_capacity(pages.len().saturating_mul(4));
            for page in pages {
                for orientation in [
                    ViewOrientation::Direct,
                    ViewOrientation::Clockwise90,
                    ViewOrientation::Clockwise180,
                    ViewOrientation::Clockwise270,
                ] {
                    views.push(ViewReference {
                        page_index: page.page_index,
                        image: Arc::clone(&page.image),
                        orientation,
                    });
                }
            }
            let worker_count = sessions.len().min(views.len()).max(1);
            let mut results = if seal_text_batching_enabled() {
                execute_views_batched(&sessions, &views, &cancellation)?
            } else {
                execute_scalar_views(&sessions[0], &views, &cancellation, MAX_CONCURRENCY)
            };
            if results.iter().any(|result| result.page_index == usize::MAX) {
                return Err(runtime_error(
                    "A seal-text preprocessing worker terminated unexpectedly.",
                ));
            }
            results.sort_by_key(|result| (result.page_index, result.orientation));
            let mut page_results = Vec::with_capacity(results.len() / 4);
            let mut receipts = Vec::with_capacity(results.len());
            for group in results.chunks_exact_mut(4) {
                let page_index = group[0].page_index;
                if group.iter().any(|result| result.page_index != page_index)
                    || group[0].orientation != ViewOrientation::Direct
                    || group[1].orientation != ViewOrientation::Clockwise90
                    || group[2].orientation != ViewOrientation::Clockwise180
                    || group[3].orientation != ViewOrientation::Clockwise270
                {
                    return Err(runtime_error(
                        "Seal-text view execution changed page or orientation cardinality.",
                    ));
                }
                let mut take = |index: usize, label: &str| {
                    std::mem::replace(
                        &mut group[index].evidence,
                        Err(runtime_error(format!(
                            "Seal-text {label} evidence was already consumed."
                        ))),
                    )
                };
                let evidence = match (
                    take(0, "direct"),
                    take(1, "clockwise-90"),
                    take(2, "clockwise-180"),
                    take(3, "clockwise-270"),
                ) {
                    (Ok(direct), Ok(clockwise90), Ok(clockwise180), Ok(clockwise270)) => {
                        receipts.extend([
                            direct.receipt.clone(),
                            clockwise90.receipt.clone(),
                            clockwise180.receipt.clone(),
                            clockwise270.receipt.clone(),
                        ]);
                        Ok(SealTextPageEvidence {
                            direct: direct.observations,
                            clockwise90: clockwise90.observations,
                            clockwise180: clockwise180.observations,
                            clockwise270: clockwise270.observations,
                            receipts: vec![
                                direct.receipt,
                                clockwise90.receipt,
                                clockwise180.receipt,
                                clockwise270.receipt,
                            ],
                        })
                    }
                    (Err(error), _, _, _)
                    | (_, Err(error), _, _)
                    | (_, _, Err(error), _)
                    | (_, _, _, Err(error)) => Err(error),
                };
                page_results.push(SealTextPageResult {
                    page_index,
                    evidence,
                });
            }
            if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
                eprintln!(
                    "A3S_OCR_SEAL_TEXT_TIMING pages={} views={} workers={} total_ms={:.3}",
                    page_results.len(),
                    results.len(),
                    worker_count,
                    started.elapsed().as_secs_f64() * 1_000.0,
                );
            }
            Ok(SealTextStageBatch {
                pages: page_results,
                receipts,
            })
        },
    )
    .await
}

fn session_count_for(
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
        .checked_div(CUDA_SESSION_BUDGET_BYTES)
        .unwrap_or_default();
    usize::try_from(funded)
        .unwrap_or(maximum_sessions)
        .clamp(1, maximum_sessions.max(1))
}

fn execute_scalar_views(
    session: &ModelSession<SealTextSession>,
    views: &[ViewReference],
    cancellation: &CancellationToken,
    worker_count: usize,
) -> Vec<ViewResult> {
    std::thread::scope(|scope| {
        let workers = (0..worker_count)
            .map(|worker| {
                scope.spawn(move || {
                    (worker..views.len())
                        .step_by(worker_count)
                        .map(|index| execute_scalar_view(session, &views[index], cancellation))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .flat_map(|worker| match worker.join() {
                Ok(results) => results,
                Err(_) => vec![ViewResult {
                    page_index: usize::MAX,
                    orientation: ViewOrientation::Direct,
                    evidence: Err(runtime_error(
                        "A seal-text preprocessing worker terminated unexpectedly.",
                    )),
                }],
            })
            .collect::<Vec<_>>()
    })
}

pub(super) fn execute_scalar_view(
    session: &ModelSession<SealTextSession>,
    reference: &ViewReference,
    cancellation: &CancellationToken,
) -> ViewResult {
    let evidence = (|| {
        check_cancelled(cancellation)?;
        let rotated;
        let image = match reference.orientation {
            ViewOrientation::Direct => reference.image.as_ref(),
            ViewOrientation::Clockwise90 => {
                rotated = image::imageops::rotate90(reference.image.as_ref());
                &rotated
            }
            ViewOrientation::Clockwise180 => {
                rotated = image::imageops::rotate180(reference.image.as_ref());
                &rotated
            }
            ViewOrientation::Clockwise270 => {
                rotated = image::imageops::rotate270(reference.image.as_ref());
                &rotated
            }
        };
        let input = detection_input_with_resize_long(
            image,
            &detection_config(),
            RESIZE_LONG,
            RESIZE_STRIDE,
        )?;
        let permit = session
            .runtime()
            .begin(cancellation)
            .map_err(|error| power_error("admit a seal-text view", error))?;
        let output = session.value().engine.infer_batch(
            input.shape,
            input.data,
            session.runtime(),
            &permit,
            cancellation,
        )?;
        let detections = detection_boxes_in_content(
            &output.tensor.values,
            &output.tensor.shape,
            input.geometry.content_width,
            input.geometry.content_height,
            input.geometry.original_width,
            input.geometry.original_height,
            &detection_config(),
        )?;
        let observations = match reference.orientation {
            ViewOrientation::Direct => direct_observations(
                &detections,
                reference.image.width(),
                reference.image.height(),
            ),
            ViewOrientation::Clockwise90 => orthogonal_observations(
                &detections,
                reference.image.width(),
                reference.image.height(),
            ),
            ViewOrientation::Clockwise180 => clockwise180_observations(
                &detections,
                reference.image.width(),
                reference.image.height(),
            ),
            ViewOrientation::Clockwise270 => clockwise270_observations(
                &detections,
                reference.image.width(),
                reference.image.height(),
            ),
        };
        Ok(ViewEvidence {
            observations,
            receipt: project_receipt(output.receipt),
        })
    })();
    ViewResult {
        page_index: reference.page_index,
        orientation: reference.orientation,
        evidence,
    }
}

fn seal_text_batching_enabled() -> bool {
    #[cfg(test)]
    if std::env::var_os("A3S_OCR_TEST_DISABLE_SEAL_TEXT_BATCHING").is_some() {
        return false;
    }
    true
}

fn pool_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} the seal-text session pool: {error}"),
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
mod tests {
    use super::*;

    #[test]
    fn seal_text_session_count_is_device_memory_bounded() {
        assert_eq!(session_count_for(RuntimeDeviceKind::Cpu, None, 4), 1);
        assert_eq!(session_count_for(RuntimeDeviceKind::Cuda, None, 4), 1);
        assert_eq!(
            session_count_for(RuntimeDeviceKind::Cuda, Some(6 << 30), 4),
            1
        );
        assert_eq!(
            session_count_for(RuntimeDeviceKind::Cuda, Some(10 << 30), 4),
            2
        );
        assert_eq!(
            session_count_for(RuntimeDeviceKind::Cuda, Some(18 << 30), 4),
            4
        );
        assert_eq!(
            session_count_for(RuntimeDeviceKind::Cuda, Some(64 << 30), 3),
            3
        );
    }
}
