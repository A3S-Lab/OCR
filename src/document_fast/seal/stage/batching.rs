use std::cmp::Reverse;
use std::collections::BTreeSet;

use a3s_power::inference::ModelSession;
use a3s_use_core::UseResult;
use rayon::prelude::*;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use super::state::{BatchRun, PageAccumulator, PreparedBatch, ViewReference, ViewResult};
use super::{power_error, runtime_error, trace_phase_timing, SealSession};
use crate::cancellation::run_blocking_with;
use crate::document_fast::seal::native::MAX_BATCH_SIZE;
use crate::document_fast::seal::preprocess::view_tensor_into;
use crate::document_fast::seal::profile::PicodetLayoutProfile;
use crate::receipt::project_receipt;
use crate::OcrExecutionReceipt;

#[derive(Clone)]
struct ViewBatch {
    index: usize,
    views: Vec<ViewReference>,
}

struct CompletedViewBatch {
    index: usize,
    views: Vec<ViewReference>,
    outcome: UseResult<BatchRun>,
}

pub(super) async fn run_view_batches(
    sessions: &[ModelSession<SealSession>],
    profile: PicodetLayoutProfile,
    views: Vec<ViewReference>,
    pages: &mut [PageAccumulator],
    receipts: &mut Vec<OcrExecutionReceipt>,
    cancellation: CancellationToken,
    collect_refinement_views: bool,
) -> UseResult<Vec<ViewReference>> {
    if views.is_empty() {
        return Ok(Vec::new());
    }
    let batches = view_batches(views);
    let completed = execute_view_batches(
        sessions,
        profile,
        &batches,
        cancellation,
        collect_refinement_views,
    )
    .await?;

    let mut refinement_views = Vec::new();
    for completed in completed {
        match completed.outcome {
            Ok(run) => apply_batch_run(run, pages, receipts, &mut refinement_views),
            Err(error) => fail_view_pages(&completed.views, pages, error),
        }
    }
    Ok(refinement_views)
}

fn view_batches(views: Vec<ViewReference>) -> Vec<ViewBatch> {
    views
        .chunks(MAX_BATCH_SIZE)
        .enumerate()
        .map(|(index, views)| ViewBatch {
            index,
            views: views.to_vec(),
        })
        .collect()
}

async fn execute_view_batches(
    sessions: &[ModelSession<SealSession>],
    profile: PicodetLayoutProfile,
    batches: &[ViewBatch],
    cancellation: CancellationToken,
    collect_refinement_views: bool,
) -> UseResult<Vec<CompletedViewBatch>> {
    if sessions.is_empty() {
        return Err(runtime_error(
            "PicoDet layout batch execution requires at least one prepared session.",
        ));
    }
    let worker_count = sessions.len().min(batches.len()).max(1);
    let assignments = least_work_assignments(batches, worker_count);
    trace_batch_plan(batches, &assignments, collect_refinement_views);

    let mut workers = JoinSet::new();
    for (worker_index, assignment) in assignments.into_iter().enumerate() {
        let assigned = assignment
            .into_iter()
            .map(|batch_index| batches[batch_index].clone())
            .collect::<Vec<_>>();
        let session = sessions[worker_index].clone();
        let cancellation = cancellation.clone();
        workers.spawn(async move {
            run_batch_assignment(
                session,
                profile,
                assigned,
                cancellation,
                collect_refinement_views,
            )
            .await
        });
    }

    let mut completed = Vec::with_capacity(batches.len());
    while let Some(worker) = workers.join_next().await {
        completed.extend(worker.map_err(|error| {
            runtime_error(format!(
                "A PicoDet layout batch worker terminated unexpectedly: {error}"
            ))
        })?);
    }
    completed.sort_by_key(|batch| batch.index);
    Ok(completed)
}

async fn run_batch_assignment(
    session: ModelSession<SealSession>,
    profile: PicodetLayoutProfile,
    batches: Vec<ViewBatch>,
    cancellation: CancellationToken,
    collect_refinement_views: bool,
) -> Vec<CompletedViewBatch> {
    let mut batches = batches.into_iter();
    let Some(mut current) = batches.next() else {
        return Vec::new();
    };
    let mut current_prepared =
        prepare_views(current.views.clone(), profile, cancellation.clone()).await;
    let mut completed = Vec::new();

    loop {
        let next = batches.next();
        let (outcome, next_prepared) = match current_prepared {
            Ok(prepared) => match next.as_ref() {
                Some(next) => {
                    let execute = execute_admitted_batch(
                        session.clone(),
                        prepared,
                        cancellation.clone(),
                        collect_refinement_views,
                    );
                    let prepare = prepare_views(next.views.clone(), profile, cancellation.clone());
                    let (execution, prepared) = tokio::join!(execute, prepare);
                    (execution, Some(prepared))
                }
                None => (
                    execute_admitted_batch(
                        session.clone(),
                        prepared,
                        cancellation.clone(),
                        collect_refinement_views,
                    )
                    .await,
                    None,
                ),
            },
            Err(error) => {
                let prepared = match next.as_ref() {
                    Some(next) => {
                        Some(prepare_views(next.views.clone(), profile, cancellation.clone()).await)
                    }
                    None => None,
                };
                (Err(error), prepared)
            }
        };
        completed.push(CompletedViewBatch {
            index: current.index,
            views: current.views,
            outcome,
        });

        let Some(next) = next else {
            break;
        };
        current = next;
        current_prepared = next_prepared.unwrap_or_else(|| {
            Err(runtime_error(
                "A queued PicoDet layout batch lost its preparation state.",
            ))
        });
    }
    completed
}

fn apply_batch_run(
    run: BatchRun,
    pages: &mut [PageAccumulator],
    receipts: &mut Vec<OcrExecutionReceipt>,
    refinement_views: &mut Vec<ViewReference>,
) {
    let receipt = project_receipt(run.receipt);
    receipts.push(receipt.clone());
    let mut receipt_pages = BTreeSet::new();
    for result in run.results {
        let ViewResult {
            page_index,
            seals,
            layout_images,
            refinement_views: result_refinement_views,
        } = result;
        refinement_views.extend(result_refinement_views);
        let page = &mut pages[page_index];
        if receipt_pages.insert(page_index) {
            page.receipts.push(receipt.clone());
        }
        match seals {
            Ok(seals) => page.add_seals(seals),
            Err(error) => page.fail(error),
        }
        match layout_images {
            Ok(images) => page.add_layout_images(images),
            Err(error) => page.fail(error),
        }
    }
}

async fn execute_admitted_batch(
    session: ModelSession<SealSession>,
    prepared: PreparedBatch,
    cancellation: CancellationToken,
    collect_refinement_views: bool,
) -> UseResult<BatchRun> {
    let permit = session
        .runtime()
        .begin_wait(&cancellation)
        .await
        .map_err(|error| power_error("admit a PicoDet layout batch", error))?;
    execute_batch(
        session,
        prepared,
        permit,
        cancellation,
        collect_refinement_views,
    )
    .await
}

async fn prepare_views(
    views: Vec<ViewReference>,
    profile: PicodetLayoutProfile,
    cancellation: CancellationToken,
) -> UseResult<PreparedBatch> {
    run_blocking_with(
        "PicoDet layout view preprocessing",
        cancellation,
        move |cancellation| {
            let started = std::time::Instant::now();
            let view_count = views.len();
            let view_elements = profile.tensor_elements_per_view();
            let tensor_elements = views.len().checked_mul(view_elements).ok_or_else(|| {
                runtime_error("PicoDet view tensor cardinality overflowed the host address space.")
            })?;
            let mut tensor = vec![0.0_f32; tensor_elements];
            tensor
                .par_chunks_mut(view_elements)
                .zip(views.par_iter())
                .try_for_each(|(tensor, view)| {
                    crate::cancellation::check_cancelled(&cancellation)?;
                    view_tensor_into(&view.image, view.view, profile, tensor)
                })?;
            trace_phase_timing("preprocess", view_count, started.elapsed());
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
    collect_refinement_views: bool,
) -> UseResult<BatchRun> {
    run_blocking_with(
        "PicoDet admitted layout batch",
        cancellation,
        move |cancellation| {
            let total_started = std::time::Instant::now();
            let engine = session
                .value()
                .engine
                .lock()
                .map_err(|_| runtime_error("The PicoDet layout engine lock is poisoned."))?;
            let inference_started = std::time::Instant::now();
            let inferred = engine.infer_batch(
                prepared.tensor,
                prepared.views.len(),
                &permit,
                &cancellation,
            )?;
            let inference_elapsed = inference_started.elapsed();
            let profile = inferred.profile;
            let decode_started = std::time::Instant::now();
            let results = super::decode::decode_inferred_views(
                prepared.views,
                &inferred.tensor.values,
                profile,
                collect_refinement_views,
                &cancellation,
            )?;
            let decode_elapsed = decode_started.elapsed();
            if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
                eprintln!(
                    "A3S_OCR_SEAL_TIMING phase=execute views={} device={} inference_ms={:.3} decode_ms={:.3} total_ms={:.3}",
                    results.len(),
                    inferred.receipt.runtime.device,
                    inference_elapsed.as_secs_f64() * 1_000.0,
                    decode_elapsed.as_secs_f64() * 1_000.0,
                    total_started.elapsed().as_secs_f64() * 1_000.0,
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

fn least_work_assignments(batches: &[ViewBatch], worker_count: usize) -> Vec<Vec<usize>> {
    if worker_count == 0 {
        return Vec::new();
    }
    let mut order = (0..batches.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| (Reverse(batches[index].views.len()), index));
    let mut loads = vec![0_usize; worker_count];
    let mut assignments = vec![Vec::new(); worker_count];
    for batch_index in order {
        let worker_index = loads
            .iter()
            .enumerate()
            .min_by_key(|(worker_index, load)| (**load, *worker_index))
            .map(|(worker_index, _)| worker_index)
            .unwrap_or(0);
        loads[worker_index] = loads[worker_index].saturating_add(batches[batch_index].views.len());
        assignments[worker_index].push(batch_index);
    }
    for assignment in &mut assignments {
        assignment.sort_unstable();
    }
    assignments
}

fn trace_batch_plan(
    batches: &[ViewBatch],
    assignments: &[Vec<usize>],
    collect_refinement_views: bool,
) {
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_none() {
        return;
    }
    let batch_sizes = batches
        .iter()
        .map(|batch| batch.views.len())
        .collect::<Vec<_>>();
    let worker_views = assignments
        .iter()
        .map(|assignment| {
            assignment.iter().fold(0_usize, |total, &index| {
                total.saturating_add(batches[index].views.len())
            })
        })
        .collect::<Vec<_>>();
    eprintln!(
        "A3S_OCR_SEAL_BATCH_PLAN collect_refinement_views={collect_refinement_views} batches={} workers={} batch_sizes={batch_sizes:?} worker_views={worker_views:?}",
        batches.len(),
        assignments.len(),
    );
}

fn fail_view_pages(
    views: &[ViewReference],
    pages: &mut [PageAccumulator],
    error: a3s_use_core::UseError,
) {
    let mut failed = BTreeSet::new();
    for view in views {
        if failed.insert(view.page_index) {
            pages[view.page_index].fail(error.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use image::RgbImage;

    use super::*;
    use crate::document_fast::seal::preprocess::SealView;
    use crate::document_fast::wired::PixelRect;

    fn batch(index: usize, views: usize) -> ViewBatch {
        let image = Arc::new(RgbImage::new(1, 1));
        ViewBatch {
            index,
            views: (0..views)
                .map(|view| ViewReference {
                    page_index: view,
                    image: image.clone(),
                    view: SealView {
                        region: PixelRect {
                            x: 0,
                            y: 0,
                            width: 1,
                            height: 1,
                        },
                    },
                    adjacent_boundary: None,
                })
                .collect(),
        }
    }

    #[test]
    fn whole_batches_are_balanced_without_changing_batch_geometry() {
        let batches = vec![
            batch(0, 32),
            batch(1, 32),
            batch(2, 32),
            batch(3, 32),
            batch(4, 17),
        ];
        let assignments = least_work_assignments(&batches, 3);
        assert_eq!(assignments, [vec![0, 3], vec![1, 4], vec![2]]);
        let mut assigned = assignments.into_iter().flatten().collect::<Vec<_>>();
        assigned.sort_unstable();
        assert_eq!(assigned, [0, 1, 2, 3, 4]);
        assert_eq!(
            batches
                .iter()
                .map(|batch| batch.views.len())
                .collect::<Vec<_>>(),
            [32, 32, 32, 32, 17]
        );
    }

    #[test]
    fn zero_workers_admit_no_assignment() {
        assert!(least_work_assignments(&[batch(0, 1)], 0).is_empty());
    }
}
