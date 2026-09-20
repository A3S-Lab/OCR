use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::ops::Range;

use a3s_power::inference::{InferenceLimits, ModelSession};
use a3s_use_core::{UseError, UseResult};
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use super::decoder::{
    clockwise180_observations, clockwise270_observations, direct_observations,
    orthogonal_observations,
};
use super::profile::{detection_config, RESIZE_LONG, RESIZE_STRIDE};
use super::stage::{
    execute_scalar_view, SealTextSession, ViewEvidence, ViewOrientation, ViewReference, ViewResult,
};
use crate::batch::MAX_BATCH_SLOTS;
use crate::cancellation::check_cancelled;
use crate::postprocess::detection_boxes_in_content;
use crate::preprocess::{
    detection_input_with_resize_long, detection_resize_long_dimensions, DetectionInput,
};
use crate::receipt::project_receipt;

struct PreparedView {
    reference: ViewReference,
    input: DetectionInput,
}

struct PreparedReferenceBatch {
    admitted: Vec<PreparedView>,
    results: Vec<ViewResult>,
}

struct ViewBatch {
    references: Vec<ViewReference>,
    estimated_work: u64,
}

pub(super) fn execute_views_batched(
    sessions: &[ModelSession<SealTextSession>],
    views: &[ViewReference],
    cancellation: &CancellationToken,
) -> UseResult<Vec<ViewResult>> {
    if sessions.is_empty() {
        return Err(runtime_error(
            "Seal-text batch execution requires at least one admitted session.",
        ));
    }
    let groups = exact_shape_groups(views)?;
    let group_count = groups.len();
    let legacy_plan = legacy_batch_plan_enabled();
    let balance_parallel_work = !legacy_plan && sessions.len() > 1;
    let mut batches = Vec::<ViewBatch>::new();
    let mut maximum_batch = 0_usize;

    for ((height, width), references) in groups {
        let batch_limit = maximum_batch_size(sessions[0].runtime().limits(), height, width)?;
        let ranges = if balance_parallel_work {
            balanced_batch_ranges(references.len(), batch_limit)
        } else {
            maximal_batch_ranges(references.len(), batch_limit)
        };
        for range in ranges {
            let batch_size = range.len();
            maximum_batch = maximum_batch.max(batch_size);
            let estimated_work = estimated_batch_work(batch_size, height, width)?;
            batches.push(ViewBatch {
                references: references[range].to_vec(),
                estimated_work,
            });
        }
    }

    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
        eprintln!(
            "A3S_OCR_SEAL_TEXT_BATCH_PLAN views={} exact_shapes={} batches={} maximum_batch={} balanced_parallel_work={}",
            views.len(),
            group_count,
            batches.len(),
            maximum_batch,
            balance_parallel_work,
        );
    }
    let worker_count = sessions.len().min(batches.len()).max(1);
    let assignments = if worker_count == 1 {
        vec![(0..batches.len()).collect()]
    } else if legacy_plan {
        round_robin_assignments(batches.len(), worker_count)
    } else {
        least_work_assignments(&batches, worker_count)
    };
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
        let worker_loads = assignment_loads(&batches, &assignments);
        let batch_sizes = batches
            .iter()
            .map(|batch| batch.references.len())
            .collect::<Vec<_>>();
        eprintln!(
            "A3S_OCR_SEAL_TEXT_LANE_PLAN workers={} preparation_window_depth=2 batch_sizes={batch_sizes:?} worker_tensor_pixels={worker_loads:?}",
            assignments.len(),
        );
    }

    let mut completed = std::thread::scope(|scope| -> UseResult<Vec<_>> {
        let batches = &batches;
        let assignments = &assignments;
        let workers = sessions
            .iter()
            .take(worker_count)
            .enumerate()
            .map(|(worker_index, session)| {
                scope.spawn(move || {
                    execute_batch_assignment(
                        session,
                        batches,
                        &assignments[worker_index],
                        cancellation,
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut completed = Vec::with_capacity(batches.len());
        for worker in workers {
            match worker.join() {
                Ok(Ok(results)) => completed.extend(results),
                Ok(Err(error)) => return Err(error),
                Err(_) => {
                    return Err(runtime_error(
                        "A seal-text batch execution worker terminated unexpectedly.",
                    ));
                }
            }
        }
        Ok(completed)
    })?;
    completed.sort_by_key(|(batch_index, _)| *batch_index);
    Ok(completed
        .into_iter()
        .flat_map(|(_, results)| results)
        .collect())
}

fn maximal_batch_ranges(item_count: usize, batch_limit: usize) -> Vec<Range<usize>> {
    (0..item_count)
        .step_by(batch_limit)
        .map(|start| start..(start + batch_limit).min(item_count))
        .collect()
}

/// Splits a cohort into the minimum number of admitted batches while keeping
/// their sizes within one item of each other. Avoiding a tiny tail batch makes
/// independent accelerator lanes carry comparable amounts of tensor work
/// without changing source order, shapes, or the resource ceiling.
fn balanced_batch_ranges(item_count: usize, batch_limit: usize) -> Vec<Range<usize>> {
    debug_assert!(batch_limit > 0);
    if item_count == 0 {
        return Vec::new();
    }
    let batch_count = item_count.div_ceil(batch_limit);
    let base_size = item_count / batch_count;
    let larger_batches = item_count % batch_count;
    let mut start = 0_usize;
    (0..batch_count)
        .map(|batch_index| {
            let size = base_size + usize::from(batch_index < larger_batches);
            let range = start..start + size;
            start = range.end;
            range
        })
        .collect()
}

fn estimated_batch_work(batch_size: usize, height: usize, width: usize) -> UseResult<u64> {
    let work = batch_size
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| runtime_error("Seal-text batch work estimate overflowed."))?;
    u64::try_from(work)
        .map_err(|_| runtime_error("Seal-text batch work estimate cannot be represented."))
}

fn round_robin_assignments(batch_count: usize, worker_count: usize) -> Vec<Vec<usize>> {
    let mut assignments = vec![Vec::new(); worker_count];
    for batch_index in 0..batch_count {
        assignments[batch_index % worker_count].push(batch_index);
    }
    assignments
}

/// Longest-processing-time scheduling using tensor pixels as the work unit.
/// Convolutional cost is proportional to N*H*W for a fixed admitted graph, so
/// this remains model-shape driven rather than document-content driven.
fn least_work_assignments(batches: &[ViewBatch], worker_count: usize) -> Vec<Vec<usize>> {
    if worker_count == 0 {
        return Vec::new();
    }
    let mut order = (0..batches.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| (Reverse(batches[index].estimated_work), index));
    let mut loads = vec![0_u64; worker_count];
    let mut assignments = vec![Vec::new(); worker_count];
    for batch_index in order {
        let worker_index = loads
            .iter()
            .enumerate()
            .min_by_key(|(worker_index, load)| (**load, *worker_index))
            .map(|(worker_index, _)| worker_index)
            .unwrap_or(0);
        loads[worker_index] =
            loads[worker_index].saturating_add(batches[batch_index].estimated_work);
        assignments[worker_index].push(batch_index);
    }
    for assignment in &mut assignments {
        assignment.sort_unstable();
    }
    assignments
}

fn assignment_loads(batches: &[ViewBatch], assignments: &[Vec<usize>]) -> Vec<u64> {
    assignments
        .iter()
        .map(|assignment| {
            assignment.iter().fold(0_u64, |load, &batch_index| {
                load.saturating_add(batches[batch_index].estimated_work)
            })
        })
        .collect()
}

fn legacy_batch_plan_enabled() -> bool {
    #[cfg(test)]
    if std::env::var_os("A3S_OCR_TEST_LEGACY_SEAL_TEXT_BATCH_PLAN").is_some() {
        return true;
    }
    false
}

fn execute_batch_assignment(
    session: &ModelSession<SealTextSession>,
    batches: &[ViewBatch],
    assignment: &[usize],
    cancellation: &CancellationToken,
) -> UseResult<Vec<(usize, Vec<ViewResult>)>> {
    let Some(&first_batch_index) = assignment.first() else {
        return Ok(Vec::new());
    };
    let mut prepared = Some(prepare_reference_batch(
        &batches[first_batch_index].references,
        cancellation,
    )?);
    let mut completed = Vec::with_capacity(assignment.len());
    for (position, &batch_index) in assignment.iter().enumerate() {
        let current = prepared.take().ok_or_else(|| {
            runtime_error("A seal-text preparation window lost its current batch.")
        })?;
        let next_batch_index = assignment.get(position + 1).copied();
        let (results, next_prepared) = std::thread::scope(|scope| -> UseResult<_> {
            let prefetch = next_batch_index.map(|next_batch_index| {
                scope.spawn(move || {
                    prepare_reference_batch(&batches[next_batch_index].references, cancellation)
                })
            });
            let results = execute_prepared_reference_batch(session, current, cancellation);
            let next_prepared = match prefetch {
                Some(worker) => Some(worker.join().map_err(|_| {
                    runtime_error("A seal-text preparation worker terminated unexpectedly.")
                })??),
                None => None,
            };
            Ok((results, next_prepared))
        })?;
        completed.push((batch_index, results));
        prepared = next_prepared;
    }
    Ok(completed)
}

fn prepare_reference_batch(
    references: &[ViewReference],
    cancellation: &CancellationToken,
) -> UseResult<PreparedReferenceBatch> {
    check_cancelled(cancellation)?;
    let preparation_started = std::time::Instant::now();
    let prepared = references
        .par_iter()
        .map(|reference| {
            let reference = (*reference).clone();
            let input = prepare_input(&reference, cancellation);
            (reference, input)
        })
        .collect::<Vec<_>>();
    if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
        eprintln!(
            "A3S_OCR_SEAL_TEXT_PREPARE_TIMING views={} total_ms={:.3}",
            references.len(),
            preparation_started.elapsed().as_secs_f64() * 1_000.0,
        );
    }
    let mut admitted = Vec::with_capacity(prepared.len());
    let mut results = Vec::with_capacity(references.len());
    for (reference, input) in prepared {
        match input {
            Ok(input) => admitted.push(PreparedView { reference, input }),
            Err(error) => results.push(ViewResult::failed(reference, error)),
        }
    }
    Ok(PreparedReferenceBatch { admitted, results })
}

fn execute_prepared_reference_batch(
    session: &ModelSession<SealTextSession>,
    mut prepared: PreparedReferenceBatch,
    cancellation: &CancellationToken,
) -> Vec<ViewResult> {
    if prepared.admitted.is_empty() {
        return prepared.results;
    }
    match execute_prepared_batch(session, prepared.admitted, cancellation) {
        Ok(batch_results) => prepared.results.extend(batch_results),
        Err((_error, admitted)) if admitted.len() > 1 && !cancellation.is_cancelled() => {
            prepared.results.extend(
                admitted.into_iter().map(|prepared| {
                    execute_scalar_view(session, &prepared.reference, cancellation)
                }),
            );
        }
        Err((error, admitted)) => prepared.results.extend(
            admitted
                .into_iter()
                .map(|prepared| ViewResult::failed(prepared.reference, error.clone())),
        ),
    }
    prepared.results
}

fn exact_shape_groups(
    views: &[ViewReference],
) -> UseResult<BTreeMap<(usize, usize), Vec<ViewReference>>> {
    let mut groups = BTreeMap::<(usize, usize), Vec<ViewReference>>::new();
    for reference in views {
        let (source_width, source_height) = reference.oriented_dimensions();
        let (width, height) = detection_resize_long_dimensions(
            source_width,
            source_height,
            RESIZE_LONG,
            RESIZE_STRIDE,
        )?;
        groups
            .entry((height as usize, width as usize))
            .or_default()
            .push(reference.clone());
    }
    Ok(groups)
}

fn maximum_batch_size(limits: &InferenceLimits, height: usize, width: usize) -> UseResult<usize> {
    let elements = 3_usize
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or_else(|| runtime_error("Seal-text input tensor size overflowed."))?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| runtime_error("Seal-text input byte size overflowed."))?;
    let maximum = MAX_BATCH_SLOTS
        .min(limits.max_tensor_elements / elements.max(1))
        .min(limits.max_input_bytes / bytes.max(1));
    if maximum == 0 {
        return Err(runtime_error(
            "Power limits cannot admit one seal-text input tensor.",
        ));
    }
    Ok(maximum)
}

fn prepare_input(
    reference: &ViewReference,
    cancellation: &CancellationToken,
) -> UseResult<DetectionInput> {
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
    detection_input_with_resize_long(image, &detection_config(), RESIZE_LONG, RESIZE_STRIDE)
}

fn execute_prepared_batch(
    session: &ModelSession<SealTextSession>,
    mut prepared: Vec<PreparedView>,
    cancellation: &CancellationToken,
) -> Result<Vec<ViewResult>, (UseError, Vec<PreparedView>)> {
    let trace = std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some();
    let started = std::time::Instant::now();
    let batch_size = prepared.len();
    let Some(first) = prepared.first() else {
        return Ok(Vec::new());
    };
    let [_, channels, height, width] = first.input.shape;
    if prepared
        .iter()
        .any(|prepared| prepared.input.shape[1..] != [channels, height, width])
    {
        return Err((
            runtime_error("A seal-text batch mixed incompatible tensor shapes."),
            prepared,
        ));
    }
    let value_count = match prepared.iter().try_fold(0_usize, |total, prepared| {
        total.checked_add(prepared.input.data.len())
    }) {
        Some(value_count) => value_count,
        None => {
            return Err((
                runtime_error("Seal-text batch value count overflowed."),
                prepared,
            ));
        }
    };
    let mut values = Vec::with_capacity(value_count);
    for prepared in &mut prepared {
        values.append(&mut prepared.input.data);
    }
    let collated_at = started.elapsed();
    let permit = match session.runtime().begin(cancellation) {
        Ok(permit) => permit,
        Err(error) => {
            return Err((power_error("admit a seal-text batch", error), prepared));
        }
    };
    let output = match session.value().engine.infer_batch(
        [batch_size, channels, height, width],
        values,
        session.runtime(),
        &permit,
        cancellation,
    ) {
        Ok(output) => output,
        Err(error) => return Err((error, prepared)),
    };
    let inferred_at = started.elapsed();
    let slot_elements = match output.tensor.shape[1..]
        .iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
    {
        Some(elements)
            if elements > 0
                && elements
                    .checked_mul(batch_size)
                    .is_some_and(|total| total == output.tensor.values.len()) =>
        {
            elements
        }
        _ => {
            return Err((
                runtime_error("Seal-text batch output cannot be partitioned by input view."),
                prepared,
            ));
        }
    };
    let output_shape = [
        1,
        output.tensor.shape[1],
        output.tensor.shape[2],
        output.tensor.shape[3],
    ];
    let receipt = project_receipt(output.receipt);
    let values = &output.tensor.values;
    let results = prepared
        .into_par_iter()
        .enumerate()
        .map(|(index, prepared)| {
            let start = index * slot_elements;
            let end = start + slot_elements;
            let evidence = detection_boxes_in_content(
                &values[start..end],
                &output_shape,
                prepared.input.geometry.content_width,
                prepared.input.geometry.content_height,
                prepared.input.geometry.original_width,
                prepared.input.geometry.original_height,
                &detection_config(),
            )
            .map(|detections| ViewEvidence {
                observations: observations(&prepared.reference, &detections),
                receipt: receipt.clone(),
            });
            ViewResult::new(prepared.reference, evidence)
        })
        .collect();
    if trace {
        let completed = started.elapsed();
        eprintln!(
            "A3S_OCR_SEAL_TEXT_BATCH_TIMING views={batch_size} height={height} width={width} collate_ms={:.3} inference_ms={:.3} postprocess_ms={:.3} total_ms={:.3}",
            collated_at.as_secs_f64() * 1_000.0,
            (inferred_at - collated_at).as_secs_f64() * 1_000.0,
            (completed - inferred_at).as_secs_f64() * 1_000.0,
            completed.as_secs_f64() * 1_000.0,
        );
    }
    Ok(results)
}

fn observations(
    reference: &ViewReference,
    detections: &[crate::postprocess::Detection],
) -> Vec<super::decoder::SealTextObservation> {
    match reference.orientation {
        ViewOrientation::Direct => direct_observations(
            detections,
            reference.image.width(),
            reference.image.height(),
        ),
        ViewOrientation::Clockwise90 => orthogonal_observations(
            detections,
            reference.image.width(),
            reference.image.height(),
        ),
        ViewOrientation::Clockwise180 => clockwise180_observations(
            detections,
            reference.image.width(),
            reference.image.height(),
        ),
        ViewOrientation::Clockwise270 => clockwise270_observations(
            detections,
            reference.image.width(),
            reference.image.height(),
        ),
    }
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
    fn balanced_batches_obey_the_resource_limit_for_general_cardinalities() {
        for item_count in 1..=128 {
            for batch_limit in 1..=32 {
                let ranges = balanced_batch_ranges(item_count, batch_limit);
                assert_eq!(ranges.len(), item_count.div_ceil(batch_limit));
                assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), item_count);
                assert!(ranges.iter().all(|range| range.len() <= batch_limit));
                let mut cursor = 0_usize;
                for range in &ranges {
                    assert_eq!(range.start, cursor);
                    assert!(range.end > range.start);
                    cursor = range.end;
                }
                assert_eq!(cursor, item_count);
                let minimum = ranges.iter().map(Range::len).min().unwrap();
                let maximum = ranges.iter().map(Range::len).max().unwrap();
                assert!(maximum - minimum <= 1);
            }
        }
    }

    #[test]
    fn least_work_scheduler_balances_shape_derived_cost_and_preserves_batch_ids() {
        let batches = [13_u64, 11, 7, 5, 3, 2]
            .into_iter()
            .map(|estimated_work| ViewBatch {
                references: Vec::new(),
                estimated_work,
            })
            .collect::<Vec<_>>();

        let assignments = least_work_assignments(&batches, 3);
        let mut batch_ids = assignments.iter().flatten().copied().collect::<Vec<_>>();
        batch_ids.sort_unstable();
        assert_eq!(batch_ids, (0..batches.len()).collect::<Vec<_>>());
        let loads = assignments
            .iter()
            .map(|assignment| assignment.len())
            .collect::<Vec<_>>();
        assert_eq!(loads, [1, 2, 3]);
        let loads = assignment_loads(&batches, &assignments);
        assert_eq!(loads, [13, 14, 14]);
    }

    #[test]
    fn zero_workers_produce_no_assignments() {
        assert!(least_work_assignments(&[], 0).is_empty());
    }

    #[test]
    fn batch_limit_is_derived_from_power_tensor_and_input_bounds() {
        let limits = InferenceLimits::default();
        let height = 768;
        let width = 640;
        let elements = 3 * height * width;
        let expected = MAX_BATCH_SLOTS
            .min(limits.max_tensor_elements / elements)
            .min(limits.max_input_bytes / (elements * std::mem::size_of::<f32>()));
        assert_eq!(
            maximum_batch_size(&limits, height, width).unwrap(),
            expected
        );
    }

    #[test]
    fn batch_limit_rejects_a_single_tensor_outside_power_authority() {
        let limits = InferenceLimits {
            max_input_bytes: 1,
            ..InferenceLimits::default()
        };
        assert_eq!(
            maximum_batch_size(&limits, 768, 640).unwrap_err().code,
            "use.ocr.runtime_failed"
        );
    }
}
