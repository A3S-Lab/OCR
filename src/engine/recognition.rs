use a3s_power::inference::{ExecutionPermit, ExecutionReceipt, RuntimeDeviceKind};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use rayon::prelude::*;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::{engine_error, EngineBlock, EngineExtraction, PpOcrV6Engine};
use crate::batch::OcrPixelWindow;
use crate::cancellation::check_cancelled;
use crate::config::RecognitionConfig;
use crate::postprocess::{decode_ctc_top1, Detection, Recognition};
use crate::ppocr_v6::native::NativeGraphOutput;
use crate::preprocess::{
    recognition_canvas_width, recognition_content_width, recognition_input_with_canvas_width,
    RecognitionInput,
};

mod crop;
#[cfg(test)]
mod orientation_probe;
mod planning;

// The reviewed recognition graph downsamples its width axis by a cumulative
// factor of eight before the terminal classifier. This graph-topology bound
// is pinned by the recognition model identity and is independent of content.
const RECOGNITION_TEMPORAL_DOWNSAMPLE: usize = 8;

// At most one successor window is prepared while the current accelerator
// window executes. Each window independently retains the exact per-lane
// tensor and input-byte admission bounds below, so host preparation remains
// finite and derived from live Power limits rather than document cardinality.
const RECOGNITION_PREPARED_WINDOW_DEPTH: usize = 2;

fn accelerator_execution_windows_enabled(device: RuntimeDeviceKind) -> bool {
    if device == RuntimeDeviceKind::Cpu {
        return false;
    }
    #[cfg(test)]
    if std::env::var_os("A3S_OCR_TEST_DISABLE_RECOGNITION_EXECUTION_WINDOWS").is_some() {
        return false;
    }
    true
}

use super::scheduling::{
    available_execution_workers, can_append_execution_job, cpu_execution_window_policy,
    CpuExecutionWindowPolicy,
};
use crop::PerspectiveCropPlan;
use planning::{plan_width_batches, window_reservations_fit_lanes, RecognitionWindowReservation};

struct RecognitionWorkItem {
    image_index: usize,
    detection_index: usize,
    detection: Detection,
    crop: PerspectiveCropPlan,
    content_width: u32,
    canvas_width: u32,
    selected: bool,
}

enum ImageRecognition {
    Failed(UseError),
    Pending {
        blocks: Vec<Option<EngineBlock>>,
        selected: Vec<bool>,
        receipts: Vec<ExecutionReceipt>,
    },
}

struct RecognizedBatch {
    items: Vec<UseResult<Recognition>>,
    receipt: ExecutionReceipt,
}

struct PreparedRecognitionBatch {
    crops: Vec<(usize, RgbImage)>,
    canvas_width: u32,
    input: Option<UseResult<RecognitionInput>>,
    input_slots: usize,
    failures: Vec<(usize, UseError)>,
    crop_preparation: Duration,
    tensor_preparation: Duration,
}

struct ActiveRecognitionBatch {
    work_indices: Vec<usize>,
    canvas_width: u32,
}

pub(super) struct RecognitionAdmission<'a> {
    permit: &'a ExecutionPermit,
    cancellation: &'a CancellationToken,
}

impl<'a> RecognitionAdmission<'a> {
    pub(super) fn new(permit: &'a ExecutionPermit, cancellation: &'a CancellationToken) -> Self {
        Self {
            permit,
            cancellation,
        }
    }
}

struct RecognitionBatchCursor {
    batches: std::vec::IntoIter<Vec<usize>>,
    deferred: Option<Vec<usize>>,
}

struct RecognitionBatchExecution {
    result: UseResult<RecognizedBatch>,
    inference: Duration,
    decoding: Duration,
    batch_executed: bool,
}

struct RecognitionWindowExecution {
    batches: Vec<RecognitionBatchExecution>,
    active_lanes: usize,
}

struct RecognitionWindowJob {
    position: usize,
    crop_count: usize,
    input: RecognitionInput,
}

enum RecognitionLanePermit<'a> {
    Primary(&'a ExecutionPermit),
    Helper(ExecutionPermit),
}

impl RecognitionLanePermit<'_> {
    fn as_ref(&self) -> &ExecutionPermit {
        match self {
            Self::Primary(permit) => permit,
            Self::Helper(permit) => permit,
        }
    }
}

struct RecognitionLane<'a> {
    engine: &'a PpOcrV6Engine,
    permit: RecognitionLanePermit<'a>,
}

#[derive(Default)]
struct RecognitionTimings {
    crop_preparation: Duration,
    tensor_preparation: Duration,
    inference: Duration,
    decoding: Duration,
    execution_wall: Duration,
    preparation_wait: Duration,
    batches: usize,
    maximum_parallel_batches: usize,
    prefetched_windows: usize,
}

impl RecognitionBatchCursor {
    fn new(batches: Vec<Vec<usize>>) -> Self {
        Self {
            batches: batches.into_iter(),
            deferred: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn take_window(
        &mut self,
        work: &[RecognitionWorkItem],
        states: &[ImageRecognition],
        config: &RecognitionConfig,
        policy: CpuExecutionWindowPolicy,
        use_accelerator_windows: bool,
        accelerator_lane_count: usize,
        maximum_input_bytes: usize,
        cancellation: &CancellationToken,
    ) -> UseResult<Option<Vec<ActiveRecognitionBatch>>> {
        loop {
            let mut selected = Vec::new();
            let mut accelerator_reservations = Vec::new();
            let mut reserved_elements = 0_usize;
            let mut reserved_input_bytes = 0_usize;
            let mut exhausted = false;
            while selected.len() < policy.maximum_parallel_jobs {
                let batch = match self.deferred.take().or_else(|| self.batches.next()) {
                    Some(batch) => batch,
                    None => {
                        exhausted = true;
                        break;
                    }
                };
                check_cancelled(cancellation)?;
                let active = active_work_indices(&batch, work, states);
                if active.is_empty() {
                    continue;
                }
                let canvas_width = planned_batch_canvas_width(&batch, work)?;
                let batch_reservation =
                    recognition_batch_reservation_elements(active.len(), canvas_width, config)?;
                let batch_input_bytes =
                    recognition_batch_input_bytes(active.len(), canvas_width, config)?;
                let fits = if use_accelerator_windows {
                    accelerator_reservations.push(RecognitionWindowReservation {
                        elements: batch_reservation,
                        input_bytes: batch_input_bytes,
                    });
                    selected.is_empty()
                        || window_reservations_fit_lanes(
                            &accelerator_reservations,
                            accelerator_lane_count,
                            policy.maximum_reserved_elements,
                            maximum_input_bytes,
                        )
                } else {
                    let input_bytes_fit = selected.is_empty()
                        || reserved_input_bytes
                            .checked_add(batch_input_bytes)
                            .is_some_and(|bytes| bytes <= maximum_input_bytes);
                    can_append_execution_job(
                        selected.len(),
                        reserved_elements,
                        batch_reservation,
                        policy,
                    ) && input_bytes_fit
                };
                if !fits {
                    accelerator_reservations.pop();
                    self.deferred = Some(batch);
                    break;
                }
                reserved_elements = reserved_elements.saturating_add(batch_reservation);
                reserved_input_bytes = reserved_input_bytes.saturating_add(batch_input_bytes);
                selected.push(ActiveRecognitionBatch {
                    work_indices: active,
                    canvas_width,
                });
            }

            if !selected.is_empty() {
                return Ok(Some(selected));
            }
            if exhausted {
                return Ok(None);
            }
        }
    }
}

impl PpOcrV6Engine {
    #[cfg(test)]
    pub(super) fn recognize_detected_batch(
        &self,
        images: &[&RgbImage],
        detections: Vec<UseResult<Vec<Detection>>>,
        detection_receipts: Vec<Vec<ExecutionReceipt>>,
        text_windows: &[Option<OcrPixelWindow>],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Vec<UseResult<EngineExtraction>>> {
        self.recognize_detected_batch_with_helpers(
            images,
            detections,
            detection_receipts,
            text_windows,
            RecognitionAdmission::new(permit, cancellation),
            &[],
        )
    }

    pub(super) fn recognize_detected_batch_with_helpers(
        &self,
        images: &[&RgbImage],
        detections: Vec<UseResult<Vec<Detection>>>,
        detection_receipts: Vec<Vec<ExecutionReceipt>>,
        text_windows: &[Option<OcrPixelWindow>],
        admission: RecognitionAdmission<'_>,
        recognition_helpers: &[&PpOcrV6Engine],
    ) -> UseResult<Vec<UseResult<EngineExtraction>>> {
        if detections.len() != images.len()
            || detection_receipts.len() != images.len()
            || text_windows.len() != images.len()
        {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 recognition received mismatched image, detection, receipt, or Text-window cardinality.",
            ));
        }

        let (mut states, work) =
            self.prepare_recognition_work(images, detections, detection_receipts, text_windows);
        let canvas_widths = work
            .iter()
            .map(|item| item.canvas_width)
            .collect::<Vec<_>>();
        let maximum_tensor_elements = self.native.maximum_tensor_elements();
        let batches = plan_width_batches(
            &canvas_widths,
            self.native.runtime_device_kind(),
            |canvas_width| {
                let per_crop = recognition_batch_reservation_elements(
                    1,
                    canvas_width,
                    &self.recognition_config,
                )?;
                Ok(maximum_tensor_elements
                    .checked_div(per_crop)
                    .unwrap_or(0)
                    .max(1))
            },
        )?;
        let trace_timings = std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some()
            || std::env::var_os("A3S_OCR_TRACE_PIPELINE_TIMINGS").is_some();
        let trace_width_histogram =
            std::env::var_os("A3S_OCR_TRACE_RECOGNITION_WIDTH_HISTOGRAM").is_some();
        let trace_summary = trace_timings
            || trace_width_histogram
            || std::env::var_os("A3S_OCR_TRACE_RECOGNITION_SUMMARY").is_some();
        if trace_summary {
            let selected_crops = work.iter().filter(|item| item.selected).count();
            let width_cohorts = canvas_widths
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let mut sorted_widths = canvas_widths.clone();
            sorted_widths.sort_unstable();
            let width_sum = sorted_widths
                .iter()
                .map(|width| u64::from(*width))
                .sum::<u64>();
            let mut sorted_content_widths = work
                .iter()
                .map(|item| item.content_width)
                .collect::<Vec<_>>();
            sorted_content_widths.sort_unstable();
            let content_width_sum = sorted_content_widths
                .iter()
                .map(|width| u64::from(*width))
                .sum::<u64>();
            let width_at = |percent: usize| {
                sorted_widths
                    .get(sorted_widths.len().saturating_sub(1) * percent / 100)
                    .copied()
                    .unwrap_or(0)
            };
            let content_width_at = |percent: usize| {
                sorted_content_widths
                    .get(sorted_content_widths.len().saturating_sub(1) * percent / 100)
                    .copied()
                    .unwrap_or(0)
            };
            let scalar_batches = batches.iter().filter(|batch| batch.len() == 1).count();
            let maximum_batch = batches.iter().map(Vec::len).max().unwrap_or(0);
            let maximum_width = canvas_widths.iter().copied().max().unwrap_or(0);
            eprintln!(
                "A3S_OCR_RECOGNITION_PLAN detections={} selected_crops={} width_cohorts={} batches={} scalar_batches={} maximum_batch={} content_width_sum={} content_width_mean={:.3} content_width_p50={} content_width_p90={} content_width_p99={} canvas_width_sum={} canvas_width_mean={:.3} canvas_width_p50={} canvas_width_p90={} canvas_width_p99={} maximum_width={}",
                work.len(),
                selected_crops,
                width_cohorts,
                batches.len(),
                scalar_batches,
                maximum_batch,
                content_width_sum,
                if sorted_content_widths.is_empty() {
                    0.0
                } else {
                    content_width_sum as f64 / sorted_content_widths.len() as f64
                },
                content_width_at(50),
                content_width_at(90),
                content_width_at(99),
                width_sum,
                if sorted_widths.is_empty() {
                    0.0
                } else {
                    width_sum as f64 / sorted_widths.len() as f64
                },
                width_at(50),
                width_at(90),
                width_at(99),
                maximum_width,
            );
            if trace_width_histogram {
                let mut counts = std::collections::BTreeMap::<u32, usize>::new();
                for width in &canvas_widths {
                    *counts.entry(*width).or_default() += 1;
                }
                let histogram = counts
                    .into_iter()
                    .map(|(width, count)| format!("{width}:{count}"))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("A3S_OCR_RECOGNITION_WIDTH_HISTOGRAM widths={histogram}");
            }
        }
        let recognition_started = Instant::now();
        let mut timings = RecognitionTimings::default();
        self.recognize_planned_batches(
            images,
            &work,
            &mut states,
            batches,
            admission.permit,
            admission.cancellation,
            recognition_helpers,
            trace_timings.then_some(&mut timings),
        )?;
        if trace_timings {
            let selected_crops = work.iter().filter(|item| item.selected).count();
            eprintln!(
                "A3S_OCR_RECOGNITION_TIMING detections={} selected_crops={} batches={} maximum_parallel_batches={} prepared_window_depth={} prefetched_windows={} crop_ms={:.3} tensor_ms={:.3} inference_work_ms={:.3} decode_work_ms={:.3} execution_wall_ms={:.3} preparation_wait_ms={:.3} total_ms={:.3}",
                work.len(),
                selected_crops,
                timings.batches,
                timings.maximum_parallel_batches,
                RECOGNITION_PREPARED_WINDOW_DEPTH,
                timings.prefetched_windows,
                timings.crop_preparation.as_secs_f64() * 1_000.0,
                timings.tensor_preparation.as_secs_f64() * 1_000.0,
                timings.inference.as_secs_f64() * 1_000.0,
                timings.decoding.as_secs_f64() * 1_000.0,
                timings.execution_wall.as_secs_f64() * 1_000.0,
                timings.preparation_wait.as_secs_f64() * 1_000.0,
                recognition_started.elapsed().as_secs_f64() * 1_000.0,
            );
        }
        let model = self.model_profile.family();
        Ok(states
            .into_iter()
            .map(|state| state.finish(model))
            .collect())
    }

    fn prepare_recognition_work(
        &self,
        images: &[&RgbImage],
        detections: Vec<UseResult<Vec<Detection>>>,
        detection_receipts: Vec<Vec<ExecutionReceipt>>,
        text_windows: &[Option<OcrPixelWindow>],
    ) -> (Vec<ImageRecognition>, Vec<RecognitionWorkItem>) {
        let mut states = Vec::with_capacity(images.len());
        let mut work = Vec::new();
        for (image_index, (detections, receipts)) in
            detections.into_iter().zip(detection_receipts).enumerate()
        {
            let detections = match detections {
                Ok(detections) => detections,
                Err(error) => {
                    states.push(ImageRecognition::Failed(error));
                    continue;
                }
            };
            let mut image_work = Vec::with_capacity(detections.len());
            let mut selected = Vec::with_capacity(detections.len());
            let mut error = None;
            for (detection_index, detection) in detections.iter().cloned().enumerate() {
                let is_selected = text_windows[image_index]
                    .map(|window| detection_intersects_window(&detection, window))
                    .unwrap_or(true);
                let crop = match PerspectiveCropPlan::new(&detection) {
                    Ok(crop) => crop,
                    Err(crop_error) => {
                        error = Some(crop_error);
                        break;
                    }
                };
                let (width, height) = crop.output_dimensions();
                let content_width =
                    match recognition_content_width(width, height, &self.recognition_config) {
                        Ok(width) => width,
                        Err(width_error) => {
                            error = Some(width_error);
                            break;
                        }
                    };
                let canvas_width =
                    match recognition_canvas_width(width, height, &self.recognition_config) {
                        Ok(width) => width,
                        Err(width_error) => {
                            error = Some(width_error);
                            break;
                        }
                    };
                image_work.push(RecognitionWorkItem {
                    image_index,
                    detection_index,
                    detection,
                    crop,
                    content_width,
                    canvas_width,
                    selected: is_selected,
                });
                selected.push(is_selected);
            }
            if let Some(error) = error {
                states.push(ImageRecognition::Failed(error));
                continue;
            }
            states.push(ImageRecognition::Pending {
                blocks: vec![None; detections.len()],
                selected,
                receipts,
            });
            work.extend(image_work);
        }
        (states, work)
    }

    #[allow(clippy::too_many_arguments)]
    fn recognize_planned_batches(
        &self,
        images: &[&RgbImage],
        work: &[RecognitionWorkItem],
        states: &mut [ImageRecognition],
        batches: Vec<Vec<usize>>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        recognition_helpers: &[&PpOcrV6Engine],
        mut timings: Option<&mut RecognitionTimings>,
    ) -> UseResult<()> {
        let device = self.native.runtime_device_kind();
        let batches =
            prioritize_recognition_batches(batches, work, states, &self.recognition_config)?;
        let use_accelerator_windows = accelerator_execution_windows_enabled(device);
        let mut policy = cpu_execution_window_policy(
            device,
            available_execution_workers(),
            self.native.maximum_tensor_elements(),
        );
        if use_accelerator_windows {
            // The finite plan cardinality is the protocol bound. Each admitted
            // model-session lane retains its own tensor and input-byte budget.
            policy.maximum_parallel_jobs = accelerator_window_job_limit(batches.len());
        }
        let maximum_input_bytes = self.native.maximum_input_bytes();
        let accelerator_lane_count = if use_accelerator_windows {
            planned_recognition_lane_count(self, recognition_helpers)
        } else {
            1
        };
        let mut cursor = RecognitionBatchCursor::new(batches);
        let Some(initial) = cursor.take_window(
            work,
            states,
            &self.recognition_config,
            policy,
            use_accelerator_windows,
            accelerator_lane_count,
            maximum_input_bytes,
            cancellation,
        )?
        else {
            return Ok(());
        };
        let mut prepared_window =
            prepare_recognition_window(images, work, initial, &self.recognition_config);

        loop {
            reconcile_prepared_window(
                &mut prepared_window,
                work,
                states,
                &self.recognition_config,
                timings.as_deref_mut(),
            );
            prepared_window.retain(|prepared| !prepared.crops.is_empty());
            if prepared_window.is_empty() {
                let Some(selected) = cursor.take_window(
                    work,
                    states,
                    &self.recognition_config,
                    policy,
                    use_accelerator_windows,
                    accelerator_lane_count,
                    maximum_input_bytes,
                    cancellation,
                )?
                else {
                    return Ok(());
                };
                prepared_window =
                    prepare_recognition_window(images, work, selected, &self.recognition_config);
                continue;
            }

            check_cancelled(cancellation)?;
            let successor = if use_accelerator_windows {
                cursor.take_window(
                    work,
                    states,
                    &self.recognition_config,
                    policy,
                    use_accelerator_windows,
                    accelerator_lane_count,
                    maximum_input_bytes,
                    cancellation,
                )?
            } else {
                None
            };
            let (execution, execution_wall, prepared_successor, preparation_wait) =
                if let Some(successor) = successor {
                    let config = &self.recognition_config;
                    std::thread::scope(|scope| -> UseResult<_> {
                        let preparation = scope.spawn(move || {
                            prepare_recognition_window(images, work, successor, config)
                        });
                        let execution_started = Instant::now();
                        let execution = self.execute_prepared_batches(
                            &mut prepared_window,
                            permit,
                            cancellation,
                            recognition_helpers,
                            use_accelerator_windows,
                        );
                        let execution_wall = execution_started.elapsed();
                        let wait_started = Instant::now();
                        let prepared_successor = preparation.join().map_err(|_| {
                            engine_error(
                                "use.ocr.runtime_failed",
                                "A recognition preparation worker terminated unexpectedly.",
                            )
                        })?;
                        Ok((
                            execution,
                            execution_wall,
                            Some(prepared_successor),
                            wait_started.elapsed(),
                        ))
                    })?
                } else {
                    let execution_started = Instant::now();
                    let execution = self.execute_prepared_batches(
                        &mut prepared_window,
                        permit,
                        cancellation,
                        recognition_helpers,
                        use_accelerator_windows,
                    );
                    (execution, execution_started.elapsed(), None, Duration::ZERO)
                };
            let execution = execution?;
            if let Some(timings) = timings.as_deref_mut() {
                timings.execution_wall += execution_wall;
                timings.preparation_wait += preparation_wait;
                timings.prefetched_windows += usize::from(prepared_successor.is_some());
                timings.maximum_parallel_batches =
                    timings.maximum_parallel_batches.max(execution.active_lanes);
            }
            self.apply_recognition_window(
                prepared_window,
                execution,
                work,
                states,
                permit,
                cancellation,
                timings.as_deref_mut(),
            )?;

            if let Some(successor) = prepared_successor {
                prepared_window = successor;
                continue;
            }
            let Some(selected) = cursor.take_window(
                work,
                states,
                &self.recognition_config,
                policy,
                use_accelerator_windows,
                accelerator_lane_count,
                maximum_input_bytes,
                cancellation,
            )?
            else {
                return Ok(());
            };
            prepared_window =
                prepare_recognition_window(images, work, selected, &self.recognition_config);
        }
    }

    fn execute_prepared_batches(
        &self,
        prepared: &mut [PreparedRecognitionBatch],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        recognition_helpers: &[&PpOcrV6Engine],
        use_accelerator_windows: bool,
    ) -> UseResult<RecognitionWindowExecution> {
        if use_accelerator_windows && prepared.len() > 1 {
            self.recognize_prepared_window(prepared, permit, cancellation, recognition_helpers)
        } else if prepared.len() == 1 {
            Ok(RecognitionWindowExecution {
                batches: vec![self.recognize_prepared_batch(
                    &mut prepared[0],
                    permit,
                    cancellation,
                )],
                active_lanes: 1,
            })
        } else {
            Ok(RecognitionWindowExecution {
                batches: prepared
                    .par_iter_mut()
                    .map(|prepared| self.recognize_prepared_batch(prepared, permit, cancellation))
                    .collect::<Vec<_>>(),
                active_lanes: prepared.len(),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_recognition_window(
        &self,
        prepared: Vec<PreparedRecognitionBatch>,
        execution: RecognitionWindowExecution,
        work: &[RecognitionWorkItem],
        states: &mut [ImageRecognition],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        mut timings: Option<&mut RecognitionTimings>,
    ) -> UseResult<()> {
        for (prepared, execution) in prepared.into_iter().zip(execution.batches) {
            if let Some(timings) = timings.as_deref_mut() {
                timings.inference += execution.inference;
                timings.decoding += execution.decoding;
                timings.batches += usize::from(execution.batch_executed);
            }
            match execution.result {
                Ok(recognized) => {
                    let work_indices = prepared
                        .crops
                        .iter()
                        .map(|(work_index, _)| *work_index)
                        .collect::<Vec<_>>();
                    apply_recognized_batch(&work_indices, recognized, work, states)?;
                }
                Err(_) if prepared.crops.len() > 1 => {
                    check_cancelled(cancellation)?;
                    self.recognize_scalar_fallback(
                        &prepared.crops,
                        work,
                        states,
                        permit,
                        cancellation,
                        timings.as_deref_mut(),
                    )?;
                }
                Err(error) => {
                    check_cancelled(cancellation)?;
                    states[work[prepared.crops[0].0].image_index].fail(error);
                }
            }
        }
        Ok(())
    }

    fn recognize_crop_batch(
        &self,
        crops: &[&RgbImage],
        canvas_width: u32,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        mut timings: Option<&mut RecognitionTimings>,
    ) -> UseResult<RecognizedBatch> {
        check_cancelled(cancellation)?;
        let started = Instant::now();
        let input =
            recognition_input_with_canvas_width(crops, &self.recognition_config, canvas_width)?;
        if let Some(timings) = timings.as_deref_mut() {
            timings.tensor_preparation += started.elapsed();
        }
        let execution_started = Instant::now();
        let execution = self.recognize_input(crops.len(), Ok(input), permit, cancellation);
        if let Some(timings) = timings {
            timings.inference += execution.inference;
            timings.decoding += execution.decoding;
            timings.execution_wall += execution_started.elapsed();
            timings.batches += usize::from(execution.batch_executed);
            timings.maximum_parallel_batches = timings.maximum_parallel_batches.max(1);
        }
        execution.result
    }

    fn recognize_prepared_batch(
        &self,
        prepared: &mut PreparedRecognitionBatch,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> RecognitionBatchExecution {
        let input = prepared.input.take().unwrap_or_else(|| {
            Err(engine_error(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 prepared recognition batch has no input tensor.",
            ))
        });
        self.recognize_input(prepared.crops.len(), input, permit, cancellation)
    }

    fn recognize_prepared_window(
        &self,
        prepared: &mut [PreparedRecognitionBatch],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        recognition_helpers: &[&PpOcrV6Engine],
    ) -> UseResult<RecognitionWindowExecution> {
        check_cancelled(cancellation)?;
        let mut executions = (0..prepared.len()).map(|_| None).collect::<Vec<_>>();
        let mut jobs = Vec::with_capacity(prepared.len());
        for (position, batch) in prepared.iter_mut().enumerate() {
            let input = batch.input.take().unwrap_or_else(|| {
                Err(engine_error(
                    "use.ocr.provider_input_invalid",
                    "PP-OCRv6 prepared recognition batch has no input tensor.",
                ))
            });
            match input {
                Ok(input) => jobs.push(RecognitionWindowJob {
                    position,
                    crop_count: batch.crops.len(),
                    input,
                }),
                Err(error) => executions[position] = Some(failed_batch_execution(error, false)),
            }
        }

        let mut lanes = vec![RecognitionLane {
            engine: self,
            permit: RecognitionLanePermit::Primary(permit),
        }];
        for helper in recognition_helpers {
            if lanes.iter().any(|lane| std::ptr::eq(lane.engine, *helper)) {
                continue;
            }
            check_cancelled(cancellation)?;
            if let Ok(helper_permit) = helper.native.begin(cancellation) {
                lanes.push(RecognitionLane {
                    engine: helper,
                    permit: RecognitionLanePermit::Helper(helper_permit),
                });
            }
        }

        let mut assignments = Vec::with_capacity(lanes.len());
        assignments.resize_with(lanes.len(), Vec::new);
        let mut assigned_work = vec![0_usize; lanes.len()];
        for job in jobs {
            let lane_index = assigned_work
                .iter()
                .enumerate()
                .min_by_key(|(lane_index, work)| (**work, *lane_index))
                .map(|(lane_index, _)| lane_index)
                .unwrap_or(0);
            let canvas_width = prepared[job.position].canvas_width;
            let reservation = recognition_batch_reservation_elements(
                job.crop_count,
                canvas_width,
                &self.recognition_config,
            )?;
            assigned_work[lane_index] = assigned_work[lane_index].saturating_add(reservation);
            assignments[lane_index].push(job);
        }
        let active_lanes = assignments.iter().filter(|jobs| !jobs.is_empty()).count();
        let lane_results = lanes
            .into_par_iter()
            .zip(assignments.into_par_iter())
            .filter(|(_, jobs)| !jobs.is_empty())
            .map(|(lane, jobs)| {
                lane.engine.execute_recognition_window_lane(
                    jobs,
                    prepared,
                    lane.permit.as_ref(),
                    cancellation,
                )
            })
            .collect::<Vec<_>>();
        for lane_result in lane_results {
            for (position, execution) in lane_result? {
                executions[position] = Some(execution);
            }
        }

        let batches = executions
            .into_iter()
            .map(|execution| {
                execution.ok_or_else(|| {
                    engine_error(
                        "use.ocr.provider_output_invalid",
                        "PP-OCRv6 recognition execution window omitted a batch result.",
                    )
                })
            })
            .collect::<UseResult<Vec<_>>>()?;
        Ok(RecognitionWindowExecution {
            batches,
            active_lanes,
        })
    }

    fn execute_recognition_window_lane(
        &self,
        jobs: Vec<RecognitionWindowJob>,
        prepared: &[PreparedRecognitionBatch],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Vec<(usize, RecognitionBatchExecution)>> {
        let graph_count = jobs.len();
        let inferred_at = Instant::now();
        let metadata = jobs
            .iter()
            .map(|job| (job.position, job.crop_count))
            .collect::<Vec<_>>();
        let inputs = jobs.into_iter().map(|job| job.input).collect::<Vec<_>>();
        match self
            .native
            .recognize_prepared_window(inputs, permit, cancellation)
        {
            Ok(outputs) => {
                let inference = inferred_at.elapsed();
                if outputs.len() != graph_count {
                    return Err(engine_error(
                        "use.ocr.provider_output_invalid",
                        "PP-OCRv6 recognition execution lane returned mismatched output cardinality.",
                    ));
                }
                Ok(metadata
                    .into_iter()
                    .zip(outputs)
                    .enumerate()
                    .map(|(output_index, ((position, crop_count), output))| {
                        let attributed_inference = if output_index == 0 {
                            inference
                        } else {
                            Duration::ZERO
                        };
                        (
                            position,
                            self.decode_recognition_output(
                                crop_count,
                                output,
                                attributed_inference,
                            ),
                        )
                    })
                    .collect())
            }
            Err(window_error) => {
                check_cancelled(cancellation)?;
                if std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some() {
                    eprintln!(
                        "A3S_OCR_RECOGNITION_WINDOW_FALLBACK graphs={graph_count} error={window_error}"
                    );
                }
                // A lane is an execution optimization, not a new failure
                // boundary. Rebuild only its exact tensors and retain the
                // existing per-batch isolation semantics.
                let mut executions = Vec::with_capacity(graph_count);
                for (position, crop_count) in metadata {
                    check_cancelled(cancellation)?;
                    let batch = &prepared[position];
                    let crops = batch.crops.iter().map(|(_, crop)| crop).collect::<Vec<_>>();
                    let input = recognition_input_with_canvas_width(
                        &crops,
                        &self.recognition_config,
                        batch.canvas_width,
                    );
                    executions.push((
                        position,
                        self.recognize_input(crop_count, input, permit, cancellation),
                    ));
                }
                Ok(executions)
            }
        }
    }

    fn recognize_input(
        &self,
        crop_count: usize,
        input: UseResult<RecognitionInput>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> RecognitionBatchExecution {
        if let Err(error) = check_cancelled(cancellation) {
            return failed_batch_execution(error, false);
        }
        let input = match input {
            Ok(input) => input,
            Err(error) => return failed_batch_execution(error, false),
        };
        let inferred_at = Instant::now();
        let recognition = match self.native.recognize_prepared(input, permit, cancellation) {
            Ok(recognition) => recognition,
            Err(error) => {
                return RecognitionBatchExecution {
                    result: Err(error),
                    inference: inferred_at.elapsed(),
                    decoding: Duration::ZERO,
                    batch_executed: true,
                }
            }
        };
        let inference = inferred_at.elapsed();
        self.decode_recognition_output(crop_count, recognition, inference)
    }

    fn decode_recognition_output(
        &self,
        crop_count: usize,
        recognition: NativeGraphOutput,
        inference: Duration,
    ) -> RecognitionBatchExecution {
        let decoded_at = Instant::now();
        let shape = recognition.tensor.shape;
        let output = recognition.tensor.values;
        if shape.len() != 3 || shape[0] != crop_count {
            return failed_decoding_execution(
                engine_error(
                "use.ocr.provider_output_invalid",
                format!(
                    "PP-OCRv6 recognition output shape must be [N, T, C] for N={}, found {shape:?}.",
                    crop_count
                ),
                ),
                inference,
                decoded_at.elapsed(),
            );
        }
        let Some(item_len) = shape[1].checked_mul(shape[2]) else {
            return failed_decoding_execution(
                engine_error(
                    "use.ocr.provider_output_invalid",
                    "PP-OCRv6 recognition output dimensions overflowed.",
                ),
                inference,
                decoded_at.elapsed(),
            );
        };
        if output.len() != crop_count.saturating_mul(item_len) {
            return failed_decoding_execution(
                engine_error(
                    "use.ocr.provider_output_invalid",
                    "PP-OCRv6 recognition output length does not match its batch shape.",
                ),
                inference,
                decoded_at.elapsed(),
            );
        }
        let items = (0..crop_count)
            .map(|index| {
                let start = index * item_len;
                decode_ctc_top1(
                    &output[start..start + item_len],
                    &[1, shape[1], shape[2]],
                    &self.recognition_config,
                )
            })
            .collect();
        RecognitionBatchExecution {
            result: Ok(RecognizedBatch {
                items,
                receipt: recognition.receipt,
            }),
            inference,
            decoding: decoded_at.elapsed(),
            batch_executed: true,
        }
    }

    fn recognize_scalar_fallback(
        &self,
        crops: &[(usize, RgbImage)],
        work: &[RecognitionWorkItem],
        states: &mut [ImageRecognition],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        mut timings: Option<&mut RecognitionTimings>,
    ) -> UseResult<()> {
        for (work_index, crop) in crops {
            check_cancelled(cancellation)?;
            let image_index = work[*work_index].image_index;
            if !states[image_index].is_pending() {
                continue;
            }
            match self.recognize_crop_batch(
                &[crop],
                work[*work_index].canvas_width,
                permit,
                cancellation,
                timings.as_deref_mut(),
            ) {
                Ok(recognized) => apply_recognized_batch(&[*work_index], recognized, work, states)?,
                Err(error) => {
                    check_cancelled(cancellation)?;
                    states[image_index].fail(error);
                }
            }
        }
        Ok(())
    }
}

fn failed_batch_execution(error: UseError, batch_executed: bool) -> RecognitionBatchExecution {
    RecognitionBatchExecution {
        result: Err(error),
        inference: Duration::ZERO,
        decoding: Duration::ZERO,
        batch_executed,
    }
}

fn failed_decoding_execution(
    error: UseError,
    inference: Duration,
    decoding: Duration,
) -> RecognitionBatchExecution {
    RecognitionBatchExecution {
        result: Err(error),
        inference,
        decoding,
        batch_executed: true,
    }
}

fn active_work_indices(
    batch: &[usize],
    work: &[RecognitionWorkItem],
    states: &[ImageRecognition],
) -> Vec<usize> {
    batch
        .iter()
        .copied()
        .filter(|index| work[*index].selected && states[work[*index].image_index].is_pending())
        .collect()
}

fn planned_recognition_lane_count(
    primary: &PpOcrV6Engine,
    recognition_helpers: &[&PpOcrV6Engine],
) -> usize {
    let mut lanes = vec![primary];
    for helper in recognition_helpers {
        if !lanes.iter().any(|lane| std::ptr::eq(*lane, *helper)) {
            lanes.push(*helper);
        }
    }
    #[cfg(test)]
    if std::env::var_os("A3S_OCR_TEST_GLOBAL_RECOGNITION_WINDOW_BUDGET").is_some() {
        return 1;
    }
    lanes.len()
}

fn accelerator_window_job_limit(planned_batches: usize) -> usize {
    planned_batches.max(1)
}

fn planned_batch_canvas_width(batch: &[usize], work: &[RecognitionWorkItem]) -> UseResult<u32> {
    batch
        .iter()
        .map(|index| work[*index].canvas_width)
        .max()
        .ok_or_else(|| {
            engine_error(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 recognition planner produced an empty batch.",
            )
        })
}

fn recognition_batch_reservation_elements(
    slot_count: usize,
    canvas_width: u32,
    config: &RecognitionConfig,
) -> UseResult<usize> {
    let width = usize::try_from(canvas_width).map_err(|_| {
        engine_error(
            "use.ocr.provider_input_invalid",
            "PP-OCRv6 recognition canvas width cannot be represented.",
        )
    })?;
    let classes = config.characters.len().checked_add(2).ok_or_else(|| {
        engine_error(
            "use.ocr.provider_input_invalid",
            "PP-OCRv6 recognition class count overflowed.",
        )
    })?;
    let input_elements = slot_count
        .checked_mul(config.channels)
        .and_then(|elements| elements.checked_mul(config.height))
        .and_then(|elements| elements.checked_mul(width));
    // The pinned operator topology creates at most ceil(width / 8) CTC steps.
    // Reserve the complete classifier output even though the current CPU
    // projection retains only bounded tiles, so this remains conservative for
    // every execution device and independent of crop content.
    let timesteps = width.div_ceil(RECOGNITION_TEMPORAL_DOWNSAMPLE);
    let classifier_elements = slot_count
        .checked_mul(timesteps)
        .and_then(|elements| elements.checked_mul(classes));
    input_elements
        .and_then(|input| classifier_elements.and_then(|output| input.checked_add(output)))
        .ok_or_else(|| {
            engine_error(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 recognition execution reservation overflowed.",
            )
        })
}

fn recognition_batch_input_bytes(
    slot_count: usize,
    canvas_width: u32,
    config: &RecognitionConfig,
) -> UseResult<usize> {
    let width = usize::try_from(canvas_width).map_err(|_| {
        engine_error(
            "use.ocr.provider_input_invalid",
            "PP-OCRv6 recognition canvas width cannot be represented.",
        )
    })?;
    slot_count
        .checked_mul(config.channels)
        .and_then(|elements| elements.checked_mul(config.height))
        .and_then(|elements| elements.checked_mul(width))
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            engine_error(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 recognition input byte reservation overflowed.",
            )
        })
}

fn prioritize_recognition_batches(
    batches: Vec<Vec<usize>>,
    work: &[RecognitionWorkItem],
    states: &[ImageRecognition],
    config: &RecognitionConfig,
) -> UseResult<Vec<Vec<usize>>> {
    let mut scheduled = Vec::with_capacity(batches.len());
    for (original_index, batch) in batches.into_iter().enumerate() {
        let active_slots = active_work_indices(&batch, work, states).len();
        let reservation = if active_slots == 0 {
            0
        } else {
            recognition_batch_reservation_elements(
                active_slots,
                planned_batch_canvas_width(&batch, work)?,
                config,
            )?
        };
        scheduled.push((original_index, reservation, batch));
    }
    sort_scheduled_batches(&mut scheduled);
    Ok(scheduled.into_iter().map(|(_, _, batch)| batch).collect())
}

fn sort_scheduled_batches<T>(scheduled: &mut [(usize, usize, T)]) {
    // List scheduling has the smallest avoidable tail when the largest
    // declared jobs enter the bounded worker window first. Equal work keeps
    // canonical planning order, so the policy is deterministic.
    scheduled.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
}

fn prepare_recognition_window(
    images: &[&RgbImage],
    work: &[RecognitionWorkItem],
    selected: Vec<ActiveRecognitionBatch>,
    config: &RecognitionConfig,
) -> Vec<PreparedRecognitionBatch> {
    selected
        .into_par_iter()
        .map(|active| {
            prepare_recognition_batch(
                images,
                work,
                active.work_indices,
                active.canvas_width,
                config,
            )
        })
        .collect()
}

fn prepare_recognition_batch(
    images: &[&RgbImage],
    work: &[RecognitionWorkItem],
    active: Vec<usize>,
    canvas_width: u32,
    config: &RecognitionConfig,
) -> PreparedRecognitionBatch {
    let crop_started = Instant::now();
    let prepared = active
        .into_par_iter()
        .map(|work_index| {
            let item = &work[work_index];
            (work_index, item.crop.execute(images[item.image_index]))
        })
        .collect::<Vec<_>>();
    let mut crops = Vec::with_capacity(prepared.len());
    let mut failures = Vec::new();
    let mut failed_images = std::collections::BTreeSet::new();
    for (work_index, crop) in prepared {
        let image_index = work[work_index].image_index;
        match crop {
            Ok(crop) => crops.push((work_index, crop)),
            Err(error) => {
                failed_images.insert(image_index);
                failures.push((image_index, error));
            }
        }
    }
    crops.retain(|(work_index, _)| !failed_images.contains(&work[*work_index].image_index));
    let crop_preparation = crop_started.elapsed();
    let tensor_started = Instant::now();
    let input = if crops.is_empty() {
        None
    } else {
        let images = crops.iter().map(|(_, crop)| crop).collect::<Vec<_>>();
        Some(recognition_input_with_canvas_width(
            &images,
            config,
            canvas_width,
        ))
    };
    let tensor_preparation = tensor_started.elapsed();
    PreparedRecognitionBatch {
        input_slots: crops.len(),
        crops,
        canvas_width,
        input,
        failures,
        crop_preparation,
        tensor_preparation,
    }
}

fn reconcile_prepared_batch(
    prepared: &mut PreparedRecognitionBatch,
    work: &[RecognitionWorkItem],
    states: &mut [ImageRecognition],
    failure_authority: &[bool],
    config: &RecognitionConfig,
    timings: Option<&mut RecognitionTimings>,
) {
    let mut additional_tensor_preparation = Duration::ZERO;
    for (image_index, error) in prepared.failures.drain(..) {
        if failure_authority.get(image_index).copied().unwrap_or(false) {
            states[image_index].fail(error);
        }
    }
    prepared
        .crops
        .retain(|(work_index, _)| states[work[*work_index].image_index].is_pending());
    if prepared.crops.is_empty() {
        prepared.input = None;
        prepared.input_slots = 0;
    } else if prepared.crops.len() != prepared.input_slots {
        let started = Instant::now();
        let images = prepared
            .crops
            .iter()
            .map(|(_, crop)| crop)
            .collect::<Vec<_>>();
        prepared.input = Some(recognition_input_with_canvas_width(
            &images,
            config,
            prepared.canvas_width,
        ));
        prepared.input_slots = prepared.crops.len();
        additional_tensor_preparation = started.elapsed();
    }
    if let Some(timings) = timings {
        timings.crop_preparation += prepared.crop_preparation;
        timings.tensor_preparation += prepared.tensor_preparation + additional_tensor_preparation;
    }
}

fn reconcile_prepared_window(
    prepared: &mut [PreparedRecognitionBatch],
    work: &[RecognitionWorkItem],
    states: &mut [ImageRecognition],
    config: &RecognitionConfig,
    mut timings: Option<&mut RecognitionTimings>,
) {
    // A successor can be prepared before its predecessor publishes failures.
    // Only images that were still pending at this window boundary may acquire
    // a preparation failure. Within the window, canonical batch order retains
    // the existing last-observed preparation error behavior.
    let failure_authority = states
        .iter()
        .map(ImageRecognition::is_pending)
        .collect::<Vec<_>>();
    for batch in prepared {
        reconcile_prepared_batch(
            batch,
            work,
            states,
            &failure_authority,
            config,
            timings.as_deref_mut(),
        );
    }
}

#[cfg(test)]
fn prepare_batch_crops(
    images: &[&RgbImage],
    work: &[RecognitionWorkItem],
    active: Vec<usize>,
    states: &mut [ImageRecognition],
) -> Vec<(usize, RgbImage)> {
    let prepared = active
        .into_par_iter()
        .map(|work_index| {
            let item = &work[work_index];
            (work_index, item.crop.execute(images[item.image_index]))
        })
        .collect::<Vec<_>>();
    let mut crops = Vec::with_capacity(prepared.len());
    for (work_index, crop) in prepared {
        let item = &work[work_index];
        match crop {
            Ok(crop) => crops.push((work_index, crop)),
            Err(error) => states[item.image_index].fail(error),
        }
    }
    crops.retain(|(work_index, _)| states[work[*work_index].image_index].is_pending());
    crops
}

fn apply_recognized_batch(
    work_indices: &[usize],
    recognized: RecognizedBatch,
    work: &[RecognitionWorkItem],
    states: &mut [ImageRecognition],
) -> UseResult<()> {
    if work_indices.len() != recognized.items.len() {
        return Err(engine_error(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 changed recognition batch result cardinality.",
        ));
    }
    let mut receipt_images = Vec::with_capacity(work_indices.len());
    for (work_index, recognition) in work_indices.iter().zip(recognized.items) {
        let item = &work[*work_index];
        if !states[item.image_index].is_pending() {
            continue;
        }
        match recognition {
            Ok(recognition) => {
                states[item.image_index].set_block(
                    item.detection_index,
                    EngineBlock {
                        polygon: item.detection.polygon,
                        text_rotation_millidegrees: item.crop.text_rotation_millidegrees(),
                        detection_confidence: item.detection.confidence,
                        text: recognition.text,
                        confidence: recognition.confidence,
                    },
                )?;
                if !receipt_images.contains(&item.image_index) {
                    receipt_images.push(item.image_index);
                }
            }
            Err(error) => states[item.image_index].fail(error),
        }
    }
    for image_index in receipt_images {
        if let ImageRecognition::Pending { receipts, .. } = &mut states[image_index] {
            receipts.push(recognized.receipt.clone());
        }
    }
    Ok(())
}

fn detection_intersects_window(detection: &Detection, window: OcrPixelWindow) -> bool {
    // This conservative source-geometry predicate is exhaustive and content-
    // independent. Boundary contact without positive area does not select a
    // block; a crossing block is recognized in full and is never clipped.
    let mut left = f32::INFINITY;
    let mut top = f32::INFINITY;
    let mut right = f32::NEG_INFINITY;
    let mut bottom = f32::NEG_INFINITY;
    for point in detection.polygon {
        if !point.x.is_finite() || !point.y.is_finite() {
            return false;
        }
        left = left.min(point.x);
        top = top.min(point.y);
        right = right.max(point.x);
        bottom = bottom.max(point.y);
    }

    left.max(window.left as f32) < right.min(window.right as f32)
        && top.max(window.top as f32) < bottom.min(window.bottom as f32)
}

impl ImageRecognition {
    fn is_pending(&self) -> bool {
        matches!(self, Self::Pending { .. })
    }

    fn fail(&mut self, error: UseError) {
        *self = Self::Failed(error);
    }

    fn set_block(&mut self, index: usize, block: EngineBlock) -> UseResult<()> {
        let Self::Pending {
            blocks, selected, ..
        } = self
        else {
            return Ok(());
        };
        if !selected.get(index).copied().ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 recognition selection index escaped its source image.",
            )
        })? {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 returned recognition output for an unselected detection.",
            ));
        }
        let target = blocks.get_mut(index).ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 recognition block index escaped its source image.",
            )
        })?;
        *target = Some(block);
        Ok(())
    }

    fn finish(self, model: &'static str) -> UseResult<EngineExtraction> {
        match self {
            Self::Failed(error) => Err(error),
            Self::Pending {
                blocks,
                selected,
                receipts,
            } => {
                if blocks.len() != selected.len() {
                    return Err(engine_error(
                        "use.ocr.provider_output_invalid",
                        "PP-OCRv6 recognition selection changed detection cardinality.",
                    ));
                }
                let mut selected_blocks =
                    Vec::with_capacity(selected.iter().filter(|is_selected| **is_selected).count());
                for (block, is_selected) in blocks.into_iter().zip(selected) {
                    if !is_selected {
                        continue;
                    }
                    selected_blocks.push(block.ok_or_else(|| {
                        engine_error(
                            "use.ocr.provider_output_invalid",
                            "PP-OCRv6 left a selected recognition block unresolved.",
                        )
                    })?);
                }
                Ok(EngineExtraction {
                    model,
                    blocks: selected_blocks,
                    receipts,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests;
