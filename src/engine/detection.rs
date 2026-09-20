use std::ops::Range;
use std::time::{Duration, Instant};

use a3s_power::inference::{ExecutionPermit, ExecutionReceipt, RuntimeDeviceKind, TensorOutput};
use a3s_use_core::UseResult;
use image::RgbImage;
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use super::scheduling::{
    available_execution_workers, can_append_execution_job, cpu_execution_window_policy,
};
use super::{engine_error, PpOcrV6Engine};
use crate::cancellation::check_cancelled;
use crate::config::DetectionConfig;
use crate::postprocess::{detection_boxes_in_content, Detection};
use crate::preprocess::{
    detection_batch_input, detection_input_with_max_side, DetectionGeometry,
    DETECTION_QUALITY_MAX_SIDE,
};

const QUALITY_RETRY_MIN_CHANNEL_RANGE: u8 = 32;

mod planning;

use planning::detection_cohort_peak_elements;
pub(crate) use planning::detection_cohort_ranges;

pub(super) struct DetectedBatch {
    pub(super) detections: Vec<UseResult<Vec<Detection>>>,
    pub(super) receipts: Vec<Vec<ExecutionReceipt>>,
    pub(super) timings: DetectionTimings,
}

#[derive(Default)]
pub(super) struct DetectionTimings {
    pub(super) cohorts: usize,
    pub(super) preprocessing: Duration,
    pub(super) inference: Duration,
    pub(super) postprocessing: Duration,
    pub(super) retry: Duration,
    pub(super) execution_wall: Duration,
    pub(super) maximum_parallel_cohorts: usize,
}

struct DetectedCohort {
    detections: Vec<UseResult<Vec<Detection>>>,
    receipts: Vec<Vec<ExecutionReceipt>>,
    timings: DetectionTimings,
}

enum DetectionLanePermit<'a> {
    Primary(&'a ExecutionPermit),
    Helper(ExecutionPermit),
}

impl DetectionLanePermit<'_> {
    fn as_ref(&self) -> &ExecutionPermit {
        match self {
            Self::Primary(permit) => permit,
            Self::Helper(permit) => permit,
        }
    }
}

struct DetectionLane<'a> {
    engine: &'a PpOcrV6Engine,
    permit: DetectionLanePermit<'a>,
}

pub(super) fn postprocess_batch(
    inputs: Vec<DetectionGeometry>,
    tensor: TensorOutput,
    config: &DetectionConfig,
) -> UseResult<Vec<UseResult<Vec<Detection>>>> {
    if inputs.is_empty()
        || tensor.shape.len() != 4
        || tensor.shape[0] != inputs.len()
        || tensor.shape[1] != 1
    {
        return Err(engine_error(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 detection postprocessing received invalid batch cardinality.",
        ));
    }
    let slot_elements = tensor.shape[1..]
        .iter()
        .try_fold(1_usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection output dimensions overflowed.",
            )
        })?;
    if slot_elements == 0
        || slot_elements
            .checked_mul(inputs.len())
            .is_none_or(|expected| expected != tensor.values.len())
    {
        return Err(engine_error(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 detection output length does not match its batch shape.",
        ));
    }
    let slot_shape = [1, tensor.shape[1], tensor.shape[2], tensor.shape[3]];
    if inputs.len() == 1 {
        let input = inputs.into_iter().next().ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection postprocessing lost its scalar input.",
            )
        })?;
        return Ok(vec![postprocess_one(
            input,
            &tensor.values,
            &slot_shape,
            config,
        )]);
    }
    std::thread::scope(|scope| {
        let workers = inputs
            .into_iter()
            .zip(tensor.values.chunks_exact(slot_elements))
            .map(|(input, values)| {
                scope.spawn(move || postprocess_one(input, values, &slot_shape, config))
            })
            .collect::<Vec<_>>();
        let completed = workers
            .into_iter()
            .map(|worker| worker.join())
            .collect::<Vec<_>>();
        Ok(completed
            .into_iter()
            .map(|result| match result {
                Ok(detections) => detections,
                Err(_) => Err(engine_error(
                    "use.ocr.runtime_failed",
                    "PP-OCRv6 detection postprocessing worker failed.",
                )),
            })
            .collect())
    })
}

fn postprocess_one(
    input: DetectionGeometry,
    values: &[f32],
    shape: &[usize],
    config: &DetectionConfig,
) -> UseResult<Vec<Detection>> {
    detection_boxes_in_content(
        values,
        shape,
        input.content_width,
        input.content_height,
        input.original_width,
        input.original_height,
        config,
    )
}

impl PpOcrV6Engine {
    #[cfg(test)]
    pub(super) fn detect_cohorts(
        &self,
        images: &[&RgbImage],
        max_tensor_elements: usize,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<DetectedBatch> {
        self.detect_cohorts_with_helpers(images, max_tensor_elements, permit, cancellation, &[])
    }

    pub(super) fn detect_cohorts_with_helpers(
        &self,
        images: &[&RgbImage],
        max_tensor_elements: usize,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        helpers: &[&Self],
    ) -> UseResult<DetectedBatch> {
        let ranges = execution_cohort_ranges(
            detection_cohort_ranges(images, max_tensor_elements)?,
            self.native.runtime_device_kind(),
        );
        let jobs = ranges
            .into_iter()
            .map(|range| {
                let reservation = detection_cohort_peak_elements(&images[range.clone()])?;
                Ok((range, reservation))
            })
            .collect::<UseResult<Vec<_>>>()?;
        let base_policy = cpu_execution_window_policy(
            self.native.runtime_device_kind(),
            available_execution_workers(),
            self.native.maximum_tensor_elements(),
        );
        let mut detections = Vec::with_capacity(images.len());
        let mut receipts = Vec::with_capacity(images.len());
        let mut timings = DetectionTimings::default();
        let mut start = 0_usize;
        while start < jobs.len() {
            check_cancelled(cancellation)?;
            let lanes = self.detection_lanes(
                permit,
                helpers,
                jobs.len().saturating_sub(start),
                cancellation,
            );
            let mut policy = base_policy;
            if self.native.runtime_device_kind() != RuntimeDeviceKind::Cpu {
                policy.maximum_parallel_jobs = lanes.len().max(1);
            }
            let mut end = start;
            let mut reserved_elements = 0_usize;
            while let Some((_, next_reservation)) = jobs.get(end) {
                if !can_append_execution_job(
                    end - start,
                    reserved_elements,
                    *next_reservation,
                    policy,
                ) {
                    break;
                }
                reserved_elements = reserved_elements.saturating_add(*next_reservation);
                end += 1;
            }
            let window = &jobs[start..end];
            let execution_started = Instant::now();
            let completed = if window.len() == 1 {
                vec![self.detect_cohort(&images[window[0].0.clone()], permit, cancellation)]
            } else if self.native.runtime_device_kind() != RuntimeDeviceKind::Cpu {
                lanes
                    .into_par_iter()
                    .zip(window.par_iter())
                    .map(|(lane, (range, _))| {
                        lane.engine.detect_cohort(
                            &images[range.clone()],
                            lane.permit.as_ref(),
                            cancellation,
                        )
                    })
                    .collect::<Vec<_>>()
            } else {
                window
                    .par_iter()
                    .map(|(range, _)| {
                        self.detect_cohort(&images[range.clone()], permit, cancellation)
                    })
                    .collect::<Vec<_>>()
            };
            timings.execution_wall += execution_started.elapsed();
            timings.maximum_parallel_cohorts = timings.maximum_parallel_cohorts.max(window.len());
            check_cancelled(cancellation)?;
            for ((range, _), completed) in window.iter().zip(completed) {
                match completed {
                    Ok(detected) => {
                        timings.include(&detected.timings);
                        detections.extend(detected.detections);
                        receipts.extend(detected.receipts);
                    }
                    Err(error) => {
                        timings.cohorts += 1;
                        detections.extend((0..range.len()).map(|_| Err(error.clone())));
                        receipts.extend((0..range.len()).map(|_| Vec::new()));
                    }
                }
            }
            start = end;
        }
        if detections.len() != images.len() || receipts.len() != images.len() {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection cohorts changed exact batch cardinality.",
            ));
        }
        Ok(DetectedBatch {
            detections,
            receipts,
            timings,
        })
    }

    fn detection_lanes<'a>(
        &'a self,
        permit: &'a ExecutionPermit,
        helpers: &[&'a Self],
        remaining_jobs: usize,
        cancellation: &CancellationToken,
    ) -> Vec<DetectionLane<'a>> {
        let mut lanes = vec![DetectionLane {
            engine: self,
            permit: DetectionLanePermit::Primary(permit),
        }];
        if self.native.runtime_device_kind() == RuntimeDeviceKind::Cpu || remaining_jobs < 2 {
            return lanes;
        }
        for helper in helpers {
            if lanes.iter().any(|lane| std::ptr::eq(lane.engine, *helper)) {
                continue;
            }
            if let Ok(helper_permit) = helper.native.begin(cancellation) {
                lanes.push(DetectionLane {
                    engine: helper,
                    permit: DetectionLanePermit::Helper(helper_permit),
                });
            }
        }
        lanes
    }

    fn detect_cohort(
        &self,
        images: &[&RgbImage],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<DetectedCohort> {
        let started = Instant::now();
        let input = detection_batch_input(images, &self.detection_config)?;
        let preprocessed = started.elapsed();
        let detection = self
            .native
            .detect_batch(input.data, input.shape, permit, cancellation)?;
        let inferred = started.elapsed();
        if detection.tensor.shape.first() != Some(&images.len())
            || input.geometries.len() != images.len()
        {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 detection changed exact cohort cardinality.",
            ));
        }
        let mut detections =
            postprocess_batch(input.geometries, detection.tensor, &self.detection_config)?;
        check_cancelled(cancellation)?;
        let postprocessed = started.elapsed();
        let mut receipts = vec![vec![detection.receipt]; images.len()];
        self.retry_empty_detections(images, &mut detections, &mut receipts, permit, cancellation)?;
        let retried = started.elapsed();
        Ok(DetectedCohort {
            detections,
            receipts,
            timings: DetectionTimings {
                cohorts: 1,
                preprocessing: preprocessed,
                inference: inferred - preprocessed,
                postprocessing: postprocessed - inferred,
                retry: retried - postprocessed,
                execution_wall: Duration::ZERO,
                maximum_parallel_cohorts: 0,
            },
        })
    }

    pub(super) fn retry_empty_detections(
        &self,
        images: &[&RgbImage],
        detections: &mut [UseResult<Vec<Detection>>],
        receipts: &mut [Vec<ExecutionReceipt>],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<()> {
        if detections.len() != images.len() || receipts.len() != images.len() {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 quality retry received mismatched batch cardinality.",
            ));
        }
        for index in 0..images.len() {
            check_cancelled(cancellation)?;
            if !should_retry_for_quality(images[index], &detections[index]) {
                continue;
            }
            match self.detect_one_for_quality(images[index], permit, cancellation) {
                Ok((quality_detections, receipt)) => {
                    detections[index] = Ok(quality_detections);
                    receipts[index].push(receipt);
                }
                Err(error) => detections[index] = Err(error),
            }
        }
        Ok(())
    }

    fn detect_one_for_quality(
        &self,
        image: &RgbImage,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<(Vec<Detection>, ExecutionReceipt)> {
        let input = detection_input_with_max_side(
            image,
            &self.detection_config,
            DETECTION_QUALITY_MAX_SIDE,
        )?;
        let output = self
            .native
            .detect_batch(input.data, input.shape, permit, cancellation)?;
        let detections = detection_boxes_in_content(
            &output.tensor.values,
            &output.tensor.shape,
            input.geometry.content_width,
            input.geometry.content_height,
            input.geometry.original_width,
            input.geometry.original_height,
            &self.detection_config,
        )?;
        Ok((detections, output.receipt))
    }
}

fn execution_cohort_ranges(
    ranges: Vec<Range<usize>>,
    device: RuntimeDeviceKind,
) -> Vec<Range<usize>> {
    if device != RuntimeDeviceKind::Cpu {
        return ranges;
    }
    ranges
        .into_iter()
        .flat_map(|range| range.map(|index| index..index + 1))
        .collect()
}

impl DetectionTimings {
    fn include(&mut self, other: &Self) {
        self.cohorts += other.cohorts;
        self.preprocessing += other.preprocessing;
        self.inference += other.inference;
        self.postprocessing += other.postprocessing;
        self.retry += other.retry;
    }
}

fn should_retry_for_quality(image: &RgbImage, detections: &UseResult<Vec<Detection>>) -> bool {
    detections.as_ref().is_ok_and(Vec::is_empty) && image_has_visual_variation(image)
}

fn image_has_visual_variation(image: &RgbImage) -> bool {
    let mut minimum = [u8::MAX; 3];
    let mut maximum = [u8::MIN; 3];
    for pixel in image.as_raw().chunks_exact(3) {
        for channel in 0..3 {
            minimum[channel] = minimum[channel].min(pixel[channel]);
            maximum[channel] = maximum[channel].max(pixel[channel]);
        }
        if (0..3).any(|channel| {
            maximum[channel].saturating_sub(minimum[channel]) >= QUALITY_RETRY_MIN_CHANNEL_RANGE
        }) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DetectionConfig {
        DetectionConfig {
            model_variant: crate::config::ModelVariant::Small,
            scale: 1.0 / 255.0,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            threshold: 0.3,
            box_threshold: 0.6,
            max_candidates: 1_000,
            unclip_ratio: 1.5,
        }
    }

    #[test]
    fn cpu_detection_executes_independent_pages_while_accelerators_keep_batches() {
        let planned = vec![0..4, 4..6];

        assert_eq!(
            execution_cohort_ranges(planned.clone(), RuntimeDeviceKind::Cpu),
            vec![0..1, 1..2, 2..3, 3..4, 4..5, 5..6]
        );
        assert_eq!(
            execution_cohort_ranges(planned.clone(), RuntimeDeviceKind::Cuda),
            planned
        );
    }

    fn detection_geometry(side: usize) -> DetectionGeometry {
        DetectionGeometry {
            original_width: side as u32,
            original_height: side as u32,
            content_width: side as u32,
            content_height: side as u32,
        }
    }

    #[test]
    fn visual_variation_distinguishes_blank_and_marked_pages() {
        let blank = RgbImage::from_pixel(64, 64, image::Rgb([248, 248, 248]));
        let mut marked = blank.clone();
        marked.put_pixel(31, 17, image::Rgb([0, 0, 0]));
        let mut slight_noise = blank.clone();
        slight_noise.put_pixel(31, 17, image::Rgb([225, 225, 225]));

        assert!(!image_has_visual_variation(&blank));
        assert!(image_has_visual_variation(&marked));
        assert!(!image_has_visual_variation(&slight_noise));
    }

    #[test]
    fn quality_retry_requires_an_empty_successful_detection_on_a_marked_page() {
        let mut marked = RgbImage::from_pixel(64, 64, image::Rgb([248, 248, 248]));
        marked.put_pixel(31, 17, image::Rgb([0, 0, 0]));
        let blank = RgbImage::from_pixel(64, 64, image::Rgb([248, 248, 248]));
        let detection = Detection {
            polygon: [imageproc::point::Point::new(0.0, 0.0); 4],
            confidence: 1.0,
        };

        assert!(should_retry_for_quality(&marked, &Ok(Vec::new())));
        assert!(!should_retry_for_quality(&blank, &Ok(Vec::new())));
        assert!(!should_retry_for_quality(&marked, &Ok(vec![detection])));
        assert!(!should_retry_for_quality(
            &marked,
            &Err(engine_error("use.ocr.fixture", "failed detection")),
        ));
    }

    #[test]
    fn batch_postprocessing_preserves_cardinality_and_isolates_slots() {
        let tensor = TensorOutput {
            shape: vec![2, 1, 32, 32],
            values: vec![0.0; 2 * 32 * 32],
        };

        let detections = postprocess_batch(
            vec![detection_geometry(32), detection_geometry(64)],
            tensor,
            &config(),
        )
        .unwrap();

        assert_eq!(detections.len(), 2);
        assert!(detections[0].as_ref().unwrap().is_empty());
        assert!(detections[1].is_err());
        assert!(postprocess_batch(
            Vec::new(),
            TensorOutput {
                shape: vec![0, 1, 32, 32],
                values: Vec::new(),
            },
            &config()
        )
        .is_err());
    }
}
