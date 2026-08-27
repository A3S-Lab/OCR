use a3s_power::inference::RuntimeDeviceKind;
use a3s_use_core::UseResult;

use super::super::engine_error;
use crate::preprocess::recognition_max_batch_size;

const RECOGNITION_CANONICAL_BATCH_SIZE: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RecognitionWindowReservation {
    pub(super) elements: usize,
    pub(super) input_bytes: usize,
}

pub(super) fn window_reservations_fit_lanes(
    reservations: &[RecognitionWindowReservation],
    lane_count: usize,
    maximum_elements_per_lane: usize,
    maximum_input_bytes_per_lane: usize,
) -> bool {
    if lane_count == 0 {
        return reservations.is_empty();
    }
    let mut lane_elements = vec![0_usize; lane_count];
    let mut lane_input_bytes = vec![0_usize; lane_count];
    for reservation in reservations {
        let lane_index = lane_elements
            .iter()
            .enumerate()
            .min_by_key(|(lane_index, elements)| (**elements, *lane_index))
            .map(|(lane_index, _)| lane_index)
            .unwrap_or(0);
        let Some(elements) = lane_elements[lane_index].checked_add(reservation.elements) else {
            return false;
        };
        let Some(input_bytes) = lane_input_bytes[lane_index].checked_add(reservation.input_bytes)
        else {
            return false;
        };
        if elements > maximum_elements_per_lane || input_bytes > maximum_input_bytes_per_lane {
            return false;
        }
        lane_elements[lane_index] = elements;
        lane_input_bytes[lane_index] = input_bytes;
    }
    true
}

pub(super) fn plan_width_batches(
    canvas_widths: &[u32],
    device: RuntimeDeviceKind,
    maximum_batch_size_for_width: impl Fn(u32) -> UseResult<usize>,
) -> UseResult<Vec<Vec<usize>>> {
    if canvas_widths.contains(&0) {
        return Err(engine_error(
            "use.ocr.image_invalid",
            "PP-OCRv6 recognition canvas widths must be positive.",
        ));
    }
    if canvas_widths.is_empty() {
        return Ok(Vec::new());
    }

    // The recognition graph has global context across its dynamic width, so
    // even bounded right padding can change decoded text. Only identical
    // input tensor widths may share one inference batch.
    let mut sorted_indices = (0..canvas_widths.len()).collect::<Vec<_>>();
    sorted_indices.sort_by_key(|index| canvas_widths[*index]);
    let mut canonical = Vec::new();
    let mut maximum_batch_sizes = std::collections::BTreeMap::new();
    let mut start = 0;
    while start < sorted_indices.len() {
        let minimum_width = canvas_widths[sorted_indices[start]];
        let resource_limit = maximum_batch_size_for_width(minimum_width)?.max(1);
        let maximum_batch_size = match device {
            // Independent scalar graphs expose more useful outer parallelism on
            // a CPU than a single NCHW batch. The split preserves each crop's
            // exact tensor width and changes neither pixels nor model arithmetic.
            RuntimeDeviceKind::Cpu => 1,
            RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal => {
                recognition_max_batch_size().min(resource_limit)
            }
        };
        maximum_batch_sizes.insert(minimum_width, maximum_batch_size);
        let canonical_batch_size = RECOGNITION_CANONICAL_BATCH_SIZE.min(maximum_batch_size);
        let mut end = start + 1;
        while end < sorted_indices.len()
            && end - start < canonical_batch_size
            && canvas_widths[sorted_indices[end]] == minimum_width
        {
            end += 1;
        }
        canonical.push(sorted_indices[start..end].to_vec());
        start = end;
    }
    Ok(coalesce_equal_canvas_batches(
        canonical,
        canvas_widths,
        &maximum_batch_sizes,
    ))
}

fn coalesce_equal_canvas_batches(
    canonical: Vec<Vec<usize>>,
    canvas_widths: &[u32],
    maximum_batch_sizes: &std::collections::BTreeMap<u32, usize>,
) -> Vec<Vec<usize>> {
    let mut batches: Vec<Vec<usize>> = Vec::with_capacity(canonical.len());
    for batch in canonical {
        let canvas_width = batch
            .iter()
            .map(|index| canvas_widths[*index])
            .max()
            .unwrap_or_default();
        let maximum_batch_size = maximum_batch_sizes.get(&canvas_width).copied().unwrap_or(1);
        let can_coalesce = batches.last().is_some_and(|previous| {
            previous.len() + batch.len() <= maximum_batch_size
                && previous.iter().map(|index| canvas_widths[*index]).max() == Some(canvas_width)
        });
        if can_coalesce {
            if let Some(previous) = batches.last_mut() {
                previous.extend(batch);
                continue;
            }
        }
        batches.push(batch);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unbounded(_: u32) -> UseResult<usize> {
        Ok(usize::MAX)
    }

    #[test]
    fn width_batches_stably_sort_compatible_crops() {
        let batches =
            plan_width_batches(&[400, 320, 384, 320], RuntimeDeviceKind::Cuda, unbounded).unwrap();

        assert_eq!(batches, vec![vec![1, 3], vec![2], vec![0]]);
    }

    #[test]
    fn exact_width_batches_separate_every_width_difference() {
        let batches = plan_width_batches(
            &[1_024, 320, 1_050, 352],
            RuntimeDeviceKind::Cuda,
            unbounded,
        )
        .unwrap();

        assert_eq!(batches, vec![vec![1], vec![3], vec![0], vec![2]]);
    }

    #[test]
    fn different_canvas_widths_never_share_an_inference_batch() {
        let batches =
            plan_width_batches(&[320, 335, 336, 352], RuntimeDeviceKind::Cuda, unbounded).unwrap();

        assert_eq!(batches, vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn width_batches_never_exceed_the_reviewed_graph_limit() {
        let limit = recognition_max_batch_size();
        let batches =
            plan_width_batches(&vec![320; limit + 1], RuntimeDeviceKind::Cuda, unbounded).unwrap();

        assert_eq!(batches, vec![(0..limit).collect::<Vec<_>>(), vec![limit]]);
    }

    #[test]
    fn coalescing_preserves_each_canonical_input_canvas() {
        let limit = recognition_max_batch_size();
        let widths = std::iter::repeat_n(320, limit + 1)
            .chain([321; 8])
            .collect::<Vec<_>>();
        let batches = plan_width_batches(&widths, RuntimeDeviceKind::Cuda, unbounded).unwrap();

        assert_eq!(
            batches,
            vec![
                (0..limit).collect::<Vec<_>>(),
                vec![limit],
                (limit + 1..limit + 9).collect::<Vec<_>>()
            ]
        );
        assert_eq!(
            batches
                .iter()
                .map(|batch| batch.iter().map(|index| widths[*index]).max().unwrap())
                .collect::<Vec<_>>(),
            vec![320, 320, 321]
        );
    }

    #[test]
    fn width_specific_resource_limits_can_only_reduce_accelerator_batches() {
        let batches = plan_width_batches(&[320; 25], RuntimeDeviceKind::Cuda, |_| Ok(16)).unwrap();

        assert_eq!(
            batches,
            vec![(0..16).collect::<Vec<_>>(), (16..25).collect::<Vec<_>>()]
        );
    }

    #[test]
    fn width_batches_accept_no_crops_and_reject_zero_width() {
        assert!(plan_width_batches(&[], RuntimeDeviceKind::Cpu, unbounded)
            .unwrap()
            .is_empty());
        assert_eq!(
            plan_width_batches(&[320, 0], RuntimeDeviceKind::Cpu, unbounded)
                .unwrap_err()
                .code,
            "use.ocr.image_invalid"
        );
    }

    #[test]
    fn cpu_plans_independent_exact_width_graphs() {
        let batches =
            plan_width_batches(&[400, 320, 320, 384], RuntimeDeviceKind::Cpu, unbounded).unwrap();

        assert_eq!(batches, vec![vec![1], vec![2], vec![3], vec![0]]);
    }

    #[test]
    fn independent_window_lanes_each_retain_their_own_resource_budget() {
        let jobs = [
            RecognitionWindowReservation {
                elements: 80,
                input_bytes: 40,
            },
            RecognitionWindowReservation {
                elements: 80,
                input_bytes: 40,
            },
            RecognitionWindowReservation {
                elements: 80,
                input_bytes: 40,
            },
        ];

        assert!(!window_reservations_fit_lanes(&jobs, 1, 100, 64));
        assert!(window_reservations_fit_lanes(&jobs, 3, 100, 64));
        assert!(!window_reservations_fit_lanes(
            &[
                jobs[0],
                jobs[1],
                jobs[2],
                RecognitionWindowReservation {
                    elements: 21,
                    input_bytes: 1,
                },
            ],
            3,
            100,
            64,
        ));
    }

    #[test]
    fn window_lane_assignment_enforces_input_bytes_independently_of_elements() {
        let jobs = [
            RecognitionWindowReservation {
                elements: 40,
                input_bytes: 60,
            },
            RecognitionWindowReservation {
                elements: 40,
                input_bytes: 60,
            },
            RecognitionWindowReservation {
                elements: 20,
                input_bytes: 5,
            },
        ];

        assert!(!window_reservations_fit_lanes(&jobs, 2, 100, 64));
        assert!(window_reservations_fit_lanes(&jobs[..2], 2, 100, 64));
        assert!(window_reservations_fit_lanes(&[], 0, 100, 64));
        assert!(!window_reservations_fit_lanes(&jobs[..1], 0, 100, 64));
    }
}
