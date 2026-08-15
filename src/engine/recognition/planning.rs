use a3s_use_core::UseResult;

use super::super::engine_error;

const RECOGNITION_BATCH_SIZE: usize = 8;
// Recognition canvases are never narrower than 320 pixels, so this admits at
// most 5% right padding while collapsing pixel-level crop-width jitter. The
// bound is covered by exact-text Parser fixtures and remains far below the
// unbounded mixed-width experiment that failed the model quality gate.
const RECOGNITION_MAX_PADDING: u32 = 16;

pub(super) fn plan_width_batches(canvas_widths: &[u32]) -> UseResult<Vec<Vec<usize>>> {
    plan_width_batches_with_padding(canvas_widths, RECOGNITION_MAX_PADDING)
}

pub(super) fn plan_width_batches_with_padding(
    canvas_widths: &[u32],
    maximum_padding: u32,
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
    // padding a crop to a wider peer's canvas can change decoded text. The
    // released planner applies only the reviewed bound above. The explicit
    // parameter keeps planning tests able to prove both exact-width and
    // bounded-padding behavior without introducing an unbounded mixed-width
    // path.
    let mut sorted_indices = (0..canvas_widths.len()).collect::<Vec<_>>();
    sorted_indices.sort_by_key(|index| canvas_widths[*index]);
    let mut batches = Vec::new();
    let mut start = 0;
    while start < sorted_indices.len() {
        let minimum_width = canvas_widths[sorted_indices[start]];
        let maximum_width = minimum_width.saturating_add(maximum_padding);
        let mut end = start + 1;
        while end < sorted_indices.len()
            && end - start < RECOGNITION_BATCH_SIZE
            && canvas_widths[sorted_indices[end]] <= maximum_width
        {
            end += 1;
        }
        batches.push(sorted_indices[start..end].to_vec());
        start = end;
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_batches_stably_sort_compatible_crops() {
        let batches = plan_width_batches(&[400, 320, 384, 320]).unwrap();

        assert_eq!(batches, vec![vec![1, 3], vec![2, 0]]);
    }

    #[test]
    fn released_width_batches_separate_width_deltas_above_bound() {
        let batches = plan_width_batches(&[1_024, 320, 1_050, 352]).unwrap();

        assert_eq!(batches, vec![vec![1], vec![3], vec![0], vec![2]]);
    }

    #[test]
    fn released_width_batches_mix_only_within_the_reviewed_padding_bound() {
        let batches = plan_width_batches(&[320, 335, 336, 352]).unwrap();

        assert_eq!(batches, vec![vec![0, 1, 2], vec![3]]);
    }

    #[test]
    fn bounded_padding_batches_never_escape_the_declared_width_delta() {
        let batches = plan_width_batches_with_padding(&[384, 320, 335, 352, 336], 16).unwrap();

        assert_eq!(batches, vec![vec![1, 2, 4], vec![3], vec![0]]);
        assert!(batches.iter().all(|batch| {
            let minimum = batch
                .iter()
                .map(|index| [384, 320, 335, 352, 336][*index])
                .min()
                .unwrap();
            let maximum = batch
                .iter()
                .map(|index| [384, 320, 335, 352, 336][*index])
                .max()
                .unwrap();
            maximum - minimum <= 16
        }));
    }

    #[test]
    fn width_batches_never_exceed_the_reviewed_graph_limit() {
        let batches = plan_width_batches(&[320; 17]).unwrap();

        assert_eq!(
            batches,
            vec![
                (0..8).collect::<Vec<_>>(),
                (8..16).collect::<Vec<_>>(),
                vec![16]
            ]
        );
    }

    #[test]
    fn width_batches_accept_no_crops_and_reject_zero_width() {
        assert!(plan_width_batches(&[]).unwrap().is_empty());
        assert_eq!(
            plan_width_batches(&[320, 0]).unwrap_err().code,
            "use.ocr.image_invalid"
        );
    }
}
