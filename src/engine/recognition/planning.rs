use a3s_use_core::UseResult;

use super::super::engine_error;

const RECOGNITION_BATCH_SIZE: usize = 8;

pub(super) fn plan_width_batches(canvas_widths: &[u32]) -> UseResult<Vec<Vec<usize>>> {
    if canvas_widths.contains(&0) {
        return Err(engine_error(
            "use.ocr.image_invalid",
            "PP-OCRv6 recognition canvas widths must be positive.",
        ));
    }
    if canvas_widths.is_empty() {
        return Ok(Vec::new());
    }

    // For a fixed maximum batch size, sorting by dynamic canvas width before
    // contiguous chunking minimizes padded columns without adding graph calls.
    let mut sorted_indices = (0..canvas_widths.len()).collect::<Vec<_>>();
    sorted_indices.sort_by_key(|index| canvas_widths[*index]);

    Ok(sorted_indices
        .chunks(RECOGNITION_BATCH_SIZE)
        .map(<[usize]>::to_vec)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_batches_stably_sort_compatible_crops() {
        let batches = plan_width_batches(&[400, 320, 384, 320]).unwrap();

        assert_eq!(batches, vec![vec![1, 3, 2, 0]]);
    }

    #[test]
    fn width_batches_put_wider_crops_last_without_adding_launches() {
        let batches = plan_width_batches(&[1_024, 320, 1_050, 352]).unwrap();

        assert_eq!(batches, vec![vec![1, 3, 0, 2]]);
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
