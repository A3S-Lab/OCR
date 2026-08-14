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

    // The recognition graph has global context across its dynamic width, so
    // padding a crop to a wider peer's canvas can change decoded text. Group
    // only identical canvas widths to preserve scalar preprocessing semantics.
    let mut sorted_indices = (0..canvas_widths.len()).collect::<Vec<_>>();
    sorted_indices.sort_by_key(|index| canvas_widths[*index]);
    let mut batches = Vec::new();
    let mut start = 0;
    while start < sorted_indices.len() {
        let width = canvas_widths[sorted_indices[start]];
        let mut end = start + 1;
        while end < sorted_indices.len() && canvas_widths[sorted_indices[end]] == width {
            end += 1;
        }
        batches.extend(
            sorted_indices[start..end]
                .chunks(RECOGNITION_BATCH_SIZE)
                .map(<[usize]>::to_vec),
        );
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

        assert_eq!(batches, vec![vec![1, 3], vec![2], vec![0]]);
    }

    #[test]
    fn width_batches_never_mix_dynamic_canvas_widths() {
        let batches = plan_width_batches(&[1_024, 320, 1_050, 352]).unwrap();

        assert_eq!(batches, vec![vec![1], vec![3], vec![0], vec![2]]);
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
