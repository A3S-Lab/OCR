use std::cmp::Ordering;

use a3s_use_core::{UseError, UseResult};

use super::profile::{
    LayoutClass, CLASSES, CLASS_COUNT, INPUT_SIDE, KEEP_TOP_K, LOCATION_COUNT, NMS_IOU_THRESHOLD,
    NMS_TOP_K, OUTPUT_WIDTH, SCORE_THRESHOLD,
};
use crate::document_fast::wired::PixelRect;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::document_fast) struct DetectedLayoutRegion {
    pub(super) region: PixelRect,
    pub(super) class: LayoutClass,
    pub(super) confidence: f32,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    region: PixelRect,
    class_index: usize,
    confidence: f32,
    row_index: usize,
}

pub(super) fn decode(
    values: &[f32],
    canvas_width: u32,
    canvas_height: u32,
) -> UseResult<Vec<DetectedLayoutRegion>> {
    if canvas_width == 0 || canvas_height == 0 {
        return Err(output_error(
            "Document-layout decoding requires a positive source canvas.",
        ));
    }
    if values.len() != LOCATION_COUNT * OUTPUT_WIDTH {
        return Err(output_error(format!(
            "One PP-DocLayout-S output requires {} values, found {}.",
            LOCATION_COUNT * OUTPUT_WIDTH,
            values.len()
        )));
    }

    let mut by_class = vec![Vec::<Candidate>::new(); CLASS_COUNT];
    for (row_index, row) in values.chunks_exact(OUTPUT_WIDTH).enumerate() {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(output_error(
                "PP-DocLayout-S emitted a non-finite score or coordinate.",
            ));
        }
        let coordinates = [row[0], row[1], row[2], row[3]];
        let Some(region) = project_box(coordinates, canvas_width, canvas_height) else {
            continue;
        };
        for (class_index, confidence) in row[4..].iter().copied().enumerate() {
            if confidence >= SCORE_THRESHOLD {
                by_class[class_index].push(Candidate {
                    region,
                    class_index,
                    confidence,
                    row_index,
                });
            }
        }
    }

    let mut retained = Vec::new();
    for candidates in &mut by_class {
        candidates.sort_by(candidate_order);
        candidates.truncate(NMS_TOP_K);
        let mut class_retained = Vec::<Candidate>::new();
        for candidate in candidates.iter().copied() {
            if class_retained.iter().any(|known| {
                intersection_over_union(known.region, candidate.region) >= NMS_IOU_THRESHOLD
            }) {
                continue;
            }
            class_retained.push(candidate);
        }
        retained.extend(class_retained);
    }
    retained.sort_by(candidate_order);
    retained.truncate(KEEP_TOP_K);
    Ok(retained
        .into_iter()
        .map(|candidate| DetectedLayoutRegion {
            region: candidate.region,
            class: CLASSES[candidate.class_index],
            confidence: candidate.confidence,
        })
        .collect())
}

fn candidate_order(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .confidence
        .total_cmp(&left.confidence)
        .then_with(|| left.class_index.cmp(&right.class_index))
        .then_with(|| left.row_index.cmp(&right.row_index))
        .then_with(|| left.region.x.cmp(&right.region.x))
        .then_with(|| left.region.y.cmp(&right.region.y))
}

fn project_box(coordinates: [f32; 4], canvas_width: u32, canvas_height: u32) -> Option<PixelRect> {
    let side = INPUT_SIDE as f32;
    let x1 = coordinates[0].clamp(0.0, side);
    let y1 = coordinates[1].clamp(0.0, side);
    let x2 = coordinates[2].clamp(0.0, side);
    let y2 = coordinates[3].clamp(0.0, side);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let scale_x = canvas_width as f32 / side;
    let scale_y = canvas_height as f32 / side;
    let left = (x1 * scale_x).floor().clamp(0.0, canvas_width as f32) as u32;
    let top = (y1 * scale_y).floor().clamp(0.0, canvas_height as f32) as u32;
    let right = (x2 * scale_x).ceil().clamp(0.0, canvas_width as f32) as u32;
    let bottom = (y2 * scale_y).ceil().clamp(0.0, canvas_height as f32) as u32;
    (right > left && bottom > top).then_some(PixelRect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

fn intersection_over_union(left: PixelRect, right: PixelRect) -> f32 {
    let intersection = intersection_area(left, right) as f64;
    if intersection == 0.0 {
        return 0.0;
    }
    let union = area(left) as f64 + area(right) as f64 - intersection;
    (intersection / union) as f32
}

pub(super) fn intersection_area(left: PixelRect, right: PixelRect) -> u64 {
    let left_right = left.x.saturating_add(left.width);
    let left_bottom = left.y.saturating_add(left.height);
    let right_right = right.x.saturating_add(right.width);
    let right_bottom = right.y.saturating_add(right.height);
    let width = left_right
        .min(right_right)
        .saturating_sub(left.x.max(right.x));
    let height = left_bottom
        .min(right_bottom)
        .saturating_sub(left.y.max(right.y));
    u64::from(width) * u64::from(height)
}

pub(super) fn area(region: PixelRect) -> u64 {
    u64::from(region.width) * u64::from(region.height)
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.document_layout_output_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_aware_nms_retains_overlapping_distinct_semantics() {
        let mut values = vec![0.0; LOCATION_COUNT * OUTPUT_WIDTH];
        let first = &mut values[..OUTPUT_WIDTH];
        first[..4].copy_from_slice(&[48.0, 48.0, 240.0, 144.0]);
        first[4] = 0.9;
        first[5] = 0.8;
        let second = &mut values[OUTPUT_WIDTH..2 * OUTPUT_WIDTH];
        second[..4].copy_from_slice(&[48.0, 48.0, 240.0, 144.0]);
        second[4] = 0.7;

        let decoded = decode(&values, 960, 480).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].class.raw_label, "paragraph_title");
        assert_eq!(decoded[1].class.raw_label, "image");
        assert_eq!(
            decoded[0].region,
            PixelRect {
                x: 96,
                y: 48,
                width: 384,
                height: 96,
            }
        );
    }

    #[test]
    fn invalid_raw_shape_is_rejected() {
        assert_eq!(
            decode(&[0.0], 100, 100).unwrap_err().code,
            "use.ocr.document_layout_output_invalid"
        );
    }
}
