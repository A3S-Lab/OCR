use std::cmp::Ordering;

use a3s_use_core::{UseError, UseResult};

use crate::{OcrCanvasEdge, OcrSealDetectionStatus};

use super::super::wired::PixelRect;
use super::native::{LOCATION_COUNT, OUTPUT_WIDTH};
use super::preprocess::{SealView, SealViewKind, INPUT_SIDE};

const FULL_PAGE_THRESHOLD: f32 = 0.10;
const BOUNDARY_THRESHOLD: f32 = 0.025;
const BOUNDARY_CONTACT_FRACTION: f32 = 0.15;
const NMS_IOU_THRESHOLD: f32 = 0.5;
const MAX_SEALS_PER_PAGE: usize = 64;

#[derive(Debug, Clone, Copy)]
pub(super) struct DecodedSeal {
    pub(super) region: PixelRect,
    pub(super) confidence: f32,
    pub(super) clipped_edge: Option<OcrCanvasEdge>,
    pub(super) status: OcrSealDetectionStatus,
}

pub(super) fn decode_page_views(
    outputs: &[(&SealView, &[f32])],
    canvas_width: u32,
    canvas_height: u32,
) -> UseResult<Vec<DecodedSeal>> {
    let mut detections = Vec::new();
    for (view, values) in outputs {
        detections.extend(decode_view(values, **view, canvas_width, canvas_height)?);
    }
    Ok(deduplicate(detections))
}

pub(super) fn merge_page_detections(
    existing: Vec<DecodedSeal>,
    additional: Vec<DecodedSeal>,
) -> Vec<DecodedSeal> {
    deduplicate(existing.into_iter().chain(additional).collect())
}

fn deduplicate(mut detections: Vec<DecodedSeal>) -> Vec<DecodedSeal> {
    detections.sort_by(|left, right| {
        status_rank(left.status)
            .cmp(&status_rank(right.status))
            .then_with(|| {
                right
                    .confidence
                    .partial_cmp(&left.confidence)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.region.x.cmp(&right.region.x))
            .then_with(|| left.region.y.cmp(&right.region.y))
    });
    let mut retained: Vec<DecodedSeal> = Vec::new();
    for detection in detections {
        if retained.iter().any(|known| {
            intersection_over_union(known.region, detection.region) >= NMS_IOU_THRESHOLD
        }) {
            continue;
        }
        retained.push(detection);
        if retained.len() == MAX_SEALS_PER_PAGE {
            break;
        }
    }
    retained
}

fn decode_view(
    values: &[f32],
    view: SealView,
    canvas_width: u32,
    canvas_height: u32,
) -> UseResult<Vec<DecodedSeal>> {
    if values.len() != LOCATION_COUNT * OUTPUT_WIDTH {
        return Err(output_error(format!(
            "One PicoDet view must contain {} raw values, found {}.",
            LOCATION_COUNT * OUTPUT_WIDTH,
            values.len()
        )));
    }
    let (threshold, status, boundary) = match view.kind {
        SealViewKind::FullPage => (FULL_PAGE_THRESHOLD, OcrSealDetectionStatus::Confirmed, None),
        SealViewKind::Boundary(edge) => (
            BOUNDARY_THRESHOLD,
            OcrSealDetectionStatus::BoundaryCandidate,
            Some(edge),
        ),
    };
    let mut detections = Vec::new();
    for row in values.chunks_exact(OUTPUT_WIDTH) {
        let score = row[6];
        if score < threshold {
            continue;
        }
        let coordinates = [row[0], row[1], row[2], row[3]];
        if !score.is_finite() || coordinates.iter().any(|value| !value.is_finite()) {
            return Err(output_error(
                "PicoDet emitted a non-finite seal score or coordinate.",
            ));
        }
        if let Some(edge) = boundary {
            let contact = INPUT_SIDE as f32 * BOUNDARY_CONTACT_FRACTION;
            let touches = match edge {
                OcrCanvasEdge::Left => coordinates[0] <= contact,
                OcrCanvasEdge::Right => coordinates[2] >= INPUT_SIDE as f32 - contact,
                OcrCanvasEdge::Top | OcrCanvasEdge::Bottom => false,
            };
            if !touches {
                continue;
            }
        }
        if let Some(detection) = project_detection(
            coordinates,
            score,
            view,
            canvas_width,
            canvas_height,
            status,
            boundary,
        ) {
            detections.push(detection);
        }
    }
    Ok(match boundary {
        Some(edge) => fuse_boundary_fragments(detections, edge),
        None => detections,
    })
}

fn fuse_boundary_fragments(detections: Vec<DecodedSeal>, edge: OcrCanvasEdge) -> Vec<DecodedSeal> {
    let broad = detections.iter().enumerate().max_by_key(|(_, detection)| {
        u64::from(detection.region.width) * u64::from(detection.region.height)
    });
    let Some((broad_index, broad)) = broad else {
        return detections;
    };
    let broad_region = broad.region;
    let broad_confidence = broad.confidence;
    let narrow = detections
        .iter()
        .enumerate()
        .filter(|(index, detection)| {
            *index != broad_index
                && detection.region.width.saturating_mul(4) <= broad_region.width
                && rect_contains(broad_region, detection.region)
        })
        .collect::<Vec<_>>();
    if narrow.len() < 2 {
        return detections;
    }
    let left = narrow
        .iter()
        .map(|(_, detection)| detection.region.x)
        .min()
        .unwrap_or(broad_region.x);
    let right = narrow
        .iter()
        .map(|(_, detection)| detection.region.x + detection.region.width)
        .max()
        .unwrap_or(left);
    if right <= left {
        return detections;
    }
    let confidence = narrow
        .iter()
        .map(|(_, detection)| detection.confidence)
        .fold(broad_confidence, f32::min);
    let consumed = narrow
        .iter()
        .map(|(index, _)| *index)
        .chain(std::iter::once(broad_index))
        .collect::<std::collections::BTreeSet<_>>();
    let mut fused = detections
        .into_iter()
        .enumerate()
        .filter_map(|(index, detection)| (!consumed.contains(&index)).then_some(detection))
        .collect::<Vec<_>>();
    fused.push(DecodedSeal {
        region: PixelRect {
            x: left,
            y: broad_region.y,
            width: right - left,
            height: broad_region.height,
        },
        confidence,
        clipped_edge: Some(edge),
        status: OcrSealDetectionStatus::BoundaryCandidate,
    });
    fused
}

fn rect_contains(outer: PixelRect, inner: PixelRect) -> bool {
    outer.x <= inner.x
        && outer.y <= inner.y
        && outer.x.saturating_add(outer.width) >= inner.x.saturating_add(inner.width)
        && outer.y.saturating_add(outer.height) >= inner.y.saturating_add(inner.height)
}

fn project_detection(
    coordinates: [f32; 4],
    confidence: f32,
    view: SealView,
    canvas_width: u32,
    canvas_height: u32,
    status: OcrSealDetectionStatus,
    boundary: Option<OcrCanvasEdge>,
) -> Option<DecodedSeal> {
    let side = INPUT_SIDE as f32;
    let x1 = coordinates[0].clamp(0.0, side);
    let y1 = coordinates[1].clamp(0.0, side);
    let x2 = coordinates[2].clamp(0.0, side);
    let y2 = coordinates[3].clamp(0.0, side);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let scale_x = view.region.width as f32 / side;
    let scale_y = view.region.height as f32 / side;
    let mut left = (view.region.x as f32 + x1 * scale_x).floor() as u32;
    let top = (view.region.y as f32 + y1 * scale_y).floor() as u32;
    let mut right = (view.region.x as f32 + x2 * scale_x).ceil() as u32;
    let bottom = (view.region.y as f32 + y2 * scale_y).ceil() as u32;
    match boundary {
        Some(OcrCanvasEdge::Left) => left = 0,
        Some(OcrCanvasEdge::Right) => right = canvas_width,
        _ => {}
    }
    left = left.min(canvas_width);
    right = right.min(canvas_width);
    let top = top.min(canvas_height);
    let bottom = bottom.min(canvas_height);
    if right <= left || bottom <= top {
        return None;
    }
    Some(DecodedSeal {
        region: PixelRect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        },
        confidence,
        clipped_edge: boundary,
        status,
    })
}

fn status_rank(status: OcrSealDetectionStatus) -> u8 {
    match status {
        OcrSealDetectionStatus::Confirmed => 0,
        OcrSealDetectionStatus::BoundaryCandidate => 1,
    }
}

fn intersection_over_union(left: PixelRect, right: PixelRect) -> f32 {
    let left_x2 = left.x.saturating_add(left.width);
    let left_y2 = left.y.saturating_add(left.height);
    let right_x2 = right.x.saturating_add(right.width);
    let right_y2 = right.y.saturating_add(right.height);
    let intersection_width = left_x2.min(right_x2).saturating_sub(left.x.max(right.x));
    let intersection_height = left_y2.min(right_y2).saturating_sub(left.y.max(right.y));
    let intersection = intersection_width.saturating_mul(intersection_height) as f32;
    if intersection == 0.0 {
        return 0.0;
    }
    let left_area = left.width.saturating_mul(left.height) as f32;
    let right_area = right.width.saturating_mul(right.height) as f32;
    intersection / (left_area + right_area - intersection)
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_model_output_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_boundary_projection_touches_the_exact_source_edge() {
        let view = SealView {
            region: PixelRect {
                x: 1_100,
                y: 0,
                width: 100,
                height: 1_600,
            },
            kind: SealViewKind::Boundary(OcrCanvasEdge::Right),
        };
        let detection = project_detection(
            [550.0, 200.0, 630.0, 300.0],
            0.5,
            view,
            1_200,
            1_600,
            OcrSealDetectionStatus::BoundaryCandidate,
            Some(OcrCanvasEdge::Right),
        )
        .unwrap();
        assert_eq!(detection.region.x + detection.region.width, 1_200);
        assert_eq!(detection.clipped_edge, Some(OcrCanvasEdge::Right));
    }

    #[test]
    fn two_contained_fragments_and_one_envelope_fuse_without_confirmation() {
        let broad = DecodedSeal {
            region: PixelRect {
                x: 1_130,
                y: 730,
                width: 60,
                height: 298,
            },
            confidence: 0.026,
            clipped_edge: Some(OcrCanvasEdge::Right),
            status: OcrSealDetectionStatus::BoundaryCandidate,
        };
        let upper = DecodedSeal {
            region: PixelRect {
                x: 1_182,
                y: 790,
                width: 8,
                height: 26,
            },
            confidence: 0.04,
            ..broad
        };
        let lower = DecodedSeal {
            region: PixelRect {
                x: 1_183,
                y: 838,
                width: 7,
                height: 21,
            },
            confidence: 0.06,
            ..broad
        };
        let fused = fuse_boundary_fragments(vec![broad, upper, lower], OcrCanvasEdge::Right);
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].region.x, 1_182);
        assert_eq!(fused[0].region.y, 730);
        assert_eq!(fused[0].region.width, 8);
        assert_eq!(fused[0].region.height, 298);
        assert_eq!(fused[0].status, OcrSealDetectionStatus::BoundaryCandidate);
    }
}
