use std::cmp::Ordering;

use a3s_use_core::{UseError, UseResult};
use image::RgbImage;

use crate::{OcrCanvasEdge, OcrSealDetectionStatus};

use super::super::wired::PixelRect;
use super::geometry::{area, center_is_inside, intersection_and_areas, intersection_over_union};
use super::preprocess::SealView;
use super::profile::{PicodetLayoutProfile, KEEP_TOP_K, NMS_IOU_THRESHOLD, SCORE_THRESHOLD};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DecodedSeal {
    pub(super) region: PixelRect,
    /// Exact immutable-source view that produced this observation. This is
    /// retained internally so cross-scale reconciliation can prefer the more
    /// resolved observation without exposing crop geometry as public evidence.
    pub(super) source_view: PixelRect,
    pub(super) confidence: f32,
    /// Log likelihood ratio of the seal class against the strongest competing
    /// layout class for the identity-bearing model observation.
    pub(super) class_log_likelihood_ratio: f32,
    /// Number of independent source views or model observations supporting
    /// this physical object. Dense rows from one model view still count once.
    pub(super) independent_observations: u16,
    pub(super) clipped_edge: Option<OcrCanvasEdge>,
    pub(super) status: OcrSealDetectionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DecodedLayoutImage {
    pub(super) region: PixelRect,
    pub(super) source_view: PixelRect,
    pub(super) confidence: f32,
    pub(super) clipped_edge: Option<OcrCanvasEdge>,
    pub(super) status: OcrSealDetectionStatus,
}

pub(super) fn decode_page_views(
    outputs: &[(&SealView, &[f32])],
    image: &RgbImage,
    profile: PicodetLayoutProfile,
) -> UseResult<Vec<DecodedSeal>> {
    let mut detections = Vec::new();
    for (view, values) in outputs {
        detections.extend(decode_view(values, **view, image, profile)?);
    }
    Ok(deduplicate(detections))
}

pub(super) fn merge_page_detections(
    existing: Vec<DecodedSeal>,
    additional: Vec<DecodedSeal>,
) -> Vec<DecodedSeal> {
    let observations = existing.into_iter().chain(additional).collect::<Vec<_>>();
    let mut retained = deduplicate(observations.clone());
    for seal in &mut retained {
        let mut source_support = Vec::<(PixelRect, u16)>::new();
        for observation in observations
            .iter()
            .filter(|observation| same_physical_object(**observation, *seal))
        {
            if let Some((_, count)) = source_support
                .iter_mut()
                .find(|(source_view, _)| *source_view == observation.source_view)
            {
                *count = (*count).max(observation.independent_observations);
            } else {
                source_support.push((
                    observation.source_view,
                    observation.independent_observations,
                ));
            }
        }
        seal.independent_observations = source_support
            .into_iter()
            .map(|(_, count)| count)
            .fold(0_u16, u16::saturating_add)
            .max(seal.independent_observations);
    }
    retained
}

/// Tests whether a caller-declared adjacent page contains the continuation of
/// one already-confirmed clipped seal identity.
///
/// The predecessor and current observations are conditionally joined as one
/// physical object: their seal-versus-competing-class log likelihood ratios
/// are added, and the shared seal identity must still win at the natural zero
/// decision boundary. Current-page geometry must independently overlap the
/// normalized predecessor view at the reviewed model NMS threshold.
pub(super) fn decode_adjacent_boundary_continuation(
    values: &[f32],
    view: SealView,
    image: &RgbImage,
    profile: PicodetLayoutProfile,
    predecessor: DecodedSeal,
) -> UseResult<Option<DecodedSeal>> {
    if predecessor.status != OcrSealDetectionStatus::BoundaryCandidate {
        return Err(output_error(
            "Adjacent seal verification requires a confirmed predecessor boundary candidate.",
        ));
    }
    let Some(edge) = predecessor.clipped_edge else {
        return Err(output_error(
            "Adjacent seal verification requires exact predecessor edge provenance.",
        ));
    };
    if !view_edge_is_source_edge(view, edge, image.width(), image.height()) {
        return Err(output_error(
            "An adjacent seal verification view must retain the predecessor source edge.",
        ));
    }
    let output_width = profile.output_width();
    let expected = profile.location_count() * output_width;
    if values.len() != expected {
        return Err(output_error(format!(
            "One adjacent PicoDet view must contain {expected} raw values, found {}.",
            values.len()
        )));
    }

    let mut best: Option<(DecodedSeal, f32)> = None;
    let seal_class_index = profile.seal_class_index();
    for row in values.chunks_exact(output_width) {
        if row[..output_width].iter().any(|value| !value.is_finite()) {
            return Err(output_error(
                "PicoDet emitted a non-finite adjacent seal score or coordinate.",
            ));
        }
        let scores = &row[4..output_width];
        let seal_score = scores[seal_class_index];
        let competing_score = scores
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != seal_class_index)
            .map(|(_, score)| *score)
            .max_by(f32::total_cmp)
            .unwrap_or(f32::NEG_INFINITY);
        let combined_log_likelihood = predecessor.class_log_likelihood_ratio
            + class_log_likelihood_ratio(seal_score, competing_score);
        if combined_log_likelihood < 0.0 {
            continue;
        }
        let Some(projected) = project_detection(
            [row[0], row[1], row[2], row[3]],
            seal_score,
            view,
            image.width(),
            image.height(),
            profile.input_side(),
        ) else {
            continue;
        };
        let overlap = intersection_over_union(projected.region, view.region);
        if overlap < NMS_IOU_THRESHOLD {
            continue;
        }
        let candidate = DecodedSeal {
            region: extend_region_to_source_edge(
                projected.region,
                edge,
                image.width(),
                image.height(),
            ),
            source_view: view.region,
            confidence: predecessor.confidence.min(seal_score),
            class_log_likelihood_ratio: combined_log_likelihood,
            independent_observations: predecessor.independent_observations.saturating_add(1),
            clipped_edge: Some(edge),
            status: OcrSealDetectionStatus::BoundaryCandidate,
        };
        let replaces = best.as_ref().is_none_or(|(known, known_overlap)| {
            combined_log_likelihood > known.class_log_likelihood_ratio
                || (combined_log_likelihood == known.class_log_likelihood_ratio
                    && overlap > *known_overlap)
        });
        if replaces {
            best = Some((candidate, overlap));
        }
    }
    let result = best.map(|(candidate, _)| candidate);
    if std::env::var_os("A3S_OCR_TRACE_SEAL_DETECTIONS").is_some() {
        eprintln!(
            "A3S_OCR_ADJACENT_SEAL source_region={:?} predecessor={:?} continuation={:?}",
            view.region, predecessor, result,
        );
    }
    Ok(result)
}

fn extend_region_to_source_edge(
    region: PixelRect,
    edge: OcrCanvasEdge,
    canvas_width: u32,
    canvas_height: u32,
) -> PixelRect {
    match edge {
        OcrCanvasEdge::Left => PixelRect {
            x: 0,
            width: region.x.saturating_add(region.width).min(canvas_width),
            ..region
        },
        OcrCanvasEdge::Top => PixelRect {
            y: 0,
            height: region.y.saturating_add(region.height).min(canvas_height),
            ..region
        },
        OcrCanvasEdge::Right => PixelRect {
            width: canvas_width.saturating_sub(region.x),
            ..region
        },
        OcrCanvasEdge::Bottom => PixelRect {
            height: canvas_height.saturating_sub(region.y),
            ..region
        },
    }
}

pub(super) fn decode_page_layout_images(
    values: &[f32],
    view: SealView,
    image: &RgbImage,
    profile: PicodetLayoutProfile,
) -> UseResult<Vec<DecodedLayoutImage>> {
    let location_count = profile.location_count();
    let output_width = profile.output_width();
    if values.len() != location_count * output_width {
        return Err(output_error(format!(
            "One PicoDet layout-image view must contain {} raw values, found {}.",
            location_count * output_width,
            values.len()
        )));
    }
    let mut observations = Vec::new();
    for row in values.chunks_exact(output_width) {
        let scores = &row[4..output_width];
        if row[..output_width].iter().any(|value| !value.is_finite()) {
            return Err(output_error(
                "PicoDet emitted a non-finite layout-image score or coordinate.",
            ));
        }
        let Some((winning_class, confidence)) = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
        else {
            continue;
        };
        if winning_class != profile.image_class_index() || confidence < SCORE_THRESHOLD {
            continue;
        }
        let Some(projected) = project_detection(
            [row[0], row[1], row[2], row[3]],
            confidence,
            view,
            image.width(),
            image.height(),
            profile.input_side(),
        ) else {
            continue;
        };
        observations.push(DecodedLayoutImage {
            region: projected.region,
            source_view: projected.source_view,
            confidence: projected.confidence,
            clipped_edge: projected.clipped_edge,
            status: projected.status,
        });
    }
    observations.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut retained = Vec::new();
    for observation in observations {
        if retained.iter().any(|known: &DecodedLayoutImage| {
            intersection_over_union(known.region, observation.region) >= NMS_IOU_THRESHOLD
        }) {
            continue;
        }
        retained.push(observation);
        if retained.len() == KEEP_TOP_K {
            break;
        }
    }
    Ok(retained)
}

fn deduplicate(mut detections: Vec<DecodedSeal>) -> Vec<DecodedSeal> {
    detections.sort_by(|left, right| {
        status_rank(left.status)
            .cmp(&status_rank(right.status))
            .then_with(|| area(left.source_view).cmp(&area(right.source_view)))
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
        if retained
            .iter()
            .any(|known| same_physical_object(*known, detection))
        {
            continue;
        }
        retained.push(detection);
        if retained.len() == KEEP_TOP_K {
            break;
        }
    }
    retained
}

fn decode_view(
    values: &[f32],
    view: SealView,
    image: &RgbImage,
    profile: PicodetLayoutProfile,
) -> UseResult<Vec<DecodedSeal>> {
    let location_count = profile.location_count();
    let output_width = profile.output_width();
    if values.len() != location_count * output_width {
        return Err(output_error(format!(
            "One PicoDet view must contain {} raw values, found {}.",
            location_count * output_width,
            values.len()
        )));
    }
    let mut detections = Vec::new();
    let mut boundary_geometry = Vec::new();
    let input_side = profile.input_side();
    let seal_class_index = profile.seal_class_index();
    for row in values.chunks_exact(output_width) {
        let class_scores = &row[4..output_width];
        let score = class_scores[seal_class_index];
        let coordinates = [row[0], row[1], row[2], row[3]];
        if row[..output_width].iter().any(|value| !value.is_finite()) {
            return Err(output_error(
                "PicoDet emitted a non-finite seal score or coordinate.",
            ));
        }
        let competing_score = class_scores
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != seal_class_index)
            .map(|(_, score)| *score)
            .max_by(f32::total_cmp)
            .unwrap_or(f32::NEG_INFINITY);
        let class_log_likelihood_ratio = class_log_likelihood_ratio(score, competing_score);
        let projected = project_detection(
            coordinates,
            score,
            view,
            image.width(),
            image.height(),
            input_side,
        );
        if let Some(mut detection) = projected {
            detection.class_log_likelihood_ratio = class_log_likelihood_ratio;
            if detection.status == OcrSealDetectionStatus::BoundaryCandidate {
                boundary_geometry.push(detection);
            }
            // The reviewed layout detector is trained with one object class
            // per location. A final seal identity still requires the seal
            // head to win the location's class decision at the official score
            // threshold. Other rows may supply only corroborating clipped
            // geometry for that accepted identity.
            if score >= SCORE_THRESHOLD && score >= competing_score {
                detections.push(detection);
            }
        }
    }
    let detections = fuse_confirmed_identity_with_boundary_geometry(detections, &boundary_geometry);
    if std::env::var_os("A3S_OCR_TRACE_SEAL_DETECTIONS").is_some() {
        let mut top_scores = values
            .chunks_exact(output_width)
            .enumerate()
            .filter_map(|(index, row)| {
                let score = row[4 + seal_class_index];
                score.is_finite().then_some((
                    index,
                    score,
                    [row[0], row[1], row[2], row[3]],
                    row[4..output_width].to_vec(),
                ))
            })
            .collect::<Vec<_>>();
        top_scores.sort_by(|left, right| right.1.total_cmp(&left.1));
        top_scores.truncate(8);
        let mut source_edge_scores = values
            .chunks_exact(output_width)
            .enumerate()
            .filter_map(|(index, row)| {
                let coordinates = [row[0], row[1], row[2], row[3]];
                let scores = row[4..output_width].to_vec();
                let edge = unique_exact_source_edge(
                    coordinates,
                    input_side as f32,
                    view,
                    image.width(),
                    image.height(),
                )??;
                scores[seal_class_index].is_finite().then_some((
                    index,
                    edge,
                    scores[seal_class_index],
                    coordinates,
                    scores,
                ))
            })
            .collect::<Vec<_>>();
        source_edge_scores.sort_by(|left, right| right.2.total_cmp(&left.2));
        source_edge_scores.truncate(8);
        eprintln!(
            "A3S_OCR_SEAL_DETECTIONS source_region={:?} detections={:?} top_raw={:?} top_source_edge_raw={:?}",
            view.region, detections, top_scores, source_edge_scores
        );
    }
    Ok(detections)
}

/// Combines identity and clipping geometry without weakening the reviewed
/// model threshold.
///
/// Dense detector rows often agree on the object while neighboring regressors
/// differ by a few source pixels at a clipped edge. A threshold-passing,
/// class-winning row establishes seal identity. A spatially equivalent raw row
/// may then establish that the same object crosses the immutable source edge.
/// The boundary row can never create evidence on its own.
fn fuse_confirmed_identity_with_boundary_geometry(
    detections: Vec<DecodedSeal>,
    boundary_geometry: &[DecodedSeal],
) -> Vec<DecodedSeal> {
    detections
        .into_iter()
        .map(|detection| {
            if detection.status == OcrSealDetectionStatus::BoundaryCandidate {
                return detection;
            }
            let proposal = boundary_geometry
                .iter()
                .filter_map(|proposal| {
                    let overlap = intersection_over_union(detection.region, proposal.region);
                    (overlap >= NMS_IOU_THRESHOLD).then_some((proposal, overlap))
                })
                .max_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.confidence.total_cmp(&right.0.confidence))
                })
                .map(|(proposal, _)| proposal);
            match proposal {
                Some(proposal) => DecodedSeal {
                    region: proposal.region,
                    source_view: proposal.source_view,
                    confidence: detection.confidence.min(proposal.confidence),
                    class_log_likelihood_ratio: detection.class_log_likelihood_ratio,
                    independent_observations: detection.independent_observations,
                    clipped_edge: proposal.clipped_edge,
                    status: OcrSealDetectionStatus::BoundaryCandidate,
                },
                None => detection,
            }
        })
        .collect()
}

fn project_detection(
    coordinates: [f32; 4],
    confidence: f32,
    view: SealView,
    canvas_width: u32,
    canvas_height: u32,
    input_side: usize,
) -> Option<DecodedSeal> {
    let side = input_side as f32;
    let boundary = unique_exact_source_edge(coordinates, side, view, canvas_width, canvas_height)?;
    let status = if boundary.is_some() {
        OcrSealDetectionStatus::BoundaryCandidate
    } else {
        OcrSealDetectionStatus::Confirmed
    };
    let x1 = coordinates[0].clamp(0.0, side);
    let y1 = coordinates[1].clamp(0.0, side);
    let x2 = coordinates[2].clamp(0.0, side);
    let y2 = coordinates[3].clamp(0.0, side);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let scale_x = view.region.width as f32 / side;
    let scale_y = view.region.height as f32 / side;
    let left = (view.region.x as f32 + x1 * scale_x).floor() as u32;
    let top = (view.region.y as f32 + y1 * scale_y).floor() as u32;
    let right = (view.region.x as f32 + x2 * scale_x).ceil() as u32;
    let bottom = (view.region.y as f32 + y2 * scale_y).ceil() as u32;
    let left = left.min(canvas_width);
    let right = right.min(canvas_width);
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
        source_view: view.region,
        confidence,
        class_log_likelihood_ratio: 0.0,
        independent_observations: 1,
        clipped_edge: boundary,
        status,
    })
}

fn class_log_likelihood_ratio(seal_score: f32, competing_score: f32) -> f32 {
    stable_logit(seal_score) - stable_logit(competing_score)
}

fn stable_logit(probability: f32) -> f32 {
    let probability = probability.clamp(f32::EPSILON, 1.0 - f32::EPSILON);
    probability.ln() - (-probability).ln_1p()
}

/// Returns `None` when a box touches multiple edges because the public seal
/// evidence contract can encode only one clipped edge. Publishing one chosen
/// edge would fabricate precision that the model did not supply.
pub(super) fn unique_exact_source_edge(
    coordinates: [f32; 4],
    input_side: f32,
    view: SealView,
    canvas_width: u32,
    canvas_height: u32,
) -> Option<Option<OcrCanvasEdge>> {
    let contacts = [
        (coordinates[0] <= 0.0, OcrCanvasEdge::Left),
        (coordinates[1] <= 0.0, OcrCanvasEdge::Top),
        (coordinates[2] >= input_side, OcrCanvasEdge::Right),
        (coordinates[3] >= input_side, OcrCanvasEdge::Bottom),
    ];
    let mut edge = None;
    for (touches, candidate) in contacts {
        if !touches {
            continue;
        }
        if !view_edge_is_source_edge(view, candidate, canvas_width, canvas_height) {
            return None;
        }
        if edge.is_some() {
            return None;
        }
        edge = Some(candidate);
    }
    Some(edge)
}

fn view_edge_is_source_edge(
    view: SealView,
    edge: OcrCanvasEdge,
    canvas_width: u32,
    canvas_height: u32,
) -> bool {
    match edge {
        OcrCanvasEdge::Left => view.region.x == 0,
        OcrCanvasEdge::Top => view.region.y == 0,
        OcrCanvasEdge::Right => view.region.x.checked_add(view.region.width) == Some(canvas_width),
        OcrCanvasEdge::Bottom => {
            view.region.y.checked_add(view.region.height) == Some(canvas_height)
        }
    }
}

fn status_rank(status: OcrSealDetectionStatus) -> u8 {
    match status {
        OcrSealDetectionStatus::Confirmed => 0,
        OcrSealDetectionStatus::BoundaryCandidate => 1,
    }
}

/// Two same-class observations from different source resolutions represent
/// the same instance when at least half of the smaller box overlaps and the
/// smaller box's center lies inside the larger one. The overlap threshold is
/// the reviewed model's own NMS contract; the added center condition prevents
/// neighboring, merely intersecting seals from being collapsed.
fn cross_scale_equivalent(left: DecodedSeal, right: DecodedSeal) -> bool {
    if left.source_view == right.source_view {
        return false;
    }
    let (intersection, left_area, right_area) = intersection_and_areas(left.region, right.region);
    let smaller_area = left_area.min(right_area);
    if smaller_area == 0 || intersection as f32 / (smaller_area as f32) < NMS_IOU_THRESHOLD {
        return false;
    }
    if left_area <= right_area {
        center_is_inside(left.region, right.region)
    } else {
        center_is_inside(right.region, left.region)
    }
}

fn same_physical_object(left: DecodedSeal, right: DecodedSeal) -> bool {
    intersection_over_union(left.region, right.region) >= NMS_IOU_THRESHOLD
        || cross_scale_equivalent(left, right)
        || complete_and_boundary_equivalent(left, right)
}

/// A source-edge observation is censored geometry, not a second instance. A
/// complete observation dominates it when the model's own overlap threshold
/// covers the smaller region and the complete object's center lies inside the
/// admitted boundary extent. Unlike cross-scale reconciliation, this remains
/// valid when both rows came from the same source view.
fn complete_and_boundary_equivalent(left: DecodedSeal, right: DecodedSeal) -> bool {
    let (complete, boundary) = match (left.status, right.status) {
        (OcrSealDetectionStatus::Confirmed, OcrSealDetectionStatus::BoundaryCandidate) => {
            (left, right)
        }
        (OcrSealDetectionStatus::BoundaryCandidate, OcrSealDetectionStatus::Confirmed) => {
            (right, left)
        }
        _ => return false,
    };
    if complete.clipped_edge.is_some() || boundary.clipped_edge.is_none() {
        return false;
    }
    let (intersection, complete_area, boundary_area) =
        intersection_and_areas(complete.region, boundary.region);
    let smaller_area = complete_area.min(boundary_area);
    smaller_area > 0
        && intersection as f32 / smaller_area as f32 >= NMS_IOU_THRESHOLD
        && center_is_inside(complete.region, boundary.region)
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
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
        };
        let detection = project_detection(
            [550.0, 200.0, 640.0, 300.0],
            0.5,
            view,
            1_200,
            1_600,
            PicodetLayoutProfile::Large.input_side(),
        )
        .unwrap();
        assert_eq!(detection.region.x + detection.region.width, 1_200);
        assert_eq!(detection.clipped_edge, Some(OcrCanvasEdge::Right));
        assert_eq!(detection.status, OcrSealDetectionStatus::BoundaryCandidate);
    }

    #[test]
    fn near_edge_detection_is_not_rewritten_as_clipped_evidence() {
        let view = SealView {
            region: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
        };
        let detection = project_detection(
            [550.0, 200.0, 639.0, 300.0],
            0.9,
            view,
            1_200,
            1_600,
            PicodetLayoutProfile::Large.input_side(),
        )
        .unwrap();
        assert_eq!(detection.clipped_edge, None);
        assert_eq!(detection.status, OcrSealDetectionStatus::Confirmed);
        assert!(detection.region.x + detection.region.width < 1_200);
    }

    #[test]
    fn boxes_touching_multiple_edges_are_not_given_ambiguous_provenance() {
        let view = SealView {
            region: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
        };
        assert!(project_detection(
            [0.0, 200.0, 640.0, 300.0],
            0.9,
            view,
            1_200,
            1_600,
            PicodetLayoutProfile::Large.input_side(),
        )
        .is_none());
    }

    #[test]
    fn interior_tile_detection_projects_with_its_exact_source_offset() {
        let view = SealView {
            region: PixelRect {
                x: 0,
                y: 494,
                width: 1_190,
                height: 1_190,
            },
        };
        let detection = project_detection(
            [64.0, 64.0, 128.0, 128.0],
            0.8,
            view,
            1_190,
            1_684,
            PicodetLayoutProfile::Large.input_side(),
        )
        .unwrap();
        assert_eq!(detection.region.x, 119);
        assert_eq!(detection.region.y, 613);
        assert_eq!(detection.clipped_edge, None);
        assert_eq!(detection.status, OcrSealDetectionStatus::Confirmed);
    }

    #[test]
    fn detection_touching_an_internal_tile_edge_is_discarded() {
        let view = SealView {
            region: PixelRect {
                x: 0,
                y: 0,
                width: 1_190,
                height: 1_190,
            },
        };
        assert!(project_detection(
            [64.0, 500.0, 128.0, 640.0],
            0.8,
            view,
            1_190,
            1_684,
            PicodetLayoutProfile::Large.input_side(),
        )
        .is_none());
    }

    #[test]
    fn tile_detection_touching_a_real_source_edge_keeps_boundary_provenance() {
        let view = SealView {
            region: PixelRect {
                x: 0,
                y: 494,
                width: 1_190,
                height: 1_190,
            },
        };
        let detection = project_detection(
            [540.0, 100.0, 640.0, 200.0],
            0.8,
            view,
            1_190,
            1_684,
            PicodetLayoutProfile::Large.input_side(),
        )
        .unwrap();
        assert_eq!(detection.region.x + detection.region.width, 1_190);
        assert_eq!(detection.clipped_edge, Some(OcrCanvasEdge::Right));
        assert_eq!(detection.status, OcrSealDetectionStatus::BoundaryCandidate);
    }

    #[test]
    fn every_single_exact_edge_maps_without_an_orientation_preference() {
        let side = PicodetLayoutProfile::Large.input_side();
        let cases = [
            ([0.0, 200.0, 100.0, 300.0], OcrCanvasEdge::Left),
            ([200.0, 0.0, 300.0, 100.0], OcrCanvasEdge::Top),
            (
                [side as f32 - 100.0, 200.0, side as f32, 300.0],
                OcrCanvasEdge::Right,
            ),
            (
                [200.0, side as f32 - 100.0, 300.0, side as f32],
                OcrCanvasEdge::Bottom,
            ),
        ];
        for (coordinates, edge) in cases {
            assert_eq!(
                unique_exact_source_edge(
                    coordinates,
                    side as f32,
                    SealView {
                        region: PixelRect {
                            x: 0,
                            y: 0,
                            width: 1_200,
                            height: 1_600,
                        },
                    },
                    1_200,
                    1_600,
                ),
                Some(Some(edge))
            );
        }
    }

    #[test]
    fn subthreshold_overlap_preserves_each_model_observation() {
        let upper = DecodedSeal {
            region: PixelRect {
                x: 100,
                y: 100,
                width: 180,
                height: 80,
            },
            source_view: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
            confidence: 0.14,
            class_log_likelihood_ratio: 0.0,
            independent_observations: 1,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        };
        let lower = DecodedSeal {
            region: PixelRect {
                y: 210,
                ..upper.region
            },
            confidence: 0.13,
            ..upper
        };
        let envelope = DecodedSeal {
            region: PixelRect {
                x: 90,
                y: 90,
                width: 200,
                height: 210,
            },
            confidence: 0.18,
            ..upper
        };
        let retained = deduplicate(vec![envelope, upper, lower]);
        assert_eq!(retained.len(), 3);
        assert!(retained.iter().any(|seal| seal.region == envelope.region));
        assert!(retained.iter().any(|seal| seal.region == upper.region));
        assert!(retained.iter().any(|seal| seal.region == lower.region));
    }

    #[test]
    fn complete_observation_dominates_overlapping_censored_geometry() {
        let complete = DecodedSeal {
            region: PixelRect {
                x: 100,
                y: 100,
                width: 180,
                height: 80,
            },
            source_view: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
            confidence: 0.9,
            class_log_likelihood_ratio: 1.0,
            independent_observations: 2,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        };
        let boundary = DecodedSeal {
            region: PixelRect {
                x: 90,
                y: 90,
                width: 300,
                height: 100,
            },
            confidence: 0.2,
            class_log_likelihood_ratio: 0.1,
            independent_observations: 1,
            clipped_edge: Some(OcrCanvasEdge::Right),
            status: OcrSealDetectionStatus::BoundaryCandidate,
            ..complete
        };
        assert!(intersection_over_union(complete.region, boundary.region) < NMS_IOU_THRESHOLD);
        assert_eq!(deduplicate(vec![boundary, complete]), vec![complete]);
    }

    #[test]
    fn distinct_boundary_geometry_without_complete_center_is_preserved() {
        let complete = DecodedSeal {
            region: PixelRect {
                x: 100,
                y: 100,
                width: 180,
                height: 80,
            },
            source_view: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
            confidence: 0.9,
            class_log_likelihood_ratio: 1.0,
            independent_observations: 2,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        };
        let boundary = DecodedSeal {
            region: PixelRect {
                x: 240,
                y: 90,
                width: 300,
                height: 100,
            },
            confidence: 0.2,
            class_log_likelihood_ratio: 0.1,
            independent_observations: 1,
            clipped_edge: Some(OcrCanvasEdge::Right),
            status: OcrSealDetectionStatus::BoundaryCandidate,
            ..complete
        };
        assert_eq!(deduplicate(vec![boundary, complete]).len(), 2);
    }

    #[test]
    fn a_competing_layout_class_cannot_be_published_as_a_seal() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::new(1_200, 1_600);
        let view = super::super::preprocess::full_page_view(&image);
        let mut values = vec![0.0; profile.location_count() * profile.output_width()];
        values[..7].copy_from_slice(&[100.0, 100.0, 200.0, 200.0, 0.9, 0.1, 0.8]);
        assert!(decode_view(&values, view, &image, profile)
            .unwrap()
            .is_empty());

        values[6] = 0.91;
        assert_eq!(
            decode_view(&values, view, &image, profile).unwrap().len(),
            1
        );
    }

    #[test]
    fn accepted_identity_and_overlapping_raw_edge_geometry_form_one_boundary_candidate() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::new(1_200, 1_600);
        let view = super::super::preprocess::full_page_view(&image);
        let mut values = vec![0.0; profile.location_count() * profile.output_width()];
        values[..7].copy_from_slice(&[550.0, 200.0, 638.0, 300.0, 0.1, 0.1, 0.9]);
        values[7..14].copy_from_slice(&[548.0, 198.0, 645.0, 302.0, 0.2, 0.1, 0.05]);

        let detections = decode_view(&values, view, &image, profile).unwrap();
        assert_eq!(detections.len(), 1);
        let detection = detections[0];
        assert_eq!(detection.status, OcrSealDetectionStatus::BoundaryCandidate);
        assert_eq!(detection.clipped_edge, Some(OcrCanvasEdge::Right));
        assert_eq!(detection.region.x + detection.region.width, image.width());
        assert_eq!(detection.confidence, 0.05);
    }

    #[test]
    fn unrelated_edge_geometry_cannot_relabel_an_interior_identity() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::new(1_200, 1_600);
        let view = super::super::preprocess::full_page_view(&image);
        let mut values = vec![0.0; profile.location_count() * profile.output_width()];
        values[..7].copy_from_slice(&[200.0, 200.0, 300.0, 300.0, 0.1, 0.1, 0.9]);
        values[7..14].copy_from_slice(&[550.0, 400.0, 645.0, 500.0, 0.2, 0.1, 0.05]);

        let detections = decode_view(&values, view, &image, profile).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].status, OcrSealDetectionStatus::Confirmed);
        assert_eq!(detections[0].clipped_edge, None);
    }

    fn predecessor_boundary(class_log_likelihood_ratio: f32) -> DecodedSeal {
        DecodedSeal {
            region: PixelRect {
                x: 1_140,
                y: 600,
                width: 60,
                height: 180,
            },
            source_view: PixelRect {
                x: 1_020,
                y: 560,
                width: 180,
                height: 180,
            },
            confidence: 0.1,
            class_log_likelihood_ratio,
            independent_observations: 1,
            clipped_edge: Some(OcrCanvasEdge::Right),
            status: OcrSealDetectionStatus::BoundaryCandidate,
        }
    }

    #[test]
    fn adjacent_observations_publish_only_when_joint_seal_likelihood_wins() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::new(1_200, 1_600);
        let view = SealView {
            region: PixelRect {
                x: 1_140,
                y: 600,
                width: 60,
                height: 180,
            },
        };
        let mut values = vec![0.0; profile.location_count() * profile.output_width()];
        values[..7].copy_from_slice(&[40.0, 20.0, 625.0, 620.0, 0.2, 0.1, 0.03]);

        let continuation = decode_adjacent_boundary_continuation(
            &values,
            view,
            &image,
            profile,
            predecessor_boundary(3.0),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            continuation.status,
            OcrSealDetectionStatus::BoundaryCandidate
        );
        assert_eq!(continuation.clipped_edge, Some(OcrCanvasEdge::Right));
        assert_eq!(
            continuation.region.x + continuation.region.width,
            image.width()
        );
        assert_eq!(continuation.confidence, 0.03);

        assert!(decode_adjacent_boundary_continuation(
            &values,
            view,
            &image,
            profile,
            predecessor_boundary(1.0),
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn joint_class_likelihood_cannot_override_unrelated_current_geometry() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::new(1_200, 1_600);
        let view = SealView {
            region: PixelRect {
                x: 1_140,
                y: 600,
                width: 60,
                height: 180,
            },
        };
        let mut values = vec![0.0; profile.location_count() * profile.output_width()];
        values[..7].copy_from_slice(&[0.0, 0.0, 200.0, 200.0, 0.01, 0.01, 0.99]);

        assert!(decode_adjacent_boundary_continuation(
            &values,
            view,
            &image,
            profile,
            predecessor_boundary(10.0),
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn cross_scale_containment_prefers_the_more_resolved_observation() {
        let coarse = DecodedSeal {
            region: PixelRect {
                x: 90,
                y: 80,
                width: 260,
                height: 240,
            },
            source_view: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
            confidence: 0.95,
            class_log_likelihood_ratio: 0.0,
            independent_observations: 1,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        };
        let refined = DecodedSeal {
            region: PixelRect {
                x: 120,
                y: 100,
                width: 180,
                height: 160,
            },
            source_view: PixelRect {
                x: 20,
                y: 20,
                width: 480,
                height: 480,
            },
            confidence: 0.55,
            ..coarse
        };

        assert_eq!(deduplicate(vec![coarse, refined]), vec![refined]);
    }

    #[test]
    fn coarse_group_envelope_does_not_collapse_independent_refined_instances() {
        let envelope = DecodedSeal {
            region: PixelRect {
                x: 80,
                y: 80,
                width: 240,
                height: 360,
            },
            source_view: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
            confidence: 0.95,
            class_log_likelihood_ratio: 0.0,
            independent_observations: 1,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        };
        let upper = DecodedSeal {
            region: PixelRect {
                x: 100,
                y: 100,
                width: 180,
                height: 100,
            },
            source_view: PixelRect {
                x: 20,
                y: 20,
                width: 480,
                height: 480,
            },
            confidence: 0.7,
            ..envelope
        };
        let lower = DecodedSeal {
            region: PixelRect {
                y: 300,
                ..upper.region
            },
            source_view: PixelRect {
                x: 20,
                y: 240,
                width: 480,
                height: 480,
            },
            ..upper
        };

        let retained = deduplicate(vec![envelope, upper, lower]);
        assert_eq!(retained.len(), 2);
        assert!(retained.contains(&upper));
        assert!(retained.contains(&lower));
    }
}
