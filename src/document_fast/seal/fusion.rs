use super::super::wired::PixelRect;
use super::decoder::{DecodedLayoutImage, DecodedSeal};
use super::geometry::{area, center_is_inside, intersection_over_smaller, intersection_over_union};
use super::profile::NMS_IOU_THRESHOLD;
use super::text::{SealTextObservation, SealTextPageEvidence};
use crate::{OcrCanvasEdge, OcrSealDetectionStatus};

/// Returns whether seal-text evidence can change the semantic result for one
/// page under the fusion contract below.
///
/// Text never creates seal semantics by itself. It can only corroborate a
/// layout-image hypothesis, a singly observed confirmed seal, or a clipped
/// boundary candidate. Pages without one of those consumers have the same
/// semantic result for every possible seal-text tensor output.
pub(super) fn has_text_fusion_consumer(
    layout_images: &[DecodedLayoutImage],
    layout_seals: &[DecodedSeal],
) -> bool {
    !layout_images.is_empty()
        || layout_seals.iter().any(|seal| {
            seal.status == OcrSealDetectionStatus::BoundaryCandidate
                || (seal.status == OcrSealDetectionStatus::Confirmed
                    && seal.independent_observations < 2)
        })
}

/// Reconciles layout and seal-text observations on one immutable source canvas.
///
/// Seal-text observations can corroborate geometry that already has layout
/// semantics, but they never create seal semantics on their own. A layout
/// boundary candidate remains eligible for publication because it carries
/// explicit incomplete-geometry provenance; one geometrically compatible text
/// observation can replace that coarse boundary geometry with the most
/// complete resolved observation.
pub(super) fn fuse_seal_evidence(
    text: &SealTextPageEvidence,
    layout_images: &[DecodedLayoutImage],
    layout_seals: &[DecodedSeal],
    source_canvas: PixelRect,
) -> Vec<DecodedSeal> {
    let primary_consensus = primary_rotation_consensus(text);
    let adjacent_consensus = adjacent_rotation_consensus(text);
    let mut fused = supported_layout_images(&primary_consensus, layout_images);
    fused.extend(supported_layout_seals(&adjacent_consensus, layout_seals));
    fused.extend(boundary_text_evidence(
        text,
        &adjacent_consensus,
        layout_images,
        layout_seals,
        source_canvas,
    ));
    fused
}

fn primary_rotation_consensus(text: &SealTextPageEvidence) -> Vec<SealTextObservation> {
    rotation_pair_consensus(&text.direct, &text.clockwise90)
}

fn adjacent_rotation_consensus(text: &SealTextPageEvidence) -> Vec<SealTextObservation> {
    let views = [
        text.direct.as_slice(),
        text.clockwise90.as_slice(),
        text.clockwise180.as_slice(),
        text.clockwise270.as_slice(),
    ];
    let mut candidates = Vec::new();
    for (left, right) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        candidates.extend(rotation_pair_consensus(views[left], views[right]));
    }
    deduplicate_text(candidates)
}

fn rotation_pair_consensus(
    left: &[SealTextObservation],
    right: &[SealTextObservation],
) -> Vec<SealTextObservation> {
    let mut candidates = Vec::new();
    for left in left {
        for right in right {
            if same_text_object(left.region, right.region) {
                candidates.push(SealTextObservation {
                    region: bounding_union(left.region, right.region),
                    confidence: left.confidence.min(right.confidence),
                });
            }
        }
    }
    deduplicate_text(candidates)
}

fn supported_layout_images(
    consensus: &[SealTextObservation],
    layout_images: &[DecodedLayoutImage],
) -> Vec<DecodedSeal> {
    let mut supported = layout_images
        .iter()
        .filter_map(|layout| {
            consensus
                .iter()
                .filter_map(|text| {
                    (area(layout.region) <= area(text.region)
                        && intersection_over_smaller(text.region, layout.region)
                            >= NMS_IOU_THRESHOLD
                        && center_is_inside(layout.region, text.region))
                    .then_some(layout.confidence.min(text.confidence))
                })
                .max_by(f32::total_cmp)
                .map(|confidence| DecodedSeal {
                    region: layout.region,
                    source_view: layout.source_view,
                    confidence,
                    class_log_likelihood_ratio: 0.0,
                    independent_observations: 3,
                    clipped_edge: layout.clipped_edge,
                    status: layout.status,
                })
        })
        .collect::<Vec<_>>();

    // A coarse detector can emit one group envelope alongside multiple
    // independent children. Once two non-overlapping supported children exist,
    // the envelope is not an instance and must not be published.
    let children = supported
        .iter()
        .map(|candidate| {
            supported
                .iter()
                .filter(|child| {
                    area(child.region) < area(candidate.region)
                        && intersection_over_smaller(child.region, candidate.region)
                            >= NMS_IOU_THRESHOLD
                        && center_is_inside(child.region, candidate.region)
                })
                .map(|child| child.region)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    supported = supported
        .into_iter()
        .zip(children)
        .filter_map(|(candidate, children)| {
            let contains_independent_pair = children.iter().enumerate().any(|(index, left)| {
                children[index + 1..]
                    .iter()
                    .any(|right| intersection_over_union(*left, *right) < NMS_IOU_THRESHOLD)
            });
            (!contains_independent_pair).then_some(candidate)
        })
        .collect();
    supported.sort_by(decoded_seal_order);
    supported
}

fn supported_layout_seals(
    consensus: &[SealTextObservation],
    layout_seals: &[DecodedSeal],
) -> Vec<DecodedSeal> {
    let mut supported = layout_seals
        .iter()
        .filter(|layout| {
            layout.status == OcrSealDetectionStatus::Confirmed
                && layout.independent_observations < 2
        })
        .filter_map(|layout| {
            consensus
                .iter()
                .filter(|text| same_text_object(layout.region, text.region))
                .map(|text| text.confidence)
                .max_by(f32::total_cmp)
                .map(|confidence| DecodedSeal {
                    confidence: layout.confidence.min(confidence),
                    independent_observations: layout.independent_observations.saturating_add(2),
                    ..*layout
                })
        })
        .collect::<Vec<_>>();
    supported.sort_by(decoded_seal_order);
    supported
}

fn boundary_text_evidence(
    text: &SealTextPageEvidence,
    adjacent_consensus: &[SealTextObservation],
    layout_images: &[DecodedLayoutImage],
    layout_seals: &[DecodedSeal],
    source_canvas: PixelRect,
) -> Vec<DecodedSeal> {
    let observations = all_text_observations(text).collect::<Vec<_>>();
    let mut supported = Vec::new();
    for layout in layout_seals
        .iter()
        .filter(|seal| seal.status == OcrSealDetectionStatus::BoundaryCandidate)
    {
        let Some((text, support_confidence)) = observations
            .iter()
            .filter_map(|text| {
                boundary_can_be_resolved_by(layout.region, text.region)
                    .then(|| {
                        boundary_support_confidence(text.region, adjacent_consensus, layout_images)
                            .map(|confidence| (*text, confidence))
                    })
                    .flatten()
            })
            .max_by(|left, right| {
                area(left.0.region)
                    .cmp(&area(right.0.region))
                    .then_with(|| left.0.confidence.total_cmp(&right.0.confidence))
                    .then_with(|| left.1.total_cmp(&right.1))
            })
        else {
            continue;
        };
        let use_text_geometry = area(text.region) <= area(layout.region);
        let region = if use_text_geometry {
            text.region
        } else {
            layout.region
        };
        let (clipped_edge, status) = if use_text_geometry {
            resolved_boundary_status(*layout, region)
        } else {
            (layout.clipped_edge, layout.status)
        };
        supported.push(DecodedSeal {
            region,
            source_view: if use_text_geometry {
                source_canvas
            } else {
                layout.source_view
            },
            confidence: layout
                .confidence
                .min(text.confidence)
                .min(support_confidence),
            class_log_likelihood_ratio: layout.class_log_likelihood_ratio,
            independent_observations: layout.independent_observations.saturating_add(2),
            clipped_edge,
            status,
        });
    }
    supported.sort_by(decoded_seal_order);
    supported
}

fn boundary_support_confidence(
    observation: PixelRect,
    adjacent_consensus: &[SealTextObservation],
    layout_images: &[DecodedLayoutImage],
) -> Option<f32> {
    adjacent_consensus
        .iter()
        .filter(|support| same_text_object(observation, support.region))
        .map(|support| support.confidence)
        .chain(
            layout_images
                .iter()
                .filter(|support| same_text_object(observation, support.region))
                .map(|support| support.confidence),
        )
        .max_by(f32::total_cmp)
}

fn all_text_observations(
    text: &SealTextPageEvidence,
) -> impl Iterator<Item = &SealTextObservation> {
    [
        text.direct.as_slice(),
        text.clockwise90.as_slice(),
        text.clockwise180.as_slice(),
        text.clockwise270.as_slice(),
    ]
    .into_iter()
    .flatten()
}

fn resolved_boundary_status(
    layout: DecodedSeal,
    resolved_region: PixelRect,
) -> (Option<OcrCanvasEdge>, OcrSealDetectionStatus) {
    let Some(edge) = layout.clipped_edge else {
        return (None, OcrSealDetectionStatus::Confirmed);
    };
    let still_touches_source_edge = match edge {
        OcrCanvasEdge::Left => resolved_region.x == layout.region.x,
        OcrCanvasEdge::Top => resolved_region.y == layout.region.y,
        OcrCanvasEdge::Right => region_right(resolved_region) == region_right(layout.region),
        OcrCanvasEdge::Bottom => region_bottom(resolved_region) == region_bottom(layout.region),
    };
    if still_touches_source_edge {
        (Some(edge), OcrSealDetectionStatus::BoundaryCandidate)
    } else {
        (None, OcrSealDetectionStatus::Confirmed)
    }
}

fn same_text_object(left: PixelRect, right: PixelRect) -> bool {
    if intersection_over_union(left, right) >= NMS_IOU_THRESHOLD {
        return true;
    }
    let (smaller, larger) = if area(left) <= area(right) {
        (left, right)
    } else {
        (right, left)
    };
    intersection_over_smaller(smaller, larger) >= NMS_IOU_THRESHOLD
        && center_is_inside(smaller, larger)
        && center_is_inside(larger, smaller)
}

/// A source-edge candidate has incomplete, edge-biased geometry, so its own
/// center cannot be required to lie inside a complete observation. Resolution
/// remains directional: the complete observation must substantially overlap
/// the candidate and its center must remain inside the candidate's admitted
/// source extent.
fn boundary_can_be_resolved_by(boundary: PixelRect, observation: PixelRect) -> bool {
    intersection_over_smaller(boundary, observation) >= NMS_IOU_THRESHOLD
        && center_is_inside(observation, boundary)
}

fn bounding_union(left: PixelRect, right: PixelRect) -> PixelRect {
    let x = left.x.min(right.x);
    let y = left.y.min(right.y);
    let right_edge = region_right(left).max(region_right(right));
    let bottom_edge = region_bottom(left).max(region_bottom(right));
    PixelRect {
        x,
        y,
        width: right_edge.saturating_sub(x),
        height: bottom_edge.saturating_sub(y),
    }
}

fn deduplicate_text(mut observations: Vec<SealTextObservation>) -> Vec<SealTextObservation> {
    observations.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| area(left.region).cmp(&area(right.region)))
            .then_with(|| left.region.x.cmp(&right.region.x))
            .then_with(|| left.region.y.cmp(&right.region.y))
    });
    let mut retained = Vec::<SealTextObservation>::new();
    for observation in observations {
        if !retained
            .iter()
            .any(|known| same_text_object(known.region, observation.region))
        {
            retained.push(observation);
        }
    }
    retained
}

fn decoded_seal_order(left: &DecodedSeal, right: &DecodedSeal) -> std::cmp::Ordering {
    right
        .confidence
        .total_cmp(&left.confidence)
        .then_with(|| left.region.x.cmp(&right.region.x))
        .then_with(|| left.region.y.cmp(&right.region.y))
}

fn region_right(region: PixelRect) -> u32 {
    region.x.saturating_add(region.width)
}

fn region_bottom(region: PixelRect) -> u32 {
    region.y.saturating_add(region.height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: u32, y: u32, width: u32, height: u32) -> PixelRect {
        PixelRect {
            x,
            y,
            width,
            height,
        }
    }

    fn canvas() -> PixelRect {
        rect(0, 0, 1_200, 1_600)
    }

    fn text(region: PixelRect, confidence: f32) -> SealTextObservation {
        SealTextObservation { region, confidence }
    }

    fn layout(region: PixelRect, confidence: f32) -> DecodedLayoutImage {
        DecodedLayoutImage {
            region,
            source_view: canvas(),
            confidence,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        }
    }

    fn evidence(direct: PixelRect, clockwise90: PixelRect) -> SealTextPageEvidence {
        SealTextPageEvidence {
            direct: vec![text(direct, 0.9)],
            clockwise90: vec![text(clockwise90, 0.8)],
            clockwise180: Vec::new(),
            clockwise270: Vec::new(),
            receipts: Vec::new(),
        }
    }

    fn boundary(region: PixelRect, edge: OcrCanvasEdge) -> DecodedSeal {
        DecodedSeal {
            region,
            source_view: canvas(),
            confidence: 0.7,
            class_log_likelihood_ratio: 1.0,
            independent_observations: 1,
            clipped_edge: Some(edge),
            status: OcrSealDetectionStatus::BoundaryCandidate,
        }
    }

    fn layout_seal(region: PixelRect) -> DecodedSeal {
        DecodedSeal {
            region,
            source_view: canvas(),
            confidence: 0.7,
            class_log_likelihood_ratio: 1.0,
            independent_observations: 1,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        }
    }

    #[test]
    fn two_layout_children_replace_a_group_envelope() {
        let envelope = rect(700, 900, 300, 400);
        let upper = rect(760, 930, 180, 120);
        let lower = rect(760, 1_120, 180, 120);
        let fused = fuse_seal_evidence(
            &evidence(envelope, envelope),
            &[
                layout(envelope, 0.4),
                layout(upper, 0.7),
                layout(lower, 0.6),
            ],
            &[],
            canvas(),
        );
        let layout_fused = fused
            .iter()
            .filter(|seal| seal.independent_observations == 3)
            .collect::<Vec<_>>();
        assert_eq!(layout_fused.len(), 2);
        assert!(layout_fused.iter().any(|seal| seal.region == upper));
        assert!(layout_fused.iter().any(|seal| seal.region == lower));
    }

    #[test]
    fn enclosing_page_image_cannot_become_a_seal_instance() {
        let seal_text = rect(500, 700, 300, 200);
        let page_image = rect(10, 10, 1_180, 1_580);
        let fused = fuse_seal_evidence(
            &evidence(seal_text, seal_text),
            &[layout(page_image, 0.8)],
            &[],
            canvas(),
        );
        assert!(fused.iter().all(|seal| seal.independent_observations != 3));
    }

    #[test]
    fn text_consensus_without_layout_semantics_is_not_complete_evidence() {
        let region = rect(100, 100, 300, 200);
        let fused = fuse_seal_evidence(&evidence(region, region), &[], &[], canvas());
        assert!(fused.is_empty());
    }

    #[test]
    fn text_execution_requires_an_exact_fusion_consumer() {
        let region = rect(100, 100, 300, 200);
        assert!(!has_text_fusion_consumer(&[], &[]));
        assert!(has_text_fusion_consumer(&[layout(region, 0.7)], &[]));

        let singly_confirmed = layout_seal(region);
        assert!(has_text_fusion_consumer(&[], &[singly_confirmed]));
        assert!(!has_text_fusion_consumer(
            &[],
            &[DecodedSeal {
                independent_observations: 2,
                ..singly_confirmed
            }]
        ));
        assert!(has_text_fusion_consumer(
            &[],
            &[boundary(region, OcrCanvasEdge::Right)]
        ));
    }

    #[test]
    fn opposite_rotations_alone_are_not_consensus() {
        let region = rect(100, 100, 300, 200);
        let evidence = SealTextPageEvidence {
            direct: Vec::new(),
            clockwise90: vec![text(region, 0.9)],
            clockwise180: Vec::new(),
            clockwise270: vec![text(region, 0.8)],
            receipts: Vec::new(),
        };
        assert!(fuse_seal_evidence(&evidence, &[], &[], canvas()).is_empty());
    }

    #[test]
    fn adjacent_text_views_can_corroborate_existing_layout_seal_semantics() {
        let region = rect(500, 700, 300, 320);
        let evidence = SealTextPageEvidence {
            direct: Vec::new(),
            clockwise90: vec![text(rect(540, 680, 250, 300), 0.9)],
            clockwise180: vec![text(rect(520, 700, 280, 290), 0.8)],
            clockwise270: Vec::new(),
            receipts: Vec::new(),
        };

        let fused = fuse_seal_evidence(&evidence, &[], &[layout_seal(region)], canvas());

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].region, region);
        assert_eq!(fused[0].independent_observations, 3);
    }

    #[test]
    fn non_primary_text_consensus_cannot_relabel_a_layout_image_as_a_seal() {
        let region = rect(500, 700, 300, 320);
        let evidence = SealTextPageEvidence {
            direct: Vec::new(),
            clockwise90: vec![text(region, 0.9)],
            clockwise180: vec![text(region, 0.8)],
            clockwise270: Vec::new(),
            receipts: Vec::new(),
        };

        assert!(fuse_seal_evidence(&evidence, &[layout(region, 0.7)], &[], canvas()).is_empty());
    }

    #[test]
    fn primary_rotations_match_comparable_cross_scale_geometry() {
        let smaller = rect(150, 130, 160, 100);
        let larger = rect(100, 100, 300, 200);
        let consensus = primary_rotation_consensus(&evidence(smaller, larger));
        assert_eq!(consensus.len(), 1);
        assert_eq!(consensus[0].region, larger);
    }

    #[test]
    fn asymmetric_containment_does_not_merge_a_fragment_with_an_envelope() {
        let fragment = rect(160, 140, 40, 20);
        let envelope = rect(100, 100, 300, 200);
        assert!(primary_rotation_consensus(&evidence(fragment, envelope)).is_empty());
    }

    #[test]
    fn boundary_layout_and_one_text_model_view_resolve_geometry() {
        let coarse = rect(500, 700, 700, 500);
        let resolved = rect(700, 800, 240, 160);
        let evidence = SealTextPageEvidence {
            direct: Vec::new(),
            clockwise90: Vec::new(),
            clockwise180: Vec::new(),
            clockwise270: vec![text(resolved, 0.8)],
            receipts: Vec::new(),
        };
        let fused = fuse_seal_evidence(
            &evidence,
            &[layout(resolved, 0.75)],
            &[boundary(coarse, OcrCanvasEdge::Right)],
            canvas(),
        );
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].region, resolved);
        assert_eq!(fused[0].independent_observations, 3);
        assert_eq!(fused[0].status, OcrSealDetectionStatus::Confirmed);
        assert_eq!(fused[0].clipped_edge, None);
    }

    #[test]
    fn edge_biased_boundary_does_not_require_mutual_center_containment() {
        let coarse = rect(534, 800, 666, 705);
        let resolved = rect(482, 750, 398, 386);
        assert!(!center_is_inside(coarse, resolved));
        assert!(center_is_inside(resolved, coarse));
        let evidence = SealTextPageEvidence {
            direct: Vec::new(),
            clockwise90: Vec::new(),
            clockwise180: Vec::new(),
            clockwise270: vec![text(resolved, 0.8)],
            receipts: Vec::new(),
        };

        let fused = fuse_seal_evidence(
            &evidence,
            &[layout(rect(525, 810, 488, 209), 0.75)],
            &[boundary(coarse, OcrCanvasEdge::Right)],
            canvas(),
        );

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].region, resolved);
        assert_eq!(fused[0].status, OcrSealDetectionStatus::Confirmed);
    }

    #[test]
    fn boundary_resolution_publishes_only_the_most_complete_matching_observation() {
        let coarse = rect(500, 700, 700, 500);
        let most_complete = rect(650, 780, 400, 240);
        let evidence = SealTextPageEvidence {
            direct: vec![text(rect(700, 800, 240, 160), 0.95)],
            clockwise90: vec![text(rect(680, 790, 300, 200), 0.9)],
            clockwise180: vec![text(most_complete, 0.8)],
            clockwise270: Vec::new(),
            receipts: Vec::new(),
        };

        let fused = fuse_seal_evidence(
            &evidence,
            &[],
            &[boundary(coarse, OcrCanvasEdge::Right)],
            canvas(),
        );

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].region, most_complete);
    }

    #[test]
    fn one_text_fragment_without_consensus_or_layout_geometry_cannot_resolve_a_boundary() {
        let coarse = rect(1_135, 808, 55, 157);
        let fragment = rect(1_144, 866, 13, 94);
        let evidence = SealTextPageEvidence {
            direct: Vec::new(),
            clockwise90: Vec::new(),
            clockwise180: Vec::new(),
            clockwise270: vec![text(fragment, 0.8)],
            receipts: Vec::new(),
        };

        assert!(fuse_seal_evidence(
            &evidence,
            &[],
            &[boundary(coarse, OcrCanvasEdge::Right)],
            canvas(),
        )
        .is_empty());
    }

    #[test]
    fn resolved_text_that_still_touches_the_source_edge_remains_incomplete() {
        let coarse = rect(500, 700, 700, 500);
        let resolved = rect(750, 800, 450, 200);
        let evidence = SealTextPageEvidence {
            direct: vec![text(resolved, 0.8)],
            clockwise90: Vec::new(),
            clockwise180: Vec::new(),
            clockwise270: Vec::new(),
            receipts: Vec::new(),
        };
        let fused = fuse_seal_evidence(
            &evidence,
            &[layout(resolved, 0.75)],
            &[boundary(coarse, OcrCanvasEdge::Right)],
            canvas(),
        );
        assert_eq!(fused[0].status, OcrSealDetectionStatus::BoundaryCandidate);
        assert_eq!(fused[0].clipped_edge, Some(OcrCanvasEdge::Right));
    }

    #[test]
    fn confidence_never_exceeds_the_weakest_independent_observation() {
        let envelope = rect(100, 100, 300, 200);
        let child = rect(150, 130, 160, 100);
        let fused = fuse_seal_evidence(
            &SealTextPageEvidence {
                direct: vec![text(envelope, 0.9)],
                clockwise90: vec![text(envelope, 0.8)],
                clockwise180: Vec::new(),
                clockwise270: Vec::new(),
                receipts: Vec::new(),
            },
            &[layout(child, 0.45)],
            &[],
            canvas(),
        );
        let layout_fused = fused
            .iter()
            .find(|seal| seal.independent_observations == 3)
            .unwrap();
        assert_eq!(layout_fused.confidence, 0.45);
    }
}
