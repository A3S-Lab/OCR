use std::cmp::Ordering;

use a3s_use_core::{UseError, UseResult};
use image::RgbImage;

use crate::OcrCanvasEdge;

use super::super::wired::PixelRect;
use super::decoder::{unique_exact_source_edge, DecodedSeal};
use super::geometry::intersection_over_union;
use super::preprocess::SealView;
use super::profile::{PicodetLayoutProfile, NMS_IOU_THRESHOLD};

/// One coarse page pass may route one independently ranked hypothesis from
/// each observation regime: the four censored source edges and the uncensored
/// page interior. The hypotheses are not evidence and are never published by
/// themselves.
pub(super) const MAX_REFINEMENT_VIEWS_PER_PAGE: usize = 5;
/// Interior hypotheses retain enough surrounding layout context to occupy one
/// quarter of the detector input. A source-clipped hypothesis is inherently
/// less observable, so it is verified at the adjacent binary-pyramid scale as
/// well. Both passes retain the model's reviewed publication threshold.
const INTERIOR_CONTEXT_FACTOR: u32 = 4;
const SOURCE_EDGE_SECONDARY_CONTEXT_FACTORS: [u32; 2] = [2, 4];

#[derive(Debug, Clone, Copy)]
struct Candidate {
    region: PixelRect,
    score: f32,
    row: usize,
    source_edge: Option<OcrCanvasEdge>,
}

/// Builds a bounded coarse-to-fine verification plan from the model's own raw
/// seal-head ranking and box geometry.
///
/// Routing never reads filenames, text, page numbers, colors, hashes, or
/// fixture identities. A routed hypothesis still has to pass the reviewed
/// model score threshold in the second pass before it becomes evidence. Scores
/// from other layout classes cannot veto this proposal phase: whole-page
/// downsampling is exactly why a higher-resolution verification pass exists.
pub(super) fn model_ranked_refinement_views(
    values: &[f32],
    source_view: SealView,
    image: &RgbImage,
    profile: PicodetLayoutProfile,
    accepted: &[DecodedSeal],
) -> UseResult<Vec<SealView>> {
    let output_width = profile.output_width();
    let expected = profile
        .location_count()
        .checked_mul(output_width)
        .ok_or_else(|| refinement_error("PicoDet refinement tensor cardinality overflowed."))?;
    if values.len() != expected {
        return Err(refinement_error(format!(
            "PicoDet refinement requires {expected} raw values, found {}.",
            values.len()
        )));
    }

    let side = profile.input_side() as f32;
    let seal_class_index = profile.seal_class_index();
    let mut candidates = Vec::new();
    for (row_index, row) in values.chunks_exact(output_width).enumerate() {
        let coordinates = [row[0], row[1], row[2], row[3]];
        let class_scores = &row[4..output_width];
        if coordinates
            .iter()
            .chain(class_scores)
            .any(|value| !value.is_finite())
        {
            return Err(refinement_error(
                "PicoDet emitted non-finite refinement coordinates or class scores.",
            ));
        }
        let seal_score = class_scores[seal_class_index];
        let Some(region) = project_box(coordinates, side, source_view, image) else {
            continue;
        };
        let source_edge = unique_exact_source_edge(
            coordinates,
            side,
            source_view,
            image.width(),
            image.height(),
        )
        .flatten();
        candidates.push(Candidate {
            region,
            score: seal_score,
            row: row_index,
            source_edge,
        });
    }
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.row.cmp(&right.row))
    });

    let mut independent = Vec::<Candidate>::new();
    for candidate in candidates {
        if accepted
            .iter()
            .any(|seal| intersection_over_union(seal.region, candidate.region) >= NMS_IOU_THRESHOLD)
            || independent.iter().any(|known| {
                known.source_edge == candidate.source_edge
                    && intersection_over_union(known.region, candidate.region) >= NMS_IOU_THRESHOLD
            })
        {
            continue;
        }
        independent.push(candidate);
    }
    if std::env::var_os("A3S_OCR_TRACE_SEAL_REFINEMENT").is_some() {
        let edge_rankings = [
            OcrCanvasEdge::Left,
            OcrCanvasEdge::Top,
            OcrCanvasEdge::Right,
            OcrCanvasEdge::Bottom,
        ]
        .map(|edge| {
            (
                edge,
                independent
                    .iter()
                    .filter(|candidate| candidate.source_edge == Some(edge))
                    .take(16)
                    .map(|candidate| (candidate.row, candidate.score, candidate.region))
                    .collect::<Vec<_>>(),
            )
        });
        eprintln!("A3S_OCR_SEAL_EDGE_RANKINGS {edge_rankings:?}");
    }

    // A globally ranked top-k is statistically biased toward fully visible
    // objects: complete interior seals can always outscore a censored edge
    // fragment. Select the best independent observation in each censoring
    // regime before spending residual budget on alternate scales or duplicate
    // hypotheses. This is derived from canvas topology, not document content.
    let mut representatives = [None; MAX_REFINEMENT_VIEWS_PER_PAGE];
    for candidate in &independent {
        let regime = observation_regime(candidate.source_edge);
        representatives[regime].get_or_insert(*candidate);
    }
    let mut representatives = representatives.into_iter().flatten().collect::<Vec<_>>();
    representatives.sort_by(candidate_order);

    let mut views = Vec::new();
    for candidate in &representatives {
        let context_factor = if candidate.source_edge.is_some() {
            1
        } else {
            INTERIOR_CONTEXT_FACTOR
        };
        route_view(&mut views, *candidate, source_view, image, context_factor);
    }

    // If one or more regimes were absent, use the remaining fixed budget to
    // verify the best censored observations at adjacent binary-pyramid scales.
    for context_factor in SOURCE_EDGE_SECONDARY_CONTEXT_FACTORS {
        for candidate in representatives
            .iter()
            .filter(|candidate| candidate.source_edge.is_some())
        {
            route_view(&mut views, *candidate, source_view, image, context_factor);
            if views.len() == MAX_REFINEMENT_VIEWS_PER_PAGE {
                return Ok(views);
            }
        }
    }

    // Finally consider additional independent hypotheses without allowing one
    // regime to consume capacity needed by a previously unseen source edge.
    for candidate in independent {
        if representatives
            .iter()
            .any(|representative| representative.row == candidate.row)
        {
            continue;
        }
        let context_factor = if candidate.source_edge.is_some() {
            1
        } else {
            INTERIOR_CONTEXT_FACTOR
        };
        route_view(&mut views, candidate, source_view, image, context_factor);
        if views.len() == MAX_REFINEMENT_VIEWS_PER_PAGE {
            break;
        }
    }
    Ok(views)
}

fn observation_regime(edge: Option<OcrCanvasEdge>) -> usize {
    match edge {
        Some(OcrCanvasEdge::Left) => 0,
        Some(OcrCanvasEdge::Top) => 1,
        Some(OcrCanvasEdge::Right) => 2,
        Some(OcrCanvasEdge::Bottom) => 3,
        None => 4,
    }
}

fn candidate_order(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.row.cmp(&right.row))
}

fn route_view(
    views: &mut Vec<SealView>,
    candidate: Candidate,
    source_view: SealView,
    image: &RgbImage,
    context_factor: u32,
) {
    if views.len() == MAX_REFINEMENT_VIEWS_PER_PAGE {
        return;
    }
    let view = focus_view(candidate.region, image, context_factor);
    if view.region != source_view.region && !views.contains(&view) {
        views.push(view);
    }
}

/// Projects one independently confirmed predecessor boundary region onto the
/// adjacent immutable canvas and admits an equally resolved verification view.
///
/// Only caller-declared adjacency, normalized source geometry, and canvas
/// dimensions participate. The view does not inspect page pixels or source
/// identity, and no evidence is published unless the current page's model pass
/// independently confirms it.
pub(super) fn adjacent_source_edge_view(
    image: &RgbImage,
    edge: crate::OcrCanvasEdge,
    predecessor_region: PixelRect,
    predecessor_width: u32,
    predecessor_height: u32,
) -> Option<SealView> {
    if predecessor_width == 0
        || predecessor_height == 0
        || !region_touches_edge(
            predecessor_region,
            predecessor_width,
            predecessor_height,
            edge,
        )
    {
        return None;
    }
    let mapped = PixelRect {
        x: scale_floor(predecessor_region.x, image.width(), predecessor_width),
        y: scale_floor(predecessor_region.y, image.height(), predecessor_height),
        width: scale_length(
            predecessor_region.x,
            predecessor_region.width,
            image.width(),
            predecessor_width,
        ),
        height: scale_length(
            predecessor_region.y,
            predecessor_region.height,
            image.height(),
            predecessor_height,
        ),
    };
    Some(focus_view_at_source_edge(mapped, image, edge))
}

fn project_box(
    coordinates: [f32; 4],
    input_side: f32,
    view: SealView,
    image: &RgbImage,
) -> Option<PixelRect> {
    let x1 = coordinates[0].clamp(0.0, input_side);
    let y1 = coordinates[1].clamp(0.0, input_side);
    let x2 = coordinates[2].clamp(0.0, input_side);
    let y2 = coordinates[3].clamp(0.0, input_side);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let scale_x = view.region.width as f32 / input_side;
    let scale_y = view.region.height as f32 / input_side;
    let left = (view.region.x as f32 + x1 * scale_x).floor() as u32;
    let top = (view.region.y as f32 + y1 * scale_y).floor() as u32;
    let right = (view.region.x as f32 + x2 * scale_x).ceil() as u32;
    let bottom = (view.region.y as f32 + y2 * scale_y).ceil() as u32;
    let left = left.min(image.width());
    let right = right.min(image.width());
    let top = top.min(image.height());
    let bottom = bottom.min(image.height());
    (right > left && bottom > top).then_some(PixelRect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

fn focus_view(region: PixelRect, image: &RgbImage, context_factor: u32) -> SealView {
    let extent = region.width.max(region.height);
    let requested_side = extent.saturating_mul(context_factor);
    let width = requested_side.min(image.width()).max(1);
    let height = requested_side.min(image.height()).max(1);
    let center_x = region.x.saturating_add(region.width / 2);
    let center_y = region.y.saturating_add(region.height / 2);
    let x = center_x
        .saturating_sub(width / 2)
        .min(image.width().saturating_sub(width));
    let y = center_y
        .saturating_sub(height / 2)
        .min(image.height().saturating_sub(height));
    SealView {
        region: PixelRect {
            x,
            y,
            width,
            height,
        },
    }
}

fn focus_view_at_source_edge(
    region: PixelRect,
    image: &RgbImage,
    edge: crate::OcrCanvasEdge,
) -> SealView {
    let width = region.width.min(image.width()).max(1);
    let height = region.height.min(image.height()).max(1);
    let centered_x = region.x.min(image.width().saturating_sub(width));
    let centered_y = region.y.min(image.height().saturating_sub(height));
    let x = match edge {
        crate::OcrCanvasEdge::Left => 0,
        crate::OcrCanvasEdge::Right => image.width().saturating_sub(width),
        crate::OcrCanvasEdge::Top | crate::OcrCanvasEdge::Bottom => centered_x,
    };
    let y = match edge {
        crate::OcrCanvasEdge::Top => 0,
        crate::OcrCanvasEdge::Bottom => image.height().saturating_sub(height),
        crate::OcrCanvasEdge::Left | crate::OcrCanvasEdge::Right => centered_y,
    };
    SealView {
        region: PixelRect {
            x,
            y,
            width,
            height,
        },
    }
}

fn region_touches_edge(
    region: PixelRect,
    canvas_width: u32,
    canvas_height: u32,
    edge: crate::OcrCanvasEdge,
) -> bool {
    match edge {
        crate::OcrCanvasEdge::Left => region.x == 0,
        crate::OcrCanvasEdge::Top => region.y == 0,
        crate::OcrCanvasEdge::Right => region.x.checked_add(region.width) == Some(canvas_width),
        crate::OcrCanvasEdge::Bottom => region.y.checked_add(region.height) == Some(canvas_height),
    }
}

fn scale_floor(value: u32, target_extent: u32, source_extent: u32) -> u32 {
    (u64::from(value) * u64::from(target_extent) / u64::from(source_extent))
        .min(u64::from(target_extent)) as u32
}

fn scale_length(start: u32, length: u32, target_extent: u32, source_extent: u32) -> u32 {
    let mapped_start = scale_floor(start, target_extent, source_extent);
    let source_end = start.saturating_add(length).min(source_extent);
    let numerator = u64::from(source_end) * u64::from(target_extent);
    let denominator = u64::from(source_extent);
    let mapped_end = numerator.saturating_add(denominator.saturating_sub(1)) / denominator;
    u32::try_from(mapped_end.min(u64::from(target_extent)))
        .unwrap_or(target_extent)
        .saturating_sub(mapped_start)
        .max(1)
}

fn refinement_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_model_output_invalid", message)
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;
    use crate::OcrSealDetectionStatus;

    fn values(profile: PicodetLayoutProfile) -> Vec<f32> {
        vec![0.0; profile.location_count() * profile.output_width()]
    }

    fn set_candidate(
        values: &mut [f32],
        profile: PicodetLayoutProfile,
        row: usize,
        coordinates: [f32; 4],
        scores: [f32; 3],
    ) {
        let start = row * profile.output_width();
        values[start..start + 4].copy_from_slice(&coordinates);
        values[start + 4..start + 7].copy_from_slice(&scores);
    }

    #[test]
    fn seal_head_ranking_routes_bounded_hypotheses_to_verification() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::from_pixel(1_200, 1_600, Rgb([255, 255, 255]));
        let source_view = super::super::preprocess::full_page_view(&image);
        let mut raw = values(profile);
        set_candidate(
            &mut raw,
            profile,
            10,
            [250.0, 250.0, 350.0, 350.0],
            [0.1, 0.2, 0.25],
        );
        set_candidate(
            &mut raw,
            profile,
            20,
            [50.0, 50.0, 150.0, 150.0],
            [0.8, 0.1, 0.7],
        );

        let views = model_ranked_refinement_views(&raw, source_view, &image, profile, &[]).unwrap();
        assert_eq!(views.len(), 2);
        assert!(views.iter().all(|view| *view != source_view));
    }

    #[test]
    fn competing_layout_class_does_not_veto_seal_verification() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::from_pixel(1_200, 1_600, Rgb([255, 255, 255]));
        let source_view = super::super::preprocess::full_page_view(&image);
        let mut raw = values(profile);
        set_candidate(
            &mut raw,
            profile,
            10,
            [250.0, 250.0, 350.0, 350.0],
            [0.8, 0.1, 0.2],
        );

        assert_eq!(
            model_ranked_refinement_views(&raw, source_view, &image, profile, &[])
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn exact_source_edge_hypothesis_uses_adjacent_binary_pyramid_scales() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::from_pixel(1_200, 1_600, Rgb([255, 255, 255]));
        let source_view = super::super::preprocess::full_page_view(&image);
        let mut raw = values(profile);
        set_candidate(
            &mut raw,
            profile,
            10,
            [550.0, 200.0, 645.0, 300.0],
            [0.8, 0.1, 0.2],
        );

        let views = model_ranked_refinement_views(&raw, source_view, &image, profile, &[]).unwrap();
        assert_eq!(views.len(), 3);
        assert!(views
            .iter()
            .all(|view| view.region.x + view.region.width == image.width()));
        assert_ne!(views[0].region.width, views[1].region.width);
    }

    #[test]
    fn complete_interior_hypotheses_cannot_starve_any_source_edge_regime() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::from_pixel(1_200, 1_600, Rgb([255, 255, 255]));
        let source_view = super::super::preprocess::full_page_view(&image);
        let mut raw = values(profile);
        for (row, score, left, top) in [
            (10, 0.95, 280.0, 280.0),
            (11, 0.90, 80.0, 80.0),
            (12, 0.85, 440.0, 80.0),
            (13, 0.80, 80.0, 440.0),
        ] {
            set_candidate(
                &mut raw,
                profile,
                row,
                [left, top, left + 80.0, top + 80.0],
                [0.02, 0.01, score],
            );
        }
        for (row, coordinates) in [
            (20, [-5.0, 200.0, 80.0, 300.0]),
            (21, [200.0, -5.0, 300.0, 80.0]),
            (22, [560.0, 200.0, 645.0, 300.0]),
            (23, [200.0, 560.0, 300.0, 645.0]),
        ] {
            set_candidate(&mut raw, profile, row, coordinates, [0.2, 0.1, 0.05]);
        }

        let views = model_ranked_refinement_views(&raw, source_view, &image, profile, &[]).unwrap();
        assert_eq!(views.len(), MAX_REFINEMENT_VIEWS_PER_PAGE);
        assert!(views.iter().any(|view| view.region.x == 0));
        assert!(views.iter().any(|view| view.region.y == 0));
        assert!(views
            .iter()
            .any(|view| view.region.x + view.region.width == image.width()));
        assert!(views
            .iter()
            .any(|view| view.region.y + view.region.height == image.height()));
        assert!(views.iter().any(|view| {
            view.region.x > 0
                && view.region.y > 0
                && view.region.x + view.region.width < image.width()
                && view.region.y + view.region.height < image.height()
        }));
    }

    #[test]
    fn accepted_model_box_is_not_reprocessed() {
        let profile = PicodetLayoutProfile::Large;
        let image = RgbImage::new(1_200, 1_600);
        let source_view = super::super::preprocess::full_page_view(&image);
        let mut raw = values(profile);
        set_candidate(
            &mut raw,
            profile,
            10,
            [250.0, 250.0, 350.0, 350.0],
            [0.1, 0.2, 0.9],
        );
        let accepted = [DecodedSeal {
            region: PixelRect {
                x: 468,
                y: 625,
                width: 188,
                height: 250,
            },
            source_view: source_view.region,
            confidence: 0.9,
            class_log_likelihood_ratio: 0.0,
            independent_observations: 1,
            clipped_edge: None,
            status: OcrSealDetectionStatus::Confirmed,
        }];
        assert!(
            model_ranked_refinement_views(&raw, source_view, &image, profile, &accepted,)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn routing_does_not_depend_on_source_pixel_values() {
        let profile = PicodetLayoutProfile::Large;
        let dark = RgbImage::from_pixel(1_200, 1_600, Rgb([0, 0, 0]));
        let colored = RgbImage::from_pixel(1_200, 1_600, Rgb([210, 20, 40]));
        let source_view = super::super::preprocess::full_page_view(&dark);
        let mut raw = values(profile);
        set_candidate(
            &mut raw,
            profile,
            10,
            [250.0, 250.0, 350.0, 350.0],
            [0.1, 0.2, 0.25],
        );
        assert_eq!(
            model_ranked_refinement_views(&raw, source_view, &dark, profile, &[]).unwrap(),
            model_ranked_refinement_views(&raw, source_view, &colored, profile, &[]).unwrap(),
        );
    }

    #[test]
    fn adjacent_view_preserves_normalized_band_and_exact_source_edge() {
        let image = RgbImage::new(2_380, 3_368);
        let view = adjacent_source_edge_view(
            &image,
            crate::OcrCanvasEdge::Right,
            PixelRect {
                x: 1_135,
                y: 808,
                width: 55,
                height: 157,
            },
            1_190,
            1_684,
        )
        .unwrap();

        assert_eq!(view.region.width, 110);
        assert_eq!(view.region.height, 314);
        assert_eq!(view.region.x + view.region.width, image.width());
        assert_eq!(view.region.y, 1_616);
    }

    #[test]
    fn adjacent_view_requires_model_geometry_on_the_declared_source_edge() {
        let image = RgbImage::new(1_200, 1_600);
        assert!(adjacent_source_edge_view(
            &image,
            crate::OcrCanvasEdge::Right,
            PixelRect {
                x: 1_100,
                y: 700,
                width: 90,
                height: 160,
            },
            1_200,
            1_600,
        )
        .is_none());
    }

    #[test]
    fn adjacent_view_admission_does_not_read_current_page_pixels() {
        let dark = RgbImage::from_pixel(1_200, 1_600, Rgb([0, 0, 0]));
        let colored = RgbImage::from_pixel(1_200, 1_600, Rgb([210, 20, 40]));
        let map = |image: &RgbImage| {
            adjacent_source_edge_view(
                image,
                crate::OcrCanvasEdge::Left,
                PixelRect {
                    x: 0,
                    y: 700,
                    width: 80,
                    height: 160,
                },
                1_200,
                1_600,
            )
        };
        assert_eq!(map(&dark), map(&colored));
    }
}
