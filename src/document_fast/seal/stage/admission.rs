use std::sync::Arc;

use super::{PageAccumulator, ViewReference};
use crate::document_fast::seal::preprocess::model_contract_views as views_for_image;
use crate::document_fast::seal::refinement::{
    adjacent_source_edge_view, MAX_REFINEMENT_VIEWS_PER_PAGE,
};
use crate::OcrSealDetectionStatus;

/// Admits bounded model-contract views for every valid immutable canvas.
///
/// Admission depends only on source authority and dimensions. It never reads
/// pixels, filenames, text, prior detections, or neighbouring-page content.
pub(super) fn model_contract_views(pages: &[PageAccumulator]) -> Vec<ViewReference> {
    pages
        .iter()
        .enumerate()
        .flat_map(|(page_index, page)| {
            page.image.as_ref().into_iter().flat_map(move |image| {
                views_for_image(image)
                    .into_iter()
                    .map(move |view| ViewReference {
                        page_index,
                        image: Arc::clone(image),
                        view,
                        adjacent_boundary: None,
                    })
            })
        })
        .collect()
}

/// Admits current-page verification only from caller-declared adjacency and a
/// predecessor boundary candidate already confirmed at model resolution.
pub(super) fn adjacent_boundary_views(pages: &[PageAccumulator]) -> Vec<ViewReference> {
    let mut views = Vec::new();
    for (page_index, page) in pages.iter().enumerate() {
        let (Some(predecessor_id), Some(image)) = (&page.adjacent_predecessor_slot_id, &page.image)
        else {
            continue;
        };
        let Some(predecessor) = pages
            .iter()
            .find(|candidate| &candidate.slot_id == predecessor_id)
        else {
            continue;
        };
        let Some(canvas) = predecessor.canvas else {
            continue;
        };
        let mut candidates = predecessor
            .seals
            .iter()
            .copied()
            .filter(|seal| {
                seal.status == OcrSealDetectionStatus::BoundaryCandidate
                    && seal.clipped_edge.is_some()
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| left.region.x.cmp(&right.region.x))
                .then_with(|| left.region.y.cmp(&right.region.y))
        });
        let mut page_views = Vec::new();
        for candidate in candidates {
            let Some(edge) = candidate.clipped_edge else {
                continue;
            };
            let Some(view) = adjacent_source_edge_view(
                image,
                edge,
                candidate.region,
                canvas.width,
                canvas.height,
            ) else {
                continue;
            };
            if page_views.contains(&view) {
                continue;
            }
            page_views.push(view);
            views.push(ViewReference {
                page_index,
                image: Arc::clone(image),
                view,
                adjacent_boundary: Some(candidate),
            });
            if page_views.len() == MAX_REFINEMENT_VIEWS_PER_PAGE {
                break;
            }
        }
    }
    views
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::document_fast::seal::stage::DecodedPage;
    use crate::OcrBatchSlotId;

    fn page(index: usize, width: u32, height: u32, pixel: Rgb<u8>) -> PageAccumulator {
        PageAccumulator::from_decoded(DecodedPage {
            slot_id: OcrBatchSlotId::new(format!("page-{index}")).unwrap(),
            adjacent_predecessor_slot_id: None,
            image: Ok(Arc::new(RgbImage::from_pixel(width, height, pixel))),
        })
    }

    #[test]
    fn every_valid_page_admits_one_exact_full_page_view() {
        let pages = vec![
            page(0, 320, 480, Rgb([235, 235, 235])),
            page(1, 640, 480, Rgb([0, 0, 0])),
            page(2, 480, 640, Rgb([220, 20, 40])),
        ];
        let views = model_contract_views(&pages);

        assert_eq!(views.len(), 3);
        assert_eq!(
            views
                .iter()
                .map(|view| (view.page_index, view.view.region))
                .collect::<Vec<_>>(),
            [
                (
                    0,
                    crate::document_fast::wired::PixelRect {
                        x: 0,
                        y: 0,
                        width: 320,
                        height: 480,
                    },
                ),
                (
                    1,
                    crate::document_fast::wired::PixelRect {
                        x: 0,
                        y: 0,
                        width: 640,
                        height: 480,
                    },
                ),
                (
                    2,
                    crate::document_fast::wired::PixelRect {
                        x: 0,
                        y: 0,
                        width: 480,
                        height: 640,
                    },
                ),
            ]
        );
    }

    #[test]
    fn source_pixels_do_not_change_view_admission() {
        let dark = vec![page(0, 1_200, 1_600, Rgb([0, 0, 0]))];
        let colored = vec![page(0, 1_200, 1_600, Rgb([220, 20, 40]))];
        let admitted = |pages: &[PageAccumulator]| {
            model_contract_views(pages)
                .into_iter()
                .map(|view| (view.page_index, view.view))
                .collect::<Vec<_>>()
        };
        assert_eq!(admitted(&dark), admitted(&colored));
    }

    #[test]
    fn declared_adjacency_projects_only_confirmed_predecessor_boundary_geometry() {
        let mut pages = vec![
            page(0, 1_190, 1_684, Rgb([255, 255, 255])),
            page(1, 1_190, 1_684, Rgb([255, 255, 255])),
        ];
        let predecessor_id = pages[0].slot_id.clone();
        pages[1].adjacent_predecessor_slot_id = Some(predecessor_id);
        pages[0]
            .seals
            .push(crate::document_fast::seal::decoder::DecodedSeal {
                region: crate::document_fast::wired::PixelRect {
                    x: 1_135,
                    y: 808,
                    width: 55,
                    height: 157,
                },
                source_view: crate::document_fast::wired::PixelRect {
                    x: 1_020,
                    y: 801,
                    width: 170,
                    height: 170,
                },
                confidence: 0.1,
                class_log_likelihood_ratio: 2.0,
                independent_observations: 1,
                clipped_edge: Some(crate::OcrCanvasEdge::Right),
                status: OcrSealDetectionStatus::BoundaryCandidate,
            });

        let views = adjacent_boundary_views(&pages);
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].page_index, 1);
        assert_eq!(views[0].view.region.x + views[0].view.region.width, 1_190);
        assert_eq!(views[0].view.region.y, 808);
        assert_eq!(views[0].view.region.height, 157);
    }

    #[test]
    fn boundary_geometry_without_declared_adjacency_admits_no_neighbor_view() {
        let mut pages = vec![
            page(0, 1_190, 1_684, Rgb([255, 255, 255])),
            page(1, 1_190, 1_684, Rgb([255, 255, 255])),
        ];
        pages[0]
            .seals
            .push(crate::document_fast::seal::decoder::DecodedSeal {
                region: crate::document_fast::wired::PixelRect {
                    x: 1_135,
                    y: 808,
                    width: 55,
                    height: 157,
                },
                source_view: crate::document_fast::wired::PixelRect {
                    x: 1_020,
                    y: 801,
                    width: 170,
                    height: 170,
                },
                confidence: 0.1,
                class_log_likelihood_ratio: 2.0,
                independent_observations: 1,
                clipped_edge: Some(crate::OcrCanvasEdge::Right),
                status: OcrSealDetectionStatus::BoundaryCandidate,
            });

        assert!(adjacent_boundary_views(&pages).is_empty());
    }
}
