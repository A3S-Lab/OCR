use std::sync::Arc;

use image::RgbImage;

use super::*;
use crate::document_fast::seal::decoder::DecodedSeal;
use crate::{OcrCanvasEdge, OcrSealDetectionStatus};

fn page(index: usize) -> PageAccumulator {
    PageAccumulator::from_decoded(DecodedPage {
        slot_id: OcrBatchSlotId::new(format!("page-{index}")).unwrap(),
        adjacent_predecessor_slot_id: None,
        image: Ok(Arc::new(RgbImage::new(1_200, 1_600))),
    })
}

fn seal(status: OcrSealDetectionStatus, independent_observations: u16) -> DecodedSeal {
    let region = PixelRect {
        x: 300,
        y: 400,
        width: 200,
        height: 200,
    };
    DecodedSeal {
        region,
        source_view: PixelRect {
            x: 0,
            y: 0,
            width: 1_200,
            height: 1_600,
        },
        confidence: 0.8,
        class_log_likelihood_ratio: 1.0,
        independent_observations,
        clipped_edge: (status == OcrSealDetectionStatus::BoundaryCandidate)
            .then_some(OcrCanvasEdge::Right),
        status,
    }
}

#[test]
fn seal_text_admission_eliminates_only_pages_without_fusion_consumers() {
    let empty = page(0);
    let mut singly_confirmed = page(1);
    singly_confirmed.add_seals(vec![seal(OcrSealDetectionStatus::Confirmed, 1)]);
    let mut independently_confirmed = page(2);
    independently_confirmed.add_seals(vec![seal(OcrSealDetectionStatus::Confirmed, 2)]);
    let mut boundary = page(3);
    boundary.add_seals(vec![seal(OcrSealDetectionStatus::BoundaryCandidate, 1)]);
    let pages = vec![empty, singly_confirmed, independently_confirmed, boundary];

    let admitted = text_page_references_for(&pages, true)
        .into_iter()
        .map(|reference| reference.page_index)
        .collect::<Vec<_>>();
    assert_eq!(admitted, [1, 3]);

    let eager = text_page_references_for(&pages, false)
        .into_iter()
        .map(|reference| reference.page_index)
        .collect::<Vec<_>>();
    assert_eq!(eager, [0, 1, 2, 3]);
}

#[test]
fn layout_session_count_preserves_a_fixed_device_reserve() {
    assert_eq!(
        layout_session_count_for(RuntimeDeviceKind::Cpu, None, MAX_LAYOUT_SESSIONS),
        1
    );
    assert_eq!(
        layout_session_count_for(RuntimeDeviceKind::Cuda, Some(6 << 30), MAX_LAYOUT_SESSIONS),
        1
    );
    assert_eq!(
        layout_session_count_for(RuntimeDeviceKind::Cuda, Some(10 << 30), MAX_LAYOUT_SESSIONS),
        2
    );
    assert_eq!(
        layout_session_count_for(RuntimeDeviceKind::Cuda, Some(14 << 30), MAX_LAYOUT_SESSIONS),
        3
    );
    assert_eq!(
        layout_session_count_for(RuntimeDeviceKind::Cuda, Some(18 << 30), MAX_LAYOUT_SESSIONS),
        4
    );
    assert_eq!(
        layout_session_count_for(RuntimeDeviceKind::Cuda, Some(22 << 30), MAX_LAYOUT_SESSIONS),
        4
    );
    assert_eq!(
        layout_session_count_for(RuntimeDeviceKind::Cuda, Some(24 << 30), MAX_LAYOUT_SESSIONS),
        4
    );
}

#[test]
fn layout_session_demand_tracks_only_bounded_independent_batches() {
    assert_eq!(layout_session_demand(1, 0, MAX_LAYOUT_SESSIONS), 1);
    assert_eq!(layout_session_demand(6, 0, MAX_LAYOUT_SESSIONS), 1);
    assert_eq!(layout_session_demand(7, 0, MAX_LAYOUT_SESSIONS), 2);
    assert_eq!(layout_session_demand(2, 2, MAX_LAYOUT_SESSIONS), 2);
    assert_eq!(layout_session_demand(29, 17, MAX_LAYOUT_SESSIONS), 4);
}

#[test]
fn layout_branch_sessions_keep_both_independent_branches_live() {
    assert_eq!(layout_branch_session_counts(2), (1, 1));
    assert_eq!(layout_branch_session_counts(3), (2, 1));
    assert_eq!(layout_branch_session_counts(4), (2, 2));
}
