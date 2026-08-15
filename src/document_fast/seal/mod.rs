mod assets;
mod decoder;
mod native;
mod preprocess;
mod projection;
mod stage;

pub(super) use projection::seal_evidence;
pub(super) use stage::{DetectedSealPage, SealStageBatch, SealStageRunner};

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::assets::PicodetLayoutAssets;
    use super::decoder::decode_page_views;
    use super::native::{NativePicodetLayout, LOCATION_COUNT, OUTPUT_WIDTH};
    use super::preprocess::{adjacent_boundary_view, page_views, view_tensor};
    use super::stage::SealStageRunner;

    #[test]
    #[ignore = "requires the pinned PicoDet bundle and real rider-seal fixture"]
    fn real_pages_execute_the_power_graph_and_emit_seal_candidates() {
        let assets = PicodetLayoutAssets::from_env().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
        let cancellation = CancellationToken::new();
        let engine = NativePicodetLayout::load(&assets).unwrap();
        for page in [1, 2] {
            let image = image::open(
                std::path::Path::new(&fixture_root).join(format!("page-{page:04}.png")),
            )
            .unwrap()
            .into_rgb8();
            let views = page_views(&image);
            let mut tensor = Vec::new();
            for view in &views {
                tensor.extend(view_tensor(&image, *view).unwrap());
            }
            let permit = engine.begin(&cancellation).unwrap();
            let output = engine
                .infer_batch(tensor, views.len(), &permit, &cancellation)
                .unwrap();
            let rows = output
                .tensor
                .values
                .chunks_exact(LOCATION_COUNT * OUTPUT_WIDTH);
            let decoded_inputs = views.iter().zip(rows).collect::<Vec<_>>();
            let detections =
                decode_page_views(&decoded_inputs, image.width(), image.height()).unwrap();
            println!("page {page}: {detections:#?}");
            assert!(!detections.is_empty());
        }
    }

    #[tokio::test]
    #[ignore = "requires the pinned PicoDet bundle and real rider-seal fixture"]
    async fn adjacent_page_hint_recovers_the_tiny_rider_seal_fragment() {
        let runner = SealStageRunner::from_env_optional().unwrap().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
        let root = std::path::Path::new(&fixture_root);
        let first_id = crate::OcrBatchSlotId::new("page-1").unwrap();
        let second_id = crate::OcrBatchSlotId::new("page-2").unwrap();
        let first = crate::OcrProviderBatchSlot {
            slot_id: first_id.clone(),
            input: crate::client::read_source(&root.join("page-0001.png"))
                .await
                .unwrap(),
            adjacent_predecessor_slot_id: None,
        };
        let second = crate::OcrProviderBatchSlot {
            slot_id: second_id,
            input: crate::client::read_source(&root.join("page-0002.png"))
                .await
                .unwrap(),
            adjacent_predecessor_slot_id: Some(first_id),
        };
        let output = runner
            .run(vec![first, second], CancellationToken::new())
            .await
            .unwrap();
        let first = output.slots[0].page.as_ref().unwrap();
        assert!(first.seals.iter().any(|seal| {
            seal.status == crate::OcrSealDetectionStatus::BoundaryCandidate
                && seal.clipped_edge == Some(crate::OcrCanvasEdge::Right)
                && seal.region.y < 850
                && seal.region.y + seal.region.height > 930
        }));
        let second = output.slots[1].page.as_ref().unwrap();
        println!("adjacent page detections: {:#?}", second.seals);
        assert_eq!(
            second
                .seals
                .iter()
                .filter(|seal| seal.status == crate::OcrSealDetectionStatus::Confirmed)
                .count(),
            3
        );
        assert!(second.seals.iter().any(|seal| {
            seal.status == crate::OcrSealDetectionStatus::BoundaryCandidate
                && seal.clipped_edge == Some(crate::OcrCanvasEdge::Right)
                && seal.region.width <= 24
                && seal.region.y < 850
                && seal.region.y + seal.region.height > 980
        }));
    }

    #[test]
    #[ignore = "requires the pinned PicoDet bundle and real rider-seal fixture"]
    fn real_local_boundary_view_exposes_raw_rider_seal_geometry() {
        let assets = PicodetLayoutAssets::from_env().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
        let image = image::open(std::path::Path::new(&fixture_root).join("page-0002.png"))
            .unwrap()
            .into_rgb8();
        let view = adjacent_boundary_view(
            &image,
            crate::OcrCanvasEdge::Right,
            super::super::wired::PixelRect {
                x: 1_132,
                y: 810,
                width: 58,
                height: 149,
            },
            1_684,
        )
        .unwrap();
        println!("local boundary view: {view:#?}");
        let cancellation = CancellationToken::new();
        let engine = NativePicodetLayout::load(&assets).unwrap();
        let permit = engine.begin(&cancellation).unwrap();
        let output = engine
            .infer_batch(
                view_tensor(&image, view).unwrap(),
                1,
                &permit,
                &cancellation,
            )
            .unwrap();
        let detections = decode_page_views(
            &[(&view, output.tensor.values.as_slice())],
            image.width(),
            image.height(),
        )
        .unwrap();
        println!("local boundary detections: {detections:#?}");
        assert!(!detections.is_empty());
    }
}
