mod assets;
mod decoder;
mod fusion;
mod geometry;
mod native;
mod preprocess;
mod profile;
mod projection;
mod refinement;
mod stage;
mod text;
#[cfg(test)]
mod text_probe;

pub(super) use projection::seal_evidence;
pub(super) use stage::{DetectedSealPage, SealStageBatch, SealStageRunner};

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use tokio_util::sync::CancellationToken;

    use super::assets::PicodetLayoutAssets;
    use super::decoder::{decode_page_views, DecodedSeal};
    use super::native::NativePicodetLayout;
    use super::preprocess::{full_page_view, view_tensor};
    use super::stage::SealStageRunner;
    use crate::document_fast::page_orientation::{PageOrientationAssets, PageOrientationRunner};
    use crate::document_fast::shared_decode::decode_slots_once;

    #[test]
    #[ignore = "requires the pinned PicoDet bundle and blind real-document fixture"]
    fn blind_pages_execute_the_power_graph_with_only_model_contract_views() {
        let assets = PicodetLayoutAssets::from_env().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the read-only fixture root");
        let cancellation = CancellationToken::new();
        let engine = NativePicodetLayout::load(&assets).unwrap();
        let profile = assets.profile;
        for page in [1, 2] {
            let image = image::open(
                std::path::Path::new(&fixture_root).join(format!("page-{page:04}.png")),
            )
            .unwrap()
            .into_rgb8();
            let view = full_page_view(&image);
            let permit = engine.begin(&cancellation).unwrap();
            let output = engine
                .infer_batch(
                    view_tensor(&image, view, profile).unwrap(),
                    1,
                    &permit,
                    &cancellation,
                )
                .unwrap();
            let detections =
                decode_page_views(&[(&view, output.tensor.values.as_slice())], &image, profile)
                    .unwrap();
            println!("page {page}: {detections:#?}");
        }
    }

    #[tokio::test]
    #[ignore = "requires the pinned PicoDet bundle and blind real-document fixture"]
    async fn blind_adjacent_boundary_pair_reports_model_and_source_edge_evidence() {
        let runner = SealStageRunner::from_env_optional().unwrap().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the read-only fixture root");
        let root = std::path::Path::new(&fixture_root);
        let first_id = crate::OcrBatchSlotId::new("page-1").unwrap();
        let second_id = crate::OcrBatchSlotId::new("page-2").unwrap();
        let slots = vec![
            crate::OcrProviderBatchSlot {
                slot_id: first_id.clone(),
                input: crate::client::read_source(&root.join("page-0001.png"))
                    .await
                    .unwrap(),
                adjacent_predecessor_slot_id: None,
                text_window: None,
            },
            crate::OcrProviderBatchSlot {
                slot_id: second_id,
                input: crate::client::read_source(&root.join("page-0002.png"))
                    .await
                    .unwrap(),
                adjacent_predecessor_slot_id: Some(first_id),
                text_window: None,
            },
        ];
        let output = runner.run(slots, CancellationToken::new()).await.unwrap();
        let mut reviewed_boundaries = Vec::new();
        for slot in output.slots {
            let page = slot.page.unwrap();
            println!(
                "{} adjacent-pair detections: {:#?}",
                slot.slot_id, page.seals
            );
            let boundaries = page
                .seals
                .iter()
                .filter(|seal| {
                    seal.status == crate::OcrSealDetectionStatus::BoundaryCandidate
                        && seal.clipped_edge == Some(crate::OcrCanvasEdge::Right)
                })
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(boundaries.len(), 1, "{} boundary evidence", slot.slot_id);
            let boundary = boundaries[0];
            assert_eq!(boundary.region.x + boundary.region.width, page.canvas.width);
            reviewed_boundaries.push((page.canvas, boundary));
        }
        let (first_canvas, first) = reviewed_boundaries[0];
        let (second_canvas, second) = reviewed_boundaries[1];
        let normalized_top = |canvas: crate::OcrImageCanvas, seal: DecodedSeal| {
            u64::from(seal.region.y) * 1_000_000 / u64::from(canvas.height)
        };
        let normalized_bottom = |canvas: crate::OcrImageCanvas, seal: DecodedSeal| {
            u64::from(seal.region.y + seal.region.height) * 1_000_000 / u64::from(canvas.height)
        };
        assert!(
            normalized_top(first_canvas, first).max(normalized_top(second_canvas, second))
                < normalized_bottom(first_canvas, first)
                    .min(normalized_bottom(second_canvas, second))
        );
    }

    #[tokio::test]
    #[ignore = "requires the pinned orientation/PicoDet bundles and blind real-document fixtures"]
    async fn blind_certificate_positions_validate_without_changing_runtime_policy() {
        let runner = SealStageRunner::from_env_optional().unwrap().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the read-only fixture root");
        let root = std::path::Path::new(&fixture_root);
        let mut slots = Vec::new();
        for page in [2, 26, 27, 28, 29] {
            slots.push(crate::OcrProviderBatchSlot {
                slot_id: crate::OcrBatchSlotId::new(format!("page-{page}")).unwrap(),
                input: crate::client::read_source(&root.join(format!("page-{page:04}.png")))
                    .await
                    .unwrap(),
                adjacent_predecessor_slot_id: None,
                text_window: None,
            });
        }
        let cancellation = CancellationToken::new();
        let source_images = decode_slots_once(&slots, cancellation.clone())
            .await
            .unwrap();
        let orientation = PageOrientationRunner::new(
            PageOrientationAssets::from_env_optional()
                .unwrap()
                .expect("the reviewed page-orientation bundle must be configured"),
        )
        .unwrap()
        .normalize_decoded(source_images.clone(), cancellation.clone())
        .await
        .unwrap();
        let transforms = orientation
            .slots
            .iter()
            .map(|slot| slot.transform)
            .collect::<Vec<_>>();
        let output = runner
            .run_source_with_oriented_layout(
                slots,
                source_images,
                orientation.images,
                transforms,
                cancellation,
            )
            .await
            .unwrap();
        for slot in output.slots {
            let page = slot.page.unwrap();
            println!("{} certificate detections: {:#?}", slot.slot_id, page.seals);
            // These reviewed positions are assertions only. Production view
            // admission, thresholds, decoding, and routing never read them.
            let expected_points: &[(u32, u32)] = match slot.slot_id.as_str() {
                "page-2" => &[(428, 1_162), (893, 1_255), (887, 1_128)],
                "page-26" => &[(650, 1_150), (730, 850)],
                "page-27" => &[(857, 674)],
                "page-28" => &[(680, 925)],
                "page-29" => &[(730, 840)],
                _ => unreachable!(),
            };
            for &(x, y) in expected_points {
                assert!(
                    page.seals.iter().any(|seal| {
                        seal.status == crate::OcrSealDetectionStatus::Confirmed
                            && seal.region.x <= x
                            && seal.region.x + seal.region.width >= x
                            && seal.region.y <= y
                            && seal.region.y + seal.region.height >= y
                    }),
                    "{} missing applied stamp at ({x},{y})",
                    slot.slot_id
                );
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires the pinned PicoDet bundle and a blind raster corpus"]
    async fn blind_corpus_reports_end_to_end_cpu_throughput_without_tuning_policy() {
        let runner = SealStageRunner::from_env_optional().unwrap().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the read-only fixture root");
        let paths = raster_paths(Path::new(&fixture_root));
        assert!(
            !paths.is_empty(),
            "the blind raster corpus must not be empty"
        );

        for phase in ["cold", "warm"] {
            let slots = load_slots(&paths).await;
            let started = Instant::now();
            let output = runner.run(slots, CancellationToken::new()).await.unwrap();
            let elapsed = started.elapsed();
            let page_count = output.slots.len();
            let detection_count = output
                .slots
                .into_iter()
                .map(|slot| slot.page.unwrap().seals.len())
                .sum::<usize>();
            println!(
                "SEAL_BLIND_CORPUS phase={phase} pages={page_count} detections={detection_count} total_ms={:.3} pps={:.3}",
                elapsed.as_secs_f64() * 1_000.0,
                page_count as f64 / elapsed.as_secs_f64(),
            );
        }
    }

    fn raster_paths(root: &Path) -> Vec<PathBuf> {
        let mut paths = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.is_file()
                    && path
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
            })
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }

    async fn load_slots(paths: &[PathBuf]) -> Vec<crate::OcrProviderBatchSlot> {
        let mut slots = Vec::with_capacity(paths.len());
        for (index, path) in paths.iter().enumerate() {
            slots.push(crate::OcrProviderBatchSlot {
                slot_id: crate::OcrBatchSlotId::new(format!("blind-page-{}", index + 1)).unwrap(),
                input: crate::client::read_source(path).await.unwrap(),
                adjacent_predecessor_slot_id: None,
                text_window: None,
            });
        }
        slots
    }
}
