use std::time::Instant;

use image::{imageops, RgbImage};
use tokio_util::sync::CancellationToken;

use super::*;

#[derive(Default)]
struct ProbeSummary {
    detections: usize,
    canvas_width_sum: u64,
    default_width_detections: usize,
    nonblank_blocks: usize,
    characters: usize,
    confidence_sum: f64,
}

#[test]
#[ignore = "requires the pinned PP-OCRv6 bundle and an explicit retained-raster directory"]
fn real_page_quarter_turn_probe_reports_recognition_work() {
    let image_root = std::path::PathBuf::from(
        std::env::var_os("A3S_PPOCR_V6_ORIENTATION_PROBE_IMAGE_ROOT")
            .expect("A3S_PPOCR_V6_ORIENTATION_PROBE_IMAGE_ROOT must name retained rasters"),
    );
    let first = std::env::var("A3S_PPOCR_V6_ORIENTATION_PROBE_FIRST")
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(1);
    let last = std::env::var("A3S_PPOCR_V6_ORIENTATION_PROBE_LAST")
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(usize::MAX);
    assert!(image_root.is_absolute());
    assert!(first > 0 && first <= last);

    let mut paths = std::fs::read_dir(&image_root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    let source_images = paths
        .into_iter()
        .enumerate()
        .filter(|(index, _)| (first..=last).contains(&(index + 1)))
        .map(|(_, path)| crate::preprocess::decode_image(&std::fs::read(path).unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert!(!source_images.is_empty());

    let assets = crate::assets::resolve_model_assets().unwrap();
    let engine = PpOcrV6Engine::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let permit = engine.native.begin(&cancellation).unwrap();

    for quarter_turns in [0_u8, 1, 3] {
        let images = source_images
            .iter()
            .map(|image| match quarter_turns {
                0 => image.clone(),
                1 => imageops::rotate90(image),
                3 => imageops::rotate270(image),
                _ => unreachable!(),
            })
            .collect::<Vec<RgbImage>>();
        let image_refs = images.iter().collect::<Vec<_>>();
        let started = Instant::now();
        let detected = engine
            .detect_cohorts(&image_refs, usize::MAX, &permit, &cancellation)
            .unwrap();
        let detection_elapsed = started.elapsed();
        let mut summary = ProbeSummary::default();
        for detections in &detected.detections {
            for detection in detections.as_ref().unwrap() {
                let crop = PerspectiveCropPlan::new(detection).unwrap();
                let (width, height) = crop.output_dimensions();
                let canvas_width =
                    recognition_canvas_width(width, height, &engine.recognition_config).unwrap();
                summary.detections += 1;
                summary.canvas_width_sum += u64::from(canvas_width);
                summary.default_width_detections +=
                    usize::from(canvas_width == engine.recognition_config.default_width as u32);
            }
        }
        let recognition_started = Instant::now();
        let outputs = engine
            .recognize_detected_batch(
                &image_refs,
                detected.detections,
                detected.receipts,
                &vec![None; image_refs.len()],
                &permit,
                &cancellation,
            )
            .unwrap();
        let recognition_elapsed = recognition_started.elapsed();
        for output in outputs {
            for block in output.unwrap().blocks {
                if block.text.trim().is_empty() {
                    continue;
                }
                summary.nonblank_blocks += 1;
                summary.characters += block.text.chars().count();
                summary.confidence_sum += f64::from(block.confidence);
            }
        }
        let total_elapsed = started.elapsed();
        let mean_confidence = if summary.nonblank_blocks == 0 {
            0.0
        } else {
            summary.confidence_sum / summary.nonblank_blocks as f64
        };
        eprintln!(
            "A3S_OCR_PAGE_ORIENTATION_PROBE pages={} quarter_turns={} detections={} canvas_width_sum={} default_width_detections={} nonblank_blocks={} characters={} mean_confidence={mean_confidence:.6} detection_ms={:.3} recognition_ms={:.3} total_ms={:.3}",
            image_refs.len(),
            quarter_turns,
            summary.detections,
            summary.canvas_width_sum,
            summary.default_width_detections,
            summary.nonblank_blocks,
            summary.characters,
            detection_elapsed.as_secs_f64() * 1_000.0,
            recognition_elapsed.as_secs_f64() * 1_000.0,
            total_elapsed.as_secs_f64() * 1_000.0,
        );
    }
}
