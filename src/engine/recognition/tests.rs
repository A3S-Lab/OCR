use a3s_power::inference::{
    ExecutionDigest, ExecutionReceipt, ModelIdentity, RuntimeDeviceKind, RuntimeIdentity,
};
use imageproc::point::Point;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::Write as _;

use super::*;

#[test]
#[ignore = "requires the pinned PP-OCRv6 bundle and an explicit inference device"]
fn official_recognition_graph_profiles_one_protocol_bounded_tensor() {
    let assets = crate::assets::resolve_model_assets().unwrap();
    let engine = PpOcrV6Engine::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let permit = engine.native.begin(&cancellation).unwrap();
    let shape = [crate::batch::MAX_BATCH_SLOTS, 3, 48, 320];
    let data = vec![0.0_f32; shape.iter().product()];
    let digest = ExecutionDigest::f32_tensor(&shape, &data);
    let output = engine
        .native
        .recognize_prepared(
            RecognitionInput {
                data,
                shape,
                digest,
            },
            &permit,
            &cancellation,
        )
        .unwrap();
    assert_eq!(output.tensor.shape[0], shape[0]);
}

#[test]
fn same_width_results_preserve_detection_order_and_share_one_receipt() {
    let work = vec![work_item(0, 0, 100.0), work_item(0, 1, 100.0)];
    let batches =
        plan_width_batches(&[320, 320], RuntimeDeviceKind::Cuda, |_| Ok(usize::MAX)).unwrap();
    assert_eq!(batches, vec![vec![0, 1]]);
    let mut states = vec![pending_image(2)];

    apply_recognized_batch(
        &batches[0],
        RecognizedBatch {
            items: vec![recognized("first"), recognized("second")],
            receipt: receipt(),
        },
        &work,
        &mut states,
    )
    .unwrap();

    let output = states
        .pop()
        .unwrap()
        .finish(crate::config::MODEL_FAMILY)
        .unwrap();
    assert_eq!(
        output
            .blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    assert_eq!(output.receipts.len(), 1);
}

#[test]
fn compatible_cross_image_results_retain_identity_and_share_one_receipt() {
    let work = vec![work_item(0, 0, 100.0), work_item(1, 0, 100.0)];
    let batches =
        plan_width_batches(&[320, 320], RuntimeDeviceKind::Cuda, |_| Ok(usize::MAX)).unwrap();
    assert_eq!(batches, vec![vec![0, 1]]);
    let mut states = vec![pending_image(1), pending_image(1)];

    apply_recognized_batch(
        &batches[0],
        RecognizedBatch {
            items: vec![recognized("first-image"), recognized("second-image")],
            receipt: receipt(),
        },
        &work,
        &mut states,
    )
    .unwrap();

    let outputs = states
        .into_iter()
        .map(|state| state.finish(crate::config::MODEL_FAMILY).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(outputs[0].blocks[0].text, "first-image");
    assert_eq!(outputs[1].blocks[0].text, "second-image");
    assert_eq!(outputs[0].receipts, vec![receipt()]);
    assert_eq!(outputs[1].receipts, vec![receipt()]);
}

#[test]
fn recognition_decode_failures_are_isolated_to_their_source_image() {
    let work = vec![work_item(0, 0, 100.0), work_item(1, 0, 100.0)];
    let mut states = vec![pending_image(1), pending_image(1)];

    apply_recognized_batch(
        &[0, 1],
        RecognizedBatch {
            items: vec![
                Err(engine_error("use.ocr.decode_failed", "invalid CTC output")),
                recognized("healthy"),
            ],
            receipt: receipt(),
        },
        &work,
        &mut states,
    )
    .unwrap();

    let mut outputs = states
        .into_iter()
        .map(|state| state.finish(crate::config::MODEL_FAMILY));
    let Err(failed) = outputs.next().unwrap() else {
        panic!("the malformed recognition item must fail its source image");
    };
    assert_eq!(failed.code, "use.ocr.decode_failed");
    let healthy = outputs.next().unwrap().unwrap();
    assert_eq!(healthy.blocks[0].text, "healthy");
    assert_eq!(healthy.receipts, vec![receipt()]);
}

#[test]
fn parallel_crop_preparation_preserves_planned_order() {
    let red = RgbImage::from_pixel(200, 50, image::Rgb([255, 0, 0]));
    let green = RgbImage::from_pixel(200, 50, image::Rgb([0, 255, 0]));
    let images = vec![&red, &green];
    let work = vec![work_item(0, 0, 100.0), work_item(1, 0, 100.0)];
    let mut states = vec![pending_image(1), pending_image(1)];

    let crops = prepare_batch_crops(&images, &work, vec![1, 0], &mut states);

    assert_eq!(
        crops.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert_eq!(crops[0].1.get_pixel(50, 10), &image::Rgb([0, 255, 0]));
    assert_eq!(crops[1].1.get_pixel(50, 10), &image::Rgb([255, 0, 0]));
}

#[test]
fn selected_results_skip_unselected_detections_without_reordering() {
    let mut work = vec![
        work_item(0, 0, 100.0),
        work_item(0, 1, 100.0),
        work_item(0, 2, 100.0),
    ];
    work[1].selected = false;
    let mut states = vec![pending_image_with_selection(vec![true, false, true])];

    apply_recognized_batch(
        &[0, 2],
        RecognizedBatch {
            items: vec![recognized("first"), recognized("third")],
            receipt: receipt(),
        },
        &work,
        &mut states,
    )
    .unwrap();

    let output = states
        .pop()
        .unwrap()
        .finish(crate::config::MODEL_FAMILY)
        .unwrap();
    assert_eq!(
        output
            .blocks
            .iter()
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "third"]
    );
}

#[test]
fn text_window_selection_requires_positive_bounding_box_intersection() {
    let window = OcrPixelWindow {
        left: 10,
        top: 10,
        right: 20,
        bottom: 20,
    };

    assert!(detection_intersects_window(
        &rectangular_detection(5.0, 12.0, 15.0, 18.0),
        window
    ));
    assert!(detection_intersects_window(
        &rectangular_detection(12.0, 12.0, 18.0, 18.0),
        window
    ));
    assert!(!detection_intersects_window(
        &rectangular_detection(0.0, 0.0, 10.0, 10.0),
        window
    ));
    assert!(!detection_intersects_window(
        &rectangular_detection(20.0, 12.0, 25.0, 18.0),
        window
    ));
}

#[test]
fn selected_subset_retains_the_exact_original_batch_canvas_width() {
    let mut work = vec![work_item(0, 0, 335.0), work_item(0, 1, 335.0)];
    work[1].selected = false;
    let states = vec![pending_image_with_selection(vec![true, false])];
    let batch = plan_width_batches(&[335, 335], RuntimeDeviceKind::Cuda, |_| Ok(usize::MAX))
        .unwrap()
        .pop()
        .unwrap();

    assert_eq!(batch, vec![0, 1]);
    assert_eq!(active_work_indices(&batch, &work, &states), vec![0]);
    assert_eq!(planned_batch_canvas_width(&batch, &work).unwrap(), 335);
}

#[test]
fn recognition_reservation_is_derived_from_shape_and_class_count() {
    let config = RecognitionConfig {
        model_variant: crate::config::ModelVariant::Small,
        channels: 3,
        height: 48,
        default_width: 320,
        characters: vec![String::new(); 10],
    };

    assert_eq!(
        recognition_batch_reservation_elements(2, 200, &config).unwrap(),
        58_200
    );
}

#[test]
fn recognition_batches_schedule_largest_declared_work_first_stably() {
    let config = RecognitionConfig {
        model_variant: crate::config::ModelVariant::Small,
        channels: 3,
        height: 48,
        default_width: 320,
        characters: vec![String::new(); 10],
    };
    let work = vec![
        work_item(0, 0, 320.0),
        work_item(0, 1, 800.0),
        work_item(0, 2, 320.0),
        work_item(0, 3, 800.0),
    ];
    let states = vec![pending_image(4)];

    let prioritized =
        prioritize_recognition_batches(vec![vec![0, 2], vec![1], vec![3]], &work, &states, &config)
            .unwrap();

    assert_eq!(prioritized, vec![vec![1], vec![3], vec![0, 2]]);
}

#[test]
fn accelerator_window_cursor_preserves_each_lane_input_budget() {
    let config = RecognitionConfig {
        model_variant: crate::config::ModelVariant::Small,
        channels: 3,
        height: 48,
        default_width: 320,
        characters: vec![String::new(); 10],
    };
    let work = vec![
        work_item(0, 0, 100.0),
        work_item(0, 1, 100.0),
        work_item(0, 2, 100.0),
    ];
    let states = vec![pending_image(3)];
    let mut cursor = RecognitionBatchCursor::new(vec![vec![0], vec![1], vec![2]]);
    let policy = CpuExecutionWindowPolicy {
        maximum_parallel_jobs: 3,
        maximum_reserved_elements: usize::MAX,
    };
    let one_input = recognition_batch_input_bytes(1, 100, &config).unwrap();
    let cancellation = CancellationToken::new();

    for expected_index in 0..3 {
        let window = cursor
            .take_window(
                &work,
                &states,
                &config,
                policy,
                true,
                1,
                one_input,
                &cancellation,
            )
            .unwrap()
            .unwrap();
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].work_indices, vec![expected_index]);
    }
    assert!(cursor
        .take_window(
            &work,
            &states,
            &config,
            policy,
            true,
            1,
            one_input,
            &cancellation,
        )
        .unwrap()
        .is_none());
    assert_eq!(RECOGNITION_PREPARED_WINDOW_DEPTH, 2);
}

#[test]
fn prefetched_window_cannot_replace_a_predecessor_failure() {
    let config = RecognitionConfig {
        model_variant: crate::config::ModelVariant::Small,
        channels: 3,
        height: 48,
        default_width: 320,
        characters: vec![String::new(); 10],
    };
    let mut states = vec![
        ImageRecognition::Failed(engine_error("predecessor", "first failure")),
        pending_image(1),
    ];
    let prepared = |image_index, code| PreparedRecognitionBatch {
        crops: Vec::new(),
        canvas_width: 320,
        input: None,
        input_slots: 0,
        failures: vec![(image_index, engine_error(code, "preparation failure"))],
        crop_preparation: Duration::ZERO,
        tensor_preparation: Duration::ZERO,
    };
    let mut window = vec![prepared(0, "speculative"), prepared(1, "successor")];

    reconcile_prepared_window(&mut window, &[], &mut states, &config, None);

    let ImageRecognition::Failed(predecessor) = &states[0] else {
        panic!("the predecessor failure must remain published");
    };
    assert_eq!(predecessor.code, "predecessor");
    let ImageRecognition::Failed(successor) = &states[1] else {
        panic!("the pending successor image must publish its preparation failure");
    };
    assert_eq!(successor.code, "successor");
}

#[test]
#[ignore = "requires the pinned PP-OCRv6 bundle and an explicit real-page raster"]
fn real_wide_crop_segmentation_probe_reports_geometry_only_accuracy() {
    let source = std::path::PathBuf::from(
        std::env::var_os("A3S_PPOCR_V6_SEGMENTATION_PROBE_IMAGE")
            .expect("A3S_PPOCR_V6_SEGMENTATION_PROBE_IMAGE must name a reviewed raster"),
    );
    let image = crate::preprocess::decode_image(&std::fs::read(source).unwrap()).unwrap();
    let assets = crate::assets::resolve_model_assets().unwrap();
    let engine = PpOcrV6Engine::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let permit = engine.native.begin(&cancellation).unwrap();
    let detected = engine
        .detect_cohorts(&[&image], usize::MAX, &permit, &cancellation)
        .unwrap();
    let detections = detected.detections.into_iter().next().unwrap().unwrap();
    let crops = detections
        .iter()
        .map(|detection| {
            PerspectiveCropPlan::new(detection)
                .unwrap()
                .execute(&image)
                .unwrap()
        })
        .collect::<Vec<_>>();

    for maximum_model_width in [640_u32, 320_u32] {
        let started = std::time::Instant::now();
        let mut evaluated = 0_usize;
        let mut exact = 0_usize;
        let mut baseline_characters = 0_usize;
        let mut edit_distance = 0_usize;
        let mut tile_count = 0_usize;
        for crop in &crops {
            let natural_width =
                recognition_content_width(crop.width(), crop.height(), &engine.recognition_config)
                    .unwrap();
            if natural_width <= maximum_model_width {
                continue;
            }
            let baseline = recognize_probe_crop(&engine, crop, &permit, &cancellation);
            let tiles = split_probe_crop_at_visual_valleys(
                crop,
                maximum_model_width,
                u32::try_from(engine.recognition_config.height).unwrap(),
            );
            let tile_recognitions = tiles
                .iter()
                .map(|tile| recognize_probe_crop(&engine, tile, &permit, &cancellation))
                .collect::<Vec<_>>();
            let candidate = tile_recognitions
                .iter()
                .map(|recognition| recognition.text.as_str())
                .collect::<String>();
            let baseline_chars = baseline.text.chars().collect::<Vec<_>>();
            let candidate_chars = candidate.chars().collect::<Vec<_>>();
            let distance = character_edit_distance(&baseline_chars, &candidate_chars);
            if distance != 0 {
                let baseline_non_whitespace = baseline_chars
                    .iter()
                    .copied()
                    .filter(|character| !character.is_whitespace())
                    .collect::<Vec<_>>();
                let candidate_non_whitespace = candidate_chars
                    .iter()
                    .copied()
                    .filter(|character| !character.is_whitespace())
                    .collect::<Vec<_>>();
                let non_whitespace_distance =
                    character_edit_distance(&baseline_non_whitespace, &candidate_non_whitespace);
                let minimum_tile_confidence = tile_recognitions
                    .iter()
                    .map(|recognition| recognition.confidence)
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                let mean_tile_confidence = tile_recognitions
                    .iter()
                    .map(|recognition| f64::from(recognition.confidence))
                    .sum::<f64>()
                    / tile_recognitions.len() as f64;
                eprintln!(
                    "A3S_OCR_WIDE_SEGMENTATION_MISMATCH maximum_model_width={maximum_model_width} crop_width={} crop_height={} natural_width={natural_width} tiles={} baseline_characters={} candidate_characters={} baseline_confidence={:.6} minimum_tile_confidence={minimum_tile_confidence:.6} mean_tile_confidence={mean_tile_confidence:.6} edit_distance={distance} non_whitespace_distance={non_whitespace_distance}",
                    crop.width(),
                    crop.height(),
                    tiles.len(),
                    baseline_chars.len(),
                    candidate_chars.len(),
                    baseline.confidence,
                );
            }
            evaluated += 1;
            exact += usize::from(distance == 0);
            baseline_characters += baseline_chars.len();
            edit_distance += distance;
            tile_count += tiles.len();
        }
        let similarity = if baseline_characters == 0 {
            1.0
        } else {
            1.0 - edit_distance as f64 / baseline_characters as f64
        };
        eprintln!(
            "A3S_OCR_WIDE_SEGMENTATION_PROBE maximum_model_width={maximum_model_width} evaluated={evaluated} tiles={tile_count} exact={exact} baseline_characters={baseline_characters} edit_distance={edit_distance} similarity={similarity:.6} elapsed_ms={:.3}",
            started.elapsed().as_secs_f64() * 1_000.0,
        );
        assert!(evaluated > 0);
    }
}

#[test]
#[ignore = "requires the pinned PP-OCRv6 bundle and the complete real rider fixture"]
fn real_recognition_batch_sizes_preserve_fixed_detection_semantics() {
    let fixture_root = std::path::PathBuf::from(
        std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root"),
    );
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    assert_eq!(pages.len(), 29);
    let images = pages
        .into_iter()
        .map(|path| crate::preprocess::decode_image(&std::fs::read(path).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let image_refs = images.iter().collect::<Vec<_>>();
    let assets = crate::assets::resolve_model_assets().unwrap();
    let engine = PpOcrV6Engine::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let permit = engine.native.begin(&cancellation).unwrap();
    let detected = engine
        .detect_cohorts(&image_refs, usize::MAX, &permit, &cancellation)
        .unwrap();
    let text_windows = vec![None; image_refs.len()];

    std::env::set_var("A3S_OCR_TEST_RECOGNITION_MAX_BATCH_SIZE", "32");
    let baseline_started = std::time::Instant::now();
    let baseline = engine
        .recognize_detected_batch(
            &image_refs,
            detected.detections.clone(),
            detected.receipts.clone(),
            &text_windows,
            &permit,
            &cancellation,
        )
        .unwrap();
    let baseline_elapsed = baseline_started.elapsed();

    std::env::set_var("A3S_OCR_TEST_RECOGNITION_MAX_BATCH_SIZE", "128");
    let candidate_started = std::time::Instant::now();
    let candidate = engine
        .recognize_detected_batch(
            &image_refs,
            detected.detections.clone(),
            detected.receipts.clone(),
            &text_windows,
            &permit,
            &cancellation,
        )
        .unwrap();
    let candidate_elapsed = candidate_started.elapsed();

    let repeated = engine
        .recognize_detected_batch(
            &image_refs,
            detected.detections.clone(),
            detected.receipts.clone(),
            &text_windows,
            &permit,
            &cancellation,
        )
        .unwrap();
    let third = engine
        .recognize_detected_batch(
            &image_refs,
            detected.detections,
            detected.receipts,
            &text_windows,
            &permit,
            &cancellation,
        )
        .unwrap();
    std::env::remove_var("A3S_OCR_TEST_RECOGNITION_MAX_BATCH_SIZE");

    let mut repeated_changed_confidences = 0_usize;
    let mut repeated_maximum_confidence_difference = 0_f32;
    for (image_index, (candidate, repeated)) in repeated.iter().zip(&third).enumerate() {
        match (candidate, repeated) {
            (Err(candidate), Err(repeated)) => assert_eq!(candidate, repeated),
            (Ok(candidate), Ok(repeated)) => {
                assert_eq!(candidate.model, repeated.model);
                assert_eq!(
                    candidate.receipts.len(),
                    repeated.receipts.len(),
                    "repeated recognition changed receipt cardinality for image {image_index}"
                );
                for (receipt_index, (candidate, repeated)) in candidate
                    .receipts
                    .iter()
                    .zip(&repeated.receipts)
                    .enumerate()
                {
                    assert_eq!(
                        candidate.input, repeated.input,
                        "repeated recognition changed input tensor digest for image {image_index}, receipt {receipt_index}"
                    );
                }
                assert_eq!(
                    candidate.blocks.len(),
                    repeated.blocks.len(),
                    "repeated recognition changed block cardinality for image {image_index}"
                );
                for (block_index, (candidate, repeated)) in
                    candidate.blocks.iter().zip(&repeated.blocks).enumerate()
                {
                    assert_eq!(
                        candidate.polygon.map(|point| [point.x.to_bits(), point.y.to_bits()]),
                        repeated
                            .polygon
                            .map(|point| [point.x.to_bits(), point.y.to_bits()]),
                        "repeated recognition changed source geometry for image {image_index}, block {block_index}"
                    );
                    assert_eq!(
                        candidate.text_rotation_millidegrees,
                        repeated.text_rotation_millidegrees
                    );
                    assert_eq!(
                        candidate.detection_confidence.to_bits(),
                        repeated.detection_confidence.to_bits()
                    );
                    assert_eq!(
                        candidate.text, repeated.text,
                        "repeated recognition changed text for image {image_index}, block {block_index}, polygon={:?}, first_confidence={}, repeated_confidence={}",
                        candidate.polygon,
                        candidate.confidence,
                        repeated.confidence,
                    );
                    repeated_changed_confidences += usize::from(
                        candidate.confidence.to_bits() != repeated.confidence.to_bits(),
                    );
                    repeated_maximum_confidence_difference = repeated_maximum_confidence_difference
                        .max((candidate.confidence - repeated.confidence).abs());
                }
            }
            _ => panic!("repeated recognition changed success status for image {image_index}"),
        }
    }
    eprintln!(
        "A3S_OCR_REPEATED_FIXED_DETECTION_PARITY pages={} batch=128 changed_confidences={repeated_changed_confidences} maximum_confidence_difference={repeated_maximum_confidence_difference:.9e}",
        image_refs.len(),
    );

    let mut compared_confidences = 0_usize;
    let mut changed_confidences = 0_usize;
    let mut maximum_confidence_difference = 0_f32;
    for (image_index, (baseline, candidate)) in baseline.iter().zip(&candidate).enumerate() {
        match (baseline, candidate) {
            (Err(baseline), Err(candidate)) => assert_eq!(baseline, candidate),
            (Ok(baseline), Ok(candidate)) => {
                assert_eq!(baseline.model, candidate.model);
                assert_eq!(
                    baseline.blocks.len(),
                    candidate.blocks.len(),
                    "recognition batch size changed block cardinality for image {image_index}"
                );
                for (block_index, (baseline, candidate)) in
                    baseline.blocks.iter().zip(&candidate.blocks).enumerate()
                {
                    assert_eq!(
                        baseline.polygon.map(|point| [point.x.to_bits(), point.y.to_bits()]),
                        candidate
                            .polygon
                            .map(|point| [point.x.to_bits(), point.y.to_bits()]),
                        "recognition batch size changed source geometry for image {image_index}, block {block_index}"
                    );
                    assert_eq!(
                        baseline.text_rotation_millidegrees,
                        candidate.text_rotation_millidegrees
                    );
                    assert_eq!(
                        baseline.detection_confidence.to_bits(),
                        candidate.detection_confidence.to_bits()
                    );
                    assert_eq!(
                        baseline.text, candidate.text,
                        "recognition batch size changed text for image {image_index}, block {block_index}, polygon={:?}, baseline_confidence={}, candidate_confidence={}",
                        baseline.polygon,
                        baseline.confidence,
                        candidate.confidence,
                    );
                    compared_confidences += 1;
                    changed_confidences += usize::from(
                        baseline.confidence.to_bits() != candidate.confidence.to_bits(),
                    );
                    maximum_confidence_difference = maximum_confidence_difference
                        .max((baseline.confidence - candidate.confidence).abs());
                }
            }
            _ => panic!("recognition batch size changed success status for image {image_index}"),
        }
    }
    eprintln!(
        "A3S_OCR_FIXED_DETECTION_BATCH_PARITY pages={} baseline_batch=32 baseline_ms={:.3} candidate_batch=128 candidate_ms={:.3} compared_confidences={compared_confidences} changed_confidences={changed_confidences} maximum_confidence_difference={maximum_confidence_difference:.9e}",
        image_refs.len(),
        baseline_elapsed.as_secs_f64() * 1_000.0,
        candidate_elapsed.as_secs_f64() * 1_000.0,
    );
}

#[test]
#[ignore = "requires the pinned PP-OCRv6 bundle and the complete real rider fixture"]
fn real_exact_width_recognition_tensor_is_repeatable() {
    let fixture_root = std::path::PathBuf::from(
        std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
            .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root"),
    );
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    assert_eq!(pages.len(), 29);
    let images = pages
        .into_iter()
        .map(|path| crate::preprocess::decode_image(&std::fs::read(path).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let image_refs = images.iter().collect::<Vec<_>>();
    let assets = crate::assets::resolve_model_assets().unwrap();
    let engine = PpOcrV6Engine::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let permit = engine.native.begin(&cancellation).unwrap();
    let detected = engine
        .detect_cohorts(&image_refs, usize::MAX, &permit, &cancellation)
        .unwrap();
    let (states, work) = engine.prepare_recognition_work(
        &image_refs,
        detected.detections,
        detected.receipts,
        &vec![None; image_refs.len()],
    );
    let indices = work
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (item.canvas_width == 320).then_some(index))
        .take(128)
        .collect::<Vec<_>>();
    assert_eq!(indices.len(), 128);
    let crops = indices
        .iter()
        .map(|index| {
            let item = &work[*index];
            item.crop.execute(image_refs[item.image_index]).unwrap()
        })
        .collect::<Vec<_>>();
    let crop_refs = crops.iter().collect::<Vec<_>>();
    let first_input =
        recognition_input_with_canvas_width(&crop_refs, &engine.recognition_config, 320).unwrap();
    let repeated_input =
        recognition_input_with_canvas_width(&crop_refs, &engine.recognition_config, 320).unwrap();
    assert_eq!(first_input.digest, repeated_input.digest);
    let first = engine
        .native
        .recognize_prepared(first_input, &permit, &cancellation)
        .unwrap();
    let repeated = engine
        .native
        .recognize_prepared(repeated_input, &permit, &cancellation)
        .unwrap();
    assert_eq!(first.tensor.shape, repeated.tensor.shape);
    let changed = first
        .tensor
        .values
        .iter()
        .zip(&repeated.tensor.values)
        .enumerate()
        .filter_map(|(index, (first, repeated))| {
            (first.to_bits() != repeated.to_bits()).then_some((index, *first, *repeated))
        })
        .collect::<Vec<_>>();
    eprintln!(
        "A3S_OCR_EXACT_WIDTH_REPEAT changed_values={} first={:?}",
        changed.len(),
        changed.first(),
    );

    let canvas_widths = work
        .iter()
        .map(|item| item.canvas_width)
        .collect::<Vec<_>>();
    let maximum_tensor_elements = engine.native.maximum_tensor_elements();
    let batches = plan_width_batches(
        &canvas_widths,
        engine.native.runtime_device_kind(),
        |canvas_width| {
            let per_crop = recognition_batch_reservation_elements(
                1,
                canvas_width,
                &engine.recognition_config,
            )?;
            Ok(maximum_tensor_elements
                .checked_div(per_crop)
                .unwrap_or(0)
                .max(1))
        },
    )
    .unwrap();
    let batches =
        prioritize_recognition_batches(batches, &work, &states, &engine.recognition_config)
            .unwrap();
    let inputs = batches
        .into_iter()
        .map(|batch| {
            let active = active_work_indices(&batch, &work, &states);
            let canvas_width = planned_batch_canvas_width(&batch, &work).unwrap();
            let prepared = prepare_recognition_batch(
                &image_refs,
                &work,
                active,
                canvas_width,
                &engine.recognition_config,
            );
            assert!(prepared.failures.is_empty());
            prepared.input.unwrap().unwrap()
        })
        .collect::<Vec<_>>();
    eprintln!(
        "A3S_OCR_EXACT_WIDTH_SEQUENCE batches={} slots={} maximum_width={}",
        inputs.len(),
        inputs.iter().map(|input| input.shape[0]).sum::<usize>(),
        inputs.iter().map(|input| input.shape[3]).max().unwrap_or(0),
    );

    let execute_outputs = || {
        inputs
            .iter()
            .cloned()
            .map(|input| {
                engine
                    .native
                    .recognize_prepared(input, &permit, &cancellation)
                    .unwrap()
                    .tensor
            })
            .collect::<Vec<_>>()
    };
    let first_outputs = execute_outputs();
    let repeated_outputs = execute_outputs();
    let changed_output_batches = first_outputs
        .iter()
        .zip(&repeated_outputs)
        .enumerate()
        .filter_map(|(index, (first, repeated))| {
            (first.shape != repeated.shape
                || first
                    .values
                    .iter()
                    .zip(&repeated.values)
                    .any(|(first, repeated)| first.to_bits() != repeated.to_bits()))
            .then_some(index)
        })
        .collect::<Vec<_>>();

    let execute_features = || {
        inputs
            .iter()
            .cloned()
            .map(|input| {
                engine
                    .native
                    .recognize_prepared_features(input, &permit, &cancellation)
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let first_features = execute_features();
    let repeated_features = execute_features();
    let changed_feature_batches = first_features
        .iter()
        .zip(&repeated_features)
        .enumerate()
        .filter_map(|(index, (first, repeated))| {
            (first.shape != repeated.shape
                || first
                    .values
                    .iter()
                    .zip(&repeated.values)
                    .any(|(first, repeated)| first.to_bits() != repeated.to_bits()))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    assert!(
        changed_output_batches.is_empty(),
        "repeated planned inference changed projected batches {changed_output_batches:?}; feature batches changed {changed_feature_batches:?}",
    );
    assert!(
        changed_feature_batches.is_empty(),
        "repeated planned feature inference changed batches {changed_feature_batches:?}",
    );
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RecognitionTensorCapture {
    schema_version: u32,
    dtype: &'static str,
    byte_order: &'static str,
    stride: usize,
    quarter_turns: u8,
    tensor_bytes: u64,
    tensor_sha256: String,
    records: Vec<RecognitionTensorRecord>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RecognitionTensorRecord {
    page: u32,
    detection_index: usize,
    shape: [usize; 4],
    byte_offset: u64,
    byte_length: u64,
    sha256: String,
    text: String,
    confidence: f32,
}

#[test]
#[ignore = "requires the pinned PP-OCRv6 bundle and an explicit retained-raster directory"]
fn real_recognition_tensor_capture_uses_exact_rust_preprocessing() {
    let image_root = std::path::PathBuf::from(
        std::env::var_os("A3S_PPOCR_V6_CAPTURE_IMAGE_ROOT")
            .expect("A3S_PPOCR_V6_CAPTURE_IMAGE_ROOT must name retained page rasters"),
    );
    let output_root = std::path::PathBuf::from(
        std::env::var_os("A3S_PPOCR_V6_CAPTURE_OUTPUT_ROOT")
            .expect("A3S_PPOCR_V6_CAPTURE_OUTPUT_ROOT must name a new output directory"),
    );
    assert!(image_root.is_absolute());
    assert!(output_root.is_absolute());
    std::fs::create_dir(&output_root).expect("capture output root must be new");
    let stride = std::env::var("A3S_PPOCR_V6_CAPTURE_STRIDE")
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(1);
    assert!(stride > 0);
    let quarter_turns = std::env::var("A3S_PPOCR_V6_CAPTURE_QUARTER_TURNS")
        .ok()
        .map(|value| value.parse::<u8>().unwrap())
        .unwrap_or(0);
    assert!(quarter_turns <= 3);

    let mut pages = std::fs::read_dir(&image_root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("page-") && name.ends_with(".png"))
        })
        .collect::<Vec<_>>();
    pages.sort();
    assert!(!pages.is_empty());

    let assets = crate::assets::resolve_model_assets().unwrap();
    let engine = PpOcrV6Engine::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let permit = engine.native.begin(&cancellation).unwrap();
    let tensor_path = output_root.join("inputs.f32le");
    let tensor_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tensor_path)
        .unwrap();
    let mut tensor_writer = std::io::BufWriter::new(tensor_file);
    let mut tensor_digest = Sha256::new();
    let mut tensor_bytes = 0_u64;
    let mut records = Vec::new();
    let mut global_detection_index = 0_usize;

    for (page_index, path) in pages.iter().enumerate() {
        let page = u32::try_from(page_index + 1).unwrap();
        let source = crate::preprocess::decode_image(&std::fs::read(path).unwrap()).unwrap();
        let image = match quarter_turns {
            0 => source,
            1 => image::imageops::rotate90(&source),
            2 => image::imageops::rotate180(&source),
            3 => image::imageops::rotate270(&source),
            _ => unreachable!(),
        };
        let detected = engine
            .detect_cohorts(&[&image], usize::MAX, &permit, &cancellation)
            .unwrap();
        let detections = detected.detections.into_iter().next().unwrap().unwrap();
        for (detection_index, detection) in detections.into_iter().enumerate() {
            let selected = global_detection_index % stride == 0;
            global_detection_index += 1;
            if !selected {
                continue;
            }
            let crop = PerspectiveCropPlan::new(&detection)
                .unwrap()
                .execute(&image)
                .unwrap();
            let canvas_width =
                recognition_canvas_width(crop.width(), crop.height(), &engine.recognition_config)
                    .unwrap();
            let input = recognition_input_with_canvas_width(
                &[&crop],
                &engine.recognition_config,
                canvas_width,
            )
            .unwrap();
            let shape = input.shape;
            let byte_offset = tensor_bytes;
            let mut bytes = Vec::with_capacity(input.data.len() * std::mem::size_of::<f32>());
            for value in &input.data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let byte_length = u64::try_from(bytes.len()).unwrap();
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            tensor_writer.write_all(&bytes).unwrap();
            tensor_digest.update(&bytes);
            tensor_bytes = tensor_bytes.checked_add(byte_length).unwrap();
            let recognition = engine
                .native
                .recognize_prepared(input, &permit, &cancellation)
                .unwrap();
            let decoded = decode_ctc_top1(
                &recognition.tensor.values,
                &recognition.tensor.shape,
                &engine.recognition_config,
            )
            .unwrap();
            records.push(RecognitionTensorRecord {
                page,
                detection_index,
                shape,
                byte_offset,
                byte_length,
                sha256,
                text: decoded.text,
                confidence: decoded.confidence,
            });
        }
        eprintln!(
            "A3S_OCR_RECOGNITION_CAPTURE page={page} detections={} captured={} tensor_bytes={tensor_bytes}",
            global_detection_index,
            records.len(),
        );
    }
    tensor_writer.flush().unwrap();
    drop(tensor_writer);
    assert_eq!(std::fs::metadata(&tensor_path).unwrap().len(), tensor_bytes);
    let manifest = RecognitionTensorCapture {
        schema_version: 1,
        dtype: "f32",
        byte_order: "little-endian",
        stride,
        quarter_turns,
        tensor_bytes,
        tensor_sha256: format!("{:x}", tensor_digest.finalize()),
        records,
    };
    assert!(!manifest.records.is_empty());
    let manifest_path = output_root.join("manifest.json");
    let mut manifest_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(manifest_path)
        .unwrap();
    serde_json::to_writer_pretty(&mut manifest_file, &manifest).unwrap();
    manifest_file.flush().unwrap();
    eprintln!(
        "A3S_OCR_RECOGNITION_CAPTURE_COMPLETE pages={} detections={} captured={} tensor_bytes={} tensor_sha256={}",
        pages.len(),
        global_detection_index,
        manifest.records.len(),
        manifest.tensor_bytes,
        manifest.tensor_sha256,
    );
}

fn recognize_probe_crop(
    engine: &PpOcrV6Engine,
    crop: &RgbImage,
    permit: &a3s_power::inference::ExecutionPermit,
    cancellation: &CancellationToken,
) -> Recognition {
    let canvas_width =
        recognition_canvas_width(crop.width(), crop.height(), &engine.recognition_config).unwrap();
    engine
        .recognize_crop_batch(&[crop], canvas_width, permit, cancellation, None)
        .unwrap()
        .items
        .into_iter()
        .next()
        .unwrap()
        .unwrap()
}

fn split_probe_crop_at_visual_valleys(
    crop: &RgbImage,
    maximum_model_width: u32,
    model_height: u32,
) -> Vec<RgbImage> {
    let maximum_source_width = u64::from(maximum_model_width)
        .saturating_mul(u64::from(crop.height()))
        .checked_div(u64::from(model_height))
        .and_then(|width| u32::try_from(width).ok())
        .unwrap()
        .max(1);
    if crop.width() <= maximum_source_width {
        return vec![crop.clone()];
    }
    let background = probe_background(crop);
    let mut ranges = Vec::new();
    let mut start = 0_u32;
    while crop.width() - start > maximum_source_width {
        let ideal = start + maximum_source_width;
        let radius = crop.height().min(maximum_source_width / 4).max(1);
        let left = ideal
            .saturating_sub(radius)
            .max(start + maximum_source_width / 2);
        let right = ideal
            .saturating_add(radius)
            .min(crop.width().saturating_sub(maximum_source_width / 3))
            .max(left + 1);
        let cut = (left..right)
            .min_by_key(|x| probe_column_activity(crop, *x, background))
            .unwrap_or(ideal);
        ranges.push(start..cut);
        start = cut;
    }
    ranges.push(start..crop.width());
    ranges
        .into_iter()
        .map(|range| {
            image::imageops::crop_imm(crop, range.start, 0, range.end - range.start, crop.height())
                .to_image()
        })
        .collect()
}

fn probe_background(crop: &RgbImage) -> [u8; 3] {
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for point in [
        (0, 0),
        (crop.width() - 1, 0),
        (0, crop.height() - 1),
        (crop.width() - 1, crop.height() - 1),
    ] {
        let pixel = crop.get_pixel(point.0, point.1);
        for channel in 0..3 {
            channels[channel].push(pixel[channel]);
        }
    }
    channels
        .each_mut()
        .iter_mut()
        .for_each(|values| values.sort_unstable());
    [channels[0][2], channels[1][2], channels[2][2]]
}

fn probe_column_activity(crop: &RgbImage, x: u32, background: [u8; 3]) -> u64 {
    (0..crop.height())
        .map(|y| {
            let pixel = crop.get_pixel(x, y);
            (0..3)
                .map(|channel| u64::from(pixel[channel].abs_diff(background[channel])))
                .sum::<u64>()
        })
        .sum()
}

fn character_edit_distance(left: &[char], right: &[char]) -> usize {
    let mut preceding = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0_usize; right.len() + 1];
    for (left_index, left_character) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_character) in right.iter().enumerate() {
            current[right_index + 1] = (preceding[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(preceding[right_index] + usize::from(left_character != right_character));
        }
        std::mem::swap(&mut preceding, &mut current);
    }
    preceding[right.len()]
}

fn pending_image(blocks: usize) -> ImageRecognition {
    pending_image_with_selection(vec![true; blocks])
}

fn pending_image_with_selection(selected: Vec<bool>) -> ImageRecognition {
    ImageRecognition::Pending {
        blocks: vec![None; selected.len()],
        selected,
        receipts: Vec::new(),
    }
}

fn rectangular_detection(left: f32, top: f32, right: f32, bottom: f32) -> Detection {
    Detection {
        polygon: [
            Point::new(left, top),
            Point::new(right, top),
            Point::new(right, bottom),
            Point::new(left, bottom),
        ],
        confidence: 0.9,
    }
}

fn work_item(image_index: usize, detection_index: usize, width: f32) -> RecognitionWorkItem {
    let detection = rectangular_detection(0.0, 0.0, width, 20.0);
    RecognitionWorkItem {
        image_index,
        detection_index,
        crop: PerspectiveCropPlan::new(&detection).unwrap(),
        detection,
        content_width: width as u32,
        canvas_width: width as u32,
        selected: true,
    }
}

fn recognized(text: &str) -> UseResult<Recognition> {
    Ok(Recognition {
        text: text.to_string(),
        confidence: 0.95,
    })
}

fn receipt() -> ExecutionReceipt {
    ExecutionReceipt {
        schema: ExecutionReceipt::SCHEMA.to_string(),
        model: ModelIdentity::new("pp-ocr-v6-small-recognition", "fixture", "0".repeat(64)),
        runtime: RuntimeIdentity {
            name: "a3s-power-native".to_string(),
            version: "fixture".to_string(),
            device: "cpu".to_string(),
        },
        input: ExecutionDigest::utf8_text("input"),
        output: ExecutionDigest::utf8_text("output"),
        accelerator: None,
        microbatch: None,
    }
}
