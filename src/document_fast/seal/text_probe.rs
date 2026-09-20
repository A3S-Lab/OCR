//! Isolated probe for an independently trained seal-text detector.
//!
//! This module is test-only until accuracy, provenance, and throughput gates
//! pass. It deliberately emits text-region observations rather than claiming
//! that those polygons are complete seal envelopes.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, ExecutionDigest, InferenceLimits, ModelIdentity,
    TensorInput, TensorOutput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::config::DetectionConfig;
use crate::postprocess::{detection_boxes_in_content, Detection};
use crate::preprocess::detection_input_with_resize_long;

use super::super::wired::PixelRect;
use super::assets::PicodetLayoutAssets;
use super::native::NativePicodetLayout;
use super::preprocess::{
    full_page_view as layout_full_page_view, view_tensor as layout_view_tensor,
};
use super::profile::{PicodetLayoutProfile, KEEP_TOP_K, NMS_IOU_THRESHOLD, SCORE_THRESHOLD};

const MODEL_ENV: &str = "A3S_OCR_SEAL_TEXT_MODEL_DIR";
const FAMILY: &str = "pp-ocr-v4-mobile-seal-det";
const ROLE: &str = "seal-text-detection";
const REVISION: &str = "paddlex-paddle3.0.0";
const SOURCE_GRAPH_SHA256: &str =
    "39854c9489b4cb0c5f47ff361b31e966d56b791b92a7ac99686f8a1a2fd8e5c1";
const GRAPH_SHA256: &str = "2e9d197579095f5816dbf8c2dd69514e3676b95bf87eaa103de0c04f4d31fbae";
const WEIGHTS_FILE_SHA256: &str =
    "22d45b9b894c9cc5eefcd52dc874448004f307297f727adad80d110e1a457e23";
const WEIGHTS_COLLECTION_SHA256: &str =
    "87a2ca81a27051c17ca5aad60f05cbc161fa0def010c983d435028a1916259db";
const WEIGHTS_BYTES: u64 = 4_709_036;
const RESIZE_LONG: u32 = 736;
const RESIZE_STRIDE: u32 = 128;

struct ProbeAssets {
    root: PathBuf,
    graph: String,
}

impl ProbeAssets {
    fn from_env() -> UseResult<Self> {
        let root = std::env::var_os(MODEL_ENV).ok_or_else(|| {
            probe_error(format!(
                "Set {MODEL_ENV} to the isolated reviewed seal-text model bundle."
            ))
        })?;
        Self::from_root(Path::new(&root))
    }

    fn from_root(root: &Path) -> UseResult<Self> {
        let root = std::fs::canonicalize(root).map_err(|error| {
            probe_error(format!(
                "Failed to resolve seal-text probe root '{}': {error}",
                root.display()
            ))
        })?;
        let graph_path = resolved_file(&root, "graph.json")?;
        let weights_path = resolved_file(&root, "model.safetensors")?;
        let graph_hash = file_sha256(&graph_path)?;
        if graph_hash != GRAPH_SHA256 {
            return Err(probe_error(format!(
                "Seal-text probe graph digest is {graph_hash}, expected {GRAPH_SHA256}."
            )));
        }
        let metadata = std::fs::metadata(&weights_path).map_err(|error| {
            probe_error(format!(
                "Failed to inspect seal-text probe weights '{}': {error}",
                weights_path.display()
            ))
        })?;
        let weights_hash = file_sha256(&weights_path)?;
        if metadata.len() != WEIGHTS_BYTES || weights_hash != WEIGHTS_FILE_SHA256 {
            return Err(probe_error(
                "Seal-text probe weights do not match the exact reviewed artifact.",
            ));
        }
        let graph = std::fs::read_to_string(&graph_path).map_err(|error| {
            probe_error(format!(
                "Failed to read seal-text probe graph '{}': {error}",
                graph_path.display()
            ))
        })?;
        Ok(Self { root, graph })
    }
}

struct ProbeRun {
    detections: Vec<Detection>,
    preprocess: Duration,
    inference: Duration,
    postprocess: Duration,
}

struct SealTextProbe {
    runtime: EmbeddedRuntime,
    graph: GraphExecutor,
    identity: ModelIdentity,
}

impl SealTextProbe {
    fn load(assets: &ProbeAssets) -> UseResult<Self> {
        Self::load_with_concurrency(assets, 1)
    }

    fn load_with_concurrency(assets: &ProbeAssets, concurrency: usize) -> UseResult<Self> {
        if concurrency == 0 {
            return Err(probe_error(
                "Seal-text probe concurrency must be greater than zero.",
            ));
        }
        let limits = InferenceLimits {
            max_concurrent_requests: concurrency,
            max_queued_requests: concurrency,
            ..InferenceLimits::default()
        };
        let runtime = EmbeddedRuntime::new(DevicePreference::Auto, limits.clone())
            .map_err(|error| power_error("initialize", error))?;
        let weights = Arc::new(
            WeightStore::open(&assets.root, &limits)
                .map_err(|error| power_error("open weights", error))?,
        );
        weights
            .verify_integrity(FAMILY, WEIGHTS_COLLECTION_SHA256)
            .map_err(|error| power_error("verify weights", error))?;
        let graph_identity = GraphIdentity::new(FAMILY, ROLE, "onnx", SOURCE_GRAPH_SHA256, 14);
        let plan = GraphPlan::parse(&assets.graph, &graph_identity, &weights, &limits)
            .map_err(|error| power_error("validate graph", error))?;
        let graph = GraphExecutor::new(plan, weights, runtime.clone())
            .map_err(|error| power_error("materialize graph", error))?;
        Ok(Self {
            runtime,
            graph,
            identity: ModelIdentity::new(FAMILY, REVISION, WEIGHTS_COLLECTION_SHA256),
        })
    }

    fn detect(&self, image: &RgbImage, cancellation: &CancellationToken) -> UseResult<ProbeRun> {
        let preprocess_started = Instant::now();
        let input =
            detection_input_with_resize_long(image, &model_config(), RESIZE_LONG, RESIZE_STRIDE)?;
        let preprocess = preprocess_started.elapsed();
        let geometry = input.geometry;
        let tensor = TensorInput::new(input.shape.to_vec(), input.data, self.runtime.limits())
            .map_err(|error| power_error("validate input", error))?;
        let input_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let permit = self
            .runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit request", error))?;
        let inference_started = Instant::now();
        let output = self
            .graph
            .run(tensor, &permit, cancellation)
            .map_err(|error| power_error("execute graph", error))?;
        let inference = inference_started.elapsed();
        if output.shape.len() != 4 || output.shape[0] != 1 || output.shape[1] != 1 {
            return Err(probe_error(format!(
                "Seal-text probe output must be [1, 1, H, W], found {:?}.",
                output.shape
            )));
        }
        let output_digest = ExecutionDigest::f32_tensor(&output.shape, &output.values);
        let _receipt = self
            .runtime
            .receipt(self.identity.clone(), input_digest, output_digest);
        let postprocess_started = Instant::now();
        let detections = detection_boxes_in_content(
            &output.values,
            &output.shape,
            geometry.content_width,
            geometry.content_height,
            geometry.original_width,
            geometry.original_height,
            &model_config(),
        )?;
        let postprocess = postprocess_started.elapsed();
        Ok(ProbeRun {
            detections,
            preprocess,
            inference,
            postprocess,
        })
    }
}

fn model_config() -> DetectionConfig {
    DetectionConfig {
        model_variant: crate::config::ModelVariant::Small,
        scale: 1.0 / 255.0,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        threshold: 0.2,
        box_threshold: 0.6,
        max_candidates: 1_000,
        unclip_ratio: 0.5,
    }
}

fn resolved_file(root: &Path, relative: &str) -> UseResult<PathBuf> {
    let requested = root.join(relative);
    let resolved = std::fs::canonicalize(&requested).map_err(|error| {
        probe_error(format!(
            "Required seal-text probe asset '{}' is unreadable: {error}",
            requested.display()
        ))
    })?;
    if !resolved.starts_with(root)
        || !std::fs::metadata(&resolved)
            .map(|metadata| metadata.is_file() && metadata.len() > 0)
            .unwrap_or(false)
    {
        return Err(probe_error(format!(
            "Required seal-text probe asset '{}' is not a bounded regular file.",
            requested.display()
        )));
    }
    Ok(resolved)
}

fn file_sha256(path: &Path) -> UseResult<String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| probe_error(format!("Failed to open '{}': {error}", path.display())))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            probe_error(format!("Failed to hash '{}': {error}", path.display()))
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn probe_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_text_probe_invalid", message)
}

fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    probe_error(format!("Failed to {action} the seal-text probe: {error}"))
}

#[test]
fn reviewed_probe_assets_are_exact_when_configured() {
    let Some(root) = std::env::var_os(MODEL_ENV) else {
        return;
    };
    let assets = ProbeAssets::from_root(Path::new(&root)).unwrap();
    let graph: serde_json::Value = serde_json::from_str(&assets.graph).unwrap();
    assert_eq!(graph["family"], FAMILY);
    assert_eq!(graph["role"], ROLE);
    assert_eq!(graph["source"]["sha256"], SOURCE_GRAPH_SHA256);
    assert_eq!(graph["nodes"].as_array().unwrap().len(), 526);
    assert_eq!(graph["initializers"].as_array().unwrap().len(), 246);
}

#[test]
#[ignore = "requires the exact seal-text bundle and a real raster fixture"]
fn real_fixture_reports_independent_seal_text_observations() {
    let assets = ProbeAssets::from_env().unwrap();
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name a read-only validation fixture");
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    assert!(!pages.is_empty());

    let detector = SealTextProbe::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let started = Instant::now();
    let mut preprocess = Duration::ZERO;
    let mut inference = Duration::ZERO;
    let mut postprocess = Duration::ZERO;
    let mut observation_count = 0_usize;
    for page in &pages {
        let image = image::open(page).unwrap().into_rgb8();
        let run = detector.detect(&image, &cancellation).unwrap();
        preprocess += run.preprocess;
        inference += run.inference;
        postprocess += run.postprocess;
        observation_count += run.detections.len();
        eprintln!(
            "SEAL_TEXT_PROBE page={} observations={} detections={:?}",
            page.file_name().unwrap().to_string_lossy(),
            run.detections.len(),
            run.detections
        );
    }
    let elapsed = started.elapsed();
    eprintln!(
        "SEAL_TEXT_PROBE_SUMMARY pages={} observations={} total_ms={:.3} pps={:.3} preprocess_ms={:.3} inference_ms={:.3} postprocess_ms={:.3}",
        pages.len(),
        observation_count,
        elapsed.as_secs_f64() * 1_000.0,
        pages.len() as f64 / elapsed.as_secs_f64(),
        preprocess.as_secs_f64() * 1_000.0,
        inference.as_secs_f64() * 1_000.0,
        postprocess.as_secs_f64() * 1_000.0,
    );
}

#[test]
#[ignore = "requires the exact seal-text bundle and a real raster fixture"]
fn real_fixture_reports_orthogonal_overlap_evidence() {
    let assets = ProbeAssets::from_env().unwrap();
    let layout_assets = PicodetLayoutAssets::from_env().unwrap();
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name a read-only validation fixture");
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    assert!(!pages.is_empty());

    let detector = SealTextProbe::load(&assets).unwrap();
    let layout_detector = NativePicodetLayout::load(&layout_assets).unwrap();
    let cancellation = CancellationToken::new();
    let started = Instant::now();
    let mut observation_count = 0_usize;
    for page in &pages {
        let image = image::open(page).unwrap().into_rgb8();
        let source_view = PixelRect {
            x: 0,
            y: 0,
            width: image.width(),
            height: image.height(),
        };
        let direct = detector.detect(&image, &cancellation).unwrap();
        let rotated_image = image::imageops::rotate90(&image);
        let orthogonal = detector.detect(&rotated_image, &cancellation).unwrap();
        let layout_view = layout_full_page_view(&image);
        let layout_permit = layout_detector.begin(&cancellation).unwrap();
        let layout_output = layout_detector
            .infer_batch(
                layout_view_tensor(&image, layout_view, layout_assets.profile).unwrap(),
                1,
                &layout_permit,
                &cancellation,
            )
            .unwrap();
        let layout_images =
            layout_image_observations(&layout_output.tensor.values, layout_assets.profile, &image);
        let restored = orthogonal
            .detections
            .iter()
            .map(|detection| {
                (
                    rotated_detection_source_bounds(detection, source_view),
                    detection.confidence,
                )
            })
            .collect::<Vec<_>>();
        let evidence = direct
            .detections
            .iter()
            .map(|detection| {
                let direct_bounds = detection_bounds(detection, image.width(), image.height());
                let best = restored
                    .iter()
                    .map(|(bounds, confidence)| {
                        (
                            intersection_over_union(direct_bounds, *bounds),
                            *bounds,
                            *confidence,
                        )
                    })
                    .max_by(|left, right| left.0.total_cmp(&right.0));
                let best_layout = layout_images
                    .iter()
                    .map(|(bounds, confidence)| {
                        (
                            intersection_over_union(direct_bounds, *bounds),
                            intersection_over_smaller(direct_bounds, *bounds),
                            *bounds,
                            *confidence,
                        )
                    })
                    .max_by(|left, right| left.0.total_cmp(&right.0));
                (direct_bounds, detection.confidence, best, best_layout)
            })
            .collect::<Vec<_>>();
        let fused = fused_layout_image_observations(
            &direct.detections,
            &restored,
            &layout_images,
            image.width(),
            image.height(),
        );
        observation_count += evidence.len();
        eprintln!(
            "SEAL_TEXT_ORTHOGONAL_OVERLAP page={} direct={} orthogonal={} layout_images={} fused={fused:?} evidence={evidence:?}",
            page.file_name().unwrap().to_string_lossy(),
            direct.detections.len(),
            restored.len(),
            layout_images.len(),
        );
    }
    let elapsed = started.elapsed();
    eprintln!(
        "SEAL_TEXT_ORTHOGONAL_OVERLAP_SUMMARY pages={} direct_observations={observation_count} total_ms={:.3} pps={:.3}",
        pages.len(),
        elapsed.as_secs_f64() * 1_000.0,
        pages.len() as f64 / elapsed.as_secs_f64(),
    );
}

#[test]
#[ignore = "requires the exact seal-text bundle and a real raster fixture"]
fn batched_fixture_inference_is_exact_and_reports_throughput() {
    let assets = ProbeAssets::from_env().unwrap();
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name a read-only validation fixture");
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    pages.truncate(8);
    assert_eq!(pages.len(), 8);

    let inputs = pages
        .iter()
        .map(|page| {
            let image = image::open(page).unwrap().into_rgb8();
            detection_input_with_resize_long(&image, &model_config(), RESIZE_LONG, RESIZE_STRIDE)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let sample_shape = inputs[0].shape;
    assert!(inputs.iter().all(|input| input.shape == sample_shape));

    let detector = SealTextProbe::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let scalar_started = Instant::now();
    let scalar_outputs = inputs
        .iter()
        .map(|input| {
            execute_probe_tensor(
                &detector,
                input.shape.to_vec(),
                input.data.clone(),
                &cancellation,
            )
        })
        .collect::<Vec<_>>();
    let scalar_elapsed = scalar_started.elapsed();
    eprintln!(
        "SEAL_TEXT_BATCH_PROBE batch_size=1 pages={} total_ms={:.3} pps={:.3} exact=true",
        inputs.len(),
        scalar_elapsed.as_secs_f64() * 1_000.0,
        inputs.len() as f64 / scalar_elapsed.as_secs_f64(),
    );

    for batch_size in [2_usize, 4, 8] {
        let started = Instant::now();
        for (batch_index, batch) in inputs.chunks(batch_size).enumerate() {
            let mut values = Vec::with_capacity(batch.iter().map(|input| input.data.len()).sum());
            for input in batch {
                values.extend_from_slice(&input.data);
            }
            let shape = vec![
                batch.len(),
                sample_shape[1],
                sample_shape[2],
                sample_shape[3],
            ];
            let output = execute_probe_tensor(&detector, shape, values, &cancellation);
            assert_eq!(
                output.shape,
                [
                    batch.len(),
                    scalar_outputs[0].shape[1],
                    scalar_outputs[0].shape[2],
                    scalar_outputs[0].shape[3],
                ]
            );
            let expected = scalar_outputs
                [batch_index * batch_size..batch_index * batch_size + batch.len()]
                .iter()
                .flat_map(|output| output.values.iter().copied())
                .collect::<Vec<_>>();
            assert_f32_bits_equal(&output.values, &expected);
        }
        let elapsed = started.elapsed();
        eprintln!(
            "SEAL_TEXT_BATCH_PROBE batch_size={batch_size} pages={} total_ms={:.3} pps={:.3} exact=true",
            inputs.len(),
            elapsed.as_secs_f64() * 1_000.0,
            inputs.len() as f64 / elapsed.as_secs_f64(),
        );
    }
}

#[test]
#[ignore = "requires the exact seal-text bundle and a real raster fixture"]
fn concurrent_fixture_inference_is_exact_and_reports_throughput() {
    let assets = ProbeAssets::from_env().unwrap();
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name a read-only validation fixture");
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    assert!(!pages.is_empty());
    let inputs = pages
        .iter()
        .map(|page| {
            let image = image::open(page).unwrap().into_rgb8();
            detection_input_with_resize_long(&image, &model_config(), RESIZE_LONG, RESIZE_STRIDE)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let sample_shape = inputs[0].shape;
    assert!(inputs.iter().all(|input| input.shape == sample_shape));

    let cancellation = CancellationToken::new();
    let scalar_detector = SealTextProbe::load(&assets).unwrap();
    let scalar_started = Instant::now();
    let expected = inputs
        .iter()
        .map(|input| {
            execute_probe_tensor(
                &scalar_detector,
                input.shape.to_vec(),
                input.data.clone(),
                &cancellation,
            )
        })
        .collect::<Vec<_>>();
    let scalar_elapsed = scalar_started.elapsed();
    eprintln!(
        "SEAL_TEXT_CONCURRENCY_PROBE concurrency=1 pages={} total_ms={:.3} pps={:.3} exact=true",
        inputs.len(),
        scalar_elapsed.as_secs_f64() * 1_000.0,
        inputs.len() as f64 / scalar_elapsed.as_secs_f64(),
    );

    for concurrency in [2_usize, 4, 8] {
        let detector = SealTextProbe::load_with_concurrency(&assets, concurrency).unwrap();
        let started = Instant::now();
        let mut actual = std::thread::scope(|scope| {
            let handles = (0..concurrency)
                .map(|worker| {
                    let detector = &detector;
                    let inputs = &inputs;
                    let cancellation = &cancellation;
                    scope.spawn(move || {
                        (worker..inputs.len())
                            .step_by(concurrency)
                            .map(|index| {
                                let input = &inputs[index];
                                (
                                    index,
                                    execute_probe_tensor(
                                        detector,
                                        input.shape.to_vec(),
                                        input.data.clone(),
                                        cancellation,
                                    ),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let elapsed = started.elapsed();
        actual.sort_by_key(|(index, _)| *index);
        assert_eq!(actual.len(), expected.len());
        for (expected_index, ((index, actual), expected)) in
            actual.iter().zip(&expected).enumerate()
        {
            assert_eq!(*index, expected_index);
            assert_eq!(actual.shape, expected.shape);
            assert_f32_bits_equal(&actual.values, &expected.values);
        }
        eprintln!(
            "SEAL_TEXT_CONCURRENCY_PROBE concurrency={concurrency} pages={} total_ms={:.3} pps={:.3} exact=true",
            inputs.len(),
            elapsed.as_secs_f64() * 1_000.0,
            inputs.len() as f64 / elapsed.as_secs_f64(),
        );
    }

    for (batch_size, concurrency) in [(2_usize, 2_usize), (4, 2), (2, 4)] {
        let detector = SealTextProbe::load_with_concurrency(&assets, concurrency).unwrap();
        let batches = inputs.chunks(batch_size).collect::<Vec<_>>();
        let started = Instant::now();
        let mut actual = std::thread::scope(|scope| {
            let handles = (0..concurrency)
                .map(|worker| {
                    let detector = &detector;
                    let batches = &batches;
                    let cancellation = &cancellation;
                    scope.spawn(move || {
                        (worker..batches.len())
                            .step_by(concurrency)
                            .map(|batch_index| {
                                let batch = batches[batch_index];
                                let mut values = Vec::with_capacity(
                                    batch.iter().map(|input| input.data.len()).sum(),
                                );
                                for input in batch {
                                    values.extend_from_slice(&input.data);
                                }
                                let shape = vec![
                                    batch.len(),
                                    sample_shape[1],
                                    sample_shape[2],
                                    sample_shape[3],
                                ];
                                (
                                    batch_index * batch_size,
                                    execute_probe_tensor(detector, shape, values, cancellation),
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let elapsed = started.elapsed();
        actual.sort_by_key(|(index, _)| *index);
        for (start, output) in &actual {
            let batch_len = output.shape[0];
            let expected_values = expected[*start..*start + batch_len]
                .iter()
                .flat_map(|output| output.values.iter().copied())
                .collect::<Vec<_>>();
            assert_f32_bits_equal(&output.values, &expected_values);
        }
        eprintln!(
            "SEAL_TEXT_CONCURRENCY_PROBE concurrency={concurrency} batch_size={batch_size} pages={} total_ms={:.3} pps={:.3} exact=true",
            inputs.len(),
            elapsed.as_secs_f64() * 1_000.0,
            inputs.len() as f64 / elapsed.as_secs_f64(),
        );
    }

    let concurrency = 8_usize;
    let detector = SealTextProbe::load_with_concurrency(&assets, concurrency).unwrap();
    let expected_detections = inputs
        .iter()
        .zip(&expected)
        .map(|(input, output)| {
            detection_boxes_in_content(
                &output.values,
                &output.shape,
                input.geometry.content_width,
                input.geometry.content_height,
                input.geometry.original_width,
                input.geometry.original_height,
                &model_config(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let mut actual = std::thread::scope(|scope| {
        let handles = (0..concurrency)
            .map(|worker| {
                let detector = &detector;
                let pages = &pages;
                let cancellation = &cancellation;
                scope.spawn(move || {
                    (worker..pages.len())
                        .step_by(concurrency)
                        .map(|index| {
                            let image = image::open(&pages[index]).unwrap().into_rgb8();
                            (index, detector.detect(&image, cancellation).unwrap())
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let elapsed = started.elapsed();
    actual.sort_by_key(|(index, _)| *index);
    let mut preprocess = Duration::ZERO;
    let mut inference = Duration::ZERO;
    let mut postprocess = Duration::ZERO;
    let mut observations = 0_usize;
    for (expected_index, ((index, run), expected)) in
        actual.iter().zip(&expected_detections).enumerate()
    {
        assert_eq!(*index, expected_index);
        assert_detections_bits_equal(&run.detections, expected);
        preprocess += run.preprocess;
        inference += run.inference;
        postprocess += run.postprocess;
        observations += run.detections.len();
    }
    eprintln!(
        "SEAL_TEXT_CONCURRENT_PIPELINE concurrency={concurrency} pages={} observations={observations} wall_ms={:.3} pps={:.3} summed_preprocess_ms={:.3} summed_inference_ms={:.3} summed_postprocess_ms={:.3} exact=true",
        pages.len(),
        elapsed.as_secs_f64() * 1_000.0,
        pages.len() as f64 / elapsed.as_secs_f64(),
        preprocess.as_secs_f64() * 1_000.0,
        inference.as_secs_f64() * 1_000.0,
        postprocess.as_secs_f64() * 1_000.0,
    );
}

#[test]
#[ignore = "requires the exact seal-text bundle and a real raster fixture"]
fn concurrent_fixture_pipeline_is_exact_and_reports_throughput() {
    let assets = ProbeAssets::from_env().unwrap();
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name a read-only validation fixture");
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    assert!(!pages.is_empty());

    let cancellation = CancellationToken::new();
    let reference = SealTextProbe::load(&assets).unwrap();
    let expected = pages
        .iter()
        .map(|page| {
            let image = image::open(page).unwrap().into_rgb8();
            reference.detect(&image, &cancellation).unwrap().detections
        })
        .collect::<Vec<_>>();

    for concurrency in [1_usize, 2, 4, 8] {
        let detector = SealTextProbe::load_with_concurrency(&assets, concurrency).unwrap();
        let started = Instant::now();
        let mut actual = std::thread::scope(|scope| {
            let handles = (0..concurrency)
                .map(|worker| {
                    let detector = &detector;
                    let pages = &pages;
                    let cancellation = &cancellation;
                    scope.spawn(move || {
                        (worker..pages.len())
                            .step_by(concurrency)
                            .map(|index| {
                                let image = image::open(&pages[index]).unwrap().into_rgb8();
                                (index, detector.detect(&image, cancellation).unwrap())
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let elapsed = started.elapsed();
        actual.sort_by_key(|(index, _)| *index);
        let mut preprocess = Duration::ZERO;
        let mut inference = Duration::ZERO;
        let mut postprocess = Duration::ZERO;
        let mut observations = 0_usize;
        for (expected_index, ((index, run), expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(*index, expected_index);
            assert_detections_bits_equal(&run.detections, expected);
            preprocess += run.preprocess;
            inference += run.inference;
            postprocess += run.postprocess;
            observations += run.detections.len();
        }
        eprintln!(
            "SEAL_TEXT_CONCURRENT_PIPELINE concurrency={concurrency} pages={} observations={observations} wall_ms={:.3} pps={:.3} summed_preprocess_ms={:.3} summed_inference_ms={:.3} summed_postprocess_ms={:.3} exact=true",
            pages.len(),
            elapsed.as_secs_f64() * 1_000.0,
            pages.len() as f64 / elapsed.as_secs_f64(),
            preprocess.as_secs_f64() * 1_000.0,
            inference.as_secs_f64() * 1_000.0,
            postprocess.as_secs_f64() * 1_000.0,
        );
    }
}

#[test]
#[ignore = "requires exact assets plus an explicit diagnostic image and artifact root"]
fn parity_fixture_exports_exact_power_tensors() {
    let assets = ProbeAssets::from_env().unwrap();
    let image_path = std::env::var_os("A3S_OCR_SEAL_TEXT_PARITY_IMAGE")
        .expect("A3S_OCR_SEAL_TEXT_PARITY_IMAGE must name one validation raster");
    let artifact_root = std::env::var_os("A3S_OCR_SEAL_TEXT_PARITY_ARTIFACT_ROOT")
        .expect("A3S_OCR_SEAL_TEXT_PARITY_ARTIFACT_ROOT must name a run-scoped artifact root");
    let artifact_root = PathBuf::from(artifact_root);
    std::fs::create_dir_all(&artifact_root).unwrap();
    let image = image::open(image_path).unwrap().into_rgb8();
    let input =
        detection_input_with_resize_long(&image, &model_config(), RESIZE_LONG, RESIZE_STRIDE)
            .unwrap();
    write_f32_le(&artifact_root.join("input.f32le"), &input.data).unwrap();
    std::fs::write(
        artifact_root.join("input-shape.json"),
        serde_json::to_vec(&input.shape).unwrap(),
    )
    .unwrap();

    let detector = SealTextProbe::load(&assets).unwrap();
    let tensor =
        TensorInput::new(input.shape.to_vec(), input.data, detector.runtime.limits()).unwrap();
    let cancellation = CancellationToken::new();
    let permit = detector.runtime.begin(&cancellation).unwrap();
    let output = detector.graph.run(tensor, &permit, &cancellation).unwrap();
    write_f32_le(&artifact_root.join("power-output.f32le"), &output.values).unwrap();
    std::fs::write(
        artifact_root.join("power-output-shape.json"),
        serde_json::to_vec(&output.shape).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires exact assets plus one explicit validation raster"]
fn one_fixture_reports_orthogonal_rotation_observations() {
    let assets = ProbeAssets::from_env().unwrap();
    let image_path = std::env::var_os("A3S_OCR_SEAL_TEXT_PARITY_IMAGE")
        .expect("A3S_OCR_SEAL_TEXT_PARITY_IMAGE must name one validation raster");
    let image = image::open(image_path).unwrap().into_rgb8();
    let rotated = [
        (0_u16, image.clone()),
        (90, image::imageops::rotate90(&image)),
        (180, image::imageops::rotate180(&image)),
        (270, image::imageops::rotate270(&image)),
    ];
    let detector = SealTextProbe::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    for (degrees, image) in rotated {
        let run = detector.detect(&image, &cancellation).unwrap();
        eprintln!(
            "SEAL_TEXT_ROTATION_PROBE degrees={degrees} dimensions={}x{} observations={} detections={:?}",
            image.width(),
            image.height(),
            run.detections.len(),
            run.detections
        );
    }
}

#[test]
#[ignore = "requires exact assets plus one explicit validation raster"]
fn one_fixture_reports_coarse_to_fine_observations() {
    let assets = ProbeAssets::from_env().unwrap();
    let layout_assets = PicodetLayoutAssets::from_env().unwrap();
    let image_path = std::env::var_os("A3S_OCR_SEAL_TEXT_PARITY_IMAGE")
        .expect("A3S_OCR_SEAL_TEXT_PARITY_IMAGE must name one validation raster");
    let image = image::open(image_path).unwrap().into_rgb8();
    let detector = SealTextProbe::load(&assets).unwrap();
    let cancellation = CancellationToken::new();
    let layout_detector = NativePicodetLayout::load(&layout_assets).unwrap();
    let layout_view = layout_full_page_view(&image);
    let layout_permit = layout_detector.begin(&cancellation).unwrap();
    let layout_output = layout_detector
        .infer_batch(
            layout_view_tensor(&image, layout_view, layout_assets.profile).unwrap(),
            1,
            &layout_permit,
            &cancellation,
        )
        .unwrap();
    eprintln!(
        "SEAL_TEXT_REFINEMENT_LAYOUT_IMAGES observations={:?}",
        layout_image_observations(&layout_output.tensor.values, layout_assets.profile, &image,)
    );
    let coarse = detector.detect(&image, &cancellation).unwrap();
    for (index, candidate) in coarse.detections.iter().enumerate() {
        let exact = detection_bounds(candidate, image.width(), image.height());
        for (view_kind, view) in [
            ("exact", exact),
            ("context", candidate_focus_view(candidate, &image)),
        ] {
            let crop = image::imageops::crop_imm(&image, view.x, view.y, view.width, view.height)
                .to_image();
            let direct = detector.detect(&crop, &cancellation).unwrap();
            let rotated = image::imageops::rotate90(&crop);
            let orthogonal = detector.detect(&rotated, &cancellation).unwrap();
            let direct_source = direct
                .detections
                .iter()
                .map(|detection| detection_source_bounds(detection, view))
                .collect::<Vec<_>>();
            let orthogonal_source = orthogonal
                .detections
                .iter()
                .map(|detection| rotated_detection_source_bounds(detection, view))
                .collect::<Vec<_>>();
            eprintln!(
                "SEAL_TEXT_REFINEMENT_PROBE candidate={index} view_kind={view_kind} coarse={exact:?} view={view:?} direct={direct_source:?} orthogonal={orthogonal_source:?}",
            );
        }
    }
}

fn candidate_focus_view(detection: &Detection, image: &RgbImage) -> PixelRect {
    let bounds = detection_bounds(detection, image.width(), image.height());
    // One candidate extent on each axis plus the same total amount of context
    // keeps the model focused while avoiding a crop-tight semantic decision.
    let requested_side = bounds.width.max(bounds.height).saturating_mul(2);
    let width = requested_side.min(image.width()).max(1);
    let height = requested_side.min(image.height()).max(1);
    let center_x = bounds.x.saturating_add(bounds.width / 2);
    let center_y = bounds.y.saturating_add(bounds.height / 2);
    let x = center_x
        .saturating_sub(width / 2)
        .min(image.width().saturating_sub(width));
    let y = center_y
        .saturating_sub(height / 2)
        .min(image.height().saturating_sub(height));
    PixelRect {
        x,
        y,
        width,
        height,
    }
}

fn detection_source_bounds(detection: &Detection, view: PixelRect) -> PixelRect {
    let local = detection_bounds(detection, view.width, view.height);
    PixelRect {
        x: view.x.saturating_add(local.x),
        y: view.y.saturating_add(local.y),
        width: local.width,
        height: local.height,
    }
}

fn rotated_detection_source_bounds(detection: &Detection, view: PixelRect) -> PixelRect {
    let restored = detection.polygon.map(|point| {
        imageproc::point::Point::new(
            point.y.clamp(0.0, view.width.saturating_sub(1) as f32),
            (view.height.saturating_sub(1) as f32 - point.x)
                .clamp(0.0, view.height.saturating_sub(1) as f32),
        )
    });
    detection_source_bounds(
        &Detection {
            polygon: restored,
            confidence: detection.confidence,
        },
        view,
    )
}

fn detection_bounds(detection: &Detection, width: u32, height: u32) -> PixelRect {
    let left = detection
        .polygon
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let top = detection
        .polygon
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let right = detection
        .polygon
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp((left + 1) as f32, width as f32) as u32;
    let bottom = detection
        .polygon
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp((top + 1) as f32, height as f32) as u32;
    PixelRect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    }
}

fn intersection_over_union(left: PixelRect, right: PixelRect) -> f32 {
    let intersection_width = left
        .x
        .saturating_add(left.width)
        .min(right.x.saturating_add(right.width))
        .saturating_sub(left.x.max(right.x));
    let intersection_height = left
        .y
        .saturating_add(left.height)
        .min(right.y.saturating_add(right.height))
        .saturating_sub(left.y.max(right.y));
    let intersection = u64::from(intersection_width).saturating_mul(u64::from(intersection_height));
    let left_area = u64::from(left.width).saturating_mul(u64::from(left.height));
    let right_area = u64::from(right.width).saturating_mul(u64::from(right.height));
    let union = left_area
        .saturating_add(right_area)
        .saturating_sub(intersection);
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn intersection_over_smaller(left: PixelRect, right: PixelRect) -> f32 {
    let intersection_width = left
        .x
        .saturating_add(left.width)
        .min(right.x.saturating_add(right.width))
        .saturating_sub(left.x.max(right.x));
    let intersection_height = left
        .y
        .saturating_add(left.height)
        .min(right.y.saturating_add(right.height))
        .saturating_sub(left.y.max(right.y));
    let intersection = u64::from(intersection_width).saturating_mul(u64::from(intersection_height));
    let smaller = u64::from(left.width)
        .saturating_mul(u64::from(left.height))
        .min(u64::from(right.width).saturating_mul(u64::from(right.height)));
    if smaller == 0 {
        0.0
    } else {
        intersection as f32 / smaller as f32
    }
}

fn layout_image_observations(
    values: &[f32],
    profile: PicodetLayoutProfile,
    image: &RgbImage,
) -> Vec<(PixelRect, f32)> {
    let mut observations = values
        .chunks_exact(profile.output_width())
        .filter_map(|row| {
            let (winning_class, confidence) = row[4..]
                .iter()
                .copied()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))?;
            if winning_class != 0 || confidence < SCORE_THRESHOLD {
                return None;
            }
            project_layout_bounds([row[0], row[1], row[2], row[3]], profile, image)
                .map(|bounds| (bounds, confidence))
        })
        .collect::<Vec<_>>();
    observations.sort_by(|left, right| right.1.total_cmp(&left.1));
    let mut retained = Vec::new();
    for observation in observations {
        if retained
            .iter()
            .any(|(known, _)| intersection_over_union(*known, observation.0) >= NMS_IOU_THRESHOLD)
        {
            continue;
        }
        retained.push(observation);
        if retained.len() == KEEP_TOP_K {
            break;
        }
    }
    retained
}

fn project_layout_bounds(
    coordinates: [f32; 4],
    profile: PicodetLayoutProfile,
    image: &RgbImage,
) -> Option<PixelRect> {
    let side = profile.input_side() as f32;
    if coordinates.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let x1 = coordinates[0].clamp(0.0, side);
    let y1 = coordinates[1].clamp(0.0, side);
    let x2 = coordinates[2].clamp(0.0, side);
    let y2 = coordinates[3].clamp(0.0, side);
    if x2 <= x1 || y2 <= y1 {
        return None;
    }
    let left = (x1 * image.width() as f32 / side).floor() as u32;
    let top = (y1 * image.height() as f32 / side).floor() as u32;
    let right = (x2 * image.width() as f32 / side).ceil() as u32;
    let bottom = (y2 * image.height() as f32 / side).ceil() as u32;
    (right > left && bottom > top).then_some(PixelRect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

fn fused_layout_image_observations(
    direct: &[Detection],
    orthogonal: &[(PixelRect, f32)],
    layout_images: &[(PixelRect, f32)],
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<(PixelRect, f32)> {
    let consensus = direct
        .iter()
        .filter_map(|detection| {
            let bounds = detection_bounds(detection, canvas_width, canvas_height);
            orthogonal
                .iter()
                .filter_map(|(other, confidence)| {
                    let overlap = intersection_over_union(bounds, *other);
                    (overlap >= NMS_IOU_THRESHOLD)
                        .then_some((bounds, detection.confidence.min(*confidence)))
                })
                .max_by(|left, right| left.1.total_cmp(&right.1))
        })
        .collect::<Vec<_>>();

    let mut supported = layout_images
        .iter()
        .filter_map(|(layout, layout_confidence)| {
            consensus
                .iter()
                .filter_map(|(text, text_confidence)| {
                    (rect_area(*layout) <= rect_area(*text)
                        && intersection_over_smaller(*text, *layout) >= NMS_IOU_THRESHOLD
                        && center_is_inside(*layout, *text))
                    .then_some(layout_confidence.min(*text_confidence))
                })
                .max_by(f32::total_cmp)
                .map(|confidence| (*layout, confidence))
        })
        .collect::<Vec<_>>();

    let grouped = supported
        .iter()
        .map(|(candidate, _)| {
            supported
                .iter()
                .filter(|(child, _)| {
                    rect_area(*child) < rect_area(*candidate)
                        && intersection_over_smaller(*child, *candidate) >= NMS_IOU_THRESHOLD
                        && smaller_center_is_inside_larger(*child, *candidate)
                })
                .map(|(child, _)| *child)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    supported = supported
        .into_iter()
        .zip(grouped)
        .filter_map(|(candidate, children)| {
            let has_independent_pair = children.iter().enumerate().any(|(index, left)| {
                children[index + 1..]
                    .iter()
                    .any(|right| intersection_over_union(*left, *right) < NMS_IOU_THRESHOLD)
            });
            (!has_independent_pair).then_some(candidate)
        })
        .collect();
    supported.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.x.cmp(&right.0.x))
            .then_with(|| left.0.y.cmp(&right.0.y))
    });
    supported
}

fn smaller_center_is_inside_larger(left: PixelRect, right: PixelRect) -> bool {
    if rect_area(left) <= rect_area(right) {
        center_is_inside(left, right)
    } else {
        center_is_inside(right, left)
    }
}

fn center_is_inside(inner: PixelRect, outer: PixelRect) -> bool {
    let center_x_twice = u64::from(inner.x)
        .saturating_mul(2)
        .saturating_add(u64::from(inner.width));
    let center_y_twice = u64::from(inner.y)
        .saturating_mul(2)
        .saturating_add(u64::from(inner.height));
    center_x_twice >= u64::from(outer.x).saturating_mul(2)
        && center_x_twice <= u64::from(outer.x.saturating_add(outer.width)).saturating_mul(2)
        && center_y_twice >= u64::from(outer.y).saturating_mul(2)
        && center_y_twice <= u64::from(outer.y.saturating_add(outer.height)).saturating_mul(2)
}

fn rect_area(region: PixelRect) -> u64 {
    u64::from(region.width).saturating_mul(u64::from(region.height))
}

fn write_f32_le(path: &Path, values: &[f32]) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()
}

fn execute_probe_tensor(
    detector: &SealTextProbe,
    shape: Vec<usize>,
    values: Vec<f32>,
    cancellation: &CancellationToken,
) -> TensorOutput {
    let tensor = TensorInput::new(shape, values, detector.runtime.limits()).unwrap();
    let permit = detector.runtime.begin(cancellation).unwrap();
    detector.graph.run(tensor, &permit, cancellation).unwrap()
}

fn assert_f32_bits_equal(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    if let Some((index, (actual, expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
    {
        panic!(
            "batched graph output differs from scalar output at flat index {index}: actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_detections_bits_equal(actual: &[Detection], expected: &[Detection]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.confidence.to_bits(),
            expected.confidence.to_bits(),
            "detection {index} confidence differs"
        );
        for (point_index, (actual, expected)) in
            actual.polygon.iter().zip(&expected.polygon).enumerate()
        {
            assert_eq!(
                (actual.x.to_bits(), actual.y.to_bits()),
                (expected.x.to_bits(), expected.y.to_bits()),
                "detection {index} point {point_index} differs"
            );
        }
    }
}
