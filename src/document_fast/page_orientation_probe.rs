//! Test-only conformance probe for the official document-orientation model.
//!
//! Production admission remains disabled until the model has an independently
//! calibrated abstention contract. This probe validates model identity,
//! preprocessing equivalence, numerical execution, and bounded throughput.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, InferenceLimits, TensorInput, TensorOutput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use super::page_orientation_preprocess as preprocess;

const MODEL_ENV: &str = "A3S_OCR_PAGE_ORIENTATION_MODEL_DIR";
const CORPUS_ENV: &str = "A3S_OCR_PAGE_ORIENTATION_CORPUS_ROOT";
const REFERENCE_ENV: &str = "A3S_OCR_PAGE_ORIENTATION_REFERENCE_ROOT";
const FAMILY: &str = "pp-lcnet-x1-doc-orientation";
const ROLE: &str = "orientation-classification";
const SOURCE_SHA256: &str = "96e898f047a0e460ba0652e9afb8c874e53872821cfd7a3fec53a5ab62df92f0";
const GRAPH_SHA256: &str = "58af7aa1ccdba05938e409f2b4ce299465740bd615ed76154167527b66389110";
const WEIGHTS_SHA256: &str = "50c0b9a20725346542c87a7b95197b725e7aa379876849d547e57d85006dd3a6";
const WEIGHTS_COLLECTION_SHA256: &str =
    "15fd6132e3afdae4a60457e12847944da3867a3a6c54a5ca89d11a4b2c30017b";
const WEIGHTS_BYTES: u64 = 6_764_348;
const REFERENCE_INPUT_SHA256: &str =
    "75357dff1d6f1229f90bc8be3df2e91d1507ea991984a2c71917604ce2bbecd6";
const REFERENCE_OUTPUT_SHA256: &str =
    "0b7f6d299ef47118fd6fb79f723ed8248c32d25f3383b07da921e43c620b9ccf";
const REFERENCE_PAGES_SHA256: &str =
    "2571439429be1479b8c406ba651965b408031daad12a68de9aabf740924abb41";
const ABSOLUTE_GRAPH_TOLERANCE: f32 = 2.0e-5;
const RELATIVE_GRAPH_TOLERANCE: f32 = 2.0e-4;
const REFERENCE_BATCH_SIZE: usize = 8;

struct ProbeAssets {
    root: PathBuf,
    graph: String,
}

impl ProbeAssets {
    fn from_env() -> UseResult<Self> {
        let root = required_directory(MODEL_ENV)?;
        let graph_path = checked_file(&root, "graph.json")?;
        let weights_path = checked_file(&root, "model.safetensors")?;
        if file_sha256(&graph_path)? != GRAPH_SHA256 {
            return Err(probe_error("Document-orientation graph identity changed."));
        }
        let metadata = std::fs::metadata(&weights_path).map_err(|error| {
            probe_error(format!("Failed to inspect orientation weights: {error}"))
        })?;
        if metadata.len() != WEIGHTS_BYTES || file_sha256(&weights_path)? != WEIGHTS_SHA256 {
            return Err(probe_error(
                "Document-orientation weights identity changed.",
            ));
        }
        let graph = std::fs::read_to_string(graph_path)
            .map_err(|error| probe_error(format!("Failed to read orientation graph: {error}")))?;
        Ok(Self { root, graph })
    }
}

struct PageOrientationProbe {
    runtime: EmbeddedRuntime,
    graph: GraphExecutor,
}

impl PageOrientationProbe {
    fn load(assets: &ProbeAssets) -> UseResult<Self> {
        let limits = InferenceLimits::default();
        let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, limits.clone())
            .map_err(|error| power_error("initialize", error))?;
        let weights = Arc::new(
            WeightStore::open(&assets.root, &limits)
                .map_err(|error| power_error("open weights", error))?,
        );
        weights
            .verify_integrity(FAMILY, WEIGHTS_COLLECTION_SHA256)
            .map_err(|error| power_error("verify weights", error))?;
        let identity = GraphIdentity::new(FAMILY, ROLE, "onnx", SOURCE_SHA256, 17);
        let plan = GraphPlan::parse(&assets.graph, &identity, &weights, &limits)
            .map_err(|error| power_error("validate graph", error))?;
        let graph = GraphExecutor::new(plan, weights, runtime.clone())
            .map_err(|error| power_error("materialize graph", error))?;
        Ok(Self { runtime, graph })
    }

    fn run(&self, images: &[&RgbImage]) -> UseResult<(TensorOutput, Duration, Duration)> {
        let preprocess_started = Instant::now();
        let values = preprocess::batch(images, images.len())?;
        let preprocess_elapsed = preprocess_started.elapsed();
        let tensor = TensorInput::new(
            vec![
                images.len(),
                3,
                preprocess::INPUT_SIDE,
                preprocess::INPUT_SIDE,
            ],
            values,
            self.runtime.limits(),
        )
        .map_err(|error| power_error("validate input", error))?;
        let cancellation = CancellationToken::new();
        let permit = self
            .runtime
            .begin(&cancellation)
            .map_err(|error| power_error("admit", error))?;
        let inference_started = Instant::now();
        let output = self
            .graph
            .run(tensor, &permit, &cancellation)
            .map_err(|error| power_error("execute", error))?;
        Ok((output, preprocess_elapsed, inference_started.elapsed()))
    }
}

#[test]
#[ignore = "requires the exact orientation bundle and an explicit blind raster corpus"]
fn blind_corpus_matches_official_preprocessing_and_reference_classes() {
    let assets = ProbeAssets::from_env().unwrap();
    let corpus = required_directory(CORPUS_ENV).unwrap();
    let reference = required_directory(REFERENCE_ENV).unwrap();
    let pages_path =
        checked_reference_file(&reference, "pages.json", REFERENCE_PAGES_SHA256).unwrap();
    let input_path =
        checked_reference_file(&reference, "input.f32le", REFERENCE_INPUT_SHA256).unwrap();
    let output_path =
        checked_reference_file(&reference, "expected.f32le", REFERENCE_OUTPUT_SHA256).unwrap();
    let page_names: Vec<String> =
        serde_json::from_slice(&std::fs::read(pages_path).unwrap()).unwrap();
    let images = page_names
        .iter()
        .map(|name| load_corpus_image(&corpus, name))
        .collect::<UseResult<Vec<_>>>()
        .unwrap();
    let image_refs = images.iter().collect::<Vec<_>>();
    let reference_input = read_f32_le(&input_path).unwrap();
    let reference_output = read_f32_le(&output_path).unwrap();
    let slot_elements = 3 * preprocess::INPUT_SIDE * preprocess::INPUT_SIDE;
    assert_eq!(reference_input.len(), image_refs.len() * slot_elements);
    assert_eq!(reference_output.len(), image_refs.len() * 4);

    let preprocess_started = Instant::now();
    let actual_input = image_refs
        .chunks(REFERENCE_BATCH_SIZE)
        .map(|batch| preprocess::batch(batch, REFERENCE_BATCH_SIZE))
        .collect::<UseResult<Vec<_>>>()
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let preprocess_elapsed = preprocess_started.elapsed();
    let mut input_maximum = 0.0_f32;
    let mut input_sum = 0.0_f64;
    for (index, (&actual, &expected)) in actual_input.iter().zip(&reference_input).enumerate() {
        let difference = (actual - expected).abs();
        input_maximum = input_maximum.max(difference);
        input_sum += f64::from(difference);
        assert!(
            difference <= 2.0e-6,
            "orientation preprocessing differs from the reviewed OpenCV contract at flat index {index}: actual={actual:?}, expected={expected:?}"
        );
    }

    let probe = PageOrientationProbe::load(&assets).unwrap();
    let first = &image_refs[..REFERENCE_BATCH_SIZE.min(image_refs.len())];
    probe.run(first).unwrap();
    let mut actual_output = Vec::with_capacity(reference_output.len());
    let mut measured_preprocess = Duration::ZERO;
    let mut measured_inference = Duration::ZERO;
    for batch in image_refs.chunks(REFERENCE_BATCH_SIZE) {
        let (output, preprocessing, inference) = probe.run(batch).unwrap();
        assert_eq!(output.shape, [batch.len(), 4]);
        actual_output.extend(output.values);
        measured_preprocess += preprocessing;
        measured_inference += inference;
    }
    let mut output_maximum = 0.0_f32;
    let mut output_sum = 0.0_f64;
    for (page, (actual, expected)) in actual_output
        .chunks_exact(4)
        .zip(reference_output.chunks_exact(4))
        .enumerate()
    {
        assert_eq!(
            top1(actual),
            top1(expected),
            "orientation class changed on page {page}"
        );
        for (&actual, &expected) in actual.iter().zip(expected) {
            let difference = (actual - expected).abs();
            output_maximum = output_maximum.max(difference);
            output_sum += f64::from(difference);
            let graph_tolerance =
                ABSOLUTE_GRAPH_TOLERANCE + RELATIVE_GRAPH_TOLERANCE * expected.abs();
            assert!(difference <= graph_tolerance);
        }
    }
    eprintln!(
        "A3S_OCR_ORIENTATION_CORPUS pages={} input_max_abs={input_maximum:.9e} input_mean_abs={:.9e} output_max_abs={output_maximum:.9e} output_mean_abs={:.9e} fused_preprocess_pps={:.3} staged_preprocess_pps={:.3} power_inference_pps={:.3}",
        image_refs.len(),
        input_sum / reference_input.len() as f64,
        output_sum / reference_output.len() as f64,
        image_refs.len() as f64 / preprocess_elapsed.as_secs_f64(),
        image_refs.len() as f64 / measured_preprocess.as_secs_f64(),
        image_refs.len() as f64 / measured_inference.as_secs_f64(),
    );
}

#[test]
#[ignore = "requires the exact orientation bundle and an explicit blind raster corpus"]
fn blind_corpus_reports_quarter_turn_group_consistency() {
    let assets = ProbeAssets::from_env().unwrap();
    let corpus = required_directory(CORPUS_ENV).unwrap();
    let reference = required_directory(REFERENCE_ENV).unwrap();
    let pages_path =
        checked_reference_file(&reference, "pages.json", REFERENCE_PAGES_SHA256).unwrap();
    let page_names: Vec<String> =
        serde_json::from_slice(&std::fs::read(pages_path).unwrap()).unwrap();
    let probe = PageOrientationProbe::load(&assets).unwrap();
    let mut consistent = 0_usize;
    let mut non_upright = 0_usize;
    let mut admitted_non_upright = 0_usize;

    for name in &page_names {
        let source = load_corpus_image(&corpus, name).unwrap();
        let rotations = [
            source.clone(),
            image::imageops::rotate90(&source),
            image::imageops::rotate180(&source),
            image::imageops::rotate270(&source),
        ];
        let image_refs = rotations.iter().collect::<Vec<_>>();
        let (output, _, _) = probe.run(&image_refs).unwrap();
        assert_eq!(output.shape, [4, 4]);
        let rows = output.values.chunks_exact(4).collect::<Vec<_>>();
        let classes = rows.iter().map(|row| top1(row)).collect::<Vec<_>>();
        let confidences = rows.iter().map(|row| row[top1(row)]).collect::<Vec<_>>();
        assert!(rows
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite()));
        let base = classes[0];
        let group_consistent = classes
            .iter()
            .enumerate()
            .all(|(turns, class)| *class == (base + turns) % 4);
        consistent += usize::from(group_consistent);
        non_upright += usize::from(base != 0);
        admitted_non_upright += usize::from(base != 0 && group_consistent);
        eprintln!(
            "A3S_OCR_ORIENTATION_GROUP page={name:?} classes={classes:?} confidences={confidences:?} consistent={group_consistent} correction={}",
            (4 - base) % 4,
        );
    }
    eprintln!(
        "A3S_OCR_ORIENTATION_GROUP_SUMMARY pages={} consistent={consistent} non_upright={non_upright} admitted_non_upright={admitted_non_upright} abstained_non_upright={}",
        page_names.len(),
        non_upright - admitted_non_upright,
    );
}

fn top1(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}

fn load_corpus_image(root: &Path, relative: &str) -> UseResult<RgbImage> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(probe_error(
            "Orientation corpus path is not a bounded relative path.",
        ));
    }
    let path = std::fs::canonicalize(root.join(relative)).map_err(|error| {
        probe_error(format!(
            "Failed to resolve orientation corpus page: {error}"
        ))
    })?;
    if !path.starts_with(root) {
        return Err(probe_error(
            "Orientation corpus page escaped its admitted root.",
        ));
    }
    image::open(path)
        .map(image::DynamicImage::into_rgb8)
        .map_err(|error| probe_error(format!("Failed to decode orientation corpus page: {error}")))
}

fn required_directory(name: &str) -> UseResult<PathBuf> {
    let value = std::env::var_os(name)
        .ok_or_else(|| probe_error(format!("Set {name} to an explicit validation directory.")))?;
    let root = std::fs::canonicalize(value)
        .map_err(|error| probe_error(format!("Failed to resolve {name}: {error}")))?;
    if !root.is_dir() {
        return Err(probe_error(format!("{name} must name a directory.")));
    }
    Ok(root)
}

fn checked_file(root: &Path, relative: &str) -> UseResult<PathBuf> {
    let path = std::fs::canonicalize(root.join(relative))
        .map_err(|error| probe_error(format!("Failed to resolve {relative}: {error}")))?;
    if !path.starts_with(root) || !path.is_file() {
        return Err(probe_error(format!(
            "{relative} escaped its admitted root."
        )));
    }
    Ok(path)
}

fn checked_reference_file(root: &Path, relative: &str, expected: &str) -> UseResult<PathBuf> {
    let path = checked_file(root, relative)?;
    if file_sha256(&path)? != expected {
        return Err(probe_error(format!(
            "Orientation reference {relative} changed."
        )));
    }
    Ok(path)
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

fn read_f32_le(path: &Path) -> UseResult<Vec<f32>> {
    let bytes = std::fs::read(path)
        .map_err(|error| probe_error(format!("Failed to read '{}': {error}", path.display())))?;
    if bytes.len() % 4 != 0 {
        return Err(probe_error(
            "Orientation reference tensor is not f32-aligned.",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("exact chunk")))
        .collect())
}

fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    probe_error(format!("Failed to {action} through a3s-power: {error}"))
}

fn probe_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.orientation_probe_failed", message)
}
