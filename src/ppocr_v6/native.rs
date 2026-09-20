//! OCR-owned PP-OCRv6 graph identity over the model-neutral Power runtime.

use std::path::Path;
use std::sync::Arc;

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
#[cfg(test)]
use a3s_power::inference::DevicePreference;
use a3s_power::inference::{
    EmbeddedRuntime, ExecutionBatchBinding, ExecutionDigest, ExecutionPermit, ExecutionReceipt,
    InferenceLimits, ModelIdentity, ModelSessionBinding, ModelSessionSpec, RuntimeDeviceKind,
    TensorInput, TensorOutput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::assets::{ModelAssets, ModelGraphAsset};
use crate::config::{ModelProfile, ModelVariant};
use crate::preprocess::RecognitionInput;

mod projection;
#[cfg(test)]
mod topology_tests;
mod window;

const SMALL_FAMILY: &str = "pp-ocr-v6-small";
const TINY_FAMILY: &str = "pp-ocr-v6-tiny";
const REVISION: &str = "paddlex-paddle3.0.0";
const SMALL_DETECTION_GRAPH: &str = include_str!("graphs/detection.json");
const SMALL_RECOGNITION_GRAPH: &str = include_str!("graphs/recognition.json");
#[cfg(test)]
const RECOGNITION_GRAPH: &str = SMALL_RECOGNITION_GRAPH;
const SMALL_DETECTION_SOURCE_SHA256: &str =
    "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e";
const SMALL_RECOGNITION_SOURCE_SHA256: &str =
    "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634";
pub(crate) const SMALL_DETECTION_WEIGHTS_SHA256: &str =
    "0439824a102e0b365ca905355553985a885773ca0ea9f6a526e5f7317fc15592";
pub(crate) const SMALL_RECOGNITION_WEIGHTS_SHA256: &str =
    "e8bf34a6900addc8cd9ec1d1ea73ea56e97cb0d668c8c45508a885924078761f";
pub(crate) const DETECTION_WEIGHTS_SHA256: &str = SMALL_DETECTION_WEIGHTS_SHA256;
pub(crate) const RECOGNITION_WEIGHTS_SHA256: &str = SMALL_RECOGNITION_WEIGHTS_SHA256;
const TINY_DETECTION_SOURCE_SHA256: &str =
    "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8";
const TINY_RECOGNITION_SOURCE_SHA256: &str =
    "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6";
pub(crate) const TINY_DETECTION_WEIGHTS_SHA256: &str =
    "565ed7331e7ceceb921ec15b0ffae5dcb6b392474bdc76a841750fcf75ef9550";
pub(crate) const TINY_RECOGNITION_WEIGHTS_SHA256: &str =
    "738a0bcb5ce1a48ef795c70956497b821ee7dfb231c59a1aefdc52ccdf73c947";
const TINY_DETECTION_GRAPH_SHA256: &str =
    "0348134a461266ecfc51293afaebc36e08efafdde15296d58e7f9fa8fb85b432";
const TINY_RECOGNITION_GRAPH_SHA256: &str =
    "b8c45305d5902983e84ae4edf869768444a23a2b8b3ba7a155913be5ca83dcce";

pub(crate) struct NativeGraphOutput {
    pub(crate) tensor: TensorOutput,
    pub(crate) receipt: ExecutionReceipt,
}

/// The model architecture, reviewed graph plans, and revision pins live here
/// in a3s-ocr. Power supplies the shared execution and security substrate.
pub(crate) struct NativePpOcrV6 {
    runtime: EmbeddedRuntime,
    detection: GraphExecutor,
    recognition: GraphExecutor,
    detection_identity: ModelIdentity,
    recognition_identity: ModelIdentity,
}

impl NativePpOcrV6 {
    #[cfg(test)]
    pub(crate) fn load(assets: &ModelAssets) -> UseResult<Self> {
        let limits = session_limits();
        let runtime = EmbeddedRuntime::new(DevicePreference::Auto, limits.clone())
            .map_err(|error| power_error("initialize the embedded runtime", error))?;
        Self::load_with_runtime(assets, runtime)
    }

    pub(crate) fn load_with_runtime(
        assets: &ModelAssets,
        runtime: EmbeddedRuntime,
    ) -> UseResult<Self> {
        let limits = runtime.limits().clone();
        let detection_spec = GraphSpec::detection(assets.profile.detection);
        let recognition_spec = GraphSpec::recognition(assets.profile.recognition);
        let detection_graph = reviewed_graph_source(assets, GraphRole::Detection)?;
        let recognition_graph = reviewed_graph_source(assets, GraphRole::Recognition)?;
        let detection = load_graph(
            &runtime,
            &limits,
            &assets.detection_weights,
            &detection_graph,
            detection_spec,
        )?;
        let recognition = load_graph(
            &runtime,
            &limits,
            &assets.recognition_weights,
            &recognition_graph,
            recognition_spec,
        )?;
        Ok(Self {
            runtime,
            detection,
            recognition,
            detection_identity: detection_spec.model_identity(),
            recognition_identity: recognition_spec.model_identity(),
        })
    }

    pub(crate) fn begin(&self, cancellation: &CancellationToken) -> UseResult<ExecutionPermit> {
        self.runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit the OCR request", error))
    }

    #[cfg(test)]
    pub(crate) fn detect(
        &self,
        data: Vec<f32>,
        shape: [usize; 4],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        self.detect_batch(data, shape, permit, cancellation)
    }

    pub(crate) fn detect_batch(
        &self,
        data: Vec<f32>,
        shape: [usize; 4],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        if shape[0] == 0 {
            return Err(UseError::new(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 detection requires at least one input tensor.",
            ));
        }
        if shape[1] != 3 {
            return Err(UseError::new(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 detection input must use NCHW tensors with three channels.",
            ));
        }
        let slot_count = shape[0];
        let input = TensorInput::new(shape.to_vec(), data, self.runtime.limits())
            .map_err(|error| power_error("validate an OCR detection tensor", error))?;
        let output = self.execute_input(
            &self.detection,
            &self.detection_identity,
            input,
            permit,
            cancellation,
        )?;
        if output.tensor.shape.len() != 4
            || output.tensor.shape[0] != slot_count
            || output.tensor.shape[1] != 1
        {
            return Err(UseError::new(
                "use.ocr.provider_output_invalid",
                format!(
                    "PP-OCRv6 detection output shape must be [N, 1, H, W] for N={slot_count}, found {:?}.",
                    output.tensor.shape
                ),
            ));
        }
        Ok(output)
    }

    #[cfg(test)]
    pub(crate) fn recognize(
        &self,
        data: Vec<f32>,
        shape: [usize; 4],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        if shape[1] != 3 || shape[2] != 48 {
            return Err(UseError::new(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 recognition input must be NCHW with three channels and height 48.",
            ));
        }
        let digest_started = std::time::Instant::now();
        let input_digest = ExecutionDigest::f32_tensor(&shape, &data);
        let digest_elapsed = digest_started.elapsed();
        let input = TensorInput::new(shape.to_vec(), data, self.runtime.limits())
            .map_err(|error| power_error("validate the OCR input tensor", error))?;
        self.execute_recognition_input(input, input_digest, digest_elapsed, permit, cancellation)
    }

    pub(crate) fn recognize_prepared(
        &self,
        input: RecognitionInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        if input.shape[1] != 3 || input.shape[2] != 48 {
            return Err(UseError::new(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 recognition input must be NCHW with three channels and height 48.",
            ));
        }
        let tensor = TensorInput::new(input.shape.to_vec(), input.data, self.runtime.limits())
            .map_err(|error| power_error("validate the prepared OCR input tensor", error))?;
        self.execute_recognition_input(
            tensor,
            input.digest,
            std::time::Duration::ZERO,
            permit,
            cancellation,
        )
    }

    #[cfg(test)]
    pub(crate) fn recognize_prepared_features(
        &self,
        input: RecognitionInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<TensorOutput> {
        if input.shape[1] != 3 || input.shape[2] != 48 {
            return Err(UseError::new(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 recognition input must be NCHW with three channels and height 48.",
            ));
        }
        let tensor = TensorInput::new(input.shape.to_vec(), input.data, self.runtime.limits())
            .map_err(|error| power_error("validate the prepared OCR feature tensor", error))?;
        self.recognition
            .run_with_terminal_matmul_bias_softmax_projection(
                tensor,
                permit,
                cancellation,
                |features, _weights, _bias| Ok(features.clone()),
            )
            .map_err(|error| power_error("execute the OCR recognition feature prefix", error))
    }

    pub(crate) fn runtime_device_kind(&self) -> RuntimeDeviceKind {
        self.runtime.device().kind()
    }

    pub(crate) fn maximum_tensor_elements(&self) -> usize {
        self.runtime.limits().max_tensor_elements
    }

    pub(crate) fn maximum_input_bytes(&self) -> usize {
        self.runtime.limits().max_input_bytes
    }

    fn execute_input(
        &self,
        graph: &GraphExecutor,
        identity: &ModelIdentity,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let tensor = graph
            .run(input, permit, cancellation)
            .map_err(|error| power_error("execute the reviewed OCR graph", error))?;
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let receipt = self
            .runtime
            .receipt(identity.clone(), input_digest, output_digest);
        Ok(NativeGraphOutput { tensor, receipt })
    }

    fn execute_recognition_input(
        &self,
        input: TensorInput,
        input_digest: ExecutionDigest,
        input_digest_elapsed: std::time::Duration,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        let trace = std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some();
        let started = std::time::Instant::now();
        let tensor = self
            .recognition
            .run_with_terminal_matmul_bias_softmax_projection(
                input,
                permit,
                cancellation,
                projection::ctc_top1_from_classifier,
            )
            .map_err(|error| power_error("execute the projected OCR recognition graph", error))?;
        let executed = started.elapsed();
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let output_digested = started.elapsed();
        let receipt = self.runtime.receipt(
            self.recognition_identity.clone(),
            input_digest,
            output_digest,
        );
        if trace {
            let completed = started.elapsed();
            eprintln!(
                "A3S_OCR_NATIVE_RECOGNITION_TIMING input_digest_ms={:.3} graph_ms={:.3} output_digest_ms={:.3} receipt_ms={:.3} total_ms={:.3}",
                input_digest_elapsed.as_secs_f64() * 1_000.0,
                executed.as_secs_f64() * 1_000.0,
                (output_digested - executed).as_secs_f64() * 1_000.0,
                (completed - output_digested).as_secs_f64() * 1_000.0,
                completed.as_secs_f64() * 1_000.0,
            );
        }
        Ok(NativeGraphOutput { tensor, receipt })
    }
}

pub(crate) fn session_limits() -> InferenceLimits {
    InferenceLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 32,
        ..InferenceLimits::default()
    }
}

pub(crate) fn session_spec(assets: &ModelAssets) -> UseResult<ModelSessionSpec> {
    let resident_bytes = file_size(&assets.detection_weights)?
        .checked_add(file_size(&assets.recognition_weights)?)
        .ok_or_else(|| model_error("PP-OCRv6 resident model bytes overflowed."))?;
    ModelSessionSpec::new(
        ModelSessionBinding::new(
            bundle_model_identity(assets.profile),
            session_execution_sha256(assets)?,
        ),
        session_limits(),
        resident_bytes,
    )
    .map_err(|error| power_error("declare the PP-OCRv6 model session", error))
}

pub(crate) fn batch_binding(weights_sha256: &str) -> UseResult<ExecutionBatchBinding> {
    ExecutionBatchBinding::new(
        weights_sha256,
        named_sha256(b"a3s-ocr-ppocr-v6-staged-slot-layout-v2\0"),
        named_sha256(b"a3s-ocr-ppocr-v6-shape-cohort-scheduler-v10\0"),
    )
    .map_err(|error| power_error("bind the PP-OCRv6 staged batch", error))
}

pub(crate) fn bundle_model_identity(profile: ModelProfile) -> ModelIdentity {
    ModelIdentity::new(
        format!("{}-bundle", profile.execution_family()),
        REVISION,
        bundle_weights_sha256(profile),
    )
}

fn bundle_weights_sha256(profile: ModelProfile) -> String {
    let detection = GraphSpec::detection(profile.detection);
    let recognition = GraphSpec::recognition(profile.recognition);
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-ppocr-v6-bundle-weights-v1\0");
    digest.update(detection.weights_sha256.as_bytes());
    digest.update(recognition.weights_sha256.as_bytes());
    format!("{:x}", digest.finalize())
}

fn session_execution_sha256(assets: &ModelAssets) -> UseResult<String> {
    let detection_graph = reviewed_graph_source(assets, GraphRole::Detection)?;
    let recognition_graph = reviewed_graph_source(assets, GraphRole::Recognition)?;
    let detection_config = std::fs::read(&assets.detection_config).map_err(|error| {
        model_error(format!(
            "Failed to read the PP-OCRv6 detection configuration: {error}"
        ))
    })?;
    let recognition_config = std::fs::read(&assets.recognition_config).map_err(|error| {
        model_error(format!(
            "Failed to read the PP-OCRv6 recognition configuration: {error}"
        ))
    })?;
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-ppocr-v6-session-execution-v3\0");
    update_bytes(&mut digest, detection_graph.as_bytes())?;
    update_bytes(&mut digest, recognition_graph.as_bytes())?;
    update_bytes(&mut digest, projection::IDENTITY)?;
    update_bytes(&mut digest, &detection_config)?;
    update_bytes(&mut digest, &recognition_config)?;
    Ok(format!("{:x}", digest.finalize()))
}

fn update_bytes(digest: &mut Sha256, bytes: &[u8]) -> UseResult<()> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| model_error("A PP-OCRv6 session input length cannot be represented."))?;
    digest.update(length.to_le_bytes());
    digest.update(bytes);
    Ok(())
}

fn file_size(path: &Path) -> UseResult<u64> {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| {
            model_error(format!(
                "Failed to inspect PP-OCRv6 model bytes '{}': {error}",
                path.display()
            ))
        })
}

fn named_sha256(domain: &[u8]) -> String {
    format!("{:x}", Sha256::digest(domain))
}

#[derive(Clone, Copy)]
struct GraphSpec {
    family: &'static str,
    role: &'static str,
    source_sha256: &'static str,
    source_opset: u32,
    weights_sha256: &'static str,
    projection_revision: Option<&'static str>,
}

impl GraphSpec {
    const fn detection(model_variant: ModelVariant) -> Self {
        Self {
            family: graph_family(model_variant),
            role: "detection",
            source_sha256: match model_variant {
                ModelVariant::Small => SMALL_DETECTION_SOURCE_SHA256,
                ModelVariant::Tiny => TINY_DETECTION_SOURCE_SHA256,
            },
            source_opset: 14,
            weights_sha256: match model_variant {
                ModelVariant::Small => SMALL_DETECTION_WEIGHTS_SHA256,
                ModelVariant::Tiny => TINY_DETECTION_WEIGHTS_SHA256,
            },
            projection_revision: None,
        }
    }

    const fn recognition(model_variant: ModelVariant) -> Self {
        Self {
            family: graph_family(model_variant),
            role: "recognition",
            source_sha256: match model_variant {
                ModelVariant::Small => SMALL_RECOGNITION_SOURCE_SHA256,
                ModelVariant::Tiny => TINY_RECOGNITION_SOURCE_SHA256,
            },
            source_opset: 11,
            weights_sha256: match model_variant {
                ModelVariant::Small => SMALL_RECOGNITION_WEIGHTS_SHA256,
                ModelVariant::Tiny => TINY_RECOGNITION_WEIGHTS_SHA256,
            },
            projection_revision: Some(projection::REVISION),
        }
    }

    fn graph_identity(self) -> GraphIdentity {
        GraphIdentity::new(
            self.family,
            self.role,
            "onnx",
            self.source_sha256,
            self.source_opset,
        )
    }

    fn model_identity(self) -> ModelIdentity {
        let revision = self.projection_revision.map_or_else(
            || REVISION.to_string(),
            |projection| format!("{REVISION}+{projection}"),
        );
        ModelIdentity::new(
            format!("{}-{}", self.family, self.role),
            revision,
            self.weights_sha256,
        )
    }
}

fn load_graph(
    runtime: &EmbeddedRuntime,
    limits: &InferenceLimits,
    weights_path: &Path,
    graph_source: &str,
    spec: GraphSpec,
) -> UseResult<GraphExecutor> {
    let root = weights_path.parent().ok_or_else(|| {
        UseError::new(
            "use.ocr.model_invalid",
            format!("PP-OCRv6 {} weights have no parent directory.", spec.role),
        )
    })?;
    let weights = Arc::new(
        WeightStore::open(root, limits)
            .map_err(|error| power_error("open the reviewed OCR weights", error))?,
    );
    weights
        .verify_integrity(
            &format!("{}-{}", spec.family, spec.role),
            spec.weights_sha256,
        )
        .map_err(|error| power_error("verify the reviewed OCR weights", error))?;
    let plan = GraphPlan::parse(graph_source, &spec.graph_identity(), &weights, limits)
        .map_err(|error| power_error("validate the reviewed OCR graph", error))?;
    GraphExecutor::new(plan, weights, runtime.clone())
        .map_err(|error| power_error("materialize the reviewed OCR graph", error))
}

#[derive(Clone, Copy)]
enum GraphRole {
    Detection,
    Recognition,
}

const fn graph_family(model_variant: ModelVariant) -> &'static str {
    match model_variant {
        ModelVariant::Small => SMALL_FAMILY,
        ModelVariant::Tiny => TINY_FAMILY,
    }
}

fn reviewed_graph_source(assets: &ModelAssets, role: GraphRole) -> UseResult<String> {
    let (variant, graph, embedded_graph, expected_sha256) = match role {
        GraphRole::Detection => (
            assets.profile.detection,
            &assets.graphs.detection,
            SMALL_DETECTION_GRAPH,
            TINY_DETECTION_GRAPH_SHA256,
        ),
        GraphRole::Recognition => (
            assets.profile.recognition,
            &assets.graphs.recognition,
            SMALL_RECOGNITION_GRAPH,
            TINY_RECOGNITION_GRAPH_SHA256,
        ),
    };
    match (variant, graph) {
        (ModelVariant::Small, ModelGraphAsset::Embedded) => Ok(embedded_graph.to_string()),
        (ModelVariant::Tiny, ModelGraphAsset::ReviewedFile(path)) => {
            let bytes = std::fs::read(path).map_err(|error| {
                model_error(format!(
                    "Failed to read reviewed PP-OCRv6 graph '{}': {error}",
                    path.display()
                ))
            })?;
            let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
            if actual_sha256 != expected_sha256 {
                return Err(model_error(format!(
                    "Reviewed PP-OCRv6 graph '{}' failed integrity verification.",
                    path.display()
                )));
            }
            String::from_utf8(bytes)
                .map_err(|_| model_error("Reviewed PP-OCRv6 graph is not UTF-8 JSON."))
        }
        _ => Err(model_error(
            "A PP-OCRv6 graph asset does not match its declared model role variant.",
        )),
    }
}

fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} through a3s-power: {error}"),
    )
}

fn model_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.model_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::OcrInstallSource;

    #[test]
    fn reviewed_graph_identity_is_ocr_owned() {
        let detection = GraphSpec::detection(ModelVariant::Small);
        let recognition = GraphSpec::recognition(ModelVariant::Small);
        assert_eq!(detection.graph_identity().role, "detection");
        assert_eq!(recognition.graph_identity().role, "recognition");
        assert_eq!(detection.source_opset, 14);
        assert_eq!(recognition.source_opset, 11);
    }

    #[test]
    fn graph_plans_keep_the_reviewed_node_inventory() {
        let detection: serde_json::Value = serde_json::from_str(SMALL_DETECTION_GRAPH).unwrap();
        let recognition: serde_json::Value = serde_json::from_str(RECOGNITION_GRAPH).unwrap();
        assert_eq!(detection["nodes"].as_array().unwrap().len(), 242);
        assert_eq!(recognition["nodes"].as_array().unwrap().len(), 481);
        assert_eq!(detection["inputs"][0]["shape"][0], "DynamicDimension.0");
        assert_eq!(
            detection["outputs"][0]["shape"][0],
            "ConvTranspose_459_o0__d0"
        );
    }

    #[test]
    fn reviewed_graphs_keep_the_fusible_gated_activation_inventory() {
        assert_eq!(
            adjacent_single_consumer_hard_sigmoid_mul(SMALL_DETECTION_GRAPH),
            13
        );
        assert_eq!(
            adjacent_single_consumer_hard_sigmoid_mul(RECOGNITION_GRAPH),
            5
        );
        assert_eq!(adjacent_single_consumer_gelu_erf(RECOGNITION_GRAPH), 13);
    }

    fn adjacent_single_consumer_hard_sigmoid_mul(graph: &str) -> usize {
        let graph: serde_json::Value = serde_json::from_str(graph).unwrap();
        let nodes = graph["nodes"].as_array().unwrap();
        nodes
            .windows(2)
            .filter(|pair| {
                if pair[0]["op"] != "HardSigmoid" || pair[1]["op"] != "Mul" {
                    return false;
                }
                let output = pair[0]["outputs"][0].as_str().unwrap();
                let multiply_inputs = pair[1]["inputs"].as_array().unwrap();
                let direct_uses = multiply_inputs
                    .iter()
                    .filter(|input| input.as_str() == Some(output))
                    .count();
                let graph_uses = nodes
                    .iter()
                    .flat_map(|node| node["inputs"].as_array().unwrap())
                    .filter(|input| input.as_str() == Some(output))
                    .count();
                direct_uses == 1 && graph_uses == 1
            })
            .count()
    }

    fn adjacent_single_consumer_gelu_erf(graph: &str) -> usize {
        let graph: serde_json::Value = serde_json::from_str(graph).unwrap();
        let nodes = graph["nodes"].as_array().unwrap();
        let graph_output = graph["outputs"][0]["name"].as_str().unwrap();
        nodes
            .windows(5)
            .filter(|window| {
                if window
                    .iter()
                    .map(|node| node["op"].as_str().unwrap())
                    .ne(["Div", "Erf", "Add", "Mul", "Mul"])
                {
                    return false;
                }
                let input = window[0]["inputs"][0].as_str().unwrap();
                let divide = window[0]["outputs"][0].as_str().unwrap();
                let erf = window[1]["outputs"][0].as_str().unwrap();
                let add = window[2]["outputs"][0].as_str().unwrap();
                let multiply = window[3]["outputs"][0].as_str().unwrap();
                if window[1]["inputs"][0].as_str() != Some(divide)
                    || !contains_once(&window[2]["inputs"], erf)
                    || !contains_once(&window[3]["inputs"], input)
                    || !contains_once(&window[3]["inputs"], add)
                    || !contains_once(&window[4]["inputs"], multiply)
                {
                    return false;
                }
                [divide, erf, add, multiply].into_iter().all(|output| {
                    output != graph_output
                        && nodes
                            .iter()
                            .flat_map(|node| node["inputs"].as_array().unwrap())
                            .filter(|input| input.as_str() == Some(output))
                            .count()
                            == 1
                })
            })
            .count()
    }

    fn contains_once(inputs: &serde_json::Value, expected: &str) -> bool {
        inputs
            .as_array()
            .unwrap()
            .iter()
            .filter(|input| input.as_str() == Some(expected))
            .count()
            == 1
    }

    #[test]
    fn pooled_session_and_batch_bindings_cover_exact_model_and_configuration() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let detection_weights = root.join("detection.safetensors");
        let recognition_weights = root.join("recognition.safetensors");
        let detection_config = root.join("detection.yml");
        let recognition_config = root.join("recognition.yml");
        std::fs::write(&detection_weights, b"detection").unwrap();
        std::fs::write(&recognition_weights, b"recognition").unwrap();
        std::fs::write(&detection_config, b"config-a").unwrap();
        std::fs::write(&recognition_config, b"config-b").unwrap();
        let assets = ModelAssets {
            root: root.to_path_buf(),
            detection_weights,
            detection_config: detection_config.clone(),
            recognition_weights,
            recognition_config,
            profile: ModelProfile::new(ModelVariant::Small, ModelVariant::Small),
            graphs: crate::assets::ModelGraphAssets::embedded(),
            source: OcrInstallSource::Environment,
        };
        let first = session_spec(&assets).unwrap();
        std::fs::write(detection_config, b"config-c").unwrap();
        let second = session_spec(&assets).unwrap();

        assert_ne!(
            first.binding().execution_sha256,
            second.binding().execution_sha256
        );
        assert_eq!(first.resident_bytes(), 20);
        assert_eq!(first.limits().max_concurrent_requests, 1);
        assert_eq!(first.limits().max_queued_requests, 32);
        assert_eq!(
            batch_binding(
                &bundle_model_identity(
                    ModelProfile::new(ModelVariant::Small, ModelVariant::Small,)
                )
                .weights_sha256,
            )
            .unwrap()
            .weights_sha256,
            bundle_model_identity(ModelProfile::new(ModelVariant::Small, ModelVariant::Small,))
                .weights_sha256
        );
    }

    #[test]
    #[ignore = "requires the pinned official PP-OCRv6 native bundle"]
    fn official_weights_execute_with_pinned_cpu_fixtures() {
        let assets = official_assets();
        let native = NativePpOcrV6::load(&assets).unwrap();
        let cancellation = CancellationToken::new();
        let permit = native.begin(&cancellation).unwrap();

        let detection = native
            .detect(
                vec![0.0; 3 * 64 * 64],
                [1, 3, 64, 64],
                &permit,
                &cancellation,
            )
            .unwrap();
        let repeated_detection = native
            .detect(
                vec![0.0; 3 * 64 * 64],
                [1, 3, 64, 64],
                &permit,
                &cancellation,
            )
            .unwrap();
        let batched_detection = native
            .detect_batch(
                vec![0.0; 2 * 3 * 64 * 64],
                [2, 3, 64, 64],
                &permit,
                &cancellation,
            )
            .unwrap();
        assert_eq!(detection.tensor.shape, [1, 1, 64, 64]);
        assert_eq!(detection.tensor, repeated_detection.tensor);
        assert_eq!(batched_detection.tensor.shape, [2, 1, 64, 64]);
        assert!(batched_detection
            .tensor
            .values
            .chunks_exact(detection.tensor.values.len())
            .all(|values| values == detection.tensor.values));
        assert_eq!(
            batched_detection.receipt.input.item_count,
            detection.receipt.input.item_count * 2
        );
        assert_eq!(
            batched_detection.receipt.output.item_count,
            detection.receipt.output.item_count * 2
        );
        assert_eq!(detection.receipt.output, repeated_detection.receipt.output);
        assert_eq!(detection.receipt.output.byte_length, 16_384);
        assert_eq!(detection.receipt.output.item_count, 4_096);
        assert_eq!(
            detection.receipt.model.weights_sha256,
            DETECTION_WEIGHTS_SHA256
        );

        assert_official_recognition_projection(&native, &permit, &cancellation);
    }

    #[test]
    #[ignore = "requires the pinned official PP-OCRv6 native bundle"]
    fn official_recognition_projection_executes_on_selected_device() {
        let native = NativePpOcrV6::load(&official_assets()).unwrap();
        let cancellation = CancellationToken::new();
        let permit = native.begin(&cancellation).unwrap();

        assert_official_recognition_projection(&native, &permit, &cancellation);
    }

    #[test]
    #[ignore = "requires the pinned official PP-OCRv6 native bundle and an explicit accelerator"]
    fn official_nonzero_recognition_batches_are_repeatable() {
        let native = NativePpOcrV6::load(&official_assets()).unwrap();
        let cancellation = CancellationToken::new();
        let permit = native.begin(&cancellation).unwrap();

        for batch in [1_usize, 8, 128] {
            let elements = batch * 3 * 48 * 320;
            let input = (0..elements)
                .map(|index| ((index % 251) as f32 - 125.0) / 127.0)
                .collect::<Vec<_>>();
            let first = native
                .recognize(input.clone(), [batch, 3, 48, 320], &permit, &cancellation)
                .unwrap();
            let repeated = native
                .recognize(input, [batch, 3, 48, 320], &permit, &cancellation)
                .unwrap();
            assert_eq!(
                first.tensor, repeated.tensor,
                "non-zero recognition changed for batch {batch}"
            );
        }
    }

    fn official_assets() -> ModelAssets {
        let root = std::env::var_os("A3S_PPOCR_V6_MODEL")
            .expect("A3S_PPOCR_V6_MODEL must name the pinned official model bundle");
        let root = std::path::PathBuf::from(root);
        ModelAssets {
            root: root.clone(),
            detection_weights: root.join("det/model.safetensors"),
            detection_config: root.join("det/inference.yml"),
            recognition_weights: root.join("rec/model.safetensors"),
            recognition_config: root.join("rec/inference.yml"),
            profile: ModelProfile::new(ModelVariant::Small, ModelVariant::Small),
            graphs: crate::assets::ModelGraphAssets::embedded(),
            source: OcrInstallSource::Environment,
        }
    }

    fn assert_official_recognition_projection(
        native: &NativePpOcrV6,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) {
        let recognition = native
            .recognize(
                vec![0.0; 3 * 48 * 320],
                [1, 3, 48, 320],
                permit,
                cancellation,
            )
            .unwrap();
        let repeated_recognition = native
            .recognize(
                vec![0.0; 3 * 48 * 320],
                [1, 3, 48, 320],
                permit,
                cancellation,
            )
            .unwrap();
        assert_eq!(recognition.tensor.shape, [1, 40, 3]);
        assert_eq!(recognition.tensor, repeated_recognition.tensor);
        assert_eq!(
            recognition.receipt.output,
            repeated_recognition.receipt.output
        );
        assert_eq!(recognition.receipt.output.byte_length, 480);
        assert_eq!(recognition.receipt.output.item_count, 120);
        assert_eq!(
            recognition.receipt.model.revision,
            "paddlex-paddle3.0.0+ctc-matmul-bias-softmax-top1-last-tie-finite-v6"
        );
        assert_eq!(
            recognition.receipt.model.weights_sha256,
            RECOGNITION_WEIGHTS_SHA256
        );
    }
}
