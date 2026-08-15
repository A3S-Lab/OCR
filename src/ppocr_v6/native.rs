//! OCR-owned PP-OCRv6 graph identity over the model-neutral Power runtime.

use std::path::Path;
use std::sync::Arc;

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
#[cfg(test)]
use a3s_power::inference::DevicePreference;
use a3s_power::inference::{
    EmbeddedRuntime, ExecutionBatchBinding, ExecutionDigest, ExecutionPermit, ExecutionReceipt,
    InferenceLimits, ModelIdentity, ModelSessionBinding, ModelSessionSpec, TensorInput,
    TensorOutput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::assets::ModelAssets;

mod projection;

const FAMILY: &str = "pp-ocr-v6-small";
const REVISION: &str = "paddlex-paddle3.0.0";
const DETECTION_GRAPH: &str = include_str!("graphs/detection.json");
const RECOGNITION_GRAPH: &str = include_str!("graphs/recognition.json");
const DETECTION_SOURCE_SHA256: &str =
    "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e";
const RECOGNITION_SOURCE_SHA256: &str =
    "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634";
pub(crate) const DETECTION_WEIGHTS_SHA256: &str =
    "0439824a102e0b365ca905355553985a885773ca0ea9f6a526e5f7317fc15592";
pub(crate) const RECOGNITION_WEIGHTS_SHA256: &str =
    "e8bf34a6900addc8cd9ec1d1ea73ea56e97cb0d668c8c45508a885924078761f";

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
        let detection = load_graph(
            &runtime,
            &limits,
            &assets.detection_weights,
            GraphSpec::detection(),
        )?;
        let recognition = load_graph(
            &runtime,
            &limits,
            &assets.recognition_weights,
            GraphSpec::recognition(),
        )?;
        Ok(Self {
            runtime,
            detection,
            recognition,
            detection_identity: GraphSpec::detection().model_identity(),
            recognition_identity: GraphSpec::recognition().model_identity(),
        })
    }

    #[cfg(test)]
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
        let input = TensorInput::new(shape.to_vec(), data, self.runtime.limits())
            .map_err(|error| power_error("validate the OCR input tensor", error))?;
        self.execute_recognition_input(input, permit, cancellation)
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
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let tensor = self
            .recognition
            .run_with_output_projection(input, permit, cancellation, projection::ctc_top1)
            .map_err(|error| power_error("execute the projected OCR recognition graph", error))?;
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let receipt = self.runtime.receipt(
            self.recognition_identity.clone(),
            input_digest,
            output_digest,
        );
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
        ModelSessionBinding::new(bundle_model_identity(), session_execution_sha256(assets)?),
        session_limits(),
        resident_bytes,
    )
    .map_err(|error| power_error("declare the PP-OCRv6 model session", error))
}

pub(crate) fn batch_binding() -> UseResult<ExecutionBatchBinding> {
    ExecutionBatchBinding::new(
        bundle_weights_sha256(),
        named_sha256(b"a3s-ocr-ppocr-v6-staged-slot-layout-v2\0"),
        named_sha256(b"a3s-ocr-ppocr-v6-shape-cohort-scheduler-v5\0"),
    )
    .map_err(|error| power_error("bind the PP-OCRv6 staged batch", error))
}

pub(crate) fn bundle_model_identity() -> ModelIdentity {
    ModelIdentity::new(
        format!("{FAMILY}-bundle"),
        REVISION,
        bundle_weights_sha256(),
    )
}

fn bundle_weights_sha256() -> String {
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-ppocr-v6-bundle-weights-v1\0");
    digest.update(DETECTION_WEIGHTS_SHA256.as_bytes());
    digest.update(RECOGNITION_WEIGHTS_SHA256.as_bytes());
    format!("{:x}", digest.finalize())
}

fn session_execution_sha256(assets: &ModelAssets) -> UseResult<String> {
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
    update_bytes(&mut digest, DETECTION_GRAPH.as_bytes())?;
    update_bytes(&mut digest, RECOGNITION_GRAPH.as_bytes())?;
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
    role: &'static str,
    plan: &'static str,
    source_sha256: &'static str,
    source_opset: u32,
    weights_sha256: &'static str,
    projection_revision: Option<&'static str>,
}

impl GraphSpec {
    const fn detection() -> Self {
        Self {
            role: "detection",
            plan: DETECTION_GRAPH,
            source_sha256: DETECTION_SOURCE_SHA256,
            source_opset: 14,
            weights_sha256: DETECTION_WEIGHTS_SHA256,
            projection_revision: None,
        }
    }

    const fn recognition() -> Self {
        Self {
            role: "recognition",
            plan: RECOGNITION_GRAPH,
            source_sha256: RECOGNITION_SOURCE_SHA256,
            source_opset: 11,
            weights_sha256: RECOGNITION_WEIGHTS_SHA256,
            projection_revision: Some(projection::REVISION),
        }
    }

    fn graph_identity(self) -> GraphIdentity {
        GraphIdentity::new(
            FAMILY,
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
            format!("{FAMILY}-{}", self.role),
            revision,
            self.weights_sha256,
        )
    }
}

fn load_graph(
    runtime: &EmbeddedRuntime,
    limits: &InferenceLimits,
    weights_path: &Path,
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
        .verify_integrity(&format!("{FAMILY}-{}", spec.role), spec.weights_sha256)
        .map_err(|error| power_error("verify the reviewed OCR weights", error))?;
    let plan = GraphPlan::parse(spec.plan, &spec.graph_identity(), &weights, limits)
        .map_err(|error| power_error("validate the reviewed OCR graph", error))?;
    GraphExecutor::new(plan, weights, runtime.clone())
        .map_err(|error| power_error("materialize the reviewed OCR graph", error))
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
        let detection = GraphSpec::detection();
        let recognition = GraphSpec::recognition();
        assert_eq!(detection.graph_identity().role, "detection");
        assert_eq!(recognition.graph_identity().role, "recognition");
        assert_eq!(detection.source_opset, 14);
        assert_eq!(recognition.source_opset, 11);
    }

    #[test]
    fn graph_plans_keep_the_reviewed_node_inventory() {
        let detection: serde_json::Value = serde_json::from_str(DETECTION_GRAPH).unwrap();
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
            adjacent_single_consumer_hard_sigmoid_mul(DETECTION_GRAPH),
            13
        );
        assert_eq!(
            adjacent_single_consumer_hard_sigmoid_mul(RECOGNITION_GRAPH),
            5
        );
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
            batch_binding().unwrap().weights_sha256,
            bundle_model_identity().weights_sha256
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
            "paddlex-paddle3.0.0+ctc-top1-last-tie-finite-v1"
        );
        assert_eq!(
            recognition.receipt.model.weights_sha256,
            RECOGNITION_WEIGHTS_SHA256
        );
    }
}
