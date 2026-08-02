//! OCR-owned PP-OCRv6 graph identity over the model-neutral Power runtime.

use std::path::Path;
use std::sync::Arc;

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, ExecutionDigest, ExecutionPermit, ExecutionReceipt,
    InferenceLimits, ModelIdentity, TensorInput, TensorOutput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use tokio_util::sync::CancellationToken;

use crate::assets::ModelAssets;

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
    pub(crate) fn load(assets: &ModelAssets) -> UseResult<Self> {
        let limits = InferenceLimits::default();
        let runtime = EmbeddedRuntime::new(DevicePreference::Auto, limits.clone())
            .map_err(|error| power_error("initialize the embedded runtime", error))?;
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

    pub(crate) fn begin(&self, cancellation: &CancellationToken) -> UseResult<ExecutionPermit> {
        self.runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit the OCR request", error))
    }

    pub(crate) fn detect(
        &self,
        data: Vec<f32>,
        shape: [usize; 4],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        self.execute(
            &self.detection,
            &self.detection_identity,
            data,
            shape,
            permit,
            cancellation,
        )
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
        self.execute(
            &self.recognition,
            &self.recognition_identity,
            data,
            shape,
            permit,
            cancellation,
        )
    }

    fn execute(
        &self,
        graph: &GraphExecutor,
        identity: &ModelIdentity,
        data: Vec<f32>,
        shape: [usize; 4],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeGraphOutput> {
        if shape[1] != 3 {
            return Err(UseError::new(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 graph input must use NCHW layout with three channels.",
            ));
        }
        let input = TensorInput::new(shape.to_vec(), data, self.runtime.limits())
            .map_err(|error| power_error("validate the OCR input tensor", error))?;
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
}

#[derive(Clone, Copy)]
struct GraphSpec {
    role: &'static str,
    plan: &'static str,
    source_sha256: &'static str,
    source_opset: u32,
    weights_sha256: &'static str,
}

impl GraphSpec {
    const fn detection() -> Self {
        Self {
            role: "detection",
            plan: DETECTION_GRAPH,
            source_sha256: DETECTION_SOURCE_SHA256,
            source_opset: 14,
            weights_sha256: DETECTION_WEIGHTS_SHA256,
        }
    }

    const fn recognition() -> Self {
        Self {
            role: "recognition",
            plan: RECOGNITION_GRAPH,
            source_sha256: RECOGNITION_SOURCE_SHA256,
            source_opset: 11,
            weights_sha256: RECOGNITION_WEIGHTS_SHA256,
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
        ModelIdentity::new(
            format!("{FAMILY}-{}", self.role),
            REVISION,
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
    }

    #[test]
    #[ignore = "requires the pinned official PP-OCRv6 native bundle"]
    fn official_weights_execute_with_pinned_cpu_fixtures() {
        let root = std::env::var_os("A3S_PPOCR_V6_MODEL")
            .expect("A3S_PPOCR_V6_MODEL must name the pinned official model bundle");
        let root = std::path::PathBuf::from(root);
        let assets = ModelAssets {
            root: root.clone(),
            detection_weights: root.join("det/model.safetensors"),
            detection_config: root.join("det/inference.yml"),
            recognition_weights: root.join("rec/model.safetensors"),
            recognition_config: root.join("rec/inference.yml"),
            source: OcrInstallSource::Environment,
        };
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
        assert_eq!(detection.tensor.shape, [1, 1, 64, 64]);
        assert_eq!(detection.tensor, repeated_detection.tensor);
        assert_eq!(detection.receipt.output, repeated_detection.receipt.output);
        assert_eq!(detection.receipt.output.byte_length, 16_384);
        assert_eq!(detection.receipt.output.item_count, 4_096);
        assert_eq!(
            detection.receipt.model.weights_sha256,
            DETECTION_WEIGHTS_SHA256
        );

        let recognition = native
            .recognize(
                vec![0.0; 3 * 48 * 320],
                [1, 3, 48, 320],
                &permit,
                &cancellation,
            )
            .unwrap();
        let repeated_recognition = native
            .recognize(
                vec![0.0; 3 * 48 * 320],
                [1, 3, 48, 320],
                &permit,
                &cancellation,
            )
            .unwrap();
        assert_eq!(recognition.tensor.shape, [1, 40, 18_710]);
        assert_eq!(recognition.tensor, repeated_recognition.tensor);
        assert_eq!(
            recognition.receipt.output,
            repeated_recognition.receipt.output
        );
        assert_eq!(recognition.receipt.output.byte_length, 2_993_600);
        assert_eq!(recognition.receipt.output.item_count, 748_400);
        assert_eq!(
            recognition.receipt.model.weights_sha256,
            RECOGNITION_WEIGHTS_SHA256
        );
    }
}
