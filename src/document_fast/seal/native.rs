//! OCR-owned PicoDet layout graph over the model-neutral Power runtime.

use std::path::Path;
use std::sync::Arc;

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    EmbeddedRuntime, ExecutionDigest, ExecutionPermit, ExecutionReceipt, InferenceLimits,
    ModelIdentity, ModelSessionBinding, ModelSessionSpec, TensorInput, TensorOutput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use super::assets::{
    model_error, PicodetLayoutAssets, GRAPH_SHA256, MODEL_FAMILY, MODEL_REVISION,
    SOURCE_GRAPH_SHA256, WEIGHTS_COLLECTION_SHA256,
};
use super::preprocess::INPUT_SIDE;

const GRAPH: &str = include_str!("graphs/picodet_l_layout_3cls.json");
const GRAPH_ROLE: &str = "layout-raw-head";
const GRAPH_OPSET: u32 = 3;
pub(super) const LOCATION_COUNT: usize = 8_500;
pub(super) const OUTPUT_WIDTH: usize = 7;

pub(super) struct NativePicodetLayout {
    runtime: EmbeddedRuntime,
    graph: GraphExecutor,
    identity: ModelIdentity,
}

pub(super) struct NativeLayoutOutput {
    pub(super) tensor: TensorOutput,
    pub(super) receipt: ExecutionReceipt,
}

impl NativePicodetLayout {
    #[cfg(test)]
    pub(super) fn load(assets: &PicodetLayoutAssets) -> UseResult<Self> {
        let runtime = EmbeddedRuntime::new(
            a3s_power::inference::DevicePreference::Auto,
            session_limits(),
        )
        .map_err(|error| power_error("initialize the embedded runtime", error))?;
        Self::load_with_runtime(assets, runtime)
    }

    pub(super) fn load_with_runtime(
        assets: &PicodetLayoutAssets,
        runtime: EmbeddedRuntime,
    ) -> UseResult<Self> {
        let limits = runtime.limits().clone();
        let weights = Arc::new(
            WeightStore::open(&assets.root, &limits)
                .map_err(|error| power_error("open the PicoDet layout weights", error))?,
        );
        weights
            .verify_integrity(MODEL_FAMILY, WEIGHTS_COLLECTION_SHA256)
            .map_err(|error| power_error("verify the PicoDet layout weights", error))?;
        let plan = GraphPlan::parse(GRAPH, &graph_identity(), &weights, &limits)
            .map_err(|error| power_error("validate the reviewed PicoDet layout graph", error))?;
        let graph = GraphExecutor::new(plan, weights, runtime.clone())
            .map_err(|error| power_error("materialize the PicoDet layout graph", error))?;
        Ok(Self {
            runtime,
            graph,
            identity: model_identity(),
        })
    }

    #[cfg(test)]
    pub(super) fn begin(&self, cancellation: &CancellationToken) -> UseResult<ExecutionPermit> {
        self.runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit the PicoDet layout request", error))
    }

    pub(super) fn infer_batch(
        &self,
        values: Vec<f32>,
        batch_size: usize,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeLayoutOutput> {
        if batch_size == 0 || batch_size > 8 {
            return Err(input_error(
                "PicoDet layout batches require 1 through 8 image views.",
            ));
        }
        let shape = vec![batch_size, 3, INPUT_SIDE, INPUT_SIDE];
        let input = TensorInput::new(shape, values, self.runtime.limits())
            .map_err(|error| power_error("validate a PicoDet layout input tensor", error))?;
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let tensor = self
            .graph
            .run(input, permit, cancellation)
            .map_err(|error| power_error("execute the reviewed PicoDet layout graph", error))?;
        if tensor.shape != [batch_size, LOCATION_COUNT, OUTPUT_WIDTH]
            || tensor.values.len() != batch_size * LOCATION_COUNT * OUTPUT_WIDTH
            || tensor.values.iter().any(|value| !value.is_finite())
        {
            return Err(output_error(format!(
                "PicoDet layout output must be finite [N,{LOCATION_COUNT},{OUTPUT_WIDTH}] for N={batch_size}, found {:?}.",
                tensor.shape
            )));
        }
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let receipt = self
            .runtime
            .receipt(self.identity.clone(), input_digest, output_digest);
        Ok(NativeLayoutOutput { tensor, receipt })
    }
}

pub(super) fn session_limits() -> InferenceLimits {
    InferenceLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 32,
        ..InferenceLimits::default()
    }
}

pub(super) fn session_spec(assets: &PicodetLayoutAssets) -> UseResult<ModelSessionSpec> {
    ModelSessionSpec::new(
        ModelSessionBinding::new(model_identity(), session_execution_sha256()),
        session_limits(),
        file_size(&assets.weights)?,
    )
    .map_err(|error| power_error("declare the PicoDet layout model session", error))
}

fn graph_identity() -> GraphIdentity {
    GraphIdentity::new(
        MODEL_FAMILY,
        GRAPH_ROLE,
        "paddle-pir",
        SOURCE_GRAPH_SHA256,
        GRAPH_OPSET,
    )
}

fn model_identity() -> ModelIdentity {
    ModelIdentity::new(MODEL_FAMILY, MODEL_REVISION, WEIGHTS_COLLECTION_SHA256)
}

fn session_execution_sha256() -> String {
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-picodet-layout-session-v1\0");
    digest.update((GRAPH.len() as u64).to_le_bytes());
    digest.update(GRAPH.as_bytes());
    digest.update(GRAPH_SHA256.as_bytes());
    format!("{:x}", digest.finalize())
}

fn file_size(path: &Path) -> UseResult<u64> {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| {
            model_error(format!(
                "Failed to inspect PicoDet layout model bytes '{}': {error}",
                path.display()
            ))
        })
}

fn input_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_model_input_invalid", message)
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_model_output_invalid", message)
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

    #[test]
    fn reviewed_graph_keeps_exact_identity_and_inventory() {
        assert_eq!(format!("{:x}", Sha256::digest(GRAPH)), GRAPH_SHA256);
        let graph: serde_json::Value = serde_json::from_str(GRAPH).unwrap();
        assert_eq!(graph["family"], MODEL_FAMILY);
        assert_eq!(graph["role"], GRAPH_ROLE);
        assert_eq!(graph["source"]["sha256"], SOURCE_GRAPH_SHA256);
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 518);
        assert_eq!(graph["initializers"].as_array().unwrap().len(), 588);
    }
}
