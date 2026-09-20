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

use super::assets::{model_error, SealTextAssets};
use super::profile::{FAMILY, REVISION, ROLE, SOURCE_GRAPH_SHA256, WEIGHTS_COLLECTION_SHA256};

const GRAPH_FORMAT: &str = "onnx";
const GRAPH_OPSET: u32 = 14;
pub(super) const MAX_CONCURRENCY: usize = 4;

pub(super) struct NativeSealText {
    graph: GraphExecutor,
    identity: ModelIdentity,
}

pub(super) struct NativeSealTextOutput {
    pub(super) tensor: TensorOutput,
    pub(super) receipt: ExecutionReceipt,
}

impl NativeSealText {
    pub(super) fn load_with_runtime(
        assets: &SealTextAssets,
        runtime: EmbeddedRuntime,
    ) -> UseResult<Self> {
        let limits = runtime.limits().clone();
        let weights = Arc::new(
            WeightStore::open(&assets.root, &limits)
                .map_err(|error| power_error("open seal-text weights", error))?,
        );
        weights
            .verify_integrity(FAMILY, WEIGHTS_COLLECTION_SHA256)
            .map_err(|error| power_error("verify seal-text weights", error))?;
        let identity =
            GraphIdentity::new(FAMILY, ROLE, GRAPH_FORMAT, SOURCE_GRAPH_SHA256, GRAPH_OPSET);
        let plan = GraphPlan::parse(&assets.graph, &identity, &weights, &limits)
            .map_err(|error| power_error("validate the reviewed seal-text graph", error))?;
        let graph = GraphExecutor::new(plan, weights, runtime)
            .map_err(|error| power_error("materialize the seal-text graph", error))?;
        Ok(Self {
            graph,
            identity: model_identity(),
        })
    }

    pub(super) fn infer_batch(
        &self,
        shape: [usize; 4],
        values: Vec<f32>,
        runtime: &EmbeddedRuntime,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeSealTextOutput> {
        if shape[0] == 0 || shape[1] != 3 || shape[2] == 0 || shape[3] == 0 {
            return Err(input_error(format!(
                "Seal-text input must be a positive [N,3,H,W] tensor, found {shape:?}."
            )));
        }
        let batch_size = shape[0];
        let input = TensorInput::new(shape.to_vec(), values, runtime.limits())
            .map_err(|error| power_error("validate a seal-text input tensor", error))?;
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let tensor = self
            .graph
            .run(input, permit, cancellation)
            .map_err(|error| power_error("execute the reviewed seal-text graph", error))?;
        if tensor.shape.len() != 4
            || tensor.shape[0] != batch_size
            || tensor.shape[1] != 1
            || tensor.shape[2] == 0
            || tensor.shape[3] == 0
            || tensor.values.iter().any(|value| !value.is_finite())
        {
            return Err(output_error(format!(
                "Seal-text output must be finite [N,1,H,W] for N={batch_size}, found {:?}.",
                tensor.shape,
            )));
        }
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let receipt = runtime.receipt(self.identity.clone(), input_digest, output_digest);
        Ok(NativeSealTextOutput { tensor, receipt })
    }
}

pub(super) fn session_limits() -> InferenceLimits {
    InferenceLimits {
        max_concurrent_requests: MAX_CONCURRENCY,
        max_queued_requests: 64,
        ..InferenceLimits::default()
    }
}

pub(super) fn session_spec(assets: &SealTextAssets) -> UseResult<ModelSessionSpec> {
    ModelSessionSpec::new(
        ModelSessionBinding::new(model_identity(), session_execution_sha256(assets)),
        session_limits(),
        file_size(&assets.weights)?,
    )
    .map_err(|error| power_error("declare the seal-text model session", error))
}

fn model_identity() -> ModelIdentity {
    ModelIdentity::new(FAMILY, REVISION, WEIGHTS_COLLECTION_SHA256)
}

fn session_execution_sha256(assets: &SealTextAssets) -> String {
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-seal-text-session-v1\0");
    digest.update((assets.graph.len() as u64).to_le_bytes());
    digest.update(assets.graph.as_bytes());
    format!("{:x}", digest.finalize())
}

fn file_size(path: &Path) -> UseResult<u64> {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| {
            model_error(format!(
                "Failed to inspect seal-text model bytes '{}': {error}",
                path.display()
            ))
        })
}

fn input_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_text_model_input_invalid", message)
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.seal_text_model_output_invalid", message)
}

fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} through a3s-power: {error}"),
    )
}
