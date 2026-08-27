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

use super::assets::{model_error, PicodetLayoutAssets};
use super::profile::PicodetLayoutProfile;

const GRAPH_ROLE: &str = "layout-raw-head";
const GRAPH_OPSET: u32 = 3;
pub(super) const MAX_BATCH_SIZE: usize = 32;

pub(super) struct NativePicodetLayout {
    runtime: EmbeddedRuntime,
    graph: GraphExecutor,
    profile: PicodetLayoutProfile,
    identity: ModelIdentity,
}

pub(super) struct NativeLayoutOutput {
    pub(super) tensor: TensorOutput,
    pub(super) receipt: ExecutionReceipt,
    pub(super) profile: PicodetLayoutProfile,
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
        let profile = assets.profile;
        let limits = runtime.limits().clone();
        let weights = Arc::new(
            WeightStore::open(&assets.root, &limits)
                .map_err(|error| power_error("open the PicoDet layout weights", error))?,
        );
        weights
            .verify_integrity(profile.family(), profile.weights_collection_sha256())
            .map_err(|error| power_error("verify the PicoDet layout weights", error))?;
        let plan = GraphPlan::parse(&assets.graph, &graph_identity(profile), &weights, &limits)
            .map_err(|error| power_error("validate the reviewed PicoDet layout graph", error))?;
        let graph = GraphExecutor::new(plan, weights, runtime.clone())
            .map_err(|error| power_error("materialize the PicoDet layout graph", error))?;
        Ok(Self {
            runtime,
            graph,
            profile,
            identity: model_identity(profile),
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
        if batch_size == 0 || batch_size > MAX_BATCH_SIZE {
            return Err(input_error(format!(
                "PicoDet layout batches require 1 through {MAX_BATCH_SIZE} image views.",
            )));
        }
        let input_side = self.profile.input_side();
        let location_count = self.profile.location_count();
        let output_width = self.profile.output_width();
        let shape = vec![batch_size, 3, input_side, input_side];
        let input = TensorInput::new(shape, values, self.runtime.limits())
            .map_err(|error| power_error("validate a PicoDet layout input tensor", error))?;
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let tensor = self
            .graph
            .run(input, permit, cancellation)
            .map_err(|error| power_error("execute the reviewed PicoDet layout graph", error))?;
        if tensor.shape != [batch_size, location_count, output_width]
            || tensor.values.len() != batch_size * location_count * output_width
            || tensor.values.iter().any(|value| !value.is_finite())
        {
            return Err(output_error(format!(
                "PicoDet layout output must be finite [N,{location_count},{output_width}] for N={batch_size}, found {:?}.",
                tensor.shape
            )));
        }
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let receipt = self
            .runtime
            .receipt(self.identity.clone(), input_digest, output_digest);
        Ok(NativeLayoutOutput {
            tensor,
            receipt,
            profile: self.profile,
        })
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
    let profile = assets.profile;
    ModelSessionSpec::new(
        ModelSessionBinding::new(
            model_identity(profile),
            session_execution_sha256(profile, &assets.graph),
        ),
        session_limits(),
        file_size(&assets.weights)?,
    )
    .map_err(|error| power_error("declare the PicoDet layout model session", error))
}

fn graph_identity(profile: PicodetLayoutProfile) -> GraphIdentity {
    GraphIdentity::new(
        profile.family(),
        GRAPH_ROLE,
        "paddle-pir",
        profile.source_graph_sha256(),
        GRAPH_OPSET,
    )
}

fn model_identity(profile: PicodetLayoutProfile) -> ModelIdentity {
    ModelIdentity::new(
        profile.family(),
        profile.revision(),
        profile.weights_collection_sha256(),
    )
}

fn session_execution_sha256(profile: PicodetLayoutProfile, graph: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-picodet-layout-session-v1\0");
    digest.update((graph.len() as u64).to_le_bytes());
    digest.update(graph.as_bytes());
    digest.update(profile.graph_sha256().as_bytes());
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
