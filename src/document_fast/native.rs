//! OCR-owned SLANet-Plus encoder graph over the model-neutral Power runtime.

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
    model_error, SlanetPlusAssets, DECODER_SHA256, DICTIONARY_SHA256, ENCODER_SOURCE_SHA256,
    ENCODER_WEIGHTS_SHA256, MODEL_FAMILY, MODEL_REVISION,
};
use super::preprocess::INPUT_SIDE;

const ENCODER_GRAPH: &str = include_str!("graphs/slanext_encoder.json");
const ENCODER_GRAPH_SHA256: &str =
    "8c0ba5d81cd3229653b2ce37d2976647336a094444b0009515fc9ed0efa48c64";
const GRAPH_FAMILY: &str = "slanet-plus";
const ENCODER_OPSET: u32 = 14;
const ENCODER_STEPS: usize = 256;
const CONTEXT_WIDTH: usize = 96;

pub(super) struct NativeSlanetPlus {
    runtime: EmbeddedRuntime,
    encoder: GraphExecutor,
    identity: ModelIdentity,
}

pub(super) struct NativeEncoderOutput {
    pub(super) tensor: TensorOutput,
    pub(super) receipt: ExecutionReceipt,
}

impl NativeSlanetPlus {
    #[cfg(test)]
    pub(super) fn load(assets: &SlanetPlusAssets) -> UseResult<Self> {
        let runtime = EmbeddedRuntime::new(
            a3s_power::inference::DevicePreference::Auto,
            session_limits(),
        )
        .map_err(|error| power_error("initialize the embedded runtime", error))?;
        Self::load_with_runtime(assets, runtime)
    }

    pub(super) fn load_with_runtime(
        assets: &SlanetPlusAssets,
        runtime: EmbeddedRuntime,
    ) -> UseResult<Self> {
        let limits = runtime.limits().clone();
        let root = assets
            .encoder_weights
            .parent()
            .ok_or_else(|| model_error("SLANet-Plus encoder weights have no parent directory."))?;
        let weights = Arc::new(
            WeightStore::open(root, &limits)
                .map_err(|error| power_error("open the SLANet-Plus encoder weights", error))?,
        );
        weights
            .verify_integrity(MODEL_FAMILY, ENCODER_WEIGHTS_SHA256)
            .map_err(|error| power_error("verify the SLANet-Plus encoder weights", error))?;
        let graph_identity = graph_identity();
        let plan = GraphPlan::parse(ENCODER_GRAPH, &graph_identity, &weights, &limits)
            .map_err(|error| power_error("validate the reviewed SLANet-Plus graph", error))?;
        let encoder = GraphExecutor::new(plan, weights, runtime.clone())
            .map_err(|error| power_error("materialize the SLANet-Plus encoder", error))?;
        Ok(Self {
            runtime,
            encoder,
            identity: encoder_identity(),
        })
    }

    #[cfg(test)]
    pub(super) fn begin(&self, cancellation: &CancellationToken) -> UseResult<ExecutionPermit> {
        self.runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit the SLANet-Plus request", error))
    }

    pub(super) fn encode_batch(
        &self,
        values: Vec<f32>,
        batch_size: usize,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeEncoderOutput> {
        if batch_size == 0 || batch_size > 16 {
            return Err(input_error(
                "SLANet-Plus encoder batches require 1 through 16 table crops.",
            ));
        }
        let shape = vec![batch_size, 3, INPUT_SIDE, INPUT_SIDE];
        let input = TensorInput::new(shape, values, self.runtime.limits())
            .map_err(|error| power_error("validate a SLANet-Plus input tensor", error))?;
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let tensor = self
            .encoder
            .run(input, permit, cancellation)
            .map_err(|error| power_error("execute the reviewed SLANet-Plus graph", error))?;
        if tensor.shape != [batch_size, ENCODER_STEPS, CONTEXT_WIDTH]
            || tensor.values.len() != batch_size * ENCODER_STEPS * CONTEXT_WIDTH
            || tensor.values.iter().any(|value| !value.is_finite())
        {
            return Err(output_error(format!(
                "SLANet-Plus encoder output must be finite [N,256,96] for N={batch_size}, found {:?}.",
                tensor.shape
            )));
        }
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let receipt = self
            .runtime
            .receipt(self.identity.clone(), input_digest, output_digest);
        Ok(NativeEncoderOutput { tensor, receipt })
    }
}

pub(super) fn session_limits() -> InferenceLimits {
    InferenceLimits {
        max_concurrent_requests: 1,
        max_queued_requests: 32,
        ..InferenceLimits::default()
    }
}

pub(super) fn session_spec(assets: &SlanetPlusAssets) -> UseResult<ModelSessionSpec> {
    let encoder_bytes = file_size(&assets.encoder_weights)?;
    let decoder_bytes = file_size(&assets.decoder_weights)?;
    let dictionary_bytes = file_size(&assets.dictionary)?;
    let resident_bytes = encoder_bytes
        .checked_add(decoder_bytes)
        .and_then(|bytes| bytes.checked_add(dictionary_bytes))
        .ok_or_else(|| model_error("SLANet-Plus resident model bytes overflowed."))?;
    ModelSessionSpec::new(
        ModelSessionBinding::new(bundle_identity(), session_execution_sha256(assets)?),
        session_limits(),
        resident_bytes,
    )
    .map_err(|error| power_error("declare the SLANet-Plus model session", error))
}

pub(super) fn bundle_identity() -> ModelIdentity {
    ModelIdentity::new(MODEL_FAMILY, MODEL_REVISION, bundle_weights_sha256())
}

fn encoder_identity() -> ModelIdentity {
    ModelIdentity::new(
        format!("{MODEL_FAMILY}-encoder"),
        MODEL_REVISION,
        ENCODER_WEIGHTS_SHA256,
    )
}

fn graph_identity() -> GraphIdentity {
    GraphIdentity::new(
        GRAPH_FAMILY,
        "table-encoder",
        "onnx",
        ENCODER_SOURCE_SHA256,
        ENCODER_OPSET,
    )
}

fn bundle_weights_sha256() -> String {
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-slanet-plus-wired-bundle-v1\0");
    digest.update(ENCODER_WEIGHTS_SHA256.as_bytes());
    digest.update(DECODER_SHA256.as_bytes());
    digest.update(DICTIONARY_SHA256.as_bytes());
    format!("{:x}", digest.finalize())
}

fn session_execution_sha256(assets: &SlanetPlusAssets) -> UseResult<String> {
    let dictionary = std::fs::read(&assets.dictionary).map_err(|error| {
        model_error(format!(
            "Failed to read the SLANet-Plus dictionary for session binding: {error}"
        ))
    })?;
    let mut digest = Sha256::new();
    digest.update(b"a3s-ocr-slanet-plus-session-v1\0");
    update_bytes(&mut digest, ENCODER_GRAPH.as_bytes())?;
    update_bytes(&mut digest, ENCODER_GRAPH_SHA256.as_bytes())?;
    update_bytes(&mut digest, DECODER_SHA256.as_bytes())?;
    update_bytes(&mut digest, &dictionary)?;
    Ok(format!("{:x}", digest.finalize()))
}

fn update_bytes(digest: &mut Sha256, bytes: &[u8]) -> UseResult<()> {
    let length = u64::try_from(bytes.len())
        .map_err(|_| model_error("SLANet-Plus session input length cannot be represented."))?;
    digest.update(length.to_le_bytes());
    digest.update(bytes);
    Ok(())
}

fn file_size(path: &Path) -> UseResult<u64> {
    std::fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| {
            model_error(format!(
                "Failed to inspect SLANet-Plus model bytes '{}': {error}",
                path.display()
            ))
        })
}

fn input_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.table_model_input_invalid", message)
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.table_model_output_invalid", message)
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
        assert_eq!(
            format!("{:x}", Sha256::digest(ENCODER_GRAPH)),
            ENCODER_GRAPH_SHA256
        );
        let graph: serde_json::Value = serde_json::from_str(ENCODER_GRAPH).unwrap();
        assert_eq!(graph["family"], "slanet-plus");
        assert_eq!(graph["role"], "table-encoder");
        assert_eq!(graph["source"]["sha256"], ENCODER_SOURCE_SHA256);
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 370);
        assert_eq!(graph["initializers"].as_array().unwrap().len(), 435);
    }

    #[test]
    fn bundle_identity_covers_encoder_decoder_and_dictionary() {
        let identity = bundle_identity();
        assert_eq!(identity.family, MODEL_FAMILY);
        assert_eq!(identity.revision, MODEL_REVISION);
        assert_eq!(identity.weights_sha256.len(), 64);
        assert_ne!(identity.weights_sha256, ENCODER_WEIGHTS_SHA256);
    }
}
