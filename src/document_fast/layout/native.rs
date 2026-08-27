use std::sync::Arc;

use a3s_power::inference::graph::{GraphExecutor, GraphIdentity, GraphPlan};
use a3s_power::inference::{
    DevicePreference, EmbeddedRuntime, ExecutionDigest, ExecutionPermit, ExecutionReceipt,
    InferenceLimits, ModelIdentity, TensorInput, TensorOutput, WeightStore,
};
use a3s_use_core::{UseError, UseResult};
use tokio_util::sync::CancellationToken;

use super::assets::DocumentLayoutAssets;
use super::profile::{
    CLASS_COUNT, FAMILY, GRAPH_OPSET, GRAPH_ROLE, INPUT_SIDE, LOCATION_COUNT, OUTPUT_WIDTH,
    REVISION, SOURCE_GRAPH_SHA256, WEIGHTS_COLLECTION_SHA256,
};

pub(super) struct NativeDocumentLayout {
    runtime: EmbeddedRuntime,
    graph: GraphExecutor,
    identity: ModelIdentity,
}

pub(super) struct NativeDocumentLayoutOutput {
    pub(super) tensor: TensorOutput,
    pub(super) receipt: ExecutionReceipt,
}

impl NativeDocumentLayout {
    pub(super) fn load(assets: &DocumentLayoutAssets) -> UseResult<Self> {
        let limits = InferenceLimits {
            max_concurrent_requests: 1,
            max_queued_requests: 32,
            ..InferenceLimits::default()
        };
        let runtime = EmbeddedRuntime::new(DevicePreference::Auto, limits.clone())
            .map_err(|error| power_error("initialize the document-layout runtime", error))?;
        let weights = Arc::new(
            WeightStore::open(&assets.root, &limits)
                .map_err(|error| power_error("open the document-layout weights", error))?,
        );
        weights
            .verify_integrity(FAMILY, WEIGHTS_COLLECTION_SHA256)
            .map_err(|error| power_error("verify the document-layout weights", error))?;
        let identity = GraphIdentity::new(
            FAMILY,
            GRAPH_ROLE,
            "paddle-pir",
            SOURCE_GRAPH_SHA256,
            GRAPH_OPSET,
        );
        let plan = GraphPlan::parse(&assets.graph, &identity, &weights, &limits)
            .map_err(|error| power_error("validate the reviewed document-layout graph", error))?;
        let graph = GraphExecutor::new(plan, weights, runtime.clone())
            .map_err(|error| power_error("materialize the document-layout graph", error))?;
        Ok(Self {
            runtime,
            graph,
            identity: ModelIdentity::new(FAMILY, REVISION, WEIGHTS_COLLECTION_SHA256),
        })
    }

    pub(super) fn limits(&self) -> &InferenceLimits {
        self.runtime.limits()
    }

    pub(super) fn begin(&self, cancellation: &CancellationToken) -> UseResult<ExecutionPermit> {
        self.runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit the document-layout request", error))
    }

    pub(super) fn infer_batch(
        &self,
        values: Vec<f32>,
        batch_size: usize,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<NativeDocumentLayoutOutput> {
        let input = TensorInput::new(
            vec![batch_size, 3, INPUT_SIDE, INPUT_SIDE],
            values,
            self.runtime.limits(),
        )
        .map_err(|error| power_error("validate a document-layout input tensor", error))?;
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        let tensor = self
            .graph
            .run(input, permit, cancellation)
            .map_err(|error| power_error("execute the reviewed document-layout graph", error))?;
        if tensor.shape != [batch_size, LOCATION_COUNT, OUTPUT_WIDTH]
            || tensor.values.len() != batch_size * LOCATION_COUNT * (4 + CLASS_COUNT)
            || tensor.values.iter().any(|value| !value.is_finite())
        {
            return Err(output_error(format!(
                "PP-DocLayout-S output must be finite [N,{LOCATION_COUNT},{OUTPUT_WIDTH}] for N={batch_size}, found {:?}.",
                tensor.shape
            )));
        }
        let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
        let receipt = self
            .runtime
            .receipt(self.identity.clone(), input_digest, output_digest);
        Ok(NativeDocumentLayoutOutput { tensor, receipt })
    }
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.document_layout_output_invalid", message)
}

fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} through a3s-power: {error}"),
    )
}
