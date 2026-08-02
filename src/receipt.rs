use a3s_power::inference::{ExecutionDigest, ExecutionReceipt, ExecutionRepresentation};

use crate::models::{
    OcrExecutionDigest, OcrExecutionModel, OcrExecutionReceipt, OcrExecutionRuntime,
};

pub(crate) fn project_receipt(receipt: ExecutionReceipt) -> OcrExecutionReceipt {
    OcrExecutionReceipt {
        schema: receipt.schema,
        model: OcrExecutionModel {
            family: receipt.model.family,
            revision: receipt.model.revision,
            weights_sha256: receipt.model.weights_sha256,
        },
        runtime: OcrExecutionRuntime {
            name: receipt.runtime.name,
            version: receipt.runtime.version,
            device: receipt.runtime.device,
        },
        input: project_digest(receipt.input),
        output: project_digest(receipt.output),
    }
}

fn project_digest(digest: ExecutionDigest) -> OcrExecutionDigest {
    let representation = match digest.representation {
        ExecutionRepresentation::F32Tensor => "f32-tensor",
        ExecutionRepresentation::ImageRequest => "image-request",
        ExecutionRepresentation::TokenIds => "token-ids",
        ExecutionRepresentation::Utf8Text => "utf8-text",
    };
    OcrExecutionDigest {
        representation: representation.to_string(),
        sha256: digest.sha256,
        byte_length: digest.byte_length,
        item_count: digest.item_count,
    }
}
