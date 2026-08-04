use a3s_power::inference::{ExecutionDigest, ExecutionReceipt, ExecutionRepresentation};

use crate::models::{
    OcrExecutionDigest, OcrExecutionModel, OcrExecutionReceipt, OcrExecutionRuntime,
    OcrMicrobatchExecutionEvidence,
};

pub(crate) fn project_receipt(receipt: ExecutionReceipt) -> OcrExecutionReceipt {
    let microbatch = receipt
        .microbatch
        .map(|evidence| OcrMicrobatchExecutionEvidence {
            schema: evidence.schema,
            session_declaration_sha256: evidence.session_declaration_sha256,
            plan_sha256: evidence.plan_sha256,
            batch_index: evidence.batch_index,
            batch_count: evidence.batch_count,
            slot_count: evidence.slot_count,
            model_admission_queued: evidence.model_admission_queued,
            device_admission_queued: evidence.device_admission_queued,
        });
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
        microbatch,
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
