use std::path::Path;
use std::sync::{Arc, Mutex};

use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;

use super::*;
use crate::{
    OcrExecutionDigest, OcrExecutionModel, OcrExecutionReceipt, OcrExecutionRuntime, OcrProvider,
    OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
};

#[derive(Clone, Copy)]
enum BatchMode {
    Exact,
    Reverse,
}

struct StagedProvider {
    calls: Arc<Mutex<Vec<Vec<String>>>>,
    mode: BatchMode,
}

impl StagedProvider {
    fn new(mode: BatchMode) -> (Self, Arc<Mutex<Vec<Vec<String>>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                calls: Arc::clone(&calls),
                mode,
            },
            calls,
        )
    }
}

#[async_trait]
impl OcrProvider for StagedProvider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        OcrProviderDescriptor::new("staged-fixture", "fixture-engine", false)
            .unwrap()
            .with_stages(vec![OcrStage::Preprocessing, OcrStage::Text])
            .unwrap()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        OcrProviderStatus {
            readiness: Readiness::Ready,
            model: Some("fixture-model".to_string()),
            model_dir: None,
            message: "ready".to_string(),
            suggestions: Vec::new(),
        }
    }

    async fn recognize(&self, _input: crate::OcrInput) -> UseResult<OcrProviderOutput> {
        unreachable!("the staged fixture overrides recognize_batch")
    }

    async fn recognize_batch(
        &self,
        request: OcrProviderBatchRequest,
    ) -> UseResult<OcrProviderBatchOutput> {
        self.calls.lock().unwrap().push(
            request
                .slots
                .iter()
                .map(|slot| slot.slot_id.to_string())
                .collect(),
        );
        assert_eq!(
            request.stages,
            vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Formula]
        );
        let mut slots = request
            .slots
            .into_iter()
            .map(|slot| {
                let fails_text = slot.slot_id.as_str() == "slot-c";
                OcrProviderBatchSlotOutput {
                    slot_id: slot.slot_id,
                    stages: vec![
                        OcrStageOutcome::completed(OcrStage::Preprocessing),
                        if fails_text {
                            OcrStageOutcome::failed(
                                OcrStage::Text,
                                UseError::new("fixture.text_failed", "text failed"),
                            )
                        } else {
                            OcrStageOutcome::completed(OcrStage::Text)
                        },
                        OcrStageOutcome::unsupported(OcrStage::Formula),
                    ],
                    output: (!fails_text).then(output),
                }
            })
            .collect::<Vec<_>>();
        if matches!(self.mode, BatchMode::Reverse) {
            slots.reverse();
        }
        Ok(OcrProviderBatchOutput {
            slots,
            execution_receipts: Vec::new(),
        })
    }
}

struct TextProvider;

#[async_trait]
impl OcrProvider for TextProvider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        OcrProviderDescriptor::new("text-fixture", "fixture-engine", false).unwrap()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        OcrProviderStatus {
            readiness: Readiness::Ready,
            model: None,
            model_dir: None,
            message: "ready".to_string(),
            suggestions: Vec::new(),
        }
    }

    async fn recognize(&self, _input: crate::OcrInput) -> UseResult<OcrProviderOutput> {
        Ok(output())
    }
}

#[tokio::test]
async fn staged_batch_preserves_order_partial_failures_and_fingerprints() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.bmp");
    let invalid = directory.path().join("invalid.bin");
    let third = directory.path().join("third.bmp");
    write(&first, b"BMfirst");
    write(&invalid, b"not-an-image");
    write(&third, b"BMthird");
    let (provider, calls) = StagedProvider::new(BatchMode::Exact);
    let client = crate::OcrClient::with_provider(provider).unwrap();

    let result = client
        .extract_batch(
            OcrBatchRequest::new(
                vec![OcrStage::Formula, OcrStage::Text, OcrStage::Preprocessing],
                vec![
                    slot("slot-a", &first),
                    slot("slot-b", &invalid),
                    slot("slot-c", &third),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(result.schema, OcrBatchResult::SCHEMA);
    assert_eq!(
        result.requested_stages,
        vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Formula]
    );
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        &[vec!["slot-a".to_string(), "slot-c".to_string()]]
    );
    assert_eq!(
        result
            .slots
            .iter()
            .map(|slot| slot.slot_id.as_str())
            .collect::<Vec<_>>(),
        vec!["slot-a", "slot-b", "slot-c"]
    );
    assert_eq!(result.slots[0].status, OcrBatchSlotStatus::Partial);
    assert!(result.slots[0].result.is_some());
    let fingerprint = result.slots[0].model_fingerprint.as_ref().unwrap();
    assert_eq!(fingerprint.model.as_deref(), Some("fixture-model"));
    assert_eq!(fingerprint.execution_models.len(), 1);
    assert_eq!(fingerprint.declaration_sha256.len(), 64);
    assert_eq!(result.slots[1].status, OcrBatchSlotStatus::Failed);
    assert!(result.slots[1].source.is_none());
    assert_eq!(
        result.slots[1].stages[0].error.as_ref().unwrap().code,
        "use.ocr.source_type_unsupported"
    );
    assert_eq!(result.slots[2].status, OcrBatchSlotStatus::Partial);
    assert!(result.slots[2].result.is_none());
    assert_eq!(
        result.slots[2].stages[1].error.as_ref().unwrap().code,
        "fixture.text_failed"
    );
    assert_eq!(result.provider.supported_stages.len(), 2);
    assert_eq!(result.provider.declaration_sha256.len(), 64);
}

#[tokio::test]
async fn malformed_provider_slot_order_fails_the_contract() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.bmp");
    let second = directory.path().join("second.bmp");
    write(&first, b"BMfirst");
    write(&second, b"BMsecond");
    let (provider, _) = StagedProvider::new(BatchMode::Reverse);
    let client = crate::OcrClient::with_provider(provider).unwrap();
    let error = client
        .extract_batch(
            OcrBatchRequest::new(
                vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Formula],
                vec![slot("slot-a", first), slot("slot-c", second)],
            )
            .unwrap(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code, "use.ocr.provider_batch_invalid");
}

#[tokio::test]
async fn default_provider_adapter_is_text_only_and_deterministic() {
    let directory = tempfile::tempdir().unwrap();
    let image = directory.path().join("image.bmp");
    write(&image, b"BMfixture");
    let client = crate::OcrClient::with_provider(TextProvider).unwrap();
    let result = client
        .extract_batch(
            OcrBatchRequest::new(
                vec![OcrStage::Table, OcrStage::Text],
                vec![slot("slot-a", image)],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.requested_stages,
        vec![OcrStage::Text, OcrStage::Table]
    );
    assert_eq!(result.slots[0].status, OcrBatchSlotStatus::Partial);
    assert_eq!(result.slots[0].stages[0].status, OcrStageStatus::Completed);
    assert_eq!(
        result.slots[0].stages[1].status,
        OcrStageStatus::Unsupported
    );
}

#[test]
fn batch_shape_and_slot_identity_are_bounded() {
    assert!(OcrBatchSlotId::new("target-a:surface-1").is_ok());
    assert!(OcrBatchSlotId::new("-invalid").is_err());
    let path = Path::new("fixture.bmp");
    assert!(OcrBatchRequest::new(Vec::new(), vec![slot("slot-a", path)]).is_err());
    assert!(OcrBatchRequest::new(
        vec![OcrStage::Text, OcrStage::Text],
        vec![slot("slot-a", path)]
    )
    .is_err());
    assert!(OcrBatchRequest::new(
        vec![OcrStage::Text],
        vec![slot("slot-a", path), slot("slot-a", path)]
    )
    .is_err());
}

#[test]
fn batch_public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OcrBatchRequest>();
    assert_send_sync::<OcrBatchResult>();
    assert_send_sync::<OcrProviderBatchRequest>();
    assert_send_sync::<OcrProviderBatchOutput>();
}

#[test]
fn receipt_v4_requires_well_formed_microbatch_evidence() {
    let mut output = output();
    output.execution_receipts[0].schema = "a3s.power.embedded-execution-receipt.v4".to_string();
    assert_eq!(
        crate::output_validation::validate_provider_output(&output)
            .unwrap_err()
            .code,
        "use.ocr.provider_output_invalid"
    );

    output.execution_receipts[0].microbatch = Some(crate::OcrMicrobatchExecutionEvidence {
        schema: "a3s.power.microbatch-execution.v1".to_string(),
        session_declaration_sha256: Some("d".repeat(64)),
        plan_sha256: "e".repeat(64),
        batch_index: 0,
        batch_count: 1,
        slot_count: 2,
        model_admission_queued: false,
        device_admission_queued: true,
    });
    crate::output_validation::validate_provider_output(&output).unwrap();
}

fn slot(id: &str, path: impl Into<std::path::PathBuf>) -> OcrBatchSlotRequest {
    OcrBatchSlotRequest::new(OcrBatchSlotId::new(id).unwrap(), path)
}

fn write(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
}

fn output() -> OcrProviderOutput {
    OcrProviderOutput {
        model: Some("fixture-model".to_string()),
        text: "fixture text".to_string(),
        blocks: Vec::new(),
        execution_receipts: vec![receipt()],
        warnings: Vec::new(),
    }
}

fn receipt() -> OcrExecutionReceipt {
    OcrExecutionReceipt {
        schema: "a3s.power.embedded-execution-receipt.v1".to_string(),
        model: OcrExecutionModel {
            family: "fixture-model".to_string(),
            revision: "revision-1".to_string(),
            weights_sha256: "a".repeat(64),
        },
        runtime: OcrExecutionRuntime {
            name: "a3s-power".to_string(),
            version: "0.7.0".to_string(),
            device: "cpu".to_string(),
        },
        input: OcrExecutionDigest {
            representation: "image-request".to_string(),
            sha256: "b".repeat(64),
            byte_length: 7,
            item_count: 1,
        },
        output: OcrExecutionDigest {
            representation: "utf8-text".to_string(),
            sha256: "c".repeat(64),
            byte_length: 12,
            item_count: 12,
        },
        microbatch: None,
    }
}
