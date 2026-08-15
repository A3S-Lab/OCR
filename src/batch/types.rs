use std::collections::BTreeSet;
use std::path::PathBuf;

use a3s_use_core::{Artifact, UseError, UseResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    OcrExecutionModel, OcrExecutionReceipt, OcrInput, OcrProviderDescriptor, OcrProviderOutput,
    OcrResult, OcrStageEvidence,
};

pub(super) const MAX_BATCH_SLOTS: usize = 256;
pub(super) const MAX_BATCH_INPUT_BYTES: u64 = 256 * 1024 * 1024;

/// Provider-neutral stages in their canonical execution and evidence order.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
pub enum OcrStage {
    Orientation,
    Preprocessing,
    Layout,
    Text,
    Table,
    Formula,
    Seal,
}

/// Stable caller-owned identity for one batch slot.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct OcrBatchSlotId(String);

impl OcrBatchSlotId {
    pub fn new(value: impl Into<String>) -> UseResult<Self> {
        let value = Self(value.into());
        value.validate()?;
        Ok(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn validate(&self) -> UseResult<()> {
        let bytes = self.0.as_bytes();
        if bytes.is_empty()
            || bytes.len() > 128
            || !bytes
                .first()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            || !bytes.iter().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b':')
            })
        {
            return Err(batch_error(
                "OCR batch slot IDs must contain 1 through 128 ASCII letters, digits, dots, hyphens, underscores, or colons and start with a letter or digit.",
            ));
        }
        Ok(())
    }
}

impl std::fmt::Display for OcrBatchSlotId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrBatchSlotRequest {
    pub slot_id: OcrBatchSlotId,
    pub path: PathBuf,
}

impl OcrBatchSlotRequest {
    pub fn new(slot_id: OcrBatchSlotId, path: impl Into<PathBuf>) -> Self {
        Self {
            slot_id,
            path: path.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrBatchRequest {
    pub stages: Vec<OcrStage>,
    pub slots: Vec<OcrBatchSlotRequest>,
}

impl OcrBatchRequest {
    pub fn new(stages: Vec<OcrStage>, slots: Vec<OcrBatchSlotRequest>) -> UseResult<Self> {
        let request = Self { stages, slots };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> UseResult<()> {
        if self.stages.is_empty() {
            return Err(batch_error("An OCR batch requires at least one stage."));
        }
        if self.slots.is_empty() || self.slots.len() > MAX_BATCH_SLOTS {
            return Err(batch_error(format!(
                "An OCR batch requires 1 through {MAX_BATCH_SLOTS} slots."
            )));
        }
        let mut stages = BTreeSet::new();
        for stage in &self.stages {
            if !stages.insert(*stage) {
                return Err(batch_error("OCR batch stages must be unique."));
            }
        }
        let mut slots = BTreeSet::new();
        for slot in &self.slots {
            slot.slot_id.validate()?;
            if !slots.insert(slot.slot_id.as_str()) {
                return Err(batch_error("OCR batch slot IDs must be unique."));
            }
        }
        Ok(())
    }

    pub(crate) fn canonical_stages(&self) -> Vec<OcrStage> {
        let mut stages = self.stages.clone();
        stages.sort_unstable();
        stages
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum OcrStageStatus {
    Completed,
    Failed,
    Unsupported,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrStageOutcome {
    pub stage: OcrStage,
    pub status: OcrStageStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<UseError>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<OcrStageEvidence>,
}

impl OcrStageOutcome {
    pub fn completed(stage: OcrStage) -> Self {
        Self {
            stage,
            status: OcrStageStatus::Completed,
            error: None,
            evidence: None,
        }
    }

    pub fn completed_with_evidence(evidence: OcrStageEvidence) -> Self {
        Self {
            stage: evidence.stage(),
            status: OcrStageStatus::Completed,
            error: None,
            evidence: Some(evidence),
        }
    }

    pub fn failed(stage: OcrStage, error: UseError) -> Self {
        Self {
            stage,
            status: OcrStageStatus::Failed,
            error: Some(error),
            evidence: None,
        }
    }

    pub fn unsupported(stage: OcrStage) -> Self {
        Self {
            stage,
            status: OcrStageStatus::Unsupported,
            error: None,
            evidence: None,
        }
    }

    pub fn skipped(stage: OcrStage, error: UseError) -> Self {
        Self {
            stage,
            status: OcrStageStatus::Skipped,
            error: Some(error),
            evidence: None,
        }
    }

    pub(crate) fn validate(&self) -> UseResult<()> {
        let expects_error = matches!(
            self.status,
            OcrStageStatus::Failed | OcrStageStatus::Skipped
        );
        if expects_error != self.error.is_some() {
            return Err(provider_batch_error(
                "Failed or skipped OCR stages require an error, while completed or unsupported stages must not carry one.",
            ));
        }
        let expects_evidence = self.status == OcrStageStatus::Completed
            && matches!(self.stage, OcrStage::Table | OcrStage::Seal);
        if expects_evidence != self.evidence.is_some() {
            return Err(provider_batch_error(
                "Completed table or seal stages require typed evidence, while every other stage outcome must not carry it.",
            ));
        }
        if let Some(evidence) = &self.evidence {
            if evidence.stage() != self.stage {
                return Err(provider_batch_error(
                    "Structured OCR evidence must match its completed stage.",
                ));
            }
            evidence.validate()?;
        }
        Ok(())
    }
}

/// One validated slot passed across the provider boundary.
#[derive(Clone)]
pub struct OcrProviderBatchSlot {
    pub slot_id: OcrBatchSlotId,
    pub input: OcrInput,
}

impl std::fmt::Debug for OcrProviderBatchSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrProviderBatchSlot")
            .field("slot_id", &self.slot_id)
            .field("input", &"validated-image")
            .finish()
    }
}

/// Canonically ordered validated slots supplied to one provider call.
pub struct OcrProviderBatchRequest {
    pub stages: Vec<OcrStage>,
    pub slots: Vec<OcrProviderBatchSlot>,
}

impl std::fmt::Debug for OcrProviderBatchRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrProviderBatchRequest")
            .field("stages", &self.stages)
            .field("slot_count", &self.slots.len())
            .finish()
    }
}

#[derive(Clone)]
pub struct OcrProviderBatchSlotOutput {
    pub slot_id: OcrBatchSlotId,
    pub stages: Vec<OcrStageOutcome>,
    pub output: Option<OcrProviderOutput>,
}

impl std::fmt::Debug for OcrProviderBatchSlotOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrProviderBatchSlotOutput")
            .field("slot_id", &self.slot_id)
            .field("stage_count", &self.stages.len())
            .field("has_output", &self.output.is_some())
            .finish()
    }
}

#[derive(Clone, Default)]
pub struct OcrProviderBatchOutput {
    pub slots: Vec<OcrProviderBatchSlotOutput>,
    pub execution_receipts: Vec<OcrExecutionReceipt>,
}

impl std::fmt::Debug for OcrProviderBatchOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrProviderBatchOutput")
            .field("slot_count", &self.slots.len())
            .field("execution_receipt_count", &self.execution_receipts.len())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrProviderFingerprint {
    pub id: String,
    pub engine: String,
    pub sends_source_off_device: bool,
    pub supported_stages: Vec<OcrStage>,
    pub declaration_sha256: String,
}

impl OcrProviderFingerprint {
    pub(crate) fn from_descriptor(descriptor: &OcrProviderDescriptor) -> UseResult<Self> {
        descriptor.validate()?;
        let supported_stages = descriptor.canonical_stages();
        let mut digest = Sha256::new();
        digest.update(b"a3s-ocr-provider-fingerprint-v1\0");
        update_text(&mut digest, &descriptor.id)?;
        update_text(&mut digest, &descriptor.engine)?;
        digest.update([u8::from(descriptor.sends_source_off_device)]);
        update_len(&mut digest, supported_stages.len())?;
        for stage in &supported_stages {
            digest.update([stage_tag(*stage)]);
        }
        Ok(Self {
            id: descriptor.id.clone(),
            engine: descriptor.engine.clone(),
            sends_source_off_device: descriptor.sends_source_off_device,
            supported_stages,
            declaration_sha256: format!("{:x}", digest.finalize()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrModelFingerprint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub execution_models: Vec<OcrExecutionModel>,
    pub declaration_sha256: String,
}

impl OcrModelFingerprint {
    pub(crate) fn from_output(output: &OcrProviderOutput) -> UseResult<Option<Self>> {
        let mut execution_models = output
            .execution_receipts
            .iter()
            .map(|receipt| receipt.model.clone())
            .collect::<Vec<_>>();
        execution_models.sort_by(|left, right| {
            (&left.family, &left.revision, &left.weights_sha256).cmp(&(
                &right.family,
                &right.revision,
                &right.weights_sha256,
            ))
        });
        execution_models.dedup();
        if output.model.is_none() && execution_models.is_empty() {
            return Ok(None);
        }
        let mut digest = Sha256::new();
        digest.update(b"a3s-ocr-model-fingerprint-v1\0");
        digest.update([u8::from(output.model.is_some())]);
        if let Some(model) = &output.model {
            update_text(&mut digest, model)?;
        }
        update_len(&mut digest, execution_models.len())?;
        for model in &execution_models {
            update_text(&mut digest, &model.family)?;
            update_text(&mut digest, &model.revision)?;
            update_text(&mut digest, &model.weights_sha256)?;
        }
        Ok(Some(Self {
            model: output.model.clone(),
            execution_models,
            declaration_sha256: format!("{:x}", digest.finalize()),
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum OcrBatchSlotStatus {
    Completed,
    Partial,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrBatchSlotResult {
    pub slot_id: OcrBatchSlotId,
    pub status: OcrBatchSlotStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Artifact>,
    pub stages: Vec<OcrStageOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_fingerprint: Option<OcrModelFingerprint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<OcrResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrBatchResult {
    pub schema: String,
    pub provider: OcrProviderFingerprint,
    pub requested_stages: Vec<OcrStage>,
    pub slots: Vec<OcrBatchSlotResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub execution_receipts: Vec<OcrExecutionReceipt>,
}

impl OcrBatchResult {
    pub const SCHEMA: &'static str = "a3s.ocr.staged-batch.v2";
}

pub(super) fn slot_status(stages: &[OcrStageOutcome]) -> OcrBatchSlotStatus {
    let completed = stages
        .iter()
        .filter(|outcome| outcome.status == OcrStageStatus::Completed)
        .count();
    if completed == stages.len() {
        OcrBatchSlotStatus::Completed
    } else if completed == 0 {
        OcrBatchSlotStatus::Failed
    } else {
        OcrBatchSlotStatus::Partial
    }
}

pub(super) fn batch_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.batch_invalid", message)
}

pub(super) fn provider_batch_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_batch_invalid", message)
}

fn update_text(digest: &mut Sha256, value: &str) -> UseResult<()> {
    update_len(digest, value.len())?;
    digest.update(value.as_bytes());
    Ok(())
}

fn update_len(digest: &mut Sha256, value: usize) -> UseResult<()> {
    let value = u64::try_from(value)
        .map_err(|_| batch_error("An OCR fingerprint length cannot be represented."))?;
    digest.update(value.to_le_bytes());
    Ok(())
}

const fn stage_tag(stage: OcrStage) -> u8 {
    match stage {
        OcrStage::Orientation => 0,
        OcrStage::Preprocessing => 1,
        OcrStage::Layout => 2,
        OcrStage::Text => 3,
        OcrStage::Table => 4,
        OcrStage::Formula => 5,
        OcrStage::Seal => 6,
    }
}
