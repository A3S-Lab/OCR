use std::sync::Arc;

use a3s_use_core::{Artifact, Readiness, UseError, UseResult};
use async_trait::async_trait;

use crate::{
    OcrBlock, OcrExecutionReceipt, OcrProviderBatchOutput, OcrProviderBatchRequest,
    OcrProviderBatchSlotOutput, OcrStage, OcrStageOutcome,
};

/// Stable identity and execution characteristics of an OCR provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrProviderDescriptor {
    pub id: String,
    pub engine: String,
    pub sends_source_off_device: bool,
    pub supported_stages: Vec<OcrStage>,
}

impl OcrProviderDescriptor {
    pub fn new(
        id: impl Into<String>,
        engine: impl Into<String>,
        sends_source_off_device: bool,
    ) -> UseResult<Self> {
        let descriptor = Self {
            id: id.into(),
            engine: engine.into(),
            sends_source_off_device,
            supported_stages: vec![OcrStage::Text],
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn validate(&self) -> UseResult<()> {
        if self.id.is_empty()
            || self.id.len() > 64
            || !self
                .id
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            || !self.id.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'.' | b'-' | b'_')
            })
        {
            return Err(UseError::new(
                "use.ocr.provider_descriptor_invalid",
                "OCR provider IDs must contain 1 through 64 lowercase ASCII letters, digits, dots, hyphens, or underscores and start with a letter or digit.",
            ));
        }
        if self.engine.trim().is_empty() || self.engine.len() > 128 {
            return Err(UseError::new(
                "use.ocr.provider_descriptor_invalid",
                "OCR provider engine names must contain 1 through 128 characters.",
            ));
        }
        if self.supported_stages.is_empty() {
            return Err(UseError::new(
                "use.ocr.provider_descriptor_invalid",
                "OCR providers must declare at least one supported stage.",
            ));
        }
        let mut stages = self.supported_stages.clone();
        stages.sort_unstable();
        stages.dedup();
        if stages != self.supported_stages {
            return Err(UseError::new(
                "use.ocr.provider_descriptor_invalid",
                "OCR provider stages must be unique and use canonical order.",
            ));
        }
        Ok(())
    }

    pub fn with_stages(mut self, mut stages: Vec<OcrStage>) -> UseResult<Self> {
        stages.sort_unstable();
        stages.dedup();
        self.supported_stages = stages;
        self.validate()?;
        Ok(self)
    }

    pub fn supports_stage(&self, stage: OcrStage) -> bool {
        self.supported_stages.binary_search(&stage).is_ok()
    }

    pub(crate) fn canonical_stages(&self) -> Vec<OcrStage> {
        self.supported_stages.clone()
    }
}

/// Provider-specific readiness projected through the common OCR interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OcrProviderStatus {
    pub readiness: Readiness,
    pub model: Option<String>,
    pub model_dir: Option<std::path::PathBuf>,
    pub message: String,
    pub suggestions: Vec<String>,
}

/// One validated image supplied to an OCR provider.
#[derive(Clone)]
pub struct OcrInput {
    source: Artifact,
    bytes: Arc<[u8]>,
}

impl std::fmt::Debug for OcrInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrInput")
            .field("source", &"validated-image")
            .field("byte_length", &self.bytes.len())
            .finish()
    }
}

impl OcrInput {
    pub(crate) fn new(source: Artifact, bytes: Vec<u8>) -> Self {
        Self {
            source,
            bytes: bytes.into(),
        }
    }

    pub fn source(&self) -> &Artifact {
        &self.source
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Provider-owned OCR content before the client adds source provenance.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OcrProviderOutput {
    pub model: Option<String>,
    pub text: String,
    pub blocks: Vec<OcrBlock>,
    pub execution_receipts: Vec<OcrExecutionReceipt>,
    pub warnings: Vec<String>,
}

/// Pluggable OCR implementation.
///
/// `OcrClient` validates and hashes the bounded local image before invoking
/// this interface. Providers only own recognition and provider-specific
/// readiness; they cannot replace the canonical source evidence.
#[async_trait]
pub trait OcrProvider: Send + Sync {
    fn descriptor(&self) -> OcrProviderDescriptor;

    fn diagnostic(&self) -> OcrProviderStatus;

    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput>;

    /// Default compatibility adapter for text-only providers.
    ///
    /// Providers with typed preprocessing, layout, table, or formula stages
    /// should override this method and return one exact slot outcome per input.
    async fn recognize_batch(
        &self,
        request: OcrProviderBatchRequest,
    ) -> UseResult<OcrProviderBatchOutput> {
        let mut slots = Vec::with_capacity(request.slots.len());
        for slot in request.slots {
            let recognition = if request.stages.contains(&OcrStage::Text) {
                Some(self.recognize(slot.input).await)
            } else {
                None
            };
            let stages = request
                .stages
                .iter()
                .map(|stage| match (*stage, &recognition) {
                    (OcrStage::Text, Some(Ok(_))) => OcrStageOutcome::completed(*stage),
                    (OcrStage::Text, Some(Err(error))) => {
                        OcrStageOutcome::failed(*stage, error.clone())
                    }
                    _ => OcrStageOutcome::unsupported(*stage),
                })
                .collect();
            slots.push(OcrProviderBatchSlotOutput {
                slot_id: slot.slot_id,
                stages,
                output: recognition.and_then(Result::ok),
            });
        }
        Ok(OcrProviderBatchOutput {
            slots,
            execution_receipts: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_ids_are_stable_machine_names() {
        assert!(OcrProviderDescriptor::new("tesseract", "tesseract", false).is_ok());
        assert!(OcrProviderDescriptor::new("Azure OCR", "azure", true).is_err());
        assert!(OcrProviderDescriptor::new("-leading", "fixture", false).is_err());
    }
}
