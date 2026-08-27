mod client;
mod types;

pub use types::{
    OcrBatchRequest, OcrBatchResult, OcrBatchSlotId, OcrBatchSlotRequest, OcrBatchSlotResult,
    OcrBatchSlotStatus, OcrModelFingerprint, OcrNormalizedWindow, OcrProviderBatchOutput,
    OcrProviderBatchRequest, OcrProviderBatchSlot, OcrProviderBatchSlotOutput,
    OcrProviderFingerprint, OcrStage, OcrStageOutcome, OcrStageStatus,
    OCR_NORMALIZED_COORDINATE_BASIS,
};
pub(crate) use types::{OcrPixelWindow, MAX_BATCH_SLOTS};

#[cfg(test)]
mod tests;
