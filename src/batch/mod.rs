mod client;
mod types;

pub use types::{
    OcrBatchRequest, OcrBatchResult, OcrBatchSlotId, OcrBatchSlotRequest, OcrBatchSlotResult,
    OcrBatchSlotStatus, OcrModelFingerprint, OcrProviderBatchOutput, OcrProviderBatchRequest,
    OcrProviderBatchSlot, OcrProviderBatchSlotOutput, OcrProviderFingerprint, OcrStage,
    OcrStageOutcome, OcrStageStatus,
};

#[cfg(test)]
mod tests;
