mod decoder;
mod ops;
mod vision;
mod weights;

pub(crate) use decoder::Decoder;
pub(crate) use vision::VisionEncoder;
pub(crate) use weights::{power_error, shared, ModelWeights};

pub(crate) const HIDDEN_SIZE: usize = 1_280;
pub(crate) const VOCAB_SIZE: usize = 129_280;
pub(crate) const DECODER_LAYERS: usize = 12;
pub(crate) const ATTENTION_HEADS: usize = 10;
pub(crate) const HEAD_DIM: usize = HIDDEN_SIZE / ATTENTION_HEADS;
pub(crate) const ROUTED_EXPERTS: usize = 64;
pub(crate) const EXPERTS_PER_TOKEN: usize = 6;
pub(crate) const EXPERT_INTERMEDIATE_SIZE: usize = 896;
pub(crate) const SHARED_EXPERT_INTERMEDIATE_SIZE: usize = 1_792;
pub(crate) const DENSE_INTERMEDIATE_SIZE: usize = 6_848;
pub(crate) const SLIDING_WINDOW: usize = 128;
pub(crate) const RMS_NORM_EPS: f64 = 1e-6;
pub(crate) const ROPE_THETA: f64 = 10_000.0;
