use std::path::Path;

use a3s_use_core::{UseError, UseResult};

pub(super) const ENCODER_STEPS: usize = 256;
pub(super) const CONTEXT_WIDTH: usize = 96;
pub(super) const HIDDEN_WIDTH: usize = 256;
pub(super) const VOCABULARY_SIZE: usize = 50;
pub(super) const LOCATION_WIDTH: usize = 8;
pub(super) const MAX_TOKENS: usize = 501;

const EXPECTED_FLOATS: usize = 547_386;

#[derive(Debug)]
pub(super) struct DecoderWeights {
    pub(super) attention_input: Vec<f32>,
    pub(super) attention_hidden: Vec<f32>,
    pub(super) attention_hidden_bias: Vec<f32>,
    pub(super) attention_score: Vec<f32>,
    pub(super) structure_hidden: Vec<f32>,
    pub(super) structure_hidden_bias: Vec<f32>,
    pub(super) structure_output: Vec<f32>,
    pub(super) structure_output_bias: Vec<f32>,
    pub(super) location_hidden: Vec<f32>,
    pub(super) location_hidden_bias: Vec<f32>,
    pub(super) location_output: Vec<f32>,
    pub(super) location_output_bias: Vec<f32>,
    pub(super) gru_input: Vec<f32>,
    pub(super) gru_hidden: Vec<f32>,
    pub(super) gru_input_bias: Vec<f32>,
    pub(super) gru_hidden_bias: Vec<f32>,
}

impl DecoderWeights {
    pub(super) fn load(path: &Path) -> UseResult<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            weights_error(format!(
                "Failed to read the SLANet-Plus decoder weights '{}': {error}",
                path.display()
            ))
        })?;
        Self::from_bytes(&bytes)
    }

    fn from_bytes(bytes: &[u8]) -> UseResult<Self> {
        if bytes.len() != EXPECTED_FLOATS * size_of::<f32>() {
            return Err(weights_error(format!(
                "The SLANet-Plus decoder must contain exactly {EXPECTED_FLOATS} little-endian float32 values."
            )));
        }
        let mut cursor = FloatCursor::new(bytes);
        let gate_width = 3 * HIDDEN_WIDTH;
        let gru_input_width = CONTEXT_WIDTH + VOCABULARY_SIZE;
        let weights = Self {
            attention_input: cursor.take(CONTEXT_WIDTH * HIDDEN_WIDTH)?,
            attention_hidden: cursor.take(HIDDEN_WIDTH * HIDDEN_WIDTH)?,
            attention_hidden_bias: cursor.take(HIDDEN_WIDTH)?,
            attention_score: cursor.take(HIDDEN_WIDTH)?,
            structure_hidden: cursor.take(HIDDEN_WIDTH * HIDDEN_WIDTH)?,
            structure_hidden_bias: cursor.take(HIDDEN_WIDTH)?,
            structure_output: cursor.take(HIDDEN_WIDTH * VOCABULARY_SIZE)?,
            structure_output_bias: cursor.take(VOCABULARY_SIZE)?,
            location_hidden: cursor.take(HIDDEN_WIDTH * HIDDEN_WIDTH)?,
            location_hidden_bias: cursor.take(HIDDEN_WIDTH)?,
            location_output: cursor.take(HIDDEN_WIDTH * LOCATION_WIDTH)?,
            location_output_bias: cursor.take(LOCATION_WIDTH)?,
            gru_input: cursor.take(gate_width * gru_input_width)?,
            gru_hidden: cursor.take(gate_width * HIDDEN_WIDTH)?,
            gru_input_bias: cursor.take(gate_width)?,
            gru_hidden_bias: cursor.take(gate_width)?,
        };
        if cursor.remaining() != 0 {
            return Err(weights_error(
                "The SLANet-Plus decoder contains trailing float32 values.",
            ));
        }
        Ok(weights)
    }
}

struct FloatCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> FloatCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> UseResult<Vec<f32>> {
        let byte_count = count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| weights_error("SLANet-Plus decoder tensor length overflowed."))?;
        let end = self
            .offset
            .checked_add(byte_count)
            .ok_or_else(|| weights_error("SLANet-Plus decoder offset overflowed."))?;
        let source = self.bytes.get(self.offset..end).ok_or_else(|| {
            weights_error("The SLANet-Plus decoder ended inside a declared tensor.")
        })?;
        let mut values = Vec::with_capacity(count);
        for chunk in source.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        self.offset = end;
        Ok(values)
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

fn weights_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.table_model_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_blob_layout_has_the_reviewed_total() {
        let gate_width = 3 * HIDDEN_WIDTH;
        let gru_input_width = CONTEXT_WIDTH + VOCABULARY_SIZE;
        let declared = CONTEXT_WIDTH * HIDDEN_WIDTH
            + HIDDEN_WIDTH * HIDDEN_WIDTH
            + HIDDEN_WIDTH
            + HIDDEN_WIDTH
            + HIDDEN_WIDTH * HIDDEN_WIDTH
            + HIDDEN_WIDTH
            + HIDDEN_WIDTH * VOCABULARY_SIZE
            + VOCABULARY_SIZE
            + HIDDEN_WIDTH * HIDDEN_WIDTH
            + HIDDEN_WIDTH
            + HIDDEN_WIDTH * LOCATION_WIDTH
            + LOCATION_WIDTH
            + gate_width * gru_input_width
            + gate_width * HIDDEN_WIDTH
            + gate_width
            + gate_width;
        assert_eq!(declared, EXPECTED_FLOATS);
    }

    #[test]
    fn truncated_decoder_is_rejected_before_tensor_access() {
        let error = DecoderWeights::from_bytes(&[0; 16]).unwrap_err();
        assert_eq!(error.code, "use.ocr.table_model_invalid");
    }
}
