mod dictionary;
mod grid;
mod weights;

use a3s_use_core::{UseError, UseResult};
use tokio_util::sync::CancellationToken;

use self::dictionary::StructureDictionary;
pub(super) use self::grid::StructureGrid;
use self::weights::{
    DecoderWeights, CONTEXT_WIDTH, ENCODER_STEPS, HIDDEN_WIDTH, LOCATION_WIDTH, MAX_TOKENS,
    VOCABULARY_SIZE,
};
use super::orientation::TableCropOrientation;
use super::wired::PixelRect;

const MAX_REPEATED_TOKEN_RUN: usize = 96;

pub(super) struct SlanetPlusDecoder {
    weights: DecoderWeights,
    dictionary: StructureDictionary,
}

impl SlanetPlusDecoder {
    pub(super) fn load(
        decoder_path: &std::path::Path,
        dictionary_path: &std::path::Path,
    ) -> UseResult<Self> {
        Ok(Self {
            weights: DecoderWeights::load(decoder_path)?,
            dictionary: StructureDictionary::load(dictionary_path)?,
        })
    }

    pub(super) fn decode(
        &self,
        features: &[f32],
        crop: PixelRect,
        orientation: TableCropOrientation,
        cancellation: &CancellationToken,
    ) -> UseResult<DecodedStructure> {
        if features.len() != ENCODER_STEPS * CONTEXT_WIDTH
            || features.iter().any(|value| !value.is_finite())
        {
            return Err(output_error(
                "SLANet-Plus encoder output must be finite [256,96] float32 features.",
            ));
        }
        if crop.width == 0 || crop.height == 0 {
            return Err(output_error(
                "SLANet-Plus decoding requires a positive source crop.",
            ));
        }

        let mut scratch = DecodeScratch::new();
        project_encoder_features(
            features,
            &self.weights.attention_input,
            &mut scratch.attention_keys,
        );
        let mut previous = self.dictionary.sos();
        let mut repeated_token = usize::MAX;
        let mut repeated_count = 0_usize;
        let mut tokens = Vec::new();
        let mut cells = Vec::new();
        let mut confidence_sum = 0.0_f32;

        for step in 0..MAX_TOKENS {
            if step % 8 == 0 && cancellation.is_cancelled() {
                return Err(UseError::new(
                    "use.ocr.cancelled",
                    "SLANet-Plus structure decoding was cancelled.",
                ));
            }
            matrix_vector_input_output(
                &scratch.hidden,
                &self.weights.attention_hidden,
                HIDDEN_WIDTH,
                HIDDEN_WIDTH,
                Some(&self.weights.attention_hidden_bias),
                &mut scratch.attention_query,
            );
            attention_context(
                features,
                &scratch.attention_keys,
                &scratch.attention_query,
                &self.weights.attention_score,
                &mut scratch.attention_energy,
                &mut scratch.context,
            );
            gru_step(
                &self.weights,
                previous,
                &scratch.context,
                &mut scratch.hidden,
                &mut scratch.gru_input,
                &mut scratch.gru_hidden,
            );
            matrix_vector_input_output(
                &scratch.hidden,
                &self.weights.structure_hidden,
                HIDDEN_WIDTH,
                HIDDEN_WIDTH,
                Some(&self.weights.structure_hidden_bias),
                &mut scratch.structure_hidden,
            );
            matrix_vector_input_output(
                &scratch.structure_hidden,
                &self.weights.structure_output,
                HIDDEN_WIDTH,
                VOCABULARY_SIZE,
                Some(&self.weights.structure_output_bias),
                &mut scratch.structure_logits,
            );
            let (best, confidence) = top_probability(&scratch.structure_logits)?;
            if best == self.dictionary.eos() {
                break;
            }
            if best == repeated_token {
                repeated_count += 1;
                if repeated_count >= MAX_REPEATED_TOKEN_RUN {
                    break;
                }
            } else {
                repeated_token = best;
                repeated_count = 1;
            }

            if best != self.dictionary.sos() {
                let token_position = tokens.len();
                tokens.push(self.dictionary.token(best)?.to_string());
                confidence_sum += confidence;
                if self.dictionary.is_cell(best) {
                    matrix_vector_input_output(
                        &scratch.hidden,
                        &self.weights.location_hidden,
                        HIDDEN_WIDTH,
                        HIDDEN_WIDTH,
                        Some(&self.weights.location_hidden_bias),
                        &mut scratch.location_hidden,
                    );
                    matrix_vector_input_output(
                        &scratch.location_hidden,
                        &self.weights.location_output,
                        HIDDEN_WIDTH,
                        LOCATION_WIDTH,
                        Some(&self.weights.location_output_bias),
                        &mut scratch.location_logits,
                    );
                    cells.push(DecodedCell {
                        token_position,
                        quad: project_quad(&scratch.location_logits, crop, orientation),
                    });
                }
            }
            previous = best;
        }
        let confidence = if tokens.is_empty() {
            0.0
        } else {
            confidence_sum / tokens.len() as f32
        };
        let decoded = DecodedStructure {
            tokens,
            cells,
            confidence,
        };
        decoded.validate()?;
        Ok(decoded)
    }
}

pub(super) struct DecodedStructure {
    pub(super) tokens: Vec<String>,
    pub(super) cells: Vec<DecodedCell>,
    pub(super) confidence: f32,
}

impl DecodedStructure {
    pub(super) fn into_grid(self) -> UseResult<StructureGrid> {
        grid::project(self)
    }

    fn validate(&self) -> UseResult<()> {
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(output_error(
                "SLANet-Plus structure confidence must be finite between zero and one.",
            ));
        }
        if self.cells.len() > self.tokens.len()
            || self
                .cells
                .iter()
                .any(|cell| cell.token_position >= self.tokens.len())
        {
            return Err(output_error(
                "SLANet-Plus cell geometry lost its structure-token identity.",
            ));
        }
        Ok(())
    }
}

pub(super) struct DecodedCell {
    pub(super) token_position: usize,
    pub(super) quad: Option<[u32; 8]>,
}

struct DecodeScratch {
    attention_keys: Vec<f32>,
    hidden: Vec<f32>,
    attention_query: Vec<f32>,
    attention_energy: Vec<f32>,
    context: Vec<f32>,
    gru_input: Vec<f32>,
    gru_hidden: Vec<f32>,
    structure_hidden: Vec<f32>,
    structure_logits: Vec<f32>,
    location_hidden: Vec<f32>,
    location_logits: Vec<f32>,
}

impl DecodeScratch {
    fn new() -> Self {
        Self {
            attention_keys: vec![0.0; ENCODER_STEPS * HIDDEN_WIDTH],
            hidden: vec![0.0; HIDDEN_WIDTH],
            attention_query: vec![0.0; HIDDEN_WIDTH],
            attention_energy: vec![0.0; ENCODER_STEPS],
            context: vec![0.0; CONTEXT_WIDTH],
            gru_input: vec![0.0; 3 * HIDDEN_WIDTH],
            gru_hidden: vec![0.0; 3 * HIDDEN_WIDTH],
            structure_hidden: vec![0.0; HIDDEN_WIDTH],
            structure_logits: vec![0.0; VOCABULARY_SIZE],
            location_hidden: vec![0.0; HIDDEN_WIDTH],
            location_logits: vec![0.0; LOCATION_WIDTH],
        }
    }
}

fn project_encoder_features(features: &[f32], weights: &[f32], output: &mut [f32]) {
    for step in 0..ENCODER_STEPS {
        matrix_vector_input_output(
            &features[step * CONTEXT_WIDTH..(step + 1) * CONTEXT_WIDTH],
            weights,
            CONTEXT_WIDTH,
            HIDDEN_WIDTH,
            None,
            &mut output[step * HIDDEN_WIDTH..(step + 1) * HIDDEN_WIDTH],
        );
    }
}

fn attention_context(
    features: &[f32],
    keys: &[f32],
    query: &[f32],
    score_weights: &[f32],
    energy: &mut [f32],
    context: &mut [f32],
) {
    for step in 0..ENCODER_STEPS {
        let key = &keys[step * HIDDEN_WIDTH..(step + 1) * HIDDEN_WIDTH];
        energy[step] = key
            .iter()
            .zip(query)
            .zip(score_weights)
            .map(|((key, query), weight)| (key + query).tanh() * weight)
            .sum();
    }
    let maximum = energy.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in energy.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    context.fill(0.0);
    if !sum.is_finite() || sum <= 0.0 {
        context.fill(f32::NAN);
        return;
    }
    for step in 0..ENCODER_STEPS {
        let probability = energy[step] / sum;
        let feature = &features[step * CONTEXT_WIDTH..(step + 1) * CONTEXT_WIDTH];
        for channel in 0..CONTEXT_WIDTH {
            context[channel] += probability * feature[channel];
        }
    }
}

fn gru_step(
    weights: &DecoderWeights,
    previous: usize,
    context: &[f32],
    hidden: &mut [f32],
    input_gates: &mut [f32],
    hidden_gates: &mut [f32],
) {
    let input_width = CONTEXT_WIDTH + VOCABULARY_SIZE;
    for (gate, input_gate) in input_gates.iter_mut().enumerate() {
        let row = &weights.gru_input[gate * input_width..(gate + 1) * input_width];
        let mut value = weights.gru_input_bias[gate];
        for channel in 0..CONTEXT_WIDTH {
            value += context[channel] * row[channel];
        }
        *input_gate = value + row[CONTEXT_WIDTH + previous];
    }
    matrix_vector_output_input(
        hidden,
        &weights.gru_hidden,
        3 * HIDDEN_WIDTH,
        HIDDEN_WIDTH,
        Some(&weights.gru_hidden_bias),
        hidden_gates,
    );
    for index in 0..HIDDEN_WIDTH {
        let reset = sigmoid(input_gates[index] + hidden_gates[index]);
        let update =
            sigmoid(input_gates[HIDDEN_WIDTH + index] + hidden_gates[HIDDEN_WIDTH + index]);
        let candidate = (input_gates[2 * HIDDEN_WIDTH + index]
            + reset * hidden_gates[2 * HIDDEN_WIDTH + index])
            .tanh();
        hidden[index] = (1.0 - update) * candidate + update * hidden[index];
    }
}

fn matrix_vector_input_output(
    input: &[f32],
    weights: &[f32],
    input_width: usize,
    output_width: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    if let Some(bias) = bias {
        output.copy_from_slice(bias);
    } else {
        output.fill(0.0);
    }
    for input_index in 0..input_width {
        let value = input[input_index];
        let row = &weights[input_index * output_width..(input_index + 1) * output_width];
        for output_index in 0..output_width {
            output[output_index] += value * row[output_index];
        }
    }
}

fn matrix_vector_output_input(
    input: &[f32],
    weights: &[f32],
    output_width: usize,
    input_width: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    for output_index in 0..output_width {
        let row = &weights[output_index * input_width..(output_index + 1) * input_width];
        let mut value = bias.map_or(0.0, |bias| bias[output_index]);
        for input_index in 0..input_width {
            value += input[input_index] * row[input_index];
        }
        output[output_index] = value;
    }
}

fn top_probability(logits: &[f32]) -> UseResult<(usize, f32)> {
    if logits.len() != VOCABULARY_SIZE || logits.iter().any(|value| !value.is_finite()) {
        return Err(output_error(
            "SLANet-Plus structure logits must be a finite 50-value vector.",
        ));
    }
    let mut best = 0_usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    let maximum = logits[best];
    let denominator = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f32>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(output_error(
            "SLANet-Plus structure softmax was not finite.",
        ));
    }
    Ok((best, 1.0 / denominator))
}

fn project_quad(
    logits: &[f32],
    crop: PixelRect,
    orientation: TableCropOrientation,
) -> Option<[u32; 8]> {
    let (oriented_width, oriented_height) = orientation.oriented_dimensions(crop);
    let scale = oriented_width.max(oriented_height) as f32;
    let mut quad = [0_u32; 8];
    for (point_index, point_logits) in logits.chunks_exact(2).enumerate() {
        let oriented_x = (sigmoid(point_logits[0]) * scale)
            .trunc()
            .clamp(0.0, oriented_width as f32) as u32;
        let oriented_y = (sigmoid(point_logits[1]) * scale)
            .trunc()
            .clamp(0.0, oriented_height as f32) as u32;
        let (source_x, source_y) = orientation.source_boundary_point(crop, oriented_x, oriented_y);
        quad[point_index * 2] = source_x;
        quad[point_index * 2 + 1] = source_y;
    }
    let left = quad.iter().step_by(2).copied().min()?;
    let right = quad.iter().step_by(2).copied().max()?;
    let top = quad.iter().skip(1).step_by(2).copied().min()?;
    let bottom = quad.iter().skip(1).step_by(2).copied().max()?;
    (right > left && bottom > top).then_some(quad)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.table_model_output_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_probability_preserves_first_argmax_tie() {
        let mut logits = vec![-3.0; VOCABULARY_SIZE];
        logits[4] = 2.0;
        logits[7] = 2.0;
        let (index, confidence) = top_probability(&logits).unwrap();
        assert_eq!(index, 4);
        assert!(confidence > 0.4 && confidence < 0.5);
    }

    #[test]
    fn quad_projection_clamps_padding_to_the_exact_crop() {
        let projected = project_quad(
            &[100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0, 100.0],
            PixelRect {
                x: 10,
                y: 20,
                width: 200,
                height: 100,
            },
            TableCropOrientation::Upright,
        )
        .unwrap();
        assert_eq!(projected, [210, 120, 210, 20, 10, 20, 10, 120]);
    }

    #[test]
    fn rotated_quad_is_mapped_back_to_the_exact_source_crop() {
        let projected = project_quad(
            &[100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0, 100.0],
            PixelRect {
                x: 10,
                y: 20,
                width: 200,
                height: 100,
            },
            TableCropOrientation::Rotate90,
        )
        .unwrap();
        assert_eq!(projected, [210, 20, 10, 20, 10, 120, 210, 120]);
    }
}
