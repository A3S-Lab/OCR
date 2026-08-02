use std::sync::Arc;

use a3s_power::inference::{ExecutionPermit, RoutedExpert, RoutedExpertBatch};
use a3s_use_core::{UseError, UseResult};
use candle_core::{DType, IndexOp, Tensor, D};
use tokio_util::sync::CancellationToken;

use super::ops::linear;
use super::weights::{expert_name, model_error, power_error};
use super::{
    ModelWeights, ATTENTION_HEADS, DECODER_LAYERS, DENSE_INTERMEDIATE_SIZE, EXPERTS_PER_TOKEN,
    EXPERT_INTERMEDIATE_SIZE, HEAD_DIM, HIDDEN_SIZE, RMS_NORM_EPS, ROPE_THETA, ROUTED_EXPERTS,
    SHARED_EXPERT_INTERMEDIATE_SIZE, SLIDING_WINDOW, VOCAB_SIZE,
};
use crate::cancellation::check_cancelled;
use crate::unlimited_ocr::ngram::{greedy_token, SlidingNoRepeatNgram};
use crate::unlimited_ocr::tokenizer::{PromptEncoding, EOS_TOKEN_ID};

pub(crate) struct Decoder {
    weights: Arc<ModelWeights>,
}

impl Decoder {
    pub(crate) fn new(weights: Arc<ModelWeights>) -> Self {
        Self { weights }
    }

    pub(crate) fn generate(
        &self,
        prompt: &PromptEncoding,
        vision: &Tensor,
        max_generated_tokens: usize,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Vec<u32>> {
        check_cancelled(cancellation)?;
        let prompt_len = prompt.token_ids.len();
        if prompt_len == 0 {
            return Err(model_error("Unlimited-OCR prompt is empty."));
        }
        let mut hidden = self.embed(&prompt.token_ids, permit, cancellation)?;
        let cached_tokens = prompt_len
            .checked_add(max_generated_tokens.min(SLIDING_WINDOW))
            .ok_or_else(|| model_error("Unlimited-OCR KV-cache token count overflowed."))?;
        let state_bytes = DECODER_LAYERS
            .checked_mul(2)
            .and_then(|value| value.checked_mul(ATTENTION_HEADS))
            .and_then(|value| value.checked_mul(cached_tokens))
            .and_then(|value| value.checked_mul(HEAD_DIM))
            .and_then(|value| value.checked_mul(hidden.dtype().size_in_bytes()))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| model_error("Unlimited-OCR KV-cache byte count overflowed."))?;
        self.weights
            .hierarchy()
            .runtime()
            .limits()
            .checked_state_bytes(state_bytes, "Unlimited-OCR decoder KV cache")
            .map_err(|error| power_error("validate decoder state bounds", error))?;
        let vision_tokens = vision
            .dim(0)
            .map_err(tensor_error("inspect vision features"))?;
        if vision_tokens != prompt.image_tokens.len() {
            return Err(model_error(format!(
                "Unlimited-OCR produced {vision_tokens} vision rows for {} image placeholders.",
                prompt.image_tokens.len()
            )));
        }
        hidden = replace_image_rows(&hidden, vision, prompt.image_tokens.clone())?
            .unsqueeze(0)
            .map_err(tensor_error("batch the fused prompt"))?;

        let mut state = DecoderState::new();
        let mut logits = self.forward(&hidden, 0, &mut state, permit, cancellation)?;
        let mut generated = Vec::with_capacity(max_generated_tokens.min(4_096));
        let mut sequence = prompt.token_ids.clone();
        let no_repeat = SlidingNoRepeatNgram::reviewed();
        for _ in 0..max_generated_tokens {
            check_cancelled(cancellation)?;
            let mut values = logits
                .to_vec1::<f32>()
                .map_err(tensor_error("read decoder token logits"))?;
            if values.len() != VOCAB_SIZE {
                return Err(model_error(format!(
                    "Unlimited-OCR decoder returned {} logits instead of {VOCAB_SIZE}.",
                    values.len()
                )));
            }
            no_repeat.ban_in_place(&sequence, &mut values)?;
            let token = greedy_token(&values)?;
            generated.push(token);
            sequence.push(token);
            if token == EOS_TOKEN_ID {
                break;
            }
            let position = prompt_len.saturating_add(generated.len()).saturating_sub(1);
            let next = self
                .embed(&[token], permit, cancellation)?
                .unsqueeze(0)
                .map_err(tensor_error("batch a decoder token"))?;
            logits = self.forward(&next, position, &mut state, permit, cancellation)?;
        }
        Ok(generated)
    }

    fn embed(
        &self,
        token_ids: &[u32],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        if token_ids
            .iter()
            .any(|token| (*token as usize) >= VOCAB_SIZE)
        {
            return Err(model_error(
                "Unlimited-OCR prompt contains a token outside the reviewed vocabulary.",
            ));
        }
        self.weights.checked_elements(
            &[token_ids.len(), HIDDEN_SIZE],
            "Unlimited-OCR decoder embeddings",
        )?;
        let table = self
            .weights
            .load_global("model.embed_tokens.weight", permit, cancellation)?;
        let indices = Tensor::from_vec(token_ids.to_vec(), token_ids.len(), table.device())
            .map_err(tensor_error("materialize token indices"))?;
        table
            .index_select(&indices, 0)
            .map_err(tensor_error("look up token embeddings"))
    }

    fn forward(
        &self,
        hidden: &Tensor,
        position: usize,
        state: &mut DecoderState,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let (batch, query_len, width) = hidden
            .dims3()
            .map_err(tensor_error("inspect decoder input"))?;
        if batch != 1 || width != HIDDEN_SIZE {
            return Err(model_error(format!(
                "Unlimited-OCR decoder input must be [1, N, {HIDDEN_SIZE}]."
            )));
        }
        let mut hidden = hidden.clone();
        for layer in 0..DECODER_LAYERS {
            check_cancelled(cancellation)?;
            hidden = self.layer(
                layer as u32,
                &hidden,
                position,
                &mut state.layers[layer],
                permit,
                cancellation,
            )?;
        }
        let norm = self
            .weights
            .load_global("model.norm.weight", permit, cancellation)?;
        let hidden = rms_norm(&hidden, norm)?;
        let last = hidden
            .i((0, query_len - 1, ..))
            .map_err(tensor_error("select the final decoder row"))?;
        let head = self
            .weights
            .load_global("lm_head.weight", permit, cancellation)?;
        linear(&last, &head, None)?
            .to_dtype(DType::F32)
            .map_err(tensor_error("convert decoder logits to f32"))
    }

    fn layer(
        &self,
        layer: u32,
        hidden: &Tensor,
        position: usize,
        cache: &mut Option<KvCache>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let prefix = format!("model.layers.{layer}");
        let input_norm = self.weights.load(
            layer,
            &format!("{prefix}.input_layernorm.weight"),
            permit,
            cancellation,
        )?;
        let normalized = rms_norm(hidden, input_norm)?;
        let attention = self.attention(
            layer,
            &prefix,
            &normalized,
            position,
            cache,
            permit,
            cancellation,
        )?;
        let hidden = (hidden + attention).map_err(tensor_error("add the attention residual"))?;
        let post_norm = self.weights.load(
            layer,
            &format!("{prefix}.post_attention_layernorm.weight"),
            permit,
            cancellation,
        )?;
        let normalized = rms_norm(&hidden, post_norm)?;
        let feed_forward = if layer == 0 {
            self.dense_mlp(
                layer,
                &format!("{prefix}.mlp"),
                &normalized,
                DENSE_INTERMEDIATE_SIZE,
                permit,
                cancellation,
            )?
        } else {
            self.moe(layer, &normalized, permit, cancellation)?
        };
        (&hidden + feed_forward).map_err(tensor_error("add the feed-forward residual"))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: u32,
        prefix: &str,
        hidden: &Tensor,
        position: usize,
        cache: &mut Option<KvCache>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let q = self.weights.load(
            layer,
            &format!("{prefix}.self_attn.q_proj.weight"),
            permit,
            cancellation,
        )?;
        let k = self.weights.load(
            layer,
            &format!("{prefix}.self_attn.k_proj.weight"),
            permit,
            cancellation,
        )?;
        let v = self.weights.load(
            layer,
            &format!("{prefix}.self_attn.v_proj.weight"),
            permit,
            cancellation,
        )?;
        let o = self.weights.load(
            layer,
            &format!("{prefix}.self_attn.o_proj.weight"),
            permit,
            cancellation,
        )?;
        let (batch, query_len, _) = hidden
            .dims3()
            .map_err(tensor_error("inspect normalized attention input"))?;
        let shape = (batch, query_len, ATTENTION_HEADS, HEAD_DIM);
        let q = linear(hidden, &q, None)?
            .reshape(shape)
            .and_then(|value| value.transpose(1, 2))
            .map_err(tensor_error("shape attention queries"))?;
        let k = linear(hidden, &k, None)?
            .reshape(shape)
            .and_then(|value| value.transpose(1, 2))
            .map_err(tensor_error("shape attention keys"))?;
        let v = linear(hidden, &v, None)?
            .reshape(shape)
            .and_then(|value| value.transpose(1, 2))
            .map_err(tensor_error("shape attention values"))?;
        let (q, k) = apply_rope(&q, &k, position)?;
        let (keys, values) = update_cache(cache, k, v, query_len > 1)?;
        let key_len = keys
            .dim(2)
            .map_err(tensor_error("inspect cached attention keys"))?;
        self.weights.checked_elements(
            &[batch, ATTENTION_HEADS, query_len, key_len],
            "Unlimited-OCR decoder attention scores",
        )?;
        let scores = q
            .contiguous()
            .and_then(|q| {
                keys.transpose(2, 3)
                    .and_then(|k| k.contiguous())
                    .and_then(|k| q.matmul(&k))
            })
            .map_err(tensor_error("compute attention scores"))?;
        let scores =
            (scores / (HEAD_DIM as f64).sqrt()).map_err(tensor_error("scale attention scores"))?;
        let scores = if query_len > 1 {
            let mask = causal_mask(query_len, scores.dtype(), scores.device())?;
            scores
                .broadcast_add(&mask)
                .map_err(tensor_error("apply the causal attention mask"))?
        } else {
            scores
        };
        let probabilities = candle_nn::ops::softmax_last_dim(
            &scores
                .to_dtype(DType::F32)
                .map_err(tensor_error("stabilize attention scores"))?,
        )
        .and_then(|value| value.to_dtype(hidden.dtype()))
        .map_err(tensor_error("normalize attention scores"))?;
        let context = probabilities
            .matmul(
                &values
                    .contiguous()
                    .map_err(tensor_error("pack cached values"))?,
            )
            .and_then(|value| value.transpose(1, 2))
            .and_then(|value| value.reshape((batch, query_len, HIDDEN_SIZE)))
            .map_err(tensor_error("merge attention heads"))?;
        linear(&context, &o, None)
    }

    fn dense_mlp(
        &self,
        layer: u32,
        prefix: &str,
        hidden: &Tensor,
        intermediate: usize,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let gate = self.weights.load(
            layer,
            &format!("{prefix}.gate_proj.weight"),
            permit,
            cancellation,
        )?;
        let up = self.weights.load(
            layer,
            &format!("{prefix}.up_proj.weight"),
            permit,
            cancellation,
        )?;
        let down = self.weights.load(
            layer,
            &format!("{prefix}.down_proj.weight"),
            permit,
            cancellation,
        )?;
        if gate.dim(0).ok() != Some(intermediate) {
            return Err(model_error(format!(
                "Unlimited-OCR layer {layer} has an invalid SwiGLU width."
            )));
        }
        swiglu(hidden, &gate, &up, &down)
    }

    fn moe(
        &self,
        layer: u32,
        hidden: &Tensor,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let gate = self.weights.load(
            layer,
            &format!("model.layers.{layer}.mlp.gate.weight"),
            permit,
            cancellation,
        )?;
        let positions = hidden
            .elem_count()
            .checked_div(HIDDEN_SIZE)
            .ok_or_else(|| model_error("Unlimited-OCR MoE input shape is invalid."))?;
        let flat = hidden
            .reshape((positions, HIDDEN_SIZE))
            .map_err(tensor_error("flatten MoE input"))?;
        let scores = linear(
            &flat
                .to_dtype(DType::F32)
                .map_err(tensor_error("convert MoE router input to f32"))?,
            &gate
                .to_dtype(DType::F32)
                .map_err(tensor_error("convert MoE router weights to f32"))?,
            None,
        )?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)
            .map_err(tensor_error("normalize MoE routes"))?;
        let probability_rows = probabilities
            .to_device(&candle_core::Device::Cpu)
            .and_then(|value| value.to_vec2::<f32>())
            .map_err(tensor_error("read MoE route probabilities"))?;
        let selections = probability_rows
            .into_iter()
            .map(|row| {
                let mut ranked = row.into_iter().enumerate().collect::<Vec<_>>();
                ranked.sort_by(|(left_index, left), (right_index, right)| {
                    right
                        .total_cmp(left)
                        .then_with(|| left_index.cmp(right_index))
                });
                ranked
                    .into_iter()
                    .take(EXPERTS_PER_TOKEN)
                    .map(|(expert, weight)| RoutedExpert {
                        expert: expert as u32,
                        weight,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let routes =
            RoutedExpertBatch::new(layer, selections, ROUTED_EXPERTS as u32, EXPERTS_PER_TOKEN)
                .map_err(|error| power_error("validate exact Unlimited-OCR routes", error))?;
        self.weights.record_routes(&routes);
        let prefetch = self
            .weights
            .prefetch_experts(&routes, permit, cancellation)?;
        let shared = self.dense_mlp(
            layer,
            &format!("model.layers.{layer}.mlp.shared_experts"),
            hidden,
            SHARED_EXPERT_INTERMEDIATE_SIZE,
            permit,
            cancellation,
        )?;
        ModelWeights::wait_prefetch(prefetch)?;

        // Upstream promotes expert outputs and route weights to f32 for the
        // top-k reduction, then converts the combined result back to the
        // hidden-state dtype. Keeping the accumulator in f32 avoids rounding
        // each probability and partial sum to bf16.
        let mut routed = Tensor::zeros(flat.shape(), DType::F32, flat.device())
            .map_err(tensor_error("allocate routed expert output"))?;
        for expert in routes.experts() {
            check_cancelled(cancellation)?;
            let assignments = routes.assignments(*expert);
            let position_ids = assignments
                .iter()
                .map(|assignment| u32::try_from(assignment.position))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| model_error("Unlimited-OCR MoE position exceeds u32."))?;
            let position_ids = Tensor::from_vec(position_ids, assignments.len(), flat.device())
                .map_err(tensor_error("materialize routed positions"))?;
            let selected = flat
                .index_select(&position_ids, 0)
                .map_err(tensor_error("gather routed tokens"))?;
            let gate = self.weights.load(
                layer,
                &expert_name(layer, *expert, "gate_proj"),
                permit,
                cancellation,
            )?;
            let up = self.weights.load(
                layer,
                &expert_name(layer, *expert, "up_proj"),
                permit,
                cancellation,
            )?;
            let down = self.weights.load(
                layer,
                &expert_name(layer, *expert, "down_proj"),
                permit,
                cancellation,
            )?;
            if gate.dim(0).ok() != Some(EXPERT_INTERMEDIATE_SIZE) {
                return Err(model_error(format!(
                    "Unlimited-OCR layer {layer} expert {expert} has an invalid SwiGLU width."
                )));
            }
            let output = swiglu(&selected, &gate, &up, &down)?;
            let output = weight_expert_output(
                &output,
                assignments
                    .iter()
                    .map(|assignment| assignment.weight)
                    .collect(),
            )?;
            routed = routed
                .index_add(&position_ids, &output, 0)
                .map_err(tensor_error("accumulate routed expert output"))?;
        }
        let routed = routed
            .to_dtype(flat.dtype())
            .map_err(tensor_error("restore routed expert precision"))?
            .reshape(hidden.shape())
            .map_err(tensor_error("restore routed output shape"))?;
        (routed + shared).map_err(tensor_error("add shared expert output"))
    }
}

fn weight_expert_output(output: &Tensor, weights: Vec<f32>) -> UseResult<Tensor> {
    let rows = output
        .dim(0)
        .map_err(tensor_error("inspect routed expert rows"))?;
    if rows != weights.len() {
        return Err(model_error(
            "Unlimited-OCR routed expert weights do not match their output rows.",
        ));
    }
    let route_weights = Tensor::from_vec(weights, (rows, 1), output.device())
        .map_err(tensor_error("materialize f32 route weights"))?;
    output
        .to_dtype(DType::F32)
        .and_then(|value| value.broadcast_mul(&route_weights))
        .map_err(tensor_error("weight routed expert output in f32"))
}

struct DecoderState {
    layers: Vec<Option<KvCache>>,
}

impl DecoderState {
    fn new() -> Self {
        Self {
            layers: (0..DECODER_LAYERS).map(|_| None).collect(),
        }
    }
}

struct KvCache {
    keys: Tensor,
    values: Tensor,
    prefill_len: usize,
}

fn update_cache(
    cache: &mut Option<KvCache>,
    keys: Tensor,
    values: Tensor,
    prefill: bool,
) -> UseResult<(Tensor, Tensor)> {
    let query_len = keys.dim(2).map_err(tensor_error("inspect new keys"))?;
    if prefill || cache.is_none() {
        *cache = Some(KvCache {
            keys: keys.clone(),
            values: values.clone(),
            prefill_len: query_len,
        });
        return Ok((keys, values));
    }
    let current = cache
        .as_ref()
        .ok_or_else(|| model_error("Missing decoder KV cache."))?;
    let current_len = current
        .keys
        .dim(2)
        .map_err(tensor_error("inspect cached keys"))?;
    let decoded = current_len.saturating_sub(current.prefill_len);
    let (next_keys, next_values) = if decoded < SLIDING_WINDOW {
        (
            Tensor::cat(&[&current.keys, &keys], 2),
            Tensor::cat(&[&current.values, &values], 2),
        )
    } else {
        let keep = SLIDING_WINDOW.saturating_sub(query_len).min(decoded);
        let recent_start = current_len.saturating_sub(keep);
        let prefix_keys = current.keys.narrow(2, 0, current.prefill_len);
        let prefix_values = current.values.narrow(2, 0, current.prefill_len);
        let recent_keys = current.keys.narrow(2, recent_start, keep);
        let recent_values = current.values.narrow(2, recent_start, keep);
        (
            prefix_keys.and_then(|prefix| {
                recent_keys.and_then(|recent| Tensor::cat(&[&prefix, &recent, &keys], 2))
            }),
            prefix_values.and_then(|prefix| {
                recent_values.and_then(|recent| Tensor::cat(&[&prefix, &recent, &values], 2))
            }),
        )
    };
    let next_keys = next_keys.map_err(tensor_error("update cached keys"))?;
    let next_values = next_values.map_err(tensor_error("update cached values"))?;
    let prefill_len = current.prefill_len;
    *cache = Some(KvCache {
        keys: next_keys.clone(),
        values: next_values.clone(),
        prefill_len,
    });
    Ok((next_keys, next_values))
}

fn replace_image_rows(
    embeddings: &Tensor,
    vision: &Tensor,
    range: std::ops::Range<usize>,
) -> UseResult<Tensor> {
    let rows = embeddings
        .dim(0)
        .map_err(tensor_error("inspect prompt embeddings"))?;
    if range.start > range.end || range.end > rows {
        return Err(model_error(
            "Unlimited-OCR image placeholder range is invalid.",
        ));
    }
    let prefix = embeddings
        .narrow(0, 0, range.start)
        .map_err(tensor_error("slice prompt prefix"))?;
    let suffix = embeddings
        .narrow(0, range.end, rows - range.end)
        .map_err(tensor_error("slice prompt suffix"))?;
    Tensor::cat(&[&prefix, vision, &suffix], 0)
        .map_err(tensor_error("fuse vision and token embeddings"))
}

fn rms_norm(hidden: &Tensor, weight: Tensor) -> UseResult<Tensor> {
    // DeepSeek's reviewed module computes the variance and normalization in
    // f32, casts the normalized activations back to the input dtype, and only
    // then applies the learned weight. Candle's slow path has that exact order;
    // its fused CPU path quantizes the denominator earlier for bf16 inputs.
    candle_nn::ops::rms_norm_slow(hidden, &weight, RMS_NORM_EPS as f32)
        .map_err(tensor_error("apply RMS normalization"))
}

fn swiglu(input: &Tensor, gate: &Tensor, up: &Tensor, down: &Tensor) -> UseResult<Tensor> {
    let gate = linear(input, gate, None)?
        .silu()
        .map_err(tensor_error("apply SwiGLU SiLU"))?;
    let up = linear(input, up, None)?;
    let hidden = (gate * up).map_err(tensor_error("multiply SwiGLU branches"))?;
    linear(&hidden, down, None)
}

fn apply_rope(q: &Tensor, k: &Tensor, position: usize) -> UseResult<(Tensor, Tensor)> {
    let query_len = q.dim(2).map_err(tensor_error("inspect RoPE queries"))?;
    let (cosine, sine) = rope_tables(position, query_len);
    let cosine = Tensor::from_vec(cosine, (1, 1, query_len, HEAD_DIM), q.device())
        .and_then(|value| value.to_dtype(q.dtype()))
        .map_err(tensor_error("materialize RoPE cosine"))?;
    let sine = Tensor::from_vec(sine, (1, 1, query_len, HEAD_DIM), q.device())
        .and_then(|value| value.to_dtype(q.dtype()))
        .map_err(tensor_error("materialize RoPE sine"))?;
    let q = (q
        .broadcast_mul(&cosine)
        .map_err(tensor_error("rotate query cosine"))?
        + rotate_half(q)?
            .broadcast_mul(&sine)
            .map_err(tensor_error("rotate query sine"))?)
    .map_err(tensor_error("combine rotated queries"))?;
    let k = (k
        .broadcast_mul(&cosine)
        .map_err(tensor_error("rotate key cosine"))?
        + rotate_half(k)?
            .broadcast_mul(&sine)
            .map_err(tensor_error("rotate key sine"))?)
    .map_err(tensor_error("combine rotated keys"))?;
    Ok((q, k))
}

fn rope_tables(position: usize, query_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = Vec::with_capacity(query_len * HEAD_DIM);
    let mut sine = Vec::with_capacity(query_len * HEAD_DIM);
    for token in 0..query_len {
        // The reviewed Llama rotary path builds frequencies, positions, and
        // trigonometric tables in f32 before casting them to bf16.
        let absolute = position.saturating_add(token) as f32;
        let mut half_cos = Vec::with_capacity(HEAD_DIM / 2);
        let mut half_sin = Vec::with_capacity(HEAD_DIM / 2);
        for pair in 0..HEAD_DIM / 2 {
            let exponent = (pair * 2) as f32 / HEAD_DIM as f32;
            let inverse = 1.0_f32 / (ROPE_THETA as f32).powf(exponent);
            let angle = absolute * inverse;
            half_cos.push(angle.cos());
            half_sin.push(angle.sin());
        }
        cosine.extend_from_slice(&half_cos);
        cosine.extend_from_slice(&half_cos);
        sine.extend_from_slice(&half_sin);
        sine.extend_from_slice(&half_sin);
    }
    (cosine, sine)
}

fn rotate_half(value: &Tensor) -> UseResult<Tensor> {
    let first = value
        .narrow(D::Minus1, 0, HEAD_DIM / 2)
        .map_err(tensor_error("slice the first RoPE half"))?;
    let second = value
        .narrow(D::Minus1, HEAD_DIM / 2, HEAD_DIM / 2)
        .and_then(|value| value.neg())
        .map_err(tensor_error("negate the second RoPE half"))?;
    Tensor::cat(&[&second, &first], D::Minus1).map_err(tensor_error("join rotated halves"))
}

fn causal_mask(length: usize, dtype: DType, device: &candle_core::Device) -> UseResult<Tensor> {
    let values = (0..length)
        .flat_map(|row| {
            (0..length).map(
                move |column| {
                    if column > row {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                },
            )
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, length, length), device)
        .and_then(|value| value.to_dtype(dtype))
        .map_err(tensor_error("materialize the causal attention mask"))
}

fn tensor_error(action: &'static str) -> impl FnOnce(candle_core::Error) -> UseError {
    move |error| {
        UseError::new(
            "use.ocr.runtime_failed",
            format!("Failed to {action} in the Unlimited-OCR Rust model: {error}"),
        )
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn rope_tables_broadcast_across_attention_heads() {
        let q = Tensor::ones((1, ATTENTION_HEADS, 3, HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
        let k = q.clone();

        let (q, k) = apply_rope(&q, &k, 4).unwrap();

        assert_eq!(q.dims(), &[1, ATTENTION_HEADS, 3, HEAD_DIM]);
        assert_eq!(k.dims(), q.dims());
    }

    #[test]
    fn routed_expert_weighting_matches_the_upstream_f32_reduction() {
        let output = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weighted = weight_expert_output(&output, vec![0.333_333_34]).unwrap();
        assert_eq!(weighted.dtype(), DType::F32);
        let values = weighted.to_vec2::<f32>().unwrap();
        assert!((values[0][0] - 0.333_333_34).abs() < 1e-7);
        assert!((values[0][1] - 0.666_666_7).abs() < 1e-7);
    }

    #[test]
    fn decoder_rms_norm_keeps_the_reviewed_f32_normalization_order() {
        let hidden = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let weight = Tensor::from_vec(vec![1.0_f32; 4], 4, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let output = rms_norm(&hidden, weight)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(
            output[0],
            [0.365_234_38, 0.730_468_75, 1.093_75, 1.460_937_5]
        );
    }

    #[test]
    fn rope_tables_follow_the_reviewed_f32_frequency_path() {
        let (cosine, sine) = rope_tables(1_000, 1);
        assert!((cosine[10] - (-0.052_861_076)).abs() < 1e-7);
        assert!((sine[10] - (-0.998_601_85)).abs() < 1e-7);
        assert_eq!(cosine[..HEAD_DIM / 2], cosine[HEAD_DIM / 2..]);
        assert_eq!(sine[..HEAD_DIM / 2], sine[HEAD_DIM / 2..]);
    }
}
