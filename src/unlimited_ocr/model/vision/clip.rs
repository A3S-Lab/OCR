use std::sync::Arc;

use a3s_power::inference::ExecutionPermit;
use a3s_use_core::UseResult;
use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use super::ops::{
    check_cancelled, layer_norm, resize_spatial_cubic, scaled_attention, tensor_error,
};
use crate::unlimited_ocr::model::ops::linear;
use crate::unlimited_ocr::model::weights::CLIP_LAYER_BASE;
use crate::unlimited_ocr::model::ModelWeights;

const WIDTH: usize = 1_024;
const LAYERS: usize = 24;
const HEADS: usize = 16;
const HEAD_DIM: usize = WIDTH / HEADS;
const MLP_WIDTH: usize = 4_096;
const POSITION_GRID: usize = 16;

/// Run the checkpoint's CLIP-L transformer using SAM output as its patch
/// embeddings. The checkpoint's raw-pixel CLIP convolution is intentionally
/// unused, matching the authoritative Unlimited-OCR forward path.
pub(super) fn encode(
    weights: &Arc<ModelWeights>,
    sam_features: &Tensor,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> UseResult<Tensor> {
    check_cancelled(cancellation)?;
    let (batch, channels, grid_h, grid_w) = sam_features
        .dims4()
        .map_err(tensor_error("inspect CLIP patch features"))?;
    if channels != WIDTH || grid_h != grid_w {
        return Err(super::ops::model_error(format!(
            "Unlimited-OCR CLIP input must be [N, {WIDTH}, Q, Q]."
        )));
    }
    let patch_tokens = sam_features
        .flatten_from(2)
        .and_then(|value| value.transpose(1, 2))
        .map_err(tensor_error("flatten CLIP patch features"))?;
    let class = weights.load(
        CLIP_LAYER_BASE,
        "model.vision_model.embeddings.class_embedding",
        permit,
        cancellation,
    )?;
    let class = class
        .reshape((1, 1, WIDTH))
        .and_then(|value| value.broadcast_as((batch, 1, WIDTH)))
        .map_err(tensor_error("expand the CLIP class embedding"))?;
    let mut hidden = Tensor::cat(&[&class, &patch_tokens], 1)
        .map_err(tensor_error("prepend the CLIP class embedding"))?;
    let positions = positions(weights, grid_h, permit, cancellation)?;
    hidden = hidden
        .broadcast_add(&positions)
        .map_err(tensor_error("add CLIP absolute positions"))?;
    let pre_weight = weights.load(
        CLIP_LAYER_BASE,
        "model.vision_model.pre_layrnorm.weight",
        permit,
        cancellation,
    )?;
    let pre_bias = weights.load(
        CLIP_LAYER_BASE,
        "model.vision_model.pre_layrnorm.bias",
        permit,
        cancellation,
    )?;
    hidden = layer_norm(&hidden, pre_weight, pre_bias, 1e-5)?;

    for layer in 0..LAYERS {
        check_cancelled(cancellation)?;
        hidden = block(weights, layer, &hidden, permit, cancellation)?;
    }
    hidden
        .narrow(1, 1, grid_h * grid_w)
        .map_err(tensor_error("drop the CLIP class output"))
}

fn positions(
    weights: &Arc<ModelWeights>,
    target_grid: usize,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> UseResult<Tensor> {
    let positions = weights.load(
        CLIP_LAYER_BASE,
        "model.vision_model.embeddings.position_embedding.weight",
        permit,
        cancellation,
    )?;
    let class = positions
        .narrow(0, 0, 1)
        .and_then(|value| value.unsqueeze(0))
        .map_err(tensor_error("select the CLIP class position"))?;
    let patches = positions
        .narrow(0, 1, POSITION_GRID * POSITION_GRID)
        .and_then(|value| value.reshape((1, POSITION_GRID, POSITION_GRID, WIDTH)))
        .and_then(|value| value.permute((0, 3, 1, 2)))
        .map_err(tensor_error("shape the CLIP patch positions"))?;
    let patches = resize_spatial_cubic(&patches, target_grid)?
        .permute((0, 2, 3, 1))
        .and_then(|value| value.reshape((1, target_grid * target_grid, WIDTH)))
        .map_err(tensor_error("flatten the resized CLIP positions"))?;
    Tensor::cat(&[&class, &patches], 1).map_err(tensor_error("join CLIP class and patch positions"))
}

fn block(
    weights: &Arc<ModelWeights>,
    layer: usize,
    hidden: &Tensor,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> UseResult<Tensor> {
    let runtime_layer = CLIP_LAYER_BASE + layer as u32;
    let prefix = format!("model.vision_model.transformer.layers.{layer}");
    let norm1_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.layer_norm1.weight"),
        permit,
        cancellation,
    )?;
    let norm1_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.layer_norm1.bias"),
        permit,
        cancellation,
    )?;
    let normalized = layer_norm(hidden, norm1_weight, norm1_bias, 1e-5)?;
    let attention = attention(
        weights,
        runtime_layer,
        &prefix,
        &normalized,
        permit,
        cancellation,
    )?;
    let hidden = (hidden + attention).map_err(tensor_error("add a CLIP attention residual"))?;
    let norm2_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.layer_norm2.weight"),
        permit,
        cancellation,
    )?;
    let norm2_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.layer_norm2.bias"),
        permit,
        cancellation,
    )?;
    let normalized = layer_norm(&hidden, norm2_weight, norm2_bias, 1e-5)?;
    let first_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.fc1.weight"),
        permit,
        cancellation,
    )?;
    let first_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.fc1.bias"),
        permit,
        cancellation,
    )?;
    let first = linear(&normalized, &first_weight, Some(&first_bias))?;
    if first.dim(candle_core::D::Minus1).ok() != Some(MLP_WIDTH) {
        return Err(super::ops::model_error(
            "Unlimited-OCR CLIP MLP has an invalid intermediate width.",
        ));
    }
    let sigmoid = candle_nn::ops::sigmoid(
        &(&first * 1.702_f64).map_err(tensor_error("scale the CLIP QuickGELU input"))?,
    )
    .map_err(tensor_error("apply the CLIP QuickGELU sigmoid"))?;
    let first = (&first * sigmoid).map_err(tensor_error("apply the CLIP QuickGELU gate"))?;
    let second_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.fc2.weight"),
        permit,
        cancellation,
    )?;
    let second_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.fc2.bias"),
        permit,
        cancellation,
    )?;
    let feed_forward = linear(&first, &second_weight, Some(&second_bias))?;
    (&hidden + feed_forward).map_err(tensor_error("add a CLIP feed-forward residual"))
}

fn attention(
    weights: &Arc<ModelWeights>,
    runtime_layer: u32,
    prefix: &str,
    hidden: &Tensor,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> UseResult<Tensor> {
    let (batch, sequence, channels) = hidden
        .dims3()
        .map_err(tensor_error("inspect CLIP attention input"))?;
    weights.checked_elements(
        &[batch, HEADS, sequence, sequence],
        "Unlimited-OCR CLIP attention scores",
    )?;
    let qkv_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.self_attn.qkv_proj.weight"),
        permit,
        cancellation,
    )?;
    let qkv_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.self_attn.qkv_proj.bias"),
        permit,
        cancellation,
    )?;
    let qkv = linear(hidden, &qkv_weight, Some(&qkv_bias))?
        .reshape((batch, sequence, 3, HEADS, HEAD_DIM))
        .and_then(|value| value.permute((2, 0, 3, 1, 4)))
        .map_err(tensor_error("shape CLIP query, key, and value tensors"))?;
    let q = qkv
        .get(0)
        .map_err(tensor_error("select CLIP attention queries"))?;
    let k = qkv
        .get(1)
        .map_err(tensor_error("select CLIP attention keys"))?;
    let v = qkv
        .get(2)
        .map_err(tensor_error("select CLIP attention values"))?;
    let context = scaled_attention(&q, &k, &v)?
        .permute((0, 2, 1, 3))
        .and_then(|value| value.reshape((batch, sequence, channels)))
        .map_err(tensor_error("merge CLIP attention heads"))?;
    let output_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.self_attn.out_proj.weight"),
        permit,
        cancellation,
    )?;
    let output_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.self_attn.out_proj.bias"),
        permit,
        cancellation,
    )?;
    linear(&context, &output_weight, Some(&output_bias))
}
