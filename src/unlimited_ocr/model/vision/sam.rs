use std::sync::Arc;

use a3s_power::inference::ExecutionPermit;
use a3s_use_core::UseResult;
use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use super::ops::{
    attention_from_scores, check_cancelled, conv2d, layer_norm, layer_norm_2d, resize_linear_1d,
    resize_spatial_cubic, tensor_error,
};
use crate::unlimited_ocr::model::ops::{linear, matmul};
use crate::unlimited_ocr::model::weights::SAM_LAYER_BASE;
use crate::unlimited_ocr::model::ModelWeights;

const WIDTH: usize = 768;
const LAYERS: usize = 12;
const HEADS: usize = 12;
const HEAD_DIM: usize = WIDTH / HEADS;
const WINDOW: usize = 14;
const MLP_WIDTH: usize = WIDTH * 4;
const GLOBAL_LAYERS: [usize; 4] = [2, 5, 8, 11];

pub(super) fn encode(
    weights: &Arc<ModelWeights>,
    input: &Tensor,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> UseResult<Tensor> {
    check_cancelled(cancellation)?;
    let patch_weight = weights.load(
        SAM_LAYER_BASE,
        "model.sam_model.patch_embed.proj.weight",
        permit,
        cancellation,
    )?;
    let patch_bias = weights.load(
        SAM_LAYER_BASE,
        "model.sam_model.patch_embed.proj.bias",
        permit,
        cancellation,
    )?;
    let input = input
        .to_dtype(patch_weight.dtype())
        .map_err(tensor_error("convert SAM input precision"))?;
    let mut hidden = conv2d(&input, patch_weight, Some(patch_bias), 16, 0)?
        .permute((0, 2, 3, 1))
        .map_err(tensor_error("move SAM patch channels last"))?;
    let grid = hidden
        .dim(1)
        .map_err(tensor_error("inspect the SAM patch grid"))?;
    let pos = weights.load(
        SAM_LAYER_BASE,
        "model.sam_model.pos_embed",
        permit,
        cancellation,
    )?;
    let pos = if pos.dim(1).ok() == Some(grid) {
        pos
    } else {
        resize_spatial_cubic(
            &pos.permute((0, 3, 1, 2))
                .map_err(tensor_error("move SAM positions channels first"))?,
            grid,
        )?
        .permute((0, 2, 3, 1))
        .map_err(tensor_error("restore SAM position channels"))?
    };
    hidden = hidden
        .broadcast_add(&pos)
        .map_err(tensor_error("add SAM absolute positions"))?;

    for layer in 0..LAYERS {
        check_cancelled(cancellation)?;
        hidden = block(weights, layer, &hidden, permit, cancellation)?;
    }

    let hidden = hidden
        .permute((0, 3, 1, 2))
        .map_err(tensor_error("move SAM features channels first"))?;
    let neck0 = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.neck.0.weight",
        permit,
        cancellation,
    )?;
    let hidden = conv2d(&hidden, neck0, None, 1, 0)?;
    let neck1_weight = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.neck.1.weight",
        permit,
        cancellation,
    )?;
    let neck1_bias = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.neck.1.bias",
        permit,
        cancellation,
    )?;
    let hidden = layer_norm_2d(&hidden, neck1_weight, neck1_bias, 1e-6)?;
    let neck2 = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.neck.2.weight",
        permit,
        cancellation,
    )?;
    let hidden = conv2d(&hidden, neck2, None, 1, 1)?;
    let neck3_weight = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.neck.3.weight",
        permit,
        cancellation,
    )?;
    let neck3_bias = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.neck.3.bias",
        permit,
        cancellation,
    )?;
    let hidden = layer_norm_2d(&hidden, neck3_weight, neck3_bias, 1e-6)?;
    let net2 = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.net_2.weight",
        permit,
        cancellation,
    )?;
    let hidden = conv2d(&hidden, net2, None, 2, 1)?;
    let net3 = weights.load(
        SAM_LAYER_BASE + LAYERS as u32,
        "model.sam_model.net_3.weight",
        permit,
        cancellation,
    )?;
    conv2d(&hidden, net3, None, 2, 1)
}

fn block(
    weights: &Arc<ModelWeights>,
    layer: usize,
    hidden: &Tensor,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> UseResult<Tensor> {
    let prefix = format!("model.sam_model.blocks.{layer}");
    let runtime_layer = SAM_LAYER_BASE + layer as u32;
    let norm1_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.norm1.weight"),
        permit,
        cancellation,
    )?;
    let norm1_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.norm1.bias"),
        permit,
        cancellation,
    )?;
    let normalized = layer_norm(hidden, norm1_weight, norm1_bias, 1e-6)?;
    let size = normalized
        .dim(1)
        .map_err(tensor_error("inspect a SAM block grid"))?;
    let windowed = !GLOBAL_LAYERS.contains(&layer);
    let (attention_input, padded) = if windowed {
        window_partition(normalized, WINDOW)?
    } else {
        (normalized, (size, size))
    };
    let attention = attention(
        weights,
        runtime_layer,
        &prefix,
        &attention_input,
        permit,
        cancellation,
    )?;
    let attention = if windowed {
        window_unpartition(attention, WINDOW, padded, (size, size))?
    } else {
        attention
    };
    let hidden = (hidden + attention).map_err(tensor_error("add a SAM attention residual"))?;
    let norm2_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.norm2.weight"),
        permit,
        cancellation,
    )?;
    let norm2_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.norm2.bias"),
        permit,
        cancellation,
    )?;
    let normalized = layer_norm(&hidden, norm2_weight, norm2_bias, 1e-6)?;
    let first_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.lin1.weight"),
        permit,
        cancellation,
    )?;
    let first_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.lin1.bias"),
        permit,
        cancellation,
    )?;
    let first = linear(&normalized, &first_weight, Some(&first_bias))?;
    if first.dim(candle_core::D::Minus1).ok() != Some(MLP_WIDTH) {
        return Err(super::ops::model_error(
            "Unlimited-OCR SAM MLP has an invalid intermediate width.",
        ));
    }
    let first = first
        .gelu_erf()
        .map_err(tensor_error("apply the SAM GELU activation"))?;
    let second_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.lin2.weight"),
        permit,
        cancellation,
    )?;
    let second_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.mlp.lin2.bias"),
        permit,
        cancellation,
    )?;
    let feed_forward = linear(&first, &second_weight, Some(&second_bias))?;
    (&hidden + feed_forward).map_err(tensor_error("add a SAM feed-forward residual"))
}

fn attention(
    weights: &Arc<ModelWeights>,
    runtime_layer: u32,
    prefix: &str,
    hidden: &Tensor,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> UseResult<Tensor> {
    let (batch, height, width, channels) = hidden
        .dims4()
        .map_err(tensor_error("inspect SAM attention input"))?;
    let sequence = height.saturating_mul(width);
    weights.checked_elements(
        &[batch, HEADS, sequence, sequence],
        "Unlimited-OCR SAM attention scores",
    )?;
    let qkv_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.attn.qkv.weight"),
        permit,
        cancellation,
    )?;
    let qkv_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.attn.qkv.bias"),
        permit,
        cancellation,
    )?;
    let qkv = linear(hidden, &qkv_weight, Some(&qkv_bias))?
        .reshape((batch, sequence, 3, HEADS, HEAD_DIM))
        .and_then(|value| value.permute((2, 0, 3, 1, 4)))
        .map_err(tensor_error("shape SAM query, key, and value tensors"))?;
    let q = qkv
        .get(0)
        .map_err(tensor_error("select SAM attention queries"))?;
    let k = qkv
        .get(1)
        .map_err(tensor_error("select SAM attention keys"))?;
    let v = qkv
        .get(2)
        .map_err(tensor_error("select SAM attention values"))?;
    let scores = q
        .contiguous()
        .and_then(|q| {
            k.transpose(2, 3)
                .and_then(|k| k.contiguous())
                .and_then(|k| matmul(&q, &k))
        })
        .and_then(|scores| scores / (HEAD_DIM as f64).sqrt())
        .map_err(tensor_error("compute SAM attention scores"))?;
    let rel_h = weights.load(
        runtime_layer,
        &format!("{prefix}.attn.rel_pos_h"),
        permit,
        cancellation,
    )?;
    let rel_w = weights.load(
        runtime_layer,
        &format!("{prefix}.attn.rel_pos_w"),
        permit,
        cancellation,
    )?;
    let scores = add_relative_positions(scores, &q, rel_h, rel_w, height, width)?;
    let context = attention_from_scores(&scores, &v)?
        .permute((0, 2, 1, 3))
        .and_then(|value| value.reshape((batch, height, width, channels)))
        .map_err(tensor_error("merge SAM attention heads"))?;
    let projection_weight = weights.load(
        runtime_layer,
        &format!("{prefix}.attn.proj.weight"),
        permit,
        cancellation,
    )?;
    let projection_bias = weights.load(
        runtime_layer,
        &format!("{prefix}.attn.proj.bias"),
        permit,
        cancellation,
    )?;
    linear(&context, &projection_weight, Some(&projection_bias))
}

fn add_relative_positions(
    scores: Tensor,
    queries: &Tensor,
    rel_h: Tensor,
    rel_w: Tensor,
    height: usize,
    width: usize,
) -> UseResult<Tensor> {
    let batch = queries
        .dim(0)
        .map_err(tensor_error("inspect SAM relative-position batch"))?;
    let batch_heads = batch.saturating_mul(HEADS);
    let rel_h = relative_table(rel_h, height)?;
    let rel_w = relative_table(rel_w, width)?;
    let queries = queries
        .reshape((batch_heads, height, width, HEAD_DIM))
        .map_err(tensor_error("shape SAM relative-position queries"))?;
    let height_positions = rel_h
        .broadcast_left(batch_heads)
        .and_then(|value| value.transpose(2, 3))
        .and_then(|value| value.contiguous())
        .map_err(tensor_error("shape SAM height positions"))?;
    let height_bias = matmul(&queries, &height_positions)
        .map_err(tensor_error("compute SAM height-relative bias"))?;
    let width_bias = queries
        .transpose(1, 2)
        .and_then(|value| value.contiguous())
        .and_then(|value| {
            rel_w
                .broadcast_left(batch_heads)
                .and_then(|positions| positions.transpose(2, 3))
                .and_then(|positions| positions.contiguous())
                .and_then(|positions| matmul(&value, &positions))
        })
        .and_then(|value| value.transpose(1, 2))
        .map_err(tensor_error("compute SAM width-relative bias"))?;
    scores
        .reshape((batch_heads, height, width, height, width))
        .and_then(|value| {
            height_bias
                .unsqueeze(4)
                .and_then(|bias| value.broadcast_add(&bias))
        })
        .and_then(|value| {
            width_bias
                .unsqueeze(3)
                .and_then(|bias| value.broadcast_add(&bias))
        })
        .and_then(|value| value.reshape((batch, HEADS, height * width, height * width)))
        .map_err(tensor_error("add decomposed SAM relative positions"))
}

fn relative_table(table: Tensor, size: usize) -> UseResult<Tensor> {
    let target = size.saturating_mul(2).saturating_sub(1);
    let table = if table.dim(0).ok() == Some(target) {
        table
    } else {
        resize_linear_1d(&table, target)?
    };
    let indices = (0..size)
        .flat_map(|query| (0..size).map(move |key| query + size - 1 - key))
        .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
        .collect::<Vec<_>>();
    let indices = Tensor::from_vec(indices, size * size, table.device())
        .map_err(tensor_error("materialize SAM relative-position indices"))?;
    table
        .index_select(&indices, 0)
        .and_then(|value| value.reshape((size, size, HEAD_DIM)))
        .map_err(tensor_error("select SAM relative positions"))
}

fn window_partition(hidden: Tensor, window: usize) -> UseResult<(Tensor, (usize, usize))> {
    let (batch, height, width, channels) = hidden
        .dims4()
        .map_err(tensor_error("inspect SAM window input"))?;
    let pad_h = (window - height % window) % window;
    let pad_w = (window - width % window) % window;
    let hidden = if pad_h > 0 {
        hidden
            .pad_with_zeros(1, 0, pad_h)
            .map_err(tensor_error("pad SAM window rows"))?
    } else {
        hidden
    };
    let hidden = if pad_w > 0 {
        hidden
            .pad_with_zeros(2, 0, pad_w)
            .map_err(tensor_error("pad SAM window columns"))?
    } else {
        hidden
    };
    let padded_h = height + pad_h;
    let padded_w = width + pad_w;
    let windows = hidden
        .reshape((
            batch,
            padded_h / window,
            window,
            padded_w / window,
            window,
            channels,
        ))
        .and_then(|value| value.permute((0, 1, 3, 2, 4, 5)))
        .and_then(|value| {
            value.reshape((
                batch * (padded_h / window) * (padded_w / window),
                window,
                window,
                channels,
            ))
        })
        .map_err(tensor_error("partition SAM attention windows"))?;
    Ok((windows, (padded_h, padded_w)))
}

fn window_unpartition(
    windows: Tensor,
    window: usize,
    (padded_h, padded_w): (usize, usize),
    (height, width): (usize, usize),
) -> UseResult<Tensor> {
    let window_count = padded_h * padded_w / window / window;
    let batch = windows
        .dim(0)
        .map_err(tensor_error("inspect SAM output windows"))?
        / window_count;
    let channels = windows
        .dim(3)
        .map_err(tensor_error("inspect SAM window channels"))?;
    let hidden = windows
        .reshape((
            batch,
            padded_h / window,
            padded_w / window,
            window,
            window,
            channels,
        ))
        .and_then(|value| value.permute((0, 1, 3, 2, 4, 5)))
        .and_then(|value| value.reshape((batch, padded_h, padded_w, channels)))
        .and_then(|value| value.narrow(1, 0, height))
        .and_then(|value| value.narrow(2, 0, width))
        .map_err(tensor_error("unpartition SAM attention windows"))?;
    Ok(hidden)
}
