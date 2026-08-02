mod clip;
mod ops;
mod sam;

use std::sync::Arc;

use a3s_power::inference::ExecutionPermit;
use a3s_use_core::UseResult;
use candle_core::{Tensor, D};
use tokio_util::sync::CancellationToken;

use self::ops::{check_cancelled, model_error, tensor_error};
use super::ops::linear;
use super::weights::PROJECTOR_LAYER;
use super::{ModelWeights, HIDDEN_SIZE};
use crate::unlimited_ocr::preprocess::{PreprocessedImage, GLOBAL_IMAGE_SIDE, TILE_IMAGE_SIDE};

const SAM_PATCH_SIDE: usize = 16;
const SAM_HEADS: usize = 12;
const FUSED_WIDTH: usize = 2_048;

pub(crate) struct VisionEncoder {
    weights: Arc<ModelWeights>,
}

impl VisionEncoder {
    pub(crate) fn new(weights: Arc<ModelWeights>) -> Self {
        Self { weights }
    }

    pub(crate) fn encode(
        &self,
        image: &PreprocessedImage,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        check_cancelled(cancellation)?;
        let projector = ProjectorWeights::load(&self.weights, permit, cancellation)?;
        let global_input = self.input_tensor(
            &image.global,
            1,
            GLOBAL_IMAGE_SIDE as usize,
            "Unlimited-OCR global image view",
        )?;
        let global = self.encode_views(&global_input, &projector, permit, cancellation)?;
        let global_grid = square_side(
            global
                .dim(1)
                .map_err(tensor_error("inspect global vision tokens"))?,
            "global",
        )?;
        let global = append_newlines(
            global
                .get(0)
                .map_err(tensor_error("select the global vision view"))?,
            global_grid,
            global_grid,
            &projector.newline,
        )?;

        let mut sections = Vec::with_capacity(3);
        if !image.tiles.is_empty() {
            if image.tiles.len()
                != (image.tile_columns as usize).saturating_mul(image.tile_rows as usize)
            {
                return Err(model_error(
                    "Unlimited-OCR tile count does not match its spatial crop grid.",
                ));
            }
            let tiles = self.encode_tiles(image, &projector, permit, cancellation)?;
            let tile_grid = square_side(
                tiles
                    .dim(1)
                    .map_err(tensor_error("inspect local vision tokens"))?,
                "local",
            )?;
            let rows = image.tile_rows as usize;
            let columns = image.tile_columns as usize;
            let mosaic = tiles
                .reshape((rows, columns, tile_grid, tile_grid, HIDDEN_SIZE))
                .and_then(|value| value.permute((0, 2, 1, 3, 4)))
                .and_then(|value| {
                    value.reshape((rows * tile_grid, columns * tile_grid, HIDDEN_SIZE))
                })
                .map_err(tensor_error("assemble the local vision mosaic"))?;
            sections.push(append_newlines(
                mosaic,
                rows * tile_grid,
                columns * tile_grid,
                &projector.newline,
            )?);
        }
        sections.push(global);
        sections.push(
            projector
                .separator
                .reshape((1, HIDDEN_SIZE))
                .map_err(tensor_error("shape the vision view separator"))?,
        );
        let references = sections.iter().collect::<Vec<_>>();
        let packed = Tensor::cat(&references, 0)
            .map_err(tensor_error("pack Unlimited-OCR vision tokens"))?;
        let actual = packed
            .dim(0)
            .map_err(tensor_error("inspect packed vision tokens"))?;
        let expected = image.image_token_count();
        if actual != expected {
            return Err(model_error(format!(
                "Unlimited-OCR packed {actual} vision tokens for {expected} prompt placeholders."
            )));
        }
        Ok(packed)
    }

    fn encode_tiles(
        &self,
        image: &PreprocessedImage,
        projector: &ProjectorWeights,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let side = TILE_IMAGE_SIDE as usize;
        let patch_grid = side / SAM_PATCH_SIDE;
        let attention_elements_per_view =
            SAM_HEADS.saturating_mul(patch_grid.saturating_mul(patch_grid).pow(2));
        let limit = self.weights.hierarchy().runtime().limits();
        let batch_limit = (limit.max_tensor_elements / attention_elements_per_view)
            .max(1)
            .min(image.tiles.len());
        let mut encoded = Vec::new();
        for chunk in image.tiles.chunks(batch_limit) {
            check_cancelled(cancellation)?;
            let values_per_tile = 3_usize.saturating_mul(side).saturating_mul(side);
            let capacity = values_per_tile
                .checked_mul(chunk.len())
                .ok_or_else(|| model_error("Unlimited-OCR tile-batch size overflowed."))?;
            let mut values = Vec::with_capacity(capacity);
            for tile in chunk {
                values.extend_from_slice(tile);
            }
            let input =
                self.input_tensor(&values, chunk.len(), side, "Unlimited-OCR local tile batch")?;
            encoded.push(self.encode_views(&input, projector, permit, cancellation)?);
        }
        let references = encoded.iter().collect::<Vec<_>>();
        Tensor::cat(&references, 0).map_err(tensor_error("join local vision tile batches"))
    }

    fn encode_views(
        &self,
        input: &Tensor,
        projector: &ProjectorWeights,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let sam = sam::encode(&self.weights, input, permit, cancellation)?;
        let clip = clip::encode(&self.weights, &sam, permit, cancellation)?;
        let sam_tokens = sam
            .flatten_from(2)
            .and_then(|value| value.transpose(1, 2))
            .map_err(tensor_error("flatten SAM fusion features"))?;
        let fused = Tensor::cat(&[&clip, &sam_tokens], D::Minus1)
            .map_err(tensor_error("fuse SAM and CLIP features"))?;
        if fused.dim(D::Minus1).ok() != Some(FUSED_WIDTH) {
            return Err(model_error(
                "Unlimited-OCR vision fusion did not produce 2048 channels.",
            ));
        }
        linear(&fused, &projector.weight, Some(&projector.bias))
    }

    fn input_tensor(
        &self,
        values: &[f32],
        batch: usize,
        side: usize,
        label: &str,
    ) -> UseResult<Tensor> {
        let shape = [batch, 3, side, side];
        self.weights
            .hierarchy()
            .runtime()
            .limits()
            .checked_elements(&shape, label)
            .map_err(|error| {
                model_error(format!("{label} exceeds the Power tensor limit: {error}"))
            })?;
        let expected = shape.iter().product::<usize>();
        if values.len() != expected {
            return Err(model_error(format!(
                "{label} contains {} values instead of {expected}.",
                values.len()
            )));
        }
        Tensor::from_slice(
            values,
            &shape,
            self.weights.hierarchy().runtime().device().tensor_device(),
        )
        .map_err(tensor_error("materialize a vision input tensor"))
    }
}

struct ProjectorWeights {
    weight: Tensor,
    bias: Tensor,
    newline: Tensor,
    separator: Tensor,
}

impl ProjectorWeights {
    fn load(
        weights: &Arc<ModelWeights>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Self> {
        Ok(Self {
            weight: weights.load(
                PROJECTOR_LAYER,
                "model.projector.layers.weight",
                permit,
                cancellation,
            )?,
            bias: weights.load(
                PROJECTOR_LAYER,
                "model.projector.layers.bias",
                permit,
                cancellation,
            )?,
            newline: weights.load_global("model.image_newline", permit, cancellation)?,
            separator: weights.load_global("model.view_seperator", permit, cancellation)?,
        })
    }
}

fn append_newlines(
    tokens: Tensor,
    rows: usize,
    columns: usize,
    newline: &Tensor,
) -> UseResult<Tensor> {
    let tokens = tokens
        .reshape((rows, columns, HIDDEN_SIZE))
        .map_err(tensor_error("shape a vision token grid"))?;
    let newline = newline
        .reshape((1, 1, HIDDEN_SIZE))
        .and_then(|value| value.broadcast_as((rows, 1, HIDDEN_SIZE)))
        .map_err(tensor_error("expand image-newline embeddings"))?;
    Tensor::cat(&[&tokens, &newline], 1)
        .and_then(|value| value.reshape((rows * (columns + 1), HIDDEN_SIZE)))
        .map_err(tensor_error("append image-newline embeddings"))
}

fn square_side(tokens: usize, label: &str) -> UseResult<usize> {
    let side = (tokens as f64).sqrt() as usize;
    if side.saturating_mul(side) != tokens {
        return Err(model_error(format!(
            "Unlimited-OCR {label} view returned a non-square token grid."
        )));
    }
    Ok(side)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn newlines_are_appended_after_each_spatial_row() {
        let tokens = Tensor::from_vec(
            (0..4 * HIDDEN_SIZE)
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            (2, 2, HIDDEN_SIZE),
            &Device::Cpu,
        )
        .unwrap();
        let newline =
            Tensor::from_vec(vec![-1.0_f32; HIDDEN_SIZE], HIDDEN_SIZE, &Device::Cpu).unwrap();
        let output = append_newlines(tokens, 2, 2, &newline).unwrap();
        assert_eq!(output.dims(), &[6, HIDDEN_SIZE]);
        let output = output.to_vec2::<f32>().unwrap();
        assert!(output[2].iter().all(|value| *value == -1.0));
        assert!(output[5].iter().all(|value| *value == -1.0));
    }
}
