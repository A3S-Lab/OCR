pub(super) use crate::cancellation::check_cancelled;
use a3s_use_core::{UseError, UseResult};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module};

pub(super) fn layer_norm(
    input: &Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: f64,
) -> UseResult<Tensor> {
    LayerNorm::new(weight, bias, epsilon)
        .forward(input)
        .map_err(tensor_error("apply vision layer normalization"))
}

pub(super) fn layer_norm_2d(
    input: &Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: f64,
) -> UseResult<Tensor> {
    input
        .permute((0, 2, 3, 1))
        .map_err(tensor_error("move vision channels for 2D normalization"))
        .and_then(|value| layer_norm(&value, weight, bias, epsilon))?
        .permute((0, 3, 1, 2))
        .map_err(tensor_error(
            "restore vision channels after 2D normalization",
        ))
}

pub(super) fn conv2d(
    input: &Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
) -> UseResult<Tensor> {
    Conv2d::new(
        weight,
        bias,
        Conv2dConfig {
            stride,
            padding,
            ..Conv2dConfig::default()
        },
    )
    .forward(input)
    .map_err(tensor_error("execute a vision convolution"))
}

pub(super) fn scaled_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> UseResult<Tensor> {
    let head_dim = q
        .dim(candle_core::D::Minus1)
        .map_err(tensor_error("inspect vision attention heads"))?;
    let scores = q
        .contiguous()
        .and_then(|q| {
            k.transpose(2, 3)
                .and_then(|k| k.contiguous())
                .and_then(|k| q.matmul(&k))
        })
        .and_then(|scores| scores / (head_dim as f64).sqrt())
        .map_err(tensor_error("compute scaled vision attention"))?;
    attention_from_scores(&scores, v)
}

pub(super) fn attention_from_scores(scores: &Tensor, values: &Tensor) -> UseResult<Tensor> {
    let probabilities = candle_nn::ops::softmax_last_dim(
        &scores
            .to_dtype(DType::F32)
            .map_err(tensor_error("stabilize vision attention scores"))?,
    )
    .and_then(|value| value.to_dtype(values.dtype()))
    .map_err(tensor_error("normalize vision attention scores"))?;
    probabilities
        .matmul(
            &values
                .contiguous()
                .map_err(tensor_error("pack vision attention values"))?,
        )
        .map_err(tensor_error("apply vision attention values"))
}

/// Resize learned spatial embeddings with an antialiased cubic kernel.
///
/// Positional tensors are small compared with activations and weights. Doing
/// the interpolation in host f32 keeps CPU, CUDA, and Metal behavior stable,
/// then returns the derived tensor to the Power-selected execution device.
pub(super) fn resize_spatial_cubic(input: &Tensor, target: usize) -> UseResult<Tensor> {
    let (batch, channels, source_h, source_w) = input
        .dims4()
        .map_err(tensor_error("inspect a spatial position embedding"))?;
    if batch != 1 || source_h == 0 || source_h != source_w || target == 0 {
        return Err(model_error(
            "Unlimited-OCR spatial position embeddings must be [1, C, S, S] with a positive target.",
        ));
    }
    if source_h == target {
        return Ok(input.clone());
    }
    let dtype = input.dtype();
    let device = input.device().clone();
    let source = host_f32(input)?;
    let horizontal = cubic_contributors(source_w, target);
    let vertical = cubic_contributors(source_h, target);
    let mut intermediate = vec![0.0_f32; channels * source_h * target];
    for channel in 0..channels {
        for row in 0..source_h {
            for (column, contributors) in horizontal.iter().enumerate() {
                let value = contributors
                    .iter()
                    .fold(0.0_f32, |sum, (source_x, weight)| {
                        let index = (channel * source_h + row) * source_w + source_x;
                        sum + source[index] * weight
                    });
                intermediate[(channel * source_h + row) * target + column] = value;
            }
        }
    }
    let mut output = vec![0.0_f32; channels * target * target];
    for channel in 0..channels {
        for (row, contributors) in vertical.iter().enumerate() {
            for column in 0..target {
                let value = contributors
                    .iter()
                    .fold(0.0_f32, |sum, (source_y, weight)| {
                        let index = (channel * source_h + source_y) * target + column;
                        sum + intermediate[index] * weight
                    });
                output[(channel * target + row) * target + column] = value;
            }
        }
    }
    tensor_on_device(output, vec![1, channels, target, target], dtype, &device)
}

pub(super) fn resize_linear_1d(input: &Tensor, target: usize) -> UseResult<Tensor> {
    let (source_len, width) = input
        .dims2()
        .map_err(tensor_error("inspect a relative position embedding"))?;
    if target == 0 {
        return Err(model_error(
            "Unlimited-OCR relative position targets must be positive.",
        ));
    }
    if source_len == target {
        return Ok(input.clone());
    }
    let dtype = input.dtype();
    let device = input.device().clone();
    let source = host_f32(input)?;
    let scale = source_len as f64 / target as f64;
    let mut output = vec![0.0_f32; target * width];
    for target_index in 0..target {
        let position = (target_index as f64 + 0.5) * scale - 0.5;
        let lower = position.floor() as isize;
        let upper = lower.saturating_add(1);
        let upper_weight = (position - lower as f64) as f32;
        let lower_weight = 1.0 - upper_weight;
        let lower = clamp_index(lower, source_len);
        let upper = clamp_index(upper, source_len);
        for column in 0..width {
            output[target_index * width + column] = source[lower * width + column] * lower_weight
                + source[upper * width + column] * upper_weight;
        }
    }
    tensor_on_device(output, vec![target, width], dtype, &device)
}

fn cubic_contributors(source: usize, target: usize) -> Vec<Vec<(usize, f32)>> {
    // Match torch.nn.functional.interpolate(..., mode="bicubic",
    // antialias=True, align_corners=False). PyTorch uses Pillow-compatible
    // Keys cubic weights, clips the contributor interval to the source image,
    // and renormalizes the remaining in-bounds weights rather than extending
    // border samples.
    let scale = source as f32 / target as f32;
    let inverse_scale = if scale >= 1.0 { scale.recip() } else { 1.0 };
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    (0..target)
        .map(|target_index| {
            let center = scale * (target_index as f32 + 0.5);
            let first = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as isize)
                .max(0)
                .min(source as isize) as usize;
            let mut contributors = (first..end)
                .map(|source_index| {
                    let distance = (source_index as f32 - center + 0.5) * inverse_scale;
                    (source_index, cubic_kernel(distance))
                })
                .collect::<Vec<_>>();
            let total = contributors.iter().map(|(_, weight)| *weight).sum::<f32>();
            if total != 0.0 {
                for (_, weight) in &mut contributors {
                    *weight /= total;
                }
            }
            contributors
        })
        .collect()
}

fn cubic_kernel(value: f32) -> f32 {
    let value = value.abs();
    // PyTorch uses A=-0.5 for antialiased bicubic interpolation to remain
    // compatible with Pillow. A=-0.75 belongs to its non-antialiased path.
    const A: f32 = -0.5;
    if value < 1.0 {
        (A + 2.0) * value.powi(3) - (A + 3.0) * value.powi(2) + 1.0
    } else if value < 2.0 {
        A * value.powi(3) - 5.0 * A * value.powi(2) + 8.0 * A * value - 4.0 * A
    } else {
        0.0
    }
}

fn clamp_index(index: isize, length: usize) -> usize {
    index.clamp(0, length.saturating_sub(1) as isize) as usize
}

fn host_f32(input: &Tensor) -> UseResult<Vec<f32>> {
    input
        .to_device(&Device::Cpu)
        .and_then(|value| value.to_dtype(DType::F32))
        .and_then(|value| value.flatten_all())
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(tensor_error("read a learned position embedding"))
}

fn tensor_on_device(
    values: Vec<f32>,
    shape: Vec<usize>,
    dtype: DType,
    device: &Device,
) -> UseResult<Tensor> {
    Tensor::from_vec(values, shape, &Device::Cpu)
        .and_then(|value| value.to_dtype(dtype))
        .and_then(|value| value.to_device(device))
        .map_err(tensor_error("materialize a resized position embedding"))
}

pub(super) fn tensor_error(action: &'static str) -> impl FnOnce(candle_core::Error) -> UseError {
    move |error| {
        UseError::new(
            "use.ocr.runtime_failed",
            format!("Failed to {action} in the Unlimited-OCR Rust model: {error}"),
        )
    }
}

pub(super) fn model_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.model_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_resize_preserves_constants() {
        let input = Tensor::from_vec(vec![3.0_f32; 4 * 4], (1, 1, 4, 4), &Device::Cpu).unwrap();
        let output = resize_spatial_cubic(&input, 3).unwrap();
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| (value - 3.0).abs() < 1e-5));
    }

    #[test]
    fn cubic_resize_uses_pillow_antialias_boundary_weights() {
        let input =
            Tensor::from_vec(vec![0.0_f32, 1.0, 2.0, 3.0], (1, 1, 2, 2), &Device::Cpu).unwrap();
        let output = resize_spatial_cubic(&input, 3).unwrap();
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = [
            -0.197_368_43,
            0.368_421_05,
            0.934_210_54,
            0.934_210_54,
            1.5,
            2.065_789_5,
            2.065_789_5,
            2.631_579,
            3.197_368_4,
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }

        let input = Tensor::from_vec(
            (0..16).map(|value| value as f32).collect::<Vec<_>>(),
            (1, 1, 4, 4),
            &Device::Cpu,
        )
        .unwrap();
        let output = resize_spatial_cubic(&input, 3).unwrap();
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = [
            0.949_656_8,
            2.259_725_3,
            3.569_794,
            6.189_931_4,
            7.5,
            8.810_069,
            11.430_205,
            12.740_274,
            14.050_343_5,
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn linear_resize_keeps_endpoints_bounded() {
        let input = Tensor::from_vec(vec![0.0_f32, 10.0], (2, 1), &Device::Cpu).unwrap();
        let output = resize_linear_1d(&input, 3).unwrap();
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0, 5.0, 10.0]
        );
    }
}
