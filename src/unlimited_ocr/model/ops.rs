use a3s_use_core::{UseError, UseResult};
use candle_core::{DType, Tensor};

/// Apply a checkpoint linear layer to the final input dimension.
///
/// Candle's Metal matmul does not broadcast a two-dimensional weight across
/// arbitrary leading activation dimensions. Keep one model-wide implementation
/// that flattens tokens before matmul and restores the original leading shape.
pub(super) fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> UseResult<Tensor> {
    let Some((&input_width, leading_dims)) = input.dims().split_last() else {
        return Err(model_error(
            "Unlimited-OCR linear inputs must have at least one dimension.",
        ));
    };
    let (output_width, weight_input_width) = weight
        .dims2()
        .map_err(tensor_error("inspect a linear weight"))?;
    if input_width == 0 || input_width != weight_input_width {
        return Err(model_error(format!(
            "Unlimited-OCR linear input width {input_width} does not match weight width {weight_input_width}.",
        )));
    }
    let rows = leading_dims
        .iter()
        .try_fold(1_usize, |rows, dimension| rows.checked_mul(*dimension));
    let rows =
        rows.ok_or_else(|| model_error("Unlimited-OCR linear input dimensions overflowed."))?;
    let mut output_shape = leading_dims.to_vec();
    output_shape.push(output_width);
    let input = input
        .contiguous()
        .and_then(|value| value.reshape((rows, input_width)))
        .map_err(tensor_error("flatten a linear input"))?;
    let weight = weight
        .transpose(0, 1)
        .and_then(|value| value.contiguous())
        .map_err(tensor_error("transpose a linear weight"))?;
    let output = matmul(&input, &weight).map_err(tensor_error("execute a linear projection"))?;
    let output = match bias {
        Some(bias) => output
            .broadcast_add(bias)
            .map_err(tensor_error("add a linear bias")),
        None => Ok(output),
    }?;
    output
        .reshape(output_shape)
        .map_err(tensor_error("restore a linear output shape"))
}

/// Execute a matrix product while preserving reviewed BF16 rounding on CPU.
///
/// Candle 0.10 cannot execute CPU BF16 matmul directly. Promote only the
/// individual operation to f32, then restore its BF16 output boundary instead
/// of silently promoting the complete model and changing autoregressive logits.
pub(super) fn matmul(left: &Tensor, right: &Tensor) -> candle_core::Result<Tensor> {
    let promote =
        left.device().is_cpu() && left.dtype() == DType::BF16 && right.dtype() == DType::BF16;
    if !promote {
        return left.matmul(right);
    }
    left.to_dtype(DType::F32)?
        .matmul(&right.to_dtype(DType::F32)?)?
        .to_dtype(DType::BF16)
}

fn tensor_error(action: &'static str) -> impl FnOnce(candle_core::Error) -> UseError {
    move |error| {
        UseError::new(
            "use.ocr.runtime_failed",
            format!("Failed to {action} in the Unlimited-OCR Rust model: {error}"),
        )
    }
}

fn model_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.model_invalid", message)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn linear_preserves_arbitrary_leading_dimensions() {
        let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let input = Tensor::from_vec(values.clone(), (2, 2, 2, 3), &Device::Cpu).unwrap();
        let weight =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0], (2, 3), &Device::Cpu).unwrap();
        let bias = Tensor::from_vec(vec![0.5_f32, -0.5], 2, &Device::Cpu).unwrap();

        let output = linear(&input, &weight, Some(&bias)).unwrap();

        assert_eq!(output.dims(), &[2, 2, 2, 2]);
        let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = values
            .chunks_exact(3)
            .flat_map(|row| [row[0] + 0.5, row[1] + row[2] - 0.5])
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn cpu_bf16_matmul_uses_f32_compute_and_restores_bf16() {
        let left = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let right = Tensor::from_vec(vec![3.0_f32, 4.0], (2, 1), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let output = matmul(&left, &right).unwrap();

        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(
            output
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            [vec![11.0]]
        );
    }
}
