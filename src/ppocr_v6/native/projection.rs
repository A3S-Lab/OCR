use a3s_power::error::{PowerError, Result as PowerResult};
use candle_core::{DType, Tensor};

pub(super) const REVISION: &str = "ctc-matmul-bias-softmax-top1-last-tie-finite-v6";
pub(super) const IDENTITY: &[u8] =
    b"a3s-ocr-ppocr-v6-ctc-matmul-bias-softmax-top1-last-tie-finite-v6\0";

/// Applies the reviewed classifier to bounded `[..., F]` feature rows, adds
/// its bias, and projects to `[..., index/probability/finite]` on the
/// execution device. The CPU implementation retains only bounded classifier
/// tiles. Leading dimensions do not participate in classifier arithmetic, so
/// Power may safely coalesce rows from exact-shape graph prefixes.
pub(super) fn ctc_top1_from_classifier(
    features: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
) -> PowerResult<Tensor> {
    if features.dtype() != DType::F32 || weights.dtype() != DType::F32 || bias.dtype() != DType::F32
    {
        return Err(projection_error(
            "PP-OCRv6 classifier projection requires F32 features, weights, and bias",
        ));
    }
    let dimensions = features.dims();
    let Some(&feature_count) = dimensions.last().filter(|_| dimensions.len() >= 2) else {
        return Err(projection_error(
            "PP-OCRv6 classifier projection requires rank-two-or-higher feature rows",
        ));
    };
    let (weight_features, classes) = weights.dims2().map_err(candle_error)?;
    if dimensions[..dimensions.len() - 1].contains(&0)
        || feature_count == 0
        || feature_count != weight_features
        || classes == 0
        || classes > (1 << 24)
        || bias.dims() != [classes]
    {
        return Err(projection_error(
            "PP-OCRv6 classifier projection received incompatible bounded shapes",
        ));
    }

    a3s_power::inference::graph::row_matmul_bias_softmax_top1_last_finite(features, weights, bias)
}

/// Adds the reviewed classifier bias to `[N, T, C]` recognition logits and
/// projects the result to
/// `[N, T, index/probability/finite]` on the execution device.
///
/// Rust's scalar CTC decoder selects the last class when scores tie. Candle's
/// reductions select the first class, so the fused projection explicitly
/// preserves the reviewed last-tie behavior. The finite marker covers every
/// source logit, including values that were not selected.
#[cfg(test)]
pub(super) fn ctc_top1_from_unbiased_logits(logits: &Tensor, bias: &Tensor) -> PowerResult<Tensor> {
    if logits.dtype() != DType::F32 {
        return Err(projection_error(format!(
            "PP-OCRv6 recognition logits projection requires F32 input, found {:?}",
            logits.dtype()
        )));
    }
    let (batch, timesteps, classes) = logits.dims3().map_err(candle_error)?;
    if batch == 0 || timesteps == 0 || classes == 0 || classes > (1 << 24) {
        return Err(projection_error(
            "PP-OCRv6 recognition logits projection received an invalid bounded shape",
        ));
    }
    if bias.dtype() != DType::F32 || bias.dims() != [classes] {
        return Err(projection_error(
            "PP-OCRv6 recognition bias must have exact F32 [classes] shape",
        ));
    }

    a3s_power::inference::graph::row_bias_softmax_top1_last_finite(logits, bias)
}

fn candle_error(error: candle_core::Error) -> PowerError {
    projection_error(format!(
        "PP-OCRv6 recognition output projection failed: {error}"
    ))
}

fn projection_error(message: impl Into<String>) -> PowerError {
    PowerError::InferenceFailed(message.into())
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn projection_preserves_last_class_ties_and_source_finiteness() {
        let logits = Tensor::from_vec(
            vec![0.1_f32, 0.8, 0.8, 0.2, 0.9, f32::NAN, 0.1, 0.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let bias = Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap();

        let projected = ctc_top1_from_unbiased_logits(&logits, &bias)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(projected[0], 2.0);
        assert!(projected[1] > 0.0 && projected[1] < 1.0);
        assert_eq!(projected[2], 1.0);
        assert_eq!(projected[5], 0.0);
    }

    #[test]
    fn bounded_classifier_projection_matches_explicit_logits_bits() {
        let features = Tensor::from_vec(
            vec![0.1_f32, 0.8, 0.2, 0.9, -0.3, 0.1],
            (1, 2, 3),
            &Device::Cpu,
        )
        .unwrap();
        let weights = Tensor::from_vec(
            vec![
                0.5_f32, -0.2, 0.1, 0.7, -0.3, 0.4, 0.8, -0.6, 0.2, 0.9, -0.5, 0.3,
            ],
            (3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let bias = Tensor::new(&[0.2_f32, -0.1, 0.3, 0.0], &Device::Cpu).unwrap();
        let logits = features.broadcast_matmul(&weights).unwrap();
        let expected = ctc_top1_from_unbiased_logits(&logits, &bias)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = ctc_top1_from_classifier(&features, &weights, &bias)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let coalesced_rows = features.reshape((2, 3)).unwrap();
        let row_actual = ctc_top1_from_classifier(&coalesced_rows, &weights, &bias)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(row_actual, expected);
    }

    #[test]
    #[ignore = "requires an explicit CUDA build and device"]
    fn reviewed_recognition_shape_projects_on_cuda() {
        let device = Device::new_cuda(0).unwrap();
        let logits = Tensor::zeros((8, 40, 18_710), DType::F32, &device).unwrap();
        let bias = Tensor::zeros(18_710, DType::F32, &device).unwrap();

        let projected = ctc_top1_from_unbiased_logits(&logits, &bias).unwrap();
        let values = projected.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(projected.dims(), [8, 40, 3]);
        assert!(values.chunks_exact(3).all(|row| {
            row[0] == 18_709.0 && (row[1] - 1.0 / 18_710.0).abs() <= 1e-7 && row[2] == 1.0
        }));
    }
}
