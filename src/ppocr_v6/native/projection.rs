use a3s_power::error::{PowerError, Result as PowerResult};
use candle_core::{DType, Tensor};

pub(super) const REVISION: &str = "ctc-top1-last-tie-finite-v1";
pub(super) const IDENTITY: &[u8] = b"a3s-ocr-ppocr-v6-ctc-top1-last-tie-finite-v1\0";

/// Projects `[N, T, C]` recognition probabilities to
/// `[N, T, index/score/finite]` on the execution device.
///
/// Rust's scalar CTC decoder selects the last class when scores tie. Candle's
/// reductions select the first class, so reversing the class axis before
/// `argmax` preserves the reviewed scalar behavior. The finite marker covers
/// every source probability, including values that were not selected.
pub(super) fn ctc_top1(output: &Tensor) -> PowerResult<Tensor> {
    if output.dtype() != DType::F32 {
        return Err(projection_error(format!(
            "PP-OCRv6 recognition projection requires F32 input, found {:?}",
            output.dtype()
        )));
    }
    let (batch, timesteps, classes) = output.dims3().map_err(candle_error)?;
    if batch == 0 || timesteps == 0 || classes == 0 || classes > (1 << 24) {
        return Err(projection_error(
            "PP-OCRv6 recognition projection received an invalid bounded shape",
        ));
    }

    let reversed = output.flip(&[2]).map_err(candle_error)?;
    let reversed_indices = reversed.argmax_keepdim(2).map_err(candle_error)?;
    let scores = reversed
        .gather(&reversed_indices, 2)
        .map_err(candle_error)?;
    let indices = reversed_indices
        .to_dtype(DType::F32)
        .and_then(|indices| indices.affine(-1.0, (classes - 1) as f64))
        .map_err(candle_error)?;
    let finite = output
        .abs()
        .and_then(|values| values.le(f32::MAX))
        .and_then(|values| values.min_keepdim(2))
        .and_then(|values| values.to_dtype(DType::F32))
        .map_err(candle_error)?;
    Tensor::cat(&[&indices, &scores, &finite], 2).map_err(candle_error)
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
        let output = Tensor::from_vec(
            vec![0.1_f32, 0.8, 0.8, 0.2, 0.9, f32::NAN, 0.1, 0.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();

        let projected = ctc_top1(&output)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(projected[..3], [2.0, 0.8, 1.0]);
        assert_eq!(projected[3], 0.0);
        assert_eq!(projected[4], 0.9);
        assert_eq!(projected[5], 0.0);
    }

    #[test]
    #[ignore = "requires an explicit CUDA build and device"]
    fn reviewed_recognition_shape_projects_on_cuda() {
        let device = Device::new_cuda(0).unwrap();
        let output = Tensor::zeros((8, 40, 18_710), DType::F32, &device).unwrap();

        let projected = ctc_top1(&output).unwrap();
        let values = projected.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(projected.dims(), [8, 40, 3]);
        assert!(values
            .chunks_exact(3)
            .all(|row| row == [18_709.0, 0.0, 1.0]));
    }
}
