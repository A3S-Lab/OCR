use a3s_power::inference::{ExecutionDigest, ExecutionPermit, TensorInput};
use a3s_use_core::{UseError, UseResult};
use tokio_util::sync::CancellationToken;

use crate::preprocess::RecognitionInput;

use super::{power_error, projection, NativeGraphOutput, NativePpOcrV6};

impl NativePpOcrV6 {
    /// Executes an ordered recognition window without placing output or
    /// upload synchronization fences between its independent graph calls.
    /// Power retains the aggregate input/output bounds and exact device order.
    pub(crate) fn recognize_prepared_window(
        &self,
        inputs: Vec<RecognitionInput>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Vec<NativeGraphOutput>> {
        if inputs.is_empty() {
            return Err(UseError::new(
                "use.ocr.provider_input_invalid",
                "PP-OCRv6 recognition execution window requires at least one input tensor.",
            ));
        }
        let graph_count = inputs.len();
        let mut tensors = Vec::with_capacity(graph_count);
        let mut input_digests = Vec::with_capacity(graph_count);
        for input in inputs {
            if input.shape[0] == 0 || input.shape[1] != 3 || input.shape[2] != 48 {
                return Err(UseError::new(
                    "use.ocr.provider_input_invalid",
                    "PP-OCRv6 recognition window inputs must be non-empty NCHW tensors with three channels and height 48.",
                ));
            }
            input_digests.push(input.digest);
            tensors.push(
                TensorInput::new(input.shape.to_vec(), input.data, self.runtime.limits()).map_err(
                    |error| power_error("validate a prepared OCR execution window tensor", error),
                )?,
            );
        }

        let trace = std::env::var_os("A3S_OCR_TRACE_STAGE_TIMINGS").is_some();
        let started = std::time::Instant::now();
        let outputs = self
            .recognition
            .run_many_with_row_coalesced_terminal_matmul_bias_softmax_projection(
                tensors,
                permit,
                cancellation,
                projection::ctc_top1_from_classifier,
            )
            .map_err(|error| {
                power_error("execute the projected OCR recognition graph window", error)
            })?;
        let executed = started.elapsed();
        if outputs.len() != input_digests.len() {
            return Err(UseError::new(
                "use.ocr.provider_output_invalid",
                format!(
                    "PP-OCRv6 recognition execution window returned {} tensors for {} inputs.",
                    outputs.len(),
                    input_digests.len()
                ),
            ));
        }

        let results = outputs
            .into_iter()
            .zip(input_digests)
            .map(|(tensor, input_digest)| {
                let output_digest = ExecutionDigest::f32_tensor(&tensor.shape, &tensor.values);
                let receipt = self.runtime.receipt(
                    self.recognition_identity.clone(),
                    input_digest,
                    output_digest,
                );
                NativeGraphOutput { tensor, receipt }
            })
            .collect::<Vec<_>>();
        if trace {
            let completed = started.elapsed();
            eprintln!(
                "A3S_OCR_NATIVE_RECOGNITION_WINDOW_TIMING graphs={} graph_ms={:.3} receipt_ms={:.3} total_ms={:.3}",
                graph_count,
                executed.as_secs_f64() * 1_000.0,
                (completed - executed).as_secs_f64() * 1_000.0,
                completed.as_secs_f64() * 1_000.0,
            );
        }
        Ok(results)
    }
}
