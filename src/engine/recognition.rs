use a3s_power::inference::{ExecutionPermit, ExecutionReceipt};
use a3s_use_core::{UseError, UseResult};
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use super::{engine_error, EngineBlock, EngineExtraction, PpOcrV6Engine};
use crate::cancellation::check_cancelled;
use crate::postprocess::{decode_ctc_top1, Detection, Recognition};
use crate::preprocess::{recognition_canvas_width, recognition_input};

mod crop;
mod planning;

use crop::PerspectiveCropPlan;
use planning::plan_width_batches;

struct RecognitionWorkItem {
    image_index: usize,
    detection_index: usize,
    detection: Detection,
    crop: PerspectiveCropPlan,
    canvas_width: u32,
}

enum ImageRecognition {
    Failed(UseError),
    Pending {
        blocks: Vec<Option<EngineBlock>>,
        receipts: Vec<ExecutionReceipt>,
    },
}

struct RecognizedBatch {
    items: Vec<UseResult<Recognition>>,
    receipt: ExecutionReceipt,
}

impl PpOcrV6Engine {
    pub(super) fn recognize_detected_batch(
        &self,
        images: &[&RgbImage],
        detections: Vec<UseResult<Vec<Detection>>>,
        detection_receipts: Vec<Vec<ExecutionReceipt>>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Vec<UseResult<EngineExtraction>>> {
        if detections.len() != images.len() || detection_receipts.len() != images.len() {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 recognition received mismatched image and detection cardinality.",
            ));
        }

        let (mut states, work) =
            self.prepare_recognition_work(images, detections, detection_receipts);
        let canvas_widths = work
            .iter()
            .map(|item| item.canvas_width)
            .collect::<Vec<_>>();
        for batch in plan_width_batches(&canvas_widths)? {
            check_cancelled(cancellation)?;
            let active = batch
                .into_iter()
                .filter(|index| states[work[*index].image_index].is_pending())
                .collect::<Vec<_>>();
            let crops = prepare_batch_crops(images, &work, active, &mut states);
            if crops.is_empty() {
                continue;
            }
            let crop_images = crops.iter().map(|(_, crop)| crop).collect::<Vec<_>>();
            match self.recognize_crop_batch(&crop_images, permit, cancellation) {
                Ok(recognized) => {
                    let work_indices = crops
                        .iter()
                        .map(|(work_index, _)| *work_index)
                        .collect::<Vec<_>>();
                    apply_recognized_batch(&work_indices, recognized, &work, &mut states)?;
                }
                Err(_) if crops.len() > 1 => {
                    check_cancelled(cancellation)?;
                    self.recognize_scalar_fallback(
                        &crops,
                        &work,
                        &mut states,
                        permit,
                        cancellation,
                    )?;
                }
                Err(error) => {
                    check_cancelled(cancellation)?;
                    states[work[crops[0].0].image_index].fail(error);
                }
            }
        }
        Ok(states.into_iter().map(ImageRecognition::finish).collect())
    }

    fn prepare_recognition_work(
        &self,
        images: &[&RgbImage],
        detections: Vec<UseResult<Vec<Detection>>>,
        detection_receipts: Vec<Vec<ExecutionReceipt>>,
    ) -> (Vec<ImageRecognition>, Vec<RecognitionWorkItem>) {
        let mut states = Vec::with_capacity(images.len());
        let mut work = Vec::new();
        for (image_index, (detections, receipts)) in
            detections.into_iter().zip(detection_receipts).enumerate()
        {
            let detections = match detections {
                Ok(detections) => detections,
                Err(error) => {
                    states.push(ImageRecognition::Failed(error));
                    continue;
                }
            };
            let mut image_work = Vec::with_capacity(detections.len());
            let mut error = None;
            for (detection_index, detection) in detections.iter().cloned().enumerate() {
                let crop = match PerspectiveCropPlan::new(&detection) {
                    Ok(crop) => crop,
                    Err(crop_error) => {
                        error = Some(crop_error);
                        break;
                    }
                };
                let (width, height) = crop.output_dimensions();
                let canvas_width =
                    match recognition_canvas_width(width, height, &self.recognition_config) {
                        Ok(width) => width,
                        Err(width_error) => {
                            error = Some(width_error);
                            break;
                        }
                    };
                image_work.push(RecognitionWorkItem {
                    image_index,
                    detection_index,
                    detection,
                    crop,
                    canvas_width,
                });
            }
            if let Some(error) = error {
                states.push(ImageRecognition::Failed(error));
                continue;
            }
            states.push(ImageRecognition::Pending {
                blocks: vec![None; detections.len()],
                receipts,
            });
            work.extend(image_work);
        }
        (states, work)
    }

    fn recognize_crop_batch(
        &self,
        crops: &[&RgbImage],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<RecognizedBatch> {
        check_cancelled(cancellation)?;
        let input = recognition_input(crops, &self.recognition_config)?;
        let recognition = self
            .native
            .recognize(input.data, input.shape, permit, cancellation)?;
        let shape = recognition.tensor.shape;
        let output = recognition.tensor.values;
        if shape.len() != 3 || shape[0] != crops.len() {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                format!(
                    "PP-OCRv6 recognition output shape must be [N, T, C] for N={}, found {shape:?}.",
                    crops.len()
                ),
            ));
        }
        let item_len = shape[1].checked_mul(shape[2]).ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 recognition output dimensions overflowed.",
            )
        })?;
        if output.len() != crops.len().saturating_mul(item_len) {
            return Err(engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 recognition output length does not match its batch shape.",
            ));
        }
        let items = (0..crops.len())
            .map(|index| {
                let start = index * item_len;
                decode_ctc_top1(
                    &output[start..start + item_len],
                    &[1, shape[1], shape[2]],
                    &self.recognition_config,
                )
            })
            .collect();
        Ok(RecognizedBatch {
            items,
            receipt: recognition.receipt,
        })
    }

    fn recognize_scalar_fallback(
        &self,
        crops: &[(usize, RgbImage)],
        work: &[RecognitionWorkItem],
        states: &mut [ImageRecognition],
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<()> {
        for (work_index, crop) in crops {
            check_cancelled(cancellation)?;
            let image_index = work[*work_index].image_index;
            if !states[image_index].is_pending() {
                continue;
            }
            match self.recognize_crop_batch(&[crop], permit, cancellation) {
                Ok(recognized) => apply_recognized_batch(&[*work_index], recognized, work, states)?,
                Err(error) => {
                    check_cancelled(cancellation)?;
                    states[image_index].fail(error);
                }
            }
        }
        Ok(())
    }
}

fn prepare_batch_crops(
    images: &[&RgbImage],
    work: &[RecognitionWorkItem],
    active: Vec<usize>,
    states: &mut [ImageRecognition],
) -> Vec<(usize, RgbImage)> {
    let mut crops = Vec::with_capacity(active.len());
    for work_index in active {
        let item = &work[work_index];
        match item.crop.execute(images[item.image_index]) {
            Ok(crop) => crops.push((work_index, crop)),
            Err(error) => states[item.image_index].fail(error),
        }
    }
    crops.retain(|(work_index, _)| states[work[*work_index].image_index].is_pending());
    crops
}

fn apply_recognized_batch(
    work_indices: &[usize],
    recognized: RecognizedBatch,
    work: &[RecognitionWorkItem],
    states: &mut [ImageRecognition],
) -> UseResult<()> {
    if work_indices.len() != recognized.items.len() {
        return Err(engine_error(
            "use.ocr.provider_output_invalid",
            "PP-OCRv6 changed recognition batch result cardinality.",
        ));
    }
    let mut receipt_images = Vec::with_capacity(work_indices.len());
    for (work_index, recognition) in work_indices.iter().zip(recognized.items) {
        let item = &work[*work_index];
        if !states[item.image_index].is_pending() {
            continue;
        }
        match recognition {
            Ok(recognition) => {
                states[item.image_index].set_block(
                    item.detection_index,
                    EngineBlock {
                        polygon: item.detection.polygon,
                        detection_confidence: item.detection.confidence,
                        text: recognition.text,
                        confidence: recognition.confidence,
                    },
                )?;
                if !receipt_images.contains(&item.image_index) {
                    receipt_images.push(item.image_index);
                }
            }
            Err(error) => states[item.image_index].fail(error),
        }
    }
    for image_index in receipt_images {
        if let ImageRecognition::Pending { receipts, .. } = &mut states[image_index] {
            receipts.push(recognized.receipt.clone());
        }
    }
    Ok(())
}

impl ImageRecognition {
    fn is_pending(&self) -> bool {
        matches!(self, Self::Pending { .. })
    }

    fn fail(&mut self, error: UseError) {
        *self = Self::Failed(error);
    }

    fn set_block(&mut self, index: usize, block: EngineBlock) -> UseResult<()> {
        let Self::Pending { blocks, .. } = self else {
            return Ok(());
        };
        let target = blocks.get_mut(index).ok_or_else(|| {
            engine_error(
                "use.ocr.provider_output_invalid",
                "PP-OCRv6 recognition block index escaped its source image.",
            )
        })?;
        *target = Some(block);
        Ok(())
    }

    fn finish(self) -> UseResult<EngineExtraction> {
        match self {
            Self::Failed(error) => Err(error),
            Self::Pending { blocks, receipts } => {
                let blocks = blocks
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        engine_error(
                            "use.ocr.provider_output_invalid",
                            "PP-OCRv6 left an unresolved recognition block.",
                        )
                    })?;
                Ok(EngineExtraction { blocks, receipts })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use a3s_power::inference::{ExecutionDigest, ExecutionReceipt, ModelIdentity, RuntimeIdentity};
    use imageproc::point::Point;

    use super::*;

    #[test]
    fn same_width_results_preserve_detection_order_and_share_one_receipt() {
        let work = vec![work_item(0, 0, 100.0), work_item(0, 1, 100.0)];
        let batches = plan_width_batches(&[320, 320]).unwrap();
        assert_eq!(batches, vec![vec![0, 1]]);
        let mut states = vec![pending_image(2)];

        apply_recognized_batch(
            &batches[0],
            RecognizedBatch {
                items: vec![recognized("first"), recognized("second")],
                receipt: receipt(),
            },
            &work,
            &mut states,
        )
        .unwrap();

        let output = states.pop().unwrap().finish().unwrap();
        assert_eq!(
            output
                .blocks
                .iter()
                .map(|block| block.text.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
        assert_eq!(output.receipts.len(), 1);
    }

    #[test]
    fn recognition_decode_failures_are_isolated_to_their_source_image() {
        let work = vec![work_item(0, 0, 100.0), work_item(1, 0, 100.0)];
        let mut states = vec![pending_image(1), pending_image(1)];

        apply_recognized_batch(
            &[0, 1],
            RecognizedBatch {
                items: vec![
                    Err(engine_error("use.ocr.decode_failed", "invalid CTC output")),
                    recognized("healthy"),
                ],
                receipt: receipt(),
            },
            &work,
            &mut states,
        )
        .unwrap();

        let mut outputs = states.into_iter().map(ImageRecognition::finish);
        let Err(failed) = outputs.next().unwrap() else {
            panic!("the malformed recognition item must fail its source image");
        };
        assert_eq!(failed.code, "use.ocr.decode_failed");
        let healthy = outputs.next().unwrap().unwrap();
        assert_eq!(healthy.blocks[0].text, "healthy");
        assert_eq!(healthy.receipts, vec![receipt()]);
    }

    fn pending_image(blocks: usize) -> ImageRecognition {
        ImageRecognition::Pending {
            blocks: vec![None; blocks],
            receipts: Vec::new(),
        }
    }

    fn work_item(image_index: usize, detection_index: usize, width: f32) -> RecognitionWorkItem {
        let detection = Detection {
            polygon: [
                Point::new(0.0, 0.0),
                Point::new(width, 0.0),
                Point::new(width, 20.0),
                Point::new(0.0, 20.0),
            ],
            confidence: 0.9,
        };
        RecognitionWorkItem {
            image_index,
            detection_index,
            crop: PerspectiveCropPlan::new(&detection).unwrap(),
            detection,
            canvas_width: width as u32,
        }
    }

    fn recognized(text: &str) -> UseResult<Recognition> {
        Ok(Recognition {
            text: text.to_string(),
            confidence: 0.95,
        })
    }

    fn receipt() -> ExecutionReceipt {
        ExecutionReceipt {
            schema: ExecutionReceipt::SCHEMA.to_string(),
            model: ModelIdentity::new("pp-ocr-v6-small-recognition", "fixture", "0".repeat(64)),
            runtime: RuntimeIdentity {
                name: "a3s-power-native".to_string(),
                version: "fixture".to_string(),
                device: "cpu".to_string(),
            },
            input: ExecutionDigest::utf8_text("input"),
            output: ExecutionDigest::utf8_text("output"),
            accelerator: None,
            microbatch: None,
        }
    }
}
