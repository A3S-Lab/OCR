//! Exact PP-DocLayout-S execution and geometry-only text ownership.

mod assets;
mod decoder;
mod native;
mod preprocess;
mod profile;

use std::path::Path;
use std::sync::{Arc, Mutex};

use a3s_power::inference::InferenceLimits;
use a3s_use_core::{UseError, UseResult};
use rayon::prelude::*;
use tokio_util::sync::CancellationToken;

use self::assets::DocumentLayoutAssets;
use self::decoder::{area, intersection_area, DetectedLayoutRegion};
use self::native::NativeDocumentLayout;
use self::profile::{INPUT_ELEMENTS_PER_IMAGE, MAX_BATCH_SIZE, OUTPUT_WIDTH};
use super::page_orientation::SourceCanvasTransform;
use super::shared_decode::SharedDecodedImage;
use super::wired::PixelRect;
use crate::cancellation::{check_cancelled, run_blocking_with};
use crate::receipt::project_receipt;
use crate::{
    OcrBatchSlotId, OcrBoundingBox, OcrEvidenceId, OcrExecutionReceipt, OcrImageCanvas,
    OcrLayoutRegionEvidence, OcrLayoutStageEvidence, OcrPoint, OcrProviderBatchSlot,
    OcrProviderOutput, OcrStageEvidence, OcrVisualRegion,
};

#[derive(Clone)]
pub(super) struct DocumentLayoutRunner {
    engine: Arc<Mutex<NativeDocumentLayout>>,
    model_root: Arc<Path>,
}

impl DocumentLayoutRunner {
    pub(super) fn from_env_optional() -> UseResult<Option<Self>> {
        DocumentLayoutAssets::from_env_optional()?
            .map(Self::new)
            .transpose()
    }

    fn new(assets: DocumentLayoutAssets) -> UseResult<Self> {
        let model_root = Arc::<Path>::from(assets.model_root());
        Ok(Self {
            engine: Arc::new(Mutex::new(NativeDocumentLayout::load(&assets)?)),
            model_root,
        })
    }

    pub(super) fn model_root(&self) -> &Path {
        &self.model_root
    }

    pub(super) async fn run_decoded(
        &self,
        slots: Vec<OcrProviderBatchSlot>,
        images: Vec<SharedDecodedImage>,
        transforms: Vec<Option<SourceCanvasTransform>>,
        cancellation: CancellationToken,
    ) -> UseResult<DocumentLayoutBatch> {
        if slots.len() != images.len() || slots.len() != transforms.len() {
            return Err(layout_error(
                "The document-layout stage changed slot, image, or transform cardinality.",
            ));
        }
        let slot_ids = slots.into_iter().map(|slot| slot.slot_id).collect();
        let engine = Arc::clone(&self.engine);
        run_blocking_with(
            "document layout inference",
            cancellation.clone(),
            move |cancellation| {
                let engine = engine
                    .lock()
                    .map_err(|_| layout_error("The document-layout engine lock is unavailable."))?;
                run_batch(&engine, slot_ids, images, transforms, &cancellation)
            },
        )
        .await
    }
}

pub(super) struct DocumentLayoutBatch {
    pub(super) slots: Vec<DocumentLayoutSlot>,
    pub(super) receipts: Vec<OcrExecutionReceipt>,
}

pub(super) struct DocumentLayoutSlot {
    pub(super) slot_id: OcrBatchSlotId,
    pub(super) page: UseResult<DetectedLayoutPage>,
}

pub(super) struct DetectedLayoutPage {
    pub(super) canvas: OcrImageCanvas,
    pub(super) regions: Vec<DetectedLayoutRegion>,
}

fn run_batch(
    engine: &NativeDocumentLayout,
    slot_ids: Vec<OcrBatchSlotId>,
    images: Vec<SharedDecodedImage>,
    transforms: Vec<Option<SourceCanvasTransform>>,
    cancellation: &CancellationToken,
) -> UseResult<DocumentLayoutBatch> {
    check_cancelled(cancellation)?;
    let maximum_batch_size = maximum_batch_size(engine.limits())?;
    let mut pages = images
        .iter()
        .map(|image| image.as_ref().err().cloned().map_or(Ok(None), Err))
        .collect::<Vec<UseResult<Option<DetectedLayoutPage>>>>();
    let mut receipts = Vec::new();
    let admitted = images
        .iter()
        .enumerate()
        .filter_map(|(index, image)| image.as_ref().ok().map(|image| (index, Arc::clone(image))))
        .collect::<Vec<_>>();
    if !admitted.is_empty() {
        let permit = engine.begin(cancellation)?;
        for entries in admitted.chunks(maximum_batch_size) {
            check_cancelled(cancellation)?;
            let mut values = vec![0.0_f32; entries.len() * INPUT_ELEMENTS_PER_IMAGE];
            values
                .par_chunks_mut(INPUT_ELEMENTS_PER_IMAGE)
                .zip(entries.par_iter())
                .try_for_each(|(tensor, (_, image))| {
                    preprocess::image_tensor_into(image, tensor)
                })?;
            let output = engine.infer_batch(values, entries.len(), &permit, cancellation)?;
            let receipt = project_receipt(output.receipt);
            for ((slot_index, image), raw) in entries.iter().zip(
                output
                    .tensor
                    .values
                    .chunks_exact(super::layout::profile::LOCATION_COUNT * OUTPUT_WIDTH),
            ) {
                let transform = transforms[*slot_index];
                let source_canvas = match transform {
                    Some(transform) => transform.source_canvas()?,
                    None => OcrImageCanvas::new(image.width(), image.height())?,
                };
                let mut regions = decoder::decode(raw, image.width(), image.height())?;
                if let Some(transform) = transform {
                    for detected in &mut regions {
                        detected.region = transform.restore_pixel_rect(detected.region)?;
                    }
                }
                pages[*slot_index] = Ok(Some(DetectedLayoutPage {
                    canvas: source_canvas,
                    regions,
                }));
            }
            receipts.push(receipt);
        }
    }

    let slots = slot_ids
        .into_iter()
        .zip(pages)
        .map(|(slot_id, page)| DocumentLayoutSlot {
            slot_id,
            page: page.and_then(|page| {
                page.ok_or_else(|| layout_error("Document-layout output omitted an admitted slot."))
            }),
        })
        .collect();
    Ok(DocumentLayoutBatch { slots, receipts })
}

fn maximum_batch_size(limits: &InferenceLimits) -> UseResult<usize> {
    let input_bytes = INPUT_ELEMENTS_PER_IMAGE
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| layout_error("Document-layout input bytes overflowed."))?;
    let maximum = MAX_BATCH_SIZE
        .min(crate::batch::MAX_BATCH_SLOTS)
        .min(limits.max_input_bytes / input_bytes)
        .min(limits.max_tensor_elements / INPUT_ELEMENTS_PER_IMAGE);
    if maximum == 0 {
        return Err(layout_error(
            "Power limits cannot admit one PP-DocLayout-S input tensor.",
        ));
    }
    Ok(maximum)
}

pub(super) fn project_evidence(
    page: DetectedLayoutPage,
    output: Option<&OcrProviderOutput>,
) -> UseResult<OcrStageEvidence> {
    let mut memberships = vec![Vec::<u32>::new(); page.regions.len()];
    if let Some(output) = output {
        for (block_index, block) in output.blocks.iter().enumerate() {
            let Some(bounds) = block.bounding_box else {
                continue;
            };
            let block_region = PixelRect {
                x: bounds.x,
                y: bounds.y,
                width: bounds.width,
                height: bounds.height,
            };
            if let Some(region_index) = owning_region(block_region, &page.regions) {
                let block_index = u32::try_from(block_index).map_err(|_| {
                    layout_error("A document-layout text-block index exceeded u32 limits.")
                })?;
                memberships[region_index].push(block_index);
            }
        }
    }
    let regions = page
        .regions
        .into_iter()
        .zip(memberships)
        .enumerate()
        .map(|(index, (detected, source_text_block_indices))| {
            let id = OcrEvidenceId::new(format!("layout:{}", index + 1))?;
            Ok(OcrLayoutRegionEvidence {
                id,
                raw_label: detected.class.raw_label.to_string(),
                role: detected.class.role,
                region: visual_region(detected.region, detected.confidence),
                source_text_block_indices,
            })
        })
        .collect::<UseResult<Vec<_>>>()?;
    Ok(OcrStageEvidence::Layout(OcrLayoutStageEvidence {
        canvas: page.canvas,
        regions,
    }))
}

fn owning_region(block: PixelRect, regions: &[DetectedLayoutRegion]) -> Option<usize> {
    let center_x_twice = u64::from(block.x) * 2 + u64::from(block.width);
    let center_y_twice = u64::from(block.y) * 2 + u64::from(block.height);
    regions
        .iter()
        .enumerate()
        .filter(|(_, detected)| contains_center(detected.region, center_x_twice, center_y_twice))
        .max_by(|(left_index, left), (right_index, right)| {
            intersection_area(left.region, block)
                .cmp(&intersection_area(right.region, block))
                .then_with(|| area(right.region).cmp(&area(left.region)))
                .then_with(|| left.confidence.total_cmp(&right.confidence))
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
}

fn contains_center(region: PixelRect, x_twice: u64, y_twice: u64) -> bool {
    let right = u64::from(region.x.saturating_add(region.width)) * 2;
    let bottom = u64::from(region.y.saturating_add(region.height)) * 2;
    x_twice >= u64::from(region.x) * 2
        && x_twice <= right
        && y_twice >= u64::from(region.y) * 2
        && y_twice <= bottom
}

fn visual_region(region: PixelRect, confidence: f32) -> OcrVisualRegion {
    let right = region.x + region.width;
    let bottom = region.y + region.height;
    OcrVisualRegion {
        bounding_box: OcrBoundingBox {
            x: region.x,
            y: region.y,
            width: region.width,
            height: region.height,
        },
        polygon: vec![
            OcrPoint {
                x: region.x,
                y: region.y,
            },
            OcrPoint {
                x: right,
                y: region.y,
            },
            OcrPoint {
                x: right,
                y: bottom,
            },
            OcrPoint {
                x: region.x,
                y: bottom,
            },
        ],
        confidence: Some(confidence),
    }
}

fn layout_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.document_layout_failed", message)
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    #[test]
    #[ignore = "requires the exact reviewed PP-DocLayout-S runtime bundle"]
    fn reviewed_model_executes_through_power_and_decodes_its_raw_head() {
        let assets = DocumentLayoutAssets::from_env_optional()
            .unwrap()
            .expect("A3S_OCR_DOCUMENT_LAYOUT_MODEL_DIR must be configured");
        let engine = NativeDocumentLayout::load(&assets).unwrap();
        let image = RgbImage::from_pixel(960, 1_280, Rgb([255, 255, 255]));
        let mut values = vec![0.0; INPUT_ELEMENTS_PER_IMAGE];
        preprocess::image_tensor_into(&image, &mut values).unwrap();
        let cancellation = CancellationToken::new();
        let permit = engine.begin(&cancellation).unwrap();
        let output = engine
            .infer_batch(values, 1, &permit, &cancellation)
            .unwrap();
        assert_eq!(
            output.tensor.shape,
            [1, profile::LOCATION_COUNT, profile::OUTPUT_WIDTH]
        );
        assert_eq!(output.receipt.model.family, profile::FAMILY);
        let regions =
            decoder::decode(&output.tensor.values, image.width(), image.height()).unwrap();
        assert!(regions.len() <= profile::KEEP_TOP_K);
    }

    #[test]
    fn geometry_ownership_prefers_maximum_overlap_then_tighter_region() {
        let block = PixelRect {
            x: 100,
            y: 100,
            width: 100,
            height: 20,
        };
        let regions = vec![
            DetectedLayoutRegion {
                region: PixelRect {
                    x: 0,
                    y: 0,
                    width: 1_000,
                    height: 1_000,
                },
                class: profile::CLASSES[2],
                confidence: 0.99,
            },
            DetectedLayoutRegion {
                region: PixelRect {
                    x: 90,
                    y: 90,
                    width: 120,
                    height: 40,
                },
                class: profile::CLASSES[0],
                confidence: 0.8,
            },
        ];
        assert_eq!(owning_region(block, &regions), Some(1));
    }
}
