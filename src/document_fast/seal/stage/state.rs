use std::sync::Arc;

use a3s_use_core::{UseError, UseResult};
use image::RgbImage;

use super::{runtime_error, DetectedSealPage, SealSlotResult};
use crate::document_fast::page_orientation::SourceCanvasTransform;
use crate::document_fast::seal::decoder::{merge_page_detections, DecodedLayoutImage, DecodedSeal};
use crate::document_fast::seal::preprocess::SealView;
use crate::{OcrBatchSlotId, OcrExecutionReceipt, OcrImageCanvas, OcrSealDetectionStatus};

const MIN_COMPLETE_SEAL_OBSERVATIONS: u16 = 2;

pub(super) struct DecodedPage {
    pub(super) slot_id: OcrBatchSlotId,
    pub(super) adjacent_predecessor_slot_id: Option<OcrBatchSlotId>,
    pub(super) image: UseResult<Arc<RgbImage>>,
}

pub(super) struct PageAccumulator {
    pub(super) slot_id: OcrBatchSlotId,
    pub(super) adjacent_predecessor_slot_id: Option<OcrBatchSlotId>,
    pub(super) image: Option<Arc<RgbImage>>,
    pub(super) canvas: Option<OcrImageCanvas>,
    pub(super) seals: Vec<DecodedSeal>,
    pub(super) layout_images: Vec<DecodedLayoutImage>,
    pub(super) receipts: Vec<OcrExecutionReceipt>,
    error: Option<UseError>,
}

impl PageAccumulator {
    pub(super) fn from_decoded(decoded: DecodedPage) -> Self {
        match decoded.image {
            Ok(image) => match OcrImageCanvas::new(image.width(), image.height()) {
                Ok(canvas) => Self {
                    slot_id: decoded.slot_id,
                    adjacent_predecessor_slot_id: decoded.adjacent_predecessor_slot_id,
                    image: Some(image),
                    canvas: Some(canvas),
                    seals: Vec::new(),
                    layout_images: Vec::new(),
                    receipts: Vec::new(),
                    error: None,
                },
                Err(error) => {
                    Self::failed(decoded.slot_id, decoded.adjacent_predecessor_slot_id, error)
                }
            },
            Err(error) => {
                Self::failed(decoded.slot_id, decoded.adjacent_predecessor_slot_id, error)
            }
        }
    }

    fn failed(
        slot_id: OcrBatchSlotId,
        adjacent_predecessor_slot_id: Option<OcrBatchSlotId>,
        error: UseError,
    ) -> Self {
        Self {
            slot_id,
            adjacent_predecessor_slot_id,
            image: None,
            canvas: None,
            seals: Vec::new(),
            layout_images: Vec::new(),
            receipts: Vec::new(),
            error: Some(error),
        }
    }

    pub(super) fn add_seals(&mut self, seals: Vec<DecodedSeal>) {
        self.seals = merge_page_detections(std::mem::take(&mut self.seals), seals);
    }

    pub(super) fn add_layout_images(&mut self, images: Vec<DecodedLayoutImage>) {
        self.layout_images.extend(images);
    }

    pub(super) fn fail(&mut self, error: UseError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    pub(super) fn merge_oriented_supplemental(
        &mut self,
        mut supplemental: Self,
        transform: SourceCanvasTransform,
    ) -> UseResult<()> {
        if self.slot_id != supplemental.slot_id
            || self.adjacent_predecessor_slot_id != supplemental.adjacent_predecessor_slot_id
        {
            return Err(runtime_error(
                "An oriented seal supplement changed source slot identity or adjacency.",
            ));
        }
        let source_canvas = transform.source_canvas()?;
        if self.canvas != Some(source_canvas) {
            return Err(runtime_error(
                "An oriented seal supplement does not match its immutable source canvas.",
            ));
        }
        if supplemental.canvas != Some(transform.oriented_canvas()?) {
            return Err(runtime_error(
                "An oriented seal supplement does not match its admitted oriented canvas.",
            ));
        }
        self.receipts.append(&mut supplemental.receipts);
        if let Some(error) = supplemental.error.take() {
            self.fail(error);
            return Ok(());
        }
        for seal in &mut supplemental.seals {
            seal.region = transform.restore_pixel_rect(seal.region)?;
            seal.source_view = transform.restore_pixel_rect(seal.source_view)?;
            seal.clipped_edge = seal
                .clipped_edge
                .map(|edge| transform.restore_canvas_edge(edge));
        }
        for image in &mut supplemental.layout_images {
            image.region = transform.restore_pixel_rect(image.region)?;
            image.source_view = transform.restore_pixel_rect(image.source_view)?;
            image.clipped_edge = image
                .clipped_edge
                .map(|edge| transform.restore_canvas_edge(edge));
        }
        self.add_seals(supplemental.seals);
        self.add_layout_images(supplemental.layout_images);
        Ok(())
    }

    pub(super) fn finish(self) -> SealSlotResult {
        if std::env::var_os("A3S_OCR_TRACE_SEAL_DETECTIONS").is_some() {
            eprintln!(
                "A3S_OCR_SEAL_PAGE_CANDIDATES slot_id={:?} candidates={:?} layout_images={:?}",
                self.slot_id, self.seals, self.layout_images
            );
        }
        let page = match (self.error, self.canvas) {
            (Some(error), _) => Err(error),
            (None, Some(canvas)) => Ok(DetectedSealPage {
                canvas,
                seals: self
                    .seals
                    .into_iter()
                    .filter(|seal| {
                        seal.status == OcrSealDetectionStatus::BoundaryCandidate
                            || seal.independent_observations >= MIN_COMPLETE_SEAL_OBSERVATIONS
                    })
                    .collect(),
                receipts: self.receipts,
            }),
            (None, None) => Err(runtime_error(
                "A seal slot lost its exact source canvas during execution.",
            )),
        };
        SealSlotResult {
            slot_id: self.slot_id,
            page,
        }
    }
}

#[derive(Clone)]
pub(super) struct ViewReference {
    pub(super) page_index: usize,
    pub(super) image: Arc<RgbImage>,
    pub(super) view: SealView,
    pub(super) adjacent_boundary: Option<DecodedSeal>,
}

pub(super) struct PreparedBatch {
    pub(super) views: Vec<ViewReference>,
    pub(super) tensor: Vec<f32>,
}

pub(super) struct ViewResult {
    pub(super) page_index: usize,
    pub(super) seals: UseResult<Vec<DecodedSeal>>,
    pub(super) layout_images: UseResult<Vec<DecodedLayoutImage>>,
    pub(super) refinement_views: Vec<ViewReference>,
}

pub(super) struct BatchRun {
    pub(super) results: Vec<ViewResult>,
    pub(super) receipt: a3s_power::inference::ExecutionReceipt,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document_fast::wired::PixelRect;
    use crate::OcrCanvasEdge;

    fn seal(x: u32, independent_observations: u16, status: OcrSealDetectionStatus) -> DecodedSeal {
        DecodedSeal {
            region: PixelRect {
                x,
                y: 100,
                width: 100,
                height: 100,
            },
            source_view: PixelRect {
                x: 0,
                y: 0,
                width: 1_200,
                height: 1_600,
            },
            confidence: 0.8,
            class_log_likelihood_ratio: 1.0,
            independent_observations,
            clipped_edge: (status == OcrSealDetectionStatus::BoundaryCandidate)
                .then_some(OcrCanvasEdge::Right),
            status,
        }
    }

    #[test]
    fn complete_geometry_requires_two_independent_observations() {
        let mut page = PageAccumulator::from_decoded(DecodedPage {
            slot_id: OcrBatchSlotId::new("page-1").unwrap(),
            adjacent_predecessor_slot_id: None,
            image: Ok(Arc::new(RgbImage::new(1_200, 1_600))),
        });
        let unsupported = seal(100, 1, OcrSealDetectionStatus::Confirmed);
        let supported = seal(400, 2, OcrSealDetectionStatus::Confirmed);
        let boundary = seal(1_100, 1, OcrSealDetectionStatus::BoundaryCandidate);
        page.add_seals(vec![unsupported, supported, boundary]);

        let detected = page.finish().page.unwrap();
        assert_eq!(detected.seals.len(), 2);
        assert!(detected.seals.contains(&supported));
        assert!(detected.seals.contains(&boundary));
        assert!(!detected.seals.contains(&unsupported));
    }

    #[test]
    fn oriented_supplement_restores_geometry_without_replacing_source_canvas() {
        let slot_id = OcrBatchSlotId::new("page-1").unwrap();
        let mut source = PageAccumulator::from_decoded(DecodedPage {
            slot_id: slot_id.clone(),
            adjacent_predecessor_slot_id: None,
            image: Ok(Arc::new(RgbImage::new(300, 200))),
        });
        let mut oriented = PageAccumulator::from_decoded(DecodedPage {
            slot_id,
            adjacent_predecessor_slot_id: None,
            image: Ok(Arc::new(RgbImage::new(200, 300))),
        });
        oriented.add_seals(vec![DecodedSeal {
            region: PixelRect {
                x: 100,
                y: 250,
                width: 50,
                height: 50,
            },
            source_view: PixelRect {
                x: 0,
                y: 0,
                width: 200,
                height: 300,
            },
            confidence: 0.8,
            class_log_likelihood_ratio: 1.0,
            independent_observations: 2,
            clipped_edge: Some(OcrCanvasEdge::Bottom),
            status: OcrSealDetectionStatus::BoundaryCandidate,
        }]);

        source
            .merge_oriented_supplemental(oriented, SourceCanvasTransform::new(300, 200, 1).unwrap())
            .unwrap();

        let detected = source.finish().page.unwrap();
        assert_eq!(detected.canvas, OcrImageCanvas::new(300, 200).unwrap());
        assert_eq!(detected.seals.len(), 1);
        assert_eq!(
            detected.seals[0].region,
            PixelRect {
                x: 250,
                y: 50,
                width: 50,
                height: 50,
            }
        );
        assert_eq!(
            detected.seals[0].source_view,
            PixelRect {
                x: 0,
                y: 0,
                width: 300,
                height: 200,
            }
        );
        assert_eq!(detected.seals[0].clipped_edge, Some(OcrCanvasEdge::Right));
    }
}
