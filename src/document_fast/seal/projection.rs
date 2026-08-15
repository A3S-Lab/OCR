use a3s_use_core::UseResult;

use super::stage::DetectedSealPage;
use crate::{
    OcrBoundingBox, OcrCanvasEdge, OcrEvidenceId, OcrSealEvidence, OcrSealKind,
    OcrSealStageEvidence, OcrStageEvidence, OcrVisualRegion,
};

pub(in crate::document_fast) fn seal_evidence(
    page: DetectedSealPage,
) -> UseResult<(OcrStageEvidence, Vec<crate::OcrExecutionReceipt>)> {
    let seals = page
        .seals
        .into_iter()
        .enumerate()
        .map(|(index, seal)| {
            Ok(OcrSealEvidence {
                id: OcrEvidenceId::new(format!("seal-{}", index + 1))?,
                kind: OcrSealKind::Unknown,
                status: seal.status,
                region: OcrVisualRegion {
                    bounding_box: OcrBoundingBox {
                        x: seal.region.x,
                        y: seal.region.y,
                        width: seal.region.width,
                        height: seal.region.height,
                    },
                    polygon: Vec::new(),
                    confidence: Some(seal.confidence),
                },
                clipped_edges: seal
                    .clipped_edge
                    .into_iter()
                    .collect::<Vec<OcrCanvasEdge>>(),
                recognized_text: None,
                recognition_confidence: None,
            })
        })
        .collect::<UseResult<Vec<_>>>()?;
    Ok((
        OcrStageEvidence::Seal(OcrSealStageEvidence {
            canvas: page.canvas,
            seals,
        }),
        page.receipts,
    ))
}
