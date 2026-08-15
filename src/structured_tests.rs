use a3s_use_core::{Readiness, UseResult};
use async_trait::async_trait;

use crate::{
    OcrBatchRequest, OcrBatchResult, OcrBatchSlotId, OcrBatchSlotRequest, OcrBoundingBox,
    OcrCanvasEdge, OcrClient, OcrEvidenceId, OcrImageCanvas, OcrInput, OcrPoint, OcrProvider,
    OcrProviderBatchOutput, OcrProviderBatchRequest, OcrProviderBatchSlotOutput,
    OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus, OcrSealEvidence, OcrSealKind,
    OcrSealStageEvidence, OcrStage, OcrStageEvidence, OcrStageOutcome, OcrStageStatus,
    OcrTableCellEvidence, OcrTableEvidence, OcrTableKind, OcrTableStageEvidence, OcrVisualRegion,
};

#[test]
fn table_evidence_preserves_merged_cells_and_exact_geometry() {
    let evidence = OcrStageEvidence::Table(table_evidence());
    let outcome = OcrStageOutcome::completed_with_evidence(evidence.clone());

    outcome.validate().unwrap();
    let value = serde_json::to_value(&outcome).unwrap();
    assert_eq!(value["stage"], "table");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["evidence"]["evidenceType"], "table");
    assert_eq!(value["evidence"]["evidence"]["canvas"]["width"], 1_000);
    assert_eq!(
        value["evidence"]["evidence"]["tables"][0]["cells"][0]["columnSpan"],
        2
    );
    assert!(value["evidence"]["evidence"]["tables"][0]["cells"][0]
        .get("region")
        .is_none());
    assert_eq!(
        serde_json::from_value::<OcrStageOutcome>(value).unwrap(),
        outcome
    );
}

#[test]
fn seal_evidence_preserves_canvas_clipping_and_position() {
    let evidence = OcrStageEvidence::Seal(seal_evidence());
    let outcome = OcrStageOutcome::completed_with_evidence(evidence);

    outcome.validate().unwrap();
    let value = serde_json::to_value(&outcome).unwrap();
    let seal = &value["evidence"]["evidence"]["seals"][0];
    assert_eq!(seal["status"], "confirmed");
    assert_eq!(seal["clippedEdges"], serde_json::json!(["right"]));
    assert_eq!(seal["region"]["boundingBox"]["x"], 900);
    assert_eq!(seal["region"]["boundingBox"]["width"], 100);
}

#[test]
fn completed_structured_stages_require_matching_valid_evidence() {
    let missing = OcrStageOutcome::completed(OcrStage::Table);
    assert_structured_error(missing.validate().unwrap_err().code.as_str(), false);

    let mismatched = OcrStageOutcome {
        stage: OcrStage::Seal,
        status: OcrStageStatus::Completed,
        error: None,
        evidence: Some(OcrStageEvidence::Table(table_evidence())),
    };
    assert_eq!(
        mismatched.validate().unwrap_err().code,
        "use.ocr.provider_batch_invalid"
    );

    let legacy: OcrStageOutcome = serde_json::from_value(serde_json::json!({
        "stage": "text",
        "status": "completed"
    }))
    .unwrap();
    legacy.validate().unwrap();
    assert!(legacy.evidence.is_none());
}

#[test]
fn malformed_geometry_grid_and_clipping_are_rejected() {
    let mut bad_polygon = table_evidence();
    bad_polygon.tables[0].region.polygon[1].x = 899;
    bad_polygon.tables[0].region.polygon[2].x = 899;
    assert_structured_stage_error(OcrStageEvidence::Table(bad_polygon));

    let mut overlap = table_evidence();
    overlap.tables[0].cells[1].row_index = 0;
    assert_structured_stage_error(OcrStageEvidence::Table(overlap));

    let mut outside_grid = table_evidence();
    outside_grid.tables[0].cells[2].column_index = 2;
    assert_structured_stage_error(OcrStageEvidence::Table(outside_grid));

    let mut duplicate_id = table_evidence();
    duplicate_id.tables[0].cells[0].id = evidence_id("table-1");
    assert_structured_stage_error(OcrStageEvidence::Table(duplicate_id));

    let mut false_clip = seal_evidence();
    false_clip.seals[0].region.bounding_box.x = 899;
    false_clip.seals[0].region.bounding_box.width = 100;
    false_clip.seals[0].region.polygon.clear();
    assert_structured_stage_error(OcrStageEvidence::Seal(false_clip));

    let mut confidence_without_text = seal_evidence();
    confidence_without_text.seals[0].recognized_text = None;
    assert_structured_stage_error(OcrStageEvidence::Seal(confidence_without_text));

    let mut ungrounded_candidate = seal_evidence();
    ungrounded_candidate.seals[0].status = crate::OcrSealDetectionStatus::BoundaryCandidate;
    ungrounded_candidate.seals[0].clipped_edges.clear();
    assert_structured_stage_error(OcrStageEvidence::Seal(ungrounded_candidate));
}

#[tokio::test]
async fn batch_v2_carries_validated_table_and_seal_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let image = directory.path().join("page.bmp");
    std::fs::write(&image, b"BMstructured-fixture").unwrap();
    let client = OcrClient::with_provider(StructuredProvider).unwrap();

    let result = client
        .extract_batch(
            OcrBatchRequest::new(
                vec![OcrStage::Seal, OcrStage::Table],
                vec![OcrBatchSlotRequest::new(
                    OcrBatchSlotId::new("page-1").unwrap(),
                    image,
                )],
            )
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(result.schema, "a3s.ocr.staged-batch.v2");
    assert_eq!(result.schema, OcrBatchResult::SCHEMA);
    assert_eq!(
        result.requested_stages,
        vec![OcrStage::Table, OcrStage::Seal]
    );
    assert_eq!(result.slots[0].stages.len(), 2);
    assert!(result.slots[0]
        .stages
        .iter()
        .all(|outcome| outcome.evidence.is_some()));
    assert!(result.slots[0].result.is_none());
}

fn table_evidence() -> OcrTableStageEvidence {
    OcrTableStageEvidence {
        canvas: OcrImageCanvas::new(1_000, 2_000).unwrap(),
        tables: vec![OcrTableEvidence {
            id: evidence_id("table-1"),
            kind: OcrTableKind::Wired,
            region: region(100, 200, 800, 900, Some(0.98)),
            row_count: Some(2),
            column_count: Some(2),
            cells: vec![
                OcrTableCellEvidence {
                    id: evidence_id("table-1:cell-1"),
                    row_index: 0,
                    column_index: 0,
                    row_span: 1,
                    column_span: 2,
                    text: Some("Merged heading".to_string()),
                    region: None,
                },
                cell("table-1:cell-2", 1, 0, "left", 120),
                cell("table-1:cell-3", 1, 1, "right", 510),
            ],
        }],
    }
}

fn seal_evidence() -> OcrSealStageEvidence {
    OcrSealStageEvidence {
        canvas: OcrImageCanvas::new(1_000, 2_000).unwrap(),
        seals: vec![OcrSealEvidence {
            id: evidence_id("seal-1"),
            kind: OcrSealKind::Circular,
            status: crate::OcrSealDetectionStatus::Confirmed,
            region: region(900, 300, 100, 120, Some(0.94)),
            clipped_edges: vec![OcrCanvasEdge::Right],
            recognized_text: Some("ACME".to_string()),
            recognition_confidence: Some(0.91),
        }],
    }
}

fn cell(id: &str, row: u32, column: u32, text: &str, x: u32) -> OcrTableCellEvidence {
    OcrTableCellEvidence {
        id: evidence_id(id),
        row_index: row,
        column_index: column,
        row_span: 1,
        column_span: 1,
        text: Some(text.to_string()),
        region: Some(region(x, 700, 370, 250, None)),
    }
}

fn region(x: u32, y: u32, width: u32, height: u32, confidence: Option<f32>) -> OcrVisualRegion {
    OcrVisualRegion {
        bounding_box: OcrBoundingBox {
            x,
            y,
            width,
            height,
        },
        polygon: vec![
            OcrPoint { x, y },
            OcrPoint { x: x + width, y },
            OcrPoint {
                x: x + width,
                y: y + height,
            },
            OcrPoint { x, y: y + height },
        ],
        confidence,
    }
}

fn evidence_id(value: &str) -> OcrEvidenceId {
    OcrEvidenceId::new(value).unwrap()
}

fn assert_structured_stage_error(evidence: OcrStageEvidence) {
    assert_eq!(
        OcrStageOutcome::completed_with_evidence(evidence)
            .validate()
            .unwrap_err()
            .code,
        "use.ocr.structured_evidence_invalid"
    );
}

fn assert_structured_error(code: &str, structured: bool) {
    assert_eq!(
        code,
        if structured {
            "use.ocr.structured_evidence_invalid"
        } else {
            "use.ocr.provider_batch_invalid"
        }
    );
}

struct StructuredProvider;

#[async_trait]
impl OcrProvider for StructuredProvider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        OcrProviderDescriptor::new("structured-fixture", "fixture-engine", false)
            .unwrap()
            .with_stages(vec![OcrStage::Table, OcrStage::Seal])
            .unwrap()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        OcrProviderStatus {
            readiness: Readiness::Ready,
            model: None,
            model_dir: None,
            message: "ready".to_string(),
            suggestions: Vec::new(),
        }
    }

    async fn recognize(&self, _input: OcrInput) -> UseResult<OcrProviderOutput> {
        unreachable!("the structured fixture overrides recognize_batch")
    }

    async fn recognize_batch(
        &self,
        request: OcrProviderBatchRequest,
    ) -> UseResult<OcrProviderBatchOutput> {
        let slots = request
            .slots
            .into_iter()
            .map(|slot| OcrProviderBatchSlotOutput {
                slot_id: slot.slot_id,
                stages: vec![
                    OcrStageOutcome::completed_with_evidence(OcrStageEvidence::Table(
                        table_evidence(),
                    )),
                    OcrStageOutcome::completed_with_evidence(OcrStageEvidence::Seal(
                        seal_evidence(),
                    )),
                ],
                output: None,
            })
            .collect();
        Ok(OcrProviderBatchOutput {
            slots,
            execution_receipts: Vec::new(),
        })
    }
}
