use crate::{
    OcrBlock, OcrBoundingBox, OcrEvidenceId, OcrImageCanvas, OcrPoint, OcrProviderOutput,
    OcrStageEvidence, OcrStageOutcome, OcrTableCellEvidence, OcrTableEvidence, OcrTableKind,
    OcrTableStageEvidence, OcrVisualRegion,
};

use super::super::{OcrBatchSlotId, OcrProviderBatchSlotOutput};

#[test]
fn exact_table_text_block_reference_contract_is_enforced() {
    let valid = slot("value", "value", true);
    super::super::client::validate_table_text_block_references(&valid).unwrap();

    let mismatched = slot("value", "different", true);
    assert_invalid(&mismatched);

    let missing_geometry = slot("value", "value", false);
    assert_invalid(&missing_geometry);

    let mut out_of_range = valid.clone();
    table_cell_mut(&mut out_of_range).source_text_block_indices = vec![1];
    assert_invalid(&out_of_range);

    let mut duplicate_owner = valid;
    let second = OcrTableCellEvidence {
        id: evidence_id("cell-2"),
        row_index: 0,
        column_index: 1,
        row_span: 1,
        column_span: 1,
        text: Some("value".to_string()),
        source_text_block_indices: vec![0],
        region: Some(region(500, 100, 400, 200)),
    };
    table_mut(&mut duplicate_owner).cells.push(second);
    assert_invalid(&duplicate_owner);
}

fn slot(cell_text: &str, block_text: &str, with_geometry: bool) -> OcrProviderBatchSlotOutput {
    OcrProviderBatchSlotOutput {
        slot_id: OcrBatchSlotId::new("page-1").unwrap(),
        stages: vec![OcrStageOutcome::completed_with_evidence(
            OcrStageEvidence::Table(OcrTableStageEvidence {
                canvas: OcrImageCanvas::new(1_000, 1_000).unwrap(),
                tables: vec![OcrTableEvidence {
                    id: evidence_id("table-1"),
                    kind: OcrTableKind::Wired,
                    region: region(100, 100, 800, 200),
                    row_count: Some(1),
                    column_count: Some(1),
                    cells: vec![OcrTableCellEvidence {
                        id: evidence_id("cell-1"),
                        row_index: 0,
                        column_index: 0,
                        row_span: 1,
                        column_span: 1,
                        text: Some(cell_text.to_string()),
                        source_text_block_indices: vec![0],
                        region: Some(region(100, 100, 800, 200)),
                    }],
                }],
            }),
        )],
        output: Some(OcrProviderOutput {
            model: None,
            text: block_text.to_string(),
            blocks: vec![OcrBlock {
                page: 1,
                text: block_text.to_string(),
                category: None,
                confidence: Some(0.9),
                detection_confidence: Some(0.9),
                text_rotation_millidegrees: None,
                polygon: None,
                bounding_box: with_geometry.then_some(OcrBoundingBox {
                    x: 120,
                    y: 140,
                    width: 200,
                    height: 40,
                }),
                bounding_boxes: Vec::new(),
            }],
            execution_receipts: Vec::new(),
            warnings: Vec::new(),
        }),
    }
}

fn table_cell_mut(slot: &mut OcrProviderBatchSlotOutput) -> &mut OcrTableCellEvidence {
    &mut table_mut(slot).cells[0]
}

fn table_mut(slot: &mut OcrProviderBatchSlotOutput) -> &mut OcrTableEvidence {
    let OcrStageEvidence::Table(stage) = slot.stages[0].evidence.as_mut().unwrap() else {
        unreachable!()
    };
    &mut stage.tables[0]
}

fn assert_invalid(slot: &OcrProviderBatchSlotOutput) {
    assert_eq!(
        super::super::client::validate_table_text_block_references(slot)
            .unwrap_err()
            .code,
        "use.ocr.provider_batch_invalid"
    );
}

fn evidence_id(value: &str) -> OcrEvidenceId {
    OcrEvidenceId::new(value).unwrap()
}

fn region(x: u32, y: u32, width: u32, height: u32) -> OcrVisualRegion {
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
        confidence: Some(0.9),
    }
}
