use crate::{
    OcrBatchSlotId, OcrBlock, OcrBoundingBox, OcrEvidenceId, OcrImageCanvas,
    OcrLayoutRegionEvidence, OcrLayoutRole, OcrLayoutStageEvidence, OcrProviderBatchSlotOutput,
    OcrProviderOutput, OcrStageEvidence, OcrStageOutcome, OcrVisualRegion,
};

#[test]
fn layout_text_ownership_requires_unique_in_range_center_containment() {
    let valid = slot(
        vec![0],
        OcrBoundingBox {
            x: 100,
            y: 100,
            width: 300,
            height: 100,
        },
    );
    super::super::client::validate_layout_text_block_references(&valid).unwrap();

    let outside = slot(
        vec![0],
        OcrBoundingBox {
            x: 0,
            y: 0,
            width: 50,
            height: 50,
        },
    );
    assert_invalid(&outside);

    let mut out_of_range = valid.clone();
    layout_mut(&mut out_of_range).source_text_block_indices = vec![1];
    assert_invalid(&out_of_range);

    let mut duplicate = valid;
    let mut second = layout_mut(&mut duplicate).clone();
    second.id = OcrEvidenceId::new("layout-2").unwrap();
    layout_stage_mut(&mut duplicate).regions.push(second);
    assert_invalid(&duplicate);
}

fn slot(references: Vec<u32>, layout_bounds: OcrBoundingBox) -> OcrProviderBatchSlotOutput {
    OcrProviderBatchSlotOutput {
        slot_id: OcrBatchSlotId::new("page-1").unwrap(),
        stages: vec![OcrStageOutcome::completed_with_evidence(
            OcrStageEvidence::Layout(OcrLayoutStageEvidence {
                canvas: OcrImageCanvas::new(1_000, 1_000).unwrap(),
                regions: vec![OcrLayoutRegionEvidence {
                    id: OcrEvidenceId::new("layout-1").unwrap(),
                    raw_label: "doc_title".to_string(),
                    role: OcrLayoutRole::Title,
                    region: OcrVisualRegion {
                        bounding_box: layout_bounds,
                        polygon: Vec::new(),
                        confidence: Some(0.9),
                    },
                    source_text_block_indices: references,
                }],
            }),
        )],
        output: Some(OcrProviderOutput {
            model: None,
            text: "title".to_string(),
            blocks: vec![OcrBlock {
                page: 1,
                text: "title".to_string(),
                category: None,
                confidence: Some(0.9),
                detection_confidence: Some(0.9),
                text_rotation_millidegrees: None,
                polygon: None,
                bounding_box: Some(OcrBoundingBox {
                    x: 180,
                    y: 130,
                    width: 80,
                    height: 20,
                }),
                bounding_boxes: Vec::new(),
            }],
            execution_receipts: Vec::new(),
            warnings: Vec::new(),
        }),
    }
}

fn layout_stage_mut(slot: &mut OcrProviderBatchSlotOutput) -> &mut OcrLayoutStageEvidence {
    let OcrStageEvidence::Layout(stage) = slot.stages[0].evidence.as_mut().unwrap() else {
        unreachable!()
    };
    stage
}

fn layout_mut(slot: &mut OcrProviderBatchSlotOutput) -> &mut OcrLayoutRegionEvidence {
    &mut layout_stage_mut(slot).regions[0]
}

fn assert_invalid(slot: &OcrProviderBatchSlotOutput) {
    assert_eq!(
        super::super::client::validate_layout_text_block_references(slot)
            .unwrap_err()
            .code,
        "use.ocr.provider_batch_invalid"
    );
}
