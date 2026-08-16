use super::*;
use crate::document_fast::decoder::GridCell;

#[test]
fn table_region_encompasses_wire_and_model_backed_cell_evidence() {
    let wire_region = PixelRect {
        x: 100,
        y: 200,
        width: 300,
        height: 400,
    };
    let grid = StructureGrid {
        row_count: 2,
        column_count: 1,
        cells: vec![
            GridCell {
                row: 0,
                column: 0,
                row_span: 1,
                column_span: 1,
                quad: Some([90, 180, 410, 180, 410, 300, 90, 300]),
            },
            GridCell {
                row: 1,
                column: 0,
                row_span: 1,
                column_span: 1,
                quad: Some([120, 300, 420, 300, 420, 630, 120, 630]),
            },
        ],
        confidence: 0.95,
    };

    assert_eq!(
        table_evidence_region(wire_region, &grid),
        PixelRect {
            x: 90,
            y: 180,
            width: 330,
            height: 450,
        }
    );
}

#[test]
fn table_region_ignores_cells_without_model_geometry() {
    let wire_region = PixelRect {
        x: 10,
        y: 20,
        width: 30,
        height: 40,
    };
    let grid = StructureGrid {
        row_count: 1,
        column_count: 1,
        cells: vec![GridCell {
            row: 0,
            column: 0,
            row_span: 1,
            column_span: 1,
            quad: None,
        }],
        confidence: 0.8,
    };

    assert_eq!(table_evidence_region(wire_region, &grid), wire_region);
}
