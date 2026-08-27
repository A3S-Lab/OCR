use std::collections::BTreeSet;

use a3s_use_core::{UseError, UseResult};

use super::decoder::StructureGrid;
use super::stage::DetectedPage;
use super::wired::PixelRect;
use crate::{
    OcrBlock, OcrBoundingBox, OcrEvidenceId, OcrPoint, OcrProviderOutput, OcrStageEvidence,
    OcrTableCellEvidence, OcrTableEvidence, OcrTableKind, OcrTableStageEvidence, OcrVisualRegion,
};

pub(super) fn table_evidence(
    page: DetectedPage,
    text: Option<&OcrProviderOutput>,
) -> UseResult<(OcrStageEvidence, Vec<crate::OcrExecutionReceipt>)> {
    let blocks = text.map_or(&[][..], |output| output.blocks.as_slice());
    let DetectedPage {
        canvas,
        tables: detected_tables,
        receipts,
    } = page;
    for detected in &detected_tables {
        validate_complete_grid(&detected.grid)?;
    }
    let quads_by_table = detected_tables
        .iter()
        .map(|detected| {
            detected
                .grid
                .cells
                .iter()
                .map(|cell| cell.quad)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let text_by_table = table_cell_texts(blocks, &quads_by_table)?;
    let mut tables = Vec::with_capacity(detected_tables.len());
    for ((table_index, detected), text_by_cell) in
        detected_tables.into_iter().enumerate().zip(text_by_table)
    {
        let table_id = format!("table-{table_index:04}");
        let mut cells = Vec::with_capacity(detected.grid.cells.len());
        for ((cell_index, cell), assignment) in detected
            .grid
            .cells
            .into_iter()
            .enumerate()
            .zip(text_by_cell)
        {
            let region = cell.quad.map(quad_region).transpose()?;
            cells.push(OcrTableCellEvidence {
                id: OcrEvidenceId::new(format!("{table_id}:cell-{cell_index:04}"))?,
                row_index: cell.row,
                column_index: cell.column,
                row_span: cell.row_span,
                column_span: cell.column_span,
                text: assignment.text,
                source_text_block_indices: assignment.source_text_block_indices,
                region,
            });
        }
        tables.push(OcrTableEvidence {
            id: OcrEvidenceId::new(table_id)?,
            kind: OcrTableKind::Wired,
            region: rectangle_region(detected.region, detected.grid.confidence)?,
            row_count: Some(detected.grid.row_count),
            column_count: Some(detected.grid.column_count),
            cells,
        });
    }
    let evidence = OcrStageEvidence::Table(OcrTableStageEvidence { canvas, tables });
    evidence.validate()?;
    Ok((evidence, receipts))
}

/// Proves that the decoded cells form the declared rectangular table exactly.
///
/// This is a structural invariant, not a quality score: every logical slot
/// must be covered once, merged cells may cover several slots, and no cell may
/// escape or overlap the declared grid. Whether cells happen to contain text
/// cannot change the verdict.
fn validate_complete_grid(grid: &StructureGrid) -> UseResult<()> {
    if grid.row_count == 0 || grid.column_count == 0 || grid.cells.is_empty() {
        return Err(projection_error(
            "A table grid requires positive dimensions and at least one cell.",
        ));
    }
    let expected_slots = u64::from(grid.row_count)
        .checked_mul(u64::from(grid.column_count))
        .ok_or_else(|| projection_error("A table grid slot count overflowed."))?;
    let mut occupied = BTreeSet::new();
    for cell in &grid.cells {
        if cell.row_span == 0 || cell.column_span == 0 {
            return Err(projection_error("A table cell span must be positive."));
        }
        let row_end = cell
            .row
            .checked_add(cell.row_span)
            .ok_or_else(|| projection_error("A table cell row span overflowed."))?;
        let column_end = cell
            .column
            .checked_add(cell.column_span)
            .ok_or_else(|| projection_error("A table cell column span overflowed."))?;
        if row_end > grid.row_count || column_end > grid.column_count {
            return Err(projection_error("A table cell escaped the declared grid."));
        }
        for row in cell.row..row_end {
            for column in cell.column..column_end {
                if !occupied.insert((row, column)) {
                    return Err(projection_error("Table cell spans overlapped."));
                }
            }
        }
    }
    if u64::try_from(occupied.len()).ok() != Some(expected_slots) {
        return Err(projection_error(
            "Table cells did not cover every declared grid slot.",
        ));
    }
    Ok(())
}

fn rectangle_region(rectangle: PixelRect, confidence: Option<f32>) -> UseResult<OcrVisualRegion> {
    let right = rectangle
        .x
        .checked_add(rectangle.width)
        .ok_or_else(|| projection_error("A table rectangle overflowed its source canvas."))?;
    let bottom = rectangle
        .y
        .checked_add(rectangle.height)
        .ok_or_else(|| projection_error("A table rectangle overflowed its source canvas."))?;
    Ok(OcrVisualRegion {
        bounding_box: OcrBoundingBox {
            x: rectangle.x,
            y: rectangle.y,
            width: rectangle.width,
            height: rectangle.height,
        },
        polygon: vec![
            OcrPoint {
                x: rectangle.x,
                y: rectangle.y,
            },
            OcrPoint {
                x: right,
                y: rectangle.y,
            },
            OcrPoint {
                x: right,
                y: bottom,
            },
            OcrPoint {
                x: rectangle.x,
                y: bottom,
            },
        ],
        confidence,
    })
}

fn quad_region(quad: [u32; 8]) -> UseResult<OcrVisualRegion> {
    let points = quad
        .chunks_exact(2)
        .map(|coordinates| OcrPoint {
            x: coordinates[0],
            y: coordinates[1],
        })
        .collect::<Vec<_>>();
    let left = points.iter().map(|point| point.x).min().ok_or_else(|| {
        projection_error("A SLANet-Plus cell quad did not contain x coordinates.")
    })?;
    let right = points.iter().map(|point| point.x).max().ok_or_else(|| {
        projection_error("A SLANet-Plus cell quad did not contain x coordinates.")
    })?;
    let top = points.iter().map(|point| point.y).min().ok_or_else(|| {
        projection_error("A SLANet-Plus cell quad did not contain y coordinates.")
    })?;
    let bottom = points.iter().map(|point| point.y).max().ok_or_else(|| {
        projection_error("A SLANet-Plus cell quad did not contain y coordinates.")
    })?;
    Ok(OcrVisualRegion {
        bounding_box: OcrBoundingBox {
            x: left,
            y: top,
            width: right.saturating_sub(left),
            height: bottom.saturating_sub(top),
        },
        polygon: points,
        confidence: None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CellTextAssignment {
    text: Option<String>,
    source_text_block_indices: Vec<u32>,
}

fn table_cell_texts(
    blocks: &[OcrBlock],
    quads_by_table: &[Vec<Option<[u32; 8]>>],
) -> UseResult<Vec<Vec<CellTextAssignment>>> {
    let flat_quads = quads_by_table
        .iter()
        .flat_map(|quads| quads.iter().copied())
        .collect::<Vec<_>>();
    let mut assignments = cell_texts(blocks, &flat_quads)?.into_iter();
    let grouped = quads_by_table
        .iter()
        .map(|quads| assignments.by_ref().take(quads.len()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if assignments.next().is_some()
        || grouped
            .iter()
            .zip(quads_by_table)
            .any(|(assignments, quads)| assignments.len() != quads.len())
    {
        return Err(projection_error(
            "Table text assignment changed the declared cell cardinality.",
        ));
    }
    Ok(grouped)
}

fn cell_texts(
    blocks: &[OcrBlock],
    quads: &[Option<[u32; 8]>],
) -> UseResult<Vec<CellTextAssignment>> {
    let mut matched = vec![Vec::<(u32, &str)>::new(); quads.len()];
    for (block_index, block) in blocks.iter().enumerate() {
        if block.text.is_empty() {
            continue;
        }
        let Some(vertices) = block_vertices(block) else {
            continue;
        };
        let mut containers = quads.iter().enumerate().filter_map(|(index, quad)| {
            quad.filter(|quad| {
                vertices
                    .iter()
                    .all(|point| point_inside_or_on_quad(*point, *quad))
            })
            .map(|_| index)
        });
        let Some(index) = containers.next() else {
            continue;
        };
        if containers.next().is_none() {
            let block_index = u32::try_from(block_index).map_err(|_| {
                projection_error("A table text-block index exceeded its published range.")
            })?;
            matched[index].push((block_index, block.text.as_str()));
        }
    }
    Ok(matched
        .into_iter()
        .map(|pieces| CellTextAssignment {
            text: (!pieces.is_empty()).then(|| {
                pieces
                    .iter()
                    .map(|(_, text)| *text)
                    .collect::<Vec<_>>()
                    .join(" ")
            }),
            source_text_block_indices: pieces.iter().map(|(index, _)| *index).collect(),
        })
        .collect())
}

fn block_vertices(block: &OcrBlock) -> Option<[OcrPoint; 4]> {
    if let Some(polygon) = block.polygon {
        return nonzero_polygon(polygon).then_some(polygon);
    }
    let bounds = block.bounding_box?;
    if bounds.width == 0 || bounds.height == 0 {
        return None;
    }
    let right = bounds.x.checked_add(bounds.width)?;
    let bottom = bounds.y.checked_add(bounds.height)?;
    Some([
        OcrPoint {
            x: bounds.x,
            y: bounds.y,
        },
        OcrPoint {
            x: right,
            y: bounds.y,
        },
        OcrPoint {
            x: right,
            y: bottom,
        },
        OcrPoint {
            x: bounds.x,
            y: bottom,
        },
    ])
}

fn nonzero_polygon(polygon: [OcrPoint; 4]) -> bool {
    polygon
        .iter()
        .zip(polygon.iter().cycle().skip(1))
        .take(polygon.len())
        .map(|(first, second)| {
            i128::from(first.x) * i128::from(second.y) - i128::from(second.x) * i128::from(first.y)
        })
        .sum::<i128>()
        != 0
}

fn point_inside_or_on_quad(point: OcrPoint, quad: [u32; 8]) -> bool {
    let vertices = [
        OcrPoint {
            x: quad[0],
            y: quad[1],
        },
        OcrPoint {
            x: quad[2],
            y: quad[3],
        },
        OcrPoint {
            x: quad[4],
            y: quad[5],
        },
        OcrPoint {
            x: quad[6],
            y: quad[7],
        },
    ];
    let mut sign = 0_i8;
    for index in 0..vertices.len() {
        let first = vertices[index];
        let second = vertices[(index + 1) % vertices.len()];
        let cross = (i128::from(second.x) - i128::from(first.x))
            * (i128::from(point.y) - i128::from(first.y))
            - (i128::from(second.y) - i128::from(first.y))
                * (i128::from(point.x) - i128::from(first.x));
        if cross == 0 {
            continue;
        }
        let current = if cross > 0 { 1 } else { -1 };
        if sign != 0 && sign != current {
            return false;
        }
        sign = current;
    }
    sign != 0
}

fn projection_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.table_model_output_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document_fast::decoder::GridCell;

    #[test]
    fn convex_quad_contains_interior_and_boundary_points() {
        let quad = [10, 10, 110, 20, 100, 80, 20, 70];
        assert!(point_inside_or_on_quad(OcrPoint { x: 60, y: 45 }, quad));
        assert!(point_inside_or_on_quad(OcrPoint { x: 10, y: 10 }, quad));
        assert!(!point_inside_or_on_quad(OcrPoint { x: 5, y: 45 }, quad));
    }

    #[test]
    fn unique_geometric_container_assigns_a_text_block_once() {
        let blocks = vec![block("cell value", 20, 20, 20, 10)];
        let matched = cell_texts(
            &blocks,
            &[
                Some([0, 0, 100, 0, 100, 100, 0, 100]),
                Some([100, 0, 200, 0, 200, 100, 100, 100]),
            ],
        )
        .unwrap();
        assert_eq!(matched[0].text.as_deref(), Some("cell value"));
        assert_eq!(matched[0].source_text_block_indices, [0]);
        assert_eq!(matched[1].text, None);
        assert!(matched[1].source_text_block_indices.is_empty());
    }

    #[test]
    fn ambiguous_or_cross_cell_text_remains_unassigned() {
        let quad = [0, 0, 100, 0, 100, 100, 0, 100];
        let duplicated = cell_texts(
            &[block("ambiguous", 20, 20, 20, 10)],
            &[Some(quad), Some(quad)],
        )
        .unwrap();
        assert!(duplicated.iter().all(|assignment| assignment.text.is_none()
            && assignment.source_text_block_indices.is_empty()));

        let crossing = cell_texts(
            &[block("boundary value", 90, 20, 20, 10)],
            &[Some(quad), Some([100, 0, 200, 0, 200, 100, 100, 100])],
        )
        .unwrap();
        assert!(crossing.iter().all(|assignment| assignment.text.is_none()
            && assignment.source_text_block_indices.is_empty()));
    }

    #[test]
    fn text_ownership_is_unique_across_tables_on_one_canvas() {
        let quad = Some([0, 0, 100, 0, 100, 100, 0, 100]);
        let matched = table_cell_texts(
            &[block("shared geometry", 20, 20, 20, 10)],
            &[vec![quad], vec![quad]],
        )
        .unwrap();

        assert!(matched.iter().flatten().all(|assignment| {
            assignment.text.is_none() && assignment.source_text_block_indices.is_empty()
        }));
    }

    #[test]
    fn text_projection_preserves_recognized_block_bytes() {
        let blocks = [
            block("  first  value  ", 10, 10, 20, 10),
            block("second   value", 10, 30, 20, 10),
        ];
        let matched = cell_texts(&blocks, &[Some([0, 0, 100, 0, 100, 100, 0, 100])]).unwrap();
        assert_eq!(
            matched[0].text.as_deref(),
            Some("  first  value   second   value")
        );
        assert_eq!(matched[0].source_text_block_indices, [0, 1]);
    }

    #[test]
    fn complete_grid_accepts_merged_cells_and_rejects_holes() {
        let complete = StructureGrid {
            row_count: 2,
            column_count: 2,
            cells: vec![
                grid_cell(0, 0, 1, 2),
                grid_cell(1, 0, 1, 1),
                grid_cell(1, 1, 1, 1),
            ],
            confidence: None,
        };
        validate_complete_grid(&complete).unwrap();

        let mut incomplete = complete;
        incomplete.cells.pop();
        let error = validate_complete_grid(&incomplete).unwrap_err();
        assert_eq!(error.code, "use.ocr.table_model_output_invalid");
        assert!(error.message.contains("every declared grid slot"));
    }

    fn block(text: &str, x: u32, y: u32, width: u32, height: u32) -> OcrBlock {
        OcrBlock {
            page: 1,
            text: text.to_string(),
            category: None,
            confidence: Some(0.9),
            detection_confidence: Some(0.9),
            text_rotation_millidegrees: None,
            polygon: None,
            bounding_box: Some(OcrBoundingBox {
                x,
                y,
                width,
                height,
            }),
            bounding_boxes: Vec::new(),
        }
    }

    fn grid_cell(row: u32, column: u32, row_span: u32, column_span: u32) -> GridCell {
        GridCell {
            row,
            column,
            row_span,
            column_span,
            quad: None,
        }
    }
}
