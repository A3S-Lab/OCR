use std::collections::BTreeSet;

use a3s_use_core::{UseError, UseResult};

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
    let mut tables = Vec::with_capacity(page.tables.len());
    for (table_index, detected) in page.tables.into_iter().enumerate() {
        let table_id = format!("table-{table_index:04}");
        let mut assigned_blocks = BTreeSet::new();
        let mut cells = Vec::with_capacity(detected.grid.cells.len());
        for (cell_index, cell) in detected.grid.cells.into_iter().enumerate() {
            let region = cell.quad.map(quad_region).transpose()?;
            let text = cell.quad.and_then(|quad| {
                cell_text(blocks, quad, &mut assigned_blocks).filter(|text| !text.is_empty())
            });
            cells.push(OcrTableCellEvidence {
                id: OcrEvidenceId::new(format!("{table_id}:cell-{cell_index:04}"))?,
                row_index: cell.row,
                column_index: cell.column,
                row_span: cell.row_span,
                column_span: cell.column_span,
                text,
                region,
            });
        }
        tables.push(OcrTableEvidence {
            id: OcrEvidenceId::new(table_id)?,
            kind: OcrTableKind::Wired,
            region: rectangle_region(detected.region, Some(detected.grid.confidence))?,
            row_count: Some(detected.grid.row_count),
            column_count: Some(detected.grid.column_count),
            cells,
        });
    }
    let evidence = OcrStageEvidence::Table(OcrTableStageEvidence {
        canvas: page.canvas,
        tables,
    });
    evidence.validate()?;
    Ok((evidence, page.receipts))
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

fn cell_text(
    blocks: &[OcrBlock],
    quad: [u32; 8],
    assigned: &mut BTreeSet<usize>,
) -> Option<String> {
    let mut matched = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if assigned.contains(&index)
            || !block_center(block).is_some_and(|point| point_inside_quad(point, quad))
        {
            continue;
        }
        assigned.insert(index);
        let text = block.text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !text.is_empty() {
            matched.push(text);
        }
    }
    (!matched.is_empty()).then(|| matched.join(" "))
}

fn block_center(block: &OcrBlock) -> Option<(f64, f64)> {
    if let Some(polygon) = block.polygon {
        let x = polygon.iter().map(|point| f64::from(point.x)).sum::<f64>() / 4.0;
        let y = polygon.iter().map(|point| f64::from(point.y)).sum::<f64>() / 4.0;
        return Some((x, y));
    }
    block.bounding_box.map(|bounds| {
        (
            f64::from(bounds.x) + f64::from(bounds.width) / 2.0,
            f64::from(bounds.y) + f64::from(bounds.height) / 2.0,
        )
    })
}

fn point_inside_quad(point: (f64, f64), quad: [u32; 8]) -> bool {
    let vertices = [
        (f64::from(quad[0]), f64::from(quad[1])),
        (f64::from(quad[2]), f64::from(quad[3])),
        (f64::from(quad[4]), f64::from(quad[5])),
        (f64::from(quad[6]), f64::from(quad[7])),
    ];
    let mut sign = 0_i8;
    for index in 0..vertices.len() {
        let first = vertices[index];
        let second = vertices[(index + 1) % vertices.len()];
        let cross =
            (second.0 - first.0) * (point.1 - first.1) - (second.1 - first.1) * (point.0 - first.0);
        if cross.abs() <= f64::EPSILON {
            continue;
        }
        let current = if cross > 0.0 { 1 } else { -1 };
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

    #[test]
    fn convex_quad_contains_cell_center_only() {
        let quad = [10, 10, 110, 20, 100, 80, 20, 70];
        assert!(point_inside_quad((60.0, 45.0), quad));
        assert!(!point_inside_quad((5.0, 45.0), quad));
    }

    #[test]
    fn matching_never_assigns_one_text_block_twice() {
        let blocks = vec![OcrBlock {
            page: 1,
            text: "cell value".to_string(),
            category: None,
            confidence: Some(0.9),
            detection_confidence: Some(0.9),
            polygon: None,
            bounding_box: Some(OcrBoundingBox {
                x: 20,
                y: 20,
                width: 20,
                height: 10,
            }),
            bounding_boxes: Vec::new(),
        }];
        let mut assigned = BTreeSet::new();
        let quad = [0, 0, 100, 0, 100, 100, 0, 100];
        assert_eq!(
            cell_text(&blocks, quad, &mut assigned).as_deref(),
            Some("cell value")
        );
        assert_eq!(cell_text(&blocks, quad, &mut assigned), None);
    }
}
