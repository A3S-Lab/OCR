use std::collections::BTreeSet;

use a3s_use_core::{UseError, UseResult};

use super::{DecodedCell, DecodedStructure};

const MAX_CELLS: usize = 4_096;
const MAX_SPAN: u32 = 20;

#[derive(Debug, Clone)]
pub(crate) struct StructureGrid {
    pub(crate) row_count: u32,
    pub(crate) column_count: u32,
    pub(crate) cells: Vec<GridCell>,
    pub(crate) confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GridCell {
    pub(crate) row: u32,
    pub(crate) column: u32,
    pub(crate) row_span: u32,
    pub(crate) column_span: u32,
    pub(crate) quad: Option<[u32; 8]>,
}

pub(super) fn project(decoded: DecodedStructure) -> UseResult<StructureGrid> {
    let rows = parse_rows(&decoded.tokens, &decoded.cells)?;
    if rows.is_empty() || rows.iter().all(Vec::is_empty) {
        return Err(grid_error(
            "SLANet-Plus produced no complete table rows with cells.",
        ));
    }
    let mut cells: Vec<GridCell> = Vec::new();
    let mut column_count = 0_u32;
    let mut row_count = u32::try_from(rows.len())
        .map_err(|_| grid_error("SLANet-Plus row count cannot be represented."))?;
    for (row_index, row) in rows.into_iter().enumerate() {
        let row_index = u32::try_from(row_index)
            .map_err(|_| grid_error("SLANet-Plus row index cannot be represented."))?;
        let mut column = 0_u32;
        for raw in row {
            while occupied(&cells, row_index, column) {
                column = column
                    .checked_add(1)
                    .ok_or_else(|| grid_error("SLANet-Plus column index overflowed."))?;
            }
            while range_occupied(&cells, row_index, column, raw.column_span)? {
                column = column
                    .checked_add(1)
                    .ok_or_else(|| grid_error("SLANet-Plus column index overflowed."))?;
                while occupied(&cells, row_index, column) {
                    column = column
                        .checked_add(1)
                        .ok_or_else(|| grid_error("SLANet-Plus column index overflowed."))?;
                }
            }
            let column_end = column
                .checked_add(raw.column_span)
                .ok_or_else(|| grid_error("SLANet-Plus column span overflowed."))?;
            let row_end = row_index
                .checked_add(raw.row_span)
                .ok_or_else(|| grid_error("SLANet-Plus row span overflowed."))?;
            cells.push(GridCell {
                row: row_index,
                column,
                row_span: raw.row_span,
                column_span: raw.column_span,
                quad: raw.quad,
            });
            column = column_end;
            column_count = column_count.max(column_end);
            row_count = row_count.max(row_end);
            if cells.len() > MAX_CELLS {
                return Err(grid_error(format!(
                    "SLANet-Plus returned more than {MAX_CELLS} cells."
                )));
            }
        }
    }
    Ok(StructureGrid {
        row_count,
        column_count,
        cells,
        confidence: decoded.confidence,
    })
}

fn parse_rows(tokens: &[String], cells: &[DecodedCell]) -> UseResult<Vec<Vec<RawCell>>> {
    let mut rows = Vec::new();
    let mut current_row: Option<Vec<RawCell>> = None;
    let mut pending: Option<RawCell> = None;
    let mut cell_cursor = 0_usize;
    let cell_positions = cells
        .iter()
        .map(|cell| cell.token_position)
        .collect::<BTreeSet<_>>();
    if cell_positions.len() != cells.len() {
        return Err(grid_error(
            "SLANet-Plus returned duplicate cell-token geometry identity.",
        ));
    }

    for (position, token) in tokens.iter().enumerate() {
        match token.as_str() {
            "<tr>" => {
                if current_row.is_some() || pending.is_some() {
                    return Err(grid_error("SLANet-Plus produced nested table rows."));
                }
                current_row = Some(Vec::new());
            }
            "</tr>" => {
                if pending.is_some() {
                    return Err(grid_error(
                        "SLANet-Plus ended a row inside a spanning cell token.",
                    ));
                }
                let row = current_row
                    .take()
                    .ok_or_else(|| grid_error("SLANet-Plus ended an unopened table row."))?;
                rows.push(row);
            }
            "<td></td>" => {
                let row = current_row
                    .as_mut()
                    .ok_or_else(|| grid_error("SLANet-Plus emitted a cell outside a table row."))?;
                row.push(RawCell {
                    row_span: 1,
                    column_span: 1,
                    quad: take_quad(cells, &mut cell_cursor, position)?,
                });
            }
            "<td" => {
                if current_row.is_none() || pending.is_some() {
                    return Err(grid_error(
                        "SLANet-Plus emitted an invalid spanning-cell start.",
                    ));
                }
                pending = Some(RawCell {
                    row_span: 1,
                    column_span: 1,
                    quad: take_quad(cells, &mut cell_cursor, position)?,
                });
            }
            ">" => {
                if pending.is_none() {
                    return Err(grid_error(
                        "SLANet-Plus emitted a cell delimiter without a cell.",
                    ));
                }
            }
            "</td>" => {
                let cell = pending
                    .take()
                    .ok_or_else(|| grid_error("SLANet-Plus ended an unopened spanning cell."))?;
                current_row
                    .as_mut()
                    .ok_or_else(|| grid_error("SLANet-Plus cell lost its table row."))?
                    .push(cell);
            }
            _ => {
                if let Some(span) = parse_span(token, " colspan=\"")? {
                    let cell = pending
                        .as_mut()
                        .ok_or_else(|| grid_error("SLANet-Plus emitted colspan outside a cell."))?;
                    cell.column_span = span;
                } else if let Some(span) = parse_span(token, " rowspan=\"")? {
                    let cell = pending
                        .as_mut()
                        .ok_or_else(|| grid_error("SLANet-Plus emitted rowspan outside a cell."))?;
                    cell.row_span = span;
                }
            }
        }
    }
    if current_row.is_some() || pending.is_some() || cell_cursor != cells.len() {
        return Err(grid_error(
            "SLANet-Plus structure ended with incomplete row or cell geometry.",
        ));
    }
    Ok(rows)
}

fn take_quad(
    cells: &[DecodedCell],
    cursor: &mut usize,
    token_position: usize,
) -> UseResult<Option<[u32; 8]>> {
    let cell = cells
        .get(*cursor)
        .ok_or_else(|| grid_error("SLANet-Plus cell token has no aligned geometry placeholder."))?;
    if cell.token_position != token_position {
        return Err(grid_error(
            "SLANet-Plus cell geometry does not align with its structure token.",
        ));
    }
    *cursor += 1;
    Ok(cell.quad)
}

fn parse_span(token: &str, prefix: &str) -> UseResult<Option<u32>> {
    let Some(value) = token
        .strip_prefix(prefix)
        .and_then(|value| value.strip_suffix('"'))
    else {
        return Ok(None);
    };
    let span = value
        .parse::<u32>()
        .map_err(|_| grid_error("SLANet-Plus emitted a non-numeric table span."))?;
    if !(1..=MAX_SPAN).contains(&span) {
        return Err(grid_error(format!(
            "SLANet-Plus table spans must be between 1 and {MAX_SPAN}."
        )));
    }
    Ok(Some(span))
}

fn occupied(cells: &[GridCell], row: u32, column: u32) -> bool {
    cells.iter().any(|cell| {
        row >= cell.row
            && row < cell.row.saturating_add(cell.row_span)
            && column >= cell.column
            && column < cell.column.saturating_add(cell.column_span)
    })
}

fn range_occupied(cells: &[GridCell], row: u32, start: u32, span: u32) -> UseResult<bool> {
    let end = start
        .checked_add(span)
        .ok_or_else(|| grid_error("SLANet-Plus column span overflowed."))?;
    Ok((start..end).any(|column| occupied(cells, row, column)))
}

struct RawCell {
    row_span: u32,
    column_span: u32,
    quad: Option<[u32; 8]>,
}

fn grid_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.table_model_output_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_cells_reserve_future_grid_slots() {
        let decoded = DecodedStructure {
            tokens: [
                "<tbody>",
                "<tr>",
                "<td",
                " rowspan=\"2\"",
                ">",
                "</td>",
                "<td></td>",
                "</tr>",
                "<tr>",
                "<td></td>",
                "</tr>",
                "</tbody>",
            ]
            .map(str::to_string)
            .to_vec(),
            cells: vec![
                DecodedCell {
                    token_position: 2,
                    quad: None,
                },
                DecodedCell {
                    token_position: 6,
                    quad: None,
                },
                DecodedCell {
                    token_position: 9,
                    quad: None,
                },
            ],
            confidence: 0.9,
        };
        let grid = project(decoded).unwrap();
        assert_eq!(grid.row_count, 2);
        assert_eq!(grid.column_count, 2);
        assert_eq!(
            grid.cells
                .iter()
                .map(|cell| (cell.row, cell.column, cell.row_span, cell.column_span))
                .collect::<Vec<_>>(),
            vec![(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1)]
        );
    }

    #[test]
    fn malformed_cell_stream_is_not_publishable() {
        let decoded = DecodedStructure {
            tokens: vec!["<tr>".to_string(), "<td".to_string()],
            cells: vec![DecodedCell {
                token_position: 1,
                quad: None,
            }],
            confidence: 0.8,
        };
        assert_eq!(
            project(decoded).unwrap_err().code,
            "use.ocr.table_model_output_invalid"
        );
    }
}
