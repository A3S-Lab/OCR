use super::decoder::{GridCell, StructureGrid};
use super::orientation::TableCropOrientation;

const MAX_MEAN_SNAP_ERROR_RATIO: f64 = 0.05;

/// Reconciles model topology with deterministic source wire positions.
///
/// SLANet owns row, column, and span semantics. The wired candidate owns the
/// exact source line coordinates. Snapping model boundary anchors to an
/// ordered subset of those source lines closes small model-quad gaps. An axis
/// without enough source evidence retains its model geometry, and no missing
/// model quad is ever fabricated from wires.
pub(super) fn align_grid_to_wires(
    grid: &mut StructureGrid,
    orientation: TableCropOrientation,
    horizontal_lines: &[u32],
    vertical_lines: &[u32],
) -> bool {
    if grid.cells.iter().any(|cell| cell.quad.is_none()) {
        return false;
    }
    let x_boundaries = source_boundaries(
        match orientation {
            TableCropOrientation::Upright => grid.column_count,
            TableCropOrientation::Rotate90 => grid.row_count,
        },
        vertical_lines,
        grid.cells.iter().filter_map(|cell| {
            let bounds = quad_bounds(cell.quad?);
            let (start, span) = match orientation {
                TableCropOrientation::Upright => (cell.column, cell.column_span),
                TableCropOrientation::Rotate90 => (cell.row, cell.row_span),
            };
            Some((start, span, bounds.left, bounds.right))
        }),
    );
    let y_boundaries = source_boundaries(
        match orientation {
            TableCropOrientation::Upright => grid.row_count,
            TableCropOrientation::Rotate90 => grid.column_count,
        },
        horizontal_lines,
        grid.cells.iter().filter_map(|cell| {
            let bounds = quad_bounds(cell.quad?);
            let (start, span) = match orientation {
                TableCropOrientation::Upright => (cell.row, cell.row_span),
                TableCropOrientation::Rotate90 => (
                    reverse_start(cell.column, cell.column_span, grid.column_count)?,
                    cell.column_span,
                ),
            };
            Some((start, span, bounds.top, bounds.bottom))
        }),
    );
    if x_boundaries.is_none() && y_boundaries.is_none() {
        return false;
    }

    let aligned = grid
        .cells
        .iter()
        .map(|cell| {
            let (left, top, right, bottom) = aligned_cell_bounds(
                cell,
                grid.row_count,
                grid.column_count,
                orientation,
                x_boundaries.as_deref(),
                y_boundaries.as_deref(),
            )?;
            Some([left, top, right, top, right, bottom, left, bottom])
        })
        .collect::<Option<Vec<_>>>();
    let Some(aligned) = aligned else {
        return false;
    };
    for (cell, quad) in grid.cells.iter_mut().zip(aligned) {
        cell.quad = Some(quad);
    }
    true
}

fn aligned_cell_bounds(
    cell: &GridCell,
    row_count: u32,
    column_count: u32,
    orientation: TableCropOrientation,
    x_boundaries: Option<&[u32]>,
    y_boundaries: Option<&[u32]>,
) -> Option<(u32, u32, u32, u32)> {
    let model = quad_bounds(cell.quad?);
    let (row_start, row_end) = address_range(cell.row, cell.row_span, row_count)?;
    let (column_start, column_end) = address_range(cell.column, cell.column_span, column_count)?;
    Some(match orientation {
        TableCropOrientation::Upright => (
            x_boundaries.map_or(model.left, |boundaries| boundaries[column_start]),
            y_boundaries.map_or(model.top, |boundaries| boundaries[row_start]),
            x_boundaries.map_or(model.right, |boundaries| boundaries[column_end]),
            y_boundaries.map_or(model.bottom, |boundaries| boundaries[row_end]),
        ),
        TableCropOrientation::Rotate90 => {
            let columns = usize::try_from(column_count).ok()?;
            (
                x_boundaries.map_or(model.left, |boundaries| boundaries[row_start]),
                y_boundaries.map_or(model.top, |boundaries| boundaries[columns - column_end]),
                x_boundaries.map_or(model.right, |boundaries| boundaries[row_end]),
                y_boundaries.map_or(model.bottom, |boundaries| {
                    boundaries[columns - column_start]
                }),
            )
        }
    })
}

fn source_boundaries(
    interval_count: u32,
    source_lines: &[u32],
    cells: impl Iterator<Item = (u32, u32, u32, u32)>,
) -> Option<Vec<u32>> {
    let interval_count = usize::try_from(interval_count).ok()?;
    let boundary_count = interval_count.checked_add(1)?;
    let mut lines = source_lines.to_vec();
    lines.sort_unstable();
    lines.dedup();
    if interval_count == 0 || lines.len() < boundary_count {
        return None;
    }
    let mut candidates = vec![Vec::<u32>::new(); boundary_count];
    for (start, span, leading, trailing) in cells {
        let (start_index, end) = address_range(start, span, u32::try_from(interval_count).ok()?)?;
        candidates[start_index].push(leading);
        candidates[end].push(trailing);
    }
    if candidates[0].is_empty() {
        candidates[0].push(*lines.first()?);
    }
    if candidates[interval_count].is_empty() {
        candidates[interval_count].push(*lines.last()?);
    }
    let mut anchors = candidates.into_iter().map(median).collect::<Vec<_>>();
    interpolate_missing(&mut anchors)?;
    let anchors = anchors.into_iter().collect::<Option<Vec<_>>>()?;
    let selected = closest_ordered_subset(&lines, &anchors)?;
    let span = f64::from(selected.last()?.saturating_sub(*selected.first()?));
    if span <= 0.0 {
        return None;
    }
    let mean_error = selected
        .iter()
        .zip(&anchors)
        .map(|(line, anchor)| line.abs_diff(*anchor) as f64)
        .sum::<f64>()
        / boundary_count as f64;
    (mean_error / span <= MAX_MEAN_SNAP_ERROR_RATIO).then_some(selected)
}

fn closest_ordered_subset(lines: &[u32], anchors: &[u32]) -> Option<Vec<u32>> {
    let selected_count = anchors.len();
    if selected_count < 2 || lines.len() < selected_count {
        return None;
    }
    let mut costs = vec![vec![f64::INFINITY; lines.len()]; selected_count];
    let mut previous = vec![vec![usize::MAX; lines.len()]; selected_count];
    let max_first = lines.len().checked_sub(selected_count)?;
    for line_index in 0..=max_first {
        costs[0][line_index] = f64::from(lines[line_index].abs_diff(anchors[0]));
    }
    for selected in 1..selected_count {
        let first = selected;
        let last = lines.len().checked_sub(selected_count - selected)?;
        for line_index in first..=last {
            let mut best = f64::INFINITY;
            let mut best_index = usize::MAX;
            for (prior, prior_cost) in costs[selected - 1]
                .iter()
                .enumerate()
                .take(line_index)
                .skip(selected - 1)
            {
                if *prior_cost < best {
                    best = *prior_cost;
                    best_index = prior;
                }
            }
            if best_index != usize::MAX {
                costs[selected][line_index] =
                    best + f64::from(lines[line_index].abs_diff(anchors[selected]));
                previous[selected][line_index] = best_index;
            }
        }
    }

    let mut line_index = costs[selected_count - 1]
        .iter()
        .enumerate()
        .skip(selected_count - 1)
        .min_by(|(_, left), (_, right)| left.total_cmp(right))?
        .0;
    if !costs[selected_count - 1][line_index].is_finite() {
        return None;
    }
    let mut indices = vec![0_usize; selected_count];
    for selected in (0..selected_count).rev() {
        indices[selected] = line_index;
        if selected > 0 {
            line_index = previous[selected][line_index];
        }
    }
    Some(indices.into_iter().map(|index| lines[index]).collect())
}

fn address_range(start: u32, span: u32, count: u32) -> Option<(usize, usize)> {
    let end = start.checked_add(span)?;
    if span == 0 || end > count {
        return None;
    }
    Some((usize::try_from(start).ok()?, usize::try_from(end).ok()?))
}

fn reverse_start(start: u32, span: u32, count: u32) -> Option<u32> {
    count.checked_sub(start.checked_add(span)?)
}

#[derive(Clone, Copy)]
struct Bounds {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

fn quad_bounds(quad: [u32; 8]) -> Bounds {
    let x = [quad[0], quad[2], quad[4], quad[6]];
    let y = [quad[1], quad[3], quad[5], quad[7]];
    Bounds {
        left: *x.iter().min().unwrap_or(&0),
        top: *y.iter().min().unwrap_or(&0),
        right: *x.iter().max().unwrap_or(&0),
        bottom: *y.iter().max().unwrap_or(&0),
    }
}

fn median(mut values: Vec<u32>) -> Option<u32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[middle])
    } else {
        u32::try_from((u64::from(values[middle - 1]) + u64::from(values[middle])) / 2).ok()
    }
}

fn interpolate_missing(values: &mut [Option<u32>]) -> Option<()> {
    let mut left = 0_usize;
    while left + 1 < values.len() {
        let right = ((left + 1)..values.len()).find(|index| values[*index].is_some())?;
        let start = u64::from(values[left]?);
        let end = u64::from(values[right]?);
        let distance = u64::try_from(right - left).ok()?;
        for (index, value_slot) in values.iter_mut().enumerate().take(right).skip(left + 1) {
            let offset = u64::try_from(index - left).ok()?;
            let value = if end >= start {
                start + ((end - start) * offset + distance / 2) / distance
            } else {
                start.saturating_sub(((start - end) * offset + distance / 2) / distance)
            };
            *value_slot = u32::try_from(value).ok();
        }
        left = right;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_grid_snaps_rows_to_x_and_reversed_columns_to_y() {
        let mut grid = grid();

        assert!(align_grid_to_wires(
            &mut grid,
            TableCropOrientation::Rotate90,
            &[10, 60, 110, 160],
            &[20, 70, 120, 170],
        ));

        assert_eq!(grid.cells[0].quad, Some([20, 60, 70, 60, 70, 110, 20, 110]));
        assert_eq!(grid.cells[1].quad, Some([20, 10, 70, 10, 70, 60, 20, 60]));
        assert_eq!(
            grid.cells[2].quad,
            Some([70, 60, 120, 60, 120, 110, 70, 110])
        );
    }

    #[test]
    fn insufficient_source_lines_leave_model_quads_unchanged() {
        let mut grid = grid();
        let original = grid.cells.clone();

        assert!(!align_grid_to_wires(
            &mut grid,
            TableCropOrientation::Rotate90,
            &[10, 110],
            &[20, 120],
        ));
        assert_eq!(grid.cells, original);
    }

    #[test]
    fn one_supported_axis_aligns_without_inventing_the_other_axis() {
        let mut grid = grid();

        assert!(align_grid_to_wires(
            &mut grid,
            TableCropOrientation::Rotate90,
            &[10, 110],
            &[20, 70, 120],
        ));

        assert_eq!(grid.cells[0].quad, Some([20, 62, 70, 62, 70, 108, 20, 108]));
        assert_eq!(grid.cells[1].quad, Some([20, 12, 70, 12, 70, 58, 20, 58]));
    }

    #[test]
    fn missing_model_geometry_is_never_fabricated_from_wires() {
        let mut grid = grid();
        grid.cells[1].quad = None;
        let original = grid.cells.clone();

        assert!(!align_grid_to_wires(
            &mut grid,
            TableCropOrientation::Rotate90,
            &[10, 60, 110],
            &[20, 70, 120],
        ));
        assert_eq!(grid.cells, original);
    }

    fn grid() -> StructureGrid {
        StructureGrid {
            row_count: 2,
            column_count: 2,
            confidence: Some(0.99),
            cells: vec![
                cell(0, 0, [23, 62, 68, 62, 68, 108, 23, 108]),
                cell(0, 1, [23, 12, 68, 12, 68, 58, 23, 58]),
                cell(1, 0, [73, 62, 118, 62, 118, 108, 73, 108]),
                cell(1, 1, [73, 12, 118, 12, 118, 58, 73, 58]),
            ],
        }
    }

    fn cell(row: u32, column: u32, quad: [u32; 8]) -> GridCell {
        GridCell {
            row,
            column,
            row_span: 1,
            column_span: 1,
            quad: Some(quad),
        }
    }
}
