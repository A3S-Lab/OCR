use std::collections::BTreeMap;

use image::RgbImage;

use super::decoder::{GridCell, StructureGrid};
use super::orientation::TableCropOrientation;
use super::wired::{LineTrack, WiredCandidate};

const LINE_BAND_RADIUS: u32 = 2;
const DUPLICATE_LINE_DISTANCE: u32 = LINE_BAND_RADIUS * 2;
const ENDPOINT_MARGIN: u32 = LINE_BAND_RADIUS + 1;
const MAX_INTERVALS_PER_AXIS: usize = 64;
const MAX_PRIMITIVE_CELLS: usize = 4_096;
const MAX_TOPOLOGY_SEARCH_OPERATIONS: usize = 2_000_000;

/// Derives exact table topology from immutable source pixels.
///
/// Candidate line coordinates establish the primitive grid. The first proof
/// uses source pixels alone. A second proof may refine unresolved edges with
/// the detector's retained long-line tracks, but a missing track is negative
/// evidence only when the primitive edge was long enough for that detector to
/// observe. Wider junction tolerance may promote a stroke only when it reaches
/// both source junctions; one-sided support cannot weaken known absence. Both
/// proofs require source-backed outer closure and exactly one rectangular
/// partition. Otherwise the structure model remains authoritative.
pub(super) fn derive_source_grid(
    image: &RgbImage,
    candidate: &WiredCandidate,
) -> Option<StructureGrid> {
    let (horizontal, vertical) = refined_source_boundaries(image, candidate);
    derive_pixel_grid(image, candidate, &horizontal, &vertical)
        .or_else(|| derive_track_grid(image, candidate, &horizontal, &vertical))
}

fn derive_pixel_grid(
    image: &RgbImage,
    candidate: &WiredCandidate,
    horizontal: &[u32],
    vertical: &[u32],
) -> Option<StructureGrid> {
    let rows = horizontal.len().checked_sub(1)?;
    let columns = vertical.len().checked_sub(1)?;
    if rows == 0
        || columns == 0
        || rows > MAX_INTERVALS_PER_AXIS
        || columns > MAX_INTERVALS_PER_AXIS
        || rows.checked_mul(columns)? > MAX_PRIMITIVE_CELLS
    {
        return None;
    }

    validate_outer_axes(image, horizontal, vertical)?;
    let mut evidence = Vec::new();
    for row in 0..rows {
        for (boundary, coordinate) in vertical.iter().enumerate().take(columns).skip(1) {
            evidence.push(BoundaryEvidence {
                first: primitive_index(row, boundary - 1, columns),
                second: primitive_index(row, boundary, columns),
                stroke: classify_vertical(image, *coordinate, horizontal[row], horizontal[row + 1]),
            });
        }
    }
    for (boundary, coordinate) in horizontal.iter().enumerate().take(rows).skip(1) {
        for column in 0..columns {
            evidence.push(BoundaryEvidence {
                first: primitive_index(boundary - 1, column, columns),
                second: primitive_index(boundary, column, columns),
                stroke: classify_horizontal(
                    image,
                    *coordinate,
                    vertical[column],
                    vertical[column + 1],
                ),
            });
        }
    }
    let cells = resolve_unique_partition(
        rows,
        columns,
        horizontal,
        vertical,
        candidate.orientation,
        &evidence,
    )?;
    let (row_count, column_count) = match candidate.orientation {
        TableCropOrientation::Upright => (rows, columns),
        TableCropOrientation::Rotate90 => (columns, rows),
    };
    Some(StructureGrid {
        row_count: u32::try_from(row_count).ok()?,
        column_count: u32::try_from(column_count).ok()?,
        cells,
        // Pixel topology is deterministic evidence, not a model probability.
        confidence: None,
    })
}

fn derive_track_grid(
    image: &RgbImage,
    candidate: &WiredCandidate,
    horizontal: &[u32],
    vertical: &[u32],
) -> Option<StructureGrid> {
    let rows = horizontal.len().checked_sub(1)?;
    let columns = vertical.len().checked_sub(1)?;
    if rows == 0
        || columns == 0
        || rows > MAX_INTERVALS_PER_AXIS
        || columns > MAX_INTERVALS_PER_AXIS
        || rows.checked_mul(columns)? > MAX_PRIMITIVE_CELLS
    {
        return None;
    }
    let minimum_horizontal_track = (image.width() / 5).max(96);
    let minimum_vertical_track = (image.height() / 12).max(64);

    validate_fused_outer_axes(
        image,
        candidate,
        horizontal,
        vertical,
        minimum_horizontal_track,
        minimum_vertical_track,
    )?;
    let mut evidence = Vec::new();
    for row in 0..rows {
        for (boundary, coordinate) in vertical.iter().enumerate().take(columns).skip(1) {
            evidence.push(BoundaryEvidence {
                first: primitive_index(row, boundary - 1, columns),
                second: primitive_index(row, boundary, columns),
                stroke: fuse_strokes(
                    classify_vertical_with_junction_tolerance(
                        image,
                        *coordinate,
                        horizontal[row],
                        horizontal[row + 1],
                    ),
                    classify_tracks(
                        *coordinate,
                        horizontal[row],
                        horizontal[row + 1],
                        &candidate.vertical_tracks,
                        minimum_vertical_track,
                    ),
                ),
            });
        }
    }
    for (boundary, coordinate) in horizontal.iter().enumerate().take(rows).skip(1) {
        for column in 0..columns {
            evidence.push(BoundaryEvidence {
                first: primitive_index(boundary - 1, column, columns),
                second: primitive_index(boundary, column, columns),
                stroke: fuse_strokes(
                    classify_horizontal_with_junction_tolerance(
                        image,
                        *coordinate,
                        vertical[column],
                        vertical[column + 1],
                    ),
                    classify_tracks(
                        *coordinate,
                        vertical[column],
                        vertical[column + 1],
                        &candidate.horizontal_tracks,
                        minimum_horizontal_track,
                    ),
                ),
            });
        }
    }
    let cells = resolve_unique_partition(
        rows,
        columns,
        horizontal,
        vertical,
        candidate.orientation,
        &evidence,
    )?;
    let (row_count, column_count) = match candidate.orientation {
        TableCropOrientation::Upright => (rows, columns),
        TableCropOrientation::Rotate90 => (columns, rows),
    };
    Some(StructureGrid {
        row_count: u32::try_from(row_count).ok()?,
        column_count: u32::try_from(column_count).ok()?,
        cells,
        confidence: None,
    })
}

fn validate_fused_outer_axes(
    image: &RgbImage,
    candidate: &WiredCandidate,
    horizontal: &[u32],
    vertical: &[u32],
    minimum_horizontal_track: u32,
    minimum_vertical_track: u32,
) -> Option<()> {
    let top = *horizontal.first()?;
    let bottom = *horizontal.last()?;
    let left = *vertical.first()?;
    let right = *vertical.last()?;
    let horizontal_intervals = vertical.len().checked_sub(1)?;
    let vertical_intervals = horizontal.len().checked_sub(1)?;
    let horizontal_side = |fixed, terminal| {
        side_has_recurring_terminal_support(fixed, &candidate.vertical_tracks, terminal)
            || side_is_backed_within_detector_resolution(
                vertical,
                minimum_horizontal_track,
                |column| {
                    fuse_strokes(
                        classify_horizontal(image, fixed, vertical[column], vertical[column + 1]),
                        classify_tracks(
                            fixed,
                            vertical[column],
                            vertical[column + 1],
                            &candidate.horizontal_tracks,
                            minimum_horizontal_track,
                        ),
                    )
                },
                |column| {
                    junction_has_terminal_support(
                        vertical[column],
                        fixed,
                        &candidate.vertical_tracks,
                        terminal,
                    ) && junction_has_terminal_support(
                        vertical[column + 1],
                        fixed,
                        &candidate.vertical_tracks,
                        terminal,
                    )
                },
            )
    };
    let vertical_side = |fixed, terminal| {
        side_has_recurring_terminal_support(fixed, &candidate.horizontal_tracks, terminal)
            || side_is_backed_within_detector_resolution(
                horizontal,
                minimum_vertical_track,
                |row| {
                    fuse_strokes(
                        classify_vertical(image, fixed, horizontal[row], horizontal[row + 1]),
                        classify_tracks(
                            fixed,
                            horizontal[row],
                            horizontal[row + 1],
                            &candidate.vertical_tracks,
                            minimum_vertical_track,
                        ),
                    )
                },
                |row| {
                    junction_has_terminal_support(
                        horizontal[row],
                        fixed,
                        &candidate.horizontal_tracks,
                        terminal,
                    ) && junction_has_terminal_support(
                        horizontal[row + 1],
                        fixed,
                        &candidate.horizontal_tracks,
                        terminal,
                    )
                },
            )
    };
    (horizontal_intervals > 0
        && vertical_intervals > 0
        && horizontal_side(top, TrackTerminal::Start)
        && horizontal_side(bottom, TrackTerminal::End)
        && vertical_side(left, TrackTerminal::Start)
        && vertical_side(right, TrackTerminal::End))
    .then_some(())
}

#[derive(Clone, Copy)]
enum TrackTerminal {
    Start,
    End,
}

fn junction_has_terminal_support(
    fixed: u32,
    terminal_coordinate: u32,
    tracks: &[LineTrack],
    terminal: TrackTerminal,
) -> bool {
    tracks.iter().any(|track| {
        let endpoint = match terminal {
            TrackTerminal::Start => track.start,
            TrackTerminal::End => track.end,
        };
        track.fixed.abs_diff(fixed) <= DUPLICATE_LINE_DISTANCE
            && endpoint.abs_diff(terminal_coordinate) <= super::wired::INTERSECTION_TOLERANCE
    })
}

fn side_has_recurring_terminal_support(
    coordinate: u32,
    tracks: &[LineTrack],
    terminal: TrackTerminal,
) -> bool {
    tracks
        .iter()
        .filter(|track| {
            let endpoint = match terminal {
                TrackTerminal::Start => track.start,
                TrackTerminal::End => track.end,
            };
            endpoint.abs_diff(coordinate) <= super::wired::INTERSECTION_TOLERANCE
        })
        .take(2)
        .count()
        == 2
}

fn side_is_backed_within_detector_resolution(
    boundaries: &[u32],
    minimum_track_length: u32,
    mut classify: impl FnMut(usize) -> Option<Stroke>,
    mut has_terminal_support: impl FnMut(usize) -> bool,
) -> bool {
    let Some(intervals) = boundaries.len().checked_sub(1) else {
        return false;
    };
    let mut has_support = false;
    let mut unsupported_start = None;
    for interval in 0..intervals {
        if classify(interval) == Some(Stroke::Present) || has_terminal_support(interval) {
            has_support = true;
            unsupported_start = None;
            continue;
        }
        let start = *unsupported_start.get_or_insert(boundaries[interval]);
        let unsupported_length = boundaries[interval + 1]
            .saturating_sub(start)
            .saturating_add(1);
        if unsupported_length >= minimum_track_length {
            return false;
        }
    }
    has_support
}

fn fuse_strokes(pixel: Option<Stroke>, track: Option<Stroke>) -> Option<Stroke> {
    match (pixel, track) {
        (Some(Stroke::Present), _) | (_, Some(Stroke::Present)) => Some(Stroke::Present),
        (Some(Stroke::Absent), _) | (None, Some(Stroke::Absent)) => Some(Stroke::Absent),
        (None, None) => None,
    }
}

fn classify_tracks(
    fixed: u32,
    start: u32,
    end: u32,
    tracks: &[LineTrack],
    minimum_track_length: u32,
) -> Option<Stroke> {
    let length = end.checked_sub(start)?;
    if length <= ENDPOINT_MARGIN.saturating_mul(2) {
        return None;
    }
    let first = start.checked_add(ENDPOINT_MARGIN)?;
    let last = end.checked_sub(ENDPOINT_MARGIN)?;
    let mut intervals = tracks
        .iter()
        .filter(|track| track.fixed.abs_diff(fixed) <= DUPLICATE_LINE_DISTANCE)
        .filter_map(|track| {
            let interval_start = track.start.max(first);
            let interval_end = track.end.min(last);
            (interval_start <= interval_end).then_some((interval_start, interval_end))
        })
        .collect::<Vec<_>>();
    if intervals.is_empty() {
        // The detector is exhaustive only for runs at least this long. A
        // missing shorter track is outside its observation authority and must
        // remain unknown rather than becoming invented negative evidence.
        return (length.saturating_add(1) >= minimum_track_length).then_some(Stroke::Absent);
    }
    intervals.sort_unstable();
    let tolerance = super::wired::LINE_GAP_TOLERANCE;
    let mut merged = Vec::<(u32, u32)>::new();
    for interval in intervals {
        if let Some(previous) = merged.last_mut() {
            if interval.0 <= previous.1.saturating_add(tolerance).saturating_add(1) {
                previous.1 = previous.1.max(interval.1);
                continue;
            }
        }
        merged.push(interval);
    }
    let start_anchored = merged.first()?.0.saturating_sub(first) <= tolerance;
    let end_anchored = last.saturating_sub(merged.last()?.1) <= tolerance;
    if start_anchored && end_anchored && merged.len() == 1 {
        Some(Stroke::Present)
    } else {
        None
    }
}

/// Adds a primitive axis only when a retained line terminal forms a complete
/// source-pixel T junction with the perpendicular grid.
///
/// A single terminal is not an axis: it may be damage or foreground touching a
/// rule. Requiring continuous perpendicular edges on both sides of an existing
/// grid line establishes a source-backed crossbar while still allowing merged
/// cells elsewhere along the newly discovered axis.
fn refined_source_boundaries(image: &RgbImage, candidate: &WiredCandidate) -> (Vec<u32>, Vec<u32>) {
    let mut horizontal = normalized_boundaries(&candidate.horizontal_lines);
    let mut vertical = normalized_boundaries(&candidate.vertical_lines);
    let base_horizontal = horizontal.clone();
    let base_vertical = vertical.clone();

    for track in &candidate.vertical_tracks {
        let Some(axis) = matching_internal_axis(&base_vertical, track.fixed) else {
            continue;
        };
        for terminal in [track.start, track.end] {
            if lies_strictly_inside(terminal, &base_horizontal)
                && classify_horizontal(
                    image,
                    terminal,
                    base_vertical[axis - 1],
                    base_vertical[axis],
                ) == Some(Stroke::Present)
                && classify_horizontal(
                    image,
                    terminal,
                    base_vertical[axis],
                    base_vertical[axis + 1],
                ) == Some(Stroke::Present)
                && classify_exact_crossbar(
                    image,
                    terminal,
                    base_vertical[axis - 1],
                    base_vertical[axis + 1],
                    false,
                ) == Some(Stroke::Present)
            {
                horizontal.push(terminal);
            }
        }
    }

    for track in &candidate.horizontal_tracks {
        let Some(axis) = matching_internal_axis(&base_horizontal, track.fixed) else {
            continue;
        };
        for terminal in [track.start, track.end] {
            if lies_strictly_inside(terminal, &base_vertical)
                && classify_vertical(
                    image,
                    terminal,
                    base_horizontal[axis - 1],
                    base_horizontal[axis],
                ) == Some(Stroke::Present)
                && classify_vertical(
                    image,
                    terminal,
                    base_horizontal[axis],
                    base_horizontal[axis + 1],
                ) == Some(Stroke::Present)
                && classify_exact_crossbar(
                    image,
                    terminal,
                    base_horizontal[axis - 1],
                    base_horizontal[axis + 1],
                    true,
                ) == Some(Stroke::Present)
            {
                vertical.push(terminal);
            }
        }
    }

    (
        normalized_boundaries(&horizontal),
        normalized_boundaries(&vertical),
    )
}

fn classify_exact_crossbar(
    image: &RgbImage,
    fixed: u32,
    start: u32,
    end: u32,
    vertical: bool,
) -> Option<Stroke> {
    classify_stroke(start, end, |position| {
        let (x, y) = if vertical {
            (fixed, position)
        } else {
            (position, fixed)
        };
        super::wired::is_dark(image, x, y)
    })
}

fn matching_internal_axis(boundaries: &[u32], coordinate: u32) -> Option<usize> {
    boundaries
        .iter()
        .enumerate()
        .take(boundaries.len().saturating_sub(1))
        .skip(1)
        .filter(|(_, boundary)| boundary.abs_diff(coordinate) <= DUPLICATE_LINE_DISTANCE)
        .min_by_key(|(_, boundary)| boundary.abs_diff(coordinate))
        .map(|(index, _)| index)
}

fn lies_strictly_inside(coordinate: u32, boundaries: &[u32]) -> bool {
    boundaries
        .first()
        .zip(boundaries.last())
        .is_some_and(|(first, last)| coordinate > *first && coordinate < *last)
}

fn normalized_boundaries(lines: &[u32]) -> Vec<u32> {
    let mut lines = lines.to_vec();
    lines.sort_unstable();
    let mut normalized = Vec::<u32>::new();
    for line in lines {
        if let Some(previous) = normalized.last_mut() {
            if line <= previous.saturating_add(DUPLICATE_LINE_DISTANCE) {
                *previous = previous.saturating_add(line) / 2;
                continue;
            }
        }
        normalized.push(line);
    }
    normalized
}

fn validate_outer_axes(image: &RgbImage, horizontal: &[u32], vertical: &[u32]) -> Option<()> {
    let top = *horizontal.first()?;
    let bottom = *horizontal.last()?;
    let left = *vertical.first()?;
    let right = *vertical.last()?;
    let horizontal_intervals = vertical.len().checked_sub(1)?;
    let vertical_intervals = horizontal.len().checked_sub(1)?;
    let top_supported = side_has_support(
        (0..horizontal_intervals)
            .map(|column| classify_horizontal(image, top, vertical[column], vertical[column + 1])),
    );
    let bottom_supported =
        side_has_support((0..horizontal_intervals).map(|column| {
            classify_horizontal(image, bottom, vertical[column], vertical[column + 1])
        }));
    let left_supported = side_has_support(
        (0..vertical_intervals)
            .map(|row| classify_vertical(image, left, horizontal[row], horizontal[row + 1])),
    );
    let right_supported = side_has_support(
        (0..vertical_intervals)
            .map(|row| classify_vertical(image, right, horizontal[row], horizontal[row + 1])),
    );
    (top_supported && bottom_supported && left_supported && right_supported).then_some(())
}

fn side_has_support(mut strokes: impl Iterator<Item = Option<Stroke>>) -> bool {
    strokes.any(|stroke| stroke == Some(Stroke::Present))
}

#[derive(Debug, Clone, Copy)]
struct BoundaryEvidence {
    first: usize,
    second: usize,
    stroke: Option<Stroke>,
}

fn resolve_unique_partition(
    rows: usize,
    columns: usize,
    horizontal: &[u32],
    vertical: &[u32],
    orientation: TableCropOrientation,
    evidence: &[BoundaryEvidence],
) -> Option<Vec<GridCell>> {
    let primitive_cells = rows.checked_mul(columns)?;
    let ambiguous = evidence
        .iter()
        .filter(|edge| edge.stroke.is_none())
        .collect::<Vec<_>>();
    let ambiguous_bits = u32::try_from(ambiguous.len()).ok()?;
    let assignments = 1_usize.checked_shl(ambiguous_bits)?;
    let work_per_assignment = primitive_cells.checked_add(evidence.len())?;
    if assignments.checked_mul(work_per_assignment)? > MAX_TOPOLOGY_SEARCH_OPERATIONS {
        return None;
    }

    let mut base = UnionFind::new(primitive_cells);
    for edge in evidence
        .iter()
        .filter(|edge| edge.stroke == Some(Stroke::Absent))
    {
        base.join(edge.first, edge.second);
    }

    let mut unique = None::<Vec<GridCell>>;
    for assignment in 0..assignments {
        let mut union = base.clone();
        for (bit, edge) in ambiguous.iter().enumerate() {
            if assignment & (1_usize << bit) != 0 {
                union.join(edge.first, edge.second);
            }
        }
        if !partition_respects_present_edges(&mut union, evidence, &ambiguous, assignment) {
            continue;
        }

        let mut components = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for row in 0..rows {
            for column in 0..columns {
                let index = primitive_index(row, column, columns);
                components
                    .entry(union.root(index))
                    .or_default()
                    .push((row, column));
            }
        }
        let Some(mut cells) = components
            .into_values()
            .map(|component| component_cell(component, horizontal, vertical, orientation))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        cells.sort_by_key(|cell| (cell.row, cell.column));
        match &unique {
            Some(previous) if previous != &cells => return None,
            Some(_) => {}
            None => unique = Some(cells),
        }
    }
    unique
}

fn partition_respects_present_edges(
    union: &mut UnionFind,
    evidence: &[BoundaryEvidence],
    ambiguous: &[&BoundaryEvidence],
    assignment: usize,
) -> bool {
    let known_present = evidence
        .iter()
        .filter(|edge| edge.stroke == Some(Stroke::Present));
    let assigned_present = ambiguous
        .iter()
        .enumerate()
        .filter_map(|(bit, edge)| (assignment & (1_usize << bit) == 0).then_some(*edge));
    known_present.chain(assigned_present).all(|edge| {
        let first = union.root(edge.first);
        let second = union.root(edge.second);
        first != second
    })
}

fn component_cell(
    component: Vec<(usize, usize)>,
    horizontal: &[u32],
    vertical: &[u32],
    orientation: TableCropOrientation,
) -> Option<GridCell> {
    let first = *component.first()?;
    let (mut row_start, mut row_end) = (first.0, first.0);
    let (mut column_start, mut column_end) = (first.1, first.1);
    for (row, column) in &component {
        row_start = row_start.min(*row);
        row_end = row_end.max(*row);
        column_start = column_start.min(*column);
        column_end = column_end.max(*column);
    }
    let row_span = row_end.checked_sub(row_start)?.checked_add(1)?;
    let column_span = column_end.checked_sub(column_start)?.checked_add(1)?;
    if component.len() != row_span.checked_mul(column_span)? {
        return None;
    }
    let left = vertical[column_start];
    let right = vertical[column_end.checked_add(1)?];
    let top = horizontal[row_start];
    let bottom = horizontal[row_end.checked_add(1)?];
    let (row, column, row_span, column_span) = match orientation {
        TableCropOrientation::Upright => (row_start, column_start, row_span, column_span),
        TableCropOrientation::Rotate90 => (
            column_start,
            horizontal
                .len()
                .checked_sub(1)?
                .checked_sub(row_start.checked_add(row_span)?)?,
            column_span,
            row_span,
        ),
    };
    Some(GridCell {
        row: u32::try_from(row).ok()?,
        column: u32::try_from(column).ok()?,
        row_span: u32::try_from(row_span).ok()?,
        column_span: u32::try_from(column_span).ok()?,
        quad: Some([left, top, right, top, right, bottom, left, bottom]),
    })
}

fn primitive_index(row: usize, column: usize, columns: usize) -> usize {
    row * columns + column
}

fn classify_vertical(image: &RgbImage, fixed: u32, start: u32, end: u32) -> Option<Stroke> {
    classify_stroke(start, end, |position| {
        band_contains_dark(image, fixed, position, true)
    })
}

fn classify_horizontal(image: &RgbImage, fixed: u32, start: u32, end: u32) -> Option<Stroke> {
    classify_stroke(start, end, |position| {
        band_contains_dark(image, fixed, position, false)
    })
}

fn classify_vertical_with_junction_tolerance(
    image: &RgbImage,
    fixed: u32,
    start: u32,
    end: u32,
) -> Option<Stroke> {
    refine_with_junction_tolerance(
        classify_vertical(image, fixed, start, end),
        classify_stroke_with_junction_tolerance(start, end, |position| {
            band_contains_dark(image, fixed, position, true)
        }),
    )
}

fn classify_horizontal_with_junction_tolerance(
    image: &RgbImage,
    fixed: u32,
    start: u32,
    end: u32,
) -> Option<Stroke> {
    refine_with_junction_tolerance(
        classify_horizontal(image, fixed, start, end),
        classify_stroke_with_junction_tolerance(start, end, |position| {
            band_contains_dark(image, fixed, position, false)
        }),
    )
}

fn refine_with_junction_tolerance(
    baseline: Option<Stroke>,
    expanded: Option<Stroke>,
) -> Option<Stroke> {
    if expanded == Some(Stroke::Present) {
        expanded
    } else {
        baseline.or(expanded)
    }
}

fn classify_stroke(start: u32, end: u32, mut selected: impl FnMut(u32) -> bool) -> Option<Stroke> {
    let tolerance = super::wired::LINE_GAP_TOLERANCE;
    classify_stroke_with_tolerance(start, end, &mut selected, tolerance)
}

fn classify_stroke_with_junction_tolerance(
    start: u32,
    end: u32,
    mut selected: impl FnMut(u32) -> bool,
) -> Option<Stroke> {
    let tolerance = super::wired::LINE_GAP_TOLERANCE.saturating_add(LINE_BAND_RADIUS);
    classify_stroke_with_tolerance(start, end, &mut selected, tolerance)
}

fn classify_stroke_with_tolerance(
    start: u32,
    end: u32,
    mut selected: impl FnMut(u32) -> bool,
    tolerance: u32,
) -> Option<Stroke> {
    let length = end.checked_sub(start)?;
    if length <= ENDPOINT_MARGIN.saturating_mul(2) {
        return None;
    }
    let first = start.checked_add(ENDPOINT_MARGIN)?;
    let last = end.checked_sub(ENDPOINT_MARGIN)?;
    let mut first_dark = None;
    let mut last_dark = None;
    let mut first_component_end = None;
    let mut current_component_start = None;
    let mut gap = 0_u32;
    let mut longest_internal_gap = 0_u32;
    for position in first..=last {
        if selected(position) {
            if first_dark.is_none() {
                first_dark = Some(position);
                current_component_start = Some(position);
            } else {
                longest_internal_gap = longest_internal_gap.max(gap);
                if gap > super::wired::LINE_GAP_TOLERANCE {
                    first_component_end.get_or_insert(last_dark?);
                    current_component_start = Some(position);
                }
            }
            last_dark = Some(position);
            gap = 0;
        } else if first_dark.is_some() {
            gap += 1;
        }
    }
    let Some(first_dark) = first_dark else {
        return Some(Stroke::Absent);
    };
    let last_dark = last_dark?;
    let first_component_end = first_component_end.unwrap_or(last_dark);
    let last_component_start = current_component_start?;
    let start_anchored = first_dark.saturating_sub(first) <= tolerance;
    let end_anchored = last.saturating_sub(last_dark) <= tolerance;
    if start_anchored && end_anchored && longest_internal_gap <= super::wired::LINE_GAP_TOLERANCE {
        Some(Stroke::Present)
    } else if !start_anchored && !end_anchored {
        // Foreground that does not connect to either grid junction is cell
        // content, not evidence for a separator.
        Some(Stroke::Absent)
    } else if endpoint_support_is_only_junction_footprint(
        first,
        last,
        first_component_end,
        last_component_start,
        start_anchored,
        end_anchored,
    ) {
        // A perpendicular rule darkens the inspected band at its junction.
        // Support that never leaves that geometry-derived footprint is not
        // evidence that a separator continues into the cell edge.
        Some(Stroke::Absent)
    } else {
        // One-sided or disconnected junction support may be a damaged rule.
        None
    }
}

fn endpoint_support_is_only_junction_footprint(
    first: u32,
    last: u32,
    first_component_end: u32,
    last_component_start: u32,
    start_anchored: bool,
    end_anchored: bool,
) -> bool {
    let footprint = LINE_BAND_RADIUS.saturating_add(super::wired::LINE_GAP_TOLERANCE);
    (start_anchored && !end_anchored && first_component_end.saturating_sub(first) <= footprint)
        || (end_anchored
            && !start_anchored
            && last.saturating_sub(last_component_start) <= footprint)
}

fn band_contains_dark(image: &RgbImage, fixed: u32, position: u32, vertical: bool) -> bool {
    let limit = if vertical {
        image.width()
    } else {
        image.height()
    };
    let first = fixed.saturating_sub(LINE_BAND_RADIUS);
    let last = fixed
        .saturating_add(LINE_BAND_RADIUS)
        .min(limit.saturating_sub(1));
    (first..=last).any(|band| {
        let (x, y) = if vertical {
            (band, position)
        } else {
            (position, band)
        };
        if x >= image.width() || y >= image.height() {
            return false;
        }
        super::wired::is_dark(image, x, y)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stroke {
    Present,
    Absent,
}

#[derive(Clone)]
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(length: usize) -> Self {
        Self {
            parent: (0..length).collect(),
        }
    }

    fn root(&mut self, node: usize) -> usize {
        let parent = self.parent[node];
        if parent != node {
            self.parent[node] = self.root(parent);
        }
        self.parent[node]
    }

    fn join(&mut self, left: usize, right: usize) {
        let left = self.root(left);
        let right = self.root(right);
        if left != right {
            self.parent[right] = left;
        }
    }
}

#[cfg(test)]
mod tests;
