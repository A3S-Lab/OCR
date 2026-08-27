use std::collections::BTreeMap;

use a3s_use_core::UseResult;
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use crate::cancellation::check_cancelled;

use super::orientation::TableCropOrientation;

pub(super) const LINE_GAP_TOLERANCE: u32 = 2;
pub(super) const INTERSECTION_TOLERANCE: u32 = 3;
const AXIS_ALIGNMENT_TOLERANCE: u32 = INTERSECTION_TOLERANCE + LINE_GAP_TOLERANCE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PixelRect {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WiredCandidate {
    pub(super) region: PixelRect,
    pub(super) inference_region: PixelRect,
    pub(super) orientation: TableCropOrientation,
    pub(super) horizontal_lines: Vec<u32>,
    pub(super) vertical_lines: Vec<u32>,
    pub(super) horizontal_tracks: Vec<LineTrack>,
    pub(super) vertical_tracks: Vec<LineTrack>,
}

impl PixelRect {
    fn right(self) -> u32 {
        self.x.saturating_add(self.width)
    }

    fn bottom(self) -> u32 {
        self.y.saturating_add(self.height)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LineTrack {
    pub(super) fixed: u32,
    pub(super) start: u32,
    pub(super) end: u32,
}

/// Finds conservative wired-table crop candidates on the immutable page.
///
/// Long horizontal and vertical strokes are intersected as a bipartite graph.
/// A component is admitted only when at least two bands exist on both axes, so
/// page rules, underlines, and ordinary text cannot become table candidates by
/// themselves. A candidate alone is never evidence: exact source-pixel
/// topology may publish only after structural proof, while every ambiguous
/// partition falls back to the downstream structure model.
pub(super) fn candidates(
    image: &RgbImage,
    cancellation: &CancellationToken,
) -> UseResult<Vec<WiredCandidate>> {
    check_cancelled(cancellation)?;
    let width = image.width();
    let height = image.height();
    if width < 96 || height < 96 {
        return Ok(Vec::new());
    }
    let minimum_horizontal = (width / 5).max(96);
    let minimum_vertical = (height / 12).max(64);
    let (horizontal, vertical) =
        scan_segments(image, minimum_horizontal, minimum_vertical, cancellation)?;
    let horizontal = cluster_segments(horizontal);
    let vertical = cluster_segments(vertical);
    check_cancelled(cancellation)?;
    connected_candidates(&horizontal, &vertical, width, height)
        .into_iter()
        .filter_map(
            |candidate| match candidate_has_salient_axes(image, &candidate, cancellation) {
                Ok(true) => Some(Ok(candidate)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

/// A structural rule must be locally darker than its parallel background.
///
/// Dense page texture can create long runs in both directions without forming
/// a semantic grid. Requiring a strict majority of retained tracks on each
/// axis to have majority-dark signal and majority-light parallel background
/// rejects that texture without consulting page content or provider identity.
fn candidate_has_salient_axes(
    image: &RgbImage,
    candidate: &WiredCandidate,
    cancellation: &CancellationToken,
) -> UseResult<bool> {
    Ok(axis_has_salient_majority(
        image,
        &candidate.horizontal_tracks,
        false,
        candidate.region,
        cancellation,
    )? && axis_has_salient_majority(
        image,
        &candidate.vertical_tracks,
        true,
        candidate.region,
        cancellation,
    )?)
}

fn axis_has_salient_majority(
    image: &RgbImage,
    tracks: &[LineTrack],
    vertical: bool,
    region: PixelRect,
    cancellation: &CancellationToken,
) -> UseResult<bool> {
    let mut salient = 0_usize;
    for track in tracks {
        check_cancelled(cancellation)?;
        salient += usize::from(track_is_locally_salient(image, *track, vertical, region));
    }
    Ok(salient.saturating_mul(2) > tracks.len())
}

fn track_is_locally_salient(
    image: &RgbImage,
    track: LineTrack,
    vertical: bool,
    region: PixelRect,
) -> bool {
    let background_offset = INTERSECTION_TOLERANCE
        .saturating_add(LINE_GAP_TOLERANCE)
        .saturating_add(1);
    let mut signal = 0_u64;
    let mut samples = 0_u64;
    let mut background = 0_u64;
    let mut background_samples = 0_u64;
    for position in track.start..=track.end {
        let (x, y) = if vertical {
            (track.fixed, position)
        } else {
            (position, track.fixed)
        };
        signal += u64::from(is_dark(image, x, y));
        samples += 1;
        for fixed in [
            track.fixed.checked_sub(background_offset),
            track.fixed.checked_add(background_offset),
        ]
        .into_iter()
        .flatten()
        {
            let inside = if vertical {
                fixed >= region.x && fixed < region.right()
            } else {
                fixed >= region.y && fixed < region.bottom()
            };
            if inside {
                let (x, y) = if vertical {
                    (fixed, position)
                } else {
                    (position, fixed)
                };
                background += u64::from(is_dark(image, x, y));
                background_samples += 1;
            }
        }
    }
    signal.saturating_mul(2) > samples
        && background_samples > 0
        && background.saturating_mul(2) < background_samples
}

/// Extracts both line directions in one row-major image pass.
///
/// Reading the RGB canvas again by columns made the vertical scan stride across
/// cache lines. Keeping one run state per column preserves the exact gap rules
/// while each source pixel is classified only once and in storage order.
fn scan_segments(
    image: &RgbImage,
    minimum_horizontal: u32,
    minimum_vertical: u32,
    cancellation: &CancellationToken,
) -> UseResult<(Vec<LineTrack>, Vec<LineTrack>)> {
    let mut horizontal = Vec::new();
    let mut vertical = Vec::new();
    let mut vertical_runs = vec![RunState::default(); image.width() as usize];
    for y in 0..image.height() {
        if y % 32 == 0 {
            check_cancelled(cancellation)?;
        }
        let mut horizontal_run = RunState::default();
        for x in 0..image.width() {
            let selected = is_dark(image, x, y);
            if let Some((start, end)) = horizontal_run.advance(x, selected, minimum_horizontal) {
                horizontal.push(LineTrack {
                    fixed: y,
                    start,
                    end,
                });
            }
            if let Some((start, end)) =
                vertical_runs[x as usize].advance(y, selected, minimum_vertical)
            {
                vertical.push(LineTrack {
                    fixed: x,
                    start,
                    end,
                });
            }
        }
        if let Some((start, end)) = horizontal_run.finish(minimum_horizontal) {
            horizontal.push(LineTrack {
                fixed: y,
                start,
                end,
            });
        }
    }
    for (x, run) in vertical_runs.into_iter().enumerate() {
        if let Some((start, end)) = run.finish(minimum_vertical) {
            vertical.push(LineTrack {
                fixed: x as u32,
                start,
                end,
            });
        }
    }
    vertical.sort_by_key(|segment| (segment.fixed, segment.start));
    Ok((horizontal, vertical))
}

#[derive(Clone, Default)]
struct RunState {
    start: Option<u32>,
    last_selected: u32,
    gap: u32,
}

impl RunState {
    fn advance(&mut self, index: u32, selected: bool, minimum: u32) -> Option<(u32, u32)> {
        if selected {
            self.start.get_or_insert(index);
            self.last_selected = index;
            self.gap = 0;
            None
        } else if self.start.is_some() {
            self.gap += 1;
            if self.gap > LINE_GAP_TOLERANCE {
                self.gap = 0;
                let first = self.start.take()?;
                return (self.last_selected.saturating_sub(first).saturating_add(1) >= minimum)
                    .then_some((first, self.last_selected));
            }
            None
        } else {
            None
        }
    }

    fn finish(mut self, minimum: u32) -> Option<(u32, u32)> {
        let first = self.start.take()?;
        (self.last_selected.saturating_sub(first).saturating_add(1) >= minimum)
            .then_some((first, self.last_selected))
    }
}

pub(super) fn is_dark(image: &RgbImage, x: u32, y: u32) -> bool {
    let pixel = image.get_pixel(x, y).0;
    let luminance = u32::from(pixel[0]) * 77 + u32::from(pixel[1]) * 150 + u32::from(pixel[2]) * 29;
    luminance < 160 * 256
}

fn cluster_segments(segments: Vec<LineTrack>) -> Vec<LineTrack> {
    let mut clusters: Vec<LineTrack> = Vec::new();
    for segment in segments {
        if let Some(previous) = clusters.last_mut() {
            if segment.fixed <= previous.fixed.saturating_add(2)
                && overlap_ratio(*previous, segment) >= 0.7
            {
                previous.fixed = midpoint(previous.fixed, segment.fixed);
                previous.start = previous.start.min(segment.start);
                previous.end = previous.end.max(segment.end);
                continue;
            }
        }
        clusters.push(segment);
    }
    clusters
}

fn overlap_ratio(left: LineTrack, right: LineTrack) -> f32 {
    let overlap = left
        .end
        .min(right.end)
        .saturating_sub(left.start.max(right.start))
        .saturating_add(1);
    let shorter = left
        .end
        .saturating_sub(left.start)
        .saturating_add(1)
        .min(right.end.saturating_sub(right.start).saturating_add(1));
    overlap as f32 / shorter.max(1) as f32
}

fn midpoint(left: u32, right: u32) -> u32 {
    left.saturating_add(right).saturating_div(2)
}

fn connected_candidates(
    horizontal: &[LineTrack],
    vertical: &[LineTrack],
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<WiredCandidate> {
    let total = horizontal.len().saturating_add(vertical.len());
    let mut union = UnionFind::new(total);
    let mut intersections = Vec::new();
    for (horizontal_index, horizontal_segment) in horizontal.iter().enumerate() {
        for (vertical_index, vertical_segment) in vertical.iter().enumerate() {
            if intersects(*horizontal_segment, *vertical_segment) {
                let vertical_node = horizontal.len() + vertical_index;
                union.join(horizontal_index, vertical_node);
                intersections.push((horizontal_index, vertical_node));
            }
        }
    }

    let mut components: BTreeMap<usize, Component> = BTreeMap::new();
    for (horizontal_index, vertical_node) in intersections {
        let root = union.root(horizontal_index);
        let component = components.entry(root).or_default();
        component.horizontal.insert(horizontal_index);
        component.vertical.insert(vertical_node - horizontal.len());
        component.intersections += 1;
    }

    let minimum_width = (canvas_width / 5).max(96);
    let minimum_height = (canvas_height / 20).max(64);
    let mut admitted = components
        .into_values()
        .filter_map(|mut component| {
            if component.horizontal.len() < 2
                || component.vertical.len() < 2
                || component.intersections < 4
            {
                return None;
            }
            extend_parallel_continuations(&mut component.horizontal, horizontal);
            extend_parallel_continuations(&mut component.vertical, vertical);
            let mut horizontal_tracks = component
                .horizontal
                .iter()
                .map(|index| horizontal[*index])
                .collect::<Vec<_>>();
            let mut vertical_tracks = component
                .vertical
                .iter()
                .map(|index| vertical[*index])
                .collect::<Vec<_>>();
            horizontal_tracks.sort_by_key(|track| (track.fixed, track.start, track.end));
            vertical_tracks.sort_by_key(|track| (track.fixed, track.start, track.end));
            let axis_left = component
                .vertical
                .iter()
                .map(|index| vertical[*index].fixed)
                .min()?;
            let axis_right = component
                .vertical
                .iter()
                .map(|index| vertical[*index].fixed)
                .max()?;
            let axis_top = component
                .horizontal
                .iter()
                .map(|index| horizontal[*index].fixed)
                .min()?;
            let axis_bottom = component
                .horizontal
                .iter()
                .map(|index| horizontal[*index].fixed)
                .max()?;
            let vertical_terminal_axes =
                recurring_outer_axes(&horizontal_tracks, axis_left, axis_right);
            let horizontal_terminal_axes =
                recurring_outer_axes(&vertical_tracks, axis_top, axis_bottom);
            let left = vertical_terminal_axes
                .first()
                .map_or(axis_left, |coordinate| axis_left.min(*coordinate));
            let right = vertical_terminal_axes
                .last()
                .map_or(axis_right, |coordinate| axis_right.max(*coordinate));
            let top = horizontal_terminal_axes
                .first()
                .map_or(axis_top, |coordinate| axis_top.min(*coordinate));
            let bottom = horizontal_terminal_axes
                .last()
                .map_or(axis_bottom, |coordinate| axis_bottom.max(*coordinate));
            let candidate = PixelRect {
                x: left,
                y: top,
                width: right.saturating_sub(left).saturating_add(1),
                height: bottom.saturating_sub(top).saturating_add(1),
            };
            (candidate.width >= minimum_width && candidate.height >= minimum_height).then(|| {
                let mut horizontal_lines = component
                    .horizontal
                    .iter()
                    .map(|index| horizontal[*index].fixed)
                    .collect::<Vec<_>>();
                let mut vertical_lines = component
                    .vertical
                    .iter()
                    .map(|index| vertical[*index].fixed)
                    .collect::<Vec<_>>();
                for coordinate in horizontal_terminal_axes {
                    add_distinct_axis(&mut horizontal_lines, coordinate);
                }
                for coordinate in vertical_terminal_axes {
                    add_distinct_axis(&mut vertical_lines, coordinate);
                }
                horizontal_lines.sort_unstable();
                vertical_lines.sort_unstable();
                let orientation = TableCropOrientation::from_grid(
                    candidate,
                    horizontal_lines.len(),
                    vertical_lines.len(),
                );
                WiredCandidate {
                    region: candidate,
                    inference_region: inference_region(
                        candidate,
                        orientation,
                        canvas_width,
                        canvas_height,
                    ),
                    orientation,
                    horizontal_lines,
                    vertical_lines,
                    horizontal_tracks,
                    vertical_tracks,
                }
            })
        })
        .collect::<Vec<_>>();
    admitted.sort_by_key(|candidate| (candidate.region.y, candidate.region.x));
    suppress_nested(&mut admitted);
    admitted
}

fn extend_parallel_continuations(
    admitted: &mut std::collections::BTreeSet<usize>,
    tracks: &[LineTrack],
) {
    let Some((minimum, maximum, typical_gap)) = axis_extent_and_typical_gap(admitted, tracks)
    else {
        return;
    };
    for extreme in [Extreme::Minimum, Extreme::Maximum] {
        let edge = match extreme {
            Extreme::Minimum => minimum,
            Extreme::Maximum => maximum,
        };
        let mut candidates = tracks
            .iter()
            .enumerate()
            .filter(|(index, track)| {
                !admitted.contains(index)
                    && match extreme {
                        Extreme::Minimum => track.fixed < edge,
                        Extreme::Maximum => track.fixed > edge,
                    }
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(_, track)| track.fixed);
        if matches!(extreme, Extreme::Minimum) {
            candidates.reverse();
        }

        let mut comparison = admitted
            .iter()
            .map(|index| tracks[*index])
            .collect::<Vec<_>>();
        let mut previous = edge;
        let mut continuation = Vec::new();
        for (index, track) in candidates {
            let gap = previous.abs_diff(track.fixed);
            if gap > typical_gap.saturating_add(AXIS_ALIGNMENT_TOLERANCE) {
                break;
            }
            if comparison
                .iter()
                .any(|existing| overlap_ratio(*existing, *track) >= 0.7)
            {
                previous = track.fixed;
                comparison.push(*track);
                continuation.push(index);
            }
        }
        let distinct_axes =
            normalized_axis_coordinates(continuation.iter().map(|index| tracks[*index].fixed));
        if distinct_axes.len() >= 2 {
            admitted.extend(continuation);
        }
    }
}

fn axis_extent_and_typical_gap(
    admitted: &std::collections::BTreeSet<usize>,
    tracks: &[LineTrack],
) -> Option<(u32, u32, u32)> {
    let coordinates =
        normalized_axis_coordinates(admitted.iter().map(|index| tracks[*index].fixed));
    let minimum = *coordinates.first()?;
    let maximum = *coordinates.last()?;
    let mut gaps = coordinates
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .collect::<Vec<_>>();
    gaps.sort_unstable();
    Some((minimum, maximum, gaps.get(gaps.len() / 2).copied()?))
}

fn normalized_axis_coordinates(coordinates: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut coordinates = coordinates.collect::<Vec<_>>();
    coordinates.sort_unstable();
    let mut normalized = Vec::<u32>::new();
    for coordinate in coordinates {
        if let Some(previous) = normalized.last_mut() {
            if coordinate <= previous.saturating_add(AXIS_ALIGNMENT_TOLERANCE) {
                *previous = midpoint(*previous, coordinate);
                continue;
            }
        }
        normalized.push(coordinate);
    }
    normalized
}

#[derive(Clone, Copy)]
enum TrackEndpoint {
    Start,
    End,
}

#[derive(Clone, Copy)]
enum Extreme {
    Minimum,
    Maximum,
}

fn recurring_terminal_axes(tracks: &[LineTrack]) -> Vec<u32> {
    let mut axes = recurring_track_terminals(tracks, TrackEndpoint::Start);
    axes.extend(recurring_track_terminals(tracks, TrackEndpoint::End));
    normalized_axis_coordinates(axes.into_iter())
}

fn recurring_outer_axes(tracks: &[LineTrack], minimum: u32, maximum: u32) -> Vec<u32> {
    let axes = recurring_terminal_axes(tracks);
    let before = axes
        .iter()
        .copied()
        .filter(|coordinate| *coordinate < minimum)
        .max();
    let after = axes
        .into_iter()
        .filter(|coordinate| *coordinate > maximum)
        .min();
    before.into_iter().chain(after).collect()
}

fn recurring_track_terminals(tracks: &[LineTrack], endpoint: TrackEndpoint) -> Vec<u32> {
    let mut coordinates = tracks
        .iter()
        .map(|track| match endpoint {
            TrackEndpoint::Start => track.start,
            TrackEndpoint::End => track.end,
        })
        .collect::<Vec<_>>();
    coordinates.sort_unstable();
    let mut clusters = Vec::<(u32, u32, usize)>::new();
    for coordinate in coordinates {
        if let Some((_, last, count)) = clusters.last_mut() {
            if coordinate <= last.saturating_add(AXIS_ALIGNMENT_TOLERANCE) {
                *last = coordinate;
                *count += 1;
                continue;
            }
        }
        clusters.push((coordinate, coordinate, 1));
    }
    clusters
        .into_iter()
        .filter_map(|(first, last, count)| (count >= 2).then_some(midpoint(first, last)))
        .collect()
}

fn add_distinct_axis(lines: &mut Vec<u32>, coordinate: u32) {
    if lines
        .iter()
        .all(|line| line.abs_diff(coordinate) > AXIS_ALIGNMENT_TOLERANCE)
    {
        lines.push(coordinate);
    }
}

fn inference_region(
    region: PixelRect,
    orientation: TableCropOrientation,
    canvas_width: u32,
    canvas_height: u32,
) -> PixelRect {
    if orientation == TableCropOrientation::Upright {
        return region;
    }
    let padding = (region.width.min(region.height) / 24).clamp(4, 32);
    let left = region.x.saturating_sub(padding);
    let top = region.y.saturating_sub(padding);
    let right = region.right().saturating_add(padding).min(canvas_width);
    let bottom = region.bottom().saturating_add(padding).min(canvas_height);
    PixelRect {
        x: left,
        y: top,
        width: right.saturating_sub(left),
        height: bottom.saturating_sub(top),
    }
}

fn intersects(horizontal: LineTrack, vertical: LineTrack) -> bool {
    within(vertical.fixed, horizontal.start, horizontal.end)
        && within(horizontal.fixed, vertical.start, vertical.end)
}

fn within(value: u32, start: u32, end: u32) -> bool {
    value.saturating_add(INTERSECTION_TOLERANCE) >= start
        && value <= end.saturating_add(INTERSECTION_TOLERANCE)
}

fn suppress_nested(candidates: &mut Vec<WiredCandidate>) {
    let snapshot = candidates.clone();
    candidates.retain(|candidate| {
        !snapshot.iter().any(|other| {
            candidate != other
                && other.region.x <= candidate.region.x
                && other.region.y <= candidate.region.y
                && other.region.right() >= candidate.region.right()
                && other.region.bottom() >= candidate.region.bottom()
        })
    });
}

#[derive(Default)]
struct Component {
    horizontal: std::collections::BTreeSet<usize>,
    vertical: std::collections::BTreeSet<usize>,
    intersections: usize,
}

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
