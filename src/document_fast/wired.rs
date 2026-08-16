use std::collections::BTreeMap;

use a3s_use_core::UseResult;
use image::RgbImage;
use tokio_util::sync::CancellationToken;

use crate::cancellation::check_cancelled;

use super::orientation::TableCropOrientation;

const MAX_LINE_GAP: usize = 2;
const INTERSECTION_TOLERANCE: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PixelRect {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WiredCandidate {
    pub(super) region: PixelRect,
    pub(super) inference_region: PixelRect,
    pub(super) orientation: TableCropOrientation,
}

impl PixelRect {
    fn right(self) -> u32 {
        self.x.saturating_add(self.width)
    }

    fn bottom(self) -> u32 {
        self.y.saturating_add(self.height)
    }
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    fixed: u32,
    start: u32,
    end: u32,
}

/// Finds conservative wired-table crop candidates on the immutable page.
///
/// Long horizontal and vertical strokes are intersected as a bipartite graph.
/// A component is admitted only when at least two bands exist on both axes, so
/// page rules, underlines, and ordinary text cannot become table candidates by
/// themselves. The downstream structure model remains the authority that can
/// turn a candidate into table evidence.
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
    Ok(connected_candidates(&horizontal, &vertical, width, height))
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
) -> UseResult<(Vec<Segment>, Vec<Segment>)> {
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
                horizontal.push(Segment {
                    fixed: y,
                    start,
                    end,
                });
            }
            if let Some((start, end)) =
                vertical_runs[x as usize].advance(y, selected, minimum_vertical)
            {
                vertical.push(Segment {
                    fixed: x,
                    start,
                    end,
                });
            }
        }
        if let Some((start, end)) = horizontal_run.finish(minimum_horizontal) {
            horizontal.push(Segment {
                fixed: y,
                start,
                end,
            });
        }
    }
    for (x, run) in vertical_runs.into_iter().enumerate() {
        if let Some((start, end)) = run.finish(minimum_vertical) {
            vertical.push(Segment {
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
    gap: usize,
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
            if self.gap > MAX_LINE_GAP {
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

fn is_dark(image: &RgbImage, x: u32, y: u32) -> bool {
    let pixel = image.get_pixel(x, y).0;
    let luminance = u32::from(pixel[0]) * 77 + u32::from(pixel[1]) * 150 + u32::from(pixel[2]) * 29;
    luminance < 160 * 256
}

fn cluster_segments(segments: Vec<Segment>) -> Vec<Segment> {
    let mut clusters: Vec<Segment> = Vec::new();
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

fn overlap_ratio(left: Segment, right: Segment) -> f32 {
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
    horizontal: &[Segment],
    vertical: &[Segment],
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
        .filter_map(|component| {
            if component.horizontal.len() < 2
                || component.vertical.len() < 2
                || component.intersections < 4
            {
                return None;
            }
            let left = component
                .vertical
                .iter()
                .map(|index| vertical[*index].fixed)
                .min()?;
            let right = component
                .vertical
                .iter()
                .map(|index| vertical[*index].fixed)
                .max()?;
            let top = component
                .horizontal
                .iter()
                .map(|index| horizontal[*index].fixed)
                .min()?;
            let bottom = component
                .horizontal
                .iter()
                .map(|index| horizontal[*index].fixed)
                .max()?;
            let candidate = PixelRect {
                x: left,
                y: top,
                width: right.saturating_sub(left).saturating_add(1),
                height: bottom.saturating_sub(top).saturating_add(1),
            };
            (candidate.width >= minimum_width && candidate.height >= minimum_height).then(|| {
                let orientation = TableCropOrientation::from_grid(
                    candidate,
                    component.horizontal.len(),
                    component.vertical.len(),
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
                }
            })
        })
        .collect::<Vec<_>>();
    admitted.sort_by_key(|candidate| (candidate.region.y, candidate.region.x));
    suppress_nested(&mut admitted);
    admitted
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

fn intersects(horizontal: Segment, vertical: Segment) -> bool {
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
