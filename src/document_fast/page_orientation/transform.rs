use a3s_use_core::{UseError, UseResult};

use super::super::wired::PixelRect;
use crate::{
    OcrBoundingBox, OcrCanvasEdge, OcrImageCanvas, OcrNormalizedWindow, OcrPoint,
    OcrProviderOutput, OCR_NORMALIZED_COORDINATE_BASIS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::document_fast) struct SourceCanvasTransform {
    width: u32,
    height: u32,
    quarter_turns: u8,
}

impl SourceCanvasTransform {
    pub(in crate::document_fast) fn new(
        width: u32,
        height: u32,
        quarter_turns: u8,
    ) -> UseResult<Self> {
        if width == 0 || height == 0 || quarter_turns > 3 {
            return Err(transform_error(
                "A page-orientation transform requires positive dimensions and zero through three quarter turns.",
            ));
        }
        Ok(Self {
            width,
            height,
            quarter_turns,
        })
    }

    pub(in crate::document_fast) fn is_identity(&self) -> bool {
        self.quarter_turns == 0
    }

    pub(in crate::document_fast) fn source_canvas(&self) -> UseResult<OcrImageCanvas> {
        OcrImageCanvas::new(self.width, self.height)
    }

    pub(in crate::document_fast) fn oriented_canvas(&self) -> UseResult<OcrImageCanvas> {
        let (width, height) = self.oriented_dimensions();
        OcrImageCanvas::new(width, height)
    }

    pub(in crate::document_fast) fn restore_pixel_rect(
        &self,
        region: PixelRect,
    ) -> UseResult<PixelRect> {
        let restored = self.restore_box(OcrBoundingBox {
            x: region.x,
            y: region.y,
            width: region.width,
            height: region.height,
        })?;
        Ok(PixelRect {
            x: restored.x,
            y: restored.y,
            width: restored.width,
            height: restored.height,
        })
    }

    pub(in crate::document_fast) fn restore_canvas_edge(
        &self,
        edge: OcrCanvasEdge,
    ) -> OcrCanvasEdge {
        self.restore_edge(edge)
    }

    pub(in crate::document_fast) fn restore_output(
        &self,
        output: &mut OcrProviderOutput,
    ) -> UseResult<()> {
        for block in &mut output.blocks {
            if let Some(polygon) = &mut block.polygon {
                for point in polygon.iter_mut() {
                    *point = self.restore_point(*point)?;
                }
                block.bounding_box = Some(envelope(polygon)?);
            } else if let Some(bounds) = block.bounding_box {
                block.bounding_box = Some(self.restore_box(bounds)?);
            }
            for bounds in &mut block.bounding_boxes {
                *bounds = self.restore_box(*bounds)?;
            }
            if let Some(rotation) = block.text_rotation_millidegrees {
                block.text_rotation_millidegrees = Some(normalized_rotation(
                    rotation - i32::from(self.quarter_turns) * 90_000,
                ));
            }
        }
        Ok(())
    }

    pub(in crate::document_fast) fn orient_window(
        &self,
        window: OcrNormalizedWindow,
    ) -> UseResult<OcrNormalizedWindow> {
        let basis = OCR_NORMALIZED_COORDINATE_BASIS;
        let (left, top, right, bottom) = match self.quarter_turns {
            0 => (window.left, window.top, window.right, window.bottom),
            1 => (
                basis - window.bottom,
                window.left,
                basis - window.top,
                window.right,
            ),
            2 => (
                basis - window.right,
                basis - window.bottom,
                basis - window.left,
                basis - window.top,
            ),
            3 => (
                window.top,
                basis - window.right,
                window.bottom,
                basis - window.left,
            ),
            _ => {
                return Err(transform_error(
                    "An orientation window transform exceeded three quarter turns.",
                ));
            }
        };
        OcrNormalizedWindow::new(left, top, right, bottom)
    }

    fn restore_edge(&self, edge: OcrCanvasEdge) -> OcrCanvasEdge {
        match (self.quarter_turns, edge) {
            (0, edge) => edge,
            (1, OcrCanvasEdge::Left) => OcrCanvasEdge::Bottom,
            (1, OcrCanvasEdge::Top) => OcrCanvasEdge::Left,
            (1, OcrCanvasEdge::Right) => OcrCanvasEdge::Top,
            (1, OcrCanvasEdge::Bottom) => OcrCanvasEdge::Right,
            (2, OcrCanvasEdge::Left) => OcrCanvasEdge::Right,
            (2, OcrCanvasEdge::Top) => OcrCanvasEdge::Bottom,
            (2, OcrCanvasEdge::Right) => OcrCanvasEdge::Left,
            (2, OcrCanvasEdge::Bottom) => OcrCanvasEdge::Top,
            (3, OcrCanvasEdge::Left) => OcrCanvasEdge::Top,
            (3, OcrCanvasEdge::Top) => OcrCanvasEdge::Right,
            (3, OcrCanvasEdge::Right) => OcrCanvasEdge::Bottom,
            (3, OcrCanvasEdge::Bottom) => OcrCanvasEdge::Left,
            _ => edge,
        }
    }

    fn restore_box(&self, bounds: OcrBoundingBox) -> UseResult<OcrBoundingBox> {
        let right = bounds
            .x
            .checked_add(bounds.width)
            .ok_or_else(|| transform_error("An oriented bounding box overflowed."))?;
        let bottom = bounds
            .y
            .checked_add(bounds.height)
            .ok_or_else(|| transform_error("An oriented bounding box overflowed."))?;
        envelope(&[
            self.restore_point(OcrPoint {
                x: bounds.x,
                y: bounds.y,
            })?,
            self.restore_point(OcrPoint {
                x: right,
                y: bounds.y,
            })?,
            self.restore_point(OcrPoint {
                x: right,
                y: bottom,
            })?,
            self.restore_point(OcrPoint {
                x: bounds.x,
                y: bottom,
            })?,
        ])
    }

    fn restore_point(&self, point: OcrPoint) -> UseResult<OcrPoint> {
        let (oriented_width, oriented_height) = self.oriented_dimensions();
        if point.x > oriented_width || point.y > oriented_height {
            return Err(transform_error(
                "An oriented evidence point escaped its immutable canvas.",
            ));
        }
        let (x, y) = match self.quarter_turns {
            0 => (point.x, point.y),
            1 => (point.y, self.height - point.x),
            2 => (self.width - point.x, self.height - point.y),
            3 => (self.width - point.y, point.x),
            _ => {
                return Err(transform_error(
                    "An orientation point transform exceeded three quarter turns.",
                ));
            }
        };
        Ok(OcrPoint { x, y })
    }

    fn oriented_dimensions(&self) -> (u32, u32) {
        if self.quarter_turns % 2 == 0 {
            (self.width, self.height)
        } else {
            (self.height, self.width)
        }
    }
}

fn envelope(points: &[OcrPoint]) -> UseResult<OcrBoundingBox> {
    let left = points
        .iter()
        .map(|point| point.x)
        .min()
        .ok_or_else(|| transform_error("A restored polygon is empty."))?;
    let right = points.iter().map(|point| point.x).max().unwrap_or(left);
    let top = points.iter().map(|point| point.y).min().unwrap_or(0);
    let bottom = points.iter().map(|point| point.y).max().unwrap_or(top);
    Ok(OcrBoundingBox {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

fn normalized_rotation(rotation: i32) -> i32 {
    (rotation + 180_000).rem_euclid(360_000) - 180_000
}

fn transform_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.orientation_transform_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_quarter_turn_round_trips_boundary_corners() {
        let width = 300;
        let height = 200;
        let expected = [
            OcrPoint { x: 0, y: 0 },
            OcrPoint { x: width, y: 0 },
            OcrPoint {
                x: width,
                y: height,
            },
            OcrPoint { x: 0, y: height },
        ];
        let oriented = [
            expected,
            [
                OcrPoint { x: height, y: 0 },
                OcrPoint {
                    x: height,
                    y: width,
                },
                OcrPoint { x: 0, y: width },
                OcrPoint { x: 0, y: 0 },
            ],
            [expected[2], expected[3], expected[0], expected[1]],
            [
                OcrPoint { x: 0, y: width },
                OcrPoint { x: 0, y: 0 },
                OcrPoint { x: height, y: 0 },
                OcrPoint {
                    x: height,
                    y: width,
                },
            ],
        ];
        for turns in 0..4 {
            let transform = SourceCanvasTransform::new(width, height, turns).unwrap();
            let restored =
                oriented[usize::from(turns)].map(|point| transform.restore_point(point).unwrap());
            assert_eq!(restored, expected);
        }
    }

    #[test]
    fn text_rotation_is_expressed_on_the_source_canvas() {
        assert_eq!(normalized_rotation(0), 0);
        assert_eq!(normalized_rotation(-90_000), -90_000);
        assert_eq!(normalized_rotation(-180_000), -180_000);
        assert_eq!(normalized_rotation(-270_000), 90_000);
    }

    #[test]
    fn normalized_text_windows_follow_the_oriented_canvas() {
        let source = OcrNormalizedWindow::new(100_000, 200_000, 400_000, 700_000).unwrap();
        let expected = [
            source,
            OcrNormalizedWindow::new(300_000, 100_000, 800_000, 400_000).unwrap(),
            OcrNormalizedWindow::new(600_000, 300_000, 900_000, 800_000).unwrap(),
            OcrNormalizedWindow::new(200_000, 600_000, 700_000, 900_000).unwrap(),
        ];
        for turns in 0..4 {
            let transform = SourceCanvasTransform::new(300, 200, turns).unwrap();
            assert_eq!(
                transform.orient_window(source).unwrap(),
                expected[usize::from(turns)]
            );
        }
    }

    #[test]
    fn oriented_seal_geometry_and_clipped_edge_restore_to_source_canvas() {
        let transform = SourceCanvasTransform::new(300, 200, 1).unwrap();
        let restored = transform
            .restore_pixel_rect(PixelRect {
                x: 100,
                y: 250,
                width: 50,
                height: 50,
            })
            .unwrap();
        assert_eq!(
            restored,
            PixelRect {
                x: 250,
                y: 50,
                width: 50,
                height: 50,
            }
        );
        assert_eq!(
            transform.restore_canvas_edge(OcrCanvasEdge::Bottom),
            OcrCanvasEdge::Right
        );
    }
}
