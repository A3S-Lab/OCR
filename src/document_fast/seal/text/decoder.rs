use imageproc::point::Point;

use crate::document_fast::wired::PixelRect;
use crate::postprocess::Detection;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::document_fast::seal) struct SealTextObservation {
    pub(in crate::document_fast::seal) region: PixelRect,
    pub(in crate::document_fast::seal) confidence: f32,
}

pub(super) fn direct_observations(
    detections: &[Detection],
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<SealTextObservation> {
    detections
        .iter()
        .filter_map(|detection| {
            detection_bounds(detection, canvas_width, canvas_height).map(|region| {
                SealTextObservation {
                    region,
                    confidence: detection.confidence,
                }
            })
        })
        .collect()
}

pub(super) fn orthogonal_observations(
    detections: &[Detection],
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<SealTextObservation> {
    rotated_observations(detections, canvas_width, canvas_height, 1)
}

pub(super) fn clockwise180_observations(
    detections: &[Detection],
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<SealTextObservation> {
    rotated_observations(detections, canvas_width, canvas_height, 2)
}

pub(super) fn clockwise270_observations(
    detections: &[Detection],
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<SealTextObservation> {
    rotated_observations(detections, canvas_width, canvas_height, 3)
}

fn rotated_observations(
    detections: &[Detection],
    canvas_width: u32,
    canvas_height: u32,
    quarter_turns: u8,
) -> Vec<SealTextObservation> {
    detections
        .iter()
        .filter_map(|detection| {
            let restored = Detection {
                polygon: detection.polygon.map(|point| match quarter_turns {
                    1 => Point::new(
                        point.y.clamp(0.0, canvas_width.saturating_sub(1) as f32),
                        (canvas_height.saturating_sub(1) as f32 - point.x)
                            .clamp(0.0, canvas_height.saturating_sub(1) as f32),
                    ),
                    2 => Point::new(
                        (canvas_width.saturating_sub(1) as f32 - point.x)
                            .clamp(0.0, canvas_width.saturating_sub(1) as f32),
                        (canvas_height.saturating_sub(1) as f32 - point.y)
                            .clamp(0.0, canvas_height.saturating_sub(1) as f32),
                    ),
                    3 => Point::new(
                        (canvas_width.saturating_sub(1) as f32 - point.y)
                            .clamp(0.0, canvas_width.saturating_sub(1) as f32),
                        point.x.clamp(0.0, canvas_height.saturating_sub(1) as f32),
                    ),
                    _ => point,
                }),
                confidence: detection.confidence,
            };
            detection_bounds(&restored, canvas_width, canvas_height).map(|region| {
                SealTextObservation {
                    region,
                    confidence: detection.confidence,
                }
            })
        })
        .collect()
}

fn detection_bounds(
    detection: &Detection,
    canvas_width: u32,
    canvas_height: u32,
) -> Option<PixelRect> {
    if canvas_width == 0
        || canvas_height == 0
        || !detection.confidence.is_finite()
        || detection
            .polygon
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return None;
    }
    let left = detection
        .polygon
        .iter()
        .map(|point| point.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, canvas_width.saturating_sub(1) as f32) as u32;
    let top = detection
        .polygon
        .iter()
        .map(|point| point.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, canvas_height.saturating_sub(1) as f32) as u32;
    let right = detection
        .polygon
        .iter()
        .map(|point| point.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp((left + 1) as f32, canvas_width as f32) as u32;
    let bottom = detection
        .polygon
        .iter()
        .map(|point| point.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp((top + 1) as f32, canvas_height as f32) as u32;
    (right > left && bottom > top).then_some(PixelRect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(points: [(f32, f32); 4]) -> Detection {
        Detection {
            polygon: points.map(|(x, y)| Point::new(x, y)),
            confidence: 0.8,
        }
    }

    #[test]
    fn direct_geometry_stays_on_the_immutable_canvas() {
        let observations = direct_observations(
            &[detection([
                (100.25, 200.5),
                (300.75, 200.5),
                (300.75, 400.5),
                (100.25, 400.5),
            ])],
            1_200,
            1_600,
        );
        assert_eq!(
            observations[0].region,
            PixelRect {
                x: 100,
                y: 200,
                width: 201,
                height: 201,
            }
        );
    }

    #[test]
    fn clockwise_rotation_is_restored_to_source_coordinates() {
        // Source box x=100..300, y=200..400 becomes rotated box
        // x=1199..1399, y=100..300 on a 1200x1600 source canvas.
        let observations = orthogonal_observations(
            &[detection([
                (1_199.0, 100.0),
                (1_399.0, 100.0),
                (1_399.0, 300.0),
                (1_199.0, 300.0),
            ])],
            1_200,
            1_600,
        );
        assert_eq!(
            observations[0].region,
            PixelRect {
                x: 100,
                y: 200,
                width: 200,
                height: 200,
            }
        );
    }

    #[test]
    fn every_non_identity_quarter_turn_restores_the_same_source_box() {
        let expected = PixelRect {
            x: 100,
            y: 200,
            width: 200,
            height: 200,
        };
        let rotated180 = detection([
            (1_099.0, 1_399.0),
            (899.0, 1_399.0),
            (899.0, 1_199.0),
            (1_099.0, 1_199.0),
        ]);
        let rotated270 = detection([
            (200.0, 1_099.0),
            (400.0, 1_099.0),
            (400.0, 899.0),
            (200.0, 899.0),
        ]);

        assert_eq!(
            clockwise180_observations(&[rotated180], 1_200, 1_600)[0].region,
            expected
        );
        assert_eq!(
            clockwise270_observations(&[rotated270], 1_200, 1_600)[0].region,
            expected
        );
    }
}
