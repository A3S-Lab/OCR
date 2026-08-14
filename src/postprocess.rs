use a3s_use_core::{UseError, UseResult};
use clipper2::{Centi, EndType, JoinType};
use image::{GrayImage, Luma};
use imageproc::contours::find_contours;
use imageproc::geometry::{contour_area, min_area_rect};
use imageproc::point::Point;

use crate::config::{DetectionConfig, RecognitionConfig};

#[derive(Debug, Clone)]
pub(crate) struct Detection {
    pub(crate) polygon: [Point<f32>; 4],
    pub(crate) confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Recognition {
    pub(crate) text: String,
    pub(crate) confidence: f32,
}

pub(crate) fn detection_boxes_in_content(
    output: &[f32],
    shape: &[usize],
    content_width: u32,
    content_height: u32,
    original_width: u32,
    original_height: u32,
    config: &DetectionConfig,
) -> UseResult<Vec<Detection>> {
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
        return Err(output_error(format!(
            "PP-OCRv6 detection output shape must be [1, 1, H, W], found {shape:?}."
        )));
    }
    let height = shape[2];
    let width = shape[3];
    let map_len = height
        .checked_mul(width)
        .ok_or_else(|| output_error("PP-OCRv6 detection output dimensions overflowed."))?;
    if width == 0 || height == 0 || output.len() != map_len {
        return Err(output_error(
            "PP-OCRv6 detection output length does not match its shape.",
        ));
    }
    let output_width = u32::try_from(width)
        .map_err(|_| output_error("PP-OCRv6 detection output width is too large."))?;
    let output_height = u32::try_from(height)
        .map_err(|_| output_error("PP-OCRv6 detection output height is too large."))?;
    if content_width == 0
        || content_height == 0
        || content_width > output_width
        || content_height > output_height
    {
        return Err(output_error(
            "PP-OCRv6 detection content extent must fit the output canvas.",
        ));
    }
    let content_width_usize = content_width as usize;
    let content_height_usize = content_height as usize;
    let mask = GrayImage::from_fn(content_width, content_height, |x, y| {
        let index = y as usize * width + x as usize;
        Luma([if output[index] > config.threshold {
            255
        } else {
            0
        }])
    });

    let mut detections = Vec::new();
    for contour in find_contours::<i32>(&mask)
        .into_iter()
        .take(config.max_candidates)
    {
        if contour.points.len() < 3 {
            continue;
        }
        let mini = order_points(min_area_rect(&contour.points));
        if minimum_side(&mini) < 3.0 {
            continue;
        }
        let score = box_score(
            output,
            width,
            content_width_usize,
            content_height_usize,
            &mini,
        );
        if score < config.box_threshold {
            continue;
        }
        let area = contour_area(&mini);
        let perimeter = polygon_perimeter(&mini);
        if !area.is_finite() || !perimeter.is_finite() || perimeter <= f64::EPSILON {
            continue;
        }
        let distance = area * f64::from(config.unclip_ratio) / perimeter;
        let path = mini
            .iter()
            .map(|point| (f64::from(point.x), f64::from(point.y)))
            .collect::<Vec<_>>();
        let inflated: Vec<Vec<(f64, f64)>> =
            clipper2::inflate::<Centi>(path, distance, JoinType::Round, EndType::Polygon, 2.0)
                .into();
        if inflated.len() != 1 || inflated[0].len() < 3 {
            continue;
        }
        let inflated = inflated[0]
            .iter()
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(x, y)| {
                Point::new(
                    x.round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32,
                    y.round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32,
                )
            })
            .collect::<Vec<_>>();
        if inflated.len() < 3 {
            continue;
        }
        let expanded = order_points(min_area_rect(&inflated));
        if minimum_side(&expanded) < 5.0 {
            continue;
        }
        let polygon = expanded.map(|point| {
            Point::new(
                (point.x as f32 / content_width as f32 * original_width as f32)
                    .round()
                    .clamp(0.0, original_width.saturating_sub(1) as f32),
                (point.y as f32 / content_height as f32 * original_height as f32)
                    .round()
                    .clamp(0.0, original_height.saturating_sub(1) as f32),
            )
        });
        detections.push(Detection {
            polygon,
            confidence: score.clamp(0.0, 1.0),
        });
    }
    sort_reading_order(&mut detections);
    Ok(detections)
}

#[cfg(test)]
pub(crate) fn decode_ctc(
    output: &[f32],
    shape: &[usize],
    config: &RecognitionConfig,
) -> UseResult<Recognition> {
    if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] == 0 {
        return Err(output_error(format!(
            "PP-OCRv6 recognition output shape must be [1, T, C], found {shape:?}."
        )));
    }
    let timesteps = shape[1];
    let classes = shape[2];
    let expected_classes = config.characters.len() + 2;
    if classes != expected_classes {
        return Err(output_error(format!(
            "PP-OCRv6 recognition class count is {classes}, but the model dictionary requires {expected_classes}."
        )));
    }
    if output.len() != timesteps.saturating_mul(classes) {
        return Err(output_error(
            "PP-OCRv6 recognition output length does not match its shape.",
        ));
    }

    let mut text = String::new();
    let mut confidence = 0.0_f32;
    let mut selected = 0_usize;
    let mut previous = usize::MAX;
    for timestep in 0..timesteps {
        let row = &output[timestep * classes..(timestep + 1) * classes];
        let (index, score) = row
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or_else(|| output_error("PP-OCRv6 recognition output row is empty."))?;
        if index != 0 && index != previous {
            if index == config.characters.len() + 1 {
                text.push(' ');
            } else if let Some(character) = config.characters.get(index - 1) {
                text.push_str(character);
            }
            confidence += score;
            selected += 1;
        }
        previous = index;
    }
    Ok(Recognition {
        text,
        confidence: if selected == 0 {
            0.0
        } else {
            (confidence / selected as f32).clamp(0.0, 1.0)
        },
    })
}

pub(crate) fn decode_ctc_top1(
    output: &[f32],
    shape: &[usize],
    config: &RecognitionConfig,
) -> UseResult<Recognition> {
    const FIELDS: usize = 3;
    if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] != FIELDS {
        return Err(output_error(format!(
            "PP-OCRv6 projected recognition output shape must be [1, T, {FIELDS}], found {shape:?}."
        )));
    }
    let timesteps = shape[1];
    if output.len() != timesteps.saturating_mul(FIELDS) {
        return Err(output_error(
            "PP-OCRv6 projected recognition output length does not match its shape.",
        ));
    }
    let classes = config.characters.len() + 2;
    let mut text = String::new();
    let mut confidence = 0.0_f32;
    let mut selected = 0_usize;
    let mut previous = usize::MAX;
    for row in output.chunks_exact(FIELDS) {
        let [index, score, finite] = [row[0], row[1], row[2]];
        if finite != 1.0 || !index.is_finite() || index < 0.0 || index.fract() != 0.0 {
            return Err(output_error(
                "PP-OCRv6 projected recognition output failed source-value validation.",
            ));
        }
        let index = index as usize;
        if index >= classes || !score.is_finite() {
            return Err(output_error(
                "PP-OCRv6 projected recognition output contains an invalid class or score.",
            ));
        }
        if index != 0 && index != previous {
            if index == config.characters.len() + 1 {
                text.push(' ');
            } else if let Some(character) = config.characters.get(index - 1) {
                text.push_str(character);
            }
            confidence += score;
            selected += 1;
        }
        previous = index;
    }
    Ok(Recognition {
        text,
        confidence: if selected == 0 {
            0.0
        } else {
            (confidence / selected as f32).clamp(0.0, 1.0)
        },
    })
}

fn box_score(
    output: &[f32],
    output_stride: usize,
    width: usize,
    height: usize,
    polygon: &[Point<i32>; 4],
) -> f32 {
    let min_x = polygon
        .iter()
        .map(|point| point.x)
        .min()
        .unwrap_or(0)
        .clamp(0, width.saturating_sub(1) as i32) as usize;
    let max_x = polygon
        .iter()
        .map(|point| point.x)
        .max()
        .unwrap_or(0)
        .clamp(0, width.saturating_sub(1) as i32) as usize;
    let min_y = polygon
        .iter()
        .map(|point| point.y)
        .min()
        .unwrap_or(0)
        .clamp(0, height.saturating_sub(1) as i32) as usize;
    let max_y = polygon
        .iter()
        .map(|point| point.y)
        .max()
        .unwrap_or(0)
        .clamp(0, height.saturating_sub(1) as i32) as usize;
    let polygon = polygon.map(|point| Point::new(point.x as f32, point.y as f32));
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            // Paddle's `box_score_fast` rasterizes the integer polygon with
            // `fillPoly` and averages every covered pixel, including its
            // boundary. Sampling integer pixel coordinates reproduces that
            // contract more closely than sampling pixel centers.
            if point_in_convex_polygon(Point::new(x as f32, y as f32), &polygon) {
                sum += output[y * output_stride + x];
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn point_in_convex_polygon(point: Point<f32>, polygon: &[Point<f32>; 4]) -> bool {
    let mut sign = 0_i8;
    for index in 0..4 {
        let start = polygon[index];
        let end = polygon[(index + 1) % 4];
        let cross =
            (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x);
        if cross.abs() <= f32::EPSILON {
            continue;
        }
        let current = if cross > 0.0 { 1 } else { -1 };
        if sign != 0 && sign != current {
            return false;
        }
        sign = current;
    }
    true
}

fn minimum_side(points: &[Point<i32>; 4]) -> f64 {
    (0..4)
        .map(|index| distance(points[index], points[(index + 1) % 4]))
        .fold(f64::INFINITY, f64::min)
}

fn polygon_perimeter(points: &[Point<i32>; 4]) -> f64 {
    (0..4)
        .map(|index| distance(points[index], points[(index + 1) % 4]))
        .sum()
}

fn distance(left: Point<i32>, right: Point<i32>) -> f64 {
    let x = f64::from(left.x - right.x);
    let y = f64::from(left.y - right.y);
    x.hypot(y)
}

fn order_points(mut points: [Point<i32>; 4]) -> [Point<i32>; 4] {
    points.sort_by(|left, right| left.x.cmp(&right.x).then(left.y.cmp(&right.y)));
    let (top_left, bottom_left) = if points[0].y <= points[1].y {
        (points[0], points[1])
    } else {
        (points[1], points[0])
    };
    let (top_right, bottom_right) = if points[2].y <= points[3].y {
        (points[2], points[3])
    } else {
        (points[3], points[2])
    };
    [top_left, top_right, bottom_right, bottom_left]
}

fn sort_reading_order(detections: &mut [Detection]) {
    detections.sort_by(|left, right| {
        left.polygon[0]
            .y
            .total_cmp(&right.polygon[0].y)
            .then_with(|| left.polygon[0].x.total_cmp(&right.polygon[0].x))
    });
    for index in 1..detections.len() {
        let mut cursor = index;
        while cursor > 0 {
            let current = detections[cursor].polygon[0];
            let previous = detections[cursor - 1].polygon[0];
            // Keep the reviewed pipeline's ten-pixel row tolerance after
            // integer coordinate projection. Native geometry can differ from
            // OpenCV by one rounding unit at the boundary, so include ten.
            if (current.y - previous.y).abs() <= 10.0 && current.x < previous.x {
                detections.swap(cursor, cursor - 1);
                cursor -= 1;
            } else {
                break;
            }
        }
    }
}

fn output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_output_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection_config() -> DetectionConfig {
        DetectionConfig {
            scale: 1.0 / 255.0,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            threshold: 0.3,
            box_threshold: 0.5,
            max_candidates: 1_000,
            unclip_ratio: 1.0,
        }
    }

    #[test]
    fn ctc_decoder_removes_blanks_and_repeated_classes() {
        let config = RecognitionConfig {
            channels: 3,
            height: 48,
            default_width: 320,
            characters: vec!["A".to_string(), "B".to_string()],
        };
        let output = [
            0.9, 0.1, 0.0, 0.0, // blank
            0.1, 0.8, 0.1, 0.0, // A
            0.9, 0.1, 0.0, 0.0, // blank
            0.1, 0.8, 0.1, 0.0, // A
            0.1, 0.1, 0.8, 0.0, // B
        ];
        let result = decode_ctc(&output, &[1, 5, 4], &config).unwrap();
        assert_eq!(result.text, "AAB");
        assert!((result.confidence - 0.8).abs() < f32::EPSILON);

        let projected = [
            0.0, 0.9, 1.0, // blank
            1.0, 0.8, 1.0, // A
            0.0, 0.9, 1.0, // blank
            1.0, 0.8, 1.0, // A
            2.0, 0.8, 1.0, // B
        ];
        let projected_result = decode_ctc_top1(&projected, &[1, 5, 3], &config).unwrap();
        assert_eq!(projected_result, result);
    }

    #[test]
    fn projected_ctc_decoder_rejects_nonfinite_source_markers_and_class_indices() {
        let config = RecognitionConfig {
            channels: 3,
            height: 48,
            default_width: 320,
            characters: vec!["A".to_string(), "B".to_string()],
        };

        assert!(decode_ctc_top1(&[1.0, 0.8, 0.0], &[1, 1, 3], &config).is_err());
        assert!(decode_ctc_top1(&[4.0, 0.8, 1.0], &[1, 1, 3], &config).is_err());
        assert!(decode_ctc_top1(&[1.5, 0.8, 1.0], &[1, 1, 3], &config).is_err());
    }

    #[test]
    fn fast_box_score_includes_integer_polygon_boundaries() {
        let output = (0..9).map(|value| value as f32).collect::<Vec<_>>();
        let polygon = [
            Point::new(1, 0),
            Point::new(2, 1),
            Point::new(1, 2),
            Point::new(0, 1),
        ];
        assert!((box_score(&output, 3, 3, 3, &polygon) - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reading_order_includes_the_integer_ten_pixel_row_boundary() {
        let mut detections = [
            Detection {
                polygon: [
                    Point::new(100.0, 0.0),
                    Point::new(110.0, 0.0),
                    Point::new(110.0, 5.0),
                    Point::new(100.0, 5.0),
                ],
                confidence: 1.0,
            },
            Detection {
                polygon: [
                    Point::new(0.0, 10.0),
                    Point::new(10.0, 10.0),
                    Point::new(10.0, 15.0),
                    Point::new(0.0, 15.0),
                ],
                confidence: 1.0,
            },
        ];
        sort_reading_order(&mut detections);
        assert_eq!(detections[0].polygon[0], Point::new(0.0, 10.0));
    }

    #[test]
    fn letterbox_padding_never_produces_detection_boxes() {
        let mut output = vec![0.0_f32; 64 * 64];
        for y in 8..24 {
            for x in 40..56 {
                output[y * 64 + x] = 1.0;
            }
        }

        let detections = detection_boxes_in_content(
            &output,
            &[1, 1, 64, 64],
            32,
            32,
            320,
            320,
            &detection_config(),
        )
        .unwrap();

        assert!(detections.is_empty());
    }

    #[test]
    fn detection_coordinates_use_each_slots_content_extent() {
        let mut output = vec![0.0_f32; 64 * 64];
        for y in 20..29 {
            for x in 24..31 {
                output[y * 64 + x] = 1.0;
            }
        }

        let detections = detection_boxes_in_content(
            &output,
            &[1, 1, 64, 64],
            32,
            32,
            320,
            320,
            &detection_config(),
        )
        .unwrap();

        assert_eq!(detections.len(), 1);
        assert!(
            detections[0]
                .polygon
                .iter()
                .map(|point| point.x)
                .fold(0.0_f32, f32::max)
                > 200.0
        );
    }
}
