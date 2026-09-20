use a3s_use_core::{UseError, UseResult};
use image::{imageops, ImageBuffer, Rgb, RgbImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Projection};
use imageproc::point::Point;

use super::super::engine_error;
use crate::postprocess::Detection;

const MAX_CROP_PIXELS: u64 = 64 * 1024 * 1024;

pub(super) struct PerspectiveCropPlan {
    projection: Projection,
    width: u32,
    height: u32,
    rotate: bool,
    text_rotation_millidegrees: i32,
}

impl PerspectiveCropPlan {
    pub(super) fn new(detection: &Detection) -> UseResult<Self> {
        let polygon = detection.polygon;
        let width = distance(polygon[0], polygon[1])
            .max(distance(polygon[2], polygon[3]))
            .max(1.0) as u32;
        let height = distance(polygon[0], polygon[3])
            .max(distance(polygon[1], polygon[2]))
            .max(1.0) as u32;
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| crop_error("PP-OCRv6 text crop dimensions overflowed."))?;
        if pixels > MAX_CROP_PIXELS {
            return Err(crop_error(
                "PP-OCRv6 text crop exceeds the 64 megapixel safety limit.",
            ));
        }
        let source = polygon.map(|point| (point.x, point.y));
        let destination = [
            (0.0, 0.0),
            (width as f32, 0.0),
            (width as f32, height as f32),
            (0.0, height as f32),
        ];
        let projection = Projection::from_control_points(source, destination).ok_or_else(|| {
            crop_error("PP-OCRv6 detected a degenerate text polygon that cannot be rectified.")
        })?;
        let rotate = f64::from(height) / f64::from(width) >= 1.5;
        let reading_end = if rotate { polygon[3] } else { polygon[1] };
        let text_rotation_millidegrees = canonical_millidegrees(
            f64::from(reading_end.y - polygon[0].y)
                .atan2(f64::from(reading_end.x - polygon[0].x))
                .to_degrees(),
        );
        Ok(Self {
            projection,
            width,
            height,
            rotate,
            text_rotation_millidegrees,
        })
    }

    pub(super) fn output_dimensions(&self) -> (u32, u32) {
        if self.rotate {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        }
    }

    pub(super) fn text_rotation_millidegrees(&self) -> i32 {
        self.text_rotation_millidegrees
    }

    pub(super) fn execute(&self, image: &RgbImage) -> UseResult<RgbImage> {
        let mut crop = ImageBuffer::new(self.width, self.height);
        warp_into(
            image,
            &self.projection,
            Interpolation::Bicubic,
            Rgb([255, 255, 255]),
            &mut crop,
        );
        if self.rotate {
            Ok(imageops::rotate270(&crop))
        } else {
            Ok(crop)
        }
    }
}

#[cfg(test)]
fn perspective_crop(image: &RgbImage, detection: &Detection) -> UseResult<RgbImage> {
    PerspectiveCropPlan::new(detection)?.execute(image)
}

fn canonical_millidegrees(degrees: f64) -> i32 {
    let mut value = (degrees * 1_000.0).round() as i32;
    while value >= 180_000 {
        value -= 360_000;
    }
    while value < -180_000 {
        value += 360_000;
    }
    value
}

fn distance(left: Point<f32>, right: Point<f32>) -> f32 {
    (left.x - right.x).hypot(left.y - right.y)
}

fn crop_error(message: impl Into<String>) -> UseError {
    engine_error("use.ocr.crop_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perspective_crop_uses_the_upstream_exclusive_extent() {
        let image = RgbImage::new(20, 20);
        let detection = Detection {
            polygon: [
                Point::new(1.0, 2.0),
                Point::new(11.0, 2.0),
                Point::new(11.0, 7.0),
                Point::new(1.0, 7.0),
            ],
            confidence: 1.0,
        };
        let crop = perspective_crop(&image, &detection).unwrap();

        assert_eq!(crop.dimensions(), (10, 5));
    }

    #[test]
    fn vertical_crops_use_the_rotated_recognition_width() {
        let detection = Detection {
            polygon: [
                Point::new(1.0, 2.0),
                Point::new(11.0, 2.0),
                Point::new(11.0, 42.0),
                Point::new(1.0, 42.0),
            ],
            confidence: 1.0,
        };
        let plan = PerspectiveCropPlan::new(&detection).unwrap();

        assert_eq!(plan.output_dimensions(), (40, 10));
        assert_eq!(plan.text_rotation_millidegrees(), 90_000);
    }

    #[test]
    fn source_rotation_follows_the_exact_recognition_x_axis() {
        let detection = Detection {
            polygon: [
                Point::new(10.0, 20.0),
                Point::new(110.0, 30.0),
                Point::new(106.0, 70.0),
                Point::new(6.0, 60.0),
            ],
            confidence: 1.0,
        };
        let plan = PerspectiveCropPlan::new(&detection).unwrap();

        assert_eq!(plan.output_dimensions(), (100, 40));
        assert_eq!(plan.text_rotation_millidegrees(), 5_711);
    }
}
