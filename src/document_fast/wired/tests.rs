use super::*;
use image::{Rgb, RgbImage};

#[test]
fn admits_a_grid_and_rejects_an_isolated_page_rule() {
    let mut image = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
    draw_horizontal(&mut image, 20, 380, 25, Rgb([0, 80, 140]));
    for y in [80, 150, 240] {
        draw_horizontal(&mut image, 40, 360, y, Rgb([0, 0, 0]));
    }
    for x in [40, 180, 360] {
        draw_vertical(&mut image, x, 80, 240, Rgb([0, 0, 0]));
    }
    assert_eq!(
        candidates(&image, &CancellationToken::new()).unwrap(),
        vec![WiredCandidate {
            region: PixelRect {
                x: 40,
                y: 80,
                width: 321,
                height: 161,
            },
            inference_region: PixelRect {
                x: 40,
                y: 80,
                width: 321,
                height: 161,
            },
            orientation: TableCropOrientation::Upright,
        }]
    );
}

#[test]
fn continuation_grid_can_touch_the_top_canvas_edge() {
    let mut image = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
    for y in [0, 100, 220] {
        draw_horizontal(&mut image, 40, 360, y, Rgb([0, 0, 0]));
    }
    for x in [40, 180, 360] {
        draw_vertical(&mut image, x, 0, 220, Rgb([0, 0, 0]));
    }
    assert_eq!(
        candidates(&image, &CancellationToken::new()).unwrap()[0]
            .region
            .y,
        0
    );
}

#[test]
fn tall_grid_with_transposed_line_counts_is_rotated_for_inference() {
    let mut image = RgbImage::from_pixel(300, 540, Rgb([255, 255, 255]));
    for y in [50, 190, 330, 490] {
        draw_horizontal(&mut image, 40, 260, y, Rgb([0, 0, 0]));
    }
    for x in (40..=260).step_by(20) {
        draw_vertical(&mut image, x, 50, 490, Rgb([0, 0, 0]));
    }
    let candidates = candidates(&image, &CancellationToken::new()).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].orientation, TableCropOrientation::Rotate90);
    assert!(candidates[0].inference_region.width > candidates[0].region.width);
    assert!(candidates[0].inference_region.height > candidates[0].region.height);
}

#[test]
fn cancelled_candidate_scan_publishes_no_region() {
    let image = RgbImage::new(400, 300);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = candidates(&image, &cancellation).unwrap_err();
    assert_eq!(error.code, "use.ocr.runtime_failed");
}

#[test]
fn real_fixture_candidates_are_close_to_reviewed_table_bounds() {
    let Some(root) = std::env::var_os("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR") else {
        return;
    };
    let expected = [
        (
            "page-0002.png",
            PixelRect {
                x: 141,
                y: 391,
                width: 1390,
                height: 527,
            },
        ),
        (
            "page-0003.png",
            PixelRect {
                x: 141,
                y: 204,
                width: 1390,
                height: 696,
            },
        ),
        (
            "page-0004.png",
            PixelRect {
                x: 141,
                y: 204,
                width: 1390,
                height: 347,
            },
        ),
    ];
    for (name, reviewed) in expected {
        let image = image::open(std::path::Path::new(&root).join(name))
            .unwrap()
            .into_rgb8();
        let actual = candidates(&image, &CancellationToken::new()).unwrap();
        assert_eq!(actual.len(), 1, "{name}: {actual:?}");
        assert!(
            intersection_over_union(actual[0].region, reviewed) >= 0.97,
            "{name}: {actual:?}"
        );
        assert_eq!(actual[0].orientation, TableCropOrientation::Upright);
    }
}

fn draw_horizontal(image: &mut RgbImage, start: u32, end: u32, y: u32, color: Rgb<u8>) {
    for x in start..=end {
        image.put_pixel(x, y, color);
    }
}

fn draw_vertical(image: &mut RgbImage, x: u32, start: u32, end: u32, color: Rgb<u8>) {
    for y in start..=end {
        image.put_pixel(x, y, color);
    }
}

fn intersection_over_union(left: PixelRect, right: PixelRect) -> f32 {
    let width = left
        .right()
        .min(right.right())
        .saturating_sub(left.x.max(right.x));
    let height = left
        .bottom()
        .min(right.bottom())
        .saturating_sub(left.y.max(right.y));
    let intersection = width.saturating_mul(height);
    let union = left
        .width
        .saturating_mul(left.height)
        .saturating_add(right.width.saturating_mul(right.height))
        .saturating_sub(intersection);
    intersection as f32 / union.max(1) as f32
}
