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
            horizontal_lines: vec![80, 150, 240],
            vertical_lines: vec![40, 180, 360],
            horizontal_tracks: vec![
                LineTrack {
                    fixed: 80,
                    start: 40,
                    end: 360,
                },
                LineTrack {
                    fixed: 150,
                    start: 40,
                    end: 360,
                },
                LineTrack {
                    fixed: 240,
                    start: 40,
                    end: 360,
                },
            ],
            vertical_tracks: vec![
                LineTrack {
                    fixed: 40,
                    start: 80,
                    end: 240,
                },
                LineTrack {
                    fixed: 180,
                    start: 80,
                    end: 240,
                },
                LineTrack {
                    fixed: 360,
                    start: 80,
                    end: 240,
                },
            ],
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
fn recurring_perpendicular_endpoints_complete_an_open_outer_axis() {
    let mut image = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
    for y in [80, 150, 240] {
        draw_horizontal(&mut image, 40, 360, y, Rgb([0, 0, 0]));
    }
    for x in [40, 180] {
        draw_vertical(&mut image, x, 80, 240, Rgb([0, 0, 0]));
    }

    let candidate = candidates(&image, &CancellationToken::new()).unwrap()[0].clone();

    assert_eq!(candidate.region.x, 40);
    assert_eq!(candidate.region.right(), 361);
    assert_eq!(candidate.vertical_lines, vec![40, 180, 360]);
    assert_eq!(candidate.vertical_tracks.len(), 2);
}

#[test]
fn recurring_terminal_coordinates_are_clustered_without_page_content() {
    let tracks = [
        LineTrack {
            fixed: 40,
            start: 20,
            end: 100,
        },
        LineTrack {
            fixed: 80,
            start: 20,
            end: 102,
        },
        LineTrack {
            fixed: 120,
            start: 20,
            end: 180,
        },
        LineTrack {
            fixed: 160,
            start: 20,
            end: 180,
        },
    ];

    assert_eq!(recurring_terminal_axes(&tracks), vec![20, 101, 180]);
}

#[test]
fn only_the_nearest_recurring_terminal_closes_each_open_side() {
    let tracks = [
        LineTrack {
            fixed: 40,
            start: 20,
            end: 150,
        },
        LineTrack {
            fixed: 80,
            start: 20,
            end: 152,
        },
        LineTrack {
            fixed: 120,
            start: 30,
            end: 180,
        },
        LineTrack {
            fixed: 160,
            start: 30,
            end: 182,
        },
    ];

    assert_eq!(recurring_outer_axes(&tracks, 40, 100), vec![30, 151]);
}

#[test]
fn recurring_internal_terminals_do_not_expand_the_primitive_axis_set() {
    let mut image = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
    for y in [40, 240] {
        draw_horizontal(&mut image, 40, 360, y, Rgb([0, 0, 0]));
    }
    for y in [100, 180] {
        draw_horizontal(&mut image, 40, 260, y, Rgb([0, 0, 0]));
    }
    for x in [40, 180, 360] {
        draw_vertical(&mut image, x, 40, 240, Rgb([0, 0, 0]));
    }

    let candidate = candidates(&image, &CancellationToken::new()).unwrap()[0].clone();

    assert_eq!(candidate.vertical_lines, vec![40, 180, 360]);
}

#[test]
fn parallel_component_extension_requires_two_spacing_consistent_axes() {
    let tracks = [
        LineTrack {
            fixed: 40,
            start: 20,
            end: 220,
        },
        LineTrack {
            fixed: 60,
            start: 20,
            end: 220,
        },
        LineTrack {
            fixed: 80,
            start: 20,
            end: 220,
        },
        LineTrack {
            fixed: 100,
            start: 20,
            end: 220,
        },
        LineTrack {
            fixed: 120,
            start: 20,
            end: 220,
        },
        LineTrack {
            fixed: 140,
            start: 230,
            end: 290,
        },
    ];
    let mut admitted = [0_usize, 1, 2].into_iter().collect();

    extend_parallel_continuations(&mut admitted, &tracks);

    assert_eq!(
        admitted.into_iter().collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );

    let mut single = [0_usize, 1, 2].into_iter().collect();
    extend_parallel_continuations(&mut single, &tracks[..4]);
    assert_eq!(single.into_iter().collect::<Vec<_>>(), vec![0, 1, 2]);
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

#[test]
fn rejects_dense_periodic_texture_without_structural_line_contrast() {
    let mut image = RgbImage::from_pixel(400, 300, Rgb([255, 255, 255]));
    for y in (0..300).step_by(6) {
        draw_horizontal(&mut image, 0, 399, y, Rgb([0, 0, 0]));
    }
    for x in (0..400).step_by(6) {
        draw_vertical(&mut image, x, 0, 299, Rgb([0, 0, 0]));
    }

    let cancellation = CancellationToken::new();
    let (horizontal, vertical) = scan_segments(&image, 96, 64, &cancellation).unwrap();
    let horizontal = cluster_segments(horizontal);
    let vertical = cluster_segments(vertical);
    assert!(!connected_candidates(&horizontal, &vertical, 400, 300).is_empty());
    assert!(candidates(&image, &cancellation).unwrap().is_empty());
}

#[test]
#[ignore = "requires retained reviewed certificate-page rasters"]
fn real_certificate_texture_does_not_become_a_wired_table() {
    let root = std::env::var_os("A3S_OCR_REAL_ROTATED_TABLE_DIR")
        .expect("A3S_OCR_REAL_ROTATED_TABLE_DIR must name the retained raster root");
    for page in [26_u32, 28, 29] {
        let name = format!("page-{page:04}.png");
        let image = image::open(std::path::Path::new(&root).join(&name))
            .unwrap()
            .into_rgb8();
        assert!(
            candidates(&image, &CancellationToken::new())
                .unwrap()
                .is_empty(),
            "{name}"
        );
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
