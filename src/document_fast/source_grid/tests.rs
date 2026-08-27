use image::{Rgb, RgbImage};

use super::*;
use crate::document_fast::wired::{LineTrack, PixelRect};

type ExpectedTopology = (TableCropOrientation, Option<(u32, u32, usize)>);

#[test]
fn exact_source_strokes_recover_rectangular_row_and_column_spans() {
    let mut image = RgbImage::from_pixel(111, 81, Rgb([255, 255, 255]));
    horizontal(&mut image, 10, 10, 100);
    horizontal(&mut image, 40, 40, 100);
    horizontal(&mut image, 70, 10, 100);
    vertical(&mut image, 10, 10, 70);
    vertical(&mut image, 40, 10, 70);
    vertical(&mut image, 70, 40, 70);
    vertical(&mut image, 100, 10, 70);
    let candidate = candidate();

    let grid = derive_source_grid(&image, &candidate).unwrap();

    assert_eq!((grid.row_count, grid.column_count), (2, 3));
    assert_eq!(grid.confidence, None);
    assert_eq!(
        grid.cells
            .iter()
            .map(|cell| (cell.row, cell.column, cell.row_span, cell.column_span))
            .collect::<Vec<_>>(),
        vec![(0, 0, 2, 1), (0, 1, 1, 2), (1, 1, 1, 1), (1, 2, 1, 1)]
    );
    assert_eq!(grid.cells[1].quad, Some([40, 10, 100, 10, 100, 40, 40, 40]));
}

#[test]
fn ambiguous_partial_strokes_require_model_fallback() {
    let mut image = RgbImage::from_pixel(111, 81, Rgb([255, 255, 255]));
    horizontal(&mut image, 10, 10, 100);
    horizontal(&mut image, 40, 10, 100);
    horizontal(&mut image, 70, 10, 100);
    vertical(&mut image, 10, 10, 70);
    vertical(&mut image, 40, 10, 70);
    vertical(&mut image, 70, 10, 58);
    vertical(&mut image, 100, 10, 70);
    let mut candidate = candidate();
    let partial = candidate
        .vertical_tracks
        .iter_mut()
        .find(|track| track.fixed == 70)
        .unwrap();
    partial.start = 10;
    partial.end = 58;

    assert!(derive_source_grid(&image, &candidate).is_none());
}

#[test]
fn source_backed_t_junction_adds_a_missing_primitive_axis() {
    let mut image = RgbImage::from_pixel(111, 81, Rgb([255, 255, 255]));
    horizontal(&mut image, 10, 10, 100);
    horizontal(&mut image, 40, 10, 100);
    horizontal(&mut image, 55, 40, 100);
    horizontal(&mut image, 70, 10, 100);
    vertical(&mut image, 10, 10, 70);
    vertical(&mut image, 40, 10, 70);
    vertical(&mut image, 70, 10, 55);
    vertical(&mut image, 100, 10, 70);
    let mut candidate = candidate();
    let partial = candidate
        .vertical_tracks
        .iter_mut()
        .find(|track| track.fixed == 70)
        .unwrap();
    partial.start = 10;
    partial.end = 55;

    let grid = derive_source_grid(&image, &candidate).unwrap();

    assert_eq!((grid.row_count, grid.column_count), (3, 3));
    assert_eq!(grid.cells.len(), 7);
    assert!(grid
        .cells
        .iter()
        .any(|cell| { (cell.row, cell.column, cell.row_span, cell.column_span) == (2, 1, 1, 2) }));
}

#[test]
fn terminal_refinement_is_symmetric_and_requires_a_two_sided_crossbar() {
    let mut image = RgbImage::from_pixel(111, 81, Rgb([255, 255, 255]));
    horizontal(&mut image, 40, 10, 55);
    vertical(&mut image, 55, 10, 70);
    let mut horizontal_terminal_candidate = candidate();
    let partial = horizontal_terminal_candidate
        .horizontal_tracks
        .iter_mut()
        .find(|track| track.fixed == 40)
        .unwrap();
    partial.start = 10;
    partial.end = 55;

    let (refined_horizontal, refined_vertical) =
        refined_source_boundaries(&image, &horizontal_terminal_candidate);

    assert_eq!(refined_horizontal, vec![10, 40, 70]);
    assert_eq!(refined_vertical, vec![10, 40, 55, 70, 100]);

    let mut one_sided = RgbImage::from_pixel(111, 81, Rgb([255, 255, 255]));
    horizontal(&mut one_sided, 55, 70, 100);
    vertical(&mut one_sided, 70, 10, 55);
    let mut one_sided_candidate = candidate();
    let partial = one_sided_candidate
        .vertical_tracks
        .iter_mut()
        .find(|track| track.fixed == 70)
        .unwrap();
    partial.start = 10;
    partial.end = 55;

    let (refined_horizontal, refined_vertical) =
        refined_source_boundaries(&one_sided, &one_sided_candidate);

    assert_eq!(refined_horizontal, vec![10, 40, 70]);
    assert_eq!(refined_vertical, vec![10, 40, 70, 100]);

    let mut band_only = RgbImage::from_pixel(111, 81, Rgb([255, 255, 255]));
    horizontal(&mut band_only, 55, 40, 55);
    horizontal(&mut band_only, 54, 56, 59);
    horizontal(&mut band_only, 55, 60, 100);
    vertical(&mut band_only, 70, 10, 55);
    let mut band_only_candidate = candidate();
    let partial = band_only_candidate
        .vertical_tracks
        .iter_mut()
        .find(|track| track.fixed == 70)
        .unwrap();
    partial.start = 10;
    partial.end = 55;
    assert_eq!(
        classify_horizontal(&band_only, 55, 40, 70),
        Some(Stroke::Present)
    );
    assert_eq!(
        classify_horizontal(&band_only, 55, 70, 100),
        Some(Stroke::Present)
    );

    let (refined_horizontal, refined_vertical) =
        refined_source_boundaries(&band_only, &band_only_candidate);

    assert_eq!(refined_horizontal, vec![10, 40, 70]);
    assert_eq!(refined_vertical, vec![10, 40, 70, 100]);
}

#[test]
fn continuity_distinguishes_a_faded_rule_from_clustered_content() {
    assert_eq!(
        classify_stroke(0, 20, |position| position % 3 != 0),
        Some(Stroke::Present)
    );
    assert_eq!(classify_stroke(0, 20, |position| position <= 11), None);
}

#[test]
fn wider_junction_tolerance_requires_complete_two_endpoint_support() {
    let mut image = RgbImage::from_pixel(21, 21, Rgb([255, 255, 255]));
    image.put_pixel(10, 7, Rgb([0, 0, 0]));
    assert_eq!(classify_vertical(&image, 10, 0, 20), Some(Stroke::Absent));
    assert_eq!(
        classify_vertical_with_junction_tolerance(&image, 10, 0, 20),
        Some(Stroke::Absent)
    );

    for y in 3..=14 {
        image.put_pixel(10, y, Rgb([0, 0, 0]));
    }
    assert_eq!(classify_vertical(&image, 10, 0, 20), None);
    assert_eq!(
        classify_vertical_with_junction_tolerance(&image, 10, 0, 20),
        Some(Stroke::Present)
    );

    let mut displaced = RgbImage::from_pixel(21, 21, Rgb([255, 255, 255]));
    for y in 7..=13 {
        displaced.put_pixel(10, y, Rgb([0, 0, 0]));
    }
    assert_eq!(
        classify_vertical(&displaced, 10, 0, 20),
        Some(Stroke::Absent)
    );
    assert_eq!(
        classify_vertical_with_junction_tolerance(&displaced, 10, 0, 20),
        Some(Stroke::Present)
    );
}

#[test]
fn missing_tracks_are_negative_evidence_only_inside_detector_authority() {
    assert_eq!(classify_tracks(10, 20, 99, &[], 80), Some(Stroke::Absent));
    assert_eq!(classify_tracks(10, 20, 98, &[], 80), None);
    assert_eq!(
        classify_tracks(
            10,
            20,
            99,
            &[LineTrack {
                fixed: 10,
                start: 20,
                end: 60,
            }],
            80,
        ),
        None
    );
    assert_eq!(
        classify_tracks(
            10,
            20,
            99,
            &[LineTrack {
                fixed: 10,
                start: 20,
                end: 99,
            }],
            80,
        ),
        Some(Stroke::Present)
    );
}

#[test]
fn an_outer_axis_requires_source_support_without_inventing_visual_closure() {
    assert!(side_has_support(
        [
            Some(Stroke::Present),
            Some(Stroke::Present),
            None,
            Some(Stroke::Present),
            Some(Stroke::Present)
        ]
        .into_iter()
    ));
    assert!(side_has_support(
        [
            Some(Stroke::Present),
            None,
            Some(Stroke::Present),
            None,
            Some(Stroke::Present)
        ]
        .into_iter()
    ));
    assert!(side_has_support(
        [Some(Stroke::Present), None, Some(Stroke::Present)].into_iter()
    ));
    assert!(side_has_support([Some(Stroke::Present), None].into_iter()));
    assert!(side_has_support(
        [Some(Stroke::Present), Some(Stroke::Present)].into_iter()
    ));
    assert!(!side_has_support(
        [Some(Stroke::Absent), None, Some(Stroke::Absent)].into_iter()
    ));
}

#[test]
fn outer_axis_inference_never_bridges_a_detector_observable_gap() {
    let boundaries = [0, 20, 40, 100];
    assert!(side_is_backed_within_detector_resolution(
        &boundaries,
        64,
        |interval| [
            Some(Stroke::Absent),
            Some(Stroke::Present),
            Some(Stroke::Present)
        ][interval],
        |_| false
    ));
    assert!(!side_is_backed_within_detector_resolution(
        &[0, 40, 80, 100],
        64,
        |interval| [
            Some(Stroke::Absent),
            Some(Stroke::Absent),
            Some(Stroke::Present)
        ][interval],
        |_| false
    ));
    assert!(!side_is_backed_within_detector_resolution(
        &boundaries,
        64,
        |_| Some(Stroke::Absent),
        |_| false
    ));
    assert!(side_is_backed_within_detector_resolution(
        &[0, 80, 160],
        64,
        |interval| [Some(Stroke::Present), Some(Stroke::Absent)][interval],
        |interval| interval == 1
    ));
}

#[test]
fn recurring_perpendicular_terminals_back_an_open_outer_axis() {
    let tracks = [
        LineTrack {
            fixed: 10,
            start: 20,
            end: 100,
        },
        LineTrack {
            fixed: 40,
            start: 20,
            end: 102,
        },
    ];
    assert!(side_has_recurring_terminal_support(
        100,
        &tracks,
        TrackTerminal::End
    ));
    assert!(!side_has_recurring_terminal_support(
        20,
        &tracks[..1],
        TrackTerminal::Start
    ));
}

#[test]
fn interior_content_without_junction_support_is_not_a_separator() {
    assert_eq!(
        classify_stroke(0, 20, |position| position == 10),
        Some(Stroke::Absent)
    );
    assert_eq!(classify_stroke(0, 20, |_| false), Some(Stroke::Absent));
}

#[test]
fn partial_junction_support_requires_model_fallback() {
    assert_eq!(classify_stroke(0, 20, |position| position <= 10), None);
    assert_eq!(classify_stroke(0, 20, |position| position >= 10), None);
    assert_eq!(
        classify_stroke(0, 20, |position| position <= 7 || position >= 13),
        None
    );
}

#[test]
fn junction_footprint_alone_is_not_a_separator() {
    assert_eq!(
        classify_stroke(0, 20, |position| (3..=7).contains(&position)),
        Some(Stroke::Absent)
    );
    assert_eq!(
        classify_stroke(0, 20, |position| (3..=8).contains(&position)),
        None
    );
    assert_eq!(
        classify_stroke(0, 20, |position| (13..=17).contains(&position)),
        Some(Stroke::Absent)
    );
}

#[test]
fn disconnected_content_does_not_extend_a_junction_footprint() {
    assert_eq!(
        classify_stroke(0, 20, |position| {
            (8..=10).contains(&position) || (16..=17).contains(&position)
        }),
        Some(Stroke::Absent)
    );
    assert_eq!(
        classify_stroke(0, 20, |position| {
            (3..=4).contains(&position) || (10..=12).contains(&position)
        }),
        Some(Stroke::Absent)
    );
}

#[test]
fn rectangular_constraints_resolve_an_unknown_edge_only_when_unique() {
    let horizontal = [10, 20, 30];
    let vertical = [10, 20, 30];
    let column_partition = [
        edge(0, 1, None),
        edge(2, 3, Some(Stroke::Present)),
        edge(0, 2, Some(Stroke::Absent)),
        edge(1, 3, Some(Stroke::Absent)),
    ];
    let columns = resolve_unique_partition(
        2,
        2,
        &horizontal,
        &vertical,
        TableCropOrientation::Upright,
        &column_partition,
    )
    .unwrap();
    assert_eq!(
        columns
            .iter()
            .map(|cell| (cell.row, cell.column, cell.row_span, cell.column_span))
            .collect::<Vec<_>>(),
        vec![(0, 0, 2, 1), (0, 1, 2, 1)]
    );

    let single_cell = [
        edge(0, 1, None),
        edge(2, 3, Some(Stroke::Absent)),
        edge(0, 2, Some(Stroke::Absent)),
        edge(1, 3, Some(Stroke::Absent)),
    ];
    let cell = resolve_unique_partition(
        2,
        2,
        &horizontal,
        &vertical,
        TableCropOrientation::Upright,
        &single_cell,
    )
    .unwrap();
    assert_eq!(
        cell.iter()
            .map(|cell| (cell.row, cell.column, cell.row_span, cell.column_span))
            .collect::<Vec<_>>(),
        vec![(0, 0, 2, 2)]
    );
}

#[test]
fn multiple_rectangular_partitions_require_model_fallback() {
    let horizontal = [10, 20, 30];
    let vertical = [10, 20, 30];
    let evidence = [
        edge(0, 1, None),
        edge(2, 3, Some(Stroke::Present)),
        edge(0, 2, Some(Stroke::Present)),
        edge(1, 3, Some(Stroke::Present)),
    ];
    assert!(resolve_unique_partition(
        2,
        2,
        &horizontal,
        &vertical,
        TableCropOrientation::Upright,
        &evidence,
    )
    .is_none());
}

fn edge(first: usize, second: usize, stroke: Option<Stroke>) -> BoundaryEvidence {
    BoundaryEvidence {
        first,
        second,
        stroke,
    }
}

#[test]
fn rotated_source_strokes_map_x_to_rows_and_reverse_y_columns() {
    let mut image = RgbImage::from_pixel(111, 81, Rgb([255, 255, 255]));
    horizontal(&mut image, 10, 10, 100);
    horizontal(&mut image, 40, 40, 100);
    horizontal(&mut image, 70, 10, 100);
    vertical(&mut image, 10, 10, 70);
    vertical(&mut image, 40, 10, 70);
    vertical(&mut image, 70, 40, 70);
    vertical(&mut image, 100, 10, 70);
    let mut candidate = candidate();
    candidate.orientation = TableCropOrientation::Rotate90;

    let grid = derive_source_grid(&image, &candidate).unwrap();

    assert_eq!((grid.row_count, grid.column_count), (3, 2));
    assert_eq!(
        grid.cells
            .iter()
            .map(|cell| (cell.row, cell.column, cell.row_span, cell.column_span))
            .collect::<Vec<_>>(),
        vec![(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 2, 1), (2, 0, 1, 1)]
    );
}

#[test]
#[ignore = "requires the reviewed real rotated-table raster fixtures"]
fn real_rotated_table_source_wires_recover_reviewed_topology_or_fall_back() {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_ROTATED_TABLE_DIR")
        .expect("A3S_OCR_REAL_ROTATED_TABLE_DIR must name the reviewed fixture root");
    let fixture_filter = std::env::var_os("A3S_OCR_REAL_ROTATED_TABLE_FILTER");
    let cancellation = tokio_util::sync::CancellationToken::new();
    let mut tested = 0_usize;
    let mut derived = 0_usize;
    let mut fallback = 0_usize;

    for page in 7_u32..=24 {
        let name = format!("page-{page:04}.png");
        if fixture_filter
            .as_deref()
            .is_some_and(|filter| filter != std::ffi::OsStr::new(&name))
        {
            continue;
        }
        tested += 1;
        let image = image::open(std::path::Path::new(&fixture_root).join(&name))
            .unwrap()
            .into_rgb8();
        let candidates = crate::document_fast::wired::candidates(&image, &cancellation).unwrap();
        let expected: &[ExpectedTopology] = match page {
            7 => &[(TableCropOrientation::Upright, Some((12, 3, 35)))],
            8 => &[(TableCropOrientation::Upright, Some((7, 3, 21)))],
            9 => &[(TableCropOrientation::Upright, Some((8, 4, 30)))],
            10 => &[(TableCropOrientation::Upright, Some((3, 1, 3)))],
            11 => &[(TableCropOrientation::Rotate90, Some((43, 13, 289)))],
            12 => &[(TableCropOrientation::Rotate90, Some((48, 13, 290)))],
            13 => &[
                (TableCropOrientation::Rotate90, Some((41, 10, 224))),
                (TableCropOrientation::Upright, Some((2, 3, 6))),
            ],
            14 => &[(TableCropOrientation::Rotate90, Some((59, 12, 265)))],
            15 => &[(TableCropOrientation::Rotate90, Some((35, 13, 193)))],
            16 => &[(TableCropOrientation::Rotate90, Some((46, 13, 279)))],
            17 => &[
                (TableCropOrientation::Rotate90, Some((52, 11, 303))),
                (TableCropOrientation::Upright, Some((3, 8, 23))),
            ],
            18 => &[(TableCropOrientation::Rotate90, Some((47, 12, 246)))],
            19 => &[(TableCropOrientation::Rotate90, Some((48, 13, 265)))],
            20 => &[(TableCropOrientation::Rotate90, Some((37, 13, 231)))],
            21 => &[(TableCropOrientation::Rotate90, Some((41, 10, 229)))],
            22 => &[
                (TableCropOrientation::Rotate90, Some((30, 10, 173))),
                (TableCropOrientation::Upright, Some((3, 4, 11))),
            ],
            23 => &[(TableCropOrientation::Rotate90, Some((50, 10, 268)))],
            24 => &[(TableCropOrientation::Rotate90, Some((37, 12, 200)))],
            _ => unreachable!("the fixture inventory is closed"),
        };
        assert_eq!(candidates.len(), expected.len(), "{name}");
        for (candidate_index, (candidate, (orientation, topology))) in
            candidates.iter().zip(expected).enumerate()
        {
            assert_eq!(
                candidate.orientation, *orientation,
                "{name}[{candidate_index}]"
            );
            let (horizontal, vertical) = refined_source_boundaries(&image, candidate);
            match (derive_source_grid(&image, candidate), *topology) {
                (None, None) => {
                    fallback += 1;
                    eprintln!(
                        "{name}[{candidate_index}] reviewed fallback: {}",
                        track_proof_diagnostics(&image, candidate, &horizontal, &vertical)
                    );
                }
                (Some(grid), Some((rows, columns, cells))) => {
                    derived += 1;
                    eprintln!("{name}[{candidate_index}] strict source grid");
                    assert_eq!(
                        (grid.row_count, grid.column_count, grid.cells.len()),
                        (rows, columns, cells),
                        "{name}[{candidate_index}] horizontal={horizontal:?} vertical={vertical:?} candidate={candidate:?}"
                    );
                    assert_eq!(grid.confidence, None);
                    for cell in grid.cells {
                        let quad = cell.quad.expect("source cells carry exact page geometry");
                        assert!(
                            [quad[0], quad[2], quad[4], quad[6]]
                                .into_iter()
                                .all(|x| vertical.contains(&x)),
                            "{name}[{candidate_index}] x geometry"
                        );
                        assert!(
                            [quad[1], quad[3], quad[5], quad[7]]
                                .into_iter()
                                .all(|y| horizontal.contains(&y)),
                            "{name}[{candidate_index}] y geometry"
                        );
                    }
                }
                (None, Some(_)) => {
                    fallback += 1;
                    eprintln!(
                        "{name}[{candidate_index}] incomplete proof; model fallback: {}; {}",
                        proof_diagnostics(&image, &horizontal, &vertical),
                        track_proof_diagnostics(&image, candidate, &horizontal, &vertical)
                    );
                }
                (Some(grid), None) => {
                    panic!(
                        "{name}[{candidate_index}] bypassed required fallback with candidate {candidate:?} and grid {}x{} ({} cells)",
                        grid.row_count,
                        grid.column_count,
                        grid.cells.len()
                    )
                }
            }
        }
    }
    assert!(tested > 0, "the rotated fixture filter matched no file");
    if fixture_filter.is_none() {
        assert_eq!(
            (derived, fallback),
            (21, 0),
            "reviewed source-grid coverage changed; inspect every changed topology before accepting it"
        );
    }
    eprintln!("strict source grids={derived} model fallbacks={fallback}");
}

fn proof_diagnostics(image: &RgbImage, horizontal: &[u32], vertical: &[u32]) -> String {
    let mut perimeter = [0_usize; 3];
    let mut internal = [0_usize; 3];
    let top = horizontal[0];
    let bottom = horizontal[horizontal.len() - 1];
    let left = vertical[0];
    let right = vertical[vertical.len() - 1];
    for column in 0..vertical.len() - 1 {
        record_stroke(
            &mut perimeter,
            classify_horizontal(image, top, vertical[column], vertical[column + 1]),
        );
        record_stroke(
            &mut perimeter,
            classify_horizontal(image, bottom, vertical[column], vertical[column + 1]),
        );
    }
    for row in 0..horizontal.len() - 1 {
        record_stroke(
            &mut perimeter,
            classify_vertical(image, left, horizontal[row], horizontal[row + 1]),
        );
        record_stroke(
            &mut perimeter,
            classify_vertical(image, right, horizontal[row], horizontal[row + 1]),
        );
    }
    for coordinate in vertical.iter().take(vertical.len() - 1).skip(1) {
        for row in 0..horizontal.len() - 1 {
            record_stroke(
                &mut internal,
                classify_vertical(image, *coordinate, horizontal[row], horizontal[row + 1]),
            );
        }
    }
    for coordinate in horizontal.iter().take(horizontal.len() - 1).skip(1) {
        for column in 0..vertical.len() - 1 {
            record_stroke(
                &mut internal,
                classify_horizontal(image, *coordinate, vertical[column], vertical[column + 1]),
            );
        }
    }
    format!(
        "perimeter present={} absent={} ambiguous={}; internal present={} absent={} ambiguous={}",
        perimeter[0], perimeter[1], perimeter[2], internal[0], internal[1], internal[2]
    )
}

fn track_proof_diagnostics(
    image: &RgbImage,
    candidate: &WiredCandidate,
    horizontal: &[u32],
    vertical: &[u32],
) -> String {
    let minimum_horizontal_track = (image.width() / 5).max(96);
    let minimum_vertical_track = (image.height() / 12).max(64);
    let outer = validate_fused_outer_axes(
        image,
        candidate,
        horizontal,
        vertical,
        minimum_horizontal_track,
        minimum_vertical_track,
    )
    .is_some();
    let mut internal = [0_usize; 3];
    let mut ambiguous = Vec::new();
    for coordinate in vertical.iter().take(vertical.len() - 1).skip(1) {
        for row in 0..horizontal.len() - 1 {
            let stroke = fuse_strokes(
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
            );
            if stroke.is_none() {
                ambiguous.push(format!(
                        "V x={coordinate} y={}..{} {} tracks={:?}",
                        horizontal[row],
                        horizontal[row + 1],
                        stroke_profile(horizontal[row], horizontal[row + 1], |position| {
                            band_contains_dark(image, *coordinate, position, true)
                        }),
                        candidate
                            .vertical_tracks
                            .iter()
                            .filter(|track| track.fixed.abs_diff(*coordinate)
                                <= DUPLICATE_LINE_DISTANCE)
                            .map(|track| (track.fixed, track.start, track.end))
                            .collect::<Vec<_>>()
                    ));
            }
            record_stroke(&mut internal, stroke);
        }
    }
    for coordinate in horizontal.iter().take(horizontal.len() - 1).skip(1) {
        for column in 0..vertical.len() - 1 {
            let stroke = fuse_strokes(
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
            );
            if stroke.is_none() {
                ambiguous.push(format!(
                        "H y={coordinate} x={}..{} {} tracks={:?}",
                        vertical[column],
                        vertical[column + 1],
                        stroke_profile(vertical[column], vertical[column + 1], |position| {
                            band_contains_dark(image, *coordinate, position, false)
                        }),
                        candidate
                            .horizontal_tracks
                            .iter()
                            .filter(|track| track.fixed.abs_diff(*coordinate)
                                <= DUPLICATE_LINE_DISTANCE)
                            .map(|track| (track.fixed, track.start, track.end))
                            .collect::<Vec<_>>()
                    ));
            }
            record_stroke(&mut internal, stroke);
        }
    }
    format!(
        "track outer={outer} internal present={} absent={} ambiguous={} unknown={ambiguous:?}",
        internal[0], internal[1], internal[2]
    )
}

fn stroke_profile(start: u32, end: u32, mut selected: impl FnMut(u32) -> bool) -> String {
    let first = start.saturating_add(ENDPOINT_MARGIN);
    let last = end.saturating_sub(ENDPOINT_MARGIN);
    let mut selected_count = 0_u32;
    let mut runs = Vec::<u32>::new();
    let mut current = 0_u32;
    let mut first_dark = None;
    let mut last_dark = None;
    for position in first..=last {
        if selected(position) {
            selected_count += 1;
            current += 1;
            first_dark.get_or_insert(position);
            last_dark = Some(position);
        } else if current > 0 {
            runs.push(current);
            current = 0;
        }
    }
    if current > 0 {
        runs.push(current);
    }
    format!(
        "dark={selected_count} runs={runs:?} first={:?} last={:?}",
        first_dark.map(|value| value.saturating_sub(first)),
        last_dark.map(|value| last.saturating_sub(value)),
    )
}

fn record_stroke(counts: &mut [usize; 3], stroke: Option<Stroke>) {
    counts[match stroke {
        Some(Stroke::Present) => 0,
        Some(Stroke::Absent) => 1,
        None => 2,
    }] += 1;
}

fn candidate() -> WiredCandidate {
    WiredCandidate {
        region: PixelRect {
            x: 10,
            y: 10,
            width: 91,
            height: 61,
        },
        inference_region: PixelRect {
            x: 10,
            y: 10,
            width: 91,
            height: 61,
        },
        orientation: TableCropOrientation::Upright,
        horizontal_lines: vec![10, 40, 70],
        vertical_lines: vec![10, 40, 70, 100],
        horizontal_tracks: vec![
            LineTrack {
                fixed: 10,
                start: 10,
                end: 100,
            },
            LineTrack {
                fixed: 40,
                start: 40,
                end: 100,
            },
            LineTrack {
                fixed: 70,
                start: 10,
                end: 100,
            },
        ],
        vertical_tracks: vec![
            LineTrack {
                fixed: 10,
                start: 10,
                end: 70,
            },
            LineTrack {
                fixed: 40,
                start: 10,
                end: 70,
            },
            LineTrack {
                fixed: 70,
                start: 40,
                end: 70,
            },
            LineTrack {
                fixed: 100,
                start: 10,
                end: 70,
            },
        ],
    }
}

fn horizontal(image: &mut RgbImage, y: u32, start: u32, end: u32) {
    for x in start..=end {
        image.put_pixel(x, y, Rgb([0, 0, 0]));
    }
}

fn vertical(image: &mut RgbImage, x: u32, start: u32, end: u32) {
    for y in start..=end {
        image.put_pixel(x, y, Rgb([0, 0, 0]));
    }
}
