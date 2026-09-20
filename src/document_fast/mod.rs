//! Fast document OCR composition owned by A3S OCR.
//!
//! The table path deliberately separates deterministic region admission from
//! structure recognition. Strict source wires provide exact topology in both
//! page orientations; ambiguous candidates retain the model-backed fallback.
//! A line candidate alone is never published as table evidence.

mod assets;
mod decoder;
mod initialization;
mod layout;
mod native;
mod opencv_cubic;
mod orientation;
mod page_orientation;
#[path = "page_orientation_probe/preprocess.rs"]
mod page_orientation_preprocess;
#[cfg(test)]
mod page_orientation_probe;
mod preprocess;
mod projection;
mod provider;
mod seal;
mod shared_decode;
mod source_grid;
mod stage;
mod text_lanes;
mod wire_geometry;
mod wired;

pub use initialization::{DocumentFastInitializationError, DocumentFastInitializationErrorKind};
pub use provider::{DocumentFastOcrProvider, DOCUMENT_FAST_PROVIDER_ID};

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};
    use tokio_util::sync::CancellationToken;

    use super::assets::SlanetPlusAssets;
    use super::decoder::SlanetPlusDecoder;
    use super::native::NativeSlanetPlus;
    use super::{orientation::TableCropOrientation, preprocess, source_grid, wire_geometry, wired};

    #[test]
    #[ignore = "requires the reviewed real cross-page table raster fixture"]
    fn real_cross_page_table_source_wires_recover_exact_merged_topology() {
        let fixture_root = std::env::var_os("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR")
            .expect("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR must name the reviewed fixture root");
        let cancellation = CancellationToken::new();
        let expected = [
            (2_u32, 6_u32, 6_u32, 28_usize),
            (3, 8, 6, 25),
            (4, 3, 6, 15),
        ];
        let required_spans = [
            vec![(2, 2, 1, 2), (3, 0, 3, 1), (3, 1, 3, 1), (3, 2, 2, 1)],
            vec![(0, 0, 2, 1), (0, 1, 2, 1), (0, 2, 2, 1), (3, 0, 5, 1)],
            vec![(1, 0, 2, 1), (1, 1, 2, 1)],
        ];

        for ((page, rows, columns, cells), required) in expected.into_iter().zip(required_spans) {
            let image = image::open(
                std::path::Path::new(&fixture_root).join(format!("page-{page:04}.png")),
            )
            .unwrap()
            .into_rgb8();
            let candidates = wired::candidates(&image, &cancellation).unwrap();
            assert_eq!(candidates.len(), 1, "page {page}");
            let grid = source_grid::derive_source_grid(&image, &candidates[0])
                .unwrap_or_else(|| panic!("page {page} source topology was ambiguous"));
            let topology = grid
                .cells
                .iter()
                .map(|cell| (cell.row, cell.column, cell.row_span, cell.column_span))
                .collect::<Vec<_>>();
            eprintln!("page-{page:04} source topology: {topology:?}");
            assert_eq!(
                (grid.row_count, grid.column_count, grid.cells.len()),
                (rows, columns, cells),
                "page {page}"
            );
            for span in required {
                assert!(topology.contains(&span), "page {page} missing {span:?}");
            }
        }
    }

    #[test]
    #[ignore = "requires the pinned SLANet-Plus bundle and real cross-page table fixture"]
    fn real_wired_table_executes_the_power_encoder_and_structure_decoder() {
        let assets = SlanetPlusAssets::from_env().unwrap();
        let fixture_root = std::env::var_os("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR")
            .expect("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR must name the reviewed fixture root");
        let image = image::open(std::path::Path::new(&fixture_root).join("page-0002.png"))
            .unwrap()
            .into_rgb8();
        let cancellation = CancellationToken::new();
        let candidates = wired::candidates(&image, &cancellation).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].orientation, TableCropOrientation::Upright);
        let input =
            preprocess::crop_tensor(&image, candidates[0].region, candidates[0].orientation)
                .unwrap();
        let encoder = NativeSlanetPlus::load(&assets).unwrap();
        let permit = encoder.begin(&cancellation).unwrap();
        let zero = encoder
            .encode_batch(
                vec![0.0; 3 * preprocess::INPUT_SIDE * preprocess::INPUT_SIDE],
                1,
                &permit,
                &cancellation,
            )
            .unwrap();
        let probe_indices = [0, 1, 95, 96, 1_024, 4_096, 8_191, 12_000, 20_000, 24_575];
        let expected_probes = [
            -0.155_649_36,
            0.970_096_4,
            -0.235_554_26,
            -0.201_624_89,
            0.334_476_38,
            -0.050_443_027,
            0.308_062_85,
            0.106_538_504,
            -0.354_751_1,
            0.060_970_064,
        ];
        for (index, expected) in probe_indices.into_iter().zip(expected_probes) {
            let actual = zero.tensor.values[index];
            assert!(
                (actual - expected).abs() <= 5e-5,
                "encoder parity probe {index} differs: expected {expected}, got {actual}"
            );
        }
        let encoded = encoder
            .encode_batch(input, 1, &permit, &cancellation)
            .unwrap();
        assert_eq!(encoded.tensor.shape, [1, 256, 96]);
        let decoder = SlanetPlusDecoder::load(&assets.decoder_weights, &assets.dictionary).unwrap();
        let decoded = decoder
            .decode(
                &encoded.tensor.values,
                candidates[0].region,
                candidates[0].orientation,
                &cancellation,
            )
            .unwrap();
        assert!(decoded.confidence >= 0.97);
        assert_eq!(decoded.tokens.len(), 56);
        assert_eq!(decoded.cells.len(), 29);
        let grid = decoded.into_grid().unwrap();
        assert_eq!((grid.row_count, grid.column_count), (6, 6));
        assert_eq!(grid.cells.len(), 29);
        assert!(grid.cells.iter().all(|cell| cell.quad.is_some()));
        assert!(grid.cells.iter().any(|cell| cell.row_span == 3));
        assert!(grid.cells.iter().any(|cell| cell.column_span == 2));
        assert_eq!(encoded.receipt.model.family, "slanet-plus-wired-encoder");
    }

    #[test]
    #[ignore = "requires the pinned SLANet-Plus bundle and reviewed wired-table fixtures"]
    fn real_rotated_wired_tables_preserve_source_cell_geometry() {
        const FIXTURES: [(&str, &str); 18] = [
            (
                "page-0007.png",
                "fdd2a90527f78454df20550644a24e2ab92ab52c678debb764f1c4d16d570913",
            ),
            (
                "page-0008.png",
                "b428a51acb533453e100a5646e3190c4eab947e3e42439362707f20a47968b83",
            ),
            (
                "page-0009.png",
                "ff0403814802d6a19ebc56a634c8a5778597ff13d28e8b1b8112b90c454154a8",
            ),
            (
                "page-0010.png",
                "fe7e062a7e46b7070ac5e3d8a29852ffa3370ec8f8c40990599a2ca22e3feb0a",
            ),
            (
                "page-0011.png",
                "c6a6d24fb89eac16ad0c1cd8c5270e3ac4e0a0908abe4495e7ff5dcd7cb98920",
            ),
            (
                "page-0012.png",
                "ce2da73a4bbb4329dcaa100339252796fbd251b762fd152db68780dbcdb95770",
            ),
            (
                "page-0013.png",
                "802ffcb8c14d6e9a0179b12dc40c207f79eca919bf2004429134df98d7894947",
            ),
            (
                "page-0014.png",
                "28a75b52a5a786876b61a79e15b97accb00594198a194429abb4abe261ea2e28",
            ),
            (
                "page-0015.png",
                "28e820e4321921553226a657c90f472c47751b7f41ce0f91dfb31956774f3134",
            ),
            (
                "page-0016.png",
                "ae10b85a388f66a25efa87bdc173f6fb54c32549f021c5bc53aa2542d9e1c6c1",
            ),
            (
                "page-0017.png",
                "ca932637c7464ab21b1a500f9f8fa7bcea2e74cea3f4b50616d80cfa3487b518",
            ),
            (
                "page-0018.png",
                "4c701e7c97d634cec14cf1dd1bd9102a2faef35051d9e6d248e0d63fccb4bddb",
            ),
            (
                "page-0019.png",
                "72afe88804cb307ad72fa1ac3dc942bb3222aef8e425cb9b0d3a5c22960b7abb",
            ),
            (
                "page-0020.png",
                "9d05ca77118f4439a58249c24d89c55cc0504165892e45d004cff8a91ccc93aa",
            ),
            (
                "page-0021.png",
                "70e1e208a9bb931b7e637d0d305acc4697f83af9171c14453d38e2d25d01b37e",
            ),
            (
                "page-0022.png",
                "5b2c91ead84d885e799f7c1019ffb45791da47b502e8b981127c6c2ee9781d5f",
            ),
            (
                "page-0023.png",
                "d2c5380138af3d3c87914b19d83e49f69d2e51f2a22a58b1eb824fcc852e8667",
            ),
            (
                "page-0024.png",
                "df63a473d7ab19f74f9087ba73ebeada07a0b0dad939169cef1ba410d2fac332",
            ),
        ];

        let fixture_root = std::env::var_os("A3S_OCR_REAL_ROTATED_TABLE_DIR")
            .expect("A3S_OCR_REAL_ROTATED_TABLE_DIR must name the reviewed fixture root");
        let fixture_filter = std::env::var_os("A3S_OCR_REAL_ROTATED_TABLE_FILTER");
        let assets = SlanetPlusAssets::from_env().unwrap();
        let cancellation = CancellationToken::new();
        let encoder = NativeSlanetPlus::load(&assets).unwrap();
        let permit = encoder.begin(&cancellation).unwrap();
        let decoder = SlanetPlusDecoder::load(&assets.decoder_weights, &assets.dictionary).unwrap();
        let mut missing_quads = 0_usize;
        let mut tested_fixtures = 0_usize;
        for (name, expected_sha256) in FIXTURES {
            if fixture_filter
                .as_deref()
                .is_some_and(|filter| filter != std::ffi::OsStr::new(name))
            {
                continue;
            }
            tested_fixtures += 1;
            let bytes = std::fs::read(std::path::Path::new(&fixture_root).join(name)).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                expected_sha256,
                "{name} changed"
            );
            let image = image::load_from_memory(&bytes).unwrap().into_rgb8();
            let candidates = wired::candidates(&image, &cancellation).unwrap();
            let expected: &[(TableCropOrientation, u32, u32, usize)] = match name {
                "page-0007.png" => &[(TableCropOrientation::Upright, 12, 3, 35)],
                "page-0008.png" => &[(TableCropOrientation::Upright, 7, 2, 14)],
                "page-0009.png" => &[(TableCropOrientation::Upright, 8, 4, 29)],
                "page-0010.png" => &[(TableCropOrientation::Upright, 3, 1, 3)],
                "page-0011.png" => &[(TableCropOrientation::Rotate90, 37, 14, 280)],
                "page-0012.png" => &[(TableCropOrientation::Rotate90, 38, 13, 253)],
                "page-0013.png" => &[
                    (TableCropOrientation::Rotate90, 40, 11, 246),
                    (TableCropOrientation::Upright, 2, 3, 5),
                ],
                "page-0014.png" => &[(TableCropOrientation::Rotate90, 51, 15, 317)],
                "page-0015.png" => &[(TableCropOrientation::Rotate90, 31, 15, 195)],
                "page-0016.png" => &[(TableCropOrientation::Rotate90, 36, 14, 244)],
                "page-0017.png" => &[
                    (TableCropOrientation::Rotate90, 48, 11, 301),
                    (TableCropOrientation::Upright, 2, 7, 14),
                ],
                "page-0018.png" => &[(TableCropOrientation::Rotate90, 40, 15, 242)],
                "page-0019.png" => &[(TableCropOrientation::Rotate90, 42, 19, 276)],
                "page-0020.png" => &[(TableCropOrientation::Rotate90, 31, 15, 203)],
                "page-0021.png" => &[(TableCropOrientation::Rotate90, 37, 12, 257)],
                "page-0022.png" => &[
                    (TableCropOrientation::Rotate90, 29, 11, 184),
                    (TableCropOrientation::Upright, 2, 2, 4),
                ],
                "page-0023.png" => &[(TableCropOrientation::Rotate90, 42, 11, 279)],
                "page-0024.png" => &[(TableCropOrientation::Rotate90, 34, 13, 218)],
                _ => unreachable!("the fixture inventory is closed"),
            };
            assert_eq!(candidates.len(), expected.len(), "{name}: {candidates:?}");
            for (candidate_index, candidate) in candidates.into_iter().enumerate() {
                let input = preprocess::crop_tensor(
                    &image,
                    candidate.inference_region,
                    candidate.orientation,
                )
                .unwrap();
                let encoded = encoder
                    .encode_batch(input, 1, &permit, &cancellation)
                    .unwrap();
                let decoded = decoder
                    .decode(
                        &encoded.tensor.values,
                        candidate.inference_region,
                        candidate.orientation,
                        &cancellation,
                    )
                    .unwrap();
                let token_count = decoded.tokens.len();
                let mut grid = decoded.into_grid().unwrap();
                assert!(
                    wire_geometry::align_grid_to_wires(
                        &mut grid,
                        candidate.orientation,
                        &candidate.horizontal_lines,
                        &candidate.vertical_lines,
                    ),
                    "{name}[{candidate_index}] could not align model topology to source wires"
                );
                let located = grid.cells.iter().filter(|cell| cell.quad.is_some()).count();
                let expected = expected[candidate_index];
                assert_eq!(
                    (
                        candidate.orientation,
                        grid.row_count,
                        grid.column_count,
                        grid.cells.len(),
                    ),
                    expected,
                    "{name}[{candidate_index}]"
                );
                for cell in grid.cells.iter().filter(|cell| cell.quad.is_none()) {
                    eprintln!(
                        "{name}[{candidate_index}] missing row={} column={} row_span={} column_span={}",
                        cell.row, cell.column, cell.row_span, cell.column_span
                    );
                }
                eprintln!(
                    "{name}[{candidate_index}] orientation={:?} wire={}x{} tokens={} grid={}x{} cells={} located={}",
                    candidate.orientation,
                    candidate.vertical_lines.len().saturating_sub(1),
                    candidate.horizontal_lines.len().saturating_sub(1),
                    token_count,
                    grid.row_count,
                    grid.column_count,
                    grid.cells.len(),
                    located
                );
                if candidate.orientation == TableCropOrientation::Rotate90 {
                    assert!(grid.row_count > grid.column_count, "{name}");
                }
                let x_aligned = grid.cells.iter().filter_map(|cell| cell.quad).all(|quad| {
                    [quad[0], quad[2], quad[4], quad[6]]
                        .into_iter()
                        .all(|x| candidate.vertical_lines.contains(&x))
                });
                let y_aligned = grid.cells.iter().filter_map(|cell| cell.quad).all(|quad| {
                    [quad[1], quad[3], quad[5], quad[7]]
                        .into_iter()
                        .all(|y| candidate.horizontal_lines.contains(&y))
                });
                assert!(x_aligned || y_aligned, "{name}: no source axis aligned");
                if candidate.orientation == TableCropOrientation::Rotate90 {
                    assert!(x_aligned, "{name}: rotated row boundaries did not align");
                }
                eprintln!(
                    "{name}[{candidate_index}] source_axis_alignment=x:{x_aligned} y:{y_aligned}"
                );
                missing_quads += grid.cells.len().saturating_sub(located);
            }
        }
        assert!(
            tested_fixtures > 0,
            "the rotated fixture filter matched no file"
        );
        assert_eq!(missing_quads, 0);
    }
}
