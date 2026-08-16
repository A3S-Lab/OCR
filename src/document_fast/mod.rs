//! Fast document OCR composition owned by A3S OCR.
//!
//! The table path deliberately separates deterministic region admission from
//! model-backed structure recognition. A line candidate alone is never
//! published as table evidence.

mod assets;
mod decoder;
mod native;
mod orientation;
mod preprocess;
mod projection;
mod provider;
mod seal;
mod stage;
mod wired;

pub use provider::{DocumentFastOcrProvider, DOCUMENT_FAST_PROVIDER_ID};

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};
    use tokio_util::sync::CancellationToken;

    use super::assets::SlanetPlusAssets;
    use super::decoder::SlanetPlusDecoder;
    use super::native::NativeSlanetPlus;
    use super::{orientation::TableCropOrientation, preprocess, wired};

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
    #[ignore = "requires the pinned SLANet-Plus bundle and reviewed rotated table fixtures"]
    fn real_rotated_wired_tables_preserve_source_cell_geometry() {
        const FIXTURES: [(&str, &str); 6] = [
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
                "page-0016.png",
                "ae10b85a388f66a25efa87bdc173f6fb54c32549f021c5bc53aa2542d9e1c6c1",
            ),
            (
                "page-0017.png",
                "ca932637c7464ab21b1a500f9f8fa7bcea2e74cea3f4b50616d80cfa3487b518",
            ),
            (
                "page-0021.png",
                "70e1e208a9bb931b7e637d0d305acc4697f83af9171c14453d38e2d25d01b37e",
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
            let expected: &[(TableCropOrientation, usize, u32, u32, usize)] = match name {
                "page-0011.png" => &[(TableCropOrientation::Rotate90, 527, 37, 14, 280)],
                "page-0012.png" => &[(TableCropOrientation::Rotate90, 430, 38, 13, 253)],
                "page-0013.png" => &[
                    (TableCropOrientation::Rotate90, 394, 40, 11, 246),
                    (TableCropOrientation::Upright, 12, 2, 3, 5),
                ],
                "page-0016.png" => &[(TableCropOrientation::Rotate90, 477, 36, 14, 244)],
                "page-0017.png" => &[
                    (TableCropOrientation::Rotate90, 531, 48, 11, 301),
                    (TableCropOrientation::Upright, 20, 2, 7, 14),
                ],
                "page-0021.png" => &[(TableCropOrientation::Rotate90, 408, 37, 12, 257)],
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
                let grid = decoded.into_grid().unwrap();
                let located = grid.cells.iter().filter(|cell| cell.quad.is_some()).count();
                let expected = expected[candidate_index];
                assert_eq!(
                    (
                        candidate.orientation,
                        token_count,
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
                    "{name}[{candidate_index}] orientation={:?} tokens={} grid={}x{} cells={} located={}",
                    candidate.orientation,
                    token_count,
                    grid.row_count,
                    grid.column_count,
                    grid.cells.len(),
                    located
                );
                if candidate.orientation == TableCropOrientation::Rotate90 {
                    assert!(grid.row_count > grid.column_count, "{name}");
                }
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
