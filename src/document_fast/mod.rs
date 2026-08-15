//! Fast document OCR composition owned by A3S OCR.
//!
//! The table path deliberately separates deterministic region admission from
//! model-backed structure recognition. A line candidate alone is never
//! published as table evidence.

mod assets;
mod decoder;
mod native;
mod preprocess;
mod projection;
mod provider;
mod seal;
mod stage;
mod wired;

pub use provider::{DocumentFastOcrProvider, DOCUMENT_FAST_PROVIDER_ID};

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::assets::SlanetPlusAssets;
    use super::decoder::SlanetPlusDecoder;
    use super::native::NativeSlanetPlus;
    use super::{preprocess, wired};

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
        let regions = wired::candidates(&image, &cancellation).unwrap();
        assert_eq!(regions.len(), 1);
        let input = preprocess::crop_tensor(&image, regions[0]).unwrap();
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
            .decode(&encoded.tensor.values, regions[0], &cancellation)
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
}
