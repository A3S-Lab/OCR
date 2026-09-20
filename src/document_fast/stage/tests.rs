use super::*;
use std::sync::Arc;

use a3s_use_core::Artifact;
use sha2::{Digest, Sha256};

use crate::document_fast::decoder::GridCell;

#[test]
fn table_region_encompasses_wire_and_model_backed_cell_evidence() {
    let wire_region = PixelRect {
        x: 100,
        y: 200,
        width: 300,
        height: 400,
    };
    let grid = StructureGrid {
        row_count: 2,
        column_count: 1,
        cells: vec![
            GridCell {
                row: 0,
                column: 0,
                row_span: 1,
                column_span: 1,
                quad: Some([90, 180, 410, 180, 410, 300, 90, 300]),
            },
            GridCell {
                row: 1,
                column: 0,
                row_span: 1,
                column_span: 1,
                quad: Some([120, 300, 420, 300, 420, 630, 120, 630]),
            },
        ],
        confidence: Some(0.95),
    };

    assert_eq!(
        table_evidence_region(wire_region, &grid),
        PixelRect {
            x: 90,
            y: 180,
            width: 330,
            height: 450,
        }
    );
}

#[test]
fn table_region_ignores_cells_without_model_geometry() {
    let wire_region = PixelRect {
        x: 10,
        y: 20,
        width: 30,
        height: 40,
    };
    let grid = StructureGrid {
        row_count: 1,
        column_count: 1,
        cells: vec![GridCell {
            row: 0,
            column: 0,
            row_span: 1,
            column_span: 1,
            quad: None,
        }],
        confidence: Some(0.8),
    };

    assert_eq!(table_evidence_region(wire_region, &grid), wire_region);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the retained reviewed 29-page rider raster fixture"]
async fn real_rider_source_table_path_has_no_model_fallback_and_profiles_cpu() {
    const PAGE_COUNT: u32 = 29;

    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the retained raster root");
    let fixture_root = std::path::Path::new(&fixture_root);
    let cancellation = CancellationToken::new();
    let mut slots = Vec::new();
    let mut images = Vec::new();

    for page in 1_u32..=PAGE_COUNT {
        let path = fixture_root.join(format!("page-{page:04}.png"));
        let bytes = std::fs::read(&path).unwrap();
        let image = image::load_from_memory(&bytes).unwrap().into_rgb8();
        let slot_id = crate::OcrBatchSlotId::new(format!("page-{page}")).unwrap();
        slots.push(OcrProviderBatchSlot {
            slot_id,
            input: crate::OcrInput::new(
                Artifact {
                    path,
                    media_type: "image/png".to_string(),
                    size: u64::try_from(bytes.len()).unwrap(),
                    sha256: format!("{:x}", Sha256::digest(&bytes)),
                },
                bytes,
            ),
            adjacent_predecessor_slot_id: None,
            text_window: None,
        });
        images.push(Ok(Arc::new(image)));
    }

    let started = std::time::Instant::now();
    let decoded = prepare_decoded_pages(slots, images, cancellation)
        .await
        .unwrap();
    let elapsed = started.elapsed();
    let mut source_tables = 0_usize;
    let mut model_fallbacks = 0_usize;
    for decoded in decoded {
        let page = decoded.page.unwrap();
        source_tables += page.candidates.len();
        model_fallbacks += page
            .candidates
            .iter()
            .filter(|candidate| candidate.source_grid.is_none())
            .count();
    }
    let pages_per_second = f64::from(PAGE_COUNT) / elapsed.as_secs_f64();

    assert_eq!(source_tables, 23);
    assert_eq!(model_fallbacks, 0);
    if let Ok(minimum) = std::env::var("A3S_OCR_REAL_TABLE_MIN_PAGES_PER_SECOND") {
        let minimum = minimum.parse::<f64>().unwrap();
        assert!(
            pages_per_second >= minimum,
            "source-table throughput {pages_per_second:.3} pages/s is below {minimum:.3} pages/s"
        );
    }
    eprintln!(
        "source-table CPU profile: pages={PAGE_COUNT} source_tables={source_tables} model_fallbacks={model_fallbacks} elapsed_ms={:.3} pages_per_second={pages_per_second:.3}",
        elapsed.as_secs_f64() * 1_000.0,
    );
}
