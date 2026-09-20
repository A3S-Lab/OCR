use super::*;
use sha2::{Digest, Sha256};

#[path = "tests/orientation_batch_probe.rs"]
mod orientation_batch_probe;
#[path = "tests/recognition_batch_probe.rs"]
mod recognition_batch_probe;

#[test]
fn explicit_provider_declares_only_implemented_stages() {
    let Some(_) = std::env::var_os("A3S_OCR_SLANET_PLUS_MODEL_DIR") else {
        return;
    };
    let provider = DocumentFastOcrProvider::from_env().unwrap();
    assert_eq!(provider.descriptor().id, DOCUMENT_FAST_PROVIDER_ID);
    let mut expected = vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Table];
    if std::env::var_os("A3S_OCR_PAGE_ORIENTATION_MODEL_DIR").is_some() {
        expected.insert(0, OcrStage::Orientation);
    }
    if std::env::var_os("A3S_OCR_PICODET_LAYOUT_MODEL_DIR").is_some() {
        expected.push(OcrStage::Seal);
    }
    assert_eq!(provider.descriptor().supported_stages, expected);
    assert_eq!(
        provider.diagnostic().model.as_deref(),
        Some(provider.model_name())
    );
    if let Some(seal) = provider.seal.as_ref() {
        let admitted_family = seal.layout_model_family();
        let declared_layout_families = provider
            .model_name()
            .split('+')
            .filter(|component| {
                matches!(
                    *component,
                    "picodet-l-layout-3cls" | "picodet-s-layout-3cls"
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(declared_layout_families, [admitted_family]);
    }
    assert_eq!(super::super::assets::MODEL_FAMILY, "slanet-plus-wired");
}

#[tokio::test]
#[ignore = "requires pinned orientation/PP-OCRv6/SLANet-Plus bundles and the real rider fixture"]
async fn real_provider_normalizes_only_group_consistent_page_orientation() {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
    assert!(std::env::var_os("A3S_OCR_PAGE_ORIENTATION_MODEL_DIR").is_some());
    let root = std::path::Path::new(&fixture_root);
    let pages = [11_u32, 28, 29];
    let slots = pages
        .into_iter()
        .map(|page| {
            crate::OcrBatchSlotRequest::new(
                crate::OcrBatchSlotId::new(format!("page-{page}")).unwrap(),
                root.join(format!("page-{page:04}.png")),
            )
        })
        .collect();
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
    let oriented = client
        .extract_batch(
            crate::OcrBatchRequest::new(vec![OcrStage::Orientation, OcrStage::Text], slots)
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(oriented.slots.len(), pages.len());
    let orientation_statuses = oriented
        .slots
        .iter()
        .map(|slot| {
            slot.stages
                .iter()
                .find(|stage| stage.stage == OcrStage::Orientation)
                .unwrap()
                .status
        })
        .collect::<Vec<_>>();
    assert_eq!(
        orientation_statuses,
        [
            crate::OcrStageStatus::Completed,
            crate::OcrStageStatus::Skipped,
            crate::OcrStageStatus::Skipped,
        ]
    );
    for (page, slot) in pages.into_iter().zip(&oriented.slots) {
        let image = image::open(root.join(format!("page-{page:04}.png")))
            .unwrap()
            .into_rgb8();
        let output = slot.result.as_ref().unwrap();
        for block in &output.blocks {
            let polygon = block.polygon.unwrap();
            assert!(polygon
                .iter()
                .all(|point| point.x <= image.width() && point.y <= image.height()));
            let bounds = block.bounding_box.unwrap();
            assert!(bounds.x + bounds.width <= image.width());
            assert!(bounds.y + bounds.height <= image.height());
        }
    }

    let baseline = client
        .extract_batch(
            crate::OcrBatchRequest::new(
                vec![OcrStage::Text],
                vec![crate::OcrBatchSlotRequest::new(
                    crate::OcrBatchSlotId::new("page-11-baseline").unwrap(),
                    root.join("page-0011.png"),
                )],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    let corrected = oriented.slots[0].result.as_ref().unwrap();
    let uncorrected = baseline.slots[0].result.as_ref().unwrap();
    assert!(corrected.text.chars().count() > uncorrected.text.chars().count());
    assert!(mean_confidence(corrected) > mean_confidence(uncorrected));
}

fn mean_confidence(output: &crate::OcrResult) -> f32 {
    let confidences = output
        .blocks
        .iter()
        .filter_map(|block| block.confidence)
        .collect::<Vec<_>>();
    confidences.iter().sum::<f32>() / confidences.len().max(1) as f32
}

fn table_stage_evidence(slot: &crate::OcrBatchSlotResult) -> &crate::OcrTableStageEvidence {
    let evidence = slot
        .stages
        .iter()
        .find(|stage| stage.stage == OcrStage::Table)
        .and_then(|stage| stage.evidence.as_ref())
        .unwrap_or_else(|| panic!("table evidence missing from stages: {:#?}", slot.stages));
    let crate::OcrStageEvidence::Table(evidence) = evidence else {
        panic!("table stage returned non-table evidence");
    };
    evidence
}

fn seal_stage_evidence(slot: &crate::OcrBatchSlotResult) -> &crate::OcrSealStageEvidence {
    let evidence = slot
        .stages
        .iter()
        .find(|stage| stage.stage == OcrStage::Seal)
        .and_then(|stage| stage.evidence.as_ref())
        .unwrap_or_else(|| panic!("seal evidence missing from stages: {:#?}", slot.stages));
    let crate::OcrStageEvidence::Seal(evidence) = evidence else {
        panic!("seal stage returned non-seal evidence");
    };
    evidence
}

#[tokio::test]
#[ignore = "requires all pinned document-fast models and the real rider fixture"]
async fn real_orientation_restores_table_and_seal_evidence_to_the_source_canvas() {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
    assert!(std::env::var_os("A3S_OCR_PAGE_ORIENTATION_MODEL_DIR").is_some());
    assert!(std::env::var_os("A3S_OCR_PICODET_LAYOUT_MODEL_DIR").is_some());
    let root = std::path::Path::new(&fixture_root);
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();

    let request = |stages: Vec<OcrStage>| {
        crate::OcrBatchRequest::new(
            stages,
            [11_u32, 26, 28, 29]
                .into_iter()
                .map(|page| {
                    crate::OcrBatchSlotRequest::new(
                        crate::OcrBatchSlotId::new(format!("page-{page}")).unwrap(),
                        root.join(format!("page-{page:04}.png")),
                    )
                })
                .collect(),
        )
        .unwrap()
    };
    let baseline = client
        .extract_batch(request(vec![OcrStage::Table, OcrStage::Seal]))
        .await
        .unwrap();
    let oriented = client
        .extract_batch(request(vec![
            OcrStage::Orientation,
            OcrStage::Table,
            OcrStage::Seal,
        ]))
        .await
        .unwrap();

    let baseline_table = table_stage_evidence(&baseline.slots[0]);
    let oriented_table = table_stage_evidence(&oriented.slots[0]);
    assert_eq!(oriented_table.canvas, baseline_table.canvas);
    assert_eq!(oriented_table.tables.len(), baseline_table.tables.len());
    for (actual, expected) in oriented_table.tables.iter().zip(&baseline_table.tables) {
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.column_count, expected.column_count);
        assert_eq!(actual.region.bounding_box, expected.region.bounding_box);
        assert_eq!(actual.cells.len(), expected.cells.len());
        assert_eq!(
            actual
                .cells
                .iter()
                .map(|cell| {
                    (
                        cell.row_index,
                        cell.column_index,
                        cell.row_span,
                        cell.column_span,
                    )
                })
                .collect::<Vec<_>>(),
            expected
                .cells
                .iter()
                .map(|cell| {
                    (
                        cell.row_index,
                        cell.column_index,
                        cell.row_span,
                        cell.column_span,
                    )
                })
                .collect::<Vec<_>>()
        );
    }

    let oriented_page_26 = seal_stage_evidence(&oriented.slots[1]);
    let source_page_26 = image::open(root.join("page-0026.png")).unwrap().into_rgb8();
    assert_eq!(
        oriented_page_26.canvas,
        crate::OcrImageCanvas::new(source_page_26.width(), source_page_26.height()).unwrap()
    );
    for (x, y) in [(650, 1_150), (730, 850)] {
        assert!(
            oriented_page_26.seals.iter().any(|seal| {
                let bounds = seal.region.bounding_box;
                bounds.x <= x
                    && bounds.x + bounds.width >= x
                    && bounds.y <= y
                    && bounds.y + bounds.height >= y
            }),
            "oriented page 26 missed reviewed seal point ({x},{y}): {:#?}",
            oriented_page_26.seals
        );
    }

    for slot_index in 2..=3 {
        assert_eq!(
            seal_stage_evidence(&oriented.slots[slot_index]),
            seal_stage_evidence(&baseline.slots[slot_index])
        );
    }
}

#[tokio::test]
#[ignore = "requires all pinned document-fast models and the real rider fixture"]
async fn real_orientation_reports_the_rotated_adjacent_seal_pair_on_source_canvases() {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
    let root = std::path::Path::new(&fixture_root);
    let first_id = crate::OcrBatchSlotId::new("page-26").unwrap();
    let slots = vec![
        crate::OcrBatchSlotRequest::new(first_id.clone(), root.join("page-0026.png")),
        crate::OcrBatchSlotRequest::new(
            crate::OcrBatchSlotId::new("page-27").unwrap(),
            root.join("page-0027.png"),
        )
        .with_adjacent_predecessor(first_id),
    ];
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
    let output = client
        .extract_batch(
            crate::OcrBatchRequest::new(vec![OcrStage::Orientation, OcrStage::Seal], slots)
                .unwrap(),
        )
        .await
        .unwrap();
    for slot in &output.slots {
        let orientation = slot
            .stages
            .iter()
            .find(|stage| stage.stage == OcrStage::Orientation)
            .unwrap();
        eprintln!(
            "{} orientation={:?} seals={:#?}",
            slot.slot_id,
            orientation.status,
            seal_stage_evidence(slot).seals,
        );
    }
}

#[tokio::test]
#[ignore = "requires pinned PP-OCRv6/SLANet-Plus bundles and the real table fixture"]
async fn real_provider_emits_source_backed_grid_geometry_and_cell_text() {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR")
        .expect("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR must name the reviewed fixture root");
    let source = std::path::Path::new(&fixture_root).join("page-0002.png");
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
    let result = client
        .extract_batch(
            crate::OcrBatchRequest::new(
                vec![OcrStage::Preprocessing, OcrStage::Text, OcrStage::Table],
                vec![crate::OcrBatchSlotRequest::new(
                    crate::OcrBatchSlotId::new("page-2").unwrap(),
                    source,
                )],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.slots[0].status, crate::OcrBatchSlotStatus::Completed);
    let table_stage = result.slots[0]
        .stages
        .iter()
        .find(|outcome| outcome.stage == OcrStage::Table)
        .unwrap();
    let crate::OcrStageEvidence::Table(evidence) = table_stage.evidence.as_ref().unwrap() else {
        panic!("table stage returned non-table evidence");
    };
    assert_eq!(evidence.tables.len(), 1);
    let table = &evidence.tables[0];
    assert_eq!((table.row_count, table.column_count), (Some(6), Some(6)));
    assert_eq!(table.cells.len(), 28);
    assert!(table.cells.iter().all(|cell| cell.region.is_some()));
    assert!(
        table
            .cells
            .iter()
            .filter(|cell| cell.text.is_some())
            .count()
            >= 20
    );
    let output = result.slots[0].result.as_ref().unwrap();
    assert!(!output.text.is_empty());
    assert!(!output
        .execution_receipts
        .iter()
        .any(|receipt| { receipt.model.family == "slanet-plus-wired-encoder" }));
    assert!(!result
        .execution_receipts
        .iter()
        .any(|receipt| { receipt.model.family == "slanet-plus-wired-encoder" }));
}

#[tokio::test]
#[ignore = "requires the pinned SLANet-Plus bundle and the real cross-page table fixture"]
async fn real_provider_batches_cross_page_table_fragments() {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR")
        .expect("A3S_OCR_REAL_CROSS_PAGE_TABLE_DIR must name the reviewed fixture root");
    let root = std::path::Path::new(&fixture_root);
    let slots = (2..=4)
        .map(|page| {
            crate::OcrBatchSlotRequest::new(
                crate::OcrBatchSlotId::new(format!("page-{page}")).unwrap(),
                root.join(format!("page-{page:04}.png")),
            )
        })
        .collect();
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
    let result = client
        .extract_batch(crate::OcrBatchRequest::new(vec![OcrStage::Table], slots).unwrap())
        .await
        .unwrap();
    assert_eq!(result.slots.len(), 3);
    let expected = [(6, 6, 28), (8, 6, 25), (3, 6, 15)];
    for (slot, (rows, columns, cells)) in result.slots.iter().zip(expected) {
        assert_eq!(slot.status, crate::OcrBatchSlotStatus::Completed);
        let table_stage = &slot.stages[0];
        let crate::OcrStageEvidence::Table(evidence) = table_stage.evidence.as_ref().unwrap()
        else {
            panic!("table stage returned non-table evidence");
        };
        assert_eq!(evidence.tables.len(), 1);
        let table = &evidence.tables[0];
        assert_eq!(
            (table.row_count, table.column_count),
            (Some(rows), Some(columns))
        );
        assert_eq!(table.cells.len(), cells);
        assert!(table.cells.iter().all(|cell| cell.region.is_some()));
    }
    let table_receipts = result
        .execution_receipts
        .iter()
        .filter(|receipt| receipt.model.family == "slanet-plus-wired-encoder")
        .collect::<Vec<_>>();
    assert!(table_receipts.is_empty());
}

#[tokio::test]
#[ignore = "requires pinned PP-OCRv6/SLANet-Plus bundles and real certificate fixtures"]
async fn real_certificate_backgrounds_do_not_publish_table_semantics() {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
    let root = std::path::Path::new(&fixture_root);
    let slots = [26, 28, 29]
        .into_iter()
        .map(|page| {
            crate::OcrBatchSlotRequest::new(
                crate::OcrBatchSlotId::new(format!("page-{page}")).unwrap(),
                root.join(format!("page-{page:04}.png")),
            )
        })
        .collect();
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
    let result = client
        .extract_batch(
            crate::OcrBatchRequest::new(vec![OcrStage::Text, OcrStage::Table], slots).unwrap(),
        )
        .await
        .unwrap();
    for slot in result.slots {
        assert_eq!(slot.status, crate::OcrBatchSlotStatus::Completed);
        let table_stage = slot
            .stages
            .iter()
            .find(|outcome| outcome.stage == OcrStage::Table)
            .unwrap();
        let crate::OcrStageEvidence::Table(evidence) = table_stage.evidence.as_ref().unwrap()
        else {
            panic!("table stage returned non-table evidence");
        };
        assert!(
            evidence.tables.is_empty(),
            "{} retained false certificate tables: {:#?}",
            slot.slot_id,
            evidence.tables
        );
    }
}

fn real_rider_full_stage_request() -> crate::OcrBatchRequest {
    let fixture_root = std::env::var_os("A3S_OCR_REAL_RIDER_SEAL_DIR")
        .expect("A3S_OCR_REAL_RIDER_SEAL_DIR must name the reviewed fixture root");
    let mut pages = std::fs::read_dir(fixture_root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("png"))
        .collect::<Vec<_>>();
    pages.sort();
    assert_eq!(pages.len(), 29);

    let mut predecessor = None;
    let slots = pages
        .into_iter()
        .enumerate()
        .map(|(index, path)| {
            let slot_id = crate::OcrBatchSlotId::new(format!("page-{}", index + 1)).unwrap();
            let mut slot = crate::OcrBatchSlotRequest::new(slot_id.clone(), path);
            if let Some(predecessor) = predecessor.replace(slot_id) {
                slot = slot.with_adjacent_predecessor(predecessor);
            }
            slot
        })
        .collect::<Vec<_>>();

    crate::OcrBatchRequest::new(
        vec![
            OcrStage::Orientation,
            OcrStage::Text,
            OcrStage::Table,
            OcrStage::Seal,
        ],
        slots,
    )
    .unwrap()
}

fn real_rider_fingerprints(output: &crate::OcrBatchResult) -> (usize, String, String) {
    assert_eq!(output.slots.len(), 29);
    assert!(output
        .slots
        .iter()
        .all(|slot| slot.status != crate::OcrBatchSlotStatus::Failed && slot.result.is_some()));

    let mut text_digest = Sha256::new();
    let mut semantic_digest = Sha256::new();
    let mut text_blocks = 0_usize;
    for slot in &output.slots {
        let result = slot.result.as_ref().unwrap();
        text_digest.update((result.text.len() as u64).to_le_bytes());
        text_digest.update(result.text.as_bytes());
        let mut canonical_result = result.clone();
        canonical_result.execution_receipts.clear();
        let semantic =
            serde_json::to_vec(&(slot.slot_id.clone(), &slot.stages, canonical_result)).unwrap();
        semantic_digest.update((semantic.len() as u64).to_le_bytes());
        semantic_digest.update(semantic);
        text_blocks += result.blocks.len();
    }
    (
        text_blocks,
        format!("{:x}", text_digest.finalize()),
        format!("{:x}", semantic_digest.finalize()),
    )
}

#[tokio::test]
#[ignore = "requires all pinned document-fast models and the complete real rider fixture"]
async fn real_rider_reports_full_stage_gpu_throughput_and_text_fingerprint() {
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
    let started = std::time::Instant::now();
    let output = client
        .extract_batch(real_rider_full_stage_request())
        .await
        .unwrap();
    let elapsed = started.elapsed();
    let (text_blocks, text_sha256, semantic_sha256) = real_rider_fingerprints(&output);
    if let Some(path) = std::env::var_os("A3S_OCR_REAL_FULL_STAGE_OUTPUT") {
        let canonical_slots = output
            .slots
            .iter()
            .map(|slot| {
                let mut result = slot.result.clone();
                if let Some(result) = result.as_mut() {
                    result.execution_receipts.clear();
                }
                (&slot.slot_id, &slot.status, &slot.stages, result)
            })
            .collect::<Vec<_>>();
        std::fs::write(path, serde_json::to_vec(&canonical_slots).unwrap()).unwrap();
    }
    eprintln!(
        "A3S_OCR_REAL_FULL_STAGE_BENCHMARK pages={} elapsed_ms={:.3} pps={:.3} text_blocks={text_blocks} text_sha256={text_sha256} semantic_sha256={semantic_sha256}",
        output.slots.len(),
        elapsed.as_secs_f64() * 1_000.0,
        output.slots.len() as f64 / elapsed.as_secs_f64(),
    );
    assert_eq!(
        text_blocks, 2_518,
        "the reviewed full-stage rider gate must retain every text block"
    );
    assert_eq!(
        text_sha256, "8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8",
        "the reviewed full-stage rider text fingerprint changed"
    );
    assert_eq!(
        semantic_sha256, "bdfedd8b50cc1bf2b863e4892ff3344fba116b1e758ebc8c33d1665a92dd7092",
        "the reviewed full-stage rider semantic fingerprint changed"
    );
}

#[tokio::test]
#[ignore = "requires all pinned document-fast models and the complete real rider fixture"]
async fn real_rider_reports_steady_state_gpu_throughput_and_text_fingerprint() {
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();
    let warmup_started = std::time::Instant::now();
    let warmup = client
        .extract_batch(real_rider_full_stage_request())
        .await
        .unwrap();
    let warmup_elapsed = warmup_started.elapsed();
    let warmup_fingerprints = real_rider_fingerprints(&warmup);
    drop(warmup);

    let started = std::time::Instant::now();
    let output = client
        .extract_batch(real_rider_full_stage_request())
        .await
        .unwrap();
    let elapsed = started.elapsed();
    let (text_blocks, text_sha256, semantic_sha256) = real_rider_fingerprints(&output);
    assert_eq!(
        (text_blocks, text_sha256.as_str(), semantic_sha256.as_str()),
        (
            warmup_fingerprints.0,
            warmup_fingerprints.1.as_str(),
            warmup_fingerprints.2.as_str(),
        ),
        "the steady-state request changed the warm-up output"
    );
    eprintln!(
        "A3S_OCR_REAL_FULL_STAGE_STEADY_BENCHMARK pages={} warmup_ms={:.3} elapsed_ms={:.3} pps={:.3} text_blocks={text_blocks} text_sha256={text_sha256} semantic_sha256={semantic_sha256}",
        output.slots.len(),
        warmup_elapsed.as_secs_f64() * 1_000.0,
        elapsed.as_secs_f64() * 1_000.0,
        output.slots.len() as f64 / elapsed.as_secs_f64(),
    );
    assert_eq!(
        text_blocks, 2_518,
        "the reviewed steady-state rider gate must retain every text block"
    );
    assert_eq!(
        text_sha256, "8e5a458d896ffee83f46e775e9fcd9f07c179d6b17aec2cf90b9845c3c22dbf8",
        "the reviewed steady-state rider text fingerprint changed"
    );
    assert_eq!(
        semantic_sha256, "bdfedd8b50cc1bf2b863e4892ff3344fba116b1e758ebc8c33d1665a92dd7092",
        "the reviewed steady-state rider semantic fingerprint changed"
    );
}
