use super::super::DocumentFastOcrProvider;

const BASELINE_BATCH_SIZE: usize = 32;
const CANDIDATE_BATCH_SIZE: usize = 128;

#[tokio::test]
#[ignore = "requires the pinned PP-OCRv6/SLANet-Plus bundles and the complete real rider fixture"]
async fn larger_recognition_batches_preserve_text_geometry_and_report_confidence_drift() {
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
    let slots = pages
        .into_iter()
        .enumerate()
        .map(|(index, path)| {
            crate::OcrBatchSlotRequest::new(
                crate::OcrBatchSlotId::new(format!("page-{}", index + 1)).unwrap(),
                path,
            )
        })
        .collect::<Vec<_>>();
    let client =
        crate::OcrClient::with_provider(DocumentFastOcrProvider::from_env().unwrap()).unwrap();

    std::env::set_var(
        "A3S_OCR_TEST_RECOGNITION_MAX_BATCH_SIZE",
        BASELINE_BATCH_SIZE.to_string(),
    );
    let baseline = client
        .extract_batch(
            crate::OcrBatchRequest::new(vec![crate::OcrStage::Text], slots.clone()).unwrap(),
        )
        .await
        .unwrap();
    std::env::set_var(
        "A3S_OCR_TEST_RECOGNITION_MAX_BATCH_SIZE",
        CANDIDATE_BATCH_SIZE.to_string(),
    );
    let candidate = client
        .extract_batch(crate::OcrBatchRequest::new(vec![crate::OcrStage::Text], slots).unwrap())
        .await
        .unwrap();
    std::env::remove_var("A3S_OCR_TEST_RECOGNITION_MAX_BATCH_SIZE");

    assert_eq!(
        without_recognition_confidence_and_receipts(baseline.clone()),
        without_recognition_confidence_and_receipts(candidate.clone()),
        "a larger exact-width batch changed text, geometry, detection evidence, or another canonical field"
    );

    let mut compared = 0_usize;
    let mut changed = 0_usize;
    let mut maximum_absolute_difference = 0_f32;
    let mut maximum_ulp_difference = 0_u32;
    for (baseline_slot, candidate_slot) in baseline.slots.iter().zip(&candidate.slots) {
        let baseline_blocks = &baseline_slot.result.as_ref().unwrap().blocks;
        let candidate_blocks = &candidate_slot.result.as_ref().unwrap().blocks;
        assert_eq!(baseline_blocks.len(), candidate_blocks.len());
        for (baseline_block, candidate_block) in baseline_blocks.iter().zip(candidate_blocks) {
            match (baseline_block.confidence, candidate_block.confidence) {
                (Some(baseline), Some(candidate)) => {
                    assert!(baseline.is_finite() && candidate.is_finite());
                    assert!((0.0..=1.0).contains(&baseline));
                    assert!((0.0..=1.0).contains(&candidate));
                    compared += 1;
                    changed += usize::from(baseline.to_bits() != candidate.to_bits());
                    maximum_absolute_difference =
                        maximum_absolute_difference.max((baseline - candidate).abs());
                    maximum_ulp_difference = maximum_ulp_difference
                        .max(baseline.to_bits().abs_diff(candidate.to_bits()));
                }
                (None, None) => {}
                _ => panic!("recognition confidence availability changed with batch size"),
            }
        }
    }
    eprintln!(
        "A3S_OCR_RECOGNITION_BATCH_PARITY baseline_batch={BASELINE_BATCH_SIZE} candidate_batch={CANDIDATE_BATCH_SIZE} compared={compared} changed={changed} max_abs={maximum_absolute_difference:.9e} max_ulp={maximum_ulp_difference}"
    );
}

fn without_recognition_confidence_and_receipts(
    mut output: crate::OcrBatchResult,
) -> crate::OcrBatchResult {
    output.execution_receipts.clear();
    for slot in &mut output.slots {
        if let Some(result) = slot.result.as_mut() {
            result.execution_receipts.clear();
            for block in &mut result.blocks {
                block.confidence = None;
            }
        }
    }
    output
}
