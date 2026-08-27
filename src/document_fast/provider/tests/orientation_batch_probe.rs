use std::time::{Duration, Instant};

use super::super::DocumentFastOcrProvider;

const BASELINE_BATCH_SIZE: usize = 8;

#[tokio::test]
#[ignore = "requires the pinned orientation/PP-OCRv6/SLANet-Plus bundles and the complete real rider fixture"]
async fn resource_bounded_orientation_batches_preserve_results_and_report_throughput() {
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

    let (baseline_reference, _) =
        extract_orientation(&client, &slots, Some(BASELINE_BATCH_SIZE)).await;
    let (candidate_reference, _) = extract_orientation(&client, &slots, None).await;
    assert_eq!(
        without_receipts(baseline_reference.clone()),
        without_receipts(candidate_reference.clone()),
        "resource-derived page-orientation batching changed a canonical result"
    );

    let mut baseline_durations = Vec::with_capacity(3);
    let mut candidate_durations = Vec::with_capacity(3);
    for _ in 0..3 {
        let (baseline, elapsed) =
            extract_orientation(&client, &slots, Some(BASELINE_BATCH_SIZE)).await;
        assert_eq!(
            without_receipts(baseline_reference.clone()),
            without_receipts(baseline)
        );
        baseline_durations.push(elapsed);

        let (candidate, elapsed) = extract_orientation(&client, &slots, None).await;
        assert_eq!(
            without_receipts(candidate_reference.clone()),
            without_receipts(candidate)
        );
        candidate_durations.push(elapsed);
    }
    std::env::remove_var("A3S_OCR_TEST_ORIENTATION_MAX_BATCH_SIZE");

    let baseline = median(&mut baseline_durations);
    let candidate = median(&mut candidate_durations);
    eprintln!(
        "A3S_OCR_ORIENTATION_BATCH_BENCHMARK pages={} baseline_batch={BASELINE_BATCH_SIZE} baseline_ms={:.3} baseline_pps={:.3} candidate_batch=power-limited candidate_ms={:.3} candidate_pps={:.3}",
        slots.len(),
        baseline.as_secs_f64() * 1_000.0,
        slots.len() as f64 / baseline.as_secs_f64(),
        candidate.as_secs_f64() * 1_000.0,
        slots.len() as f64 / candidate.as_secs_f64(),
    );
}

async fn extract_orientation(
    client: &crate::OcrClient,
    slots: &[crate::OcrBatchSlotRequest],
    batch_cap: Option<usize>,
) -> (crate::OcrBatchResult, Duration) {
    match batch_cap {
        Some(batch_cap) => std::env::set_var(
            "A3S_OCR_TEST_ORIENTATION_MAX_BATCH_SIZE",
            batch_cap.to_string(),
        ),
        None => std::env::remove_var("A3S_OCR_TEST_ORIENTATION_MAX_BATCH_SIZE"),
    }
    let started = Instant::now();
    let output = client
        .extract_batch(
            crate::OcrBatchRequest::new(vec![crate::OcrStage::Orientation], slots.to_vec())
                .unwrap(),
        )
        .await
        .unwrap();
    (output, started.elapsed())
}

fn without_receipts(mut output: crate::OcrBatchResult) -> crate::OcrBatchResult {
    output.execution_receipts.clear();
    for slot in &mut output.slots {
        if let Some(result) = slot.result.as_mut() {
            result.execution_receipts.clear();
        }
    }
    output
}

fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}
