use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use a3s_use_core::Readiness;
use a3s_use_ocr::{OcrBlock, OcrClient, OcrExecutionReceipt, OcrRequest, OcrResult};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::memory::{resident_bytes, ResidentMemorySampler};
use super::report::{
    throughput_milli, BenchmarkConfiguration, BuildEvidence, BuildProfile, DeterminismEvidence,
    ExecutionBaselineReport, ExecutionFingerprint, FixtureEvidence, OcrSample, ProviderEvidence,
    SamplePhase, SystemEvidence, WarmSummary, BASELINE_SCHEMA, EVIDENCE_SCOPE, EXPECTED_BLOCKS,
    EXPECTED_RECEIPTS, FIXTURE_BYTES, FIXTURE_HEIGHT, FIXTURE_ID, FIXTURE_MEDIA_TYPE,
    FIXTURE_SHA256, FIXTURE_WIDTH, PROVIDER_CLASS,
};
use super::Arguments;

pub(crate) async fn run_suite(arguments: &Arguments) -> Result<ExecutionBaselineReport> {
    let fixture = inspect_fixture(&arguments.fixture)?;
    let client = OcrClient::from_env().context("could not construct the PP-OCRv6 client")?;
    let diagnostic = client.diagnostic();
    if diagnostic.readiness != Readiness::Ready {
        bail!("the PP-OCRv6 provider is not ready: {}", diagnostic.message);
    }
    let descriptor = client.provider().clone();

    let cold = measure(&client, &arguments.fixture, SamplePhase::ColdStart, 0).await?;
    validate_result(&cold.result, &fixture)?;
    let reference = cold.canonical.clone();

    for index in 0..arguments.warmup_samples {
        let result = client
            .extract(OcrRequest {
                path: arguments.fixture.clone(),
            })
            .await
            .with_context(|| format!("warmup OCR sample {} failed", index + 1))?;
        validate_result(&result, &fixture)?;
        let canonical = canonical_output(&result)?;
        if canonical != reference {
            bail!("warmup OCR output diverged from the cold-start result");
        }
    }

    let mut warm_samples = Vec::with_capacity(arguments.samples);
    for index in 1..=arguments.samples {
        let measured = measure(&client, &arguments.fixture, SamplePhase::Warm, index).await?;
        validate_result(&measured.result, &fixture)?;
        if measured.canonical != reference {
            bail!("measured warm OCR output diverged from the cold-start result");
        }
        warm_samples.push(measured.sample);
    }

    let fingerprints = execution_fingerprints(&cold.result)?;
    let output_sha256 = format!("{:x}", Sha256::digest(&reference));
    let output_bytes = u64::try_from(reference.len())
        .context("canonical OCR output size cannot be represented")?;
    let report = ExecutionBaselineReport {
        schema: BASELINE_SCHEMA.to_owned(),
        evidence_scope: EVIDENCE_SCOPE.to_owned(),
        provider_class: PROVIDER_CLASS.to_owned(),
        build: BuildEvidence {
            ocr_version: env!("CARGO_PKG_VERSION").to_owned(),
            ocr_commit: arguments.ocr_commit.clone(),
            source_tree_state: arguments.source_tree_state,
            profile: BuildProfile::current(),
        },
        system: SystemEvidence {
            host_label: arguments.host_label.clone(),
            os: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            cpu_model: arguments.cpu_model.clone(),
            logical_cpus: std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1),
            ram_bytes: arguments.ram_bytes,
        },
        fixture,
        provider: ProviderEvidence {
            id: descriptor.id,
            engine: descriptor.engine,
            model: cold
                .result
                .model
                .clone()
                .context("PP-OCRv6 returned no model identity")?,
            sends_source_off_device: descriptor.sends_source_off_device,
            execution_fingerprints: fingerprints,
        },
        configuration: BenchmarkConfiguration {
            warmup_samples: arguments.warmup_samples,
            measured_samples: arguments.samples,
            includes_public_client_io: true,
            process_rss_sampling_interval_ms: 1,
        },
        cold_start: cold.sample,
        warm_summary: WarmSummary::from_samples(&warm_samples)?,
        warm_samples,
        determinism: DeterminismEvidence {
            byte_stable: true,
            output_sha256,
            output_bytes,
        },
    };
    report.validate()?;
    Ok(report)
}

struct MeasuredSample {
    sample: OcrSample,
    result: OcrResult,
    canonical: Vec<u8>,
}

async fn measure(
    client: &OcrClient,
    fixture: &Path,
    phase: SamplePhase,
    index: usize,
) -> Result<MeasuredSample> {
    let resident_bytes_before = resident_bytes()?;
    let memory = ResidentMemorySampler::start()?;
    let started = Instant::now();
    let result = client
        .extract(OcrRequest {
            path: fixture.to_path_buf(),
        })
        .await
        .with_context(|| format!("OCR {phase:?} sample {index} failed"))?;
    let elapsed_nanos = duration_nanos(started.elapsed());
    let resident_bytes_after = resident_bytes()?;
    let peak_resident_bytes = memory
        .finish()?
        .max(resident_bytes_before)
        .max(resident_bytes_after);
    let canonical = canonical_output(&result)?;
    let output_sha256 = format!("{:x}", Sha256::digest(&canonical));
    let output_bytes = u64::try_from(canonical.len())
        .context("canonical OCR output size cannot be represented")?;
    let block_count =
        u32::try_from(result.blocks.len()).context("OCR block count cannot be represented")?;
    let execution_receipt_count = u32::try_from(result.execution_receipts.len())
        .context("OCR execution receipt count cannot be represented")?;
    Ok(MeasuredSample {
        sample: OcrSample {
            phase,
            index,
            elapsed_nanos,
            // OcrClient publishes one atomic result, so its first observable
            // result is the completed extraction rather than an internal block.
            time_to_first_result_nanos: elapsed_nanos,
            images_per_second_milli: throughput_milli(elapsed_nanos)?,
            resident_bytes_before,
            peak_resident_bytes,
            resident_bytes_after,
            block_count,
            execution_receipt_count,
            output_sha256,
            output_bytes,
        },
        result,
        canonical,
    })
}

fn inspect_fixture(path: &Path) -> Result<FixtureEvidence> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("could not inspect fixture '{}'", path.display()))?;
    if !metadata.is_file() || metadata.len() != FIXTURE_BYTES {
        bail!("the fixture must be the {FIXTURE_BYTES}-byte pinned {FIXTURE_ID} image");
    }
    let bytes = std::fs::read(path)
        .with_context(|| format!("could not read fixture '{}'", path.display()))?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    if sha256 != FIXTURE_SHA256 {
        bail!("the fixture SHA-256 does not match the pinned official image");
    }
    let image = image::load_from_memory(&bytes).context("could not decode the pinned fixture")?;
    if image.width() != FIXTURE_WIDTH || image.height() != FIXTURE_HEIGHT {
        bail!("the pinned fixture dimensions are invalid");
    }
    Ok(FixtureEvidence {
        id: FIXTURE_ID.to_owned(),
        media_type: FIXTURE_MEDIA_TYPE.to_owned(),
        sha256,
        byte_length: metadata.len(),
        width: image.width(),
        height: image.height(),
    })
}

fn validate_result(result: &OcrResult, fixture: &FixtureEvidence) -> Result<()> {
    let mut mismatches = Vec::new();
    if result.provider != "pp-ocr-v6" {
        mismatches.push("provider");
    }
    if result.engine != "a3s-power-native" {
        mismatches.push("engine");
    }
    if result.model.as_deref() != Some("PP-OCRv6_small") {
        mismatches.push("model");
    }
    if result.source.media_type != fixture.media_type {
        mismatches.push("mediaType");
    }
    if result.source.size != fixture.byte_length {
        mismatches.push("sourceSize");
    }
    if result.source.sha256 != fixture.sha256 {
        mismatches.push("sourceSha256");
    }
    if result.text.trim().is_empty() {
        mismatches.push("emptyText");
    }
    if result.blocks.len() != EXPECTED_BLOCKS as usize {
        mismatches.push("blockCount");
    }
    if result.execution_receipts.len() != EXPECTED_RECEIPTS as usize {
        mismatches.push("receiptCount");
    }
    if !result.warnings.is_empty() {
        mismatches.push("warnings");
    }
    if !mismatches.is_empty() {
        bail!(
            "PP-OCRv6 result evidence mismatched fields {mismatches:?}: provider={:?}, engine={:?}, model={:?}, mediaType={:?}, sourceSize={}, sourceSha256={}, emptyText={}, blockCount={}, receiptCount={}, warningCount={}",
            result.provider,
            result.engine,
            result.model,
            result.source.media_type,
            result.source.size,
            result.source.sha256,
            result.text.trim().is_empty(),
            result.blocks.len(),
            result.execution_receipts.len(),
            result.warnings.len()
        );
    }
    if result
        .execution_receipts
        .iter()
        .any(|receipt| receipt.schema != "a3s.power.embedded-execution-receipt.v1")
    {
        bail!("PP-OCRv6 returned an unexpected Power receipt schema");
    }
    Ok(())
}

fn execution_fingerprints(result: &OcrResult) -> Result<Vec<ExecutionFingerprint>> {
    let fingerprints = result
        .execution_receipts
        .iter()
        .map(|receipt| ExecutionFingerprint {
            model_family: receipt.model.family.clone(),
            model_revision: receipt.model.revision.clone(),
            weights_sha256: receipt.model.weights_sha256.clone(),
            runtime_name: receipt.runtime.name.clone(),
            runtime_version: receipt.runtime.version.clone(),
            device: receipt.runtime.device.clone(),
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if fingerprints.len() != 2 {
        bail!("the pinned PP-OCRv6 run did not expose two model fingerprints");
    }
    Ok(fingerprints)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CanonicalOutput<'a> {
    provider: &'a str,
    engine: &'a str,
    model: &'a Option<String>,
    source: CanonicalSource<'a>,
    text: &'a str,
    blocks: &'a [OcrBlock],
    execution_receipts: &'a [OcrExecutionReceipt],
    warnings: &'a [String],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CanonicalSource<'a> {
    media_type: &'a str,
    size: u64,
    sha256: &'a str,
}

fn canonical_output(result: &OcrResult) -> Result<Vec<u8>> {
    serde_json::to_vec(&CanonicalOutput {
        provider: &result.provider,
        engine: &result.engine,
        model: &result.model,
        source: CanonicalSource {
            media_type: &result.source.media_type,
            size: result.source.size,
            sha256: &result.source.sha256,
        },
        text: &result.text,
        blocks: &result.blocks,
        execution_receipts: &result.execution_receipts,
        warnings: &result.warnings,
    })
    .context("could not encode canonical OCR output evidence")
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}
