use std::collections::BTreeSet;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub(crate) const BASELINE_SCHEMA: &str = "a3s.ocr.execution-baseline.v1";
pub(crate) const EVIDENCE_SCOPE: &str = "a3s-ocr-real-single-image";
pub(crate) const PROVIDER_CLASS: &str = "embedded-native";
pub(crate) const FIXTURE_ID: &str = "paddleocr-general-ocr-002";
pub(crate) const FIXTURE_SHA256: &str =
    "4425af33dd163cf73bdff502bd35ee527e9bdd5725501db1da78bfdae9f538f4";
pub(crate) const FIXTURE_BYTES: u64 = 128_713;
pub(crate) const FIXTURE_WIDTH: u32 = 896;
pub(crate) const FIXTURE_HEIGHT: u32 = 528;
pub(crate) const FIXTURE_MEDIA_TYPE: &str = "image/jpeg";
pub(crate) const EXPECTED_BLOCKS: u32 = 30;
pub(crate) const EXPECTED_RECEIPTS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum SourceTreeState {
    Clean,
    Modified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum BuildProfile {
    Debug,
    Release,
}

impl BuildProfile {
    pub(crate) fn current() -> Self {
        if cfg!(debug_assertions) {
            Self::Debug
        } else {
            Self::Release
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum SamplePhase {
    ColdStart,
    Warm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BuildEvidence {
    pub ocr_version: String,
    pub ocr_commit: String,
    pub source_tree_state: SourceTreeState,
    pub profile: BuildProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SystemEvidence {
    pub host_label: String,
    pub os: String,
    pub architecture: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub ram_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FixtureEvidence {
    pub id: String,
    pub media_type: String,
    pub sha256: String,
    pub byte_length: u64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ExecutionFingerprint {
    pub model_family: String,
    pub model_revision: String,
    pub weights_sha256: String,
    pub runtime_name: String,
    pub runtime_version: String,
    pub device: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ProviderEvidence {
    pub id: String,
    pub engine: String,
    pub model: String,
    pub sends_source_off_device: bool,
    pub execution_fingerprints: Vec<ExecutionFingerprint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BenchmarkConfiguration {
    pub warmup_samples: usize,
    pub measured_samples: usize,
    pub includes_public_client_io: bool,
    pub process_rss_sampling_interval_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct OcrSample {
    pub phase: SamplePhase,
    pub index: usize,
    pub elapsed_nanos: u64,
    pub time_to_first_result_nanos: u64,
    /// Integer milli-images per second derived from the measured interval.
    pub images_per_second_milli: u64,
    pub resident_bytes_before: u64,
    pub peak_resident_bytes: u64,
    pub resident_bytes_after: u64,
    pub block_count: u32,
    pub execution_receipt_count: u32,
    pub output_sha256: String,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MetricSummary {
    pub minimum: u64,
    pub p50: u64,
    pub p95: u64,
    pub maximum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct WarmSummary {
    pub elapsed_nanos: MetricSummary,
    pub images_per_second_milli: MetricSummary,
    pub peak_resident_bytes: MetricSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct DeterminismEvidence {
    pub byte_stable: bool,
    pub output_sha256: String,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ExecutionBaselineReport {
    pub schema: String,
    pub evidence_scope: String,
    pub provider_class: String,
    pub build: BuildEvidence,
    pub system: SystemEvidence,
    pub fixture: FixtureEvidence,
    pub provider: ProviderEvidence,
    pub configuration: BenchmarkConfiguration,
    pub cold_start: OcrSample,
    pub warm_samples: Vec<OcrSample>,
    pub warm_summary: WarmSummary,
    pub determinism: DeterminismEvidence,
}

impl ExecutionBaselineReport {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != BASELINE_SCHEMA
            || self.evidence_scope != EVIDENCE_SCOPE
            || self.provider_class != PROVIDER_CLASS
            || self.build.ocr_version != env!("CARGO_PKG_VERSION")
            || !is_revision(&self.build.ocr_commit)
            || self.system.host_label.is_empty()
            || self.system.os.is_empty()
            || self.system.architecture.is_empty()
            || self.system.cpu_model.trim().is_empty()
            || self.system.logical_cpus == 0
            || self.system.ram_bytes == 0
        {
            bail!("the OCR execution-baseline report header is invalid");
        }
        if self.fixture.id != FIXTURE_ID
            || self.fixture.media_type != FIXTURE_MEDIA_TYPE
            || self.fixture.sha256 != FIXTURE_SHA256
            || self.fixture.byte_length != FIXTURE_BYTES
            || self.fixture.width != FIXTURE_WIDTH
            || self.fixture.height != FIXTURE_HEIGHT
        {
            bail!("the OCR execution-baseline fixture evidence is invalid");
        }
        if self.provider.id != "pp-ocr-v6"
            || self.provider.engine != "a3s-power-native"
            || self.provider.model != "PP-OCRv6_small"
            || self.provider.sends_source_off_device
            || self.provider.execution_fingerprints.len() != 2
        {
            bail!("the OCR execution-baseline provider evidence is invalid");
        }
        let unique = self
            .provider
            .execution_fingerprints
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if unique.len() != self.provider.execution_fingerprints.len()
            || unique
                .iter()
                .any(|fingerprint| !valid_fingerprint(fingerprint))
            || unique.iter().cloned().collect::<Vec<_>>() != self.provider.execution_fingerprints
        {
            bail!("execution fingerprints must be valid, unique, and sorted");
        }
        if self.configuration.warmup_samples > 20
            || !(1..=100).contains(&self.configuration.measured_samples)
            || !self.configuration.includes_public_client_io
            || self.configuration.process_rss_sampling_interval_ms != 1
            || self.warm_samples.len() != self.configuration.measured_samples
        {
            bail!("the OCR execution-baseline configuration is invalid");
        }
        validate_sample(&self.cold_start, SamplePhase::ColdStart, 0)?;
        for (offset, sample) in self.warm_samples.iter().enumerate() {
            validate_sample(sample, SamplePhase::Warm, offset + 1)?;
        }
        let expected_summary = WarmSummary::from_samples(&self.warm_samples)?;
        if self.warm_summary != expected_summary {
            bail!("the warm summary is not derived from the measured samples");
        }
        let samples = std::iter::once(&self.cold_start).chain(&self.warm_samples);
        let all_stable = samples.clone().all(|sample| {
            sample.output_sha256 == self.determinism.output_sha256
                && sample.output_bytes == self.determinism.output_bytes
        });
        if !self.determinism.byte_stable
            || !all_stable
            || !is_sha256(&self.determinism.output_sha256)
            || self.determinism.output_bytes == 0
        {
            bail!("the OCR execution-baseline output is not byte-stable");
        }
        Ok(())
    }
}

impl WarmSummary {
    pub(crate) fn from_samples(samples: &[OcrSample]) -> Result<Self> {
        if samples.is_empty() {
            bail!("a warm summary requires at least one measured sample");
        }
        Ok(Self {
            elapsed_nanos: summarize(samples.iter().map(|sample| sample.elapsed_nanos))?,
            images_per_second_milli: summarize(
                samples.iter().map(|sample| sample.images_per_second_milli),
            )?,
            peak_resident_bytes: summarize(
                samples.iter().map(|sample| sample.peak_resident_bytes),
            )?,
        })
    }
}

pub(crate) fn throughput_milli(elapsed_nanos: u64) -> Result<u64> {
    if elapsed_nanos == 0 {
        bail!("throughput requires a positive elapsed interval");
    }
    u64::try_from(1_000_000_000_000_u128 / u128::from(elapsed_nanos))
        .context("throughput cannot be represented")
}

fn validate_sample(sample: &OcrSample, phase: SamplePhase, index: usize) -> Result<()> {
    if sample.phase != phase
        || sample.index != index
        || sample.elapsed_nanos == 0
        || sample.time_to_first_result_nanos != sample.elapsed_nanos
        || sample.images_per_second_milli != throughput_milli(sample.elapsed_nanos)?
        || sample.resident_bytes_before == 0
        || sample.peak_resident_bytes < sample.resident_bytes_before
        || sample.peak_resident_bytes < sample.resident_bytes_after
        || sample.block_count != EXPECTED_BLOCKS
        || sample.execution_receipt_count != EXPECTED_RECEIPTS
        || !is_sha256(&sample.output_sha256)
        || sample.output_bytes == 0
    {
        bail!("an OCR execution-baseline sample is invalid");
    }
    Ok(())
}

fn valid_fingerprint(fingerprint: &ExecutionFingerprint) -> bool {
    !fingerprint.model_family.is_empty()
        && !fingerprint.model_revision.is_empty()
        && is_sha256(&fingerprint.weights_sha256)
        && fingerprint.runtime_name == "a3s-power-native"
        && !fingerprint.runtime_version.is_empty()
        && !fingerprint.device.is_empty()
}

fn summarize(values: impl Iterator<Item = u64>) -> Result<MetricSummary> {
    let mut values = values.collect::<Vec<_>>();
    if values.is_empty() || values.contains(&0) {
        bail!("metric summaries require positive samples");
    }
    values.sort_unstable();
    Ok(MetricSummary {
        minimum: values[0],
        p50: percentile(&values, 50),
        p95: percentile(&values, 95),
        maximum: *values.last().expect("non-empty metric samples"),
    })
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    let rank = sorted.len().saturating_mul(percentile).saturating_add(99) / 100;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn is_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(phase: SamplePhase, index: usize, elapsed_nanos: u64) -> OcrSample {
        OcrSample {
            phase,
            index,
            elapsed_nanos,
            time_to_first_result_nanos: elapsed_nanos,
            images_per_second_milli: throughput_milli(elapsed_nanos).unwrap(),
            resident_bytes_before: 10,
            peak_resident_bytes: 20,
            resident_bytes_after: 15,
            block_count: EXPECTED_BLOCKS,
            execution_receipt_count: EXPECTED_RECEIPTS,
            output_sha256: "a".repeat(64),
            output_bytes: 100,
        }
    }

    fn report() -> ExecutionBaselineReport {
        let warm_samples = vec![
            sample(SamplePhase::Warm, 1, 3_000_000_000),
            sample(SamplePhase::Warm, 2, 1_000_000_000),
            sample(SamplePhase::Warm, 3, 2_000_000_000),
        ];
        ExecutionBaselineReport {
            schema: BASELINE_SCHEMA.to_owned(),
            evidence_scope: EVIDENCE_SCOPE.to_owned(),
            provider_class: PROVIDER_CLASS.to_owned(),
            build: BuildEvidence {
                ocr_version: env!("CARGO_PKG_VERSION").to_owned(),
                ocr_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                source_tree_state: SourceTreeState::Clean,
                profile: BuildProfile::Release,
            },
            system: SystemEvidence {
                host_label: "a3s-lab-ws-01".to_owned(),
                os: "linux".to_owned(),
                architecture: "x86_64".to_owned(),
                cpu_model: "Example CPU".to_owned(),
                logical_cpus: 8,
                ram_bytes: 1024,
            },
            fixture: FixtureEvidence {
                id: FIXTURE_ID.to_owned(),
                media_type: FIXTURE_MEDIA_TYPE.to_owned(),
                sha256: FIXTURE_SHA256.to_owned(),
                byte_length: FIXTURE_BYTES,
                width: FIXTURE_WIDTH,
                height: FIXTURE_HEIGHT,
            },
            provider: ProviderEvidence {
                id: "pp-ocr-v6".to_owned(),
                engine: "a3s-power-native".to_owned(),
                model: "PP-OCRv6_small".to_owned(),
                sends_source_off_device: false,
                execution_fingerprints: vec![
                    ExecutionFingerprint {
                        model_family: "PP-OCRv6_small_det".to_owned(),
                        model_revision: "paddlex-paddle3.0.0-native-v1".to_owned(),
                        weights_sha256: "b".repeat(64),
                        runtime_name: "a3s-power-native".to_owned(),
                        runtime_version: "0.7.0".to_owned(),
                        device: "cpu".to_owned(),
                    },
                    ExecutionFingerprint {
                        model_family: "PP-OCRv6_small_rec".to_owned(),
                        model_revision: "paddlex-paddle3.0.0-native-v1".to_owned(),
                        weights_sha256: "c".repeat(64),
                        runtime_name: "a3s-power-native".to_owned(),
                        runtime_version: "0.7.0".to_owned(),
                        device: "cpu".to_owned(),
                    },
                ],
            },
            configuration: BenchmarkConfiguration {
                warmup_samples: 1,
                measured_samples: warm_samples.len(),
                includes_public_client_io: true,
                process_rss_sampling_interval_ms: 1,
            },
            cold_start: sample(SamplePhase::ColdStart, 0, 4_000_000_000),
            warm_summary: WarmSummary::from_samples(&warm_samples).unwrap(),
            warm_samples,
            determinism: DeterminismEvidence {
                byte_stable: true,
                output_sha256: "a".repeat(64),
                output_bytes: 100,
            },
        }
    }

    #[test]
    fn validates_a_strict_path_free_real_provider_report() {
        let report = report();
        report.validate().unwrap();
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(!encoded.to_ascii_lowercase().contains("path"));
        assert!(!encoded.contains("text"));
    }

    #[test]
    fn derives_nearest_rank_warm_summary_and_rejects_drift() {
        let mut report = report();
        assert_eq!(report.warm_summary.elapsed_nanos.p50, 2_000_000_000);
        assert_eq!(report.warm_summary.elapsed_nanos.p95, 3_000_000_000);
        report.warm_samples[0].output_sha256 = "d".repeat(64);
        assert!(report.validate().is_err());
    }
}
