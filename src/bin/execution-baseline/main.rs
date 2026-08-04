mod memory;
mod report;
mod runner;

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use report::SourceTreeState;
use runner::run_suite;

const USAGE: &str = "\
Record a path-free real PP-OCRv6 single-image execution baseline.\n\
\n\
Usage:\n\
  a3s-use-ocr-execution-bench \\\n+    --ocr-commit <40-or-64-character-revision> \\\n+    --source-tree-state <clean|modified> \\\n+    --host-label <stable-host-name> \\\n+    --cpu-model <model-name> \\\n+    --ram-bytes <bytes> \\\n+    --fixture <general_ocr_002.png> \\\n+    [--warmup-samples <count>] \\\n+    [--samples <count>]\n";

#[derive(Debug, Clone)]
pub(crate) struct Arguments {
    pub ocr_commit: String,
    pub source_tree_state: SourceTreeState,
    pub host_label: String,
    pub cpu_model: String,
    pub ram_bytes: u64,
    pub fixture: PathBuf,
    pub warmup_samples: usize,
    pub samples: usize,
}

impl Arguments {
    fn parse_from<I>(arguments: I) -> Result<Self>
    where
        I: IntoIterator<Item = OsString>,
    {
        let mut arguments = arguments.into_iter();
        let mut ocr_commit = None;
        let mut source_tree_state = None;
        let mut host_label = None;
        let mut cpu_model = None;
        let mut ram_bytes = None;
        let mut fixture = None;
        let mut warmup_samples = None;
        let mut samples = None;

        while let Some(argument) = arguments.next() {
            let flag = argument
                .into_string()
                .map_err(|_| anyhow::anyhow!("argument names must be valid UTF-8"))?;
            let mut value = || {
                arguments
                    .next()
                    .with_context(|| format!("{flag} requires a value"))
            };
            match flag.as_str() {
                "--ocr-commit" => {
                    set_once(&mut ocr_commit, utf8(value()?, "--ocr-commit")?, &flag)?
                }
                "--source-tree-state" => set_once(
                    &mut source_tree_state,
                    parse_tree_state(&utf8(value()?, "--source-tree-state")?)?,
                    &flag,
                )?,
                "--host-label" => {
                    set_once(&mut host_label, utf8(value()?, "--host-label")?, &flag)?
                }
                "--cpu-model" => set_once(&mut cpu_model, utf8(value()?, "--cpu-model")?, &flag)?,
                "--ram-bytes" => set_once(
                    &mut ram_bytes,
                    parse_number(value()?, "--ram-bytes")?,
                    &flag,
                )?,
                "--fixture" => set_once(&mut fixture, PathBuf::from(value()?), &flag)?,
                "--warmup-samples" => set_once(
                    &mut warmup_samples,
                    parse_number(value()?, "--warmup-samples")?,
                    &flag,
                )?,
                "--samples" => set_once(&mut samples, parse_number(value()?, "--samples")?, &flag)?,
                _ => bail!("unknown argument '{flag}'"),
            }
        }

        let parsed = Self {
            ocr_commit: ocr_commit.context("--ocr-commit is required")?,
            source_tree_state: source_tree_state.context("--source-tree-state is required")?,
            host_label: host_label.context("--host-label is required")?,
            cpu_model: cpu_model.context("--cpu-model is required")?,
            ram_bytes: ram_bytes.context("--ram-bytes is required")?,
            fixture: fixture.context("--fixture is required")?,
            warmup_samples: warmup_samples.unwrap_or(1),
            samples: samples.unwrap_or(5),
        };
        parsed.validate()?;
        Ok(parsed)
    }

    fn validate(&self) -> Result<()> {
        let revision_is_valid = matches!(self.ocr_commit.len(), 40 | 64)
            && self
                .ocr_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
        if !revision_is_valid {
            bail!("--ocr-commit must be a lowercase 40- or 64-character hexadecimal revision");
        }
        if self.host_label.is_empty()
            || self.host_label.len() > 128
            || !self
                .host_label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            bail!("--host-label must use 1 through 128 ASCII letters, digits, dots, underscores, or hyphens");
        }
        if self.cpu_model.trim().is_empty()
            || self.cpu_model.len() > 256
            || self.cpu_model.chars().any(char::is_control)
        {
            bail!("--cpu-model must contain 1 through 256 non-control characters");
        }
        if self.ram_bytes == 0 || self.warmup_samples > 20 || !(1..=100).contains(&self.samples) {
            bail!("RAM, warmup, and measured sample bounds are invalid");
        }
        if self.fixture.as_os_str().is_empty() {
            bail!("--fixture must name the pinned official image");
        }
        Ok(())
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<()> {
    if slot.replace(value).is_some() {
        bail!("{flag} must not be repeated");
    }
    Ok(())
}

fn utf8(value: OsString, flag: &str) -> Result<String> {
    value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{flag} must be valid UTF-8"))
}

fn parse_number<T>(value: OsString, flag: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    utf8(value, flag)?
        .parse::<T>()
        .map_err(|error| anyhow::anyhow!("{flag} must be a base-10 integer: {error}"))
}

fn parse_tree_state(value: &str) -> Result<SourceTreeState> {
    match value {
        "clean" => Ok(SourceTreeState::Clean),
        "modified" => Ok(SourceTreeState::Modified),
        _ => bail!("--source-tree-state must be 'clean' or 'modified'"),
    }
}

fn main() {
    if let Err(error) = execute() {
        let _ = writeln!(
            std::io::stderr().lock(),
            "OCR execution baseline failed: {error:#}"
        );
        std::process::exit(2);
    }
}

fn execute() -> Result<()> {
    if std::env::args_os()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        print!("{USAGE}");
        return Ok(());
    }
    let arguments = Arguments::parse_from(std::env::args_os().skip(1))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("could not create the benchmark async runtime")?;
    let report = runtime.block_on(run_suite(&arguments))?;
    report.validate()?;
    let bytes = serde_json::to_vec_pretty(&report)
        .context("could not encode the OCR execution-baseline report")?;
    let fixture = arguments.fixture.to_string_lossy();
    if !fixture.is_empty()
        && String::from_utf8_lossy(&bytes)
            .to_ascii_lowercase()
            .contains(&fixture.to_ascii_lowercase())
    {
        bail!("the OCR execution-baseline report leaked the fixture path");
    }
    std::io::stdout()
        .lock()
        .write_all(&bytes)
        .context("could not write the OCR execution-baseline report")?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_arguments() -> Vec<OsString> {
        [
            "--ocr-commit",
            "0123456789abcdef0123456789abcdef01234567",
            "--source-tree-state",
            "clean",
            "--host-label",
            "a3s-lab-ws-01",
            "--cpu-model",
            "Example CPU",
            "--ram-bytes",
            "68719476736",
            "--fixture",
            "fixture.png",
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    #[test]
    fn parses_required_evidence_and_bounded_defaults() {
        let parsed = Arguments::parse_from(valid_arguments()).unwrap();
        assert_eq!(parsed.source_tree_state, SourceTreeState::Clean);
        assert_eq!(parsed.warmup_samples, 1);
        assert_eq!(parsed.samples, 5);
    }

    #[test]
    fn rejects_duplicate_or_ambiguous_evidence() {
        let mut duplicate = valid_arguments();
        duplicate.extend([OsString::from("--samples"), OsString::from("3")]);
        duplicate.extend([OsString::from("--samples"), OsString::from("4")]);
        assert!(Arguments::parse_from(duplicate).is_err());

        let mut invalid_host = valid_arguments();
        let index = invalid_host
            .iter()
            .position(|value| value == "a3s-lab-ws-01")
            .unwrap();
        invalid_host[index] = OsString::from("C:\\host");
        assert!(Arguments::parse_from(invalid_host).is_err());
    }
}
