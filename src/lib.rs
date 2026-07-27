//! Provider-oriented optical character recognition for A3S.
//!
//! [`OcrClient`] owns bounded source validation and evidence while recognition
//! is delegated to an injected [`OcrProvider`]. The optional PP-OCRv6 feature
//! supplies the local default used by A3S Use without making that model the
//! library's architectural boundary.

#[cfg(feature = "ppocr-v6")]
mod assets;
#[cfg(feature = "cli")]
pub mod cli;
mod client;
#[cfg(feature = "ppocr-v6")]
mod config;
#[cfg(feature = "ppocr-v6")]
mod engine;
#[cfg(feature = "ppocr-v6")]
mod install;
#[cfg(feature = "mcp")]
pub mod mcp;
mod models;
#[cfg(feature = "ppocr-v6")]
mod postprocess;
#[cfg(feature = "ppocr-v6")]
mod ppocr_v6;
#[cfg(feature = "ppocr-v6")]
mod preprocess;
mod provider;

#[cfg(feature = "ppocr-v6")]
pub use assets::{ocr_status, OcrInstallSource, OcrRuntimeStatus};
pub use client::OcrClient;
#[cfg(feature = "ppocr-v6")]
pub use install::{install_ppocr_v6, repair_ppocr_v6, uninstall_managed_ppocr_v6};
#[cfg(feature = "mcp")]
pub use mcp::OcrMcpServer;
pub use models::{OcrBlock, OcrBoundingBox, OcrDiagnostic, OcrPoint, OcrRequest, OcrResult};
#[cfg(feature = "ppocr-v6")]
pub use ppocr_v6::{PpOcrV6Provider, PP_OCR_V6_PROVIDER_ID};
pub use provider::{
    OcrInput, OcrProvider, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
};

pub use a3s_use_core::{Artifact, Readiness, UseError, UseResult};

/// Source-tree Skill root used by development hosts.
///
/// Installed products should use their release-packaged Skill directory. This
/// path is intentionally only a fallback for source builds and tests.
pub fn source_skill_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("skills")
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_skill_is_owned_by_this_repository() {
        assert!(super::source_skill_root()
            .join("a3s-use-ocr/SKILL.md")
            .is_file());
    }
}
