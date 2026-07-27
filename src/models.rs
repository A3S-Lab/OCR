use std::path::PathBuf;

use a3s_use_core::{Artifact, Readiness};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OcrRequest {
    #[schemars(description = "Local PNG, JPEG, WebP, GIF, BMP, or TIFF image path")]
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OcrPoint {
    pub x: u32,
    pub y: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OcrBoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OcrBlock {
    pub page: u32,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Provider recognition confidence from 0 through 1, when available")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "Provider text-detection confidence from 0 through 1, when available"
    )]
    pub detection_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Four polygon vertices in source-image coordinates, when available")]
    pub polygon: Option<[OcrPoint; 4]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounding_box: Option<OcrBoundingBox>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OcrResult {
    pub provider: String,
    pub engine: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[schemars(with = "OcrArtifactSchema")]
    pub source: Artifact,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<OcrBlock>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct OcrDiagnostic {
    #[schemars(with = "OcrReadinessSchema")]
    pub readiness: Readiness,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_dir: Option<PathBuf>,
    pub sends_source_off_device: bool,
    pub message: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<String>,
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
struct OcrArtifactSchema {
    path: PathBuf,
    media_type: String,
    size: u64,
    sha256: String,
}

#[derive(schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[allow(dead_code)]
enum OcrReadinessSchema {
    Ready,
    Missing,
    Broken,
    Unknown,
}
