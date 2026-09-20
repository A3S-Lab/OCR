use std::collections::BTreeSet;

use a3s_use_core::UseResult;
use serde::{Deserialize, Serialize};

use crate::output_validation::validate_category_label;

use super::{structured_error, validate_region, OcrEvidenceId, OcrImageCanvas, OcrVisualRegion};

const MAX_LAYOUT_REGIONS_PER_SLOT: usize = 512;
const MAX_TEXT_BLOCK_REFERENCES_PER_REGION: usize = 10_000;

/// Conservative provider-neutral meaning of one page-layout region.
///
/// The exact provider class remains in `raw_label`; this role carries only
/// semantics that can be projected without inspecting text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum OcrLayoutRole {
    Text,
    Title,
    Heading,
    Paragraph,
    Table,
    Caption,
    EquationBlock,
    Figure,
    Chart,
    Header,
    Footer,
    Footnote,
    PageNumber,
    CodeBlock,
    Note,
    Seal,
    HeaderFigure,
    FooterFigure,
    Unknown,
}

/// One provider-supplied page-layout region and its geometry-established text
/// membership. Text block indices address the Text-stage output from the same
/// slot and never arise from text matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrLayoutRegionEvidence {
    pub id: OcrEvidenceId,
    pub raw_label: String,
    pub role: OcrLayoutRole,
    pub region: OcrVisualRegion,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_text_block_indices: Vec<u32>,
}

/// Complete page-local output of one layout stage, including a negative
/// result. Every region is bound to this exact immutable source canvas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrLayoutStageEvidence {
    pub canvas: OcrImageCanvas,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<OcrLayoutRegionEvidence>,
}

pub(super) fn validate_layout_stage(evidence: &OcrLayoutStageEvidence) -> UseResult<()> {
    evidence.canvas.validate()?;
    if evidence.regions.len() > MAX_LAYOUT_REGIONS_PER_SLOT {
        return Err(structured_error(format!(
            "One OCR layout stage must not return more than {MAX_LAYOUT_REGIONS_PER_SLOT} regions."
        )));
    }
    let mut ids = BTreeSet::new();
    for region in &evidence.regions {
        region.id.validate()?;
        if !ids.insert(region.id.as_str()) {
            return Err(structured_error(
                "OCR layout-region IDs must be unique within one slot.",
            ));
        }
        validate_category_label(&region.raw_label)?;
        validate_region(&region.region, evidence.canvas)?;
        if region.source_text_block_indices.len() > MAX_TEXT_BLOCK_REFERENCES_PER_REGION
            || region
                .source_text_block_indices
                .windows(2)
                .any(|indices| indices[0] >= indices[1])
        {
            return Err(structured_error(
                "OCR layout-region text-block references must be bounded, unique, and canonical.",
            ));
        }
    }
    Ok(())
}
