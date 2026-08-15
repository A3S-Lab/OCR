use std::collections::BTreeSet;

use a3s_use_core::{UseError, UseResult};
use serde::{Deserialize, Serialize};

use crate::{OcrBoundingBox, OcrPoint, OcrStage};

const MAX_CANVAS_AXIS: u32 = 100_000;
const MAX_REGION_POLYGON_POINTS: usize = 64;
const MAX_TABLES_PER_SLOT: usize = 128;
const MAX_CELLS_PER_SLOT: usize = 4_096;
const MAX_GRID_AXIS: u32 = 4_096;
const MAX_GRID_SLOTS_PER_TABLE: usize = 16_384;
const MAX_SEALS_PER_SLOT: usize = 256;
const MAX_EVIDENCE_TEXT_BYTES: usize = 1_048_576;
const MAX_ITEM_TEXT_BYTES: usize = 4_096;

/// Exact source-image canvas used by one structured OCR stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrImageCanvas {
    pub width: u32,
    pub height: u32,
}

impl OcrImageCanvas {
    pub fn new(width: u32, height: u32) -> UseResult<Self> {
        let canvas = Self { width, height };
        canvas.validate()?;
        Ok(canvas)
    }

    fn validate(self) -> UseResult<()> {
        if self.width == 0
            || self.height == 0
            || self.width > MAX_CANVAS_AXIS
            || self.height > MAX_CANVAS_AXIS
        {
            return Err(structured_error(format!(
                "Structured OCR canvases must have positive axes no larger than {MAX_CANVAS_AXIS} pixels."
            )));
        }
        Ok(())
    }
}

/// Stable provider-local identity for one structured evidence object.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct OcrEvidenceId(String);

impl OcrEvidenceId {
    pub fn new(value: impl Into<String>) -> UseResult<Self> {
        let value = Self(value.into());
        value.validate()?;
        Ok(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn validate(&self) -> UseResult<()> {
        let bytes = self.0.as_bytes();
        if bytes.is_empty()
            || bytes.len() > 128
            || !bytes
                .first()
                .is_some_and(|byte| byte.is_ascii_alphanumeric())
            || !bytes.iter().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b':')
            })
        {
            return Err(structured_error(
                "Structured OCR evidence IDs must contain 1 through 128 ASCII letters, digits, dots, hyphens, underscores, or colons and start with a letter or digit.",
            ));
        }
        Ok(())
    }
}

/// Provider-supplied source-pixel geometry for one detected object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrVisualRegion {
    pub bounding_box: OcrBoundingBox,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub polygon: Vec<OcrPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Provider-neutral table appearance class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum OcrTableKind {
    Wired,
    Wireless,
    Unknown,
}

/// One provider-supplied table cell. Geometry is optional when unavailable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrTableCellEvidence {
    pub id: OcrEvidenceId,
    pub row_index: u32,
    pub column_index: u32,
    pub row_span: u32,
    pub column_span: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<OcrVisualRegion>,
}

/// One page-local table and any grid evidence actually supplied by a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrTableEvidence {
    pub id: OcrEvidenceId,
    pub kind: OcrTableKind,
    pub region: OcrVisualRegion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cells: Vec<OcrTableCellEvidence>,
}

/// Complete page-local output of one table stage, including a negative result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrTableStageEvidence {
    pub canvas: OcrImageCanvas,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<OcrTableEvidence>,
}

/// Conservative provider-neutral seal shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum OcrSealKind {
    Circular,
    Rectangular,
    Other,
    Unknown,
}

/// Source-canvas boundary on which a detected seal is visibly clipped.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "kebab-case")]
pub enum OcrCanvasEdge {
    Left,
    Top,
    Right,
    Bottom,
}

/// One page-local seal position and optional provider-supplied recognition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrSealEvidence {
    pub id: OcrEvidenceId,
    pub kind: OcrSealKind,
    pub region: OcrVisualRegion,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clipped_edges: Vec<OcrCanvasEdge>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recognized_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recognition_confidence: Option<f32>,
}

/// Complete page-local output of one seal stage, including a negative result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OcrSealStageEvidence {
    pub canvas: OcrImageCanvas,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seals: Vec<OcrSealEvidence>,
}

/// Typed payload carried by a completed structured OCR stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(
    tag = "evidenceType",
    content = "evidence",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum OcrStageEvidence {
    Table(OcrTableStageEvidence),
    Seal(OcrSealStageEvidence),
}

impl OcrStageEvidence {
    pub fn stage(&self) -> OcrStage {
        match self {
            Self::Table(_) => OcrStage::Table,
            Self::Seal(_) => OcrStage::Seal,
        }
    }

    pub(crate) fn validate(&self) -> UseResult<()> {
        match self {
            Self::Table(evidence) => validate_table_stage(evidence),
            Self::Seal(evidence) => validate_seal_stage(evidence),
        }
    }
}

fn validate_table_stage(evidence: &OcrTableStageEvidence) -> UseResult<()> {
    evidence.canvas.validate()?;
    if evidence.tables.len() > MAX_TABLES_PER_SLOT {
        return Err(structured_error(format!(
            "One OCR table stage must not return more than {MAX_TABLES_PER_SLOT} tables."
        )));
    }
    let mut evidence_ids = BTreeSet::new();
    let mut total_cells = 0_usize;
    let mut total_text_bytes = 0_usize;
    for table in &evidence.tables {
        table.id.validate()?;
        if !evidence_ids.insert(table.id.as_str()) {
            return Err(structured_error(
                "OCR table and cell IDs must be unique within one slot.",
            ));
        }
        validate_region(&table.region, evidence.canvas)?;
        validate_table_grid(
            table,
            evidence.canvas,
            &mut total_cells,
            &mut total_text_bytes,
            &mut evidence_ids,
        )?;
    }
    Ok(())
}

fn validate_table_grid<'a>(
    table: &'a OcrTableEvidence,
    canvas: OcrImageCanvas,
    total_cells: &mut usize,
    total_text_bytes: &mut usize,
    evidence_ids: &mut BTreeSet<&'a str>,
) -> UseResult<()> {
    *total_cells = total_cells
        .checked_add(table.cells.len())
        .ok_or_else(|| structured_error("OCR table cell count overflowed."))?;
    if *total_cells > MAX_CELLS_PER_SLOT {
        return Err(structured_error(format!(
            "One OCR table stage must not return more than {MAX_CELLS_PER_SLOT} cells."
        )));
    }
    let (row_count, column_count) = match (table.row_count, table.column_count) {
        (Some(rows), Some(columns))
            if rows > 0 && columns > 0 && rows <= MAX_GRID_AXIS && columns <= MAX_GRID_AXIS =>
        {
            (Some(rows), Some(columns))
        }
        (None, None) if table.cells.is_empty() => (None, None),
        _ => {
            return Err(structured_error(
                "OCR tables require both positive bounded grid dimensions whenever cell evidence is present.",
            ));
        }
    };
    let mut occupied = BTreeSet::new();
    for cell in &table.cells {
        cell.id.validate()?;
        if !evidence_ids.insert(cell.id.as_str()) {
            return Err(structured_error(
                "OCR table and cell IDs must be unique within one slot.",
            ));
        }
        if cell.row_span == 0 || cell.column_span == 0 {
            return Err(structured_error("OCR table cell spans must be positive."));
        }
        let row_end = cell
            .row_index
            .checked_add(cell.row_span)
            .ok_or_else(|| structured_error("OCR table row span overflowed."))?;
        let column_end = cell
            .column_index
            .checked_add(cell.column_span)
            .ok_or_else(|| structured_error("OCR table column span overflowed."))?;
        if row_end > row_count.unwrap_or(0) || column_end > column_count.unwrap_or(0) {
            return Err(structured_error(
                "OCR table cells must remain inside the declared grid.",
            ));
        }
        for row in cell.row_index..row_end {
            for column in cell.column_index..column_end {
                if occupied.len() >= MAX_GRID_SLOTS_PER_TABLE {
                    return Err(structured_error(format!(
                        "One OCR table must not occupy more than {MAX_GRID_SLOTS_PER_TABLE} grid slots."
                    )));
                }
                if !occupied.insert((row, column)) {
                    return Err(structured_error("OCR table cell spans must not overlap."));
                }
            }
        }
        if let Some(text) = &cell.text {
            validate_text(text, "table cell", total_text_bytes)?;
        }
        if let Some(region) = &cell.region {
            validate_region(region, canvas)?;
            if !contains(table.region.bounding_box, region.bounding_box)? {
                return Err(structured_error(
                    "OCR table cell regions must remain inside their table region.",
                ));
            }
        }
    }
    Ok(())
}

fn validate_seal_stage(evidence: &OcrSealStageEvidence) -> UseResult<()> {
    evidence.canvas.validate()?;
    if evidence.seals.len() > MAX_SEALS_PER_SLOT {
        return Err(structured_error(format!(
            "One OCR seal stage must not return more than {MAX_SEALS_PER_SLOT} seals."
        )));
    }
    let mut ids = BTreeSet::new();
    let mut total_text_bytes = 0_usize;
    for seal in &evidence.seals {
        seal.id.validate()?;
        if !ids.insert(seal.id.as_str()) {
            return Err(structured_error(
                "OCR seal IDs must be unique within one slot.",
            ));
        }
        validate_region(&seal.region, evidence.canvas)?;
        validate_clipped_edges(seal, evidence.canvas)?;
        match (&seal.recognized_text, seal.recognition_confidence) {
            (Some(text), confidence) => {
                validate_text(text, "seal recognition", &mut total_text_bytes)?;
                validate_confidence(confidence, "seal recognition confidence")?;
            }
            (None, None) => {}
            (None, Some(_)) => {
                return Err(structured_error(
                    "OCR seal recognition confidence requires recognized text.",
                ));
            }
        }
    }
    Ok(())
}

fn validate_region(region: &OcrVisualRegion, canvas: OcrImageCanvas) -> UseResult<()> {
    let bounds = region.bounding_box;
    let right = bounds
        .x
        .checked_add(bounds.width)
        .ok_or_else(|| structured_error("Structured OCR region width overflowed."))?;
    let bottom = bounds
        .y
        .checked_add(bounds.height)
        .ok_or_else(|| structured_error("Structured OCR region height overflowed."))?;
    if bounds.width == 0 || bounds.height == 0 || right > canvas.width || bottom > canvas.height {
        return Err(structured_error(
            "Structured OCR regions must cover positive area inside the exact source canvas.",
        ));
    }
    validate_confidence(region.confidence, "region confidence")?;
    if region.polygon.is_empty() {
        return Ok(());
    }
    if !(3..=MAX_REGION_POLYGON_POINTS).contains(&region.polygon.len()) {
        return Err(structured_error(format!(
            "Structured OCR polygons must contain 3 through {MAX_REGION_POLYGON_POINTS} points."
        )));
    }
    let mut left = u32::MAX;
    let mut top = u32::MAX;
    let mut polygon_right = 0_u32;
    let mut polygon_bottom = 0_u32;
    for point in &region.polygon {
        if point.x > canvas.width || point.y > canvas.height {
            return Err(structured_error(
                "Structured OCR polygon points must remain inside the exact source canvas.",
            ));
        }
        left = left.min(point.x);
        top = top.min(point.y);
        polygon_right = polygon_right.max(point.x);
        polygon_bottom = polygon_bottom.max(point.y);
    }
    if left != bounds.x
        || top != bounds.y
        || polygon_right.checked_sub(left) != Some(bounds.width)
        || polygon_bottom.checked_sub(top) != Some(bounds.height)
    {
        return Err(structured_error(
            "A structured OCR bounding box must equal its provider polygon envelope.",
        ));
    }
    Ok(())
}

fn validate_clipped_edges(seal: &OcrSealEvidence, canvas: OcrImageCanvas) -> UseResult<()> {
    let mut sorted = seal.clipped_edges.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted != seal.clipped_edges {
        return Err(structured_error(
            "OCR seal clipped edges must be unique and use canonical order.",
        ));
    }
    let bounds = seal.region.bounding_box;
    let right = bounds.x.checked_add(bounds.width);
    let bottom = bounds.y.checked_add(bounds.height);
    for edge in &seal.clipped_edges {
        let touches = match edge {
            OcrCanvasEdge::Left => bounds.x == 0,
            OcrCanvasEdge::Top => bounds.y == 0,
            OcrCanvasEdge::Right => right == Some(canvas.width),
            OcrCanvasEdge::Bottom => bottom == Some(canvas.height),
        };
        if !touches {
            return Err(structured_error(
                "A declared OCR seal clipped edge must touch that exact canvas boundary.",
            ));
        }
    }
    Ok(())
}

fn validate_confidence(value: Option<f32>, label: &str) -> UseResult<()> {
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(structured_error(format!(
            "Structured OCR {label} must be between 0 and 1."
        )));
    }
    Ok(())
}

fn validate_text(text: &str, label: &str, total: &mut usize) -> UseResult<()> {
    if text.is_empty() || text.len() > MAX_ITEM_TEXT_BYTES || text.chars().any(char::is_control) {
        return Err(structured_error(format!(
            "Structured OCR {label} text must contain 1 through {MAX_ITEM_TEXT_BYTES} control-free UTF-8 bytes."
        )));
    }
    *total = total
        .checked_add(text.len())
        .ok_or_else(|| structured_error("Structured OCR text length overflowed."))?;
    if *total > MAX_EVIDENCE_TEXT_BYTES {
        return Err(structured_error(format!(
            "One structured OCR stage must not return more than {MAX_EVIDENCE_TEXT_BYTES} text bytes."
        )));
    }
    Ok(())
}

fn contains(outer: OcrBoundingBox, inner: OcrBoundingBox) -> UseResult<bool> {
    let outer_right = outer
        .x
        .checked_add(outer.width)
        .ok_or_else(|| structured_error("Outer OCR region width overflowed."))?;
    let outer_bottom = outer
        .y
        .checked_add(outer.height)
        .ok_or_else(|| structured_error("Outer OCR region height overflowed."))?;
    let inner_right = inner
        .x
        .checked_add(inner.width)
        .ok_or_else(|| structured_error("Inner OCR region width overflowed."))?;
    let inner_bottom = inner
        .y
        .checked_add(inner.height)
        .ok_or_else(|| structured_error("Inner OCR region height overflowed."))?;
    Ok(outer.x <= inner.x
        && outer.y <= inner.y
        && outer_right >= inner_right
        && outer_bottom >= inner_bottom)
}

fn structured_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.structured_evidence_invalid", message)
}
