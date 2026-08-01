use std::io::Cursor;

use a3s_use_core::UseResult;
use image::{metadata::Orientation, ImageDecoder, ImageReader};

mod coordinates;

use crate::OcrBlock;

use super::{provider_output_error, STOP_TOKEN};
use coordinates::{parse_coordinates, CoordinateParseError, NormalizedBox};

const REF_OPEN: &str = "<|ref|>";
const REF_CLOSE: &str = "<|/ref|>";
const DET_OPEN: &str = "<|det|>";
const DET_CLOSE: &str = "<|/det|>";
const MAX_GROUNDING_LABEL_BYTES: usize = 128;
const MAX_GROUNDING_TOKEN_SPANS: usize = 20_000;
const MAX_COORDINATE_BOXES: usize = 40_000;
const INVALID_GROUNDING_WARNING: &str =
    "Unlimited-OCR output did not provide grounding that could be represented as bounded source-image text blocks.";

pub(super) const MAX_GROUNDING_BLOCK_TEXT_BYTES: usize = 1024 * 1024;
pub(super) const MAX_GROUNDING_MARKERS: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GroundingGeometry {
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Debug, PartialEq)]
pub(super) struct ParsedModelOutput {
    pub text: String,
    pub blocks: Vec<OcrBlock>,
    pub warnings: Vec<String>,
}

pub(super) fn source_grounding_geometry(bytes: &[u8]) -> UseResult<Option<GroundingGeometry>> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| {
            provider_output_error(
                "Unlimited-OCR requires an image with bounded, decodable pixel dimensions.",
            )
        })?;
    let mut decoder = reader.into_decoder().map_err(|_| {
        provider_output_error(
            "Unlimited-OCR requires an image with bounded, decodable pixel dimensions.",
        )
    })?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 {
        return Err(provider_output_error(
            "Unlimited-OCR source-image dimensions must be positive.",
        ));
    }
    // The reviewed upstream loader applies EXIF orientation before it computes
    // grounding. OcrBlock coordinates are defined against the untransformed
    // source image, so transformed inputs must not produce misleading boxes.
    let orientation = decoder.orientation();
    if !matches!(orientation, Ok(Orientation::NoTransforms)) {
        return Ok(None);
    }
    Ok(Some(GroundingGeometry { width, height }))
}

pub(super) fn parse_model_output(
    raw: &str,
    geometry: Option<GroundingGeometry>,
) -> UseResult<ParsedModelOutput> {
    let scanned = scan_grounding(raw)?;
    let mut malformed = scanned.malformed || scanned.markers.is_empty();
    let mut blocks = Vec::with_capacity(scanned.markers.len());
    for (index, marker) in scanned.markers.iter().enumerate() {
        let end = scanned
            .markers
            .get(index + 1)
            .map(|next| next.text_start)
            .unwrap_or(scanned.text.len());
        let Some(segment) = scanned.text.get(marker.text_start..end) else {
            return Err(provider_output_error(
                "Unlimited-OCR grounding text boundaries are invalid.",
            ));
        };
        let text = normalize_text(segment);
        let Some(label) = marker.label.as_deref() else {
            malformed = true;
            continue;
        };
        if label.eq_ignore_ascii_case("image") {
            malformed = true;
            continue;
        }
        let Some(bounds) = marker.bounds else {
            malformed = true;
            continue;
        };
        if text.is_empty() {
            malformed = true;
            continue;
        }
        if text.len() > MAX_GROUNDING_BLOCK_TEXT_BYTES {
            malformed = true;
            continue;
        }
        let Some(geometry) = geometry else {
            malformed = true;
            continue;
        };
        let Some(bounding_box) = bounds.to_source_pixels(geometry.width, geometry.height) else {
            malformed = true;
            continue;
        };
        blocks.push(OcrBlock {
            page: 1,
            text,
            confidence: None,
            detection_confidence: None,
            polygon: None,
            bounding_box: Some(bounding_box),
        });
    }

    Ok(ParsedModelOutput {
        text: normalize_text(&scanned.text),
        blocks,
        warnings: malformed
            .then(|| INVALID_GROUNDING_WARNING.to_string())
            .into_iter()
            .collect(),
    })
}

struct ScannedGrounding {
    text: String,
    markers: Vec<GroundingMarker>,
    malformed: bool,
}

struct GroundingMarker {
    text_start: usize,
    label: Option<String>,
    bounds: Option<NormalizedBox>,
}

fn scan_grounding(raw: &str) -> UseResult<ScannedGrounding> {
    let mut text = String::with_capacity(raw.len());
    let mut markers = Vec::new();
    let mut cursor = 0_usize;
    let mut token_spans = 0_usize;
    let mut coordinate_boxes = 0_usize;
    let mut malformed = false;

    while cursor < raw.len() {
        let Some((position, kind)) = next_marker(raw, cursor) else {
            text.push_str(&raw[cursor..]);
            break;
        };
        text.push_str(&raw[cursor..position]);
        match kind {
            MarkerKind::Reference => {
                let content_start = position + REF_OPEN.len();
                let Some(close_offset) = raw[content_start..].find(REF_CLOSE) else {
                    text.push_str(&raw[position..]);
                    malformed = true;
                    break;
                };
                bump_token_span(&mut token_spans)?;
                let content_end = content_start + close_offset;
                let reference_end = content_end + REF_CLOSE.len();
                if raw[reference_end..].starts_with(DET_OPEN) {
                    let detection_start = reference_end;
                    let payload_start = detection_start + DET_OPEN.len();
                    let Some(detection_close_offset) = raw[payload_start..].find(DET_CLOSE) else {
                        text.push_str(&raw[position..]);
                        malformed = true;
                        break;
                    };
                    bump_token_span(&mut token_spans)?;
                    bump_grounding_marker(&markers)?;
                    let payload_end = payload_start + detection_close_offset;
                    let (marker, invalid) = build_marker(
                        text.len(),
                        &raw[content_start..content_end],
                        &raw[payload_start..payload_end],
                        &mut coordinate_boxes,
                    )?;
                    markers.push(marker);
                    malformed |= invalid;
                    cursor = payload_end + DET_CLOSE.len();
                } else {
                    malformed = true;
                    cursor = reference_end;
                }
            }
            MarkerKind::Detection => {
                let payload_start = position + DET_OPEN.len();
                let Some(close_offset) = raw[payload_start..].find(DET_CLOSE) else {
                    text.push_str(&raw[position..]);
                    malformed = true;
                    break;
                };
                bump_token_span(&mut token_spans)?;
                bump_grounding_marker(&markers)?;
                let payload_end = payload_start + close_offset;
                let payload = &raw[payload_start..payload_end];
                let (label, coordinates) = split_direct_payload(payload);
                let (marker, invalid) = build_marker(
                    text.len(),
                    label.unwrap_or_default(),
                    coordinates.unwrap_or_default(),
                    &mut coordinate_boxes,
                )?;
                markers.push(marker);
                malformed |= invalid || label.is_none() || coordinates.is_none();
                cursor = payload_end + DET_CLOSE.len();
            }
        }
    }

    Ok(ScannedGrounding {
        text,
        markers,
        malformed,
    })
}

fn bump_token_span(token_spans: &mut usize) -> UseResult<()> {
    *token_spans = token_spans
        .checked_add(1)
        .ok_or_else(grounding_limit_error)?;
    if *token_spans > MAX_GROUNDING_TOKEN_SPANS {
        return Err(grounding_limit_error());
    }
    Ok(())
}

fn bump_grounding_marker(markers: &[GroundingMarker]) -> UseResult<()> {
    if markers.len() >= MAX_GROUNDING_MARKERS {
        return Err(grounding_limit_error());
    }
    Ok(())
}

fn build_marker(
    text_start: usize,
    raw_label: &str,
    raw_coordinates: &str,
    coordinate_boxes: &mut usize,
) -> UseResult<(GroundingMarker, bool)> {
    let label = parse_label(raw_label);
    let coordinates = parse_bounded_coordinates(raw_coordinates)?;
    if let Some(coordinates) = coordinates {
        *coordinate_boxes = coordinate_boxes
            .checked_add(coordinates.box_count)
            .ok_or_else(grounding_limit_error)?;
        if *coordinate_boxes > MAX_COORDINATE_BOXES {
            return Err(grounding_limit_error());
        }
        let invalid = label.is_none();
        Ok((
            GroundingMarker {
                text_start,
                label,
                bounds: Some(coordinates.bounds),
            },
            invalid,
        ))
    } else {
        Ok((
            GroundingMarker {
                text_start,
                label,
                bounds: None,
            },
            true,
        ))
    }
}

fn split_direct_payload(payload: &str) -> (Option<&str>, Option<&str>) {
    let payload = payload.trim();
    let Some(coordinates_start) = payload.find('[') else {
        return (None, None);
    };
    let label = payload[..coordinates_start].trim();
    let coordinates = payload[coordinates_start..].trim();
    ((!label.is_empty()).then_some(label), Some(coordinates))
}

fn parse_label(raw: &str) -> Option<String> {
    let label = raw.trim();
    if label.is_empty()
        || label.len() > MAX_GROUNDING_LABEL_BYTES
        || !label
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return None;
    }
    Some(label.to_string())
}

fn parse_bounded_coordinates(raw: &str) -> UseResult<Option<coordinates::ParsedCoordinates>> {
    match parse_coordinates(raw) {
        Ok(parsed) => Ok(Some(parsed)),
        Err(CoordinateParseError::Invalid) => Ok(None),
        Err(CoordinateParseError::Limit) => Err(grounding_limit_error()),
    }
}

#[derive(Clone, Copy)]
enum MarkerKind {
    Reference,
    Detection,
}

fn next_marker(raw: &str, cursor: usize) -> Option<(usize, MarkerKind)> {
    let remaining = raw.get(cursor..)?;
    let reference = remaining
        .find(REF_OPEN)
        .map(|offset| (cursor + offset, MarkerKind::Reference));
    let detection = remaining
        .find(DET_OPEN)
        .map(|offset| (cursor + offset, MarkerKind::Detection));
    match (reference, detection) {
        (Some(reference), Some(detection)) => Some(if reference.0 <= detection.0 {
            reference
        } else {
            detection
        }),
        (Some(reference), None) => Some(reference),
        (None, Some(detection)) => Some(detection),
        (None, None) => None,
    }
}

fn normalize_text(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    while text.ends_with(STOP_TOKEN) {
        text.truncate(text.len() - STOP_TOKEN.len());
        text = text.trim_end().to_string();
    }
    text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
}

fn grounding_limit_error() -> a3s_use_core::UseError {
    provider_output_error("Unlimited-OCR grounding exceeds its bounded marker or coordinate limit.")
}
