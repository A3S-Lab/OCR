use std::path::Path;

use a3s_use_core::{UseError, UseResult};

const VOCABULARY_SIZE: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::document_fast) enum StructureToken {
    StartOfSequence,
    EndOfSequence,
    TableHeadStart,
    TableHeadEnd,
    TableBodyStart,
    TableBodyEnd,
    RowStart,
    RowEnd,
    CellStart,
    CellDelimiter,
    CellEnd,
    EmptyCell,
    ColumnSpan(u32),
    RowSpan(u32),
}

impl StructureToken {
    pub(super) fn carries_cell_geometry(self) -> bool {
        matches!(self, Self::CellStart | Self::EmptyCell)
    }
}

#[derive(Debug, Clone)]
pub(super) struct StructureDictionary {
    tokens: Vec<StructureToken>,
}

impl StructureDictionary {
    pub(super) fn load(path: &Path) -> UseResult<Self> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            dictionary_error(format!(
                "Failed to read the SLANet-Plus structure dictionary '{}': {error}",
                path.display()
            ))
        })?;
        Self::from_text(&text)
    }

    pub(super) fn from_text(text: &str) -> UseResult<Self> {
        if text.contains('\r') {
            return Err(dictionary_error(
                "The pinned SLANet-Plus dictionary must use LF line endings.",
            ));
        }
        let base = text
            .split('\n')
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();
        let mut tokens = Vec::with_capacity(base.len() + 2);
        tokens.push(StructureToken::StartOfSequence);
        for token in base {
            if token == "<td>" {
                continue;
            }
            tokens.push(parse_token(token)?);
        }
        tokens.push(StructureToken::EmptyCell);
        tokens.push(StructureToken::EndOfSequence);
        let dictionary = Self { tokens };
        let reviewed = reviewed_tokens();
        if dictionary.tokens.len() != VOCABULARY_SIZE || dictionary.tokens != reviewed {
            return Err(dictionary_error(format!(
                "The SLANet-Plus structure dictionary must realize the exact reviewed {VOCABULARY_SIZE}-token index map, found {} tokens or a different order.",
                dictionary.tokens.len()
            )));
        }
        Ok(dictionary)
    }

    pub(super) fn token(&self, index: usize) -> UseResult<StructureToken> {
        self.tokens.get(index).copied().ok_or_else(|| {
            dictionary_error("A decoded SLANet-Plus token escaped the pinned vocabulary.")
        })
    }

    pub(super) fn sos(&self) -> usize {
        0
    }

    pub(super) fn eos(&self) -> usize {
        self.tokens.len() - 1
    }
}

fn parse_token(token: &str) -> UseResult<StructureToken> {
    let structural = match token {
        "<thead>" => StructureToken::TableHeadStart,
        "</thead>" => StructureToken::TableHeadEnd,
        "<tbody>" => StructureToken::TableBodyStart,
        "</tbody>" => StructureToken::TableBodyEnd,
        "<tr>" => StructureToken::RowStart,
        "</tr>" => StructureToken::RowEnd,
        "<td" => StructureToken::CellStart,
        ">" => StructureToken::CellDelimiter,
        "</td>" => StructureToken::CellEnd,
        _ => {
            if let Some(span) = parse_span(token, " colspan=\"")? {
                StructureToken::ColumnSpan(span)
            } else if let Some(span) = parse_span(token, " rowspan=\"")? {
                StructureToken::RowSpan(span)
            } else {
                return Err(dictionary_error(
                    "The pinned SLANet-Plus dictionary contains an unsupported structure token.",
                ));
            }
        }
    };
    Ok(structural)
}

fn parse_span(token: &str, prefix: &str) -> UseResult<Option<u32>> {
    let Some(value) = token
        .strip_prefix(prefix)
        .and_then(|value| value.strip_suffix('"'))
    else {
        return Ok(None);
    };
    let span = value.parse::<u32>().map_err(|_| {
        dictionary_error("The pinned SLANet-Plus dictionary contains a non-numeric table span.")
    })?;
    if !(2..=20).contains(&span) {
        return Err(dictionary_error(
            "The pinned SLANet-Plus dictionary table span is outside the reviewed range.",
        ));
    }
    Ok(Some(span))
}

fn reviewed_tokens() -> Vec<StructureToken> {
    let mut tokens = vec![
        StructureToken::StartOfSequence,
        StructureToken::TableHeadStart,
        StructureToken::TableHeadEnd,
        StructureToken::TableBodyStart,
        StructureToken::TableBodyEnd,
        StructureToken::RowStart,
        StructureToken::RowEnd,
        StructureToken::CellStart,
        StructureToken::CellDelimiter,
        StructureToken::CellEnd,
    ];
    tokens.extend((2..=20).map(StructureToken::ColumnSpan));
    tokens.extend((2..=20).map(StructureToken::RowSpan));
    tokens.push(StructureToken::EmptyCell);
    tokens.push(StructureToken::EndOfSequence);
    tokens
}

fn dictionary_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.table_model_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_no_span_vocabulary_has_exact_model_indices() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/slanet_plus_dictionary.txt"
        ));
        let dictionary = StructureDictionary::from_text(source).unwrap();
        assert_eq!(
            dictionary.token(dictionary.sos()).unwrap(),
            StructureToken::StartOfSequence
        );
        assert_eq!(
            dictionary.token(dictionary.eos()).unwrap(),
            StructureToken::EndOfSequence
        );
        assert!(dictionary.tokens.contains(&StructureToken::EmptyCell));
        assert_eq!(
            dictionary.token(dictionary.eos() - 1).unwrap(),
            StructureToken::EmptyCell
        );
    }

    #[test]
    fn unsupported_dictionary_tokens_fail_before_inference() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/slanet_plus_dictionary.txt"
        ));
        let source = source.replacen("<tbody>", "<unknown>", 1);

        assert_eq!(
            StructureDictionary::from_text(&source).unwrap_err().code,
            "use.ocr.table_model_invalid"
        );
    }

    #[test]
    fn reordered_dictionary_tokens_fail_before_inference() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/slanet_plus_dictionary.txt"
        ));
        let source = source.replacen("<thead>\n</thead>", "</thead>\n<thead>", 1);

        assert_eq!(
            StructureDictionary::from_text(&source).unwrap_err().code,
            "use.ocr.table_model_invalid"
        );
    }
}
