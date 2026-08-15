use std::path::Path;

use a3s_use_core::{UseError, UseResult};

const VOCABULARY_SIZE: usize = 50;

#[derive(Debug, Clone)]
pub(super) struct StructureDictionary {
    tokens: Vec<String>,
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
        let mut base = text
            .split('\n')
            .filter(|token| !token.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        if !base.iter().any(|token| token == "<td></td>") {
            base.push("<td></td>".to_string());
        }
        base.retain(|token| token != "<td>");
        let mut tokens = Vec::with_capacity(base.len() + 2);
        tokens.push("sos".to_string());
        tokens.extend(base);
        tokens.push("eos".to_string());
        let dictionary = Self { tokens };
        if dictionary.tokens.len() != VOCABULARY_SIZE {
            return Err(dictionary_error(format!(
                "The SLANet-Plus structure dictionary must realize exactly {VOCABULARY_SIZE} tokens, found {}.",
                dictionary.tokens.len()
            )));
        }
        Ok(dictionary)
    }

    pub(super) fn token(&self, index: usize) -> UseResult<&str> {
        self.tokens.get(index).map(String::as_str).ok_or_else(|| {
            dictionary_error("A decoded SLANet-Plus token escaped the pinned vocabulary.")
        })
    }

    pub(super) fn sos(&self) -> usize {
        0
    }

    pub(super) fn eos(&self) -> usize {
        self.tokens.len() - 1
    }

    pub(super) fn is_cell(&self, index: usize) -> bool {
        self.tokens
            .get(index)
            .is_some_and(|token| matches!(token.as_str(), "<td" | "<td>" | "<td></td>"))
    }
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
        assert_eq!(dictionary.token(dictionary.sos()).unwrap(), "sos");
        assert_eq!(dictionary.token(dictionary.eos()).unwrap(), "eos");
        assert!(dictionary.tokens.iter().any(|token| token == "<td></td>"));
        assert!(!dictionary.tokens.iter().any(|token| token == "<td>"));
    }
}
