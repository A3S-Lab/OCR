use std::fmt;

use a3s_use_core::UseError;

/// Stable product disposition for DocumentFast provider construction.
///
/// The enum keeps callers independent of the serialized `UseError::code`
/// representation. It classifies only provider construction; document
/// admission, evidence, and inference failures retain their existing typed
/// contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFastInitializationErrorKind {
    RequiredModelMissing,
    ConfigurationInvalid,
}

/// Typed DocumentFast construction failure with the original OCR error.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentFastInitializationError {
    kind: DocumentFastInitializationErrorKind,
    source: UseError,
}

impl DocumentFastInitializationError {
    pub fn kind(&self) -> DocumentFastInitializationErrorKind {
        self.kind
    }

    pub fn as_use_error(&self) -> &UseError {
        &self.source
    }

    pub fn into_use_error(self) -> UseError {
        self.source
    }

    pub(super) fn required_model_missing(source: UseError) -> Self {
        Self {
            kind: DocumentFastInitializationErrorKind::RequiredModelMissing,
            source,
        }
    }

    pub(super) fn configuration_invalid(source: UseError) -> Self {
        Self {
            kind: DocumentFastInitializationErrorKind::ConfigurationInvalid,
            source,
        }
    }
}

impl fmt::Display for DocumentFastInitializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(formatter)
    }
}

impl std::error::Error for DocumentFastInitializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_kind_is_independent_of_the_legacy_machine_code() {
        let missing = DocumentFastInitializationError::required_model_missing(UseError::new(
            "legacy.missing.wire",
            "missing",
        ));
        assert_eq!(
            missing.kind(),
            DocumentFastInitializationErrorKind::RequiredModelMissing
        );
        assert_eq!(missing.as_use_error().code, "legacy.missing.wire");

        let invalid = DocumentFastInitializationError::configuration_invalid(UseError::new(
            "legacy.invalid.wire",
            "invalid",
        ));
        assert_eq!(
            invalid.kind(),
            DocumentFastInitializationErrorKind::ConfigurationInvalid
        );
        assert_eq!(invalid.into_use_error().code, "legacy.invalid.wire");
    }
}
