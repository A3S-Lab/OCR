use std::path::Path;
use std::sync::Arc;

use a3s_use_core::{Artifact, UseError, UseResult};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;

use crate::models::{OcrDiagnostic, OcrRequest, OcrResult};
use crate::provider::{OcrInput, OcrProvider, OcrProviderDescriptor, OcrProviderOutput};

const MAX_INPUT_BYTES: u64 = 32 * 1024 * 1024;

/// Provider-neutral OCR client.
///
/// The client owns source validation and evidence. Recognition is delegated to
/// one injected [`OcrProvider`].
#[derive(Clone)]
pub struct OcrClient {
    provider: Arc<dyn OcrProvider>,
    descriptor: OcrProviderDescriptor,
}

impl OcrClient {
    pub fn with_provider<P>(provider: P) -> UseResult<Self>
    where
        P: OcrProvider + 'static,
    {
        Self::from_provider(Arc::new(provider))
    }

    pub fn from_provider(provider: Arc<dyn OcrProvider>) -> UseResult<Self> {
        let descriptor = provider.descriptor();
        descriptor.validate()?;
        Ok(Self {
            provider,
            descriptor,
        })
    }

    #[cfg(feature = "ppocr-v6")]
    pub fn from_env() -> UseResult<Self> {
        Self::with_provider(crate::PpOcrV6Provider::from_env()?)
    }

    pub fn provider(&self) -> &OcrProviderDescriptor {
        &self.descriptor
    }

    pub fn diagnostic(&self) -> OcrDiagnostic {
        let status = self.provider.diagnostic();
        OcrDiagnostic {
            readiness: status.readiness,
            provider: Some(self.descriptor.id.clone()),
            engine: Some(self.descriptor.engine.clone()),
            model: status.model,
            model_dir: status.model_dir,
            sends_source_off_device: self.descriptor.sends_source_off_device,
            message: status.message,
            suggestions: status.suggestions,
        }
    }

    pub async fn extract(&self, request: OcrRequest) -> UseResult<OcrResult> {
        let input = read_source(&request.path).await?;
        let source = input.source().clone();
        let output = self.provider.recognize(input).await?;
        validate_provider_output(&output)?;
        Ok(OcrResult {
            provider: self.descriptor.id.clone(),
            engine: self.descriptor.engine.clone(),
            model: output.model,
            source,
            text: output.text,
            blocks: output.blocks,
            warnings: output.warnings,
        })
    }
}

fn validate_provider_output(output: &OcrProviderOutput) -> UseResult<()> {
    for block in &output.blocks {
        if block.page == 0 {
            return Err(provider_output_error(
                "OCR providers must number pages starting at 1.",
            ));
        }
        for (name, value) in [
            ("confidence", block.confidence),
            ("detection confidence", block.detection_confidence),
        ] {
            if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
                return Err(provider_output_error(format!(
                    "OCR provider {name} must be between 0 and 1."
                )));
            }
        }
    }
    Ok(())
}

fn provider_output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_output_invalid", message)
}

async fn read_source(path: &Path) -> UseResult<OcrInput> {
    let canonical = tokio::fs::canonicalize(path).await.map_err(|error| {
        UseError::new(
            "use.ocr.source_unreadable",
            format!("Failed to resolve OCR source '{}': {error}", path.display()),
        )
    })?;
    let metadata = tokio::fs::metadata(&canonical).await.map_err(|error| {
        UseError::new(
            "use.ocr.source_unreadable",
            format!(
                "Failed to inspect OCR source '{}': {error}",
                canonical.display()
            ),
        )
    })?;
    if !metadata.is_file() {
        return Err(UseError::new(
            "use.ocr.source_invalid",
            format!(
                "OCR source '{}' is not a regular file.",
                canonical.display()
            ),
        ));
    }
    if metadata.len() == 0 || metadata.len() > MAX_INPUT_BYTES {
        return Err(UseError::new(
            "use.ocr.source_too_large",
            format!(
                "OCR source '{}' must contain between 1 byte and 32 MiB.",
                canonical.display()
            ),
        )
        .with_detail("size", metadata.len()));
    }
    let file = tokio::fs::File::open(&canonical).await.map_err(|error| {
        UseError::new(
            "use.ocr.source_unreadable",
            format!(
                "Failed to open OCR source '{}': {error}",
                canonical.display()
            ),
        )
    })?;
    let mut bytes = Vec::with_capacity(metadata.len().min(MAX_INPUT_BYTES) as usize);
    file.take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|error| {
            UseError::new(
                "use.ocr.source_unreadable",
                format!(
                    "Failed to read OCR source '{}': {error}",
                    canonical.display()
                ),
            )
        })?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err(UseError::new(
            "use.ocr.source_too_large",
            format!(
                "OCR source '{}' must not exceed 32 MiB.",
                canonical.display()
            ),
        )
        .with_detail("sizeAtLeast", MAX_INPUT_BYTES + 1));
    }
    let media_type = detect_image_type(&bytes).ok_or_else(|| {
        UseError::new(
            "use.ocr.source_type_unsupported",
            "OCR accepts PNG, JPEG, WebP, GIF, BMP, and TIFF image bytes.",
        )
    })?;
    let digest = Sha256::digest(&bytes);
    Ok(OcrInput::new(
        Artifact {
            path: canonical,
            media_type: media_type.to_string(),
            size: bytes.len() as u64,
            sha256: format!("{digest:x}"),
        },
        bytes,
    ))
}

fn detect_image_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        Some("image/tiff")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use a3s_use_core::Readiness;
    use async_trait::async_trait;

    use crate::{OcrBlock, OcrProviderStatus};

    use super::*;

    struct FixtureProvider;

    #[async_trait]
    impl OcrProvider for FixtureProvider {
        fn descriptor(&self) -> OcrProviderDescriptor {
            OcrProviderDescriptor::new("fixture-tesseract", "tesseract", true).unwrap()
        }

        fn diagnostic(&self) -> OcrProviderStatus {
            OcrProviderStatus {
                readiness: Readiness::Ready,
                model: Some("eng".to_string()),
                model_dir: None,
                message: "ready".to_string(),
                suggestions: Vec::new(),
            }
        }

        async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
            assert_eq!(input.source().media_type, "image/bmp");
            assert!(input.bytes().starts_with(b"BM"));
            Ok(OcrProviderOutput {
                model: Some("eng".to_string()),
                text: "fixture text".to_string(),
                blocks: vec![OcrBlock {
                    page: 1,
                    text: "fixture text".to_string(),
                    confidence: None,
                    detection_confidence: None,
                    polygon: None,
                    bounding_box: None,
                }],
                warnings: Vec::new(),
            })
        }
    }

    #[test]
    fn detects_supported_image_signatures() {
        assert_eq!(
            detect_image_type(b"\x89PNG\r\n\x1a\nrest"),
            Some("image/png")
        );
        assert_eq!(detect_image_type(b"\xff\xd8\xffrest"), Some("image/jpeg"));
        assert_eq!(detect_image_type(b"not an image"), None);
    }

    #[tokio::test]
    async fn injected_provider_receives_validated_source_and_owns_recognition() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("scan.bmp");
        std::fs::write(&source, base64_free_white_bmp()).unwrap();
        let client = OcrClient::with_provider(FixtureProvider).unwrap();

        let result = client.extract(OcrRequest { path: source }).await.unwrap();

        assert_eq!(result.provider, "fixture-tesseract");
        assert_eq!(result.engine, "tesseract");
        assert_eq!(result.model.as_deref(), Some("eng"));
        assert_eq!(result.text, "fixture text");
        assert_eq!(result.blocks[0].confidence, None);
        assert!(client.diagnostic().sends_source_off_device);
    }

    fn base64_free_white_bmp() -> Vec<u8> {
        vec![
            0x42, 0x4d, 0x3a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
            0x28, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00,
        ]
    }
}
