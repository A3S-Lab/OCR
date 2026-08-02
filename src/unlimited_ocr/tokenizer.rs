use std::ops::Range;
use std::path::Path;

use a3s_use_core::{UseError, UseResult};
use tokenizers::Tokenizer;

use super::preprocess::PreprocessedImage;

pub(crate) const BOS_TOKEN_ID: u32 = 0;
pub(crate) const EOS_TOKEN_ID: u32 = 1;
pub(crate) const IMAGE_TOKEN_ID: u32 = 128_815;
pub(crate) const PROMPT: &str = "<image>document parsing.";

#[derive(Debug, Clone)]
pub(crate) struct PromptEncoding {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) image_tokens: Range<usize>,
}

pub(crate) struct UnlimitedTokenizer {
    inner: Tokenizer,
}

impl UnlimitedTokenizer {
    pub(crate) fn load(path: &Path) -> UseResult<Self> {
        let inner = Tokenizer::from_file(path).map_err(|error| {
            tokenizer_error(format!(
                "Failed to load the reviewed Unlimited-OCR tokenizer '{}': {error}",
                path.display()
            ))
        })?;
        if inner.token_to_id("<image>") != Some(IMAGE_TOKEN_ID) {
            return Err(tokenizer_error(format!(
                "Unlimited-OCR tokenizer must map <image> to token {IMAGE_TOKEN_ID}."
            )));
        }
        Ok(Self { inner })
    }

    pub(crate) fn encode_prompt(&self, image: &PreprocessedImage) -> UseResult<PromptEncoding> {
        let (prefix, suffix) = PROMPT.split_once("<image>").ok_or_else(|| {
            tokenizer_error("The reviewed Unlimited-OCR prompt has no image placeholder.")
        })?;
        if suffix.contains("<image>") {
            return Err(tokenizer_error(
                "The reviewed Unlimited-OCR prompt contains more than one image placeholder.",
            ));
        }
        let mut token_ids = vec![BOS_TOKEN_ID];
        token_ids.extend(self.encode_text(prefix)?);
        let image_start = token_ids.len();
        token_ids.extend(image.image_token_ids(IMAGE_TOKEN_ID));
        let image_end = token_ids.len();
        token_ids.extend(self.encode_text(suffix)?);
        Ok(PromptEncoding {
            token_ids,
            image_tokens: image_start..image_end,
        })
    }

    pub(crate) fn decode(&self, token_ids: &[u32]) -> UseResult<String> {
        let token_ids = without_terminal_eos(token_ids);
        self.inner
            .decode(token_ids, false)
            .map(|text| text.trim().to_string())
            .map_err(|error| {
                tokenizer_error(format!(
                    "Failed to decode Unlimited-OCR output tokens: {error}"
                ))
            })
    }

    fn encode_text(&self, text: &str) -> UseResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| {
                tokenizer_error(format!(
                    "Failed to encode the Unlimited-OCR prompt: {error}"
                ))
            })
    }
}

fn without_terminal_eos(token_ids: &[u32]) -> &[u32] {
    token_ids.strip_suffix(&[EOS_TOKEN_ID]).unwrap_or(token_ids)
}

fn tokenizer_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.tokenizer_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_eos_is_removed_before_visible_text_decoding() {
        assert_eq!(without_terminal_eos(&[7, 8, EOS_TOKEN_ID]), [7, 8]);
        assert_eq!(without_terminal_eos(&[7, EOS_TOKEN_ID, 8]), [7, 1, 8]);
        assert!(without_terminal_eos(&[EOS_TOKEN_ID]).is_empty());
    }
}
