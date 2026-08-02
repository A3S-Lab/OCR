mod grounding;

use std::fmt;
use std::net::IpAddr;
use std::time::Duration;

use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::{StatusCode, Url};
use serde::Deserialize;

#[cfg(test)]
use grounding::MAX_GROUNDING_MARKERS;
use grounding::{parse_model_output, source_grounding_geometry};

use crate::provider::{
    OcrInput, OcrProvider, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
};

pub const UNLIMITED_OCR_PROVIDER_ID: &str = "unlimited-ocr";
pub const UNLIMITED_OCR_MODEL: &str = "baidu/Unlimited-OCR";

const ENGINE_NAME: &str = "vllm-openai";
const DEFAULT_MAX_TOKENS: u32 = 8_192;
const MAX_MAX_TOKENS: u32 = 32_768;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(1_200);
const MAX_TIMEOUT: Duration = Duration::from_secs(3_600);
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const PROMPT: &str = "<image>document parsing.";
const STOP_TOKEN: &str = "<｜end▁of▁sentence｜>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndpointScope {
    Local,
    Remote,
}

/// Typed configuration for one externally managed Unlimited-OCR vLLM server.
///
/// Use [`Self::local`] for a loopback server. [`Self::remote`] requires HTTPS
/// and records that source image bytes leave the device.
#[derive(Clone)]
pub struct UnlimitedOcrConfig {
    base_url: Url,
    scope: EndpointScope,
    model: String,
    bearer_token: Option<String>,
    max_tokens: u32,
    timeout: Duration,
}

impl UnlimitedOcrConfig {
    pub fn local(base_url: impl AsRef<str>) -> UseResult<Self> {
        let base_url = parse_base_url(base_url.as_ref())?;
        if !is_loopback(&base_url) {
            return Err(config_error(
                "Local Unlimited-OCR endpoints must use localhost or a loopback IP address. Use UnlimitedOcrConfig::remote for other hosts.",
            ));
        }
        Ok(Self::new(base_url, EndpointScope::Local))
    }

    pub fn remote(base_url: impl AsRef<str>) -> UseResult<Self> {
        let base_url = parse_base_url(base_url.as_ref())?;
        if base_url.scheme() != "https" {
            return Err(config_error(
                "Remote Unlimited-OCR endpoints must use HTTPS because OCR source bytes are transmitted to the server.",
            ));
        }
        Ok(Self::new(base_url, EndpointScope::Remote))
    }

    fn new(base_url: Url, scope: EndpointScope) -> Self {
        Self {
            base_url,
            scope,
            model: UNLIMITED_OCR_MODEL.to_string(),
            bearer_token: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Set the model name exposed by the vLLM server.
    ///
    /// The default is `baidu/Unlimited-OCR`. This is useful when the server was
    /// started with a different `--served-model-name`.
    pub fn with_model(mut self, model: impl Into<String>) -> UseResult<Self> {
        let model = model.into();
        if model.trim().is_empty() || model.len() > 256 {
            return Err(config_error(
                "Unlimited-OCR served model names must contain 1 through 256 characters.",
            ));
        }
        self.model = model;
        Ok(self)
    }

    /// Add a bearer token without exposing it through diagnostics or `Debug`.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> UseResult<Self> {
        let token = token.into();
        bearer_header(&token)?;
        self.bearer_token = Some(token);
        Ok(self)
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> UseResult<Self> {
        if !(1..=MAX_MAX_TOKENS).contains(&max_tokens) {
            return Err(config_error(format!(
                "Unlimited-OCR max_tokens must be between 1 and {MAX_MAX_TOKENS}."
            )));
        }
        self.max_tokens = max_tokens;
        Ok(self)
    }

    pub fn with_timeout(mut self, timeout: Duration) -> UseResult<Self> {
        if timeout.is_zero() || timeout > MAX_TIMEOUT {
            return Err(config_error(
                "Unlimited-OCR request timeouts must be greater than zero and no longer than 3600 seconds.",
            ));
        }
        self.timeout = timeout;
        Ok(self)
    }

    pub fn base_url(&self) -> &str {
        self.base_url.as_str()
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn sends_source_off_device(&self) -> bool {
        self.scope == EndpointScope::Remote
    }
}

impl fmt::Debug for UnlimitedOcrConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("UnlimitedOcrConfig")
            .field("base_url", &self.base_url)
            .field("scope", &self.scope)
            .field("model", &self.model)
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("max_tokens", &self.max_tokens)
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Unlimited-OCR provider backed by an externally managed vLLM server.
///
/// The provider uses the official OpenAI-compatible serving contract. It does
/// not start Python, Docker, vLLM, or a model download on behalf of the caller.
#[derive(Clone)]
pub struct UnlimitedOcrProvider {
    descriptor: OcrProviderDescriptor,
    config: UnlimitedOcrConfig,
    completions_url: Url,
    client: reqwest::Client,
}

impl UnlimitedOcrProvider {
    pub fn new(config: UnlimitedOcrConfig) -> UseResult<Self> {
        let descriptor = OcrProviderDescriptor::new(
            UNLIMITED_OCR_PROVIDER_ID,
            ENGINE_NAME,
            config.sends_source_off_device(),
        )?;
        let completions_url = config.base_url.join("chat/completions").map_err(|error| {
            config_error(format!(
                "Failed to resolve the Unlimited-OCR chat completions endpoint: {error}"
            ))
        })?;
        let mut headers = HeaderMap::new();
        if let Some(token) = &config.bearer_token {
            headers.insert(AUTHORIZATION, bearer_header(token)?);
        }
        let mut client_builder = reqwest::Client::builder()
            .default_headers(headers)
            .redirect(reqwest::redirect::Policy::none())
            .timeout(config.timeout)
            .user_agent(concat!("a3s-use-ocr/", env!("CARGO_PKG_VERSION")));
        if config.scope == EndpointScope::Local {
            client_builder = client_builder.no_proxy();
        }
        let client = client_builder.build().map_err(|error| {
            config_error(format!(
                "Failed to build the Unlimited-OCR HTTP client: {error}"
            ))
        })?;
        Ok(Self {
            descriptor,
            config,
            completions_url,
            client,
        })
    }

    pub fn config(&self) -> &UnlimitedOcrConfig {
        &self.config
    }
}

#[async_trait]
impl OcrProvider for UnlimitedOcrProvider {
    fn descriptor(&self) -> OcrProviderDescriptor {
        self.descriptor.clone()
    }

    fn diagnostic(&self) -> OcrProviderStatus {
        OcrProviderStatus {
            readiness: Readiness::Unknown,
            model: Some(self.config.model.clone()),
            model_dir: None,
            message: format!(
                "Unlimited-OCR is configured at '{}'; endpoint reachability is checked during extraction.",
                self.config.base_url
            ),
            suggestions: vec![
                "Start the official baidu/Unlimited-OCR vLLM server and verify its OpenAI-compatible /v1 endpoint."
                    .to_string(),
            ],
        }
    }

    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
        let source_geometry = source_grounding_geometry(input.bytes())?;
        let image_url = format!(
            "data:{};base64,{}",
            input.source().media_type,
            BASE64_STANDARD.encode(input.bytes())
        );
        let payload = serde_json::json!({
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": PROMPT },
                    { "type": "image_url", "image_url": { "url": image_url } }
                ]
            }],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.0,
            "skip_special_tokens": false,
            "vllm_xargs": {
                "ngram_size": 35,
                "window_size": 128
            }
        });
        let response = self
            .client
            .post(self.completions_url.clone())
            .json(&payload)
            .send()
            .await
            .map_err(|error| request_error(&self.completions_url, error))?;
        let status = response.status();
        let body = read_bounded_response(response).await?;
        if !status.is_success() {
            return Err(http_status_error(status, &body));
        }
        let completion: ChatCompletionResponse =
            serde_json::from_slice(&body).map_err(|error| {
                provider_output_error(format!(
                    "Unlimited-OCR returned an invalid chat completion response: {error}"
                ))
            })?;
        let raw = completion
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .and_then(ChatMessageContent::into_text)
            .ok_or_else(|| {
                provider_output_error(
                    "Unlimited-OCR returned no textual content in its first completion choice.",
                )
            })?;
        let parsed = parse_model_output(&raw, source_geometry)?;
        Ok(OcrProviderOutput {
            model: Some(self.config.model.clone()),
            text: parsed.text,
            blocks: parsed.blocks,
            execution_receipts: Vec::new(),
            warnings: parsed.warnings,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: Option<ChatMessageContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatMessagePart>),
}

impl ChatMessageContent {
    fn into_text(self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(parts) => {
                let text = parts
                    .into_iter()
                    .filter_map(|part| part.text)
                    .collect::<Vec<_>>()
                    .join("");
                (!text.is_empty()).then_some(text)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatMessagePart {
    #[serde(default)]
    text: Option<String>,
}

fn parse_base_url(value: &str) -> UseResult<Url> {
    if value.len() > 2_048 {
        return Err(config_error(
            "Unlimited-OCR endpoint URLs must not exceed 2048 characters.",
        ));
    }
    let mut url = Url::parse(value)
        .map_err(|error| config_error(format!("Invalid Unlimited-OCR endpoint URL: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(config_error(
            "Unlimited-OCR endpoint URLs must be absolute HTTP or HTTPS URLs.",
        ));
    }
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(config_error(
            "Unlimited-OCR endpoint URLs must not contain credentials, a query, or a fragment.",
        ));
    }
    if url.path() == "/" {
        url.set_path("/v1/");
    } else if !url.path().ends_with('/') {
        let normalized = format!("{}/", url.path());
        url.set_path(&normalized);
    }
    Ok(url)
}

fn is_loopback(url: &Url) -> bool {
    url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    })
}

fn bearer_header(token: &str) -> UseResult<HeaderValue> {
    if token.trim().is_empty() || token.len() > 8_192 {
        return Err(config_error(
            "Unlimited-OCR bearer tokens must contain 1 through 8192 characters.",
        ));
    }
    HeaderValue::from_str(&format!("Bearer {token}"))
        .map_err(|_| config_error("Unlimited-OCR bearer tokens must be valid HTTP header values."))
}

async fn read_bounded_response(mut response: reqwest::Response) -> UseResult<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err(provider_output_error(format!(
            "Unlimited-OCR response exceeds the {MAX_RESPONSE_BYTES}-byte limit."
        )));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| {
        provider_output_error(format!(
            "Failed to read the Unlimited-OCR response body: {error}"
        ))
    })? {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(provider_output_error(format!(
                "Unlimited-OCR response exceeds the {MAX_RESPONSE_BYTES}-byte limit."
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn request_error(endpoint: &Url, error: reqwest::Error) -> UseError {
    UseError::new(
        "use.ocr.provider_unavailable",
        format!(
            "Failed to call the Unlimited-OCR endpoint '{}': {error}",
            endpoint
        ),
    )
    .with_suggestion("Verify that the baidu/Unlimited-OCR vLLM server is running and reachable.")
}

fn http_status_error(status: StatusCode, body: &[u8]) -> UseError {
    let message = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "The server did not return a structured error message.".to_string());
    UseError::new(
        "use.ocr.provider_request_failed",
        format!("Unlimited-OCR returned HTTP {}: {message}", status.as_u16()),
    )
    .with_detail("status", u64::from(status.as_u16()))
}

fn config_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.unlimited_ocr_config_invalid", message)
}

fn provider_output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_output_invalid", message)
}

#[cfg(test)]
mod tests;
