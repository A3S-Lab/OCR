mod assets;
mod config;
mod grounding;
mod model;
mod ngram;
mod preprocess;
mod tokenizer;

use std::sync::{Arc, Mutex};

use a3s_power::inference::{
    EmbeddedRuntime, ExecutionDigest, ModelIdentity, ResidencyPolicy, WeightHierarchy, WeightStore,
};
use a3s_use_core::{Readiness, UseError, UseResult};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use self::assets::{inspect_assets, verify_weight_store, MODEL_REVISION};
pub use self::config::UnlimitedOcrConfig;
use self::grounding::{parse_model_output, source_grounding_geometry};
use self::model::{power_error, shared, Decoder, VisionEncoder};
use self::preprocess::preprocess;
use self::tokenizer::UnlimitedTokenizer;
use crate::cancellation::run_blocking;
use crate::provider::{
    OcrInput, OcrProvider, OcrProviderDescriptor, OcrProviderOutput, OcrProviderStatus,
};
use crate::receipt::project_receipt;

pub const UNLIMITED_OCR_PROVIDER_ID: &str = "unlimited-ocr";
pub const UNLIMITED_OCR_MODEL: &str = "baidu/Unlimited-OCR";

const ENGINE_NAME: &str = "a3s-power-native";
const STOP_TOKEN: &str = "<｜end▁of▁sentence｜>";

/// Embedded native-Rust Unlimited-OCR provider.
///
/// Provider creation performs no download and opens no listener. The reviewed
/// local checkpoint is loaded lazily on the first recognition request through
/// A3S Power's model-neutral embedded runtime.
#[derive(Clone)]
pub struct UnlimitedOcrProvider {
    descriptor: OcrProviderDescriptor,
    config: UnlimitedOcrConfig,
    loaded: Arc<Mutex<Option<Arc<UnlimitedOcrSession>>>>,
}

impl UnlimitedOcrProvider {
    pub fn new(config: UnlimitedOcrConfig) -> UseResult<Self> {
        Ok(Self {
            descriptor: OcrProviderDescriptor::new(UNLIMITED_OCR_PROVIDER_ID, ENGINE_NAME, false)?,
            config,
            loaded: Arc::new(Mutex::new(None)),
        })
    }

    pub fn from_env() -> UseResult<Self> {
        Self::new(UnlimitedOcrConfig::from_env()?)
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
        match inspect_assets(self.config.model_dir()) {
            Ok(assets) => OcrProviderStatus {
                readiness: Readiness::Ready,
                model: Some(UNLIMITED_OCR_MODEL.to_string()),
                model_dir: Some(assets.root),
                message: format!(
                    "Reviewed Unlimited-OCR revision {MODEL_REVISION} is ready for embedded a3s-power inference."
                ),
                suggestions: Vec::new(),
            },
            Err(error) => OcrProviderStatus {
                readiness: if self.config.model_dir().exists() {
                    Readiness::Broken
                } else {
                    Readiness::Missing
                },
                model: Some(UNLIMITED_OCR_MODEL.to_string()),
                model_dir: Some(self.config.model_dir().to_path_buf()),
                message: error.to_string(),
                suggestions: vec![
                    "Restore the exact reviewed baidu/Unlimited-OCR checkpoint and set A3S_UNLIMITED_OCR_MODEL_DIR to its local directory."
                        .to_string(),
                ],
            },
        }
    }

    async fn recognize(&self, input: OcrInput) -> UseResult<OcrProviderOutput> {
        let loaded = Arc::clone(&self.loaded);
        let config = self.config.clone();
        run_blocking("embedded Unlimited-OCR inference", move |cancellation| {
            let session = {
                let mut loaded = loaded.lock().map_err(|_| {
                    UseError::new(
                        "use.ocr.runtime_failed",
                        "The embedded Unlimited-OCR session lock is poisoned.",
                    )
                })?;
                if loaded.is_none() {
                    *loaded = Some(Arc::new(UnlimitedOcrSession::load(&config)?));
                }
                loaded.as_ref().cloned().ok_or_else(|| {
                    UseError::new(
                        "use.ocr.runtime_failed",
                        "The embedded Unlimited-OCR session failed to initialize.",
                    )
                })?
            };
            session.recognize(input.bytes(), &cancellation)
        })
        .await
    }
}

struct UnlimitedOcrSession {
    runtime: EmbeddedRuntime,
    identity: ModelIdentity,
    limits: a3s_power::inference::InferenceLimits,
    max_generated_tokens: usize,
    weights: Arc<model::ModelWeights>,
    vision: VisionEncoder,
    decoder: Decoder,
    tokenizer: UnlimitedTokenizer,
}

impl UnlimitedOcrSession {
    fn load(config: &UnlimitedOcrConfig) -> UseResult<Self> {
        let assets = inspect_assets(config.model_dir())?;
        let runtime = EmbeddedRuntime::new(config.device, config.limits.clone())
            .map_err(|error| power_error("initialize the embedded runtime", error))?;
        let residency = resolve_residency(config, &runtime)?;
        let store = Arc::new(
            WeightStore::open_config(&config.weight_store, &config.limits)
                .map_err(|error| power_error("open the reviewed Unlimited-OCR weights", error))?,
        );
        verify_weight_store(&store, &assets)?;
        let identity = ModelIdentity::new(
            UNLIMITED_OCR_MODEL,
            MODEL_REVISION,
            store.sha256().to_string(),
        );
        let hierarchy = WeightHierarchy::new(store, runtime.clone(), residency)
            .map_err(|error| power_error("initialize the shared weight hierarchy", error))?;
        let weights = shared(hierarchy);
        let vision = VisionEncoder::new(Arc::clone(&weights));
        let decoder = Decoder::new(Arc::clone(&weights));
        let tokenizer = UnlimitedTokenizer::load(&assets.tokenizer)?;
        Ok(Self {
            runtime,
            identity,
            limits: config.limits.clone(),
            max_generated_tokens: config.max_generated_tokens,
            weights,
            vision,
            decoder,
            tokenizer,
        })
    }

    fn recognize(
        &self,
        bytes: &[u8],
        cancellation: &CancellationToken,
    ) -> UseResult<OcrProviderOutput> {
        let permit = self
            .runtime
            .begin(cancellation)
            .map_err(|error| power_error("admit the Unlimited-OCR request", error))?;
        self.weights.refresh_residency(&permit, cancellation)?;
        let geometry = source_grounding_geometry(bytes)?;
        let image = preprocess(bytes, &self.limits, cancellation)?;
        let prompt = self.tokenizer.encode_prompt(&image)?;
        let total_tokens = prompt
            .token_ids
            .len()
            .checked_add(self.max_generated_tokens)
            .ok_or_else(|| generation_error("Unlimited-OCR context length overflowed."))?;
        if total_tokens > self.limits.max_context_tokens {
            return Err(generation_error(format!(
                "Unlimited-OCR requires up to {total_tokens} context tokens, exceeding the {} token Power limit.",
                self.limits.max_context_tokens
            )));
        }
        let vision = self.vision.encode(&image, &permit, cancellation)?;
        let generated = self.decoder.generate(
            &prompt,
            &vision,
            self.max_generated_tokens,
            &permit,
            cancellation,
        )?;
        let raw = self.tokenizer.decode(&generated)?;
        let parsed = parse_model_output(&raw, geometry)?;
        let receipt = self.runtime.receipt(
            self.identity.clone(),
            ExecutionDigest::image_request(bytes, 1),
            ExecutionDigest::utf8_text(&parsed.text),
        );
        Ok(OcrProviderOutput {
            model: Some(UNLIMITED_OCR_MODEL.to_string()),
            text: parsed.text,
            blocks: parsed.blocks,
            execution_receipts: vec![project_receipt(receipt)],
            warnings: parsed.warnings,
        })
    }
}

fn resolve_residency(
    config: &UnlimitedOcrConfig,
    runtime: &EmbeddedRuntime,
) -> UseResult<ResidencyPolicy> {
    let Some(policy) = &config.residency_budget else {
        return Ok(config.residency.clone());
    };
    let plan = runtime
        .plan_residency_budget(policy)
        .map_err(|error| power_error("plan the hardware-aware residency budget", error))?;
    plan.apply_to(&config.residency)
        .map_err(|error| power_error("apply the hardware-aware residency budget", error))
}

fn provider_output_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.provider_output_invalid", message)
}

fn generation_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.generation_failed", message)
}

#[cfg(test)]
mod numerical_tests;
#[cfg(test)]
mod tests;
