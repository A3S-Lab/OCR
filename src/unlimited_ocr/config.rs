use std::path::{Path, PathBuf};

use a3s_power::inference::{
    DevicePreference, InferenceLimits, ResidencyBudgetPolicy, ResidencyPolicy, WeightStoreConfig,
};
use a3s_use_core::{UseError, UseResult};

use super::UNLIMITED_OCR_MODEL;

const DEFAULT_MAX_GENERATED_TOKENS: usize = 8_192;

/// Typed configuration for one embedded Unlimited-OCR session.
///
/// The model root owns the reviewed checkpoint and tokenizer assets. Device,
/// limits, replicas, and residency are delegated to A3S Power's typed
/// embedded-inference controls; no endpoint or external service is involved.
#[derive(Clone)]
pub struct UnlimitedOcrConfig {
    pub(crate) weight_store: WeightStoreConfig,
    pub(crate) device: DevicePreference,
    pub(crate) limits: InferenceLimits,
    pub(crate) residency: ResidencyPolicy,
    pub(crate) residency_budget: Option<ResidencyBudgetPolicy>,
    pub(crate) max_generated_tokens: usize,
}

impl UnlimitedOcrConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> UseResult<Self> {
        let model_dir = absolute_model_dir(model_dir.into())?;
        let limits = InferenceLimits::default();
        Ok(Self {
            weight_store: WeightStoreConfig::new(model_dir),
            device: DevicePreference::Auto,
            limits,
            residency: ResidencyPolicy::default(),
            residency_budget: None,
            max_generated_tokens: DEFAULT_MAX_GENERATED_TOKENS,
        })
    }

    /// Resolve an explicit local checkpoint from the environment.
    ///
    /// Model download is intentionally not hidden inside provider creation.
    pub fn from_env() -> UseResult<Self> {
        let model_dir = std::env::var_os("A3S_UNLIMITED_OCR_MODEL_DIR")
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                config_error(
                    "A3S_UNLIMITED_OCR_MODEL_DIR must point to a reviewed local baidu/Unlimited-OCR checkpoint.",
                )
            })?;
        Self::new(PathBuf::from(model_dir))
    }

    pub fn with_device(mut self, device: DevicePreference) -> Self {
        self.device = device;
        self
    }

    pub fn with_limits(mut self, limits: InferenceLimits) -> UseResult<Self> {
        limits
            .validate()
            .map_err(|error| config_error(format!("Invalid embedded inference limits: {error}")))?;
        let context_generation_limit = limits.max_context_tokens.saturating_sub(1);
        if self.max_generated_tokens > limits.max_generated_tokens
            || self.max_generated_tokens > context_generation_limit
        {
            return Err(config_error(format!(
                "Unlimited-OCR max generated tokens ({}) exceed the embedded runtime generation/context limits ({}/{}).",
                self.max_generated_tokens,
                limits.max_generated_tokens,
                context_generation_limit,
            )));
        }
        validate_residency(&self.residency, &limits)?;
        self.limits = limits;
        Ok(self)
    }

    pub fn with_residency_policy(mut self, residency: ResidencyPolicy) -> UseResult<Self> {
        residency
            .validate()
            .map_err(|error| config_error(format!("Invalid weight residency policy: {error}")))?;
        if self.residency_budget.is_some() && has_explicit_cache_budget(&residency) {
            return Err(config_error(
                "Manual host/device cache bytes cannot be combined with an automatic residency budget.",
            ));
        }
        validate_residency(&residency, &self.limits)?;
        self.residency = residency;
        Ok(self)
    }

    /// Opt into Power's native hardware-aware cache budget planning.
    ///
    /// Manual host/device cache bytes are mutually exclusive with this policy.
    /// Other residency controls, including eviction, prefetch, and telemetry,
    /// remain owned by [`ResidencyPolicy`] and are retained when the derived
    /// byte budget is applied during lazy session loading.
    pub fn with_residency_budget_policy(
        mut self,
        policy: ResidencyBudgetPolicy,
    ) -> UseResult<Self> {
        policy.validate().map_err(|error| {
            config_error(format!(
                "Invalid automatic residency budget policy: {error}"
            ))
        })?;
        if has_explicit_cache_budget(&self.residency) {
            return Err(config_error(
                "An automatic residency budget cannot replace explicit host/device cache bytes.",
            ));
        }
        self.residency_budget = Some(policy);
        Ok(self)
    }

    /// Replace the primary checkpoint and optional byte-identical replicas.
    ///
    /// A3S Power validates replica count, complete byte identity, deterministic
    /// source selection, and fallback behavior when the model is opened.
    pub fn with_weight_store_config(mut self, config: WeightStoreConfig) -> UseResult<Self> {
        if config.primary.root.as_os_str().is_empty() {
            return Err(config_error(
                "Unlimited-OCR primary weight root must not be empty.",
            ));
        }
        self.weight_store = config;
        Ok(self)
    }

    pub fn with_max_generated_tokens(mut self, max_generated_tokens: usize) -> UseResult<Self> {
        let maximum = self
            .limits
            .max_generated_tokens
            .min(self.limits.max_context_tokens.saturating_sub(1));
        if max_generated_tokens == 0 || max_generated_tokens > maximum {
            return Err(config_error(format!(
                "Unlimited-OCR max generated tokens must be between 1 and {}.",
                maximum
            )));
        }
        self.max_generated_tokens = max_generated_tokens;
        Ok(self)
    }

    pub fn model_dir(&self) -> &Path {
        &self.weight_store.primary.root
    }

    pub fn model(&self) -> &'static str {
        UNLIMITED_OCR_MODEL
    }

    pub fn max_generated_tokens(&self) -> usize {
        self.max_generated_tokens
    }

    pub fn residency_budget_policy(&self) -> Option<&ResidencyBudgetPolicy> {
        self.residency_budget.as_ref()
    }

    pub fn sends_source_off_device(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for UnlimitedOcrConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnlimitedOcrConfig")
            .field("weight_store", &self.weight_store)
            .field("device", &self.device)
            .field("limits", &self.limits)
            .field("residency", &self.residency)
            .field("residency_budget", &self.residency_budget)
            .field("max_generated_tokens", &self.max_generated_tokens)
            .finish()
    }
}

fn absolute_model_dir(path: PathBuf) -> UseResult<PathBuf> {
    if path.as_os_str().is_empty() {
        return Err(config_error(
            "Unlimited-OCR model directory must not be empty.",
        ));
    }
    if path.is_absolute() {
        return Ok(path);
    }
    std::env::current_dir()
        .map(|directory| directory.join(path))
        .map_err(|error| {
            config_error(format!(
                "Failed to resolve the Unlimited-OCR model directory: {error}"
            ))
        })
}

fn validate_residency(policy: &ResidencyPolicy, limits: &InferenceLimits) -> UseResult<()> {
    let resident_bytes = policy
        .host_cache_bytes
        .checked_add(policy.device_cache_bytes)
        .ok_or_else(|| config_error("Unlimited-OCR residency byte budget overflowed."))?;
    if resident_bytes > limits.max_resident_weight_bytes {
        return Err(config_error(format!(
            "Unlimited-OCR residency requires {resident_bytes} bytes, exceeding the {} byte embedded runtime limit.",
            limits.max_resident_weight_bytes
        )));
    }
    Ok(())
}

fn has_explicit_cache_budget(policy: &ResidencyPolicy) -> bool {
    policy.host_cache_bytes > 0 || policy.device_cache_bytes > 0
}

fn config_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.unlimited_ocr_config_invalid", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_is_local_and_bounded() {
        let config = UnlimitedOcrConfig::new("fixture-model")
            .unwrap()
            .with_max_generated_tokens(512)
            .unwrap();
        assert!(config.model_dir().is_absolute());
        assert_eq!(config.model(), UNLIMITED_OCR_MODEL);
        assert_eq!(config.max_generated_tokens(), 512);
        assert!(!config.sends_source_off_device());
        assert!(UnlimitedOcrConfig::new("").is_err());
        assert!(config.clone().with_max_generated_tokens(0).is_err());
    }

    #[test]
    fn residency_cannot_escape_the_shared_runtime_limit() {
        let config = UnlimitedOcrConfig::new("fixture-model").unwrap();
        let policy = ResidencyPolicy {
            host_cache_bytes: config.limits.max_resident_weight_bytes,
            device_cache_bytes: 1,
            ..ResidencyPolicy::default()
        };
        assert!(config.with_residency_policy(policy).is_err());
    }

    #[test]
    fn automatic_and_manual_cache_budgets_are_mutually_exclusive() {
        let automatic = ResidencyBudgetPolicy::new(5_000, 0).unwrap();
        let manual = ResidencyPolicy {
            host_cache_bytes: 1,
            ..ResidencyPolicy::default()
        };

        assert!(UnlimitedOcrConfig::new("fixture-model")
            .unwrap()
            .with_residency_policy(manual.clone())
            .unwrap()
            .with_residency_budget_policy(automatic.clone())
            .is_err());
        assert!(UnlimitedOcrConfig::new("fixture-model")
            .unwrap()
            .with_residency_budget_policy(automatic)
            .unwrap()
            .with_residency_policy(manual)
            .is_err());
    }

    #[test]
    fn automatic_residency_is_explicitly_opt_in() {
        let config = UnlimitedOcrConfig::new("fixture-model").unwrap();
        assert!(config.residency_budget_policy().is_none());
        assert_eq!(config.residency.host_cache_bytes, 0);
        assert_eq!(config.residency.device_cache_bytes, 0);

        let automatic = ResidencyBudgetPolicy::new(5_000, 0).unwrap();
        let config = config
            .with_residency_budget_policy(automatic.clone())
            .unwrap();
        assert_eq!(config.residency_budget_policy(), Some(&automatic));
    }
}
