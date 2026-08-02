use std::sync::Arc;

use a3s_power::inference::{
    ExecutionPermit, PlacementPreference, PrefetchTask, ResidencyCandidate, ResidencyPolicy,
    RoutedExpertBatch, TelemetryMode, WeightHierarchy, WeightKey, WeightRequest,
};
use a3s_use_core::{UseError, UseResult};
use candle_core::Tensor;
use tokio_util::sync::CancellationToken;

use super::{DECODER_LAYERS, ROUTED_EXPERTS};

const GLOBAL_LAYER: u32 = 100;
pub(crate) const SAM_LAYER_BASE: u32 = 200;
pub(crate) const CLIP_LAYER_BASE: u32 = 300;
pub(crate) const PROJECTOR_LAYER: u32 = 400;

#[derive(Clone)]
pub(crate) struct ModelWeights {
    hierarchy: WeightHierarchy,
}

impl ModelWeights {
    pub(crate) fn new(hierarchy: WeightHierarchy) -> Self {
        Self { hierarchy }
    }

    pub(crate) fn hierarchy(&self) -> &WeightHierarchy {
        &self.hierarchy
    }

    pub(crate) fn load_global(
        &self,
        name: &str,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        self.load(GLOBAL_LAYER, name, permit, cancellation)
    }

    pub(crate) fn load(
        &self,
        layer: u32,
        name: &str,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Tensor> {
        let descriptor = self.hierarchy.store().descriptor(name).ok_or_else(|| {
            model_error(format!("Unlimited-OCR reviewed tensor '{name}' is absent."))
        })?;
        self.checked_elements(&descriptor.shape, &format!("Unlimited-OCR weight '{name}'"))?;
        self.hierarchy
            .load(
                // Model operations execute on Power's selected tensor device.
                // An explicit Device request still resolves to Host on CPU,
                // while avoiding a host-only result on accelerators whose
                // device cache budget is intentionally zero.
                &WeightRequest::new(WeightKey::new(layer, name), PlacementPreference::Device),
                permit,
                cancellation,
            )
            .map(|weight| weight.into_tensor())
            .map_err(|error| power_error("load a reviewed model tensor", error))
    }

    pub(crate) fn record_routes(&self, routes: &RoutedExpertBatch) {
        self.hierarchy.record_routes(routes);
    }

    pub(crate) fn checked_elements(&self, shape: &[usize], label: &str) -> UseResult<usize> {
        self.hierarchy
            .runtime()
            .limits()
            .checked_elements(shape, label)
            .map_err(|error| power_error("validate model tensor bounds", error))
    }

    pub(crate) fn prefetch_experts(
        &self,
        routes: &RoutedExpertBatch,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<Option<PrefetchTask>> {
        let policy = self.hierarchy.policy();
        if !prefetch_can_be_reused(policy) {
            return Ok(None);
        }
        let requests = routes
            .experts()
            .iter()
            .flat_map(|expert| expert_keys(routes.layer(), *expert))
            .map(|key| WeightRequest::new(key, PlacementPreference::Device))
            .collect::<Vec<_>>();
        if requests.len() > policy.max_prefetch_items {
            return Ok(None);
        }
        let bytes = requests.iter().try_fold(0_u64, |total, request| {
            let descriptor = self
                .hierarchy
                .store()
                .descriptor(&request.key.name)
                .ok_or_else(|| {
                    model_error(format!(
                        "Unlimited-OCR expert tensor '{}' is absent.",
                        request.key.name
                    ))
                })?;
            total
                .checked_add(descriptor.bytes)
                .ok_or_else(|| model_error("Unlimited-OCR expert prefetch bytes overflowed."))
        })?;
        if bytes > policy.max_prefetch_bytes {
            return Ok(None);
        }
        self.hierarchy
            .start_prefetch(requests, permit, cancellation.clone())
            .map(Some)
            .map_err(|error| power_error("start bounded expert prefetch", error))
    }

    pub(crate) fn wait_prefetch(task: Option<PrefetchTask>) -> UseResult<()> {
        let Some(task) = task else {
            return Ok(());
        };
        let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
            model_error(format!(
                "Unlimited-OCR expert prefetch has no active Tokio runtime: {error}"
            ))
        })?;
        runtime
            .block_on(task.wait())
            .map(|_| ())
            .map_err(|error| power_error("complete bounded expert prefetch", error))
    }

    /// Reconcile the plan-owned expert hot set at a request boundary.
    pub(crate) fn refresh_residency(
        &self,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> UseResult<()> {
        if self.hierarchy.policy().telemetry != TelemetryMode::Detailed {
            return Ok(());
        }
        let history = self
            .hierarchy
            .routing_history()
            .map_err(|error| power_error("read private routing heat", error))?;
        if history.entries.is_empty() {
            return Ok(());
        }
        let candidates = history
            .entries
            .into_iter()
            .filter(|entry| {
                (entry.key.layer as usize) < DECODER_LAYERS
                    && (entry.key.expert as usize) < ROUTED_EXPERTS
                    && entry.selections > 0
            })
            .map(|entry| {
                ResidencyCandidate::new(
                    format!("decoder.{}.expert.{}", entry.key.layer, entry.key.expert),
                    entry.selections,
                    expert_keys(entry.key.layer, entry.key.expert),
                )
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(());
        }
        let plan = self
            .hierarchy
            .plan_residency(&candidates)
            .map_err(|error| power_error("plan the atomic expert hot set", error))?;
        self.hierarchy
            .apply_residency_plan(&plan, permit, cancellation)
            .map(|_| ())
            .map_err(|error| power_error("apply the atomic expert hot set", error))
    }
}

fn prefetch_can_be_reused(policy: &ResidencyPolicy) -> bool {
    policy.host_cache_bytes > 0 || policy.device_cache_bytes > 0
}

pub(crate) fn expert_name(layer: u32, expert: u32, projection: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight")
}

fn expert_keys(layer: u32, expert: u32) -> Vec<WeightKey> {
    ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .map(|projection| WeightKey::new(layer, expert_name(layer, expert, projection)))
        .collect()
}

pub(crate) fn power_error(action: &str, error: impl std::fmt::Display) -> UseError {
    UseError::new(
        "use.ocr.runtime_failed",
        format!("Failed to {action} through a3s-power: {error}"),
    )
}

pub(crate) fn model_error(message: impl Into<String>) -> UseError {
    UseError::new("use.ocr.model_invalid", message)
}

pub(crate) fn shared(weights: WeightHierarchy) -> Arc<ModelWeights> {
    Arc::new(ModelWeights::new(weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_prefetch_requires_a_residency_budget() {
        let mut policy = ResidencyPolicy::default();
        assert!(!prefetch_can_be_reused(&policy));

        policy.host_cache_bytes = 1;
        assert!(prefetch_can_be_reused(&policy));

        policy.host_cache_bytes = 0;
        policy.device_cache_bytes = 1;
        assert!(prefetch_can_be_reused(&policy));
    }
}
