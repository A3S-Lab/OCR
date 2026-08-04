use a3s_use_core::{UseError, UseResult};
use tokio_util::sync::CancellationToken;

/// Run bounded synchronous inference without detaching it from the caller's
/// async lifetime.
///
/// Tokio cannot abort a `spawn_blocking` task once it has started. This guard
/// therefore cancels the one request token when the awaiting future is dropped;
/// native stages observe that token at their bounded cancellation points.
#[cfg(any(feature = "unlimited-ocr", test))]
pub(crate) async fn run_blocking<T, F>(label: &'static str, operation: F) -> UseResult<T>
where
    T: Send + 'static,
    F: FnOnce(CancellationToken) -> UseResult<T> + Send + 'static,
{
    let guard = CancellationScope::new();
    let result = run_blocking_with(label, guard.token(), operation).await;
    guard.disarm();
    result
}

pub(crate) async fn run_blocking_with<T, F>(
    label: &'static str,
    cancellation: CancellationToken,
    operation: F,
) -> UseResult<T>
where
    T: Send + 'static,
    F: FnOnce(CancellationToken) -> UseResult<T> + Send + 'static,
{
    let worker_cancellation = cancellation;
    let result = tokio::task::spawn_blocking(move || operation(worker_cancellation))
        .await
        .map_err(|error| {
            UseError::new(
                "use.ocr.runtime_failed",
                format!("The {label} blocking task failed: {error}"),
            )
        })?;
    result
}

pub(crate) fn check_cancelled(cancellation: &CancellationToken) -> UseResult<()> {
    if cancellation.is_cancelled() {
        Err(UseError::new(
            "use.ocr.runtime_failed",
            "OCR inference was cancelled.",
        ))
    } else {
        Ok(())
    }
}

pub(crate) struct CancellationScope {
    cancellation: CancellationToken,
    armed: bool,
}

impl CancellationScope {
    pub(crate) fn new() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            armed: true,
        }
    }

    pub(crate) fn token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub(crate) fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for CancellationScope {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::Notify;

    use super::*;

    #[tokio::test]
    async fn dropping_the_async_waiter_cancels_the_blocking_worker() {
        let started = Arc::new(Notify::new());
        let observed = Arc::new(Notify::new());
        let worker_started = Arc::clone(&started);
        let worker_observed = Arc::clone(&observed);
        let task = tokio::spawn(async move {
            run_blocking("cancellation fixture", move |cancellation| {
                worker_started.notify_one();
                while !cancellation.is_cancelled() {
                    std::thread::yield_now();
                }
                worker_observed.notify_one();
                check_cancelled(&cancellation)
            })
            .await
        });

        started.notified().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        observed.notified().await;
    }

    #[test]
    fn successful_completion_disarms_cancellation() {
        let scope = CancellationScope::new();
        let observed = scope.token();
        scope.disarm();
        assert!(!observed.is_cancelled());
    }
}
