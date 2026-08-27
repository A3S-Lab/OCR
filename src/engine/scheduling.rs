use a3s_power::inference::RuntimeDeviceKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CpuExecutionWindowPolicy {
    pub(super) maximum_parallel_jobs: usize,
    pub(super) maximum_reserved_elements: usize,
}

pub(super) fn available_execution_workers() -> usize {
    std::thread::available_parallelism()
        .map(|workers| workers.get())
        .unwrap_or(1)
        .min(rayon::current_num_threads())
        .max(1)
}

pub(super) fn cpu_execution_window_policy(
    device: RuntimeDeviceKind,
    available_workers: usize,
    maximum_tensor_elements: usize,
) -> CpuExecutionWindowPolicy {
    let maximum_parallel_jobs = if device == RuntimeDeviceKind::Cpu {
        available_workers.max(1)
    } else {
        1
    };
    CpuExecutionWindowPolicy {
        maximum_parallel_jobs,
        maximum_reserved_elements: maximum_tensor_elements.max(1),
    }
}

pub(super) fn can_append_execution_job(
    current_jobs: usize,
    current_reserved_elements: usize,
    next_reserved_elements: usize,
    policy: CpuExecutionWindowPolicy,
) -> bool {
    if current_jobs == 0 {
        return true;
    }
    current_jobs < policy.maximum_parallel_jobs
        && current_reserved_elements
            .checked_add(next_reserved_elements)
            .is_some_and(|elements| elements <= policy.maximum_reserved_elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallelism_uses_only_cpu_workers_and_tensor_budget() {
        assert_eq!(
            cpu_execution_window_policy(RuntimeDeviceKind::Cpu, 6, 1_000),
            CpuExecutionWindowPolicy {
                maximum_parallel_jobs: 6,
                maximum_reserved_elements: 1_000,
            }
        );
        for accelerator in [RuntimeDeviceKind::Cuda, RuntimeDeviceKind::Metal] {
            assert_eq!(
                cpu_execution_window_policy(accelerator, 6, 1_000),
                CpuExecutionWindowPolicy {
                    maximum_parallel_jobs: 1,
                    maximum_reserved_elements: 1_000,
                }
            );
        }
    }

    #[test]
    fn windows_never_exceed_workers_or_aggregate_tensor_budget() {
        let policy = CpuExecutionWindowPolicy {
            maximum_parallel_jobs: 3,
            maximum_reserved_elements: 100,
        };

        assert!(can_append_execution_job(0, 0, 120, policy));
        assert!(can_append_execution_job(1, 40, 60, policy));
        assert!(!can_append_execution_job(1, 40, 61, policy));
        assert!(!can_append_execution_job(3, 40, 10, policy));
    }
}
