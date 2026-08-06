use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{bail, Context, Result};

pub(crate) struct ResidentMemorySampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl ResidentMemorySampler {
    pub(crate) fn start() -> Result<Self> {
        let initial = resident_bytes()?;
        if initial == 0 {
            bail!("the operating system returned a zero resident-memory sample");
        }
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(initial));
        let thread_stop = Arc::clone(&stop);
        let thread_peak = Arc::clone(&peak);
        let handle = std::thread::Builder::new()
            .name("a3s-ocr-rss-sampler".to_owned())
            .spawn(move || {
                while !thread_stop.load(Ordering::Acquire) {
                    if let Ok(sample) = resident_bytes() {
                        thread_peak.fetch_max(sample, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                if let Ok(sample) = resident_bytes() {
                    thread_peak.fetch_max(sample, Ordering::Relaxed);
                }
            })
            .context("could not start the resident-memory sampler")?;
        Ok(Self {
            stop,
            peak,
            handle: Some(handle),
        })
    }

    pub(crate) fn finish(mut self) -> Result<u64> {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("the resident-memory sampler panicked"))?;
        }
        let peak = self.peak.load(Ordering::Relaxed);
        if peak == 0 {
            bail!("the resident-memory sampler recorded no usable sample");
        }
        Ok(peak)
    }
}

impl Drop for ResidentMemorySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn resident_bytes() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")
        .context("could not read Linux process memory status")?;
    let kibibytes = status
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix("VmRSS:")?.trim();
            value.strip_suffix("kB")?.trim().parse::<u64>().ok()
        })
        .context("Linux process memory status did not contain VmRSS")?;
    kibibytes
        .checked_mul(1_024)
        .context("Linux resident-memory byte count overflowed")
}

#[cfg(target_os = "macos")]
pub(crate) fn resident_bytes() -> Result<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to writable storage for one `rusage`, and
    // `RUSAGE_SELF` asks the kernel only about this process.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        bail!("macOS getrusage failed for the current process");
    }
    // SAFETY: a successful `getrusage` initialized the complete structure.
    let usage = unsafe { usage.assume_init() };
    u64::try_from(usage.ru_maxrss).context("macOS returned a negative resident-memory sample")
}

#[cfg(windows)]
pub(crate) fn resident_bytes() -> Result<u64> {
    use std::ffi::c_void;

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    let mut counters = std::mem::MaybeUninit::<ProcessMemoryCounters>::zeroed();
    let size = u32::try_from(std::mem::size_of::<ProcessMemoryCounters>())
        .context("Windows process-memory structure size cannot be represented")?;
    // SAFETY: zero is valid for every integer field, and `cb` is the one input
    // field required by `GetProcessMemoryInfo`.
    unsafe {
        (*counters.as_mut_ptr()).cb = size;
    }
    // SAFETY: the pseudo-handle is valid for this process and the output
    // pointer is writable for `size` bytes.
    let result = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), counters.as_mut_ptr(), size) };
    if result == 0 {
        bail!("Windows GetProcessMemoryInfo failed for the current process");
    }
    // SAFETY: a successful call initialized the complete structure.
    let counters = unsafe { counters.assume_init() };
    u64::try_from(counters.working_set_size)
        .context("Windows resident-memory byte count cannot be represented")
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub(crate) fn resident_bytes() -> Result<u64> {
    bail!("resident-memory measurement is unsupported on this platform")
}
