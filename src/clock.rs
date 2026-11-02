//! The host monotonic clock that frames and triggers share.
//!
//! QPC on Windows, `CLOCK_MONOTONIC` on Unix: the same timebases the platform
//! capture APIs use for their own frame timestamps (Media Foundation's device
//! reference time is QPC-based; V4L2 stamps buffers with `CLOCK_MONOTONIC`).
//! Callers stamp trigger times with [`now_ns`] so they compare directly with
//! [`Frame::host_ns`](crate::Frame::host_ns).

/// Current host monotonic time in nanoseconds.
///
/// ```
/// let a = stance::clock::now_ns();
/// let b = stance::clock::now_ns();
/// assert!(b >= a);
/// ```
#[cfg(windows)]
#[must_use]
pub fn now_ns() -> u64 {
    use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    let mut count = 0i64;
    let mut freq = 0i64;
    // SAFETY: both calls only write to the provided i64s; they cannot fail on
    // Windows XP and later.
    unsafe {
        let _ = QueryPerformanceCounter(&raw mut count);
        let _ = QueryPerformanceFrequency(&raw mut freq);
    }
    qpc_to_ns(count, freq)
}

/// Current host monotonic time in nanoseconds.
///
/// ```
/// let a = stance::clock::now_ns();
/// let b = stance::clock::now_ns();
/// assert!(b >= a);
/// ```
#[cfg(unix)]
#[must_use]
pub fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime only writes to the provided timespec.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut ts);
    }
    // time_t/c_long are i64 on 64-bit targets but narrower elsewhere.
    #[allow(clippy::useless_conversion)]
    timespec_ns(i64::from(ts.tv_sec), i64::from(ts.tv_nsec))
}

/// Current host monotonic time in nanoseconds (process-relative on platforms
/// without a native clock binding).
#[cfg(not(any(windows, unix)))]
#[must_use]
pub fn now_ns() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Converts a QPC tick count to nanoseconds without overflowing.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn qpc_to_ns(count: i64, freq: i64) -> u64 {
    if freq <= 0 || count <= 0 {
        return 0;
    }
    let ns = i128::from(count) * 1_000_000_000 / i128::from(freq);
    u64::try_from(ns).unwrap_or(u64::MAX)
}

/// Converts seconds + nanoseconds to a nanosecond count, clamping negatives to 0.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn timespec_ns(sec: i64, nsec: i64) -> u64 {
    let ns = i128::from(sec) * 1_000_000_000 + i128::from(nsec);
    u64::try_from(ns.max(0)).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_clock_advances() {
        let a = now_ns();
        let b = now_ns();
        assert!(a > 0);
        assert!(b >= a);
    }

    #[test]
    fn qpc_conversion() {
        assert_eq!(qpc_to_ns(10_000_000, 10_000_000), 1_000_000_000);
        assert_eq!(qpc_to_ns(3, 3_000_000), 1000);
        assert_eq!(qpc_to_ns(i64::MAX, 10_000_000), u64::MAX);
        assert_eq!(qpc_to_ns(5, 0), 0);
    }

    #[test]
    fn timespec_conversion() {
        assert_eq!(timespec_ns(2, 5), 2_000_000_005);
        assert_eq!(timespec_ns(-1, 0), 0);
    }
}
