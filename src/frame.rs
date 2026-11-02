//! Frames and capture modes.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// One compressed frame, as the source delivered it.
///
/// Cloning is cheap: the JPEG bytes are shared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// The MJPEG frame exactly as the camera sent it. Many webcams omit the
    /// Huffman tables; run [`jpeg::ensure_huffman`](crate::jpeg::ensure_huffman)
    /// before handing it to a strict decoder.
    pub jpeg: Arc<[u8]>,
    /// Host monotonic time (ns, see `clock::now_ns`) when the frame arrived.
    pub host_ns: u64,
    /// The driver's capture time for the frame on the same host monotonic
    /// clock, when the platform provides one.
    pub device_ns: Option<u64>,
    /// Frame counter, increasing by one per delivered frame.
    ///
    /// A [`Source`](crate::Source) counts from 0. A [`Capture`](crate::Capture)
    /// re-stamps every frame from a process-wide counter: each capture gets a
    /// block of numbers above every block handed out before it, so a cursor
    /// kept across a reconnect (a new `Capture` for the same camera) never
    /// hides the new capture's frames. Within one capture the numbers are
    /// consecutive.
    pub seq: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// One capture mode: pixel format × size × frame rate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Format {
    /// Pixel format fourcc, e.g. `MJPG`, `YUY2`, `NV12`.
    pub fourcc: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate numerator (`fps = fps_num / fps_den`).
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
}

impl Format {
    /// The fourcc of MJPEG modes.
    pub const MJPG: &'static str = "MJPG";

    /// An MJPEG mode at a whole frame rate.
    #[must_use]
    pub fn mjpeg(width: u32, height: u32, fps: u32) -> Self {
        Self {
            fourcc: Self::MJPG.to_owned(),
            width,
            height,
            fps_num: fps,
            fps_den: 1,
        }
    }

    /// Frames per second.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            0.0
        } else {
            f64::from(self.fps_num) / f64::from(self.fps_den)
        }
    }

    /// Nominal time between frames in nanoseconds (0 for an unknown rate).
    #[must_use]
    pub fn frame_period_ns(&self) -> u64 {
        if self.fps_num == 0 {
            0
        } else {
            u64::from(self.fps_den) * 1_000_000_000 / u64::from(self.fps_num)
        }
    }

    /// Pixel count.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// True for MJPEG modes.
    #[must_use]
    pub fn is_mjpeg(&self) -> bool {
        self.fourcc.eq_ignore_ascii_case(Self::MJPG)
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}x{} @ {}",
            self.fourcc,
            self.width,
            self.height,
            fmt_fps(self.fps())
        )
    }
}

/// Gap threshold, in frame periods, when both frames carry a device time.
pub(crate) const DEVICE_GAP: f64 = 1.5;
/// Gap threshold, in frame periods, on arrival times: USB and driver delivery
/// jitter (frames arriving in pairs) must not count as drops.
pub(crate) const ARRIVAL_GAP: f64 = 2.5;

/// Frames missing between two consecutive frames of a stream whose nominal
/// frame period is `period_ns`. Uses the device times when both frames have
/// one (gaps over [`DEVICE_GAP`] periods count), the arrival times otherwise
/// (gaps over [`ARRIVAL_GAP`] periods count). A counted gap is
/// `round(gap / period) − 1` frames.
pub(crate) fn missing_between(a: &Frame, b: &Frame, period_ns: f64) -> u64 {
    let (gap, threshold) = match (a.device_ns, b.device_ns) {
        (Some(x), Some(y)) => (y.saturating_sub(x), DEVICE_GAP),
        _ => (b.host_ns.saturating_sub(a.host_ns), ARRIVAL_GAP),
    };
    missing_in_gap(gap, period_ns, threshold)
}

/// Frames missing in a gap of `gap_ns` longer than `threshold` periods.
pub(crate) fn missing_in_gap(gap_ns: u64, period_ns: f64, threshold: f64) -> u64 {
    if period_ns.is_nan() || period_ns <= 0.0 {
        return 0;
    }
    // Gaps and periods are far below 2^52 ns.
    #[allow(clippy::cast_precision_loss)]
    let ratio = gap_ns as f64 / period_ns;
    if ratio <= threshold {
        return 0;
    }
    // Rounded and positive; `as` saturates for huge ratios.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let periods = ratio.round() as u64;
    periods.saturating_sub(1)
}

/// `60`, `29.97`.
pub(crate) fn fmt_fps(fps: f64) -> String {
    let s = format!("{fps:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_and_period() {
        let f = Format::mjpeg(1280, 720, 120);
        assert!((f.fps() - 120.0).abs() < 1e-9);
        assert_eq!(f.frame_period_ns(), 8_333_333);
        assert_eq!(f.to_string(), "MJPG 1280x720 @ 120");
        let ntsc = Format {
            fps_num: 30_000,
            fps_den: 1001,
            ..f
        };
        assert_eq!(ntsc.to_string(), "MJPG 1280x720 @ 29.97");
        let zero = Format {
            fps_num: 0,
            fps_den: 0,
            ..ntsc
        };
        assert!(zero.fps().abs() < f64::EPSILON);
        assert_eq!(zero.frame_period_ns(), 0);
    }

    fn at(host_ns: u64, device_ns: Option<u64>) -> Frame {
        Frame {
            jpeg: Arc::from(vec![0u8; 1]),
            host_ns,
            device_ns,
            seq: 0,
            width: 1,
            height: 1,
        }
    }

    #[test]
    fn gaps_use_device_times_when_present() {
        let p = 10_000_000.0;
        // Device times: one missing frame in a two-period gap.
        assert_eq!(
            missing_between(&at(0, Some(0)), &at(20_000_000, Some(20_000_000)), p),
            1
        );
        // Device times are trusted over jittery arrival times.
        assert_eq!(
            missing_between(&at(0, Some(0)), &at(26_000_000, Some(10_000_000)), p),
            0
        );
        // Arrival times only: a two-period gap is delivery jitter...
        assert_eq!(missing_between(&at(0, None), &at(20_000_000, None), p), 0);
        // ...three periods is two missing frames.
        assert_eq!(missing_between(&at(0, None), &at(30_000_000, None), p), 2);
        assert_eq!(missing_in_gap(5, 0.0, DEVICE_GAP), 0);
        assert_eq!(missing_in_gap(u64::MAX, 1.0, DEVICE_GAP), u64::MAX - 1);
    }
}
