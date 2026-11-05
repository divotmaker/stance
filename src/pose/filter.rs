//! Keypoint smoothing: One Euro for live use, zero-phase Butterworth for
//! whole clips.
//!
//! Both work on one coordinate series at a time. Run them on `x` and `y`
//! of each keypoint separately; scores are not filtered.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// One Euro filter parameters (Casiez, Roussel and Vogel, CHI 2012).
///
/// The cutoff frequency rises with speed: `cutoff = min_cutoff + beta ×
/// |speed|`. A low `min_cutoff` removes jitter while the point is still; a
/// higher `beta` cuts lag when it moves fast. Speeds are in the signal's units
/// per second (pixels per second for keypoints).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OneEuroParams {
    /// Cutoff at rest (Hz).
    pub min_cutoff: f64,
    /// How fast the cutoff rises with speed (Hz per unit/s).
    pub beta: f64,
    /// Cutoff for the speed estimate itself (Hz).
    pub d_cutoff: f64,
}

impl Default for OneEuroParams {
    /// 1 Hz at rest, `beta` 0.05 per px/s, 1 Hz on the speed estimate:
    /// steady keypoints for a still golfer, small lag during the swing.
    fn default() -> Self {
        Self {
            min_cutoff: 1.0,
            beta: 0.05,
            d_cutoff: 1.0,
        }
    }
}

/// A One Euro filter for one value.
#[derive(Debug, Clone, PartialEq)]
pub struct OneEuro {
    params: OneEuroParams,
    last: Option<(f64, f64, f64)>,
}

impl OneEuro {
    /// A filter with no history.
    #[must_use]
    pub fn new(params: OneEuroParams) -> Self {
        Self { params, last: None }
    }

    /// Filters sample `x` taken at time `t` (seconds). The first sample, and
    /// any sample not later than the previous one or not finite, passes
    /// through and restarts the filter from there.
    pub fn filter(&mut self, t: f64, x: f64) -> f64 {
        let Some((t0, x0, dx0)) = self.last else {
            self.last = x.is_finite().then_some((t, x, 0.0));
            return x;
        };
        let dt = t - t0;
        if dt.is_nan() || dt <= 0.0 || !x.is_finite() || !dt.is_finite() {
            self.last = x.is_finite().then_some((t, x, 0.0));
            return x;
        }
        let dx = (x - x0) / dt;
        let dx_hat = dx0 + alpha(self.params.d_cutoff, dt) * (dx - dx0);
        let cutoff = self.params.min_cutoff + self.params.beta * dx_hat.abs();
        let x_hat = x0 + alpha(cutoff, dt) * (x - x0);
        self.last = Some((t, x_hat, dx_hat));
        x_hat
    }

    /// Forgets the history.
    pub fn reset(&mut self) {
        self.last = None;
    }
}

/// Smoothing factor of a first-order low-pass at `cutoff` Hz for a step of
/// `dt` seconds.
fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff.max(1e-6));
    1.0 / (1.0 + tau / dt)
}

/// A Butterworth low-pass run forwards and backwards (zero phase), for a
/// whole clip.
///
/// `order` is the order of the filter designed (1–8); running it twice
/// doubles the effective order and squares the magnitude response, so the
/// attenuation at `cutoff_hz` is 6 dB. The ends are padded with an odd
/// reflection of the signal and each section starts at its steady state for
/// the first padded sample, so a constant signal passes unchanged and the
/// ends don't ring.
///
/// The filter assumes evenly spaced samples. For a swing, filter the
/// segments before and after impact separately, so the filter doesn't smear
/// the impact frames.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Butterworth {
    /// Cutoff frequency (Hz).
    pub cutoff_hz: f64,
    /// Order of the designed filter, 1–8.
    pub order: u32,
}

/// One second-order section, transposed direct form II.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
}

impl Biquad {
    /// Runs the section over `x` in place, starting at the steady state for
    /// a constant input `x[0]` (the low-pass DC gain is 1).
    fn run(&self, x: &mut [f64]) {
        let Some(&first) = x.first() else { return };
        let [b0, b1, b2] = self.b;
        let [a1, a2] = self.a;
        let mut z1 = (1.0 - b0) * first;
        let mut z2 = (b2 - a2) * first;
        for v in x.iter_mut() {
            let input = *v;
            let y = b0 * input + z1;
            z1 = b1 * input - a1 * y + z2;
            z2 = b2 * input - a2 * y;
            *v = y;
        }
    }
}

impl Butterworth {
    /// Second-order sections of the digital low-pass (bilinear transform
    /// with pre-warping) for `sample_rate_hz`.
    fn sections(&self, sample_rate_hz: f64) -> Result<Vec<Biquad>> {
        let nyquist = sample_rate_hz / 2.0;
        if !(1..=8).contains(&self.order)
            || self.cutoff_hz.is_nan()
            || self.cutoff_hz <= 0.0
            || self.cutoff_hz >= nyquist
            || !sample_rate_hz.is_finite()
        {
            return Err(Error::InvalidArgument(format!(
                "Butterworth order {} at {} Hz needs order 1-8 and 0 < cutoff < {nyquist} Hz",
                self.order, self.cutoff_hz
            )));
        }
        // Pre-warped analog cutoff for a unit sample period, K = tan(π fc / fs).
        let k = (std::f64::consts::PI * self.cutoff_hz / sample_rate_hz).tan();
        let n = self.order;
        let mut sections = Vec::new();
        for i in 0..n / 2 {
            // Pole pair of the analog prototype: s² + 2 sin(θ) s + 1.
            let theta = std::f64::consts::PI * f64::from(2 * i + 1) / f64::from(2 * n);
            let q2 = 2.0 * theta.sin();
            let norm = 1.0 / (1.0 + q2 * k + k * k);
            let b0 = k * k * norm;
            sections.push(Biquad {
                b: [b0, 2.0 * b0, b0],
                a: [2.0 * (k * k - 1.0) * norm, (1.0 - q2 * k + k * k) * norm],
            });
        }
        if n % 2 == 1 {
            // Real pole: a first-order section written as a biquad.
            let norm = 1.0 / (1.0 + k);
            sections.push(Biquad {
                b: [k * norm, k * norm, 0.0],
                a: [(k - 1.0) * norm, 0.0],
            });
        }
        Ok(sections)
    }

    /// Filters the series forwards and backwards.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an order outside 1–8 or a cutoff not
    /// strictly between 0 and half the sample rate.
    pub fn filtfilt(&self, samples: &[f64], sample_rate_hz: f64) -> Result<Vec<f64>> {
        let sections = self.sections(sample_rate_hz)?;
        let n = samples.len();
        if n < 2 {
            return Ok(samples.to_vec());
        }
        // SciPy's default pad length: three times the filter length.
        let pad = (3 * (2 * sections.len() + 1)).min(n - 1);
        let (first, last) = (samples[0], samples[n - 1]);
        let mut x = Vec::with_capacity(n + 2 * pad);
        x.extend((1..=pad).rev().map(|i| 2.0 * first - samples[i]));
        x.extend_from_slice(samples);
        x.extend((1..=pad).map(|i| 2.0 * last - samples[n - 1 - i]));
        for s in &sections {
            s.run(&mut x);
        }
        x.reverse();
        for s in &sections {
            s.run(&mut x);
        }
        x.reverse();
        Ok(x[pad..pad + n].to_vec())
    }
}

#[cfg(test)]
// Pass-through samples are compared exactly.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn sine(freq: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / fs;
                (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect()
    }

    fn rms(v: &[f64]) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let n = v.len() as f64;
        (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt()
    }

    #[test]
    fn one_euro_follows_a_step_and_damps_jitter() {
        let mut f = OneEuro::new(OneEuroParams::default());
        let fs = 120.0;
        // Still with ±1 px jitter: the output varies much less.
        let mut out = Vec::new();
        for i in 0..240 {
            let x = 100.0 + if i % 2 == 0 { 1.0 } else { -1.0 };
            out.push(f.filter(f64::from(i) / fs, x));
        }
        let tail = &out[120..];
        let spread = tail.iter().copied().fold(f64::MIN, f64::max)
            - tail.iter().copied().fold(f64::MAX, f64::min);
        assert!(spread < 0.3, "spread {spread}");
        // A 200 px step: reached (within 2 px) inside 0.25 s, never overshot.
        let mut last = 0.0;
        for i in 240..360 {
            last = f.filter(f64::from(i) / fs, 300.0);
            assert!(last <= 300.0 + 1e-9);
            if i == 270 {
                assert!(last > 298.0, "after 0.25 s: {last}");
            }
        }
        assert!((last - 300.0).abs() < 0.01);
    }

    #[test]
    fn one_euro_restarts_on_bad_time() {
        let mut f = OneEuro::new(OneEuroParams::default());
        assert_eq!(f.filter(1.0, 5.0), 5.0);
        let y = f.filter(1.1, 10.0);
        assert!(y > 5.0 && y < 10.0);
        assert_eq!(f.filter(1.0, 42.0), 42.0, "time went backwards");
        assert!(f.filter(1.0, f64::NAN).is_nan());
        assert_eq!(f.filter(2.0, 7.0), 7.0, "restarted after NaN");
        f.reset();
        assert_eq!(f.filter(0.0, -3.0), -3.0);
    }

    #[test]
    fn butterworth_passes_dc_and_low_frequencies() {
        let bw = Butterworth {
            cutoff_hz: 15.0,
            order: 4,
        };
        let flat = vec![3.5; 50];
        for (a, b) in bw.filtfilt(&flat, 120.0).unwrap().iter().zip(&flat) {
            assert!((a - b).abs() < 1e-9);
        }
        // 2 Hz passes almost unchanged and in phase.
        let low = sine(2.0, 120.0, 600);
        let y = bw.filtfilt(&low, 120.0).unwrap();
        let err: Vec<f64> = y.iter().zip(&low).map(|(a, b)| a - b).collect();
        assert!(rms(&err[60..540]) < 0.01, "rms err {}", rms(&err[60..540]));
    }

    #[test]
    fn butterworth_cuts_high_frequencies_with_zero_phase() {
        let bw = Butterworth {
            cutoff_hz: 10.0,
            order: 2,
        };
        // At the cutoff the forward-backward gain is 0.5 (−6 dB).
        let at = sine(10.0, 240.0, 2400);
        let y = bw.filtfilt(&at, 240.0).unwrap();
        let gain = rms(&y[240..2160]) / rms(&at[240..2160]);
        assert!((gain - 0.5).abs() < 0.02, "gain {gain}");
        // Zero phase: the filtered sine peaks where the input does.
        let corr: f64 = y[240..2160]
            .iter()
            .zip(&at[240..2160])
            .map(|(a, b)| a * b)
            .sum();
        assert!(corr > 0.0);
        let peak_in = (240..264).max_by(|&a, &b| at[a].total_cmp(&at[b])).unwrap();
        let peak_out = (240..264).max_by(|&a, &b| y[a].total_cmp(&y[b])).unwrap();
        assert_eq!(peak_in, peak_out);
        // Well above the cutoff almost nothing is left.
        let high = sine(60.0, 240.0, 2400);
        let y = bw.filtfilt(&high, 240.0).unwrap();
        assert!(rms(&y[240..2160]) < 0.01);
    }

    #[test]
    fn butterworth_step_settles_without_ringing_at_the_ends() {
        let bw = Butterworth {
            cutoff_hz: 6.0,
            order: 3,
        };
        let step: Vec<f64> = (0..240).map(|i| if i < 120 { 0.0 } else { 1.0 }).collect();
        let y = bw.filtfilt(&step, 120.0).unwrap();
        assert!(y[0].abs() < 1e-6 && (y[239] - 1.0).abs() < 1e-6);
        // Symmetric around the step (zero phase), halfway at the edge.
        assert!((y[119] + y[120] - 1.0).abs() < 0.02);
        assert!(y.iter().all(|v| (-0.1..=1.1).contains(v)));
    }

    #[test]
    fn butterworth_rejects_bad_parameters_and_handles_short_input() {
        let bad = |cutoff_hz, order| Butterworth { cutoff_hz, order }.filtfilt(&[1.0, 2.0], 100.0);
        assert!(bad(50.0, 2).is_err());
        assert!(bad(0.0, 2).is_err());
        assert!(bad(10.0, 0).is_err());
        assert!(bad(10.0, 9).is_err());
        assert!(bad(f64::NAN, 2).is_err());
        let bw = Butterworth {
            cutoff_hz: 10.0,
            order: 4,
        };
        assert_eq!(bw.filtfilt(&[], 100.0).unwrap(), Vec::<f64>::new());
        assert_eq!(bw.filtfilt(&[4.0], 100.0).unwrap(), vec![4.0]);
        assert_eq!(bw.filtfilt(&[4.0, 4.0, 4.0], 100.0).unwrap().len(), 3);
    }
}
