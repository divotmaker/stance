//! Gap filling and zero-phase smoothing of a clip's keypoints.

use serde::{Deserialize, Serialize};

use super::body::{Body, Handed, Pt};
use crate::pose::{Butterworth, Keypoint, PoseFrame, PoseTrack, Skeleton};
use crate::{Error, Result};

/// How a clip's keypoints are cleaned before analysis.
///
/// A keypoint scoring below `min_score` is a gap, never a zero. Gaps up to
/// `max_gap_ns` long (from the last good sample to the next) are filled by
/// linear interpolation; longer gaps stay empty. Each unbroken stretch of a
/// keypoint is then filtered with a zero-phase [`Butterworth`] low-pass,
/// separately before and after the impact, so the filter doesn't smear the
/// fastest frames of the swing. The filter's sample rate is the median
/// frame interval: frames are not resampled, so a camera that drops frames
/// gets a hole treated as one step (clip times are regular in practice).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Smoothing {
    /// Keypoints scoring below this are gaps.
    pub min_score: f32,
    /// The longest gap filled by interpolation (ns).
    pub max_gap_ns: u64,
    /// Low-pass cutoff (Hz on the track's time base), capped at 0.4 × the
    /// frame rate; `0` turns filtering off.
    pub cutoff_hz: f64,
    /// Butterworth order (1–8), applied forwards and backwards.
    pub order: u32,
}

impl Default for Smoothing {
    /// Score 0.3, gaps up to 150 ms, an 8 Hz second-order low-pass.
    fn default() -> Self {
        Self {
            min_score: 0.3,
            max_gap_ns: 150_000_000,
            cutoff_hz: 8.0,
            order: 2,
        }
    }
}

/// A smoothed track: per frame, each keypoint's position (where usable) and
/// score (interpolated keypoints carry half the lower score of their two
/// neighbours, gaps 0).
#[derive(Debug, Clone)]
pub(crate) struct Frames {
    pub t_ns: Vec<u64>,
    pub bodies: Vec<Body>,
}

impl Frames {
    pub fn len(&self) -> usize {
        self.t_ns.len()
    }

    /// The index of the frame nearest `t_ns`.
    pub fn nearest(&self, t_ns: u64) -> Option<usize> {
        let i = self.t_ns.partition_point(|&t| t < t_ns);
        match (i.checked_sub(1), (i < self.len()).then_some(i)) {
            (Some(a), Some(b)) => Some(if t_ns - self.t_ns[a] <= self.t_ns[b] - t_ns {
                a
            } else {
                b
            }),
            (a, b) => a.or(b),
        }
    }
}

fn median_interval(t_ns: &[u64]) -> Option<f64> {
    let mut d: Vec<u64> = t_ns
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]))
        .filter(|&d| d > 0)
        .collect();
    if d.is_empty() {
        return None;
    }
    d.sort_unstable();
    // Nanosecond intervals are far below 2^52.
    #[allow(clippy::cast_precision_loss, reason = "ns intervals fit f64")]
    Some(d[d.len() / 2] as f64 * 1e-9)
}

/// Fills short gaps and smooths every keypoint; `split_at_ns` (the impact)
/// ends one filter segment, so the frame at or before it and the frame after
/// it are filtered apart.
pub(crate) fn smooth(
    track: &PoseTrack,
    handed: Handed,
    smoothing: &Smoothing,
    split_at_ns: Option<u64>,
) -> Result<Frames> {
    if track.skeleton != Skeleton::Halpe26 {
        return Err(Error::InvalidArgument(
            "golf analysis needs Halpe-26 keypoints".into(),
        ));
    }
    let n = track.frames.len();
    let t_ns: Vec<u64> = track.frames.iter().map(|f| f.t_ns).collect();
    if t_ns.windows(2).any(|w| w[1] < w[0]) {
        return Err(Error::InvalidArgument(
            "pose track frame times go backwards".into(),
        ));
    }
    let raw: Vec<Body> = track
        .frames
        .iter()
        .map(|f| {
            Body::from_keypoints(&f.keypoints, handed, smoothing.min_score).unwrap_or(Body {
                pts: [None; 26],
                scores: [0.0; 26],
                handed,
            })
        })
        .collect();
    let rate_hz = median_interval(&t_ns).map(|dt| 1.0 / dt);
    let filter = match rate_hz {
        Some(fs) if smoothing.cutoff_hz > 0.0 => Some((
            Butterworth {
                cutoff_hz: smoothing.cutoff_hz.min(0.4 * fs),
                order: smoothing.order,
            },
            fs,
        )),
        _ => None,
    };
    // The first frame after the split starts a new segment.
    let split = split_at_ns.map(|s| t_ns.partition_point(|&t| t <= s));
    let mut bodies = raw.clone();
    for k in 0..26 {
        let (pts, scores) = fill_gaps(&t_ns, &raw, k, smoothing.max_gap_ns);
        let mut i = 0;
        while i < n {
            if pts[i].is_none() {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < n && pts[j].is_some() && (j == i || Some(j) != split) {
                j += 1;
            }
            let seg: Vec<Pt> = pts[i..j].iter().flatten().copied().collect();
            let out = match &filter {
                Some((bw, fs)) => {
                    let xs: Vec<f64> = seg.iter().map(|p| p.x).collect();
                    let ys: Vec<f64> = seg.iter().map(|p| p.y).collect();
                    let (xs, ys) = (bw.filtfilt(&xs, *fs)?, bw.filtfilt(&ys, *fs)?);
                    xs.into_iter().zip(ys).map(|(x, y)| Pt::new(x, y)).collect()
                }
                None => seg,
            };
            for (m, p) in out.into_iter().enumerate() {
                bodies[i + m].pts[k] = Some(p);
                bodies[i + m].scores[k] = scores[i + m];
            }
            i = j;
        }
    }
    Ok(Frames { t_ns, bodies })
}

/// Keypoint `k` per frame with gaps up to `max_gap_ns` interpolated.
fn fill_gaps(t_ns: &[u64], raw: &[Body], k: usize, max_gap_ns: u64) -> (Vec<Option<Pt>>, Vec<f32>) {
    let mut pts: Vec<Option<Pt>> = raw.iter().map(|b| b.pts[k]).collect();
    let mut scores: Vec<f32> = raw.iter().map(|b| b.scores[k]).collect();
    let mut last: Option<usize> = None;
    for i in 0..pts.len() {
        let Some(b) = pts[i] else { continue };
        if let Some(a) = last
            && i > a + 1
            && t_ns[i] - t_ns[a] <= max_gap_ns
            && let Some(pa) = pts[a]
        {
            let span = (t_ns[i] - t_ns[a]).max(1);
            let score = 0.5 * scores[a].min(scores[i]);
            for m in a + 1..i {
                // ns offsets inside a gap fit f64 exactly.
                #[allow(clippy::cast_precision_loss, reason = "ns offsets fit f64")]
                let u = (t_ns[m] - t_ns[a]) as f64 / span as f64;
                pts[m] = Some(Pt::new(pa.x + u * (b.x - pa.x), pa.y + u * (b.y - pa.y)));
                scores[m] = score;
            }
        }
        last = Some(i);
    }
    (pts, scores)
}

/// The smoothed keypoints as a [`PoseTrack`] (same frames, times and
/// metadata): what [`analyze_swing`](super::analyze_swing) measures on, for
/// drawing or per-frame [`posture`](super::posture).
///
/// Keypoints in a gap longer than [`Smoothing::max_gap_ns`] keep their raw
/// position with score 0; interpolated keypoints score half the lower score
/// of their neighbours. `split_at_ns` is the impact time (ns on the track's
/// time base), where filtering is split.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a skeleton other than Halpe-26, frame
/// times that go backwards, or bad [`Smoothing`] filter parameters.
pub fn smooth_track(
    track: &PoseTrack,
    smoothing: &Smoothing,
    split_at_ns: Option<u64>,
) -> Result<PoseTrack> {
    // Handedness only renames joints; the right-handed view keeps indices.
    let frames = smooth(track, Handed::Right, smoothing, split_at_ns)?;
    let out = track
        .frames
        .iter()
        .zip(&frames.bodies)
        .map(|(raw, body)| {
            if body.pts.iter().all(Option::is_none) {
                return PoseFrame::empty(raw.t_ns);
            }
            let keypoints: Vec<Keypoint> = (0..26)
                .map(|k| match body.pts[k] {
                    // Pixel coordinates fit f32.
                    #[allow(clippy::cast_possible_truncation, reason = "pixels fit f32")]
                    Some(p) => Keypoint {
                        x: p.x as f32,
                        y: p.y as f32,
                        score: body.scores[k],
                    },
                    None => Keypoint {
                        score: 0.0,
                        ..raw.keypoints.get(k).copied().unwrap_or_default()
                    },
                })
                .collect();
            #[allow(clippy::cast_precision_loss, reason = "26 keypoints")]
            let score = keypoints.iter().map(|k| k.score).sum::<f32>() / keypoints.len() as f32;
            PoseFrame {
                t_ns: raw.t_ns,
                score: Some(score),
                bbox: raw.bbox,
                keypoints,
            }
        })
        .collect();
    Ok(PoseTrack {
        version: track.version,
        skeleton: track.skeleton,
        model: track.model.clone(),
        detector: track.detector.clone(),
        width: track.width,
        height: track.height,
        frames: out,
    })
}
