//! Swing events from the hands' trajectory, and tempo.

use serde::{Deserialize, Serialize};

use super::body::{Body, Joint, Pt};
use super::smooth::Frames;

/// A swing event: when it happened and how sure the detector is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SwingEvent {
    /// Time on the track's time base (ns from the clip's first frame).
    pub t_ns: u64,
    /// The index of the frame nearest that time.
    pub frame: usize,
    /// `0..=1`: the wrists' keypoint scores, times how clearly the event
    /// stands out in the trajectory.
    pub confidence: f32,
}

/// Where the impact time came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImpactSource {
    /// Given by the caller and used as is, at confidence 0.5: the hands show
    /// no clear low point within 0.3 s of it. Unconfirmed, so no swing was
    /// found around it: the other events (and so the tempo) are `None`.
    Given,
    /// Given by the caller (typically the launch monitor's shot time, an
    /// estimate itself) and moved to the hands' low point within 0.3 s of
    /// it.
    Refined,
    /// Estimated from the hands alone.
    Estimated,
}

/// The swing's key moments, found from the hands (the wrists' midpoint,
/// or the one wrist found). Thresholds are in torso lengths (`L`), so they
/// don't depend on image size or distance; durations are on the track's
/// time base. Hand speeds are measured over at least 50 ms (at least one
/// frame either side).
///
/// - **impact**: the hands bottom out a little before the ball, so impact
///   is 25 ms after the hands' low point (sub-frame, a parabola through its
///   neighbours). Of the low points, the one with the clearest swing
///   through it: the smaller of the drop from its top and the rise in the
///   follow-through after it (within two downswings, at most 0.6 s), times
///   the downswing's peak hand speed over half the clip's peak (at most 1,
///   so lowering the hands slowly to address scores low), times the
///   downswing's plausibility (below, from the top's onset; at least a
///   quarter, as a slowed time base makes every downswing long). With a
///   given impact time the search is a prior: only low points within 0.3 s
///   of it count (or reached by the hands still descending past that
///   window), each weighted down the further it is (to half at 0.3 s), and
///   the result is [`ImpactSource::Refined`]. With no clear low point
///   there, the given time stands ([`ImpactSource::Given`]) and no other
///   event is derived from it.
/// - **top**: the start of the downswing. Walking back from the impact's
///   low point, the hands' highest point (until they are 0.3 L below it
///   again) bounds the backswing; walking back from the downswing's fastest
///   descent, the top is where the hands' descent speed falls below 10 % of
///   that peak (between frames, never before the highest point's sub-frame
///   time). A pause at the top, even one where the hands sag, ends
///   where the downswing starts. When the speeds are unusable, the last
///   frame within 0.03 L of the highest point.
/// - **address**: walking back from the top, the first time the hands were
///   at rest (within 0.03 L for 0.2 s, at least 0.5 L below the top); the
///   address is the last frame they are within 0.015 L of that rest
///   position before the backswing. A clip that starts less than 0.2 s
///   before the takeaway (but at least two frames) counts as at rest from
///   its first frame, at reduced confidence (down to half).
/// - **takeaway**: the frame after address, where the hands leave the rest
///   position for good.
/// - **arm parallel back** (P3) and **arm parallel down** (P5): the lead
///   arm parallel to the ground, where the lead wrist passes the lead
///   shoulder's height (in either view: an arm level with the ground has
///   its wrist at shoulder height from any level camera). P3 is the last
///   time the wrist rises through it after the takeaway (or the first
///   frame) and at least one frame before the top, so a pose glitch early in
///   the clip can't stand in for it; P5 is the first time it falls through
///   it at least one frame after the top and up to impact, so hands
///   hovering at shoulder height over the top don't put either on the top
///   frame. Both are sub-frame (linear between the frames either side).
///   The lead wrist falls back to the hands and the lead shoulder to the
///   neck keypoint (between the shoulders) where they are missing, at half
///   the confidence. `None` when the hands never reach shoulder height (a
///   chip, a pitch) or there is no top (or, for P5, no impact). On real
///   swings P5 comes a little before the frame people label "arm parallel"
///   by eye (a few frames at 120 fps): the arm is moving fast there, and
///   by the frame that looks parallel the wrist is already below the
///   shoulder in the image.
/// - **finish**: after impact, once the hands are above mid-torso and past
///   their highest point, the first frame where their speed falls below 3 %
///   of the swing's peak hand speed. When they never settle, the highest
///   point, at half the confidence.
///
/// Confidence is the wrists' mean keypoint score (for the top and impact,
/// the lowest of the frame and its neighbours, so an event beside
/// interpolated hands carries their halved score), scaled down where the
/// trajectory is unclear: a top less than 1 L above the address hands, a
/// downswing (top → impact) outside 120–700 ms (to 0 at 60 ms and 1 s), a
/// low point with less than 1 L of swing through it or a slow downswing,
/// a finish where the hands stay settled (under 6 % of the peak speed) for
/// less than 0.3 s. For P3 and P5, it is the wrist's and shoulder's mean
/// score either side of the crossing, times how clearly the wrist crosses
/// (at least 0.02 L of height change over that step for full confidence,
/// so a wrist drifting along the shoulder's height scores low).
///
/// Events missing from the clip (a clip that starts mid-swing, a golfer out
/// of frame) are `None`, as are all but the impact when it is
/// [`ImpactSource::Given`].
///
/// Limitations:
///
/// - An estimated impact trusts the keypoints: a pose model that latches
///   onto something else for a stretch (both wrists jumping by a body
///   length between frames and holding there) makes drops and rises that
///   can outscore the real swing, at a confidence that looks sound. A
///   given impact time keeps the search near the shot; give one whenever
///   it is known.
/// - Two low points of similar clarity within 0.3 s of a given time (a
///   golfer partly out of frame, a crowded follow-through) are told apart
///   by their distance from it, so the impact (and the top and tempo with
///   it) can move between them as the given time moves.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SwingEvents {
    /// Hands still, before the takeaway.
    pub address: Option<SwingEvent>,
    /// The hands start moving.
    pub takeaway: Option<SwingEvent>,
    /// P3: the lead arm parallel to the ground in the backswing. `None`
    /// when absent from JSON.
    #[serde(default)]
    pub arm_parallel_back: Option<SwingEvent>,
    /// The top of the backswing.
    pub top: Option<SwingEvent>,
    /// P5: the lead arm parallel to the ground in the downswing. `None`
    /// when absent from JSON.
    #[serde(default)]
    pub arm_parallel_down: Option<SwingEvent>,
    /// Impact.
    pub impact: Option<SwingEvent>,
    /// Where [`SwingEvents::impact`] came from.
    pub impact_source: ImpactSource,
    /// The hands come to rest after the follow-through.
    pub finish: Option<SwingEvent>,
}

/// Swing tempo: backswing (takeaway → top) against downswing (top →
/// impact). Good players are typically near 3:1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Tempo {
    /// Takeaway to top (ns).
    pub backswing_ns: u64,
    /// Top to impact (ns).
    pub downswing_ns: u64,
    /// `backswing_ns / downswing_ns`.
    pub ratio: f64,
}

impl Tempo {
    /// The tempo of a swing with a takeaway, top and impact in that order.
    /// `None` when the top's confidence is 0 (its downswing is too short or
    /// too long for a swing), and so for an unconfirmed given impact
    /// ([`ImpactSource::Given`]), which has no top. A low but non-zero top
    /// confidence still gives a tempo; weigh it by
    /// [`SwingEvents::top`]'s confidence.
    #[must_use]
    pub fn from_events(events: &SwingEvents) -> Option<Self> {
        let (k, t, i) = (events.takeaway?, events.top?, events.impact?);
        if t.confidence <= 0.0 {
            return None;
        }
        let backswing_ns = t.t_ns.checked_sub(k.t_ns).filter(|&d| d > 0)?;
        let downswing_ns = i.t_ns.checked_sub(t.t_ns).filter(|&d| d > 0)?;
        // Swing phases last well under 2^52 ns.
        #[allow(clippy::cast_precision_loss, reason = "durations fit f64")]
        let ratio = backswing_ns as f64 / downswing_ns as f64;
        Some(Self {
            backswing_ns,
            downswing_ns,
            ratio,
        })
    }
}

/// Rest before the swing: hands within this many torso lengths...
const STILL_TORSO: f64 = 0.03;
/// ...for this long (s).
const STILL_S: f64 = 0.2;
/// At rest, the hands are at least this many torso lengths below the top
/// (a pause at the top is not address).
const ADDRESS_BELOW_TOP_TORSO: f64 = 0.5;
/// Address ends (and the takeaway starts) when the hands leave their rest
/// position by more than this many torso lengths for good.
const ONSET_TORSO: f64 = 0.015;
/// Walking back from impact, the top is passed once the hands are this many
/// torso lengths below the highest point so far.
const TOP_DROP_TORSO: f64 = 0.3;
/// Fallback top: the last frame before the downswing with the hands within
/// this many torso lengths of their highest point.
const TOP_HOLD_TORSO: f64 = 0.03;
/// The downswing starts where the hands' descent speed passes this
/// fraction of its peak.
const TOP_ONSET_FRACTION: f64 = 0.1;
/// The top is fully clear once it is this many torso lengths above the
/// address (or impact) hands.
const TOP_RISE_TORSO: f64 = 1.0;
/// Downswings (top → impact) this long (s) are plausible...
const DOWNSWING_S: (f64, f64) = (0.12, 0.7);
/// ...and these long (s) are not (the top's confidence ramps to 0).
const DOWNSWING_IMPLAUSIBLE_S: (f64, f64) = (0.06, 1.0);
/// Finish: the hands have settled once their speed is below this fraction
/// of the swing's peak hand speed...
const FINISH_SPEED_FRACTION: f64 = 0.03;
/// ...they stay settled until it is back above this multiple of that
/// fraction...
const FINISH_UNSETTLED: f64 = 2.0;
/// ...and the finish is fully clear when they stay settled this long (s).
const FINISH_HOLD_S: f64 = 0.3;
/// Hand speeds are measured over at least this long (s).
const SPEED_WINDOW_S: f64 = 0.05;
/// An impact is fully clear when the hands drop this many torso lengths
/// from the top and rise as much again in the follow-through.
const IMPACT_SWING_TORSO: f64 = 1.0;
/// ...and when the downswing's peak hand speed reaches this fraction of the
/// clip's peak.
const IMPACT_SPEED_FRACTION: f64 = 0.5;
/// The follow-through window after a low point, as a multiple of the
/// downswing's duration...
const FOLLOW_THROUGH_DOWNSWINGS: f64 = 2.0;
/// ...and at most this long (s).
const FOLLOW_THROUGH_MAX_S: f64 = 0.6;
/// Impact is this long (s) after the hands' low point.
const IMPACT_AFTER_LOW_S: f64 = 0.025;
/// A given impact time is searched this far (s) either side for the
/// hands' low point.
const IMPACT_SEARCH_S: f64 = 0.3;
/// A low point near a given impact time needs at least this clarity
/// (`0..=1`) to replace it.
const IMPACT_MIN_CLEAR: f64 = 0.2;
/// A low point's clarity counts an implausible downswing duration down to
/// this factor (not to 0: on a slowed time base every downswing is long).
const IMPACT_MIN_PLAUSIBILITY: f64 = 0.25;
/// The confidence of a given impact time the hands don't confirm.
const GIVEN_UNCONFIRMED_CONFIDENCE: f64 = 0.5;

/// The lead arm counts as parallel once a crossing of the shoulder's height
/// changes the wrist's height by this many torso lengths per step (full
/// confidence).
const ARM_PARALLEL_CLEAR_TORSO: f64 = 0.02;

/// What [`detect`] knows about the impact.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ImpactHint {
    /// Nothing: estimate it.
    Estimate,
    /// The caller's time (ns): refine it.
    Prior(u64),
    /// Already found (by an earlier pass): keep it.
    Known(SwingEvent, ImpactSource),
}

/// The trajectory the detector works on.
struct Hands<'a> {
    frames: &'a Frames,
    h: Vec<Option<Pt>>,
    t: Vec<f64>,
    /// Torso length (px), the unit of every threshold.
    scale: f64,
    /// The clip's peak hand speed (torso lengths per second).
    peak_speed: f64,
}

impl<'a> Hands<'a> {
    fn new(frames: &'a Frames, scale: f64) -> Self {
        // ns → s for speeds; precision is ample.
        #[allow(clippy::cast_precision_loss, reason = "ns times fit f64")]
        let t: Vec<f64> = frames.t_ns.iter().map(|&t| t as f64 * 1e-9).collect();
        let mut hands = Self {
            frames,
            h: frames.bodies.iter().map(super::body::Body::hands).collect(),
            t,
            scale,
            peak_speed: 0.0,
        };
        hands.peak_speed = (0..hands.len())
            .filter_map(|i| hands.speed(i))
            .fold(0.0, f64::max);
        hands
    }
}

impl Hands<'_> {
    fn len(&self) -> usize {
        self.h.len()
    }

    fn event(frame: usize, t_ns: u64, confidence: f64) -> SwingEvent {
        // Clamped to 0..=1 first.
        #[allow(clippy::cast_possible_truncation, reason = "a 0..=1 value")]
        let confidence = confidence.clamp(0.0, 1.0) as f32;
        SwingEvent {
            t_ns,
            frame,
            confidence,
        }
    }

    fn at_frame(&self, frame: usize, confidence: f64) -> SwingEvent {
        Self::event(frame, self.frames.t_ns[frame], confidence)
    }

    /// The wrists' mean score at a frame.
    fn wrist_score(&self, i: usize) -> f64 {
        let b = &self.frames.bodies[i];
        f64::from(b.score(Joint::LeadWrist) + b.score(Joint::TrailWrist)) / 2.0
    }

    /// The lowest wrist score at a frame and its neighbours: an extremum
    /// beside interpolated hands is no surer than they are.
    fn wrist_score_near(&self, i: usize) -> f64 {
        (i.saturating_sub(1)..=(i + 1).min(self.len() - 1))
            .map(|j| self.wrist_score(j))
            .fold(f64::INFINITY, f64::min)
    }

    /// The frames either side of `i` spanning at least [`SPEED_WINDOW_S`]
    /// (at least one frame each way, within the clip).
    fn window(&self, i: usize) -> (usize, usize) {
        let half = SPEED_WINDOW_S / 2.0;
        let mut a = i.saturating_sub(1);
        while a > 0 && self.t[i] - self.t[a] < half {
            a -= 1;
        }
        let mut b = (i + 1).min(self.len() - 1);
        while b + 1 < self.len() && self.t[b] - self.t[i] < half {
            b += 1;
        }
        (a, b)
    }

    /// Hand speed at a frame, torso lengths per second.
    fn speed(&self, i: usize) -> Option<f64> {
        let (a, b) = self.window(i);
        let (pa, pb) = (self.h[a]?, self.h[b]?);
        let dt = self.t[b] - self.t[a];
        (dt > 0.0).then(|| pa.dist(pb) / dt / self.scale)
    }

    /// The hands' downward speed at a frame (negative rising), torso
    /// lengths per second.
    fn descent(&self, i: usize) -> Option<f64> {
        let (a, b) = self.window(i);
        let dt = self.t[b] - self.t[a];
        (dt > 0.0).then_some((self.y(b)? - self.y(a)?) / dt / self.scale)
    }

    fn y(&self, i: usize) -> Option<f64> {
        self.h[i].map(|p| p.y)
    }

    /// Whether frame `i` is a low point of the hands (a local maximum of
    /// image `y`).
    fn is_low(&self, i: usize) -> bool {
        if i == 0 || i + 1 >= self.len() {
            return false;
        }
        matches!(
            (self.y(i - 1), self.y(i), self.y(i + 1)),
            (Some(a), Some(y), Some(c)) if y >= a && y >= c
        )
    }

    /// The time of the extremum of `y` at frame `i`, refined with a
    /// parabola through its neighbours (the frame's own time where `i` is
    /// not an extremum).
    fn refine(&self, i: usize) -> u64 {
        let t_ns = self.frames.t_ns[i];
        if i == 0 || i + 1 >= self.len() {
            return t_ns;
        }
        let (Some(a), Some(b), Some(c)) = (self.y(i - 1), self.y(i), self.y(i + 1)) else {
            return t_ns;
        };
        let curvature = a - 2.0 * b + c;
        if curvature.abs() < 1e-9 || (b - a) * (b - c) < 0.0 {
            return t_ns;
        }
        let delta = (0.5 * (a - c) / curvature).clamp(-0.5, 0.5);
        let half_step = (self.t[i + 1] - self.t[i - 1]) / 2.0;
        Self::ns(self.t[i] + delta * half_step)
    }

    fn ns(s: f64) -> u64 {
        // Track times are non-negative ns well below 2^63.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "non-negative ns"
        )]
        let ns = (s.max(0.0) * 1e9).round() as u64;
        ns
    }

    /// Walking back from `end` (exclusive), the hands' highest point until
    /// they have dropped well below it again.
    fn highest_before(&self, end: usize) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for i in (0..end).rev() {
            let Some(y) = self.y(i) else { continue };
            match best {
                Some((_, top)) if y > top + TOP_DROP_TORSO * self.scale => break,
                Some((_, top)) if y >= top => {}
                _ => best = Some((i, y)),
            }
        }
        best.map(|(i, _)| i)
    }

    /// The top of the backswing that ends at `end` (the impact frame) and
    /// its time.
    fn top_before(&self, end: usize) -> Option<(usize, u64)> {
        // Back to the low point when `end` is past it, so the hands still
        // falling towards it don't end the backswing.
        let mut low = end;
        while low > 0
            && self.t[end] - self.t[low - 1] <= IMPACT_SEARCH_S
            && matches!((self.y(low - 1), self.y(low)), (Some(a), Some(b)) if a > b)
        {
            low -= 1;
        }
        let highest = self.highest_before(low)?;
        if let Some(top) = self.downswing_onset(highest, low) {
            return Some(top);
        }
        let top_y = self.y(highest)?;
        let i = (highest..low)
            .rev()
            .find(|&i| {
                self.y(i)
                    .is_some_and(|y| y <= top_y + TOP_HOLD_TORSO * self.scale)
            })
            .unwrap_or(highest);
        Some((i, self.refine(i)))
    }

    /// The downswing's start between the highest point and the low point,
    /// and its time: walking back from the fastest descent, where the hands'
    /// descent speed falls below [`TOP_ONSET_FRACTION`] of it (between
    /// frames, never before the highest point's sub-frame time).
    fn downswing_onset(&self, highest: usize, low: usize) -> Option<(usize, u64)> {
        let (fastest, peak) = (highest..=low)
            .filter_map(|i| Some((i, self.descent(i)?)))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        if peak <= 0.0 {
            return None;
        }
        let level = TOP_ONSET_FRACTION * peak;
        let i = (highest..=fastest)
            .rev()
            .find(|&i| self.descent(i).is_some_and(|d| d < level))?;
        let (d0, t0) = (self.descent(i)?, self.t[i]);
        let t_ns = match self.descent(i + 1) {
            Some(d1) if i < fastest && d1 > d0 => {
                Self::ns(t0 + (level - d0) / (d1 - d0) * (self.t[i + 1] - t0))
            }
            _ => self.frames.t_ns[i],
        }
        .max(self.refine(highest));
        let frame = self.frames.nearest(t_ns).unwrap_or(i).clamp(highest, low);
        Some((frame, t_ns))
    }

    /// How clearly the hands swing through the low point at `i`, `0..=1`:
    /// the smaller of the drop from the top before it and the rise in the
    /// follow-through after it (in [`IMPACT_SWING_TORSO`]s, at most 1),
    /// times the downswing's peak hand speed as a fraction of
    /// [`IMPACT_SPEED_FRACTION`] of the clip's peak (at most 1), times how
    /// plausible the downswing's duration is (at least
    /// [`IMPACT_MIN_PLAUSIBILITY`]). `None` unless `i` is a low point after
    /// a top.
    fn swing_through(&self, i: usize) -> Option<f64> {
        if !self.is_low(i) {
            return None;
        }
        let y = self.y(i)?;
        let top = self.highest_before(i)?;
        let top_y = self.y(top)?;
        // The downswing runs from its onset (a hold at the top is not part
        // of it).
        // Track times fit f64.
        #[allow(clippy::cast_precision_loss, reason = "ns times fit f64")]
        let onset = self
            .downswing_onset(top, i)
            .map_or(self.t[top], |(_, t)| t as f64 * 1e-9);
        let downswing = self.t[i] - onset;
        let horizon = self.t[i] + (FOLLOW_THROUGH_DOWNSWINGS * downswing).min(FOLLOW_THROUGH_MAX_S);
        let follow_y = (i + 1..self.len())
            .take_while(|&j| self.t[j] <= horizon)
            .filter_map(|j| self.y(j))
            .min_by(f64::total_cmp)?;
        let swing = ((y - top_y).min(y - follow_y) / (IMPACT_SWING_TORSO * self.scale)).max(0.0);
        let fastest = (top..=i).filter_map(|j| self.speed(j)).fold(0.0, f64::max);
        let speed = if self.peak_speed > 0.0 {
            fastest / (IMPACT_SPEED_FRACTION * self.peak_speed)
        } else {
            0.0
        };
        let plausible = downswing_plausibility(downswing).max(IMPACT_MIN_PLAUSIBILITY);
        Some(swing.min(1.0) * speed.min(1.0) * plausible)
    }

    /// The impact 25 ms after the low point at `low`.
    fn impact_event(&self, low: usize, clear: f64) -> SwingEvent {
        // Track times fit f64.
        #[allow(clippy::cast_precision_loss, reason = "ns times fit f64")]
        let t = self.refine(low) as f64 * 1e-9 + IMPACT_AFTER_LOW_S;
        let t_ns = Self::ns(t);
        let frame = self.frames.nearest(t_ns).unwrap_or(low);
        Self::event(frame, t_ns, self.wrist_score_near(low) * clear)
    }

    /// Estimates impact: the low point with the clearest swing through it,
    /// and that clarity.
    fn estimate_impact(&self) -> Option<(usize, f64)> {
        (1..self.len().saturating_sub(1))
            .filter_map(|i| Some((i, self.swing_through(i)?)))
            .fold(None, |best: Option<(usize, f64)>, (i, s)| {
                if best.is_none_or(|(_, b)| s > b) {
                    Some((i, s))
                } else {
                    best
                }
            })
    }

    /// Refines a given impact (`frame` nearest `t_ns`): of the low points
    /// within [`IMPACT_SEARCH_S`] of it, or reached by the hands still
    /// descending out of that window, the clearest, weighted down the
    /// further it is from the given time. Returns the low point and its
    /// clarity.
    fn refine_given(&self, frame: usize, t_ns: u64) -> Option<(usize, f64)> {
        // Track times fit f64.
        #[allow(clippy::cast_precision_loss, reason = "ns times fit f64")]
        let given = t_ns as f64 * 1e-9;
        let inside = |i: usize| (self.t[i] - given).abs() <= IMPACT_SEARCH_S;
        let first = (0..=frame).rev().take_while(|&i| inside(i)).last()?;
        let last = (frame..self.len()).take_while(|&i| inside(i)).last()?;
        let downhill = |mut i: usize, forwards: bool| {
            loop {
                let next = if forwards {
                    (i + 1 < self.len()).then_some(i + 1)
                } else {
                    i.checked_sub(1)
                };
                match next.map(|n| (n, self.y(n), self.y(i))) {
                    Some((n, Some(a), Some(b))) if a > b => i = n,
                    _ => return i,
                }
            }
        };
        let candidates = (first..=last)
            .filter(|&i| self.is_low(i))
            .chain([downhill(first, false), downhill(last, true)]);
        let mut best: Option<(usize, f64, f64)> = None;
        for i in candidates {
            let Some(clear) = self.swing_through(i) else {
                continue;
            };
            let off = ((self.t[i] - given).abs() / IMPACT_SEARCH_S).min(1.0);
            let score = clear * (1.0 - 0.5 * off * off);
            if best.is_none_or(|(_, _, s)| score > s) {
                best = Some((i, clear, score));
            }
        }
        best.filter(|&(_, clear, _)| clear >= IMPACT_MIN_CLEAR)
            .map(|(i, clear, _)| (i, clear))
    }
}

/// `0..=1`: how plausible a downswing (top → impact) of `s` seconds is.
fn downswing_plausibility(s: f64) -> f64 {
    let ((lo, hi), (lo0, hi0)) = (DOWNSWING_S, DOWNSWING_IMPLAUSIBLE_S);
    if s < lo {
        ((s - lo0) / (lo - lo0)).clamp(0.0, 1.0)
    } else if s > hi {
        ((hi0 - s) / (hi0 - hi)).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Finds the events.
pub(crate) fn detect(frames: &Frames, hint: ImpactHint) -> SwingEvents {
    let mut events = SwingEvents {
        address: None,
        takeaway: None,
        arm_parallel_back: None,
        top: None,
        arm_parallel_down: None,
        impact: None,
        impact_source: match hint {
            ImpactHint::Estimate => ImpactSource::Estimated,
            ImpactHint::Prior(_) => ImpactSource::Given,
            ImpactHint::Known(_, source) => source,
        },
        finish: None,
    };
    match hint {
        ImpactHint::Prior(t) => {
            events.impact = frames.nearest(t).map(|frame| SwingEvent {
                t_ns: t,
                frame,
                // A 0..=1 constant.
                #[allow(clippy::cast_possible_truncation, reason = "a 0..=1 value")]
                confidence: GIVEN_UNCONFIRMED_CONFIDENCE as f32,
            });
        }
        ImpactHint::Known(e, _) => events.impact = Some(e),
        ImpactHint::Estimate => {}
    }
    let Some(scale) = body_scale(frames) else {
        return events;
    };
    let hands = Hands::new(frames, scale);
    if hands.len() < 3 {
        return events;
    }

    // The frame the backswing is searched back from.
    let end = match (hint, events.impact) {
        (ImpactHint::Estimate, _) => {
            let Some((low, clear)) = hands.estimate_impact() else {
                return events;
            };
            events.impact = Some(hands.impact_event(low, clear));
            low
        }
        (ImpactHint::Prior(t), Some(given)) => {
            // An unconfirmed time is no impact to measure the swing from.
            let Some((low, clear)) = hands.refine_given(given.frame, t) else {
                return events;
            };
            events.impact = Some(hands.impact_event(low, clear));
            events.impact_source = ImpactSource::Refined;
            low
        }
        // An unconfirmed given impact (from the first pass), or none.
        (ImpactHint::Known(_, ImpactSource::Given), _) | (_, None) => return events,
        (_, Some(known)) => known.frame,
    };
    let Some(impact) = events.impact else {
        return events;
    };

    let top = hands.top_before(end);
    let address = top.and_then(|(top, _)| hands.address_before(top));
    if let Some((a, k, rest)) = address {
        events.address = Some(hands.at_frame(a, hands.wrist_score(a) * rest));
        events.takeaway = Some(hands.at_frame(k, hands.wrist_score(k) * rest));
    }
    // The hands' height at address, else at impact: the top's baseline.
    let base_y = address
        .and_then(|(a, _, _)| hands.y(a))
        .or_else(|| hands.y(end));
    if let Some((top, t_ns)) = top {
        let rise = match (base_y, hands.y(top)) {
            (Some(b), Some(y)) => ((b - y) / (TOP_RISE_TORSO * scale)).clamp(0.0, 1.0),
            _ => 0.0,
        };
        // ns durations fit f64.
        #[allow(clippy::cast_precision_loss, reason = "ns durations fit f64")]
        let downswing = (impact.t_ns as f64 - t_ns as f64) * 1e-9;
        events.top = Some(Hands::event(
            top,
            t_ns,
            hands.wrist_score_near(top) * rise * downswing_plausibility(downswing),
        ));
    }
    if let Some((top, _)) = top {
        let from = events.takeaway.map_or(0, |e| e.frame);
        events.arm_parallel_back = top
            .checked_sub(1)
            .and_then(|last| hands.arm_parallel(from, last, true));
        events.arm_parallel_down = hands.arm_parallel(top + 1, impact.frame.max(end), false);
    }
    events.finish = hands.finish_after(top.map_or(end, |(t, _)| t), impact.frame.max(end));
    events
}

/// The lead wrist's height above the lead shoulder at a frame (pixels,
/// positive with the wrist higher) and the two keypoints' mean score, each
/// halved where it falls back (the hands for the wrist, the neck for the
/// shoulder).
fn lead_arm_height(body: &Body) -> Option<(f64, f64)> {
    let (wrist, wrist_score) = match body.at(Joint::LeadWrist) {
        Some(p) => (p, f64::from(body.score(Joint::LeadWrist))),
        None => (
            body.at(Joint::TrailWrist)?,
            0.5 * f64::from(body.score(Joint::TrailWrist)),
        ),
    };
    let (shoulder, shoulder_score) = match body.at(Joint::LeadShoulder) {
        Some(p) => (p, f64::from(body.score(Joint::LeadShoulder))),
        None => (
            body.at(Joint::Neck)?,
            0.5 * f64::from(body.score(Joint::Neck)),
        ),
    };
    Some((
        shoulder.y - wrist.y,
        f64::midpoint(wrist_score, shoulder_score),
    ))
}

impl Hands<'_> {
    /// The last time between frames `from` and `to` (inclusive) that the
    /// lead wrist rises through the lead shoulder's height (`up`), or the
    /// first time it falls through it, interpolated between the frames
    /// either side (across frames without the keypoints).
    fn arm_parallel(&self, from: usize, to: usize, up: bool) -> Option<SwingEvent> {
        let to = to.min(self.len().saturating_sub(1));
        let mut prev: Option<(usize, f64, f64)> = None;
        let mut found = None;
        for i in from..=to {
            let Some((a, score)) = lead_arm_height(&self.frames.bodies[i]) else {
                continue;
            };
            if let Some((j, a0, score0)) = prev {
                let crossed = if up {
                    a0 < 0.0 && a >= 0.0
                } else {
                    a0 >= 0.0 && a < 0.0
                };
                if crossed {
                    let step = a - a0;
                    let u = (-a0 / step).clamp(0.0, 1.0);
                    let t_ns = Self::ns(self.t[j] + u * (self.t[i] - self.t[j]));
                    let frame = self.frames.nearest(t_ns).unwrap_or(i).clamp(j, i);
                    let clear = (step.abs() / (ARM_PARALLEL_CLEAR_TORSO * self.scale)).min(1.0);
                    found = Some(Self::event(
                        frame,
                        t_ns,
                        f64::midpoint(score0, score) * clear,
                    ));
                    if !up {
                        break;
                    }
                }
            }
            prev = Some((i, a, score));
        }
        found
    }
}

impl Hands<'_> {
    /// How long (s) the hands stay within `tolerance` torso lengths of
    /// frame `from`'s position walking back, up to `duration`: `Some(1.0)`
    /// when they rest for the whole duration, `Some(fraction)` when the
    /// clip starts first (after at least two frames at rest), `None` when
    /// they move or vanish.
    fn rest_before(&self, from: usize, duration: f64, tolerance: f64) -> Option<f64> {
        let origin = self.h[from]?;
        let limit = tolerance * self.scale;
        for i in (0..from).rev() {
            if self.h[i]?.dist(origin) >= limit {
                return None;
            }
            let covered = self.t[from] - self.t[i];
            if covered >= duration {
                return Some(1.0);
            }
            if i == 0 {
                return (from >= 2).then(|| covered / duration);
            }
        }
        None
    }

    /// The address and takeaway frames before the top, and how much of the
    /// rest before address the clip shows (`0..=1`): walking back from the
    /// top, the first moment the hands were at rest well below it; address
    /// is the last frame they are still near that rest position, the
    /// takeaway the next.
    fn address_before(&self, top: usize) -> Option<(usize, usize, f64)> {
        let low = self.y(top)? + ADDRESS_BELOW_TOP_TORSO * self.scale;
        let (anchor, seen) = (0..top).rev().find_map(|i| {
            if !self.y(i).is_some_and(|y| y > low) {
                return None;
            }
            Some((i, self.rest_before(i, STILL_S, STILL_TORSO)?))
        })?;
        let start = (0..anchor)
            .rev()
            .find(|&i| self.t[anchor] - self.t[i] >= STILL_S)
            .unwrap_or(0);
        let window: Vec<Pt> = self.h[start..=anchor].iter().flatten().copied().collect();
        // A handful of frames.
        #[allow(clippy::cast_precision_loss, reason = "a short window")]
        let n = window.len() as f64;
        let rest = Pt::new(
            window.iter().map(|p| p.x).sum::<f64>() / n,
            window.iter().map(|p| p.y).sum::<f64>() / n,
        );
        let address = (start..top)
            .rev()
            .find(|&i| self.h[i].is_some_and(|p| p.dist(rest) < ONSET_TORSO * self.scale))?;
        Some((address, (address + 1).min(top), 0.5 + 0.5 * seen))
    }

    /// The finish: after impact, once the hands are high (above mid-torso)
    /// and past their highest point, the first frame where their speed
    /// falls below a small fraction of the swing's peak.
    fn finish_after(&self, top: usize, impact: usize) -> Option<SwingEvent> {
        let high = |i: usize| -> bool {
            let b = &self.frames.bodies[i];
            match (self.h[i], b.shoulders(), b.pelvis()) {
                (Some(h), Some(s), Some(p)) => h.y < f64::midpoint(s.y, p.y),
                _ => false,
            }
        };
        let first_high = (impact + 1..self.len()).find(|&i| high(i))?;
        let end = (first_high..self.len())
            .find(|&i| !high(i))
            .unwrap_or(self.len());
        let highest = (first_high..end)
            .filter_map(|i| Some((i, self.y(i)?)))
            .min_by(|a, b| a.1.total_cmp(&b.1))?
            .0;
        let peak = (top..first_high)
            .filter_map(|i| self.speed(i))
            .fold(0.0, f64::max);
        let calm = |i: usize| {
            self.speed(i)
                .is_some_and(|v| v < FINISH_SPEED_FRACTION * peak)
        };
        let Some(settled) = (highest..end).find(|&i| calm(i)) else {
            return Some(self.at_frame(highest, 0.5 * self.wrist_score(highest)));
        };
        // Settled until the speed is back above twice the threshold.
        let until = (settled..self.len())
            .find(|&i| {
                self.speed(i)
                    .is_none_or(|v| v >= FINISH_SPEED_FRACTION * FINISH_UNSETTLED * peak)
            })
            .unwrap_or(self.len() - 1);
        let held = ((self.t[until] - self.t[settled]) / FINISH_HOLD_S).min(1.0);
        Some(self.at_frame(settled, self.wrist_score(settled) * held))
    }
}

/// The typical torso length (px) over the clip: the median over frames
/// where it is visible.
pub(crate) fn body_scale(frames: &Frames) -> Option<f64> {
    let mut v: Vec<f64> = frames
        .bodies
        .iter()
        .filter_map(super::body::Body::torso)
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}
