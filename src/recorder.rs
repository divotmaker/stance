//! Cutting clips around shot triggers.
//!
//! [`ClipRecorder`] is pure and deterministic: it is fed frames and triggers
//! with their host times and hands back finished clips. It keeps its own
//! pre-roll [`Ring`], so it needs nothing but the frames, in capture order.
//!
//! The clip rules and the caller's duties are documented on [`ClipRecorder`].

use std::time::Duration;

use crate::Frame;
use crate::frame::missing_between;
use crate::ring::{Ring, duration_ns};

/// Recorder settings.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RecorderConfig {
    /// Time kept before the impact estimate. Default 4 s.
    pub pre_roll: Duration,
    /// Time kept after the latest trigger. Default 1.5 s.
    pub post_roll: Duration,
    /// Largest gap between two sources' impact estimates for one swing.
    /// Default 500 ms.
    pub merge_window: Duration,
    /// How much history the pre-roll ring keeps. Must be at least `pre_roll`
    /// plus the largest impact lead, or clips start late (see
    /// [`RecorderConfig::buffer_covers`] and [`Clip::short_buffer`]).
    /// Default 10 s.
    pub buffer: Duration,
    /// Byte cap of the pre-roll ring. Default 512 MiB.
    pub max_bytes: usize,
    /// How long [`ClipRecorder::poll`] waits past a clip's window for a frame
    /// before finishing it as incomplete. Default 1 s.
    pub stall: Duration,
    /// Nominal frame rate, used to count dropped frames. `None` estimates the
    /// rate from each clip's own frame spacing.
    pub fps: Option<f64>,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            pre_roll: Duration::from_secs(4),
            post_roll: Duration::from_millis(1500),
            merge_window: Duration::from_millis(500),
            buffer: Duration::from_secs(10),
            max_bytes: 512 << 20,
            stall: Duration::from_secs(1),
            fps: None,
        }
    }
}

impl RecorderConfig {
    /// True if `buffer` holds the whole pre-roll for a trigger with this
    /// impact lead (`buffer ≥ pre_roll + impact_lead`).
    ///
    /// ```
    /// use std::time::Duration;
    /// use stance::RecorderConfig;
    /// let config = RecorderConfig::default(); // 4 s pre-roll, 10 s buffer
    /// assert!(config.buffer_covers(Duration::from_secs(3)));
    /// assert!(!config.buffer_covers(Duration::from_secs(7)));
    /// ```
    #[must_use]
    pub fn buffer_covers(&self, impact_lead: Duration) -> bool {
        self.buffer >= self.pre_roll.saturating_add(impact_lead)
    }
}

/// A shot trigger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trigger<T> {
    /// When the trigger arrived, on the host monotonic clock (`clock::now_ns`).
    pub at_ns: u64,
    /// How long before `at_ns` the impact happened (0 for monitors that
    /// trigger at impact).
    pub impact_lead_ns: u64,
    /// Who triggered (a launch monitor id). Decides joining and truncation.
    pub source: String,
    /// Caller data carried into the clip (a shot key).
    pub tag: T,
    /// Overrides the recorder's pre-roll for this trigger.
    pub pre_roll: Option<Duration>,
    /// Overrides the recorder's post-roll for this trigger.
    pub post_roll: Option<Duration>,
}

impl<T> Trigger<T> {
    /// A trigger at `at_ns` with no impact lead and the recorder's rolls.
    #[must_use]
    pub fn new(at_ns: u64, source: impl Into<String>, tag: T) -> Self {
        Self {
            at_ns,
            impact_lead_ns: 0,
            source: source.into(),
            tag,
            pre_roll: None,
            post_roll: None,
        }
    }

    /// Sets how long before the trigger the impact happened.
    #[must_use]
    pub fn with_impact_lead(mut self, lead: Duration) -> Self {
        self.impact_lead_ns = duration_ns(lead);
        self
    }

    /// Overrides the pre-roll for this trigger.
    #[must_use]
    pub fn with_pre_roll(mut self, pre_roll: Duration) -> Self {
        self.pre_roll = Some(pre_roll);
        self
    }

    /// Overrides the post-roll for this trigger.
    #[must_use]
    pub fn with_post_roll(mut self, post_roll: Duration) -> Self {
        self.post_roll = Some(post_roll);
        self
    }

    /// The impact estimate, `at_ns − impact_lead_ns`.
    #[must_use]
    pub fn impact_ns(&self) -> u64 {
        self.at_ns.saturating_sub(self.impact_lead_ns)
    }
}

/// Identifies a clip within one recorder. Increasing from 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClipId(pub u64);

/// What a trigger did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerOutcome {
    /// A new clip started.
    Started(ClipId),
    /// The trigger joined an open clip from another source.
    Joined(ClipId),
    /// The trigger came from a source already in `previous`, whose window
    /// was still running; `previous` now ends at the trigger's time and
    /// `clip` started.
    Truncated {
        /// The clip that was cut short.
        previous: ClipId,
        /// The clip that started.
        clip: ClipId,
    },
}

/// A finished clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip<T> {
    /// The clip's id.
    pub id: ClipId,
    /// Tags of every trigger in the clip, in trigger order.
    pub tags: Vec<T>,
    /// The frames, oldest first. Never empty.
    pub frames: Vec<Frame>,
    /// Index of the first frame at or after the first trigger, or
    /// `frames.len()` when frames stopped between the impact and the trigger.
    pub trigger_index: usize,
    /// Index of the first frame at or after the first trigger's impact
    /// estimate. Always a valid index.
    pub impact_index: usize,
    /// Host time of the first trigger.
    pub trigger_ns: u64,
    /// The first trigger's impact estimate.
    pub impact_ns: u64,
    /// The whole window was captured: frames past it arrived, and no frames
    /// are missing across the trigger or the impact. `false` when frames
    /// stopped first or a gap spans either instant.
    pub complete: bool,
    /// Frames missing inside the clip, estimated from gaps in the frame times
    /// (device times when the frames carry them).
    pub dropped: u32,
    /// A later trigger from the same source cut the clip short.
    pub truncated: bool,
    /// The recorder's `buffer` was shorter than a trigger's pre-roll plus
    /// impact lead, so frames at the start of the window may be missing.
    /// Raise [`RecorderConfig::buffer`].
    pub short_buffer: bool,
    /// The recorder's nominal frame rate, if it was given one.
    pub fps: Option<f64>,
}

impl<T> Clip<T> {
    /// Time from the first to the last frame.
    #[must_use]
    pub fn duration(&self) -> Duration {
        match (self.frames.first(), self.frames.last()) {
            (Some(a), Some(b)) => Duration::from_nanos(b.host_ns.saturating_sub(a.host_ns)),
            _ => Duration::ZERO,
        }
    }

    /// Frames per second over the clip, from its first and last frame times.
    #[must_use]
    pub fn measured_fps(&self) -> Option<f64> {
        let span = self.duration().as_secs_f64();
        // Frame counts are far below 2^52.
        #[allow(clippy::cast_precision_loss)]
        let intervals = self.frames.len().saturating_sub(1) as f64;
        (span > 0.0).then(|| intervals / span)
    }

    /// The frame at the first trigger, if frames reached it.
    #[must_use]
    pub fn trigger_frame(&self) -> Option<&Frame> {
        self.frames.get(self.trigger_index)
    }

    /// The frame at the impact estimate.
    #[must_use]
    pub fn impact_frame(&self) -> Option<&Frame> {
        self.frames.get(self.impact_index)
    }
}

/// A clip that was dropped without being finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discarded<T> {
    /// The clip's id.
    pub id: ClipId,
    /// Tags of every trigger in the clip.
    pub tags: Vec<T>,
    /// Why.
    pub reason: DiscardReason,
}

/// Why a clip was discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiscardReason {
    /// No frame of the window arrived at all.
    NoFrames,
    /// Frames stopped (or a gap ran past the window) before the impact
    /// estimate's frame arrived.
    NoImpactFrame,
}

/// A clip leaving the recorder.
#[derive(Debug, Clone, PartialEq)]
pub enum RecorderEvent<T> {
    /// A finished clip.
    Finished(Clip<T>),
    /// A clip dropped for lack of frames.
    Discarded(Discarded<T>),
}

#[derive(Debug)]
struct OpenClip<T> {
    id: ClipId,
    tags: Vec<T>,
    sources: Vec<String>,
    trigger_ns: u64,
    impact_ns: u64,
    start_ns: u64,
    end_ns: u64,
    truncated: bool,
    short_buffer: bool,
    frames: Vec<Frame>,
}

/// Cuts clips around triggers. `T` is the per-trigger tag (e.g. a shot key).
///
/// # Rules
///
/// - A trigger's **impact estimate** is `at_ns − impact_lead_ns`: launch
///   monitors that report after the ball is gone trigger late, and the lead
///   moves the clip back to the swing.
/// - A clip's **window** is `[impact − pre_roll, latest trigger + post_roll]`.
/// - A trigger from a source that already has a trigger in an open clip
///   **truncates** that clip at the trigger's time, if its window is still
///   running then, and starts a new clip. A clip whose window already ended
///   is left alone and finishes normally.
/// - Otherwise a trigger **joins** the newest open, untruncated clip that has
///   no trigger from the same source and whose impact estimate is within
///   `merge_window` of the trigger's (two launch monitors seeing one swing);
///   its tag is added. Every open clip is considered, newest first.
/// - Otherwise a new clip **starts**. Overlapping clips share frames.
/// - A clip **finishes** when a frame past its window arrives, or through
///   [`ClipRecorder::poll`] / [`ClipRecorder::flush`] when frames stop.
///   A finished clip holds its **impact frame** (the first frame at or after
///   the impact estimate); a clip without one is discarded and reported as
///   [`RecorderEvent::Discarded`].
/// - A clip is **complete** when frames past its window arrived and no frames
///   are missing across its trigger or impact. It may lack a trigger frame
///   (frames stopped between impact and a late trigger); `trigger_index` is
///   then `frames.len()`.
///
/// # Caller duties
///
/// - Call [`ClipRecorder::poll`] on a timer (every few hundred ms), not only
///   when frames arrive: when a camera stops, no frame comes to finish the
///   open clips, and they stay open until `poll` (or `flush`) finishes them.
/// - Deduplicate triggers. The recorder treats every trigger as a new event:
///   the same source triggering twice for one shot truncates the first clip
///   and starts a second one.
///
/// # Example
///
/// ```
/// # use std::sync::Arc;
/// # use std::time::Duration;
/// use stance::{ClipRecorder, Frame, RecorderConfig, RecorderEvent, Trigger, TriggerOutcome};
///
/// let mut config = RecorderConfig::default();
/// config.pre_roll = Duration::from_millis(200);
/// config.post_roll = Duration::from_millis(100);
/// let mut rec = ClipRecorder::new(config);
/// let frame = |seq: u64| Frame {
///     jpeg: Arc::from(vec![0u8; 16]),
///     host_ns: seq * 10_000_000, // 100 fps
///     device_ns: None,
///     seq,
///     width: 8,
///     height: 8,
/// };
/// let mut clips = Vec::new();
/// for seq in 0..100 {
///     if seq == 50 {
///         let outcome = rec.trigger(Trigger::new(500_000_000, "radar", "shot 1"));
///         assert!(matches!(outcome, TriggerOutcome::Started(_)));
///     }
///     clips.extend(rec.push(frame(seq)));
/// }
/// let [RecorderEvent::Finished(clip)] = clips.as_slice() else { panic!() };
/// assert_eq!(clip.tags, ["shot 1"]);
/// assert_eq!(clip.frames.len(), 31); // 300..=600 ms
/// assert_eq!(clip.frames[clip.trigger_index].host_ns, 500_000_000);
/// assert!(clip.complete);
/// ```
#[derive(Debug)]
pub struct ClipRecorder<T> {
    config: RecorderConfig,
    ring: Ring,
    open: Vec<OpenClip<T>>,
    next_id: u64,
}

impl<T> ClipRecorder<T> {
    /// A recorder with no frames and no open clips.
    #[must_use]
    pub fn new(config: RecorderConfig) -> Self {
        Self {
            ring: Ring::new(config.buffer, config.max_bytes),
            config,
            open: Vec::new(),
            next_id: 0,
        }
    }

    /// The current settings.
    #[must_use]
    pub fn config(&self) -> &RecorderConfig {
        &self.config
    }

    /// Replaces the settings. Rolls apply to later triggers; the ring bounds
    /// apply now.
    pub fn set_config(&mut self, config: RecorderConfig) {
        self.ring.set_limits(config.buffer, config.max_bytes);
        self.config = config;
    }

    /// Sets the pre-roll for later triggers.
    pub fn set_pre_roll(&mut self, pre_roll: Duration) {
        self.config.pre_roll = pre_roll;
    }

    /// Sets the post-roll for later triggers.
    pub fn set_post_roll(&mut self, post_roll: Duration) {
        self.config.post_roll = post_roll;
    }

    /// The pre-roll ring.
    #[must_use]
    pub fn ring(&self) -> &Ring {
        &self.ring
    }

    /// Number of clips waiting for frames. Without frames they finish only
    /// through [`ClipRecorder::poll`] or [`ClipRecorder::flush`].
    #[must_use]
    pub fn open_clips(&self) -> usize {
        self.open.len()
    }

    /// Applies a trigger (see [`ClipRecorder`] for the rules). Every trigger
    /// counts: deduplicate repeated triggers for one shot first.
    pub fn trigger(&mut self, t: Trigger<T>) -> TriggerOutcome {
        let impact = t.impact_ns();
        let pre = duration_ns(t.pre_roll.unwrap_or(self.config.pre_roll));
        let post = duration_ns(t.post_roll.unwrap_or(self.config.post_roll));
        let end = t.at_ns.saturating_add(post);

        // The source's clip ends here if its window is still running; one
        // whose window already ended finishes on its own.
        let mut previous = None;
        for c in self.open.iter_mut().rev() {
            if c.truncated || !c.sources.contains(&t.source) {
                continue;
            }
            let cut = t.at_ns.max(c.trigger_ns);
            if cut < c.end_ns {
                c.end_ns = cut;
                c.truncated = true;
                c.frames.retain(|f| f.host_ns <= cut);
                previous.get_or_insert(c.id);
            }
        }
        if let Some(previous) = previous {
            let clip = self.start(t, impact, pre, end);
            return TriggerOutcome::Truncated { previous, clip };
        }

        let merge = duration_ns(self.config.merge_window);
        let buffer = duration_ns(self.config.buffer);
        if let Some(c) = self.open.iter_mut().rev().find(|c| {
            !c.truncated && !c.sources.contains(&t.source) && c.impact_ns.abs_diff(impact) <= merge
        }) {
            let start = impact.saturating_sub(pre);
            if start < c.start_ns {
                let mut earlier = self.ring.range(start, c.start_ns.saturating_sub(1));
                earlier.append(&mut c.frames);
                c.frames = earlier;
                c.start_ns = start;
                c.short_buffer |= t.at_ns.saturating_sub(start) > buffer;
            }
            c.end_ns = c.end_ns.max(end);
            c.tags.push(t.tag);
            c.sources.push(t.source);
            return TriggerOutcome::Joined(c.id);
        }

        TriggerOutcome::Started(self.start(t, impact, pre, end))
    }

    fn start(&mut self, t: Trigger<T>, impact: u64, pre: u64, end: u64) -> ClipId {
        let id = ClipId(self.next_id);
        self.next_id += 1;
        let start = impact.saturating_sub(pre);
        self.open.push(OpenClip {
            id,
            tags: vec![t.tag],
            sources: vec![t.source],
            trigger_ns: t.at_ns,
            impact_ns: impact,
            start_ns: start,
            end_ns: end,
            truncated: false,
            short_buffer: t.at_ns.saturating_sub(start) > duration_ns(self.config.buffer),
            frames: self.ring.range(start, end),
        });
        id
    }

    /// Feeds the next frame. Returns the clips this frame finished.
    ///
    /// Frames must arrive in capture order; a frame not newer than the newest
    /// one seen is ignored.
    pub fn push(&mut self, frame: Frame) -> Vec<RecorderEvent<T>> {
        if self
            .ring
            .newest()
            .is_some_and(|n| frame.host_ns <= n.host_ns)
        {
            return Vec::new();
        }
        let mut done = Vec::new();
        let mut i = 0;
        while i < self.open.len() {
            let c = &mut self.open[i];
            if frame.host_ns > c.end_ns {
                let c = self.open.remove(i);
                done.push(self.finish(c, true));
                continue;
            }
            if frame.host_ns >= c.start_ns {
                c.frames.push(frame.clone());
            }
            i += 1;
        }
        self.ring.push(frame);
        done
    }

    /// Finishes clips whose window ended at least `stall` before `now_ns`
    /// (the camera stopped delivering), and clips a late trigger placed
    /// entirely in the past. Call it on a timer: when frames stop, nothing
    /// else finishes the open clips.
    pub fn poll(&mut self, now_ns: u64) -> Vec<RecorderEvent<T>> {
        let stall = duration_ns(self.config.stall);
        let mut done = Vec::new();
        let mut i = 0;
        while i < self.open.len() {
            let c = &self.open[i];
            if self.captured_past(c) || now_ns >= c.end_ns.saturating_add(stall) {
                let c = self.open.remove(i);
                let past = self.captured_past(&c);
                done.push(self.finish(c, past));
            } else {
                i += 1;
            }
        }
        done
    }

    /// Finishes every open clip now (the source is gone). Clips whose window
    /// was fully captured are complete; the rest are not.
    pub fn flush(&mut self) -> Vec<RecorderEvent<T>> {
        let open = std::mem::take(&mut self.open);
        open.into_iter()
            .map(|c| {
                let past = self.captured_past(&c);
                self.finish(c, past)
            })
            .collect()
    }

    /// A frame past the clip's window has been seen.
    fn captured_past(&self, c: &OpenClip<T>) -> bool {
        self.ring.newest().is_some_and(|n| n.host_ns > c.end_ns)
    }

    /// Finishes a clip. `frames_past`: a frame past its window arrived.
    fn finish(&self, c: OpenClip<T>, frames_past: bool) -> RecorderEvent<T> {
        let n = c.frames.len();
        let first_at = |ns: u64| c.frames.partition_point(|f| f.host_ns < ns);
        let impact_at = first_at(c.impact_ns);
        let missing = if n == 0 {
            Some(DiscardReason::NoFrames)
        } else if impact_at == n {
            Some(DiscardReason::NoImpactFrame)
        } else {
            None
        };
        if let Some(reason) = missing {
            return RecorderEvent::Discarded(Discarded {
                id: c.id,
                tags: c.tags,
                reason,
            });
        }
        let trigger_at = first_at(c.trigger_ns);
        let period = frame_period_ns(&c.frames, self.config.fps);
        // Frames are missing right before frame `i` (or there is no frame `i`).
        let gap_before = |i: usize| {
            i == n
                || (i > 0
                    && period
                        .is_some_and(|p| missing_between(&c.frames[i - 1], &c.frames[i], p) > 0))
        };
        let complete = frames_past && !gap_before(trigger_at) && !gap_before(impact_at);
        let dropped = count_dropped(&c.frames, period);
        RecorderEvent::Finished(Clip {
            id: c.id,
            tags: c.tags,
            trigger_index: trigger_at,
            impact_index: impact_at,
            frames: c.frames,
            trigger_ns: c.trigger_ns,
            impact_ns: c.impact_ns,
            complete,
            dropped,
            truncated: c.truncated,
            short_buffer: c.short_buffer,
            fps: self.config.fps,
        })
    }
}

/// The nominal frame period: from `fps` if given, else the median spacing of
/// the frames' arrival times.
fn frame_period_ns(frames: &[Frame], fps: Option<f64>) -> Option<f64> {
    if let Some(fps) = fps.filter(|f| *f > 0.0) {
        return Some(1e9 / fps);
    }
    let mut gaps: Vec<u64> = frames
        .windows(2)
        .map(|w| w[1].host_ns.saturating_sub(w[0].host_ns))
        .collect();
    gaps.sort_unstable();
    // Frame periods are far below 2^52 ns.
    #[allow(clippy::cast_precision_loss)]
    gaps.get(gaps.len() / 2)
        .filter(|g| **g > 0)
        .map(|&g| g as f64)
}

/// Frames missing between consecutive frames (see `frame::missing_between`).
fn count_dropped(frames: &[Frame], period: Option<f64>) -> u32 {
    let Some(period) = period else {
        return 0;
    };
    let dropped = frames
        .windows(2)
        .map(|w| missing_between(&w[0], &w[1], period))
        .fold(0u64, u64::saturating_add);
    u32::try_from(dropped).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests;
