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
/// let config = RecorderConfig {
///     pre_roll: Duration::from_millis(200),
///     post_roll: Duration::from_millis(100),
///     ..RecorderConfig::default()
/// };
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
mod tests {
    use std::sync::Arc;

    use super::*;

    const MS: u64 = 1_000_000;

    /// 100 fps frames: frame `i` at `i * 10` ms.
    fn frame(seq: u64) -> Frame {
        frame_at(seq, seq * 10 * MS)
    }

    fn frame_at(seq: u64, host_ns: u64) -> Frame {
        Frame {
            jpeg: Arc::from(vec![u8::try_from(seq % 256).unwrap(); 8]),
            host_ns,
            device_ns: None,
            seq,
            width: 8,
            height: 8,
        }
    }

    /// As `frame`, with the device time equal to the arrival time.
    fn device_frame(seq: u64) -> Frame {
        Frame {
            device_ns: Some(seq * 10 * MS),
            ..frame(seq)
        }
    }

    fn config() -> RecorderConfig {
        RecorderConfig {
            pre_roll: Duration::from_millis(200),
            post_roll: Duration::from_millis(100),
            merge_window: Duration::from_millis(500),
            buffer: Duration::from_secs(5),
            max_bytes: usize::MAX,
            stall: Duration::from_millis(300),
            fps: Some(100.0),
        }
    }

    fn feed(
        rec: &mut ClipRecorder<&'static str>,
        seqs: std::ops::Range<u64>,
    ) -> Vec<RecorderEvent<&'static str>> {
        seqs.flat_map(|s| rec.push(frame(s))).collect()
    }

    fn finished(events: Vec<RecorderEvent<&'static str>>) -> Vec<Clip<&'static str>> {
        events
            .into_iter()
            .map(|e| match e {
                RecorderEvent::Finished(c) => c,
                RecorderEvent::Discarded(d) => panic!("unexpected discard {d:?}"),
            })
            .collect()
    }

    fn seqs(c: &Clip<&str>) -> (u64, u64) {
        (c.frames[0].seq, c.frames[c.frames.len() - 1].seq)
    }

    #[test]
    fn starts_and_finishes_on_frames_past_the_window() {
        let mut rec = ClipRecorder::new(config());
        assert!(feed(&mut rec, 0..100).is_empty());
        // Trigger at 1000 ms: window 800..=1100 ms.
        assert_eq!(
            rec.trigger(Trigger::new(1000 * MS, "a", "s1")),
            TriggerOutcome::Started(ClipId(0))
        );
        assert!(feed(&mut rec, 100..111).is_empty(), "1100 ms is inside");
        assert_eq!(rec.open_clips(), 1);
        let clips = finished(rec.push(frame(111)));
        assert_eq!(clips.len(), 1);
        let c = &clips[0];
        assert_eq!(seqs(c), (80, 110));
        assert_eq!(c.frames.len(), 31);
        assert_eq!(c.frames[c.trigger_index].seq, 100);
        assert_eq!(c.impact_index, c.trigger_index);
        assert!(c.complete);
        assert!(!c.truncated);
        assert_eq!(c.dropped, 0);
        assert_eq!(c.tags, ["s1"]);
        assert_eq!(rec.open_clips(), 0);
        assert!((c.measured_fps().unwrap() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn impact_lead_shifts_the_window() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..200);
        // Trigger arrives at 1990 ms, impact 1.5 s earlier: window 290..=2090 ms.
        let t = Trigger::new(1990 * MS, "a", "s1").with_impact_lead(Duration::from_millis(1500));
        assert_eq!(t.impact_ns(), 490 * MS);
        rec.trigger(t);
        let c = finished(feed(&mut rec, 200..300)).remove(0);
        assert_eq!(seqs(&c), (29, 209));
        assert_eq!(c.frames[c.impact_index].seq, 49);
        assert_eq!(c.frames[c.trigger_index].seq, 199);
        assert_eq!(c.impact_ns, 490 * MS);
    }

    #[test]
    fn per_trigger_and_live_rolls() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(
            Trigger::new(990 * MS, "a", "s1")
                .with_pre_roll(Duration::from_millis(50))
                .with_post_roll(Duration::from_millis(20)),
        );
        let c = finished(feed(&mut rec, 100..110)).remove(0);
        assert_eq!(seqs(&c), (94, 101));
        rec.set_pre_roll(Duration::from_millis(30));
        rec.set_post_roll(Duration::from_millis(30));
        rec.trigger(Trigger::new(1090 * MS, "a", "s2"));
        let c = finished(feed(&mut rec, 110..120)).remove(0);
        assert_eq!(seqs(&c), (106, 112));
    }

    #[test]
    fn different_source_inside_the_merge_window_joins() {
        let mut rec = ClipRecorder::new(config());
        let post = Duration::from_millis(1500);
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "radar", "r1").with_post_roll(post));
        assert!(feed(&mut rec, 100..130).is_empty());
        // The other monitor reports 1.3 s late but sees the same impact (40 ms apart).
        let late = Trigger::new(2250 * MS, "camera", "c1")
            .with_impact_lead(Duration::from_millis(1300))
            .with_post_roll(post);
        assert_eq!(rec.trigger(late), TriggerOutcome::Joined(ClipId(0)));
        // The window now runs from the earlier impact estimate (950 ms) minus
        // the pre-roll to 1.5 s after the late trigger: 750..=3750 ms.
        assert!(feed(&mut rec, 130..376).is_empty());
        let c = finished(rec.push(frame(376))).remove(0);
        assert_eq!(c.tags, ["r1", "c1"]);
        assert_eq!(seqs(&c), (75, 375));
        assert_eq!(c.frames.len(), 301, "every frame once");
        assert_eq!(
            c.frames[c.trigger_index].seq, 99,
            "anchored on the first trigger"
        );
    }

    #[test]
    fn different_source_outside_the_merge_window_starts_a_new_clip() {
        let mut rec = ClipRecorder::new(config());
        let post = Duration::from_millis(1500);
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "radar", "r1").with_post_roll(post));
        feed(&mut rec, 100..130);
        // 1.29 s apart: another swing, seen by the other monitor only.
        let other = Trigger::new(2280 * MS, "camera", "c1").with_post_roll(post);
        assert_eq!(rec.trigger(other), TriggerOutcome::Started(ClipId(1)));
        assert_eq!(rec.open_clips(), 2);
        let clips = finished(feed(&mut rec, 130..400));
        assert_eq!(clips.len(), 2);
        assert_eq!(
            (clips[0].tags.as_slice(), seqs(&clips[0])),
            (&["r1"][..], (79, 249))
        );
        assert_eq!(
            (clips[1].tags.as_slice(), seqs(&clips[1])),
            (&["c1"][..], (208, 378))
        );
    }

    #[test]
    fn overlapping_clips_share_frames() {
        let mut cfg = config();
        cfg.merge_window = Duration::from_millis(100);
        let mut rec = ClipRecorder::new(cfg);
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1")); // 790..=1090 ms
        rec.trigger(Trigger::new(1190 * MS, "b", "b1")); // 990..=1290 ms
        let clips = finished(feed(&mut rec, 100..200));
        assert_eq!(seqs(&clips[0]), (79, 109));
        assert_eq!(seqs(&clips[1]), (99, 129));
        let in_both = |c: &Clip<&str>| c.frames.iter().find(|f| f.seq == 105).unwrap().jpeg.clone();
        assert!(Arc::ptr_eq(&in_both(&clips[0]), &in_both(&clips[1])));
    }

    #[test]
    fn joining_with_an_earlier_impact_extends_the_start() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        // Impact estimate 300 ms earlier: start moves from 790 to 490 ms.
        let t = Trigger::new(990 * MS, "b", "b1").with_impact_lead(Duration::from_millis(300));
        assert_eq!(rec.trigger(t), TriggerOutcome::Joined(ClipId(0)));
        let c = finished(feed(&mut rec, 100..120)).remove(0);
        assert_eq!(seqs(&c), (49, 109));
        assert_eq!(c.frames.len(), 61);
    }

    #[test]
    fn same_source_truncates_and_starts_a_new_clip() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "putter", "p1"));
        feed(&mut rec, 100..105); // up to 1040 ms
        // The next putt 50 ms later: p1 ends at 1040 ms, p2 starts.
        assert_eq!(
            rec.trigger(Trigger::new(1040 * MS, "putter", "p2")),
            TriggerOutcome::Truncated {
                previous: ClipId(0),
                clip: ClipId(1)
            }
        );
        let clips = finished(feed(&mut rec, 105..200));
        assert_eq!(clips.len(), 2);
        let (p1, p2) = (&clips[0], &clips[1]);
        assert_eq!(p1.tags, ["p1"]);
        assert!(p1.truncated && p1.complete);
        assert_eq!(seqs(p1), (79, 104));
        assert_eq!(p2.tags, ["p2"]);
        assert!(!p2.truncated);
        assert_eq!(seqs(p2), (84, 114), "overlaps p1");
    }

    #[test]
    fn same_source_truncation_drops_frames_already_past_the_cut() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        feed(&mut rec, 100..108); // frames up to 1070 ms are in the clip
        rec.trigger(Trigger::new(1020 * MS, "a", "a2")); // arrived late
        let clips = finished(feed(&mut rec, 108..200));
        assert_eq!(seqs(&clips[0]), (79, 102));
    }

    #[test]
    fn poll_finishes_incomplete_clips_when_frames_stop() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        feed(&mut rec, 100..103); // camera dies at 1020 ms; window ends 1090 ms
        assert!(rec.poll(1300 * MS).is_empty(), "stall not reached");
        let clips = finished(rec.poll(1390 * MS));
        let c = &clips[0];
        assert!(!c.complete);
        assert_eq!(seqs(c), (79, 102));
        assert_eq!(c.frames[c.trigger_index].seq, 99);
        assert_eq!(rec.open_clips(), 0);
    }

    #[test]
    fn clips_without_their_impact_frame_are_discarded() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100); // last frame at 990 ms
        rec.trigger(Trigger::new(995 * MS, "a", "a1"));
        let events = rec.flush();
        assert_eq!(
            events,
            vec![RecorderEvent::Discarded(Discarded {
                id: ClipId(0),
                tags: vec!["a1"],
                reason: DiscardReason::NoImpactFrame
            })]
        );
        // No frames at all.
        let mut rec: ClipRecorder<&str> = ClipRecorder::new(config());
        rec.trigger(Trigger::new(MS, "a", "a1"));
        assert_eq!(
            rec.poll(u64::MAX),
            vec![RecorderEvent::Discarded(Discarded {
                id: ClipId(0),
                tags: vec!["a1"],
                reason: DiscardReason::NoFrames
            })]
        );
    }

    #[test]
    fn counts_dropped_frames() {
        let mut rec = ClipRecorder::new(config());
        let mut events = Vec::new();
        for s in (0..120).filter(|s| !(95..98).contains(s) && *s != 104) {
            events.extend(rec.push(device_frame(s)));
            if s == 99 {
                rec.trigger(Trigger::new(990 * MS, "a", "a1"));
            }
        }
        let c = finished(events).remove(0);
        assert_eq!(c.dropped, 4, "3 at 950-970 ms, 1 at 1040 ms");
        assert!(c.complete);
        assert_eq!(c.fps, Some(100.0));
        // Without a nominal rate the clip's own median spacing is used.
        let mut cfg = config();
        cfg.fps = None;
        let mut rec = ClipRecorder::new(cfg);
        for s in (0..105).filter(|s| *s != 102) {
            rec.push(device_frame(s));
        }
        rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        let c = finished(rec.flush()).remove(0);
        assert_eq!(c.dropped, 1);
        assert!(!c.complete, "flushed before the window ended");
        assert_eq!(c.fps, None);
    }

    #[test]
    fn arrival_times_tolerate_delivery_jitter() {
        let mut rec = ClipRecorder::new(config());
        let mut events = Vec::new();
        // No device times. 1040 ms is missing (a two-period gap, which
        // delivery jitter also produces) and 950..=970 ms (four periods).
        for s in (0..120).filter(|s| !(95..98).contains(s) && *s != 104) {
            events.extend(rec.push(frame(s)));
            if s == 99 {
                rec.trigger(Trigger::new(990 * MS, "a", "a1"));
            }
        }
        let c = finished(events).remove(0);
        assert_eq!(c.dropped, 3);
    }

    #[test]
    fn a_trigger_entirely_in_the_past_finishes_on_poll() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..200);
        rec.trigger(Trigger::new(500 * MS, "a", "a1"));
        let c = finished(rec.poll(0)).remove(0);
        assert!(c.complete);
        assert_eq!(seqs(&c), (30, 60));
    }

    #[test]
    fn ignores_out_of_order_and_repeated_frames() {
        let mut rec: ClipRecorder<&str> = ClipRecorder::new(config());
        assert!(rec.push(frame(10)).is_empty());
        rec.push(frame(10));
        rec.push(frame_at(11, 50 * MS));
        assert_eq!(rec.ring().len(), 1);
    }

    #[test]
    fn same_source_after_the_window_ended_starts_a_new_clip() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..101); // frames to 1000 ms
        rec.trigger(Trigger::new(990 * MS, "a", "a1")); // window 790..=1090 ms
        // Frames stall; the same monitor fires again at 1500 ms, after the
        // first window ended.
        assert_eq!(
            rec.trigger(Trigger::new(1500 * MS, "a", "a2")),
            TriggerOutcome::Started(ClipId(1))
        );
        // Frames resume at 1010 ms.
        let clips = finished(feed(&mut rec, 101..200));
        assert_eq!(clips.len(), 2);
        let (a1, a2) = (&clips[0], &clips[1]);
        assert_eq!(a1.tags, ["a1"]);
        assert_eq!(seqs(a1), (79, 109), "ends at its own window");
        assert!(a1.complete && !a1.truncated);
        assert_eq!(a2.tags, ["a2"]);
        assert_eq!(seqs(a2), (130, 160));
    }

    #[test]
    fn same_source_at_the_window_end_does_not_truncate() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1")); // ends 1090 ms
        assert_eq!(
            rec.trigger(Trigger::new(1090 * MS, "a", "a2")),
            TriggerOutcome::Started(ClipId(1))
        );
        // One nanosecond earlier cuts it.
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        assert!(matches!(
            rec.trigger(Trigger::new(1090 * MS - 1, "a", "a2")),
            TriggerOutcome::Truncated { .. }
        ));
    }

    #[test]
    fn duplicate_triggers_are_the_callers_to_drop() {
        // The same source twice at the same instant: the recorder can't tell
        // a duplicate from a second shot, so the first clip is cut at the
        // trigger and a second clip starts.
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        assert_eq!(
            rec.trigger(Trigger::new(990 * MS, "a", "a1-again")),
            TriggerOutcome::Truncated {
                previous: ClipId(0),
                clip: ClipId(1)
            }
        );
        let clips = finished(feed(&mut rec, 100..200));
        assert_eq!(clips.len(), 2);
        assert!(clips[0].truncated);
        assert_eq!(seqs(&clips[0]), (79, 99));
        assert_eq!(seqs(&clips[1]), (79, 109));
    }

    #[test]
    fn join_considers_every_open_clip() {
        let mut cfg = config();
        cfg.post_roll = Duration::from_millis(1500);
        let mut rec = ClipRecorder::new(cfg);
        feed(&mut rec, 0..100);
        rec.trigger(Trigger::new(990 * MS, "a", "a1")); // impact 990 ms
        feed(&mut rec, 100..200);
        rec.trigger(Trigger::new(1990 * MS, "b", "b1")); // another swing
        // A third monitor reports the first swing 1.1 s late: impact 990 ms.
        let late = Trigger::new(2090 * MS, "c", "c1").with_impact_lead(Duration::from_millis(1100));
        assert_eq!(rec.trigger(late), TriggerOutcome::Joined(ClipId(0)));
        // The newest matching clip wins when several match.
        let near = Trigger::new(1700 * MS, "d", "d1").with_impact_lead(Duration::ZERO);
        assert_eq!(rec.trigger(near), TriggerOutcome::Joined(ClipId(1)));
        // A clip that already has the source is skipped.
        let again = Trigger::new(1995 * MS, "b", "b2");
        assert!(matches!(
            rec.trigger(again),
            TriggerOutcome::Truncated { .. }
        ));
    }

    #[test]
    fn a_clip_ending_after_its_impact_is_kept_incomplete() {
        // A monitor reporting 2 s after impact; the camera dies 0.5 s after
        // the impact, before the trigger arrives.
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..150); // to 1490 ms
        let t = Trigger::new(3000 * MS, "r10", "s1").with_impact_lead(Duration::from_millis(2000));
        rec.trigger(t);
        let c = finished(rec.poll(5000 * MS)).remove(0);
        assert!(!c.complete);
        assert_eq!(seqs(&c), (80, 149));
        assert_eq!(c.frames[c.impact_index].seq, 100);
        assert_eq!(c.trigger_index, c.frames.len(), "no trigger frame");
        assert!(c.trigger_frame().is_none());
        assert_eq!(c.impact_frame().map(|f| f.seq), Some(100));
    }

    #[test]
    fn a_gap_across_the_trigger_is_incomplete() {
        // Frames to 940 ms, trigger at 990 ms, frames resume at 1200 ms:
        // inside a 1.5 s post-roll, the clip is kept but not complete.
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..95);
        let impact = Trigger::new(990 * MS, "a", "a1")
            .with_impact_lead(Duration::from_millis(100))
            .with_post_roll(Duration::from_millis(1500));
        rec.trigger(impact);
        let c = finished(feed(&mut rec, 120..300)).remove(0);
        assert!(!c.complete, "frames are missing across the trigger");
        assert_eq!(c.frames[c.impact_index].seq, 89);
        assert_eq!(c.frames[c.trigger_index].seq, 120, "first frame after it");
        assert_eq!(c.dropped, 25);

        // The same gap past a short window: the clip has its impact frame
        // (880 ms) but no trigger frame.
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..95);
        rec.trigger(Trigger::new(990 * MS, "a", "a1").with_impact_lead(Duration::from_millis(100)));
        let c = finished(feed(&mut rec, 120..130)).remove(0);
        assert!(!c.complete);
        assert_eq!(c.trigger_index, c.frames.len());

        // Without a lead the impact frame is missing too: discarded.
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..95);
        rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        let events = feed(&mut rec, 120..130);
        assert!(matches!(
            events.as_slice(),
            [RecorderEvent::Discarded(Discarded {
                reason: DiscardReason::NoImpactFrame,
                ..
            })]
        ));
    }

    #[test]
    fn open_clips_wait_for_poll_when_frames_stop() {
        let mut rec = ClipRecorder::new(config());
        feed(&mut rec, 0..100);
        for i in 0..1000u64 {
            rec.trigger(Trigger::new(990 * MS + i * 600 * MS, "a", "x"));
        }
        assert_eq!(
            rec.open_clips(),
            1000,
            "nothing finishes them without frames"
        );
        let events = rec.poll(u64::MAX);
        assert_eq!(events.len(), 1000);
        assert_eq!(rec.open_clips(), 0);
        assert!(matches!(events[0], RecorderEvent::Finished(_)));
        assert!(
            events[1..]
                .iter()
                .all(|e| matches!(e, RecorderEvent::Discarded(_)))
        );
    }

    #[test]
    fn flags_a_buffer_shorter_than_pre_roll_plus_lead() {
        let mut cfg = config();
        cfg.buffer = Duration::from_millis(1000);
        assert!(cfg.buffer_covers(Duration::from_millis(800)));
        assert!(!cfg.buffer_covers(Duration::from_millis(801)));
        let mut rec = ClipRecorder::new(cfg);
        feed(&mut rec, 0..300);
        // 200 ms pre-roll + 900 ms lead > 1 s buffer.
        rec.trigger(
            Trigger::new(2990 * MS, "a", "a1").with_impact_lead(Duration::from_millis(900)),
        );
        let c = finished(feed(&mut rec, 300..320)).remove(0);
        assert!(c.short_buffer);
        assert_eq!(c.frames[0].seq, 199, "starts at the ring's oldest frame");
        // A lead the buffer covers.
        rec.trigger(
            Trigger::new(3190 * MS, "a", "a2").with_impact_lead(Duration::from_millis(500)),
        );
        let c = finished(feed(&mut rec, 320..340)).remove(0);
        assert!(!c.short_buffer);
        assert_eq!(c.frames[0].seq, 249);
    }
}
