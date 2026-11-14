//! Running a [`Source`] on its own thread.
//!
//! The capture thread reads frames as fast as the source delivers them and
//! pushes each one straight into a [`Ring`]. That ring is the only buffer:
//! memory is bounded by its duration and byte caps, and a slow consumer loses
//! the oldest frames rather than growing a queue. Consumers pull frames newer
//! than the last one they saw ([`Capture::frames_after`]).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::clock;
use crate::frame::missing_between;
use crate::source::Interrupter;
use crate::{CameraInfo, Error, Format, Frame, Result, Ring, Source};

/// Sequence numbers per capture: each [`Capture`] stamps its frames from its
/// own block, above every block handed out before it.
const SEQ_BLOCK: u64 = 1 << 40;
/// The next block's index.
static NEXT_SEQ_BLOCK: AtomicU64 = AtomicU64::new(0);

/// What the capture thread reports once the source is open.
type Opened = (CameraInfo, Format, Option<Interrupter>);

/// How long [`Capture::stop`] and drop wait for the thread before leaving it
/// to finish on its own.
const STOP_WAIT: Duration = Duration::from_secs(3);

/// Capture settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureConfig {
    /// How much history the hand-off ring keeps. Consumers must pull more
    /// often than this. Default 2 s.
    pub ring_duration: Duration,
    /// Byte cap of the ring. Default 256 MiB.
    pub ring_bytes: usize,
    /// How long one read may wait before the thread checks for a stop
    /// request. Default 200 ms. Media Foundation cameras (Windows) ignore it:
    /// their reads wait for the camera; [`Capture::request_stop`] ends such a
    /// read by shutting the camera down instead.
    pub read_timeout: Duration,
    /// How long the source may deliver nothing before the capture fails with
    /// [`Error::Stalled`] and releases the camera. Default 3 s;
    /// `Duration::ZERO` never fails. A read blocked inside the platform API
    /// counts too: the stall is noticed by [`Capture::status`] and
    /// [`Capture::wait_frames_after`].
    pub stall: Duration,
    /// Name of the capture thread. Default `stance`.
    pub thread_name: String,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            ring_duration: Duration::from_secs(2),
            ring_bytes: 256 << 20,
            read_timeout: Duration::from_millis(200),
            stall: Duration::from_secs(3),
            thread_name: "stance".to_owned(),
        }
    }
}

/// What the capture thread is doing.
#[derive(Debug, Clone)]
pub enum CaptureStatus {
    /// Delivering frames.
    Running,
    /// Stopped through [`Capture::request_stop`], [`Capture::stop`] or drop.
    Stopped,
    /// The source ended, failed (camera unplugged, file finished) or stalled
    /// ([`Error::Stalled`]). No more frames will arrive; the error says why.
    Failed(Arc<Error>),
}

/// Capture counters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CaptureStats {
    /// Frames delivered since start.
    pub frames: u64,
    /// Frames missing since start, from gaps between consecutive frames: on
    /// the device times when the frames carry them (gaps over 1.5 nominal
    /// periods), on arrival times otherwise (gaps over 2.5 periods, so
    /// delivery jitter doesn't count).
    pub dropped: u64,
    /// Frame rate over the last second of frames in the ring.
    pub fps: f64,
    /// Host time of the newest frame.
    pub last_frame_ns: Option<u64>,
    /// Frames in the ring.
    pub ring_frames: usize,
    /// JPEG bytes in the ring.
    pub ring_bytes: usize,
}

/// The capture's state and rules, without threads or a clock: sequence
/// stamping, drop counting, the stall rule and the ring push. Every method
/// takes the host monotonic time it runs at.
struct Core {
    ring: Ring,
    latest: Option<Frame>,
    frames: u64,
    dropped: u64,
    status: CaptureStatus,
    /// The next frame's sequence number.
    next_seq: u64,
    /// The source's nominal frame period, for drop counting.
    period_ns: f64,
    stall: Duration,
    /// When the last frame arrived (or the source opened).
    last_frame_ns: u64,
}

impl Core {
    fn new(config: &CaptureConfig, first_seq: u64, now_ns: u64) -> Self {
        Self {
            ring: Ring::new(config.ring_duration, config.ring_bytes),
            latest: None,
            frames: 0,
            dropped: 0,
            status: CaptureStatus::Running,
            next_seq: first_seq,
            period_ns: 0.0,
            stall: config.stall,
            last_frame_ns: now_ns,
        }
    }

    fn running(&self) -> bool {
        matches!(self.status, CaptureStatus::Running)
    }

    /// The source opened in `format`: frames are counted against its period
    /// and the stall clock starts.
    fn opened(&mut self, format: &Format, now_ns: u64) {
        // Frame periods are far below 2^52 ns.
        #[allow(clippy::cast_precision_loss)]
        let period_ns = format.frame_period_ns() as f64;
        self.period_ns = period_ns;
        self.last_frame_ns = now_ns;
    }

    /// Stamps `frame`, counts the frames missing before it and pushes it into
    /// the ring. False, with the frame discarded, once the capture is no
    /// longer running.
    fn on_frame(&mut self, mut frame: Frame, now_ns: u64) -> bool {
        if !self.running() {
            return false;
        }
        frame.seq = self.next_seq;
        self.next_seq += 1;
        self.dropped += self
            .latest
            .as_ref()
            .map_or(0, |p| missing_between(p, &frame, self.period_ns));
        self.frames += 1;
        self.last_frame_ns = now_ns;
        self.latest = Some(frame.clone());
        self.ring.push(frame);
        true
    }

    /// Time left at `now_ns` before the stall runs out; `None` if stall
    /// detection is off.
    fn stall_left(&self, now_ns: u64) -> Option<Duration> {
        let silent = Duration::from_nanos(now_ns.saturating_sub(self.last_frame_ns));
        (!self.stall.is_zero()).then(|| self.stall.saturating_sub(silent))
    }

    /// The stall rule, applied at `now_ns` while no frame is arriving: true
    /// when this call failed a running capture with [`Error::Stalled`].
    fn on_idle(&mut self, now_ns: u64) -> bool {
        let stalled = self.running() && self.stall_left(now_ns).is_some_and(|d| d.is_zero());
        if stalled {
            self.status = CaptureStatus::Failed(Arc::new(Error::Stalled(self.stall)));
        }
        stalled
    }

    /// Fails the capture with `error`, unless it already ended.
    fn fail(&mut self, error: Error) {
        if self.running() {
            self.status = CaptureStatus::Failed(Arc::new(error));
        }
    }

    /// Marks the capture stopped, unless it already ended.
    fn stop(&mut self) {
        if self.running() {
            self.status = CaptureStatus::Stopped;
        }
    }
}

struct Shared {
    state: Mutex<Core>,
    frame_ready: Condvar,
    stop: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Core> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Marks the capture failed (if it is still running) and wakes waiters.
    fn fail(&self, error: Error) {
        self.lock().fail(error);
        self.frame_ready.notify_all();
    }
}

/// A [`Source`] running on its own thread, filling a bounded [`Ring`].
///
/// Stops on [`Capture::request_stop`], [`Capture::stop`] or drop. When the
/// source fails, ends or stalls, the thread exits and [`Capture::status`]
/// reports the error.
///
/// # Blocking
///
/// [`Capture::start`] blocks while the source opens (a camera can take
/// seconds). [`Capture::stop`] and drop wait for the thread to exit, up to
/// 3 s. From async code, call `start` through a blocking task, and stop with
/// [`Capture::request_stop`], which never blocks, then drop the capture once
/// [`Capture::is_finished`] (dropping a finished capture doesn't wait). The
/// other methods only take a short lock, except
/// [`Capture::wait_frames_after`], which waits by design.
///
/// # Sequence numbers
///
/// Frames are re-stamped with sequence numbers that are consecutive within a
/// capture and higher than those of every capture started before it in the
/// process. A cursor (the last `seq` pulled) therefore stays valid across a
/// reconnect: [`Capture::frames_after`] on the new capture returns all of its
/// frames.
///
/// ```
/// use std::time::Duration;
/// use stance::{Capture, CaptureConfig, TestPattern, TestPatternConfig};
///
/// let pattern = TestPatternConfig { width: 64, height: 48, fps: 100, ..TestPatternConfig::default() };
/// let capture = Capture::start(CaptureConfig::default(), move || TestPattern::new(pattern))?;
/// let frames = capture.wait_frames_after(None, Duration::from_secs(2));
/// assert!(!frames.is_empty());
/// let still = capture.latest().expect("a frame arrived");
/// assert_eq!((still.width, still.height), (64, 48));
/// capture.stop();
/// # Ok::<(), stance::Error>(())
/// ```
pub struct Capture {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    interrupter: Option<Interrupter>,
    info: CameraInfo,
    format: Format,
}

impl std::fmt::Debug for Capture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capture")
            .field("info", &self.info)
            .field("format", &self.format)
            .finish_non_exhaustive()
    }
}

impl Capture {
    /// Starts a capture thread, opens the source on it with `open` and waits
    /// until it is open. Sources are opened on the thread that reads them
    /// because Media Foundation cameras belong to the thread that opened them.
    ///
    /// Blocks the calling thread while the source opens, which for a camera
    /// can take seconds; from async code call it in a blocking task.
    ///
    /// ```no_run
    /// use stance::{Capture, CaptureConfig, FormatRequest};
    /// let id = "/dev/v4l/by-id/usb-cam-video-index0".to_owned();
    /// let capture = Capture::start(CaptureConfig::default(), move || {
    ///     stance::open(&id, &FormatRequest::Auto)
    /// })?;
    /// # Ok::<(), stance::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Whatever `open` returns, or [`Error::Io`] if the thread can't be spawned.
    pub fn start<S, F>(config: CaptureConfig, open: F) -> Result<Self>
    where
        S: Source,
        F: FnOnce() -> Result<S> + Send + 'static,
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(Core::new(&config, next_first_seq(), clock::now_ns())),
            frame_ready: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<Opened>>(1);
        let thread_shared = Arc::clone(&shared);
        let read_timeout = config.read_timeout;
        let thread = thread::Builder::new()
            .name(config.thread_name)
            .spawn(move || {
                let mut source = match open() {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                thread_shared
                    .lock()
                    .opened(source.format(), clock::now_ns());
                let opened = (
                    source.info().clone(),
                    source.format().clone(),
                    source.interrupter(),
                );
                if ready_tx.send(Ok(opened)).is_err() {
                    return;
                }
                run(&mut source, &thread_shared, read_timeout);
            })
            .map_err(|e| Error::io("spawning the capture thread", e))?;
        match ready_rx.recv() {
            Ok(Ok((info, format, interrupter))) => Ok(Self {
                shared,
                thread: Some(thread),
                interrupter,
                info,
                format,
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err(Error::io(
                    "opening the source",
                    std::io::Error::other("the capture thread exited while opening"),
                ))
            }
        }
    }

    /// The source's description.
    #[must_use]
    pub fn info(&self) -> &CameraInfo {
        &self.info
    }

    /// The mode frames arrive in.
    #[must_use]
    pub fn format(&self) -> &Format {
        &self.format
    }

    /// The newest frame (for a still image).
    #[must_use]
    pub fn latest(&self) -> Option<Frame> {
        self.shared.lock().latest.clone()
    }

    /// Frames newer than `seq` still in the ring (all of them for `None`),
    /// oldest first. A gap between `seq` and the first frame's `seq` means
    /// frames left the ring before they were pulled (or `seq` came from an
    /// earlier capture).
    #[must_use]
    pub fn frames_after(&self, seq: Option<u64>) -> Vec<Frame> {
        self.shared.lock().ring.after(seq)
    }

    /// As [`Capture::frames_after`], but waits up to `timeout` for a frame if
    /// there is none yet. Returns early, possibly empty, when the capture
    /// stops, fails or stalls.
    #[must_use]
    pub fn wait_frames_after(&self, seq: Option<u64>, timeout: Duration) -> Vec<Frame> {
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.lock();
        loop {
            let frames = state.ring.after(seq);
            if !frames.is_empty() || !state.running() {
                return frames;
            }
            let now_ns = clock::now_ns();
            if state.on_idle(now_ns) {
                drop(state);
                self.stalled();
                return frames;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return frames;
            }
            let wait = state.stall_left(now_ns).map_or(left, |s| s.min(left));
            state = match self.shared.frame_ready.wait_timeout(state, wait) {
                Ok((guard, _)) => guard,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
    }

    /// What the capture thread is doing. Notices a stalled source (see
    /// [`CaptureConfig::stall`]): the capture then fails and releases the
    /// camera.
    #[must_use]
    pub fn status(&self) -> CaptureStatus {
        let mut state = self.shared.lock();
        if state.on_idle(clock::now_ns()) {
            let status = state.status.clone();
            drop(state);
            self.stalled();
            return status;
        }
        if state.running() && self.is_finished() {
            // Only a panic ends the thread without setting a status.
            state.fail(Error::Device {
                context: "capture thread".into(),
                message: "exited unexpectedly".into(),
            });
            self.shared.frame_ready.notify_all();
        }
        state.status.clone()
    }

    /// After the capture failed as stalled: wakes waiters and ends the
    /// blocked read.
    fn stalled(&self) {
        self.shared.frame_ready.notify_all();
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(i) = &self.interrupter {
            i.interrupt();
        }
    }

    /// Counters, and the frame rate over the last second.
    #[must_use]
    pub fn stats(&self) -> CaptureStats {
        let state = self.shared.lock();
        let last = state.ring.newest().map(|f| f.host_ns);
        let fps = last.map_or(0.0, |newest| {
            let recent: Vec<u64> = state
                .ring
                .iter()
                .rev()
                .map(|f| f.host_ns)
                .take_while(|t| newest - t <= 1_000_000_000)
                .collect();
            match (recent.first(), recent.last()) {
                // Frame counts and spans are far below 2^52.
                #[allow(clippy::cast_precision_loss)]
                (Some(&a), Some(&b)) if a > b => (recent.len() - 1) as f64 * 1e9 / (a - b) as f64,
                _ => 0.0,
            }
        });
        CaptureStats {
            frames: state.frames,
            dropped: state.dropped,
            fps,
            last_frame_ns: last,
            ring_frames: state.ring.len(),
            ring_bytes: state.ring.bytes(),
        }
    }

    /// Asks the thread to stop and returns at once. The status becomes
    /// [`CaptureStatus::Stopped`] (unless the capture already failed) and no
    /// more frames arrive. A read blocked in the platform API is ended through
    /// the source's [`Interrupter`] (Media Foundation cameras are shut down),
    /// so the camera is free to reopen as soon as [`Capture::is_finished`].
    pub fn request_stop(&self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        self.shared.lock().stop();
        self.shared.frame_ready.notify_all();
        if let Some(i) = &self.interrupter {
            i.interrupt();
        }
    }

    /// True once the capture thread has exited and released the source.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Stops the thread and waits for it to exit, up to 3 s; a read hung
    /// inside a driver beyond that is left to finish on its own. This blocks:
    /// call it from a blocking context (in async code, use
    /// [`Capture::request_stop`] and [`Capture::is_finished`]).
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.request_stop();
        if let Some(t) = self.thread.take() {
            let deadline = Instant::now() + STOP_WAIT;
            while !t.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            if t.is_finished() {
                let _ = t.join();
            }
        }
    }
}

impl Drop for Capture {
    /// As [`Capture::stop`]: waits up to 3 s unless the thread already exited.
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The next capture's first sequence number: the start of a block above every
/// block handed out before.
fn next_first_seq() -> u64 {
    NEXT_SEQ_BLOCK
        .fetch_add(1, Ordering::Relaxed)
        .saturating_mul(SEQ_BLOCK)
}

/// The capture loop: read, hand each frame to the core, repeat until stopped,
/// or until the source fails or stalls.
fn run<S: Source>(source: &mut S, shared: &Shared, read_timeout: Duration) {
    while !shared.stop.load(Ordering::SeqCst) {
        match source.next_frame(read_timeout) {
            Ok(Some(frame)) => {
                if !shared.lock().on_frame(frame, clock::now_ns()) {
                    break;
                }
                shared.frame_ready.notify_all();
            }
            Ok(None) => {
                if shared.lock().on_idle(clock::now_ns()) {
                    shared.frame_ready.notify_all();
                    return;
                }
            }
            Err(e) => {
                shared.fail(e);
                return;
            }
        }
    }
    shared.lock().stop();
}

#[cfg(test)]
mod tests {
    use std::sync::Condvar;

    use super::*;
    use crate::{TestPattern, TestPatternConfig};

    const MS: u64 = 1_000_000;

    fn pattern(frames: u64) -> TestPatternConfig {
        TestPatternConfig {
            width: 32,
            height: 32,
            fps: 100,
            paced: false,
            frames: Some(frames),
            origin_ns: Some(1_000_000_000),
            quality: 50,
        }
    }

    fn large_ring() -> CaptureConfig {
        CaptureConfig {
            ring_duration: Duration::from_secs(60),
            ..CaptureConfig::default()
        }
    }

    fn frame_at(host_ms: u64) -> Frame {
        Frame {
            jpeg: Arc::from(vec![0u8; 4]),
            host_ns: host_ms * MS,
            device_ns: None,
            seq: 0,
            width: 32,
            height: 32,
        }
    }

    /// A core for a 100 fps source opened at `t = 0`.
    fn core(config: &CaptureConfig) -> Core {
        let mut core = Core::new(config, next_first_seq(), 0);
        let format = TestPattern::new(pattern(1)).unwrap().format().clone();
        core.opened(&format, 0);
        core
    }

    #[test]
    fn core_stamps_counts_and_keeps_frames() {
        let mut c = core(&large_ring());
        let first = c.next_seq;
        // Arrival times only: a four-period gap is three missing frames.
        for ms in [0, 10, 20, 60] {
            assert!(c.on_frame(frame_at(ms), ms * MS));
        }
        let seqs: Vec<u64> = c.ring.after(None).iter().map(|f| f.seq - first).collect();
        assert_eq!(seqs, [0, 1, 2, 3], "consecutive");
        assert_eq!((c.frames, c.dropped), (4, 3));
        assert_eq!(c.latest.as_ref().unwrap().seq, first + 3);
    }

    #[test]
    fn cursors_survive_a_new_capture() {
        let mut first = core(&large_ring());
        for ms in 0..5 {
            first.on_frame(frame_at(ms * 10), 0);
        }
        let cursor = first.ring.after(None).last().map(|f| f.seq);
        // A reconnect: a new capture, the consumer keeps its cursor.
        let mut second = core(&large_ring());
        for ms in 0..5 {
            second.on_frame(frame_at(ms * 10), 0);
        }
        let frames = second.ring.after(cursor);
        assert_eq!(frames.len(), 5, "every frame of the new capture");
        assert!(frames.windows(2).all(|w| w[1].seq == w[0].seq + 1));
    }

    #[test]
    fn a_silent_source_fails_as_stalled() {
        let mut c = core(&CaptureConfig {
            stall: Duration::from_millis(100),
            ..CaptureConfig::default()
        });
        assert!(!c.on_idle(99 * MS), "silent since opening, not yet stalled");
        assert_eq!(c.stall_left(40 * MS), Some(Duration::from_millis(60)));
        // A frame restarts the stall clock.
        assert!(c.on_frame(frame_at(50), 50 * MS));
        assert!(!c.on_idle(149 * MS));
        assert!(c.on_idle(150 * MS));
        assert!(matches!(&c.status, CaptureStatus::Failed(e) if matches!(**e, Error::Stalled(_))));
        assert!(!c.on_idle(500 * MS), "reported once");
        assert!(
            !c.on_frame(frame_at(500), 500 * MS),
            "no frames after a failure"
        );
        assert_eq!(c.frames, 1);
    }

    #[test]
    fn zero_stall_never_fails_and_stopping_ends_frames() {
        let mut c = core(&CaptureConfig {
            stall: Duration::ZERO,
            ..CaptureConfig::default()
        });
        assert_eq!(c.stall_left(u64::MAX), None);
        assert!(!c.on_idle(u64::MAX));
        c.stop();
        assert!(matches!(c.status, CaptureStatus::Stopped));
        assert!(!c.on_frame(frame_at(0), 0));
        c.fail(Error::Ended);
        assert!(
            matches!(c.status, CaptureStatus::Stopped),
            "stopped stays stopped"
        );
    }

    /// Delivers nothing: every read waits until interrupted, whatever its
    /// timeout (a Media Foundation camera gone silent).
    struct Hung {
        interrupted: Arc<(Mutex<bool>, Condvar)>,
        /// Signaled when the first read begins.
        reading: mpsc::Sender<()>,
        info: CameraInfo,
        format: Format,
    }

    impl Source for Hung {
        fn next_frame(&mut self, _: Duration) -> Result<Option<Frame>> {
            let _ = self.reading.send(());
            let (lock, cv) = &*self.interrupted;
            let mut stopped = lock.lock().unwrap();
            while !*stopped {
                stopped = cv.wait(stopped).unwrap();
            }
            Err(Error::Device {
                context: "reading a frame".into(),
                message: "shut down".into(),
            })
        }

        fn info(&self) -> &CameraInfo {
            &self.info
        }

        fn format(&self) -> &Format {
            &self.format
        }

        fn interrupter(&self) -> Option<Interrupter> {
            let flag = Arc::clone(&self.interrupted);
            Some(Interrupter::new(move || {
                *flag.0.lock().unwrap() = true;
                flag.1.notify_all();
            }))
        }
    }

    #[test]
    fn reports_source_end_instead_of_spinning() {
        let err = Capture::start(CaptureConfig::default(), || {
            TestPattern::new(TestPatternConfig {
                fps: 0,
                ..TestPatternConfig::default()
            })
        })
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "open errors are returned"
        );

        let cap = Capture::start(large_ring(), || TestPattern::new(pattern(20))).unwrap();
        let mut seen = Vec::new();
        let mut cursor = None;
        while seen.len() < 20 {
            let frames = cap.wait_frames_after(cursor, Duration::from_secs(5));
            assert!(!frames.is_empty(), "frames stopped early");
            cursor = frames.last().map(|f| f.seq);
            seen.extend(frames);
        }
        assert!(
            cap.wait_frames_after(cursor, Duration::from_secs(5))
                .is_empty()
        );
        assert!(matches!(cap.status(), CaptureStatus::Failed(e) if matches!(*e, Error::Ended)));
        let first = seen[0].seq;
        let seqs: Vec<u64> = seen.iter().map(|f| f.seq - first).collect();
        assert_eq!(seqs, (0..20).collect::<Vec<_>>(), "consecutive");
        let stats = cap.stats();
        assert_eq!(stats.frames, 20);
        assert_eq!(stats.dropped, 0);
        assert_eq!(stats.ring_frames, 20);
        assert!((stats.fps - 100.0).abs() < 1e-6, "{}", stats.fps);
        assert_eq!(cap.latest().unwrap().seq, first + 19);
        assert_eq!(cap.info().id, crate::test_pattern::TEST_PATTERN_ID);
    }

    #[test]
    fn stop_ends_a_hung_read_through_the_interrupter() {
        let config = CaptureConfig {
            stall: Duration::ZERO,
            ..CaptureConfig::default()
        };
        let (reading, in_read) = mpsc::channel();
        let mut cap = Capture::start(config, || {
            let pattern = TestPattern::new(pattern(1))?;
            Ok(Hung {
                interrupted: Arc::new((Mutex::new(false), Condvar::new())),
                reading,
                info: pattern.info().clone(),
                format: pattern.format().clone(),
            })
        })
        .unwrap();
        in_read.recv().unwrap();
        cap.request_stop();
        assert!(matches!(cap.status(), CaptureStatus::Stopped));
        // The thread exiting is the proof: a hung read would never return.
        let thread = cap.thread.take().unwrap();
        thread.join().unwrap();
    }
}
