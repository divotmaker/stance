//! USB camera capture for swing replay.
//!
//! `stance` reads MJPEG frames from USB (UVC) cameras, keeps the last
//! few seconds in a pre-roll buffer, cuts a clip around each shot trigger and
//! writes it as an MJPEG AVI with a JSON sidecar. Frames are stored exactly as
//! the camera compressed them: no decoding or re-encoding on the capture path.
//!
//! - [`list`] and [`open`] enumerate and open cameras (Media Foundation on
//!   Windows, V4L2 on Linux). [`TestPattern`] and [`AviFile`] are sources too;
//!   all of them implement [`Source`].
//! - [`Capture`] runs a source on its own thread and fills a bounded [`Ring`].
//!   Starting and [`Capture::stop`] block; [`Capture::request_stop`] doesn't.
//! - [`ClipRecorder`] turns frames and [`Trigger`]s into [`Clip`]s. It is pure:
//!   no threads, no clock of its own. Callers call [`ClipRecorder::poll`] on
//!   a timer and deduplicate triggers.
//! - [`write_clip`] stores a clip; [`sidecar::Sidecar`] describes it, with
//!   the byte offset of every frame inside the AVI.
//! - [`jpeg::ensure_huffman`] makes webcam frames decodable by strict
//!   decoders.
//! - With the `frp` feature, [`frp`] maps Flight Relay Protocol
//!   `shot_trigger` events onto triggers.
//! - With the `pose` feature (alpha), `pose` estimates body keypoints
//!   (RTMPose, pure-Rust inference): live per frame, or over a recorded clip
//!   on several threads. Models are downloaded separately and verified by
//!   SHA-256; `pose-download` adds fetching them over HTTPS. `golf` turns
//!   keypoints into swing events, tempo and per-view body metrics (at the
//!   key positions and per frame).
//!
//! All times are host monotonic nanoseconds from [`clock::now_ns`].
//!
//! # Platforms
//!
//! Camera capture works on Windows and 64-bit Linux. On macOS [`list`] and
//! [`open`] return [`Error::Unsupported`]; everything else works. On wasm32
//! the crate builds the platform-neutral parts only: frame and format types,
//! [`Ring`], [`ClipRecorder`], [`sidecar`], [`jpeg`] and the camera types
//! ([`list`] and [`open`] return [`Error::Unsupported`]).
//!
//! # Example
//!
//! ```
//! use std::time::Duration;
//! use stance::{
//!     Capture, CaptureConfig, ClipRecorder, RecorderConfig, RecorderEvent, TestPattern,
//!     TestPatternConfig, Trigger,
//! };
//!
//! let pattern = TestPatternConfig { width: 64, height: 48, fps: 100, ..TestPatternConfig::default() };
//! let capture = Capture::start(CaptureConfig::default(), move || TestPattern::new(pattern))?;
//! let mut recorder = ClipRecorder::new(RecorderConfig {
//!     pre_roll: Duration::from_millis(100),
//!     post_roll: Duration::from_millis(100),
//!     ..RecorderConfig::default()
//! });
//!
//! let mut cursor = None;
//! let mut triggered = false;
//! let clip = 'capture: loop {
//!     for frame in capture.wait_frames_after(cursor, Duration::from_millis(100)) {
//!         cursor = Some(frame.seq);
//!         for event in recorder.push(frame) {
//!             if let RecorderEvent::Finished(clip) = event {
//!                 break 'capture clip;
//!             }
//!         }
//!     }
//!     if !triggered && recorder.ring().span() >= Duration::from_millis(150) {
//!         // A launch monitor reports a shot: stamp it on arrival.
//!         let now = stance::clock::now_ns();
//!         recorder.trigger(Trigger::new(now, "launch monitor", "shot 1"));
//!         triggered = true;
//!     }
//! };
//! assert_eq!(clip.tags, ["shot 1"]);
//! assert!(clip.complete);
//! let dir = std::env::temp_dir().join("stance-doc");
//! let files = stance::write_clip(&dir, "shot-1", &clip)?;
//! assert_eq!(files.sidecar.frames.len(), clip.frames.len());
//! # std::fs::remove_dir_all(&dir).ok();
//! # Ok::<(), stance::Error>(())
//! ```

#[cfg(not(target_family = "wasm"))]
pub mod avi;
pub mod camera;
#[cfg(not(target_family = "wasm"))]
mod capture;
#[cfg(not(target_family = "wasm"))]
pub mod clock;
mod error;
mod frame;
#[cfg(feature = "frp")]
pub mod frp;
#[cfg(all(feature = "pose", not(target_family = "wasm")))]
pub mod golf;
pub mod jpeg;
#[cfg(all(feature = "pose", not(target_family = "wasm")))]
pub mod pose;
mod recorder;
mod ring;
pub mod sidecar;
mod source;
#[cfg(not(target_family = "wasm"))]
mod storage;
#[cfg(not(target_family = "wasm"))]
mod test_pattern;
#[cfg(test)]
mod testutil;

#[cfg(not(target_family = "wasm"))]
pub use avi::AviFile;
pub use camera::{Camera, CameraInfo, FormatRequest, choose_format, list, open};
#[cfg(not(target_family = "wasm"))]
pub use capture::{Capture, CaptureConfig, CaptureStats, CaptureStatus};
pub use error::{Error, Result};
pub use frame::{Format, Frame};
pub use recorder::{
    Clip, ClipId, ClipRecorder, DiscardReason, Discarded, RecorderConfig, RecorderEvent, Trigger,
    TriggerOutcome,
};
pub use ring::Ring;
pub use source::{Interrupter, Source};
#[cfg(not(target_family = "wasm"))]
pub use storage::{ClipFiles, open_clip, read_sidecar, write_clip};
#[cfg(not(target_family = "wasm"))]
pub use test_pattern::{TEST_PATTERN_ID, TEST_PATTERN_NAME, TestPattern, TestPatternConfig};

/// This crate's version, e.g. for recording which `stance` analyzed a clip.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compiles the README's examples as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
