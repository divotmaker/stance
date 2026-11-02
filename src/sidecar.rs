//! The JSON sidecar written next to each clip.
//!
//! It records capture facts only: timing, size and where each frame's JPEG
//! sits inside the AVI file, so a client can fetch single frames with HTTP
//! `Range` requests instead of downloading the whole clip. Builds on every
//! target, including wasm.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// The sidecar format version this crate writes.
pub const SIDECAR_VERSION: u32 = 1;

/// Capture facts about one clip.
///
/// ```
/// use stance::sidecar::Sidecar;
///
/// let json = r#"{
///   "version": 1, "fourcc": "MJPG", "width": 1280, "height": 720,
///   "fps": { "nominal": 120.0, "measured": 119.6 },
///   "start_ns": 5000000000,
///   "frames": [
///     { "t_ns": 0, "offset": 232, "len": 41000 },
///     { "t_ns": 8360000, "offset": 41240, "len": 40876 }
///   ],
///   "trigger_index": 1, "impact_index": 0, "trigger_ns": 8000000, "impact_ns": 0,
///   "complete": true, "dropped": 0, "truncated": false
/// }"#;
/// let sidecar = Sidecar::from_json(json)?;
/// let trigger = sidecar.trigger_frame().expect("frames reached the trigger");
/// // Fetch the trigger frame: `Range: bytes=41240-82115`.
/// assert_eq!((trigger.offset, trigger.end()), (41240, 82116));
/// # Ok::<(), stance::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sidecar {
    /// Sidecar format version ([`SIDECAR_VERSION`]).
    pub version: u32,
    /// Pixel format of the frames (`MJPG`).
    pub fourcc: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rates.
    pub fps: SidecarFps,
    /// Host monotonic time of the first frame (ns). Comparable between clips
    /// captured on the same host, e.g. to line up two cameras.
    pub start_ns: u64,
    /// Every frame, in file order. Never empty; offsets increase and the
    /// frames don't overlap.
    pub frames: Vec<SidecarFrame>,
    /// Index of the first frame at or after the first trigger, or
    /// `frames.len()` when the clip ends before the trigger (frames stopped
    /// between the impact and a late trigger). Use
    /// [`Sidecar::trigger_frame`] to index safely.
    pub trigger_index: usize,
    /// Index of the first frame at or after the impact estimate. Always a
    /// valid index.
    pub impact_index: usize,
    /// Time of the first trigger relative to the first frame (ns).
    pub trigger_ns: i64,
    /// Time of the impact estimate relative to the first frame (ns; negative
    /// if it precedes the clip).
    pub impact_ns: i64,
    /// The whole window was captured, with no frames missing across the
    /// trigger or the impact.
    pub complete: bool,
    /// Frames missing inside the clip.
    pub dropped: u32,
    /// A later trigger from the same source cut the clip short.
    pub truncated: bool,
}

/// Nominal and measured frame rates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SidecarFps {
    /// The camera mode's rate, if known.
    pub nominal: Option<f64>,
    /// Frames per second actually captured over the clip.
    pub measured: f64,
}

/// One frame's time and location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarFrame {
    /// Host time relative to the first frame (ns).
    pub t_ns: u64,
    /// Byte offset of the frame's JPEG data inside the AVI file.
    pub offset: u64,
    /// Length of the JPEG data in bytes.
    pub len: u32,
}

impl SidecarFrame {
    /// One past the last byte of the JPEG data (saturating at `u64::MAX`;
    /// [`Sidecar::validate`] rejects frames where that happens).
    #[must_use]
    pub fn end(&self) -> u64 {
        self.offset.saturating_add(u64::from(self.len))
    }
}

impl Sidecar {
    /// Parses a sidecar.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSidecar`] for malformed JSON, a newer major version, or
    /// indices that don't point at a frame.
    pub fn from_json(json: &str) -> Result<Self> {
        let s: Self =
            serde_json::from_str(json).map_err(|e| Error::InvalidSidecar(e.to_string()))?;
        s.validate()?;
        Ok(s)
    }

    /// Serialises the sidecar as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSidecar`] if serialisation fails (non-finite rates).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| Error::InvalidSidecar(e.to_string()))
    }

    /// Checks the version, that the indices point at frames, and that frame
    /// offsets increase without overlapping or overflowing.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSidecar`] describing the first problem found.
    pub fn validate(&self) -> Result<()> {
        if self.version == 0 || self.version > SIDECAR_VERSION {
            return Err(Error::InvalidSidecar(format!(
                "version {} (this library reads 1..={SIDECAR_VERSION})",
                self.version
            )));
        }
        if self.frames.is_empty() {
            return Err(Error::InvalidSidecar("no frames".into()));
        }
        let n = self.frames.len();
        if self.trigger_index > n || self.impact_index >= n {
            return Err(Error::InvalidSidecar(format!(
                "trigger/impact index out of range ({n} frames)"
            )));
        }
        let mut end = 0u64;
        for (i, f) in self.frames.iter().enumerate() {
            if f.offset < end {
                return Err(Error::InvalidSidecar(format!(
                    "frame {i} starts at {} inside the previous frame (which ends at {end})",
                    f.offset
                )));
            }
            end = f
                .offset
                .checked_add(u64::from(f.len))
                .ok_or_else(|| Error::InvalidSidecar(format!("frame {i} ends past 2^64")))?;
        }
        Ok(())
    }

    /// The frame at the first trigger, if the clip reaches it.
    #[must_use]
    pub fn trigger_frame(&self) -> Option<&SidecarFrame> {
        self.frames.get(self.trigger_index)
    }

    /// The frame at the impact estimate.
    #[must_use]
    pub fn impact_frame(&self) -> Option<&SidecarFrame> {
        self.frames.get(self.impact_index)
    }

    /// Duration from the first to the last frame (ns).
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.frames.last().map_or(0, |f| f.t_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Sidecar {
        Sidecar {
            version: SIDECAR_VERSION,
            fourcc: "MJPG".into(),
            width: 640,
            height: 480,
            fps: SidecarFps {
                nominal: Some(30.0),
                measured: 29.9,
            },
            start_ns: 7,
            frames: vec![
                SidecarFrame {
                    t_ns: 0,
                    offset: 232,
                    len: 10,
                },
                SidecarFrame {
                    t_ns: 33_000_000,
                    offset: 250,
                    len: 11,
                },
            ],
            trigger_index: 1,
            impact_index: 0,
            trigger_ns: 30_000_000,
            impact_ns: -5,
            complete: true,
            dropped: 0,
            truncated: false,
        }
    }

    #[test]
    fn json_round_trip() {
        let s = sample();
        let json = s.to_json().unwrap();
        assert!(json.contains(r#""nominal": 30.0"#), "{json}");
        assert_eq!(Sidecar::from_json(&json).unwrap(), s);
        assert_eq!(s.duration_ns(), 33_000_000);
        assert_eq!(s.frames[1].end(), 261);
    }

    #[test]
    fn rejects_bad_sidecars() {
        let mut s = sample();
        s.version = 2;
        assert!(Sidecar::from_json(&s.to_json().unwrap()).is_err());
        let mut s = sample();
        s.trigger_index = 3;
        assert!(Sidecar::from_json(&s.to_json().unwrap()).is_err());
        let mut s = sample();
        s.impact_index = 2;
        assert!(s.validate().is_err());
        let mut s = sample();
        s.frames.clear();
        assert!(s.validate().is_err());
        assert!(Sidecar::from_json("{}").is_err());
    }

    #[test]
    fn a_clip_ending_before_its_trigger_is_valid() {
        let mut s = sample();
        s.trigger_index = 2;
        s.validate().unwrap();
        assert!(s.trigger_frame().is_none());
        assert_eq!(s.impact_frame(), Some(&s.frames[0]));
    }

    #[test]
    fn rejects_overflowing_and_overlapping_offsets() {
        let mut s = sample();
        s.frames[1].offset = u64::MAX;
        assert_eq!(s.frames[1].end(), u64::MAX, "saturates, no panic");
        assert!(matches!(s.validate(), Err(Error::InvalidSidecar(_))));
        let mut s = sample();
        s.frames[1].offset = u64::MAX - 5;
        assert!(s.validate().is_err(), "ends past u64::MAX");
        let mut s = sample();
        s.frames[1].offset = 235; // inside frame 0 (232..242)
        assert!(s.validate().is_err());
        let mut s = sample();
        s.frames.swap(0, 1);
        assert!(s.validate().is_err(), "offsets go backwards");
        let mut s = sample();
        s.frames[1].offset = 242; // directly after frame 0
        s.validate().unwrap();
    }
}
