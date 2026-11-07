//! The swing report: events, tempo and metrics for one clip.

use std::borrow::Cow;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Deserializer, Serialize};

use super::body::{Handed, Setup, View};
use super::events::{ImpactHint, ImpactSource, SwingEvents, Tempo, detect};
use super::metrics::{BodyScale, KeyFrames, Metric, MetricId, Series, measure};
use super::posture::unmirror_keypoints;
use super::smooth::{Smoothing, smooth};
use super::view::{Guess, guess_handed, guess_view_body};
use crate::pose::PoseTrack;
use crate::{Error, Result};

/// The swing-report format version this crate writes.
pub const SWING_REPORT_VERSION: u32 = 1;

/// The largest swing report [`SwingReport::read`] accepts (bytes); a
/// report with its [`Series`] is tens of kilobytes (about 130 bytes per
/// frame face-on).
pub const MAX_SWING_REPORT_BYTES: u64 = 4 << 20;

/// What the caller knows about the swing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SwingOptions {
    /// The camera view; `None` guesses it from the address pose. When the
    /// guess is unsure too, only the metrics that work in either view are
    /// measured.
    pub view: Option<View>,
    /// The golfer's handedness.
    pub handed: Handed,
    /// The impact time on the track's time base (ns from the first frame),
    /// typically the launch monitor's shot time. It is a prior: the impact
    /// is refined to the hands' low point within 0.3 s of it
    /// ([`SwingEvents`]); with no clear low point there, the report has the
    /// impact alone ([`ImpactSource::Given`]).
    /// `None` estimates it from the hands.
    pub impact_ns: Option<u64>,
    /// The golfer's height (cm), for distances in centimeters.
    pub height_cm: Option<f64>,
    /// The track comes from a mirrored image (a selfie-style preview): it is
    /// unmirrored before analysis, so [`SwingOptions::handed`] is the
    /// golfer's true hand and the report describes the real swing. `false`
    /// when absent from JSON.
    #[serde(default)]
    pub mirrored: bool,
    /// Keypoint cleaning; the default when absent from JSON.
    #[serde(default)]
    pub smoothing: Smoothing,
}

impl Default for SwingOptions {
    /// View guessed, right-handed, impact estimated, no height, not
    /// mirrored.
    fn default() -> Self {
        Self {
            view: None,
            handed: Handed::Right,
            impact_ns: None,
            height_cm: None,
            mirrored: false,
            smoothing: Smoothing::default(),
        }
    }
}

/// The analysis of one swing, stored beside its clip as `<clip>.swing.json`.
///
/// ```json
/// {
///   "version": 1,
///   "view": "face_on",
///   "handed": "right",
///   "view_guess": { "value": "face_on", "confidence": 0.8 },
///   "handed_guess": { "value": "right", "confidence": 1.0 },
///   "smoothing": { "min_score": 0.3, "max_gap_ns": 150000000, "cutoff_hz": 8.0, "order": 2 },
///   "scale": { "torso_px": 200.0, "upright_px": 230.0, "cm_per_px": 0.22 },
///   "events": {
///     "address": { "t_ns": 1000000000, "frame": 120, "confidence": 0.8 },
///     "takeaway": { "t_ns": 1008333333, "frame": 121, "confidence": 0.8 },
///     "arm_parallel_back": { "t_ns": 1560000000, "frame": 187, "confidence": 0.9 },
///     "top": { "t_ns": 1850000000, "frame": 222, "confidence": 0.9 },
///     "arm_parallel_down": { "t_ns": 2050000000, "frame": 246, "confidence": 0.9 },
///     "impact": { "t_ns": 2150000000, "frame": 258, "confidence": 0.9 },
///     "impact_source": "refined",
///     "finish": { "t_ns": 2800000000, "frame": 336, "confidence": 0.7 }
///   },
///   "tempo": { "backswing_ns": 841666667, "downswing_ns": 300000000, "ratio": 2.81 },
///   "metrics": [
///     { "id": "hip_sway", "view": "face_on", "unit": "torso",
///       "values": { "arm_parallel_back": -0.04, "top": -0.05, "arm_parallel_down": 0.05, "impact": 0.12 },
///       "cm": { "arm_parallel_back": -2.0, "top": -2.5, "arm_parallel_down": 2.5, "impact": 6.0 },
///       "confidence": 0.8 },
///     { "id": "lead_arm_angle", "view": "face_on", "unit": "degrees",
///       "values": { "address": 175.0, "top": 150.0, "impact": 176.0 }, "confidence": 0.8 }
///   ],
///   "series": {
///     "t_ns": [0, 8333333, 16666667],
///     "metrics": [
///       { "id": "hip_sway", "unit": "torso", "values": [null, 0.0, -0.004] },
///       { "id": "lead_arm_angle", "unit": "degrees", "values": [174.2, 175.0, 175.1] }
///     ]
///   }
/// }
/// ```
///
/// Reading skips metrics (and series) with an id this crate doesn't know
/// (written by a later version), so new metrics don't need a new format
/// version; a report without `series` reads with an empty one, and one
/// without the lead-arm-parallel events and positions or
/// `scale.upright_px` reads with them `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SwingReport {
    /// Format version ([`SWING_REPORT_VERSION`]).
    pub version: u32,
    /// The view the metrics were measured for; `None` when it was neither
    /// given nor guessed with confidence (only either-view metrics then).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<View>,
    /// The golfer's handedness (the true hand, also for a mirrored image).
    pub handed: Handed,
    /// Whether the track came from a mirrored image
    /// ([`SwingOptions::mirrored`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub mirrored: bool,
    /// The view the address pose suggests (whether or not the caller gave
    /// one); `None` when the pose is ambiguous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_guess: Option<Guess<View>>,
    /// The handedness the swing suggests for [`SwingReport::view`];
    /// disagreeing with [`SwingReport::handed`] hints at a wrong hand or a
    /// mirrored image that wasn't declared. Face-on it needs a top with
    /// confidence 0.5 or more; `None` for an unconfirmed impact
    /// ([`ImpactSource::Given`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handed_guess: Option<Guess<Handed>>,
    /// The golfer's height (cm), when given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height_cm: Option<f64>,
    /// The keypoint cleaning used.
    #[serde(default)]
    pub smoothing: Smoothing,
    /// The body scale; `None` when the torso was never visible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<BodyScale>,
    /// The swing's key moments.
    pub events: SwingEvents,
    /// Backswing against downswing, when the takeaway, top and impact were
    /// found ([`Tempo::from_events`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo: Option<Tempo>,
    /// The view's metrics, in [`MetricId::ALL`] order; a metric with no
    /// value at any position is left out.
    #[serde(deserialize_with = "known_metrics")]
    pub metrics: Vec<Metric>,
    /// The view's metrics at every frame ([`Series`]); empty when no
    /// metric was measured.
    #[serde(default, skip_serializing_if = "Series::is_empty")]
    pub series: Series,
}

/// The metrics with an id this crate knows; malformed known metrics are
/// errors.
fn known_metrics<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<Metric>, D::Error> {
    let raw = Vec::<serde_json::Value>::deserialize(d)?;
    let mut out = Vec::with_capacity(raw.len());
    for value in raw {
        let known = value
            .get("id")
            .is_some_and(|id| MetricId::deserialize(id).is_ok());
        if known {
            out.push(Metric::deserialize(value).map_err(serde::de::Error::custom)?);
        }
    }
    Ok(out)
}

/// The track as an unmirrored camera would have seen it.
fn unmirror_track(track: &PoseTrack) -> PoseTrack {
    let mut out = track.clone();
    // Pixel widths fit f32.
    #[allow(clippy::cast_precision_loss, reason = "pixel widths fit f32")]
    let width = track.width as f32;
    for f in &mut out.frames {
        f.keypoints = unmirror_keypoints(track.skeleton, &f.keypoints, width);
        if let Some(b) = &mut f.bbox {
            let (x1, x2) = (b.x1, b.x2);
            b.x1 = width - x2;
            b.x2 = width - x1;
        }
    }
    out
}

/// Analyzes one swing from its clip's pose track.
///
/// 1. A mirrored track ([`SwingOptions::mirrored`]) is unmirrored.
/// 2. Keypoints are cleaned and smoothed ([`Smoothing`]).
/// 3. The impact is found from the hands ([`SwingEvents`]): the given time
///    refined against them, or estimated.
/// 4. The track is smoothed again, split at that impact, and the other
///    events are found.
/// 5. The view is guessed from the address pose when not given, and
///    handedness is checked.
/// 6. The view's metrics are measured at address, P3, top, P5, impact and
///    finish ([`MetricId`]), and at every frame ([`Series`]).
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a skeleton other than Halpe-26, frame
/// times that go backwards, bad [`Smoothing`] filter parameters, or a track
/// with fewer than three frames with a person.
pub fn analyze_swing(track: &PoseTrack, options: &SwingOptions) -> Result<SwingReport> {
    let posed = track
        .frames
        .iter()
        .filter(|f| !f.keypoints.is_empty())
        .count();
    if posed < 3 {
        return Err(Error::InvalidArgument(format!(
            "swing analysis needs at least 3 frames with a person, the track has {posed}"
        )));
    }
    let track = if options.mirrored {
        Cow::Owned(unmirror_track(track))
    } else {
        Cow::Borrowed(track)
    };
    let handed = options.handed;
    // The impact is found on the track smoothed in one piece, so where a
    // given time falls doesn't move it; the events are then found on the
    // track smoothed apart either side of that impact.
    let mut frames = smooth(&track, handed, &options.smoothing, None)?;
    let hint = options
        .impact_ns
        .map_or(ImpactHint::Estimate, ImpactHint::Prior);
    let mut events = detect(&frames, hint);
    if let Some(found) = events.impact {
        frames = smooth(&track, handed, &options.smoothing, Some(found.t_ns))?;
        events = detect(&frames, ImpactHint::Known(found, events.impact_source));
    }
    let keys = KeyFrames {
        address: events.address.map(|e| e.frame),
        arm_parallel_back: events.arm_parallel_back.map(|e| e.frame),
        top: events.top.map(|e| e.frame),
        arm_parallel_down: events.arm_parallel_down.map(|e| e.frame),
        impact: events.impact.map(|e| e.frame),
        finish: events.finish.map(|e| e.frame),
    };
    // The address pose, else the last frame with a torso before the top.
    let reference = keys.address.or_else(|| {
        let end = keys.top.or(keys.impact).unwrap_or(frames.len());
        (0..end).rev().find(|&i| frames.bodies[i].torso().is_some())
    });
    let view_guess = reference.and_then(|i| guess_view_body(&frames.bodies[i]));
    let view = options.view.or(view_guess.map(|g| g.value));
    // An unconfirmed impact found no swing to check the hand against.
    let swing_found = events.impact_source != ImpactSource::Given;
    let handed_guess = view.filter(|_| swing_found).and_then(|view| {
        guess_handed(
            view,
            reference.map(|i| &frames.bodies[i]),
            events.top.map(|e| (&frames.bodies[e.frame], e.confidence)),
        )
    });
    let scale = BodyScale::measure(&frames, keys.address, options.height_cm);
    let (metrics, series) = scale.map_or_else(
        || (Vec::new(), Series::default()),
        |s| measure(&frames, keys, view.map(|view| Setup { view, handed }), s),
    );
    Ok(SwingReport {
        version: SWING_REPORT_VERSION,
        view,
        handed,
        mirrored: options.mirrored,
        view_guess,
        handed_guess,
        height_cm: options.height_cm,
        smoothing: options.smoothing,
        scale,
        tempo: Tempo::from_events(&events),
        events,
        metrics,
        series,
    })
}

/// Just the version, read before the rest so a newer format gives the
/// version message rather than a parse error.
#[derive(Deserialize)]
struct VersionOnly {
    version: u32,
}

impl SwingReport {
    /// The view and handedness the metrics were measured for; `None` when
    /// the view is unknown.
    #[must_use]
    pub fn setup(&self) -> Option<Setup> {
        self.view.map(|view| Setup {
            view,
            handed: self.handed,
        })
    }

    /// A metric by id.
    #[must_use]
    pub fn metric(&self, id: MetricId) -> Option<&Metric> {
        self.metrics.iter().find(|m| m.id == id)
    }

    /// `<dir>/<stem>.swing.json` for the clip `<dir>/<stem>.avi`.
    #[must_use]
    pub fn path_for(clip_avi: &Path) -> PathBuf {
        let stem = clip_avi
            .file_stem()
            .map_or_else(|| "clip".into(), |s| s.to_string_lossy().into_owned());
        clip_avi.with_file_name(format!("{stem}.swing.json"))
    }

    /// Serializes the report.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if serialization fails (a non-finite
    /// number).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| Error::InvalidArgument(format!("swing report: {e}")))
    }

    /// Parses a report. Metrics with an unknown id are skipped.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for malformed JSON or an unknown version.
    pub fn from_json(json: &str) -> Result<Self> {
        let bad = |e: serde_json::Error| Error::InvalidArgument(format!("swing report: {e}"));
        let version = serde_json::from_str::<VersionOnly>(json)
            .map_err(bad)?
            .version;
        if version != SWING_REPORT_VERSION {
            return Err(Error::InvalidArgument(format!(
                "swing report version {version}, this crate reads {SWING_REPORT_VERSION}"
            )));
        }
        serde_json::from_str(json).map_err(bad)
    }

    /// Writes the report to [`SwingReport::path_for`] the clip (through a
    /// synced temporary file named for this process and a rename, so
    /// concurrent writers never interleave) and returns the path.
    ///
    /// # Errors
    ///
    /// As [`SwingReport::to_json`]; [`Error::Io`] if writing fails.
    pub fn write_beside(&self, clip_avi: &Path) -> Result<PathBuf> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = Self::path_for(clip_avi);
        let part = path.with_extension(format!(
            "json.part-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let json = self.to_json()?;
        let written = (|| {
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&part)?;
            f.write_all(json.as_bytes())?;
            f.sync_all()
        })();
        if let Err(e) = written {
            let _ = fs::remove_file(&part);
            return Err(Error::io(format!("writing {}", part.display()), e));
        }
        fs::rename(&part, &path)
            .map_err(|e| Error::io(format!("renaming {}", part.display()), e))?;
        Ok(path)
    }

    /// Reads a report written by [`SwingReport::write_beside`], at most
    /// [`MAX_SWING_REPORT_BYTES`] long.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file can't be read; [`Error::InvalidArgument`]
    /// if it is too long or not UTF-8; as [`SwingReport::from_json`].
    pub fn read(path: &Path) -> Result<Self> {
        let io = |e| Error::io(format!("reading {}", path.display()), e);
        let mut bytes = Vec::new();
        fs::File::open(path)
            .map_err(io)?
            .take(MAX_SWING_REPORT_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(io)?;
        if bytes.len() as u64 > MAX_SWING_REPORT_BYTES {
            return Err(Error::InvalidArgument(format!(
                "{} is longer than {MAX_SWING_REPORT_BYTES} bytes",
                path.display()
            )));
        }
        let json = String::from_utf8(bytes)
            .map_err(|e| Error::InvalidArgument(format!("swing report {}: {e}", path.display())))?;
        Self::from_json(&json)
    }
}
