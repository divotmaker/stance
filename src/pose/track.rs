//! Keypoints of a whole clip, stored beside it as JSON.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use super::geometry::BBox;
use super::image::Rotation;
use super::skeleton::Skeleton;
use super::{Keypoint, Pose};
use crate::{Error, Result};

/// The pose-track format version this crate writes.
pub const POSE_TRACK_VERSION: u32 = 1;

/// The golfer's raw keypoints for every frame of a clip.
///
/// Stored as `<clip>.pose.json` next to `<clip>.avi` and `<clip>.json`.
/// Keypoints are the model's output, unfiltered: smoothing is derived when
/// reading, never stored.
///
/// ```json
/// {
///   "version": 1,
///   "skeleton": "halpe26",
///   "model": "rtmpose-m-halpe26",
///   "detector": "yolox-tiny",
///   "width": 1280,
///   "height": 720,
///   "rotation": 90,
///   "frames": [
///     { "t_ns": 0, "score": 0.81, "bbox": { "x1": 512.0, "y1": 96.0, "x2": 790.0, "y2": 700.0 },
///       "keypoints": [[651.2, 160.4, 0.93], "… 26 in all"] },
///     { "t_ns": 8358000 }
///   ]
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseTrack {
    /// Format version ([`POSE_TRACK_VERSION`]).
    pub version: u32,
    /// The keypoint layout.
    pub skeleton: Skeleton,
    /// The pose model's name.
    pub model: String,
    /// The person detector's name.
    pub detector: String,
    /// Frame width in pixels, after [`PoseTrack::rotation`].
    pub width: u32,
    /// Frame height in pixels, after [`PoseTrack::rotation`].
    pub height: u32,
    /// The turn applied to the clip's frames before pose estimation
    /// ([`PoseConfig::rotation`](super::PoseConfig::rotation)): keypoints
    /// are in the turned frame. Written only when not 0; a file without it
    /// reads as 0.
    #[serde(default, skip_serializing_if = "Rotation::is_none")]
    pub rotation: Rotation,
    /// One entry per clip frame, in clip order.
    pub frames: Vec<PoseFrame>,
}

/// One frame of a [`PoseTrack`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseFrame {
    /// The frame's time relative to the clip's first frame (the sidecar's
    /// `t_ns`).
    pub t_ns: u64,
    /// Mean keypoint score; absent when no person was found or the pose
    /// scored below the clip pass's
    /// [`PoseConfig::min_pose_score`](super::PoseConfig::min_pose_score).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// The box the pose model looked in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<BBox>,
    /// `[x, y, score]` per skeleton index in image pixels; empty when the
    /// score is absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keypoints: Vec<Keypoint>,
}

impl PoseFrame {
    /// A frame with a pose.
    #[must_use]
    pub fn with_pose(t_ns: u64, pose: &Pose) -> Self {
        Self {
            t_ns,
            score: Some(pose.score),
            bbox: Some(pose.bbox),
            keypoints: pose.keypoints.clone(),
        }
    }

    /// A frame without a pose: nobody found, or only a low-score pose.
    #[must_use]
    pub fn empty(t_ns: u64) -> Self {
        Self {
            t_ns,
            score: None,
            bbox: None,
            keypoints: Vec::new(),
        }
    }

    /// The frame's pose, `None` where nobody was found.
    #[must_use]
    pub fn pose(&self, skeleton: Skeleton) -> Option<Pose> {
        (self.keypoints.len() == skeleton.len()).then(|| Pose {
            skeleton,
            keypoints: self.keypoints.clone(),
            bbox: self.bbox.unwrap_or_default(),
            score: self.score.unwrap_or_default(),
        })
    }
}

impl PoseTrack {
    /// `<dir>/<stem>.pose.json` for the clip `<dir>/<stem>.avi`.
    #[must_use]
    pub fn path_for(clip_avi: &Path) -> PathBuf {
        let stem = clip_avi
            .file_stem()
            .map_or_else(|| "clip".into(), |s| s.to_string_lossy().into_owned());
        clip_avi.with_file_name(format!("{stem}.pose.json"))
    }

    /// Serializes the track.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if serialization fails (a non-finite
    /// number).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| Error::InvalidArgument(format!("pose track: {e}")))
    }

    /// Parses a track, checking that every pose has one keypoint per
    /// skeleton index.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for malformed JSON, an unknown version or
    /// a wrong keypoint count.
    pub fn from_json(json: &str) -> Result<Self> {
        let track: Self = serde_json::from_str(json)
            .map_err(|e| Error::InvalidArgument(format!("pose track: {e}")))?;
        if track.version != POSE_TRACK_VERSION {
            return Err(Error::InvalidArgument(format!(
                "pose track version {}, this crate reads {POSE_TRACK_VERSION}",
                track.version
            )));
        }
        let n = track.skeleton.len();
        if let Some(f) = track
            .frames
            .iter()
            .find(|f| !f.keypoints.is_empty() && f.keypoints.len() != n)
        {
            return Err(Error::InvalidArgument(format!(
                "pose track frame at {} ns has {} keypoints, the skeleton has {n}",
                f.t_ns,
                f.keypoints.len()
            )));
        }
        Ok(track)
    }

    /// Writes the track to [`PoseTrack::path_for`] the clip (through a
    /// synced temporary file named for this process and a rename, so
    /// concurrent writers never interleave) and returns the path.
    ///
    /// # Errors
    ///
    /// As [`PoseTrack::to_json`]; [`Error::Io`] if writing fails.
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

    /// Reads a track written by [`PoseTrack::write_beside`].
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file can't be read; as [`PoseTrack::from_json`].
    pub fn read(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)
            .map_err(|e| Error::io(format!("reading {}", path.display()), e))?;
        Self::from_json(&json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn track() -> PoseTrack {
        let kps: Vec<Keypoint> = (0..26u8)
            .map(|i| Keypoint {
                x: f32::from(i) * 1.25 + 0.1,
                y: 400.0 - f32::from(i) / 3.0,
                score: f32::from(i) / 26.0,
            })
            .collect();
        let pose = Pose::new(Skeleton::Halpe26, kps, BBox::new(1.0, 2.0, 300.5, 400.25));
        PoseTrack {
            version: POSE_TRACK_VERSION,
            skeleton: Skeleton::Halpe26,
            model: "rtmpose-m-halpe26".into(),
            detector: "yolox-tiny".into(),
            width: 640,
            height: 480,
            rotation: Rotation::None,
            frames: vec![
                PoseFrame::with_pose(0, &pose),
                PoseFrame::empty(8_333_333),
                PoseFrame::with_pose(16_666_666, &pose),
            ],
        }
    }

    #[test]
    fn json_round_trip() {
        let t = track();
        let json = t.to_json().unwrap();
        assert!(json.contains("\"skeleton\":\"halpe26\""));
        assert!(
            json.contains("{\"t_ns\":8333333}"),
            "empty frames stay small"
        );
        assert_eq!(PoseTrack::from_json(&json).unwrap(), t);
        let pose = t.frames[0].pose(Skeleton::Halpe26).unwrap();
        assert_eq!(pose.keypoints.len(), 26);
        assert!(t.frames[1].pose(Skeleton::Halpe26).is_none());
        // Upright tracks don't write the rotation; turned ones do.
        assert!(!json.contains("rotation"));
        let turned = PoseTrack {
            rotation: Rotation::Ccw90,
            ..t.clone()
        };
        let json = turned.to_json().unwrap();
        assert!(json.contains("\"rotation\":270"), "{json}");
        assert_eq!(PoseTrack::from_json(&json).unwrap(), turned);
        let bad = json.replace("\"rotation\":270", "\"rotation\":45");
        assert!(PoseTrack::from_json(&bad).is_err());
    }

    #[test]
    fn rejects_bad_tracks() {
        let mut t = track();
        t.version = 99;
        assert!(PoseTrack::from_json(&t.to_json().unwrap()).is_err());
        let mut t = track();
        t.frames[0].keypoints.pop();
        assert!(PoseTrack::from_json(&t.to_json().unwrap()).is_err());
        assert!(PoseTrack::from_json("{").is_err());
        assert!(PoseTrack::from_json("[]").is_err());
    }

    #[test]
    fn written_beside_the_clip() {
        let dir = TempDir::new("pose-track");
        let avi = dir.path().join("shot-0001.avi");
        assert_eq!(
            PoseTrack::path_for(&avi),
            dir.path().join("shot-0001.pose.json")
        );
        let t = track();
        let path = t.write_beside(&avi).unwrap();
        assert_eq!(PoseTrack::read(&path).unwrap(), t);
        // Only the clip's track is left.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
