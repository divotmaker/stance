//! Pose estimation (`pose` feature, alpha): body keypoints from camera
//! frames, live and over recorded clips.
//!
//! The pipeline is top-down, as rtmlib runs it: a person detector
//! ([`Yolox`]) finds the golfer, a pose model ([`RtmPose`]) finds 26 Halpe
//! keypoints (including heels and toes) inside that box. Inference runs on
//! the CPU in pure Rust (tract), so applications stay a single executable.
//!
//! - [`decode_jpeg`] turns a frame into an [`Image`].
//! - [`PoseTracker`] follows one golfer frame by frame, for a live view: it
//!   detects on the first frame, every few frames and whenever tracking is
//!   lost, and otherwise crops around the previous frame's keypoints.
//!   Optional One Euro smoothing.
//! - [`analyze_clip`] runs a recorded clip on several threads and returns a
//!   [`PoseTrack`] (raw keypoints per frame, on the sidecar's time base),
//!   which [`PoseTrack::write_beside`] stores as `<clip>.pose.json`.
//! - A pose scoring below [`PoseConfig::min_pose_score`] is absent: `None`
//!   live, an empty frame in a track.
//! - [`OneEuro`] and [`Butterworth`] smooth keypoint series; filtering is
//!   derived from a track, never stored in it.
//! - [`draw`] annotates images: skeletons, lines, disks and text.
//!
//! The pipeline is generic over [`PoseEngine`] and [`PersonDetector`]:
//! [`PoseModels::load`] gives the default pair, [`PoseModels::new`] any
//! other.
//!
//! # Models
//!
//! Models are not bundled. [`ModelSpec`] lists each model's source, SHA-256
//! and input size; loading reads `<dir>/<name>.onnx` and verifies the hash.
//! With the `pose-download` feature, [`fetch_model`] downloads a model over
//! HTTPS into a cache directory first; [`fetch_model_with`] also reports
//! its progress and can be canceled.
//!
//! | Model | Use | License |
//! |---|---|---|
//! | [`RTMPOSE_M_HALPE26`] (default), [`RTMPOSE_S_HALPE26`] | keypoints | Apache-2.0 weights (OpenMMLab), trained on public datasets with mixed licenses |
//! | [`YOLOX_TINY`] (default), [`YOLOX_NANO`] | person boxes | Apache-2.0 (Megvii), COCO |
//!
//! # Attribution
//!
//! The RTMPose pre- and post-processing, the YOLOX decoding and the tracking
//! scheme are ported from rtmlib (<https://github.com/Tau-J/rtmlib>,
//! Apache-2.0, by Tau-J and contributors), which derives them from
//! `MMPose` (OpenMMLab) and YOLOX (Megvii), both Apache-2.0; see `NOTICE`.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use stance::pose::{PoseConfig, PoseModels, decode_jpeg};
//!
//! let models = PoseModels::load(Path::new("models"), &PoseConfig::default())?;
//! let mut tracker = models.tracker(PoseConfig::default())?;
//! let (mut avi, sidecar) = stance::open_clip(Path::new("clips"), "shot-0001")?;
//! for (i, frame) in sidecar.frames.iter().enumerate() {
//!     let image = decode_jpeg(&avi.read_frame(i)?)?;
//!     let pose = tracker.track(&image, frame.t_ns)?;
//!     if let Some(wrist) = pose.as_ref().and_then(|p| p.keypoint("left_wrist")) {
//!         println!("{} ns: left wrist at ({:.0}, {:.0})", frame.t_ns, wrist.x, wrist.y);
//!     }
//! }
//! # Ok::<(), stance::Error>(())
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};

mod clip;
#[cfg(feature = "pose-download")]
mod download;
pub mod draw;
mod filter;
mod geometry;
mod image;
mod model;
mod rtmpose;
mod runtime;
mod skeleton;
mod track;
mod tracker;
mod yolox;

pub use clip::{WorkerHook, analyze_clip, analyze_frames};
#[cfg(feature = "pose-download")]
pub use download::{FetchProgress, fetch_model, fetch_model_with};
pub use filter::{Butterworth, OneEuro, OneEuroParams};
pub use geometry::{BBox, Detection};
pub use image::{Image, MAX_DECODE_DIMENSION, Rotation, decode_jpeg};
pub use model::{
    MODELS, ModelKind, ModelSpec, RTMPOSE_M_HALPE26, RTMPOSE_S_HALPE26, YOLOX_NANO, YOLOX_TINY,
};
pub use rtmpose::RtmPose;
pub use runtime::InferencePool;
pub use skeleton::{Side, Skeleton};
pub use track::{POSE_TRACK_VERSION, PoseFrame, PoseTrack};
pub use tracker::PoseTracker;
pub use yolox::Yolox;

use crate::Result;

/// One keypoint in source-image pixels.
///
/// Serialized as `[x, y, score]`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(from = "[f32; 3]", into = "[f32; 3]")]
pub struct Keypoint {
    /// Column, pixels from the left edge.
    pub x: f32,
    /// Row, pixels from the top edge.
    pub y: f32,
    /// Model confidence, roughly `[0, 1]`.
    pub score: f32,
}

impl From<[f32; 3]> for Keypoint {
    fn from([x, y, score]: [f32; 3]) -> Self {
        Self { x, y, score }
    }
}

impl From<Keypoint> for [f32; 3] {
    fn from(k: Keypoint) -> Self {
        [k.x, k.y, k.score]
    }
}

/// One person's keypoints in one image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// The keypoint layout.
    pub skeleton: Skeleton,
    /// One keypoint per skeleton index.
    pub keypoints: Vec<Keypoint>,
    /// The box the pose model looked in.
    pub bbox: BBox,
    /// Mean keypoint score.
    pub score: f32,
}

impl Pose {
    /// A pose whose score is the mean of its keypoint scores.
    #[must_use]
    pub fn new(skeleton: Skeleton, keypoints: Vec<Keypoint>, bbox: BBox) -> Self {
        let score = mean_score(&keypoints);
        Self {
            skeleton,
            keypoints,
            bbox,
            score,
        }
    }

    /// A keypoint by name.
    #[must_use]
    pub fn keypoint(&self, name: &str) -> Option<&Keypoint> {
        self.skeleton
            .index(name)
            .and_then(|i| self.keypoints.get(i))
    }

    /// The box around the keypoints scoring at least `min_score` (all of
    /// them when fewer than three do), scaled by `expansion` about its
    /// center: where to look for the person in the next frame.
    #[must_use]
    pub fn keypoint_box(&self, min_score: f32, expansion: f32) -> Option<BBox> {
        let confident = self.keypoints.iter().filter(|k| k.score >= min_score);
        let points: Vec<(f32, f32)> = if confident.clone().count() >= 3 {
            confident.map(|k| (k.x, k.y)).collect()
        } else {
            self.keypoints.iter().map(|k| (k.x, k.y)).collect()
        };
        BBox::around(points)
            .map(|b| b.expand(expansion))
            .filter(BBox::is_valid)
    }
}

fn mean_score(keypoints: &[Keypoint]) -> f32 {
    if keypoints.is_empty() {
        return 0.0;
    }
    // Keypoint counts are tiny.
    #[allow(clippy::cast_precision_loss)]
    let n = keypoints.len() as f32;
    keypoints.iter().map(|k| k.score).sum::<f32>() / n
}

/// How much the box around the keypoints grows to give the next crop
/// (rtmlib's `pose_to_bbox` expansion).
pub(crate) const KEYPOINT_BOX_EXPANSION: f32 = 1.25;
/// A keypoint box smaller than this fraction of the detector's box (by area)
/// has collapsed.
const MIN_CROP_AREA_FRACTION: f32 = 0.36;

/// Where to look for the person next: the box around `pose`'s keypoints
/// scoring at least `keypoint_threshold`, or `None` when it has collapsed
/// onto a part of the body and the detector's box should be used instead.
///
/// It has collapsed when fewer than half the keypoints score, or when it
/// covers less than [`MIN_CROP_AREA_FRACTION`] of `detected` (the detector's
/// box for the same person). At the top of the backswing the arms and club
/// can hide the body so that only the head scores well; a crop around the
/// head alone would lose the rest of the golfer.
pub(crate) fn next_crop(
    pose: &Pose,
    detected: Option<&BBox>,
    keypoint_threshold: f32,
) -> Option<BBox> {
    let confident = pose
        .keypoints
        .iter()
        .filter(|k| k.score >= keypoint_threshold)
        .count();
    if confident * 2 < pose.keypoints.len() {
        return None;
    }
    let crop = pose.keypoint_box(keypoint_threshold, KEYPOINT_BOX_EXPANSION)?;
    match detected {
        Some(d) if crop.area() < MIN_CROP_AREA_FRACTION * d.area() => None,
        _ => Some(crop),
    }
}

/// A top-down pose model: keypoints of the person in a box.
///
/// [`RtmPose`] is the built-in engine; [`PoseModels`], [`PoseTracker`] and
/// [`analyze_frames`] work with any implementation.
pub trait PoseEngine: Send {
    /// The keypoint layout it produces.
    fn skeleton(&self) -> Skeleton;
    /// The model's name.
    fn model_name(&self) -> &str;
    /// Estimates the keypoints of the person in `bbox` (image pixels; the
    /// model pads it).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for an
    /// empty or non-finite box, [`Error::Model`](crate::Error::Model) if
    /// inference fails.
    fn estimate(&mut self, image: &Image, bbox: &BBox) -> Result<Pose>;
    /// Another instance sharing this one's weights, with its own run state
    /// on the calling thread, for another thread.
    ///
    /// # Errors
    ///
    /// [`Error::Model`](crate::Error::Model) if the run state can't be
    /// created.
    fn fork(&self) -> Result<Self>
    where
        Self: Sized;
    /// Runs each inference on `pool`, which lowers the latency of one frame
    /// (for a live view). Engines that can't use it ignore it, the default.
    ///
    /// # Errors
    ///
    /// [`Error::Model`](crate::Error::Model) if the engine can't switch.
    fn set_inference_pool(&mut self, pool: &InferencePool) -> Result<()> {
        let _ = pool;
        Ok(())
    }
}

/// A person detector.
///
/// [`Yolox`] is the built-in detector; [`PoseModels`], [`PoseTracker`] and
/// [`analyze_frames`] work with any implementation.
pub trait PersonDetector: Send {
    /// The model's name.
    fn model_name(&self) -> &str;
    /// The people in the image, best first.
    ///
    /// # Errors
    ///
    /// [`Error::Model`](crate::Error::Model) if inference fails.
    fn detect(&mut self, image: &Image) -> Result<Vec<Detection>>;
    /// Another instance sharing this one's weights and settings, with its
    /// own run state on the calling thread, for another thread.
    ///
    /// # Errors
    ///
    /// [`Error::Model`](crate::Error::Model) if the run state can't be
    /// created.
    fn fork(&self) -> Result<Self>
    where
        Self: Sized;
    /// Runs each inference on `pool`, as [`PoseEngine::set_inference_pool`].
    ///
    /// # Errors
    ///
    /// [`Error::Model`](crate::Error::Model) if the detector can't switch.
    fn set_inference_pool(&mut self, pool: &InferencePool) -> Result<()> {
        let _ = pool;
        Ok(())
    }
}

/// Pose settings shared by the live tracker and the clip pass.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseConfig {
    /// The pose model.
    pub pose_model: ModelSpec,
    /// The person detector.
    pub detector: ModelSpec,
    /// Minimum detector score for a person.
    pub detection_threshold: f32,
    /// Run the detector at least every this many frames (live: also when
    /// tracking is lost; clip pass: the key-frame interval). At least 1.
    pub detect_every: u32,
    /// Keypoints scoring at least this outline the person for the next
    /// frame's crop.
    pub keypoint_threshold: f32,
    /// A pose whose mean score is below this is absent: the live tracker
    /// returns `None` (after one retry) and the clip pass redoes it from a
    /// neighboring frame's keypoints, then leaves a gap in the
    /// [`PoseTrack`]. Low-score poses are keypoints scattered over the
    /// background, e.g. after the golfer walks out of frame.
    pub min_pose_score: f32,
    /// Minimum IoU for a new detection to be the same person.
    pub min_iou: f32,
    /// One Euro smoothing for the live tracker's output, `None` for raw
    /// keypoints. The clip pass always stores raw keypoints.
    pub smoothing: Option<OneEuroParams>,
    /// Worker threads for the clip pass, each running its own model
    /// instances; 0 uses the available parallelism (at most 8).
    pub threads: usize,
    /// Cores one inference of the live tracker may use, which lowers its
    /// latency; 0 picks half the available parallelism, at most 4. The clip
    /// pass runs single-threaded inferences on [`PoseConfig::threads`]
    /// workers instead.
    pub inference_threads: usize,
    /// Runs at the start of every worker thread of the clip pass, on that
    /// thread, before it takes any frame; `None` runs nothing. Use it to
    /// set up the workers, e.g. lower their priority so the pass yields
    /// the CPU to other programs. The live tracker doesn't use it.
    pub on_worker_start: Option<WorkerHook>,
    /// Quarter turns applied to every frame before detection and pose, for
    /// a camera mounted on its side: by the clip pass on every frame it
    /// decodes, and by the live tracker on every image it is given. The
    /// track's keypoints, [`PoseTrack::width`] and [`PoseTrack::height`]
    /// are in the turned frame, and [`PoseTrack::rotation`] records it; the
    /// live tracker's poses are in the turned frame too.
    pub rotation: Rotation,
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            pose_model: RTMPOSE_M_HALPE26,
            detector: YOLOX_TINY,
            detection_threshold: Yolox::DEFAULT_SCORE_THRESHOLD,
            detect_every: 10,
            keypoint_threshold: 0.3,
            min_pose_score: 0.3,
            min_iou: 0.3,
            smoothing: None,
            threads: 0,
            inference_threads: 0,
            on_worker_start: None,
            rotation: Rotation::None,
        }
    }
}

impl PoseConfig {
    /// The number of clip-pass workers this config asks for.
    pub(crate) fn worker_count(&self) -> usize {
        if self.threads > 0 {
            self.threads
        } else {
            std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
        }
    }

    /// The number of cores one live inference uses.
    pub(crate) fn inference_thread_count(&self) -> usize {
        if self.inference_threads > 0 {
            self.inference_threads
        } else {
            std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).clamp(1, 4))
        }
    }
}

/// A pose model and a person detector, by default [`RtmPose`] and
/// [`Yolox`].
///
/// Load once; [`PoseModels::fork`] and [`PoseModels::tracker`] make
/// instances sharing the weights, for other threads. Other engines plug in
/// through [`PoseModels::new`].
#[derive(Debug)]
pub struct PoseModels<P = RtmPose, D = Yolox> {
    /// The pose model.
    pub pose: P,
    /// The person detector.
    pub detector: D,
}

impl PoseModels {
    /// Loads the config's two models from `<dir>/<name>.onnx`, verifying
    /// both, and applies the config's detection threshold.
    ///
    /// # Errors
    ///
    /// As [`RtmPose::load`] and [`Yolox::load`].
    pub fn load(dir: &Path, config: &PoseConfig) -> Result<Self> {
        let pose = RtmPose::load(&config.pose_model, dir)?;
        let mut detector = Yolox::load(&config.detector, dir)?;
        detector.set_thresholds(config.detection_threshold, Yolox::DEFAULT_NMS_THRESHOLD);
        Ok(Self { pose, detector })
    }

    /// Downloads the config's two models into `cache_dir` if they aren't
    /// there yet (see [`fetch_model`]), then loads them.
    ///
    /// # Errors
    ///
    /// As [`fetch_model`] and [`PoseModels::load`].
    #[cfg(feature = "pose-download")]
    pub fn download(cache_dir: &Path, config: &PoseConfig) -> Result<Self> {
        fetch_model(&config.pose_model, cache_dir)?;
        fetch_model(&config.detector, cache_dir)?;
        Self::load(cache_dir, config)
    }
}

impl<P: PoseEngine, D: PersonDetector> PoseModels<P, D> {
    /// Pairs a pose model with a person detector.
    #[must_use]
    pub fn new(pose: P, detector: D) -> Self {
        Self { pose, detector }
    }

    /// Instances sharing these weights, each running on the calling thread.
    ///
    /// # Errors
    ///
    /// As [`PoseEngine::fork`] and [`PersonDetector::fork`].
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            pose: self.pose.fork()?,
            detector: self.detector.fork()?,
        })
    }

    /// A live tracker on forks of these models. Both share one
    /// [`InferencePool`] of [`PoseConfig::inference_threads`] cores (they
    /// never run at the same time).
    ///
    /// # Errors
    ///
    /// As [`PoseModels::fork`], [`InferencePool::new`] and
    /// [`PoseEngine::set_inference_pool`].
    pub fn tracker(&self, config: PoseConfig) -> Result<PoseTracker<P, D>> {
        let mut m = self.fork()?;
        let pool = InferencePool::new(config.inference_thread_count())?;
        m.pose.set_inference_pool(&pool)?;
        m.detector.set_inference_pool(&pool)?;
        Ok(PoseTracker::new(m.pose, m.detector, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypoints_serialize_as_triples() {
        let k = Keypoint {
            x: 1.5,
            y: -2.0,
            score: 0.25,
        };
        assert_eq!(serde_json::to_string(&k).unwrap(), "[1.5,-2.0,0.25]");
        assert_eq!(
            serde_json::from_str::<Keypoint>("[1.5,-2.0,0.25]").unwrap(),
            k
        );
    }

    #[test]
    fn pose_score_and_keypoint_box() {
        let mut kps = vec![
            Keypoint {
                x: 10.0,
                y: 10.0,
                score: 0.9,
            };
            26
        ];
        kps[1] = Keypoint {
            x: 30.0,
            y: 50.0,
            score: 0.9,
        };
        kps[2] = Keypoint {
            x: 20.0,
            y: 30.0,
            score: 0.9,
        };
        // A far-off low-score point is ignored.
        kps[3] = Keypoint {
            x: 500.0,
            y: 500.0,
            score: 0.05,
        };
        let pose = Pose::new(Skeleton::Halpe26, kps, BBox::default());
        assert!((pose.score - (25.0 * 0.9 + 0.05) / 26.0).abs() < 1e-6);
        assert_eq!(
            pose.keypoint_box(0.3, 1.0),
            Some(BBox::new(10.0, 10.0, 30.0, 50.0))
        );
        assert_eq!(
            pose.keypoint_box(0.3, 1.5),
            Some(BBox::new(5.0, 0.0, 35.0, 60.0))
        );
        assert_eq!(pose.keypoint("left_eye").map(|k| k.x), Some(30.0));
        assert!(pose.keypoint("tail").is_none());
        // Everything below the threshold: all points count.
        assert_eq!(
            pose.keypoint_box(0.95, 1.0),
            Some(BBox::new(10.0, 10.0, 500.0, 500.0))
        );
    }

    #[test]
    fn worker_count_is_at_least_one() {
        assert!(PoseConfig::default().worker_count() >= 1);
        let c = PoseConfig {
            threads: 3,
            ..PoseConfig::default()
        };
        assert_eq!(c.worker_count(), 3);
        assert!((1..=4).contains(&PoseConfig::default().inference_thread_count()));
    }
}
