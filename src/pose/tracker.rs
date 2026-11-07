//! Following one golfer from frame to frame.

use super::filter::OneEuro;
use super::geometry::{BBox, Detection};
use super::image::Image;
use super::{PersonDetector, Pose, PoseConfig, PoseEngine, next_crop};
use crate::Result;

/// People smaller than this fraction of the image are ignored when picking
/// the golfer.
const MIN_AREA_FRACTION: f32 = 0.005;

/// Tracks a single golfer for a live view.
///
/// Frames are turned by [`PoseConfig::rotation`] first, so a camera on its
/// side gives an upright golfer, as in the clip pass.
///
/// The detector runs on the first frame, every
/// [`PoseConfig::detect_every`] frames and whenever tracking is lost; in
/// between, the crop comes from the previous frame's keypoints (rtmlib's
/// `PoseTracker` scheme). When that keypoint box collapses onto part of the
/// body (fewer than half the keypoints score, or it is much smaller than the
/// detector's box, as when the arms hide the body at the top of the
/// backswing), the next frame uses the detector's box and detects again.
///
/// The golfer is the most confident large person near the frame centre;
/// periodic detections keep the same person by IoU with the tracked box. A
/// pose whose mean score falls below [`PoseConfig::min_pose_score`] gets
/// one retry (a tracked crop with a fresh detection, a fresh detection with
/// the tracked crop); if that fails too, tracking is lost and
/// [`PoseTracker::track`] returns `None`.
///
/// Once lost, the next detection picks the golfer afresh: if the golfer
/// has left the frame and someone else is in view, the tracker follows that
/// person. A `Some` after a `None` is therefore a re-acquisition, possibly
/// of a different person, and a UI should present it as one.
#[derive(Debug)]
pub struct PoseTracker<P, D> {
    pose: P,
    detector: D,
    config: PoseConfig,
    /// Where to look in the next frame.
    target: Option<BBox>,
    /// The detector's latest box for the tracked person.
    detected: Option<BBox>,
    /// Detect on the next frame regardless of the interval.
    redetect: bool,
    frames_since_detect: u32,
    smoothers: Vec<[OneEuro; 2]>,
}

impl<P: PoseEngine, D: PersonDetector> PoseTracker<P, D> {
    /// A tracker with no person yet. [`PoseModels::tracker`] makes one from
    /// loaded models.
    ///
    /// [`PoseModels::tracker`]: super::PoseModels::tracker
    #[must_use]
    pub fn new(pose: P, detector: D, config: PoseConfig) -> Self {
        Self {
            pose,
            detector,
            config,
            target: None,
            detected: None,
            redetect: false,
            frames_since_detect: 0,
            smoothers: Vec::new(),
        }
    }

    /// The settings.
    #[must_use]
    pub fn config(&self) -> &PoseConfig {
        &self.config
    }

    /// Forgets the tracked person and the smoothing history.
    pub fn reset(&mut self) {
        self.lose();
        self.frames_since_detect = 0;
    }

    /// The golfer's pose in this frame, `None` when nobody is found or the
    /// pose scores below [`PoseConfig::min_pose_score`].
    ///
    /// `image` is the frame as the camera sent it: the tracker first turns
    /// it by [`PoseConfig::rotation`], as the clip pass does, so the pose's
    /// keypoints are in the upright frame.
    ///
    /// `t_ns` is the frame's time (any monotonic nanosecond clock); it only
    /// drives the optional smoothing.
    ///
    /// # Errors
    ///
    /// [`Error::Model`](crate::Error::Model) if inference fails.
    pub fn track(&mut self, image: &Image, t_ns: u64) -> Result<Option<Pose>> {
        if self.config.rotation.is_none() {
            self.track_upright(image, t_ns)
        } else {
            let upright = image.rotated(self.config.rotation);
            self.track_upright(&upright, t_ns)
        }
    }

    /// [`PoseTracker::track`] on a frame that is already upright.
    fn track_upright(&mut self, image: &Image, t_ns: u64) -> Result<Option<Pose>> {
        let every = self.config.detect_every.max(1);
        let detect_due =
            self.target.is_none() || self.redetect || self.frames_since_detect >= every;
        let mut crop = self.target;
        let mut fresh = false;
        if detect_due {
            // Periodic detections only refine the box of the person being
            // tracked; a stranger never takes over.
            let strict = self.target.is_some();
            if let Some(b) = self.detect(image, strict)? {
                crop = Some(b);
                fresh = true;
                self.detected = Some(b);
            }
            self.frames_since_detect = 0;
            self.redetect = false;
        }
        let Some(b) = crop else {
            self.lose();
            return Ok(None);
        };
        let mut pose = self.pose.estimate(image, &b)?;
        if pose.score < self.config.min_pose_score {
            // One retry: a tracked crop with a fresh detection, a fresh
            // detection with the tracked crop.
            let (retry_box, from_detector) = if fresh {
                (self.target.filter(|t| *t != b), false)
            } else {
                self.frames_since_detect = 0;
                (self.detect(image, false)?, true)
            };
            if let Some(r) = retry_box {
                let retry = self.pose.estimate(image, &r)?;
                if retry.score > pose.score {
                    pose = retry;
                    if from_detector {
                        self.detected = Some(r);
                    }
                }
            }
        }
        if pose.score < self.config.min_pose_score {
            self.lose();
            return Ok(None);
        }
        let crop = next_crop(
            &pose,
            self.detected.as_ref(),
            self.config.keypoint_threshold,
        );
        // A collapsed crop: pose the detector's box and detect again.
        self.redetect = crop.is_none();
        self.target = crop.or(self.detected).or(Some(b));
        self.frames_since_detect = self.frames_since_detect.saturating_add(1);
        if let Some(params) = self.config.smoothing {
            if self.smoothers.len() != pose.keypoints.len() {
                self.smoothers =
                    vec![[OneEuro::new(params), OneEuro::new(params)]; pose.keypoints.len()];
            }
            // Nanosecond clocks stay exact in f64 for months of uptime.
            #[allow(clippy::cast_precision_loss)]
            let t = t_ns as f64 / 1e9;
            for (k, [fx, fy]) in pose.keypoints.iter_mut().zip(&mut self.smoothers) {
                // Pixel coordinates round-trip through f64 exactly.
                #[allow(clippy::cast_possible_truncation)]
                {
                    k.x = fx.filter(t, f64::from(k.x)) as f32;
                    k.y = fy.filter(t, f64::from(k.y)) as f32;
                }
            }
        }
        Ok(Some(pose))
    }

    fn lose(&mut self) {
        self.target = None;
        self.detected = None;
        self.redetect = false;
        self.smoothers.clear();
    }

    /// Runs the detector and picks the box to pose: with `strict`, only a
    /// box matching the tracked one by IoU; otherwise the IoU match if there
    /// is one, else the best golfer candidate.
    fn detect(&mut self, image: &Image, strict: bool) -> Result<Option<BBox>> {
        let dets = self.detector.detect(image)?;
        let matched = self
            .target
            .and_then(|t| match_by_iou(&dets, &t, self.config.min_iou));
        if matched.is_some() || strict {
            return Ok(matched);
        }
        Ok(pick_golfer(&dets, image.width(), image.height()))
    }
}

/// The detection overlapping `target` most, if by at least `min_iou`.
pub(crate) fn match_by_iou(dets: &[Detection], target: &BBox, min_iou: f32) -> Option<BBox> {
    dets.iter()
        .map(|d| (d.bbox.iou(target), d.bbox))
        .filter(|(iou, _)| *iou >= min_iou)
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, b)| b)
}

/// The most likely golfer: confident, large and near the frame centre.
pub(crate) fn pick_golfer(dets: &[Detection], width: u32, height: u32) -> Option<BBox> {
    // Frame sizes are far below f32's exact-integer range.
    #[allow(clippy::cast_precision_loss)]
    let (w, h) = (width as f32, height as f32);
    let image_area = (w * h).max(1.0);
    let half_diag = (w * w + h * h).sqrt() * 0.5;
    dets.iter()
        .filter(|d| d.bbox.is_valid() && d.bbox.area() / image_area >= MIN_AREA_FRACTION)
        .map(|d| {
            let (cx, cy) = d.bbox.center();
            let off = ((cx - w * 0.5).powi(2) + (cy - h * 0.5).powi(2)).sqrt() / half_diag.max(1.0);
            let rank = d.score * (d.bbox.area() / image_area).sqrt() / (1.0 + 2.0 * off);
            (rank, d.bbox)
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, b)| b)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::pose::{Keypoint, Rotation, Skeleton};

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Detection {
        Detection {
            bbox: BBox::new(x1, y1, x2, y2),
            score,
        }
    }

    #[test]
    fn picks_the_central_large_confident_person() {
        let golfer = det(270.0, 100.0, 370.0, 400.0, 0.8);
        let bystander = det(0.0, 150.0, 60.0, 330.0, 0.95);
        let speck = det(320.0, 240.0, 324.0, 248.0, 0.99);
        assert_eq!(
            pick_golfer(&[bystander, speck, golfer], 640, 480),
            Some(golfer.bbox)
        );
        assert_eq!(pick_golfer(&[speck], 640, 480), None);
        assert_eq!(pick_golfer(&[], 640, 480), None);
    }

    #[test]
    fn iou_matching_stays_on_the_same_person() {
        let a = det(100.0, 100.0, 200.0, 400.0, 0.5);
        let b = det(400.0, 100.0, 500.0, 400.0, 0.9);
        let tracked = BBox::new(110.0, 95.0, 210.0, 405.0);
        assert_eq!(match_by_iou(&[b, a], &tracked, 0.3), Some(a.bbox));
        assert_eq!(match_by_iou(&[b], &tracked, 0.3), None);
    }

    /// A pose engine that puts a 26-point pose inside whatever box it is
    /// given, with a scripted score per call. A `head_only` call scores the
    /// first four keypoints high and the rest low.
    struct FakePose {
        scores: VecDeque<f32>,
        head_only: VecDeque<bool>,
        boxes: Vec<BBox>,
    }

    impl FakePose {
        fn new(scores: impl IntoIterator<Item = f32>) -> Self {
            Self {
                scores: scores.into_iter().collect(),
                head_only: VecDeque::new(),
                boxes: Vec::new(),
            }
        }
    }

    impl PoseEngine for FakePose {
        fn skeleton(&self) -> Skeleton {
            Skeleton::Halpe26
        }
        fn model_name(&self) -> &'static str {
            "fake-pose"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self::new(self.scores.clone()))
        }
        fn estimate(&mut self, _image: &Image, bbox: &BBox) -> Result<Pose> {
            self.boxes.push(*bbox);
            let score = self.scores.pop_front().unwrap_or(0.9);
            let head_only = self.head_only.pop_front().unwrap_or(false);
            let (cx, cy) = bbox.center();
            let kps = (0..26u8)
                .map(|i| Keypoint {
                    x: cx + (f32::from(i % 5) - 2.0) * bbox.width() / 6.0,
                    y: cy + (f32::from(i / 5) - 2.5) * bbox.height() / 7.0,
                    score: match (head_only, i < 4) {
                        (false, _) => score,
                        (true, true) => 0.95,
                        (true, false) => 0.25,
                    },
                })
                .collect();
            Ok(Pose::new(Skeleton::Halpe26, kps, *bbox))
        }
    }

    struct FakeDetector {
        calls: usize,
        dets: Vec<Detection>,
    }

    impl PersonDetector for FakeDetector {
        fn model_name(&self) -> &'static str {
            "fake-detector"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self {
                calls: 0,
                dets: self.dets.clone(),
            })
        }
        fn detect(&mut self, _image: &Image) -> Result<Vec<Detection>> {
            self.calls += 1;
            Ok(self.dets.clone())
        }
    }

    fn image() -> Image {
        Image::new(640, 480, vec![0; 640 * 480 * 3]).unwrap()
    }

    #[test]
    fn detects_every_n_frames_and_tracks_in_between() {
        let golfer = det(270.0, 100.0, 370.0, 400.0, 0.8);
        let mut t = PoseTracker::new(
            FakePose::new([]),
            FakeDetector {
                calls: 0,
                dets: vec![golfer],
            },
            PoseConfig {
                detect_every: 4,
                ..PoseConfig::default()
            },
        );
        let img = image();
        for i in 0..9u64 {
            assert!(t.track(&img, i * 8_333_333).unwrap().is_some());
        }
        // Frames 0, 4 and 8 detect.
        assert_eq!(t.detector.calls, 3);
        assert_eq!(t.pose.boxes[0], golfer.bbox);
        assert_eq!(t.pose.boxes[4], golfer.bbox);
        // Frame 1 crops around frame 0's keypoints instead.
        assert_ne!(t.pose.boxes[1], golfer.bbox);
    }

    #[test]
    fn low_score_redetects_then_loses() {
        let golfer = det(270.0, 100.0, 370.0, 400.0, 0.8);
        let mut t = PoseTracker::new(
            // Frame 0 fine; frame 1 tracked crop bad, retry good; frame 2
            // tracked bad and retry bad: lost.
            FakePose::new([0.9, 0.1, 0.8, 0.1, 0.1]),
            FakeDetector {
                calls: 0,
                dets: vec![golfer],
            },
            PoseConfig::default(),
        );
        let img = image();
        assert!(t.track(&img, 0).unwrap().is_some());
        let p = t.track(&img, 1).unwrap().unwrap();
        assert!((p.score - 0.8).abs() < 1e-6);
        assert_eq!(t.detector.calls, 2);
        assert!(t.track(&img, 2).unwrap().is_none());
        assert_eq!(t.detector.calls, 3);
        // Lost: the next frame detects again.
        assert!(t.track(&img, 3).unwrap().is_some());
        assert_eq!(t.detector.calls, 4);
    }

    #[test]
    fn weak_pose_on_a_fresh_detection_retries_the_tracked_crop() {
        let golfer = det(270.0, 100.0, 370.0, 400.0, 0.8);
        let mut t = PoseTracker::new(
            // Frame 2's periodic detection poses badly; the tracked crop
            // from frame 1 does better.
            FakePose::new([0.9, 0.9, 0.1, 0.9]),
            FakeDetector {
                calls: 0,
                dets: vec![golfer],
            },
            PoseConfig {
                detect_every: 2,
                ..PoseConfig::default()
            },
        );
        let img = image();
        for i in 0..3 {
            assert!(t.track(&img, i).unwrap().is_some(), "frame {i}");
        }
        assert_eq!(t.detector.calls, 2);
        assert_eq!(t.pose.boxes[2], golfer.bbox);
        // The retry posed the crop around frame 1's keypoints, which sits
        // inside frame 1's own crop.
        let (retry, before) = (t.pose.boxes[3], t.pose.boxes[1]);
        assert_ne!(retry, golfer.bbox);
        assert!(retry.area() < before.area());
        assert!(retry.iou(&before) > 0.5);
    }

    #[test]
    fn a_collapsed_crop_falls_back_to_the_detector() {
        let golfer = det(270.0, 100.0, 370.0, 400.0, 0.8);
        let mut pose = FakePose::new([]);
        // Frame 1: only the head scores (the body hidden at the top).
        pose.head_only = VecDeque::from([false, true]);
        let mut t = PoseTracker::new(
            pose,
            FakeDetector {
                calls: 0,
                dets: vec![golfer],
            },
            PoseConfig::default(),
        );
        let img = image();
        assert!(t.track(&img, 0).unwrap().is_some());
        let p = t
            .track(&img, 1)
            .unwrap()
            .expect("still above the threshold");
        assert!(p.score >= PoseConfig::default().min_pose_score);
        assert_eq!(t.detector.calls, 1);
        // Frame 2 detects again and poses the detector's box, not a crop
        // around the head.
        assert!(t.track(&img, 2).unwrap().is_some());
        assert_eq!(t.detector.calls, 2);
        assert_eq!(t.pose.boxes[2], golfer.bbox);
    }

    #[test]
    fn nobody_in_view_is_none() {
        let mut t = PoseTracker::new(
            FakePose::new([]),
            FakeDetector {
                calls: 0,
                dets: Vec::new(),
            },
            PoseConfig::default(),
        );
        assert!(t.track(&image(), 0).unwrap().is_none());
        assert!(t.pose.boxes.is_empty());
    }

    /// A detector and pose engine that find the one bright pixel block and
    /// record the size of every image they see.
    #[derive(Default)]
    struct Blob {
        sizes: Vec<(u32, u32)>,
    }

    impl Blob {
        fn centre(&mut self, image: &Image) -> (f32, f32) {
            self.sizes.push((image.width(), image.height()));
            let (mut sx, mut sy, mut n) = (0.0f32, 0.0f32, 0.0f32);
            for y in 0..image.height() {
                for x in 0..image.width() {
                    if image.pixel(x, y).unwrap()[0] > 128 {
                        // Test images are small.
                        #[allow(clippy::cast_precision_loss)]
                        {
                            sx += x as f32;
                            sy += y as f32;
                        }
                        n += 1.0;
                    }
                }
            }
            (sx / n, sy / n)
        }
    }

    impl PoseEngine for Blob {
        fn skeleton(&self) -> Skeleton {
            Skeleton::Halpe26
        }
        fn model_name(&self) -> &'static str {
            "blob"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self::default())
        }
        fn estimate(&mut self, image: &Image, bbox: &BBox) -> Result<Pose> {
            let (x, y) = self.centre(image);
            let kps = (0..26u8)
                .map(|i| Keypoint {
                    x: x + f32::from(i % 3),
                    y: y + f32::from(i / 3),
                    score: 0.9,
                })
                .collect();
            Ok(Pose::new(Skeleton::Halpe26, kps, *bbox))
        }
    }

    impl PersonDetector for Blob {
        fn model_name(&self) -> &'static str {
            "blob"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self::default())
        }
        fn detect(&mut self, image: &Image) -> Result<Vec<Detection>> {
            let (x, y) = self.centre(image);
            Ok(vec![det(x - 30.0, y - 30.0, x + 30.0, y + 30.0, 0.9)])
        }
    }

    #[test]
    fn the_tracker_turns_frames_by_the_config_rotation() {
        // An upright 160 × 120 scene with a bright block at (100..110, 20..30).
        let mut data = vec![0u8; 160 * 120 * 3];
        for y in 20..30 {
            for x in 100..110 {
                let p = (y * 160 + x) * 3;
                data[p..p + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        let scene = Image::new(160, 120, data).unwrap();
        let mut upright = PoseTracker::new(Blob::default(), Blob::default(), PoseConfig::default());
        let want = upright.track(&scene, 0).unwrap().unwrap();
        assert!((want.keypoints[0].x - 104.5).abs() < 1e-3);
        for (rotation, mounted) in [
            (Rotation::Cw90, Rotation::Ccw90),
            (Rotation::Half, Rotation::Half),
            (Rotation::Ccw90, Rotation::Cw90),
        ] {
            // A camera mounted turned by `mounted` sends the scene turned
            // that way; the config's `rotation` turns it back.
            let sent = scene.rotated(mounted);
            let mut t = PoseTracker::new(
                Blob::default(),
                Blob::default(),
                PoseConfig {
                    rotation,
                    ..PoseConfig::default()
                },
            );
            for i in 0..3 {
                let pose = t.track(&sent, i).unwrap().unwrap();
                for (k, w) in pose.keypoints.iter().zip(&want.keypoints) {
                    assert!(
                        (k.x - w.x).abs() < 1e-3 && (k.y - w.y).abs() < 1e-3,
                        "{rotation:?}: {k:?} vs {w:?}"
                    );
                }
            }
            assert!(t.detector.sizes.iter().all(|&s| s == (160, 120)));
            assert!(t.pose.sizes.iter().all(|&s| s == (160, 120)));
            // Without the rotation the models see the sideways frame.
            let mut raw = PoseTracker::new(Blob::default(), Blob::default(), PoseConfig::default());
            let pose = raw.track(&sent, 0).unwrap().unwrap();
            assert_eq!(raw.pose.sizes[0], mounted.dimensions(160, 120));
            if mounted != Rotation::Half {
                assert!((pose.keypoints[0].x - want.keypoints[0].x).abs() > 1.0);
            }
        }
    }

    #[test]
    fn smoothing_damps_jitter() {
        let golfer = det(270.0, 100.0, 370.0, 400.0, 0.8);
        let mut t = PoseTracker::new(
            FakePose::new([]),
            FakeDetector {
                calls: 0,
                dets: vec![golfer],
            },
            PoseConfig {
                detect_every: 1,
                smoothing: Some(crate::pose::OneEuroParams::default()),
                ..PoseConfig::default()
            },
        );
        // The detector alternates the box by ±6 px; the smoothed nose moves
        // far less.
        let img = image();
        let mut xs = Vec::new();
        for i in 0..60u64 {
            #[allow(clippy::cast_precision_loss)]
            let dx = if i % 2 == 0 { 6.0 } else { -6.0 };
            t.detector.dets = vec![det(270.0 + dx, 100.0, 370.0 + dx, 400.0, 0.8)];
            let p = t.track(&img, i * 8_333_333).unwrap().unwrap();
            xs.push(p.keypoints[0].x);
        }
        let tail = &xs[30..];
        let spread = tail.iter().copied().fold(f32::MIN, f32::max)
            - tail.iter().copied().fold(f32::MAX, f32::min);
        assert!(spread < 3.0, "spread {spread}");
    }
}
