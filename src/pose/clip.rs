//! The offline pass over a recorded clip.

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::geometry::{BBox, Detection};
use super::image::DecodeBudget;
use super::track::{POSE_TRACK_VERSION, PoseFrame, PoseTrack};
use super::tracker::{match_by_iou, pick_golfer};
use super::{PersonDetector, Pose, PoseConfig, PoseEngine, PoseModels, next_crop};
use crate::{Error, Result};

/// The most decoded RGB frame bytes the clip pass holds at once, across all
/// its workers.
pub(crate) const DECODE_BUDGET: usize = 256 * 1024 * 1024;

/// A function each worker thread of the clip pass runs once when it starts
/// ([`PoseConfig::on_worker_start`]), on that thread, before it takes any
/// frame.
///
/// The pass starts its workers afresh for each parallel stage (detection on
/// the key frames, then pose on every frame), so one clip runs the hook up
/// to twice [`PoseConfig::threads`] times. The sequential fix-up runs on the
/// calling thread, which the hook never sees.
///
/// Clones share the function; two hooks are equal when they share it.
///
/// ```
/// use stance::pose::{PoseConfig, WorkerHook};
///
/// let config = PoseConfig {
///     on_worker_start: Some(WorkerHook::new(|| {
///         // e.g. lower the calling thread's priority
///     })),
///     ..PoseConfig::default()
/// };
/// # let _ = config;
/// ```
#[derive(Clone)]
pub struct WorkerHook(Arc<dyn Fn() + Send + Sync>);

impl WorkerHook {
    /// A hook running `f`.
    #[must_use]
    pub fn new(f: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// Runs the hook on the calling thread.
    pub fn run(&self) {
        (self.0)();
    }
}

impl fmt::Debug for WorkerHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WorkerHook(..)")
    }
}

impl PartialEq for WorkerHook {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Runs pose estimation over the clip `<dir>/<name>.avi` (frame times from
/// its sidecar `<dir>/<name>.json`). See [`analyze_frames`].
///
/// # Errors
///
/// As [`open_clip`](crate::open_clip) and [`analyze_frames`].
pub fn analyze_clip<P: PoseEngine, D: PersonDetector>(
    models: &PoseModels<P, D>,
    clip_avi: &Path,
    config: &PoseConfig,
) -> Result<PoseTrack> {
    let dir = clip_avi.parent().unwrap_or_else(|| Path::new("."));
    let name = clip_avi
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::InvalidArgument(format!("clip path {}", clip_avi.display())))?;
    let (mut avi, sidecar) = crate::open_clip(dir, name)?;
    let frames = sidecar
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| Ok((f.t_ns, avi.read_frame(i)?)))
        .collect::<Result<Vec<_>>>()?;
    analyze_frames(models, &frames, config)
}

/// Runs pose estimation over a clip's frames: `(t_ns, jpeg)` in clip order.
///
/// Every decoded frame is first turned by [`PoseConfig::rotation`]; the
/// track is in the turned frame.
///
/// 1. The detector runs on every [`PoseConfig::detect_every`]-th frame and
///    the last, in parallel. The golfer is picked on the first of these
///    frames with a person and followed through the others by IoU.
/// 2. Boxes for the frames in between are interpolated linearly.
/// 3. The pose model runs on every frame in parallel, each worker with its
///    own model instances ([`PoseConfig::threads`], made with
///    [`PoseEngine::fork`] and [`PersonDetector::fork`]). Every worker
///    thread of steps 1 and 3 first runs [`PoseConfig::on_worker_start`].
/// 4. A sequential fix-up pass, forwards then backwards, redoes every frame
///    whose pose scores below [`PoseConfig::min_pose_score`] (or has none)
///    with a box around the neighbouring frame's keypoints, keeping the
///    better result. A neighbour whose keypoint box has collapsed onto part
///    of the body is not used.
///
/// Frames that fail to decode, frames where nobody is found and frames
/// whose pose still scores below [`PoseConfig::min_pose_score`] are empty
/// in the track ([`PoseFrame::empty`]): a low-score pose is keypoints
/// scattered over the background, never stored. Keypoints are stored raw.
///
/// Decoded frames held at once, across all workers, stay under 256 MiB.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for no frames or no decodable frame;
/// [`Error::Model`] if inference fails.
pub fn analyze_frames<P, D, J>(
    models: &PoseModels<P, D>,
    frames: &[(u64, J)],
    config: &PoseConfig,
) -> Result<PoseTrack>
where
    P: PoseEngine,
    D: PersonDetector,
    J: AsRef<[u8]> + Sync,
{
    let n = frames.len();
    if n == 0 {
        return Err(Error::InvalidArgument("the clip has no frames".into()));
    }
    let every = config.detect_every.max(1) as usize;
    let mut keyframes: Vec<usize> = (0..n).step_by(every).collect();
    if keyframes.last() != Some(&(n - 1)) {
        keyframes.push(n - 1);
    }
    let budget = DecodeBudget::new(DECODE_BUDGET);
    let decode = |i: usize| budget.decode(frames[i].1.as_ref(), config.rotation).ok();
    let workers = config.worker_count();
    let hook = config.on_worker_start.as_ref();

    // 1. Detection on the key frames.
    let detections: Vec<Option<(u32, u32, Vec<Detection>)>> =
        parallel_map(&keyframes, workers, hook, models, |m, &i| {
            let Some(image) = decode(i) else {
                return Ok(None);
            };
            let dets = m.detector.detect(&image)?;
            Ok(Some((image.width(), image.height(), dets)))
        })?;
    let Some((width, height)) = detections.iter().flatten().map(|d| (d.0, d.1)).next() else {
        return Err(Error::InvalidArgument(
            "no frame of the clip decodes".into(),
        ));
    };
    let chosen = follow_golfer(&detections, width, height, config.min_iou);

    // 2. A box for every frame.
    let anchors: Vec<(usize, BBox)> = keyframes
        .iter()
        .zip(&chosen)
        .filter_map(|(&i, b)| b.map(|b| (i, b)))
        .collect();
    let boxes: Vec<Option<BBox>> = (0..n).map(|i| interpolate(&anchors, i)).collect();

    // 3. Pose on every frame.
    let indices: Vec<usize> = (0..n).collect();
    let mut poses: Vec<Option<Pose>> = parallel_map(&indices, workers, hook, models, |m, &i| {
        let Some(bbox) = boxes[i] else {
            return Ok(None);
        };
        let Some(image) = decode(i) else {
            return Ok(None);
        };
        m.pose.estimate(&image, &bbox).map(Some)
    })?;

    // 4. Sequential fix-up from neighbouring keypoints.
    let mut m = models.fork()?;
    let threshold = config.min_pose_score;
    let weak = |p: &Option<Pose>| p.as_ref().is_none_or(|p| p.score < threshold);
    let order: Vec<(usize, Option<usize>)> = (0..n)
        .map(|i| (i, i.checked_sub(1)))
        .chain((0..n).rev().map(|i| (i, (i + 1 < n).then_some(i + 1))))
        .collect();
    for (i, neighbour) in order {
        let Some(j) = neighbour else { continue };
        if !weak(&poses[i]) || weak(&poses[j]) {
            continue;
        }
        let Some(bbox) = poses[j]
            .as_ref()
            .and_then(|p| next_crop(p, boxes[i].as_ref(), config.keypoint_threshold))
        else {
            continue;
        };
        let Some(image) = decode(i) else { continue };
        let retry = m.pose.estimate(&image, &bbox)?;
        if poses[i].as_ref().is_none_or(|p| retry.score > p.score) {
            poses[i] = Some(retry);
        }
    }

    Ok(PoseTrack {
        version: POSE_TRACK_VERSION,
        skeleton: m.pose.skeleton(),
        model: m.pose.model_name().to_owned(),
        detector: m.detector.model_name().to_owned(),
        width,
        height,
        rotation: config.rotation,
        frames: frames
            .iter()
            .zip(&poses)
            .map(|((t, _), p)| match p {
                Some(p) if p.score >= threshold => PoseFrame::with_pose(*t, p),
                _ => PoseFrame::empty(*t),
            })
            .collect(),
    })
}

/// The golfer's box on each key frame: picked on the first key frame with a
/// person, then the best IoU match with the last box found (`None` where
/// nothing matches).
fn follow_golfer(
    detections: &[Option<(u32, u32, Vec<Detection>)>],
    width: u32,
    height: u32,
    min_iou: f32,
) -> Vec<Option<BBox>> {
    let mut last: Option<BBox> = None;
    detections
        .iter()
        .map(|d| {
            let dets = d.as_ref().map_or(&[][..], |d| &d.2[..]);
            let found = match last {
                Some(prev) => match_by_iou(dets, &prev, min_iou),
                None => pick_golfer(dets, width, height),
            };
            if found.is_some() {
                last = found;
            }
            found
        })
        .collect()
}

/// The box for frame `i`: interpolated between the nearest anchors before
/// and after it, or the nearest one when it lies outside them.
fn interpolate(anchors: &[(usize, BBox)], i: usize) -> Option<BBox> {
    let after = anchors.partition_point(|(k, _)| *k < i);
    match (
        after.checked_sub(1).and_then(|a| anchors.get(a)),
        anchors.get(after),
    ) {
        (_, Some((k, b))) if *k == i => Some(*b),
        (Some((ka, a)), Some((kb, b))) => {
            // Frame indices are far below f32's exact-integer range.
            #[allow(clippy::cast_precision_loss)]
            let t = (i - ka) as f32 / (kb - ka) as f32;
            Some(a.lerp(b, t))
        }
        (Some((_, a)), None) => Some(*a),
        (None, Some((_, b))) => Some(*b),
        (None, None) => None,
    }
}

/// Maps `f` over `items` on up to `workers` threads, each with its own fork
/// of the models and running `hook` first. Results keep the input order;
/// the first error stops the other workers and is returned.
fn parallel_map<P, D, T, R, F>(
    items: &[T],
    workers: usize,
    hook: Option<&WorkerHook>,
    models: &PoseModels<P, D>,
    f: F,
) -> Result<Vec<R>>
where
    P: PoseEngine,
    D: PersonDetector,
    T: Sync,
    R: Send,
    F: Fn(&mut PoseModels<P, D>, &T) -> Result<R> + Sync,
{
    let workers = workers.clamp(1, items.len().max(1));
    // Forks are made here: model state moves to a thread but isn't shared.
    let forks = (0..workers)
        .map(|_| models.fork())
        .collect::<Result<Vec<_>>>()?;
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    let first_error: Mutex<Option<Error>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for mut m in forks {
            let (next, failed, results, first_error, f) =
                (&next, &failed, &results, &first_error, &f);
            scope.spawn(move || {
                if let Some(hook) = hook {
                    hook.run();
                }
                while !failed.load(Ordering::SeqCst) {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(item) = items.get(i) else { break };
                    match f(&mut m, item) {
                        Ok(r) => {
                            let mut res = results.lock().unwrap_or_else(PoisonError::into_inner);
                            if let Some(slot) = res.get_mut(i) {
                                *slot = Some(r);
                            }
                        }
                        Err(e) => {
                            failed.store(true, Ordering::SeqCst);
                            first_error
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .get_or_insert(e);
                            break;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = first_error
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
    {
        return Err(e);
    }
    results
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .into_iter()
        .map(|r| r.ok_or_else(|| Error::InvalidArgument("a worker stopped early".into())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{Image, Keypoint, Rotation, Skeleton};

    /// A pose engine whose score is the image's brightness: dark frames
    /// pose badly wherever they are cropped.
    struct BrightnessPose;

    impl PoseEngine for BrightnessPose {
        fn skeleton(&self) -> Skeleton {
            Skeleton::Halpe26
        }
        fn model_name(&self) -> &'static str {
            "brightness"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self)
        }
        fn estimate(&mut self, image: &Image, bbox: &BBox) -> Result<Pose> {
            let score = f32::from(image.pixel(0, 0).unwrap_or_default()[0]) / 255.0;
            let (cx, cy) = bbox.center();
            let kps = (0..26u8)
                .map(|i| Keypoint {
                    x: cx + (f32::from(i % 5) - 2.0) * bbox.width() / 5.0,
                    y: cy + (f32::from(i / 5) - 2.5) * bbox.height() / 6.0,
                    score,
                })
                .collect();
            Ok(Pose::new(Skeleton::Halpe26, kps, *bbox))
        }
    }

    /// A detector that always sees the same person.
    struct OnePerson;

    impl PersonDetector for OnePerson {
        fn model_name(&self) -> &'static str {
            "one-person"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self)
        }
        fn detect(&mut self, _image: &Image) -> Result<Vec<Detection>> {
            Ok(vec![Detection {
                bbox: BBox::new(16.0, 8.0, 48.0, 40.0),
                score: 0.9,
            }])
        }
    }

    fn grey_jpeg(level: u8) -> Vec<u8> {
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 90)
            .encode(&[level; 64 * 48 * 3], 64, 48, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        out
    }

    #[test]
    fn any_engine_runs_the_pass_and_low_scores_are_gaps() {
        let levels = [230u8, 230, 230, 230, 230, 230, 20, 20, 20, 230, 230, 230];
        let frames: Vec<(u64, Vec<u8>)> = (0u64..)
            .zip(levels)
            .map(|(i, l)| (i * 8_333_333, grey_jpeg(l)))
            .collect();
        let config = PoseConfig {
            detect_every: 4,
            threads: 3,
            ..PoseConfig::default()
        };
        let models = PoseModels::new(BrightnessPose, OnePerson);
        let track = analyze_frames(&models, &frames, &config).unwrap();
        assert_eq!(
            (track.model.as_str(), track.detector.as_str()),
            ("brightness", "one-person")
        );
        assert_eq!((track.width, track.height), (64, 48));
        for (f, l) in track.frames.iter().zip(levels) {
            if l < 100 {
                assert_eq!(*f, PoseFrame::empty(f.t_ns), "dark frame at {}", f.t_ns);
            } else {
                assert!(f.score.unwrap() > 0.85, "frame at {}", f.t_ns);
                assert_eq!(f.keypoints.len(), 26);
            }
        }
        assert!(matches!(
            analyze_frames(&models, &frames[..0], &config),
            Err(Error::InvalidArgument(_))
        ));
    }

    std::thread_local! {
        /// Set by the test's hook on the thread running it.
        static HOOKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// A detector that sees the same person and records, per call, whether
    /// its thread had run the hook.
    struct Recorded(Arc<Mutex<Vec<bool>>>);

    impl PersonDetector for Recorded {
        fn model_name(&self) -> &'static str {
            "recorded"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self(Arc::clone(&self.0)))
        }
        fn detect(&mut self, image: &Image) -> Result<Vec<Detection>> {
            self.0.lock().unwrap().push(HOOKED.get());
            OnePerson.detect(image)
        }
    }

    #[test]
    fn the_hook_runs_once_on_every_worker_thread_before_its_work() {
        let frames: Vec<(u64, Vec<u8>)> = (0u64..12).map(|i| (i, grey_jpeg(230))).collect();
        let hooked: Arc<Mutex<Vec<std::thread::ThreadId>>> = Arc::default();
        let detected: Arc<Mutex<Vec<bool>>> = Arc::default();
        let record = Arc::clone(&hooked);
        let hook = WorkerHook::new(move || {
            HOOKED.set(true);
            record.lock().unwrap().push(std::thread::current().id());
        });
        assert_eq!(hook, hook.clone());
        assert_ne!(hook, WorkerHook::new(|| {}));
        assert_eq!(format!("{hook:?}"), "WorkerHook(..)");
        let config = PoseConfig {
            detect_every: 4,
            threads: 3,
            on_worker_start: Some(hook),
            ..PoseConfig::default()
        };
        let models = PoseModels::new(BrightnessPose, Recorded(Arc::clone(&detected)));
        analyze_frames(&models, &frames, &config).unwrap();
        let hooked = hooked.lock().unwrap().clone();
        // Detection on 4 key frames, then pose on 12 frames: 3 workers each.
        assert_eq!(hooked.len(), 6);
        let distinct: std::collections::HashSet<_> = hooked.iter().collect();
        assert_eq!(distinct.len(), hooked.len(), "once per thread");
        assert!(!distinct.contains(&std::thread::current().id()));
        assert_eq!(*detected.lock().unwrap(), [true; 4], "hooked before work");
        assert!(!HOOKED.get());
    }

    #[test]
    fn no_hook_is_the_default() {
        assert_eq!(PoseConfig::default().on_worker_start, None);
    }

    /// The centroid of the image's bright pixels.
    fn blob(image: &Image) -> (f32, f32) {
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

    /// A pose engine and detector that find the one bright blob: keypoints
    /// at fixed offsets from it, in image pixels.
    struct BlobPose;

    impl PoseEngine for BlobPose {
        fn skeleton(&self) -> Skeleton {
            Skeleton::Halpe26
        }
        fn model_name(&self) -> &'static str {
            "blob"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self)
        }
        fn estimate(&mut self, image: &Image, bbox: &BBox) -> Result<Pose> {
            let (x, y) = blob(image);
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

    impl PersonDetector for BlobPose {
        fn model_name(&self) -> &'static str {
            "blob"
        }
        fn fork(&self) -> Result<Self> {
            Ok(Self)
        }
        fn detect(&mut self, image: &Image) -> Result<Vec<Detection>> {
            let (x, y) = blob(image);
            Ok(vec![Detection {
                bbox: BBox::new(x - 10.0, y - 10.0, x + 10.0, y + 10.0),
                score: 0.9,
            }])
        }
    }

    fn encode(image: &Image) -> Vec<u8> {
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 95)
            .encode(
                image.data(),
                u16::try_from(image.width()).unwrap(),
                u16::try_from(image.height()).unwrap(),
                jpeg_encoder::ColorType::Rgb,
            )
            .unwrap();
        out
    }

    #[test]
    fn rotated_frames_give_the_upright_track() {
        // An upright 64 × 48 scene with a bright square that moves.
        let scene = |i: u32| {
            let mut data = vec![0u8; 64 * 48 * 3];
            for y in 10..18 {
                for x in 20 + 2 * i..28 + 2 * i {
                    let p = ((y * 64 + x) * 3) as usize;
                    data[p..p + 3].copy_from_slice(&[255, 255, 255]);
                }
            }
            Image::new(64, 48, data).unwrap()
        };
        let models = PoseModels::new(BlobPose, BlobPose);
        let config = PoseConfig {
            detect_every: 2,
            threads: 2,
            ..PoseConfig::default()
        };
        let frames = |turn: Rotation| -> Vec<(u64, Vec<u8>)> {
            (0..6)
                .map(|i| (u64::from(i) * 8_333_333, encode(&scene(i).rotated(turn))))
                .collect()
        };
        let upright = analyze_frames(&models, &frames(Rotation::None), &config).unwrap();
        assert_eq!((upright.width, upright.height), (64, 48));
        assert_eq!(upright.rotation, Rotation::None);
        for (rotation, mounted) in [
            (Rotation::Cw90, Rotation::Ccw90),
            (Rotation::Half, Rotation::Half),
            (Rotation::Ccw90, Rotation::Cw90),
        ] {
            // A camera mounted turned by `mounted` records the scene turned
            // that way; `rotation` turns it back.
            let stored = frames(mounted);
            let raw = analyze_frames(&models, &stored, &config).unwrap();
            assert_eq!((raw.width, raw.height), mounted.dimensions(64, 48));
            let config = PoseConfig {
                rotation,
                ..config.clone()
            };
            let track = analyze_frames(&models, &stored, &config).unwrap();
            assert_eq!(track.rotation, rotation);
            assert_eq!((track.width, track.height), (64, 48), "{rotation:?}");
            for (f, g) in track.frames.iter().zip(&upright.frames) {
                assert_eq!(f.t_ns, g.t_ns);
                assert_eq!(f.keypoints.len(), 26);
                for (k, u) in f.keypoints.iter().zip(&g.keypoints) {
                    assert!(
                        (k.x - u.x).abs() < 0.3 && (k.y - u.y).abs() < 0.3,
                        "{rotation:?} at {}: {k:?} vs {u:?}",
                        f.t_ns
                    );
                }
            }
        }
    }

    fn b(x: f32) -> BBox {
        BBox::new(x, 0.0, x + 10.0, 20.0)
    }

    #[test]
    fn interpolates_between_anchors() {
        let anchors = [(2, b(0.0)), (6, b(40.0))];
        assert_eq!(interpolate(&anchors, 0), Some(b(0.0)));
        assert_eq!(interpolate(&anchors, 2), Some(b(0.0)));
        assert_eq!(interpolate(&anchors, 4), Some(b(20.0)));
        assert_eq!(interpolate(&anchors, 5), Some(b(30.0)));
        assert_eq!(interpolate(&anchors, 6), Some(b(40.0)));
        assert_eq!(interpolate(&anchors, 9), Some(b(40.0)));
        assert_eq!(interpolate(&[], 3), None);
    }

    #[test]
    fn follows_the_golfer_by_iou() {
        let d = |x: f32, s: f32| Detection {
            bbox: BBox::new(x, 100.0, x + 100.0, 400.0),
            score: s,
        };
        let frames = vec![
            None,
            Some((640, 480, vec![d(270.0, 0.8), d(0.0, 0.9)])),
            Some((640, 480, vec![d(0.0, 0.9)])),
            // The golfer moved a little; a bystander scores higher.
            Some((640, 480, vec![d(10.0, 0.95), d(285.0, 0.6)])),
        ];
        let chosen = follow_golfer(&frames, 640, 480, 0.3);
        assert_eq!(
            chosen,
            vec![
                None,
                Some(d(270.0, 0.0).bbox),
                None,
                Some(d(285.0, 0.0).bbox)
            ]
        );
    }
}
