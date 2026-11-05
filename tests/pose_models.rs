//! Pose tests against the real models.
//!
//! They run only when `STANCE_MODEL_DIR` names a directory holding the
//! models as `<name>.onnx` (see `ModelSpec`); otherwise each prints a note
//! and passes. `tests/data/golfer.rtmlib.json` holds rtmlib's keypoints for
//! `tests/data/golfer.jpg` (see `tests/data/README.md`).

#![cfg(feature = "pose")]

use std::path::PathBuf;
use std::sync::Arc;

use serde::Deserialize;
use stance::pose::{
    BBox, Image, ModelSpec, PersonDetector, PoseConfig, PoseEngine, PoseModels, PoseTrack,
    RTMPOSE_M_HALPE26, RTMPOSE_S_HALPE26, RtmPose, YOLOX_TINY, Yolox, analyze_clip, decode_jpeg,
};
use stance::{Clip, ClipId, Frame};

/// Keypoints with a score above this are compared with rtmlib.
const PARITY_MIN_SCORE: f32 = 0.3;
/// Largest keypoint distance from rtmlib's, in image pixels.
const PARITY_MAX_PX: f32 = 2.0;
/// Largest mean keypoint distance from rtmlib's, in image pixels.
const PARITY_MEAN_PX: f32 = 0.5;
/// Largest keypoint score difference from rtmlib's.
const PARITY_MAX_SCORE: f32 = 0.03;

fn model_dir(test: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("STANCE_MODEL_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("{test}: skipped, set STANCE_MODEL_DIR to a directory with the models");
    }
    dir
}

fn has(dir: &std::path::Path, spec: &ModelSpec, test: &str) -> bool {
    let ok = spec.path_in(dir).exists();
    if !ok {
        eprintln!("{test}: skipping {}, not in {}", spec.name, dir.display());
    }
    ok
}

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn golfer() -> Image {
    decode_jpeg(&std::fs::read(data("golfer.jpg")).unwrap()).unwrap()
}

fn flip(image: &Image) -> Image {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let src = image.data();
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        for x in 0..w {
            let (s, d) = ((y * w + x) * 3, (y * w + (w - 1 - x)) * 3);
            out[d..d + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
    Image::new(image.width(), image.height(), out).unwrap()
}

#[derive(Deserialize)]
struct Reference {
    frames: Vec<RefFrame>,
}

#[derive(Deserialize)]
struct RefFrame {
    name: String,
    flip: bool,
    detections: Vec<[f32; 4]>,
    poses: Vec<RefPose>,
}

#[derive(Deserialize)]
struct RefPose {
    model: String,
    bbox_name: String,
    bbox: [f32; 4],
    keypoints: Vec<[f32; 2]>,
    scores: Vec<f32>,
}

fn reference() -> Reference {
    serde_json::from_str(&std::fs::read_to_string(data("golfer.rtmlib.json")).unwrap()).unwrap()
}

fn bbox(b: [f32; 4]) -> BBox {
    BBox::new(b[0], b[1], b[2], b[3])
}

#[test]
fn rtmpose_matches_rtmlib() {
    let test = "rtmpose_matches_rtmlib";
    let Some(dir) = model_dir(test) else { return };
    let image = golfer();
    let flipped = flip(&image);
    let reference = reference();
    let mut compared = 0;
    for spec in [RTMPOSE_S_HALPE26, RTMPOSE_M_HALPE26] {
        if !has(&dir, &spec, test) {
            continue;
        }
        let mut model = RtmPose::load(&spec, &dir).unwrap();
        let (mut worst, mut sum, mut n, mut worst_score) = (0.0f32, 0.0f32, 0usize, 0.0f32);
        for frame in &reference.frames {
            let img = if frame.flip { &flipped } else { &image };
            for r in frame.poses.iter().filter(|p| p.model == spec.name) {
                let pose = model.estimate(img, &bbox(r.bbox)).unwrap();
                let (mut fw, mut fs) = (0.0f32, 0.0f32);
                for (k, (xy, s)) in pose.keypoints.iter().zip(r.keypoints.iter().zip(&r.scores)) {
                    let ds = (k.score - s).abs();
                    fs = fs.max(ds);
                    if *s < PARITY_MIN_SCORE {
                        continue;
                    }
                    let d = ((k.x - xy[0]).powi(2) + (k.y - xy[1]).powi(2)).sqrt();
                    fw = fw.max(d);
                    sum += d;
                    n += 1;
                }
                eprintln!(
                    "{} {} {}: max {fw:.3} px, max score diff {fs:.4}",
                    spec.name, frame.name, r.bbox_name
                );
                worst = worst.max(fw);
                worst_score = worst_score.max(fs);
                compared += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / n.max(1) as f32;
        eprintln!(
            "{}: {n} keypoints, mean {mean:.3} px, max {worst:.3} px, max score diff {worst_score:.4}",
            spec.name
        );
        assert!(worst <= PARITY_MAX_PX, "{}: max {worst} px", spec.name);
        assert!(mean <= PARITY_MEAN_PX, "{}: mean {mean} px", spec.name);
        assert!(
            worst_score <= PARITY_MAX_SCORE,
            "{}: score diff {worst_score}",
            spec.name
        );
    }
    eprintln!("{test}: compared {compared} poses");
}

#[test]
fn detector_finds_the_golfer() {
    let test = "detector_finds_the_golfer";
    let Some(dir) = model_dir(test) else { return };
    if !has(&dir, &YOLOX_TINY, test) {
        return;
    }
    let mut det = Yolox::load(&YOLOX_TINY, &dir).unwrap();
    let image = golfer();
    let flipped = flip(&image);
    for frame in reference().frames {
        let img = if frame.flip { &flipped } else { &image };
        let dets = det.detect(img).unwrap();
        let expect = bbox(frame.detections[0]);
        let best = dets.first().expect("a person");
        let iou = best.bbox.iou(&expect);
        eprintln!(
            "{test} {}: {} people, best {:.3} at {:?}, IoU with rtmlib's {iou:.4}",
            frame.name,
            dets.len(),
            best.score,
            best.bbox
        );
        assert!(iou > 0.98, "IoU {iou}");
    }
}

/// A clip of the test image sliding right by 2 px per frame at 120 fps.
fn sliding_clip(frames: usize) -> Clip<&'static str> {
    let image = golfer();
    let (w, h) = (image.width() as usize, image.height() as usize);
    let frames: Vec<Frame> = (0..frames)
        .map(|i| {
            let shift = 2 * i;
            let mut rgb = vec![0u8; w * h * 3];
            for y in 0..h {
                let row = y * w * 3;
                rgb[row + shift * 3..row + w * 3]
                    .copy_from_slice(&image.data()[row..row + (w - shift) * 3]);
            }
            let mut jpeg = Vec::new();
            jpeg_encoder::Encoder::new(&mut jpeg, 85)
                .encode(
                    &rgb,
                    u16::try_from(w).unwrap(),
                    u16::try_from(h).unwrap(),
                    jpeg_encoder::ColorType::Rgb,
                )
                .unwrap();
            Frame {
                jpeg: Arc::from(jpeg),
                host_ns: 1_000_000_000 + i as u64 * 8_333_333,
                device_ns: None,
                seq: i as u64,
                width: image.width(),
                height: image.height(),
            }
        })
        .collect();
    let n = frames.len();
    Clip {
        id: ClipId(1),
        tags: vec!["shot"],
        trigger_ns: frames[n / 2].host_ns,
        impact_ns: frames[n / 2].host_ns,
        frames,
        trigger_index: n / 2,
        impact_index: n / 2,
        complete: true,
        dropped: 0,
        truncated: false,
        short_buffer: false,
        fps: Some(120.0),
    }
}

#[test]
fn analyze_clip_tracks_a_generated_clip() {
    let test = "analyze_clip_tracks_a_generated_clip";
    let Some(dir) = model_dir(test) else { return };
    let config = PoseConfig {
        pose_model: RTMPOSE_S_HALPE26,
        detect_every: 5,
        threads: 3,
        ..PoseConfig::default()
    };
    if !has(&dir, &config.pose_model, test) || !has(&dir, &config.detector, test) {
        return;
    }
    let models = PoseModels::load(&dir, &config).unwrap();
    let out = std::env::temp_dir().join(format!("stance-pose-clip-{}", std::process::id()));
    let clip = sliding_clip(24);
    let files = stance::write_clip(&out, "slide", &clip).unwrap();
    let t0 = std::time::Instant::now();
    let track = analyze_clip(&models, &files.avi, &config).unwrap();
    eprintln!(
        "{test}: {} frames in {:?}",
        track.frames.len(),
        t0.elapsed()
    );
    assert_eq!(track.frames.len(), 24);
    assert_eq!((track.width, track.height), (640, 480));
    assert_eq!(track.model, "rtmpose-s-halpe26");
    for (f, s) in track.frames.iter().zip(&files.sidecar.frames) {
        assert_eq!(f.t_ns, s.t_ns);
        assert!(f.score.unwrap() > 0.5, "score {:?} at {}", f.score, f.t_ns);
    }
    // The left hip follows the 2 px/frame slide.
    let hip = |i: usize| track.frames[i].keypoints[11].x;
    let moved = hip(23) - hip(0);
    eprintln!("{test}: left hip moved {moved:.1} px over 23 frames (46 expected)");
    assert!((moved - 46.0).abs() < 6.0, "moved {moved}");
    let path = track.write_beside(&files.avi).unwrap();
    assert_eq!(path, out.join("slide.pose.json"));
    assert_eq!(PoseTrack::read(&path).unwrap(), track);

    // The live tracker over the same frames agrees with the clip pass.
    let mut tracker = models.tracker(config).unwrap();
    for (i, frame) in clip.frames.iter().enumerate() {
        let image = decode_jpeg(&frame.jpeg).unwrap();
        let pose = tracker
            .track(&image, frame.host_ns)
            .unwrap()
            .expect("tracked");
        let d = (pose.keypoints[11].x - hip(i)).abs();
        assert!(d < 4.0, "frame {i}: live and clip pass differ by {d} px");
    }
    std::fs::remove_dir_all(&out).ok();
}
