//! Golf posture on real keypoints.
//!
//! Runs only when `STANCE_MODEL_DIR` names a directory holding the default
//! models as `<name>.onnx`; otherwise it prints a note and passes. The test
//! image (`tests/data/golfer.jpg`) shows a golfer from the side, facing the
//! image's left, at the top of the backswing: down the line, that is a
//! left-hander's view (or a right-hander's seen from the other side).

#![cfg(feature = "pose")]

use std::path::PathBuf;

use stance::golf::{Handed, View, posture};
use stance::pose::{Image, PoseConfig, PoseModels, decode_jpeg};

fn models(test: &str) -> Option<PoseModels> {
    let Some(dir) = std::env::var_os("STANCE_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("{test}: skipped, set STANCE_MODEL_DIR to a directory with the models");
        return None;
    };
    let config = PoseConfig::default();
    if !config.pose_model.path_in(&dir).exists() || !config.detector.path_in(&dir).exists() {
        eprintln!(
            "{test}: skipped, the default models are not in {}",
            dir.display()
        );
        return None;
    }
    Some(PoseModels::load(&dir, &config).unwrap())
}

fn golfer() -> Image {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/golfer.jpg");
    decode_jpeg(&std::fs::read(path).unwrap()).unwrap()
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

#[test]
fn posture_from_real_keypoints_mirrors() {
    let Some(models) = models("posture_from_real_keypoints_mirrors") else {
        return;
    };
    let mut tracker = models.tracker(PoseConfig::default()).unwrap();
    let pose = tracker.track(&golfer(), 0).unwrap().expect("a golfer");
    let left = posture(&pose, View::DownTheLine, Handed::Left);
    let bend = left.spine_angle.expect("spine visible");
    // Bent towards the ball (the image's left), as golfers are.
    assert!((10.0..60.0).contains(&bend), "forward bend {bend}");
    assert!(left.lead_knee_flex.is_some() && left.trail_knee_flex.is_some());
    assert!(left.torso_px.is_some_and(|t| t > 50.0));

    // The mirror image read for a right-hander gives the same angles, up to
    // the model's own left/right asymmetry (largest at the elbows, folded
    // at the top of the swing).
    let mut tracker = models.tracker(PoseConfig::default()).unwrap();
    let flipped = tracker
        .track(&flip(&golfer()), 0)
        .unwrap()
        .expect("a golfer");
    let right = posture(&flipped, View::DownTheLine, Handed::Right);
    let pairs = [
        (left.spine_angle, right.spine_angle, 6.0),
        (left.lead_knee_flex, right.lead_knee_flex, 6.0),
        (left.trail_knee_flex, right.trail_knee_flex, 6.0),
        (left.lead_elbow_angle, right.lead_elbow_angle, 15.0),
        (left.trail_elbow_angle, right.trail_elbow_angle, 15.0),
    ];
    for (a, b, tolerance) in pairs {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a - b).abs() < tolerance, "{a} vs {b}");
    }
}
