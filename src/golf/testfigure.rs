//! Test support: a parametric stick figure with known angles, shifts and
//! event times, and the helpers the golf tests share.

// Test fixtures use small exact values.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "synthetic fixtures"
)]

use std::f64::consts::PI;

use super::*;
use crate::pose::{
    BBox, Keypoint, POSE_TRACK_VERSION, Pose, PoseFrame, PoseTrack, Rotation, Skeleton,
};

pub(super) const FPS: f64 = 120.0;
pub(super) const DURATION_S: f64 = 3.5;
pub(super) const WIDTH: u32 = 1280;
pub(super) const TORSO: f64 = 200.0;
pub(super) const ARM: f64 = 190.0;
pub(super) const HAND_SWING: f64 = 150.0;

pub(super) const TAKEAWAY_S: f64 = 1.0;
pub(super) const TOP_S: f64 = 2.0;
pub(super) const IMPACT_S: f64 = 2.4;
pub(super) const FINISH_S: f64 = 2.8;
/// The face-on figure's spine lean away from the target at the finish
/// (degrees).
pub(super) const FINISH_LEAN: f64 = 33.0;
/// Impact is reported this long after the hands' low point.
pub(super) const IMPACT_LAG_S: f64 = 0.025;

/// Smooth 0 → 1 between `a` and `b` seconds.
pub(super) fn ramp(t: f64, a: f64, b: f64) -> f64 {
    let u = ((t - a) / (b - a)).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// A value per phase: address until 1.2 s, top from 1.7 to 2.1 s, impact
/// from 2.25 to 2.55 s, finish from 2.8 s; smooth ramps in between.
#[derive(Clone, Copy)]
pub(super) struct Phases {
    pub(super) address: f64,
    pub(super) top: f64,
    pub(super) impact: f64,
    pub(super) finish: f64,
}

impl Phases {
    pub(super) fn flat(v: f64) -> Self {
        Self {
            address: v,
            top: v,
            impact: v,
            finish: v,
        }
    }

    pub(super) fn at(&self, t: f64) -> f64 {
        let v = self.address + (self.top - self.address) * ramp(t, 1.2, 1.7);
        let v = v + (self.impact - self.top) * ramp(t, 2.1, 2.25);
        v + (self.finish - self.impact) * ramp(t, 2.55, 2.8)
    }
}

/// The hands' angle about the shoulders (radians, 0 hanging down, positive
/// toward the right-hander's lead side, +x) for a swing reaching `swing`
/// degrees: still, a cosine backswing to the trail side, an accelerating
/// downswing that is fastest at impact, and a decelerating follow-through
/// to the lead side, then still.
pub(super) fn hand_angle(t: f64, swing: f64) -> f64 {
    let a = swing.to_radians();
    if t < TAKEAWAY_S {
        0.0
    } else if t < TOP_S {
        let u = (t - TAKEAWAY_S) / (TOP_S - TAKEAWAY_S);
        -a * (1.0 - (PI * u).cos()) / 2.0
    } else if t < IMPACT_S {
        let u = (t - TOP_S) / (IMPACT_S - TOP_S);
        -a * (1.0 - u * u)
    } else if t < FINISH_S {
        let u = (t - IMPACT_S) / (FINISH_S - IMPACT_S);
        a * (1.0 - (1.0 - u) * (1.0 - u))
    } else {
        a
    }
}

/// The figure's settings for one view (right-handed; positions in pixels,
/// angles in degrees).
#[derive(Clone, Copy)]
pub(super) struct Figure {
    pub(super) view: View,
    /// Pelvis x shift (+x is the right-hander's target face-on, the ball
    /// down the line).
    pub(super) pelvis_dx: Phases,
    /// Pelvis y shift (image y: positive is down).
    pub(super) pelvis_dy: Phases,
    /// Hip half-width (the true half-length of the hip line).
    pub(super) hip_half: Phases,
    /// Hip line tilt, positive with the lead hip higher.
    pub(super) hip_tilt: Phases,
    /// Hip line turn about the vertical (degrees from square).
    pub(super) hip_turn: Phases,
    /// Shoulder half-width (the true half-length of the shoulder line).
    pub(super) shoulder_half: Phases,
    /// Spine tilt from vertical, positive toward +x.
    pub(super) spine: Phases,
    /// Shoulder midpoint x shift on top of the spine tilt.
    pub(super) chest_dx: Phases,
    /// Shoulder line tilt, positive with the lead shoulder higher.
    pub(super) shoulder_tilt: Phases,
    /// Shoulder line turn about the vertical (degrees from square).
    pub(super) shoulder_turn: Phases,
    /// The hands' swing (degrees each way, see [`hand_angle`]).
    pub(super) hand_swing: f64,
    pub(super) head_dx: Phases,
    pub(super) head_dy: Phases,
    /// Elbow angles (180 = straight).
    pub(super) lead_elbow: Phases,
    pub(super) trail_elbow: Phases,
    /// Knee flexion (0 = straight).
    pub(super) lead_knee: Phases,
    pub(super) trail_knee: Phases,
}

impl Figure {
    pub(super) fn face_on() -> Self {
        Self {
            view: View::FaceOn,
            pelvis_dx: Phases {
                address: 0.0,
                top: -20.0,
                impact: 30.0,
                finish: 40.0,
            },
            pelvis_dy: Phases::flat(0.0),
            hip_half: Phases::flat(45.0),
            hip_tilt: Phases::flat(0.0),
            // The hips show 0.8, 0.9 and 0.5 of their width.
            hip_turn: Phases {
                address: 0.0,
                top: 0.8f64.acos().to_degrees(),
                impact: 0.9f64.acos().to_degrees(),
                finish: 60.0,
            },
            shoulder_half: Phases::flat(85.0),
            // Leaning back away from the target at the finish: a 33° bend
            // back (the "reverse C"), seen side-on.
            spine: Phases {
                address: -6.0,
                top: -2.0,
                impact: -12.0,
                finish: -FINISH_LEAN,
            },
            chest_dx: Phases::flat(0.0),
            shoulder_tilt: Phases {
                address: 10.0,
                top: -40.0,
                impact: 25.0,
                finish: 0.0,
            },
            shoulder_turn: Phases {
                address: 0.0,
                top: 70.0,
                impact: 30.0,
                finish: 60.0,
            },
            hand_swing: HAND_SWING,
            head_dx: Phases {
                address: 0.0,
                top: -10.0,
                impact: 6.0,
                finish: 40.0,
            },
            head_dy: Phases {
                address: 0.0,
                top: 4.0,
                impact: 12.0,
                finish: -10.0,
            },
            lead_elbow: Phases {
                address: 175.0,
                top: 160.0,
                impact: 178.0,
                finish: 120.0,
            },
            trail_elbow: Phases::flat(170.0),
            lead_knee: Phases::flat(15.0),
            trail_knee: Phases::flat(15.0),
        }
    }

    pub(super) fn down_the_line() -> Self {
        Self {
            view: View::DownTheLine,
            pelvis_dx: Phases {
                address: 0.0,
                top: 4.0,
                impact: 16.0,
                finish: 10.0,
            },
            pelvis_dy: Phases::flat(0.0),
            hip_half: Phases::flat(4.0),
            hip_tilt: Phases::flat(0.0),
            hip_turn: Phases::flat(0.0),
            shoulder_half: Phases::flat(5.0),
            spine: Phases {
                address: 35.0,
                top: 33.0,
                impact: 30.0,
                finish: 10.0,
            },
            chest_dx: Phases::flat(0.0),
            shoulder_tilt: Phases::flat(0.0),
            shoulder_turn: Phases::flat(0.0),
            hand_swing: HAND_SWING,
            head_dx: Phases::flat(0.0),
            head_dy: Phases {
                address: 0.0,
                top: -2.0,
                impact: 8.0,
                finish: -30.0,
            },
            lead_elbow: Phases::flat(170.0),
            trail_elbow: Phases {
                address: 165.0,
                top: 80.0,
                impact: 150.0,
                finish: 100.0,
            },
            lead_knee: Phases {
                address: 20.0,
                top: 35.0,
                impact: 10.0,
                finish: 2.0,
            },
            trail_knee: Phases {
                address: 20.0,
                top: 18.0,
                impact: 30.0,
                finish: 5.0,
            },
        }
    }

    /// The pelvis at address.
    pub(super) fn pelvis0(&self) -> (f64, f64) {
        match self.view {
            View::FaceOn => (640.0, 520.0),
            View::DownTheLine => (600.0, 520.0),
        }
    }

    pub(super) fn pelvis(&self, t: f64) -> (f64, f64) {
        let (x, y) = self.pelvis0();
        (x + self.pelvis_dx.at(t), y + self.pelvis_dy.at(t))
    }

    pub(super) fn shoulders(&self, t: f64) -> (f64, f64) {
        let (px, py) = self.pelvis(t);
        let s = self.spine.at(t).to_radians();
        (
            px + TORSO * s.sin() + self.chest_dx.at(t),
            py - TORSO * s.cos(),
        )
    }

    pub(super) fn hands(&self, t: f64) -> (f64, f64) {
        let (sx, sy) = self.shoulders(t);
        let a = hand_angle(t, self.hand_swing);
        (sx + ARM * a.sin(), sy + ARM * a.cos())
    }

    /// A body line's lead-end offset from its midpoint (x, image y) with
    /// true half-length `half`, tilted by `tilt` and turned by `turn`
    /// (degrees): `half·cos(tilt)·cos(turn)` across, `half·sin(tilt)` up.
    pub(super) fn line(half: f64, tilt: f64, turn: f64) -> (f64, f64) {
        let (tilt, turn) = (tilt.to_radians(), turn.to_radians());
        (half * tilt.cos() * turn.cos(), -half * tilt.sin())
    }

    /// The lead and trail hips.
    pub(super) fn hips(&self, t: f64) -> ((f64, f64), (f64, f64)) {
        let (px, py) = self.pelvis(t);
        let (dx, dy) = Self::line(
            self.hip_half.at(t),
            self.hip_tilt.at(t),
            self.hip_turn.at(t),
        );
        ((px + dx, py + dy), (px - dx, py - dy))
    }

    /// The lead and trail shoulders.
    pub(super) fn shoulder_points(&self, t: f64) -> ((f64, f64), (f64, f64)) {
        let (sx, sy) = self.shoulders(t);
        let (dx, dy) = Self::line(
            self.shoulder_half.at(t),
            self.shoulder_tilt.at(t),
            self.shoulder_turn.at(t),
        );
        ((sx + dx, sy + dy), (sx - dx, sy - dy))
    }

    /// The right-hander's 26 keypoints at `t`.
    pub(super) fn keypoints(&self, t: f64) -> Vec<Keypoint> {
        let mut k = vec![(0.0, 0.0); 26];
        let set = |k: &mut Vec<(f64, f64)>, j: Joint, p: (f64, f64)| k[j.index(Handed::Right)] = p;
        let (px, py) = self.pelvis(t);
        let (sx, sy) = self.shoulders(t);
        let (lead_hip, trail_hip) = self.hips(t);
        set(&mut k, Joint::Hip, (px, py));
        set(&mut k, Joint::LeadHip, lead_hip);
        set(&mut k, Joint::TrailHip, trail_hip);
        let (lead_shoulder, trail_shoulder) = self.shoulder_points(t);
        set(&mut k, Joint::LeadShoulder, lead_shoulder);
        set(&mut k, Joint::TrailShoulder, trail_shoulder);
        set(&mut k, Joint::Neck, (sx, sy));
        let (x0, y0) = self.pelvis0();
        let head0 = match self.view {
            View::FaceOn => (x0 - TORSO * 6f64.to_radians().sin(), y0 - TORSO - 95.0),
            View::DownTheLine => (x0 + 160.0, y0 - 230.0),
        };
        let head = (head0.0 + self.head_dx.at(t), head0.1 + self.head_dy.at(t));
        set(&mut k, Joint::Head, head);
        set(&mut k, Joint::Nose, (head.0, head.1 + 30.0));
        set(&mut k, Joint::LeadEye, (head.0 + 10.0, head.1 + 22.0));
        set(&mut k, Joint::TrailEye, (head.0 - 10.0, head.1 + 22.0));
        set(&mut k, Joint::LeadEar, (head.0 + 20.0, head.1 + 25.0));
        set(&mut k, Joint::TrailEar, (head.0 - 20.0, head.1 + 25.0));
        let (hx, hy) = self.hands(t);
        let lead_wrist = (hx + 3.0, hy);
        let trail_wrist = (hx - 3.0, hy);
        set(&mut k, Joint::LeadWrist, lead_wrist);
        set(&mut k, Joint::TrailWrist, trail_wrist);
        set(
            &mut k,
            Joint::LeadElbow,
            joint_between(lead_shoulder, lead_wrist, self.lead_elbow.at(t), 1.0),
        );
        set(
            &mut k,
            Joint::TrailElbow,
            joint_between(trail_shoulder, trail_wrist, self.trail_elbow.at(t), -1.0),
        );
        let (lead_ankle, trail_ankle) = match self.view {
            View::FaceOn => ((x0 + 70.0, 760.0), (x0 - 70.0, 760.0)),
            View::DownTheLine => ((x0 + 12.0, 760.0), (x0 + 8.0, 760.0)),
        };
        set(&mut k, Joint::LeadAnkle, lead_ankle);
        set(&mut k, Joint::TrailAnkle, trail_ankle);
        // Knees bend forwards (+x) down the line, outwards face-on.
        let (lead_out, trail_out) = match self.view {
            View::FaceOn => (-1.0, 1.0),
            View::DownTheLine => (-1.0, -1.0),
        };
        set(
            &mut k,
            Joint::LeadKnee,
            joint_between(lead_hip, lead_ankle, 180.0 - self.lead_knee.at(t), lead_out),
        );
        set(
            &mut k,
            Joint::TrailKnee,
            joint_between(
                trail_hip,
                trail_ankle,
                180.0 - self.trail_knee.at(t),
                trail_out,
            ),
        );
        for (ankle, toe, small, heel) in [
            (
                lead_ankle,
                Joint::LeadBigToe,
                Joint::LeadSmallToe,
                Joint::LeadHeel,
            ),
            (
                trail_ankle,
                Joint::TrailBigToe,
                Joint::TrailSmallToe,
                Joint::TrailHeel,
            ),
        ] {
            set(&mut k, toe, (ankle.0 + 25.0, ankle.1 + 25.0));
            set(&mut k, small, (ankle.0 + 15.0, ankle.1 + 28.0));
            set(&mut k, heel, (ankle.0 - 10.0, ankle.1 + 15.0));
        }
        k.into_iter()
            .map(|(x, y)| Keypoint {
                x: x as f32,
                y: y as f32,
                score: 0.9,
            })
            .collect()
    }

    pub(super) fn track(&self) -> PoseTrack {
        let n = (DURATION_S * FPS) as usize;
        let frames = (0..n)
            .map(|i| {
                let t = i as f64 / FPS;
                let pose = Pose::new(
                    Skeleton::Halpe26,
                    self.keypoints(t),
                    BBox::new(0.0, 0.0, 1280.0, 800.0),
                );
                PoseFrame::with_pose(t_ns(t), &pose)
            })
            .collect();
        PoseTrack {
            version: POSE_TRACK_VERSION,
            skeleton: Skeleton::Halpe26,
            model: "synthetic".into(),
            detector: "synthetic".into(),
            width: WIDTH,
            height: 800,
            rotation: Rotation::None,
            frames,
        }
    }

    pub(super) fn pose(&self, t: f64) -> Pose {
        Pose::new(Skeleton::Halpe26, self.keypoints(t), BBox::default())
    }
}

/// The middle joint of a two-segment limb of equal segments from `a` to
/// `c` with the angle `deg` at it, bent to the `side` (±1) of `a → c`.
pub(super) fn joint_between(start: (f64, f64), end: (f64, f64), deg: f64, side: f64) -> (f64, f64) {
    let mid = (f64::midpoint(start.0, end.0), f64::midpoint(start.1, end.1));
    let span = (end.0 - start.0).hypot(end.1 - start.1);
    let segment = span / (2.0 * (deg.to_radians() / 2.0).sin());
    let offset = (segment * segment - span * span / 4.0).max(0.0).sqrt();
    let (ux, uy) = ((end.0 - start.0) / span, (end.1 - start.1) / span);
    (mid.0 - side * uy * offset, mid.1 + side * ux * offset)
}

pub(super) fn t_ns(t: f64) -> u64 {
    (t * 1e9).round() as u64
}

pub(super) fn secs(e: Option<SwingEvent>) -> f64 {
    e.expect("event found").t_ns as f64 * 1e-9
}

/// The track as a left-hander would produce it: mirrored image, labels
/// swapped.
pub(super) fn mirrored(track: &PoseTrack) -> PoseTrack {
    let mirror = Skeleton::Halpe26.mirror();
    let w = track.width as f32;
    let mut out = track.clone();
    for f in &mut out.frames {
        let original = f.keypoints.clone();
        for (i, k) in original.iter().enumerate() {
            f.keypoints[mirror[i]] = Keypoint { x: w - k.x, ..*k };
        }
    }
    out
}

pub(super) fn options(view: View, handed: Handed, impact: Option<f64>) -> SwingOptions {
    SwingOptions {
        view: Some(view),
        handed,
        impact_ns: impact.map(t_ns),
        height_cm: None,
        mirrored: false,
        smoothing: Smoothing::default(),
    }
}

pub(super) fn raw() -> Smoothing {
    Smoothing {
        cutoff_hz: 0.0,
        ..Smoothing::default()
    }
}

pub(super) fn value(report: &SwingReport, id: MetricId) -> Positions {
    report
        .metric(id)
        .unwrap_or_else(|| panic!("{id:?} measured"))
        .values
}

pub(super) fn near(a: Option<f64>, b: f64, tolerance: f64) -> bool {
    a.is_some_and(|a| (a - b).abs() <= tolerance)
}
