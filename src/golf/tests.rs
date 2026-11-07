//! Synthetic swings: a parametric stick figure with known angles, shifts
//! and event times.

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
use crate::testutil::TempDir;

const FPS: f64 = 120.0;
const DURATION_S: f64 = 3.5;
const WIDTH: u32 = 1280;
const TORSO: f64 = 200.0;
const ARM: f64 = 190.0;
const HAND_SWING: f64 = 150.0;

const TAKEAWAY_S: f64 = 1.0;
const TOP_S: f64 = 2.0;
const IMPACT_S: f64 = 2.4;
const FINISH_S: f64 = 2.8;
/// The face-on figure's spine lean away from the target at the finish
/// (degrees).
const FINISH_LEAN: f64 = 33.0;
/// Impact is reported this long after the hands' low point.
const IMPACT_LAG_S: f64 = 0.025;

/// Smooth 0 → 1 between `a` and `b` seconds.
fn ramp(t: f64, a: f64, b: f64) -> f64 {
    let u = ((t - a) / (b - a)).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// A value per phase: address until 1.2 s, top from 1.7 to 2.1 s, impact
/// from 2.25 to 2.55 s, finish from 2.8 s; smooth ramps in between.
#[derive(Clone, Copy)]
struct Phases {
    address: f64,
    top: f64,
    impact: f64,
    finish: f64,
}

impl Phases {
    fn flat(v: f64) -> Self {
        Self {
            address: v,
            top: v,
            impact: v,
            finish: v,
        }
    }

    fn at(&self, t: f64) -> f64 {
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
fn hand_angle(t: f64, swing: f64) -> f64 {
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
struct Figure {
    view: View,
    /// Pelvis x shift (+x is the right-hander's target face-on, the ball
    /// down the line).
    pelvis_dx: Phases,
    /// Pelvis y shift (image y: positive is down).
    pelvis_dy: Phases,
    /// Hip half-width (the true half-length of the hip line).
    hip_half: Phases,
    /// Hip line tilt, positive with the lead hip higher.
    hip_tilt: Phases,
    /// Hip line turn about the vertical (degrees from square).
    hip_turn: Phases,
    /// Shoulder half-width (the true half-length of the shoulder line).
    shoulder_half: Phases,
    /// Spine tilt from vertical, positive toward +x.
    spine: Phases,
    /// Shoulder midpoint x shift on top of the spine tilt.
    chest_dx: Phases,
    /// Shoulder line tilt, positive with the lead shoulder higher.
    shoulder_tilt: Phases,
    /// Shoulder line turn about the vertical (degrees from square).
    shoulder_turn: Phases,
    /// The hands' swing (degrees each way, see [`hand_angle`]).
    hand_swing: f64,
    head_dx: Phases,
    head_dy: Phases,
    /// Elbow angles (180 = straight).
    lead_elbow: Phases,
    trail_elbow: Phases,
    /// Knee flexion (0 = straight).
    lead_knee: Phases,
    trail_knee: Phases,
}

impl Figure {
    fn face_on() -> Self {
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

    fn down_the_line() -> Self {
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
    fn pelvis0(&self) -> (f64, f64) {
        match self.view {
            View::FaceOn => (640.0, 520.0),
            View::DownTheLine => (600.0, 520.0),
        }
    }

    fn pelvis(&self, t: f64) -> (f64, f64) {
        let (x, y) = self.pelvis0();
        (x + self.pelvis_dx.at(t), y + self.pelvis_dy.at(t))
    }

    fn shoulders(&self, t: f64) -> (f64, f64) {
        let (px, py) = self.pelvis(t);
        let s = self.spine.at(t).to_radians();
        (
            px + TORSO * s.sin() + self.chest_dx.at(t),
            py - TORSO * s.cos(),
        )
    }

    fn hands(&self, t: f64) -> (f64, f64) {
        let (sx, sy) = self.shoulders(t);
        let a = hand_angle(t, self.hand_swing);
        (sx + ARM * a.sin(), sy + ARM * a.cos())
    }

    /// A body line's lead-end offset from its midpoint (x, image y) with
    /// true half-length `half`, tilted by `tilt` and turned by `turn`
    /// (degrees): `half·cos(tilt)·cos(turn)` across, `half·sin(tilt)` up.
    fn line(half: f64, tilt: f64, turn: f64) -> (f64, f64) {
        let (tilt, turn) = (tilt.to_radians(), turn.to_radians());
        (half * tilt.cos() * turn.cos(), -half * tilt.sin())
    }

    /// The lead and trail hips.
    fn hips(&self, t: f64) -> ((f64, f64), (f64, f64)) {
        let (px, py) = self.pelvis(t);
        let (dx, dy) = Self::line(
            self.hip_half.at(t),
            self.hip_tilt.at(t),
            self.hip_turn.at(t),
        );
        ((px + dx, py + dy), (px - dx, py - dy))
    }

    /// The lead and trail shoulders.
    fn shoulder_points(&self, t: f64) -> ((f64, f64), (f64, f64)) {
        let (sx, sy) = self.shoulders(t);
        let (dx, dy) = Self::line(
            self.shoulder_half.at(t),
            self.shoulder_tilt.at(t),
            self.shoulder_turn.at(t),
        );
        ((sx + dx, sy + dy), (sx - dx, sy - dy))
    }

    /// The right-hander's 26 keypoints at `t`.
    fn keypoints(&self, t: f64) -> Vec<Keypoint> {
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

    fn track(&self) -> PoseTrack {
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

    fn pose(&self, t: f64) -> Pose {
        Pose::new(Skeleton::Halpe26, self.keypoints(t), BBox::default())
    }
}

/// The middle joint of a two-segment limb of equal segments from `a` to
/// `c` with the angle `deg` at it, bent to the `side` (±1) of `a → c`.
fn joint_between(start: (f64, f64), end: (f64, f64), deg: f64, side: f64) -> (f64, f64) {
    let mid = (f64::midpoint(start.0, end.0), f64::midpoint(start.1, end.1));
    let span = (end.0 - start.0).hypot(end.1 - start.1);
    let segment = span / (2.0 * (deg.to_radians() / 2.0).sin());
    let offset = (segment * segment - span * span / 4.0).max(0.0).sqrt();
    let (ux, uy) = ((end.0 - start.0) / span, (end.1 - start.1) / span);
    (mid.0 - side * uy * offset, mid.1 + side * ux * offset)
}

fn t_ns(t: f64) -> u64 {
    (t * 1e9).round() as u64
}

fn secs(e: Option<SwingEvent>) -> f64 {
    e.expect("event found").t_ns as f64 * 1e-9
}

/// The track as a left-hander would produce it: mirrored image, labels
/// swapped.
fn mirrored(track: &PoseTrack) -> PoseTrack {
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

fn options(view: View, handed: Handed, impact: Option<f64>) -> SwingOptions {
    SwingOptions {
        view: Some(view),
        handed,
        impact_ns: impact.map(t_ns),
        height_cm: None,
        mirrored: false,
        smoothing: Smoothing::default(),
    }
}

fn raw() -> Smoothing {
    Smoothing {
        cutoff_hz: 0.0,
        ..Smoothing::default()
    }
}

fn value(report: &SwingReport, id: MetricId) -> Positions {
    report
        .metric(id)
        .unwrap_or_else(|| panic!("{id:?} measured"))
        .values
}

fn near(a: Option<f64>, b: f64, tolerance: f64) -> bool {
    a.is_some_and(|a| (a - b).abs() <= tolerance)
}

#[test]
fn events_on_a_known_swing() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        let track = figure.track();
        let report =
            analyze_swing(&track, &options(figure.view, Handed::Right, Some(IMPACT_S))).unwrap();
        let e = report.events;
        assert_eq!(e.impact_source, ImpactSource::Refined);
        let impact = secs(e.impact);
        assert!(
            (impact - IMPACT_S - IMPACT_LAG_S).abs() <= 1.0 / FPS,
            "impact {impact}"
        );
        let address = secs(e.address);
        assert!((0.95..=1.08).contains(&address), "address {address}");
        assert_eq!(e.takeaway.unwrap().frame, e.address.unwrap().frame + 1);
        let top = secs(e.top);
        assert!((1.98..=2.1).contains(&top), "top {top}");
        let finish = secs(e.finish);
        assert!((2.75..=2.95).contains(&finish), "finish {finish}");
        for ev in [e.address, e.takeaway, e.top, e.impact, e.finish] {
            assert!(ev.unwrap().confidence > 0.8, "{ev:?}");
        }

        let estimated = analyze_swing(&track, &options(figure.view, Handed::Right, None)).unwrap();
        assert_eq!(estimated.events.impact_source, ImpactSource::Estimated);
        let impact = secs(estimated.events.impact);
        assert!(
            (impact - IMPACT_S - IMPACT_LAG_S).abs() <= 1.5 / FPS,
            "impact {impact}"
        );
        assert!(estimated.events.impact.unwrap().confidence > 0.8);
        assert!((secs(estimated.events.top) - top).abs() < 0.02);
    }
}

#[test]
fn tempo() {
    let report = analyze_swing(
        &Figure::face_on().track(),
        &options(View::FaceOn, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    let tempo = report.tempo.unwrap();
    let e = report.events;
    assert_eq!(
        tempo.backswing_ns,
        e.top.unwrap().t_ns - e.takeaway.unwrap().t_ns
    );
    assert_eq!(
        tempo.downswing_ns,
        e.impact.unwrap().t_ns - e.top.unwrap().t_ns
    );
    assert!((tempo.ratio - tempo.backswing_ns as f64 / tempo.downswing_ns as f64).abs() < 1e-12);
    // 1.0 s up and 0.4 s down (2.5:1): the takeaway's first millimeters,
    // the downswing's gentle start and the impact 25 ms after the hands'
    // low point shift it a little.
    assert!((2.2..=2.9).contains(&tempo.ratio), "ratio {}", tempo.ratio);

    let event = |t: f64| SwingEvent {
        t_ns: t_ns(t),
        frame: 0,
        confidence: 1.0,
    };
    let mut events = SwingEvents {
        address: None,
        takeaway: Some(event(1.0)),
        arm_parallel_back: None,
        top: Some(event(1.9)),
        arm_parallel_down: None,
        impact: Some(event(2.2)),
        impact_source: ImpactSource::Refined,
        finish: None,
    };
    let tempo = Tempo::from_events(&events).unwrap();
    assert_eq!(
        (tempo.backswing_ns, tempo.downswing_ns),
        (900_000_000, 300_000_000)
    );
    assert!((tempo.ratio - 3.0).abs() < 1e-12);
    events.top = Some(SwingEvent {
        confidence: 0.0,
        ..event(1.9)
    });
    assert!(Tempo::from_events(&events).is_none(), "implausible top");
    events.top = Some(SwingEvent {
        confidence: 0.05,
        ..event(1.9)
    });
    assert!(Tempo::from_events(&events).is_some(), "doubtful top");
    events.top = Some(event(2.3));
    assert!(Tempo::from_events(&events).is_none(), "top after impact");
}

#[test]
fn face_on_metrics() {
    let figure = Figure::face_on();
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    opts.height_cm = Some(180.0);
    let r = analyze_swing(&figure.track(), &opts).unwrap();
    assert_eq!(r.view, Some(View::FaceOn));
    let scale = r.scale.unwrap();
    assert!((scale.torso_px - TORSO).abs() < 0.01);
    // The torso is never longer than at address here.
    assert!(near(scale.upright_px, TORSO, 0.01), "{scale:?}");
    let cm = scale.cm_per_px.unwrap();
    assert!((cm - TORSO_PER_HEIGHT * 180.0 / TORSO).abs() < 1e-6);
    let tol = 1e-3;
    // Hip sway: 20 px away from the target at the top, 30 px toward it at
    // impact.
    let sway = r.metric(MetricId::HipSway).unwrap();
    assert!(near(sway.values.top, -20.0 / TORSO, tol), "{sway:?}");
    assert!(near(sway.values.impact, 30.0 / TORSO, tol), "{sway:?}");
    assert!(sway.values.address.is_none() && sway.values.finish.is_none());
    let sway_cm = sway.cm.unwrap();
    assert!(near(sway_cm.top, -20.0 * cm, 1e-3) && near(sway_cm.impact, 30.0 * cm, 1e-3));
    assert_eq!(sway.unit, Unit::Torso);
    assert_eq!(sway.view, Some(View::FaceOn));
    assert!(sway.confidence > 0.89);
    let head = value(&r, MetricId::HeadLateral);
    assert!(near(head.top, -10.0 / TORSO, tol) && near(head.impact, 6.0 / TORSO, tol));
    let head = value(&r, MetricId::HeadVertical);
    assert!(near(head.top, -4.0 / TORSO, tol) && near(head.impact, -12.0 / TORSO, tol));
    let tilt = value(&r, MetricId::LateralSpineTilt);
    assert!(
        near(tilt.address, 6.0, 0.05) && near(tilt.top, 2.0, 0.05) && near(tilt.impact, 12.0, 0.05)
    );
    // Seen side-on at the finish, the spine's lean away from the target is
    // the bend back.
    assert!(near(tilt.finish, FINISH_LEAN, 0.05), "{tilt:?}");
    let shoulders = value(&r, MetricId::ShoulderTilt);
    assert!(near(shoulders.address, 10.0, 0.05), "{shoulders:?}");
    // The figure holds 25° from 2.25 to 2.55 s, around the impact, and
    // −40° at the top, both with the shoulders turned.
    assert!(near(shoulders.impact, 25.0, 0.05));
    assert!(near(shoulders.top, -40.0, 0.05), "{shoulders:?}");
    assert!(shoulders.finish.is_none());
    let arm = value(&r, MetricId::LeadArmAngle);
    assert!(
        near(arm.address, 175.0, 0.05)
            && near(arm.top, 160.0, 0.05)
            && near(arm.impact, 178.0, 0.05)
    );
    assert!(arm.finish.is_none());
    assert!(near(
        value(&r, MetricId::StanceWidth).address,
        140.0 / 170.0,
        tol
    ));
    let hips = value(&r, MetricId::HipWidthProxy);
    assert!(near(hips.top, 0.8, tol) && near(hips.impact, 0.9, tol) && near(hips.finish, 0.5, tol));
    // Only face-on (and either-view) metrics.
    assert!(r.metrics.iter().all(|m| m.id.applies_to(View::FaceOn)));
    assert!(r.metric(MetricId::SpineForwardBend).is_none());
    let order: Vec<usize> = r
        .metrics
        .iter()
        .map(|m| MetricId::ALL.iter().position(|&id| id == m.id).unwrap())
        .collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "ALL order");

    // Default smoothing changes held values only slightly.
    let smoothed = analyze_swing(
        &figure.track(),
        &options(View::FaceOn, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    assert!(near(value(&smoothed, MetricId::HipSway).top, -0.1, 0.005));
    let tilt = value(&smoothed, MetricId::LateralSpineTilt).top;
    assert!(
        near(tilt, 2.0, 0.5),
        "smoothed spine tilt at the top {tilt:?}"
    );
}

#[test]
fn down_the_line_metrics() {
    let figure = Figure::down_the_line();
    let mut opts = options(View::DownTheLine, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let r = analyze_swing(&figure.track(), &opts).unwrap();
    assert!(r.scale.unwrap().cm_per_px.is_none());
    assert!(r.metrics.iter().all(|m| m.cm.is_none()));
    let bend = value(&r, MetricId::SpineForwardBend);
    assert!(
        near(bend.address, 35.0, 0.05)
            && near(bend.top, 33.0, 0.05)
            && near(bend.impact, 30.0, 0.05)
    );
    let loss = value(&r, MetricId::LossOfPosture);
    assert!(near(loss.top, 2.0, 0.05) && near(loss.impact, 5.0, 0.05));
    assert!(loss.address.is_none());
    let ee = value(&r, MetricId::EarlyExtension);
    assert!(near(ee.top, 4.0 / TORSO, 1e-3) && near(ee.impact, 16.0 / TORSO, 1e-3));
    let lead = value(&r, MetricId::LeadKneeFlex);
    assert!(near(lead.address, 20.0, 0.05) && near(lead.top, 35.0, 0.05));
    assert!(near(lead.impact, 10.0, 0.05) && near(lead.finish, 2.0, 0.05));
    let trail = value(&r, MetricId::TrailKneeFlex);
    assert!(near(trail.address, 20.0, 0.05) && near(trail.top, 18.0, 0.05));
    assert!(near(trail.impact, 30.0, 0.05) && near(trail.finish, 5.0, 0.05));
    let elbow = value(&r, MetricId::TrailElbowAngle);
    assert!(
        near(elbow.address, 165.0, 0.05)
            && near(elbow.top, 80.0, 0.05)
            && near(elbow.impact, 150.0, 0.05)
    );
    let top_t = r.events.top.unwrap().frame as f64 / FPS;
    // The shoulder line is level, so the trail shoulder is at the
    // shoulders' height.
    let expected = (figure.shoulders(top_t).1 - figure.hands(top_t).1) / TORSO;
    assert!(near(value(&r, MetricId::HandHeight).top, expected, 2e-3));
    let head = value(&r, MetricId::HeadVertical);
    assert!(near(head.impact, -8.0 / TORSO, 1e-3));
    assert!(r.metrics.iter().all(|m| m.id.applies_to(View::DownTheLine)));
    assert!(r.metric(MetricId::HipSway).is_none());
}

/// The value of every metric's series at `frame`.
fn series_at(report: &SwingReport, id: MetricId, frame: usize) -> Option<f64> {
    report
        .series
        .metric(id)
        .unwrap_or_else(|| panic!("{id:?} has a series"))
        .values[frame]
}

#[test]
fn every_metric_has_a_value_per_frame_matching_its_positions() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        let track = figure.track();
        for smoothing in [raw(), Smoothing::default()] {
            let mut opts = options(figure.view, Handed::Right, Some(IMPACT_S));
            opts.smoothing = smoothing;
            let r = analyze_swing(&track, &opts).unwrap();
            let s = &r.series;
            let times: Vec<u64> = track.frames.iter().map(|f| f.t_ns).collect();
            assert_eq!(s.t_ns, times, "one entry per track frame");
            // Every measured metric has a series, in the same order.
            let ids = |v: Vec<MetricId>| v;
            assert_eq!(
                ids(s.metrics.iter().map(|m| m.id).collect()),
                ids(r.metrics.iter().map(|m| m.id).collect())
            );
            let e = r.events;
            for m in &r.metrics {
                let series = s.metric(m.id).unwrap();
                assert_eq!(series.values.len(), times.len());
                assert_eq!(series.unit, m.unit);
                for (event, value) in [
                    (e.address, m.values.address),
                    (e.arm_parallel_back, m.values.arm_parallel_back),
                    (e.top, m.values.top),
                    (e.arm_parallel_down, m.values.arm_parallel_down),
                    (e.impact, m.values.impact),
                    (e.finish, m.values.finish),
                ] {
                    if let Some(v) = value {
                        let frame = event.unwrap().frame;
                        assert_eq!(series.values[frame], Some(v), "{:?} at {frame}", m.id);
                        assert_eq!(s.value_at(m.id, e_t(event)), Some(v));
                    }
                }
                // Centimeters follow the body scale as the positions' do.
                if let (Some(cm), Some(top)) = (m.cm, m.values.top) {
                    assert_eq!(r.scale.unwrap().to_cm(top), cm.top);
                }
            }
            // Both lead-arm-parallel positions are measured.
            assert!(
                r.metrics
                    .iter()
                    .any(|m| m.values.arm_parallel_back.is_some())
            );
            assert!(
                r.metrics
                    .iter()
                    .any(|m| m.values.arm_parallel_down.is_some())
            );
            // Shifts are from the address frame.
            let address = e.address.unwrap().frame;
            for id in [
                MetricId::HipSway,
                MetricId::ChestSway,
                MetricId::EarlyExtension,
                MetricId::HeadVertical,
                MetricId::PelvisLift,
            ] {
                if let Some(series) = s.metric(id) {
                    assert_eq!(series.values[address], Some(0.0), "{id:?}");
                }
            }
        }
    }

    // In the raw track, a frame's value is the figure's at that time.
    let figure = Figure::face_on();
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let r = analyze_swing(&figure.track(), &opts).unwrap();
    for frame in [150, 230, 300, 400] {
        let t = frame as f64 / FPS;
        let (px, _) = figure.pelvis(t);
        let sway = series_at(&r, MetricId::HipSway, frame);
        assert!(near(sway, (px - 640.0) / TORSO, 1e-4), "{frame}: {sway:?}");
        let arm = series_at(&r, MetricId::LeadArmAngle, frame);
        assert!(near(arm, figure.lead_elbow.at(t), 0.05), "{frame}: {arm:?}");
    }
}

/// The time of an event.
fn e_t(e: Option<SwingEvent>) -> u64 {
    e.unwrap().t_ns
}

#[test]
fn turns_from_one_face_on_camera() {
    let figure = Figure::face_on();
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let r = analyze_swing(&figure.track(), &opts).unwrap();
    let hips = r.metric(MetricId::HipTurn).unwrap();
    assert_eq!((hips.unit, hips.view), (Unit::Degrees, Some(View::FaceOn)));
    // The hips show 0.8 and 0.9 of their address width.
    assert!(
        near(hips.values.top, 0.8f64.acos().to_degrees(), 0.01),
        "{hips:?}"
    );
    assert!(near(hips.values.impact, 0.9f64.acos().to_degrees(), 0.01));
    assert!(hips.values.address.is_none() && hips.values.finish.is_none());
    let shoulders = value(&r, MetricId::ShoulderTurn);
    // 70° at the top with the shoulder line tilted 40°, 30° at impact with
    // it tilted 25°: the solve recovers both; the line's length alone
    // (`acos(hypot(dx, dy) / w₀)`) reads the top as about 46°.
    assert!(near(shoulders.top, 70.0, 0.01), "{shoulders:?}");
    assert!(near(shoulders.impact, 30.0, 0.01));
    assert!(shoulders.address.is_none() && shoulders.finish.is_none());
    let top_t = r.events.top.unwrap().frame as f64 / FPS;
    let (lead, trail) = figure.shoulder_points(top_t);
    let blind = ((lead.0 - trail.0).hypot(lead.1 - trail.1) / 170.0)
        .acos()
        .to_degrees();
    assert!((blind - 46.0).abs() < 0.5, "{blind}");
    // The hip width proxy is unchanged: the turn is its arc cosine.
    let proxy = value(&r, MetricId::HipWidthProxy);
    assert!(near(
        hips.values.top,
        proxy.top.unwrap().acos().to_degrees(),
        1e-2
    ));

    // Every frame: the known turn of the figure at that time.
    for (frame, &t_ns) in r.series.t_ns.iter().enumerate() {
        let t = t_ns as f64 * 1e-9;
        let hip = series_at(&r, MetricId::HipTurn, frame);
        assert!(near(hip, figure.hip_turn.at(t), 0.1), "{t}: {hip:?}");
        let sh = series_at(&r, MetricId::ShoulderTurn, frame);
        assert!(near(sh, figure.shoulder_turn.at(t), 0.1), "{t}: {sh:?}");
    }
    // The finish: 60° of shoulder turn, the hips at 0.5 (60°).
    let last = r.series.t_ns.len() - 1;
    assert!(near(
        series_at(&r, MetricId::ShoulderTurn, last),
        60.0,
        0.01
    ));
    assert!(near(series_at(&r, MetricId::HipTurn, last), 60.0, 0.01));

    // Default smoothing keeps the held turns.
    let smoothed = analyze_swing(
        &figure.track(),
        &options(View::FaceOn, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    assert!(near(
        value(&smoothed, MetricId::ShoulderTurn).top,
        70.0,
        0.5
    ));
    assert!(near(value(&smoothed, MetricId::HipTurn).top, 36.87, 0.5));

    // Wider than at address (a body that wasn't square at address, or a
    // drifting keypoint) has no turn; the frames that are narrower still do.
    let mut wide = figure;
    (wide.hip_half.impact, wide.hip_turn.impact) = (50.0, 0.0);
    (wide.shoulder_half.top, wide.shoulder_turn.top) = (90.0, 0.0);
    let r = analyze_swing(&wide.track(), &opts).unwrap();
    let hips = value(&r, MetricId::HipTurn);
    assert!(hips.impact.is_none() && hips.top.is_some());
    assert!(value(&r, MetricId::ShoulderTurn).top.is_none());
    let impact = r.events.impact.unwrap().frame;
    assert_eq!(series_at(&r, MetricId::HipTurn, impact), None);
    assert!(series_at(&r, MetricId::HipTurn, impact - 40).is_some());

    // Missing keypoints: no value at those frames.
    let mut track = figure.track();
    let hip = Joint::TrailHip.index(Handed::Right);
    for f in &mut track.frames[100..110] {
        f.keypoints[hip].score = 0.0;
    }
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = Smoothing {
        max_gap_ns: 0,
        ..raw()
    };
    let r = analyze_swing(&track, &opts).unwrap();
    assert!((100..110).all(|i| series_at(&r, MetricId::HipTurn, i).is_none()));
    assert!(series_at(&r, MetricId::ShoulderTurn, 105).is_some());

    // Down the line has no turn estimates.
    let r = analyze_swing(
        &Figure::down_the_line().track(),
        &options(View::DownTheLine, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    for id in [MetricId::HipTurn, MetricId::ShoulderTurn] {
        assert!(r.metric(id).is_none() && r.series.metric(id).is_none());
    }
}

#[test]
fn the_series_reads_back_and_stays_compact() {
    let report = analyze_swing(
        &Figure::face_on().track(),
        &options(View::FaceOn, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    let json = report.to_json().unwrap();
    assert!(
        json.contains("\"series\":{\"t_ns\":[0,8333333,"),
        "{}",
        &json[..200]
    );
    assert!(json.contains("{\"id\":\"hip_turn\",\"unit\":\"degrees\",\"values\":["));
    assert!(json.contains("{\"id\":\"hip_sway\",\"unit\":\"torso\",\"values\":["));
    let back = SwingReport::from_json(&json).unwrap();
    assert_eq!(back, report);
    // 420 frames, 14 face-on metrics at 4 decimals: under 130 bytes per
    // frame.
    let frames = report.series.t_ns.len();
    assert_eq!(frames, 420);
    assert!(json.len() < frames * 130, "{} bytes", json.len());

    // A report without a series (an earlier writer) reads with an empty
    // one, and writes none.
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("series");
    let old = SwingReport::from_json(&value.to_string()).unwrap();
    assert!(old.series.is_empty() && old.series.t_ns.is_empty());
    assert_eq!(old.metrics, report.metrics);
    assert!(!old.to_json().unwrap().contains("series"));
    assert_eq!(old.series.frame_at(5), None);

    // Series with ids this crate doesn't know are skipped; a known one of
    // the wrong length is an error.
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let n = value["series"]["metrics"].as_array().unwrap().len();
    for id in ["x_factor", "sway", "slide"] {
        value["series"]["metrics"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({ "id": id, "unit": "degrees", "values": [1.0] }));
    }
    let parsed = SwingReport::from_json(&value.to_string()).unwrap();
    assert_eq!(parsed.series.metrics.len(), n);
    assert_eq!(parsed, report);
    value["series"]["metrics"][0]["values"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(SwingReport::from_json(&value.to_string()).is_err());

    // No metrics, no series.
    let mut track = Figure::face_on().track();
    for f in &mut track.frames {
        for (i, k) in f.keypoints.iter_mut().enumerate() {
            if i != Joint::LeadWrist.index(Handed::Right)
                && i != Joint::TrailWrist.index(Handed::Right)
            {
                k.score = 0.0;
            }
        }
    }
    let r = analyze_swing(
        &track,
        &options(View::FaceOn, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    assert!(r.metrics.is_empty() && r.series.is_empty() && r.series.t_ns.is_empty());

    // The nearest frame (the second is at 8,333,333 ns).
    let s = &report.series;
    assert_eq!(s.frame_at(0), Some(0));
    assert_eq!(s.frame_at(4_166_666), Some(0));
    assert_eq!(s.frame_at(4_166_667), Some(1));
    let tie = Series {
        t_ns: vec![0, 10],
        metrics: Vec::new(),
    };
    assert_eq!(tie.frame_at(5), Some(0), "a tie goes to the earlier frame");
    assert_eq!(s.frame_at(u64::MAX), Some(frames - 1));
}

#[test]
fn left_handers_mirror_right_handers() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        let right = figure.track();
        let left = mirrored(&right);
        for impact in [Some(IMPACT_S), None] {
            let r = analyze_swing(&right, &options(figure.view, Handed::Right, impact)).unwrap();
            let l = analyze_swing(&left, &options(figure.view, Handed::Left, impact)).unwrap();
            assert_eq!(r.events, l.events);
            assert_eq!(l.handed, Handed::Left);
            assert_eq!(r.metrics.len(), l.metrics.len());
            assert!(!r.metrics.is_empty());
            for (a, b) in r.metrics.iter().zip(&l.metrics) {
                assert_eq!(a.id, b.id);
                for (x, y) in [
                    (a.values.address, b.values.address),
                    (a.values.arm_parallel_back, b.values.arm_parallel_back),
                    (a.values.top, b.values.top),
                    (a.values.arm_parallel_down, b.values.arm_parallel_down),
                    (a.values.impact, b.values.impact),
                    (a.values.finish, b.values.finish),
                ] {
                    assert_eq!(x.is_some(), y.is_some(), "{:?}", a.id);
                    if let (Some(x), Some(y)) = (x, y) {
                        assert!((x - y).abs() < 1e-3, "{:?}: {x} vs {y}", a.id);
                    }
                }
            }
            assert_eq!(r.series.t_ns, l.series.t_ns);
            assert_eq!(r.series.metrics.len(), l.series.metrics.len());
            for (a, b) in r.series.metrics.iter().zip(&l.series.metrics) {
                assert_eq!(a.id, b.id);
                for (x, y) in a.values.iter().zip(&b.values) {
                    assert_eq!(x.is_some(), y.is_some(), "{:?}", a.id);
                    if let (Some(x), Some(y)) = (x, y) {
                        assert!((x - y).abs() < 1e-3, "{:?}: {x} vs {y}", a.id);
                    }
                }
            }
            if figure.view == View::FaceOn {
                // Hip sway is signed by the target, not the image: away from
                // it at the top, toward it at impact, for either hand.
                for report in [&r, &l] {
                    let sway = value(report, MetricId::HipSway);
                    assert!(
                        sway.top.unwrap() < 0.0 && sway.impact.unwrap() > 0.0,
                        "{sway:?}"
                    );
                }
            }
            assert_eq!(l.handed_guess.unwrap().value, Handed::Left);
            assert_eq!(r.handed_guess.unwrap().value, Handed::Right);
            // The right-hander's mirror image analyzed as right-handed is
            // flagged.
            let wrong = analyze_swing(&left, &options(figure.view, Handed::Right, impact)).unwrap();
            assert_eq!(wrong.handed_guess.unwrap().value, Handed::Left);
        }
    }
}

#[test]
fn gaps_are_filled_or_left_empty_never_zero() {
    let figure = Figure::face_on();
    let clean = figure.track();
    let mut track = clean.clone();
    let wrist = Joint::LeadWrist.index(Handed::Right);
    let frame_at = |t: f64| (t * FPS).round() as usize;
    // A short gap in the backswing (3 frames, 25 ms), as zeros with a low
    // score.
    for i in frame_at(1.5)..frame_at(1.5) + 3 {
        track.frames[i].keypoints[wrist] = Keypoint {
            x: 0.0,
            y: 0.0,
            score: 0.05,
        };
    }
    // A long dropout after the finish: nobody found for 0.4 s.
    for i in frame_at(3.0)..frame_at(3.4) {
        track.frames[i] = PoseFrame::empty(track.frames[i].t_ns);
    }
    // A long gap of one keypoint (0.3 s) at address.
    let ankle = Joint::TrailAnkle.index(Handed::Right);
    for i in frame_at(0.3)..frame_at(0.6) {
        track.frames[i].keypoints[ankle].score = 0.1;
    }
    let smoothed = smooth_track(&track, &Smoothing::default(), Some(t_ns(IMPACT_S))).unwrap();
    let truth = smooth_track(&clean, &Smoothing::default(), Some(t_ns(IMPACT_S))).unwrap();
    for i in frame_at(1.5)..frame_at(1.5) + 3 {
        let (k, want) = (
            smoothed.frames[i].keypoints[wrist],
            truth.frames[i].keypoints[wrist],
        );
        assert!(
            (k.x - want.x).abs() < 1.5 && (k.y - want.y).abs() < 1.5,
            "{k:?} vs {want:?}"
        );
        assert!(
            (k.score - 0.45).abs() < 1e-6,
            "interpolated: half the neighbors' score"
        );
    }
    assert!(smoothed.frames[frame_at(3.2)].keypoints.is_empty());
    let gap = smoothed.frames[frame_at(0.45)].keypoints[ankle];
    assert_eq!(gap.score, 0.0, "a long gap stays empty");
    assert_eq!(smoothed.frames.len(), track.frames.len());
    assert_eq!(smoothed.frames[10].t_ns, track.frames[10].t_ns);

    let a = analyze_swing(
        &clean,
        &options(View::FaceOn, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    let b = analyze_swing(
        &track,
        &options(View::FaceOn, Handed::Right, Some(IMPACT_S)),
    )
    .unwrap();
    assert_eq!(
        a.events.address.unwrap().frame,
        b.events.address.unwrap().frame
    );
    assert_eq!(a.events.top.unwrap().frame, b.events.top.unwrap().frame);
    assert_eq!(
        a.events.finish.unwrap().frame,
        b.events.finish.unwrap().frame
    );
    assert!(near(
        value(&b, MetricId::HipSway).top,
        value(&a, MetricId::HipSway).top.unwrap(),
        1e-3
    ));

    // Too little to analyze.
    let mut empty = clean.clone();
    for f in &mut empty.frames[2..] {
        *f = PoseFrame::empty(f.t_ns);
    }
    assert!(analyze_swing(&empty, &SwingOptions::default()).is_err());
    let mut backwards = clean;
    backwards.frames.swap(5, 6);
    assert!(analyze_swing(&backwards, &SwingOptions::default()).is_err());
}

#[test]
fn view_is_guessed_from_the_address_pose() {
    let fo = Figure::face_on();
    let dtl = Figure::down_the_line();
    assert_eq!(guess_view(&fo.pose(0.5)).unwrap().value, View::FaceOn);
    assert_eq!(guess_view(&dtl.pose(0.5)).unwrap().value, View::DownTheLine);
    assert!(guess_view(&fo.pose(0.5)).unwrap().confidence > 0.3);
    let report = analyze_swing(&dtl.track(), &SwingOptions::default()).unwrap();
    assert_eq!(report.view, Some(View::DownTheLine));
    assert_eq!(report.view_guess.unwrap().value, View::DownTheLine);
    let report = analyze_swing(&fo.track(), &SwingOptions::default()).unwrap();
    assert_eq!(report.view, Some(View::FaceOn));
    // A given view wins over the guess.
    let forced = analyze_swing(
        &fo.track(),
        &options(View::DownTheLine, Handed::Right, None),
    )
    .unwrap();
    assert_eq!(forced.view, Some(View::DownTheLine));
    assert_eq!(forced.view_guess.unwrap().value, View::FaceOn);
    let mut nobody = fo.pose(0.5);
    nobody.keypoints.truncate(3);
    assert!(guess_view(&nobody).is_none());
}

#[test]
fn report_json_round_trip() {
    let mut opts = options(View::FaceOn, Handed::Left, Some(IMPACT_S));
    opts.height_cm = Some(165.0);
    let report = analyze_swing(&mirrored(&Figure::face_on().track()), &opts).unwrap();
    let json = report.to_json().unwrap();
    assert!(json.starts_with("{\"version\":1,"));
    assert!(json.contains("\"view\":\"face_on\",\"handed\":\"left\","));
    assert!(json.contains("\"impact_source\":\"refined\""));
    assert!(json.contains("\"smoothing\":{\"min_score\":0.3,"));
    assert!(!json.contains("mirrored"));
    assert!(json.contains("\"id\":\"lateral_spine_tilt\""));
    assert!(json.contains("\"unit\":\"degrees\""));
    assert_eq!(SwingReport::from_json(&json).unwrap(), report);

    // The lead-arm-parallel events and positions and the upright torso are
    // written...
    assert!(json.contains("\"arm_parallel_back\":{\"t_ns\":"));
    assert!(json.contains("\"arm_parallel_down\":{\"t_ns\":"));
    assert!(json.contains("\"upright_px\":"));
    // ...and a report without them or the metrics that need them (an
    // earlier writer) still reads.
    let new_ids = ["chest_sway", "sway_gap", "pelvis_lift", "hip_tilt"];
    let mut old: serde_json::Value = serde_json::from_str(&json).unwrap();
    old["scale"].as_object_mut().unwrap().remove("upright_px");
    old["metrics"]
        .as_array_mut()
        .unwrap()
        .retain(|m| !new_ids.contains(&m["id"].as_str().unwrap()));
    old["series"]["metrics"]
        .as_array_mut()
        .unwrap()
        .retain(|m| !new_ids.contains(&m["id"].as_str().unwrap()));
    for key in ["arm_parallel_back", "arm_parallel_down"] {
        old["events"].as_object_mut().unwrap().remove(key);
        for m in old["metrics"].as_array_mut().unwrap() {
            for positions in ["values", "cm"] {
                if let Some(p) = m.get_mut(positions) {
                    p.as_object_mut().unwrap().remove(key);
                }
            }
        }
    }
    let old_json = old.to_string();
    assert!(!old_json.contains("arm_parallel") && !old_json.contains("upright_px"));
    let old = SwingReport::from_json(&old_json).unwrap();
    assert!(old.scale.unwrap().upright_px.is_none());
    assert_eq!(
        old.scale.unwrap().cm_per_px,
        report.scale.unwrap().cm_per_px
    );
    assert!(old.events.arm_parallel_back.is_none() && old.events.arm_parallel_down.is_none());
    assert_eq!(
        (old.events.top, old.events.impact),
        (report.events.top, report.events.impact)
    );
    assert!(old.metric(MetricId::ChestSway).is_none());
    let tilt = old.metric(MetricId::ShoulderTilt).unwrap();
    assert!(tilt.values.arm_parallel_back.is_none() && tilt.values.top.is_some());
    assert_eq!(
        old.metric(MetricId::HipSway).unwrap().values.top,
        report.metric(MetricId::HipSway).unwrap().values.top
    );

    let dir = TempDir::new("swing-report");
    let avi = dir.path().join("shot-0007.avi");
    assert_eq!(
        SwingReport::path_for(&avi),
        dir.path().join("shot-0007.swing.json")
    );
    let path = report.write_beside(&avi).unwrap();
    assert_eq!(SwingReport::read(&path).unwrap(), report);
    assert!(!dir.path().join("shot-0007.swing.json.part").exists());

    assert!(
        dir.path().read_dir().unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".part")),
        "no temporary file left"
    );

    // A later format's version is reported before anything else is parsed.
    let newer = "{\"version\":99,\"anything\":[1,2]}";
    let err = SwingReport::from_json(newer).unwrap_err().to_string();
    assert!(err.contains("version 99"), "{err}");
    assert!(SwingReport::from_json("{").is_err());

    // Metrics with ids this crate doesn't know (e.g. from a later version)
    // are skipped; a malformed known one is an error.
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let metrics = value["metrics"].as_array_mut().unwrap();
    let n = metrics.len();
    metrics.push(serde_json::json!({
        "id": "x_factor", "view": "face_on", "unit": "radians",
        "values": { "top": 0.8 }, "confidence": 0.9
    }));
    for (id, position) in [("sway", "top"), ("slide", "impact")] {
        metrics.push(serde_json::json!({
            "id": id, "view": "face_on", "unit": "torso",
            "values": { position: 0.1 }, "confidence": 0.9
        }));
    }
    let parsed = SwingReport::from_json(&value.to_string()).unwrap();
    assert_eq!(parsed.metrics.len(), n);
    assert_eq!(parsed, report);
    value["metrics"][0]["unit"] = "furlongs".into();
    assert!(SwingReport::from_json(&value.to_string()).is_err());

    // Reading is bounded.
    let big = dir.path().join("big.swing.json");
    let mut padded = json.clone();
    padded.push_str(&" ".repeat(MAX_SWING_REPORT_BYTES as usize));
    std::fs::write(&big, padded).unwrap();
    assert!(SwingReport::read(&big).is_err());
}

#[test]
fn posture_on_constructed_poses() {
    let fo = Figure::face_on();
    let address = posture(&fo.pose(0.5), View::FaceOn, Handed::Right);
    assert!(near(address.spine_angle, 6.0, 1e-3), "{address:?}");
    assert!(near(address.shoulder_tilt, 10.0, 1e-3));
    assert!(near(address.lead_elbow_angle, 175.0, 1e-3));
    assert!(near(address.trail_elbow_angle, 170.0, 1e-3));
    assert!(near(address.lead_knee_flex, 15.0, 1e-3));
    assert!(near(address.torso_px, TORSO, 1e-3));
    assert_eq!(
        address.head_shift(&address).map(|s| (s.forward, s.up)),
        Some((0.0, 0.0))
    );
    let impact = posture(&fo.pose(IMPACT_S), View::FaceOn, Handed::Right);
    let shift = impact.head_shift(&address).unwrap();
    assert!((shift.forward - 6.0 / TORSO).abs() < 1e-4 && (shift.up + 12.0 / TORSO).abs() < 1e-4);
    // The mirror image reads the same for a left-hander.
    let mirror = Skeleton::Halpe26.mirror();
    let mut left = fo.pose(IMPACT_S);
    for (i, k) in fo.pose(IMPACT_S).keypoints.iter().enumerate() {
        left.keypoints[mirror[i]] = Keypoint {
            x: WIDTH as f32 - k.x,
            ..*k
        };
    }
    let mut left_address = fo.pose(0.5);
    for (i, k) in fo.pose(0.5).keypoints.iter().enumerate() {
        left_address.keypoints[mirror[i]] = Keypoint {
            x: WIDTH as f32 - k.x,
            ..*k
        };
    }
    let l = posture(&left, View::FaceOn, Handed::Left);
    let l_shift = l
        .head_shift(&posture(&left_address, View::FaceOn, Handed::Left))
        .unwrap();
    assert!((l_shift.forward - shift.forward).abs() < 1e-4);
    assert!(near(l.spine_angle, impact.spine_angle.unwrap(), 1e-3));
    assert!(near(l.shoulder_tilt, impact.shoulder_tilt.unwrap(), 1e-3));

    let dtl = posture(
        &Figure::down_the_line().pose(0.5),
        View::DownTheLine,
        Handed::Right,
    );
    assert!(near(dtl.spine_angle, 35.0, 1e-3));
    assert!(near(dtl.lead_knee_flex, 20.0, 1e-3) && near(dtl.trail_knee_flex, 20.0, 1e-3));
    assert!(near(dtl.trail_elbow_angle, 165.0, 1e-3));

    // Low scores and wrong sizes give nothing.
    let mut faint = fo.pose(0.5);
    for k in &mut faint.keypoints {
        k.score = 0.1;
    }
    let p = posture(&faint, View::FaceOn, Handed::Right);
    assert!(p.spine_angle.is_none() && p.head.is_none() && p.head_shift(&address).is_none());
    faint.keypoints.pop();
    assert!(
        posture(&faint, View::FaceOn, Handed::Right)
            .torso_px
            .is_none()
    );

    // The filter passes a steady posture and resets on gaps.
    let mut filter = PostureFilter::new(crate::pose::OneEuroParams::default());
    let mut out = address;
    for i in 0..30 {
        out = filter.filter(i * 8_333_333, &address);
    }
    assert_eq!(out.spine_angle, address.spine_angle);
    assert_eq!(out.head, address.head);
    let gap = filter.filter(31 * 8_333_333, &p);
    assert!(gap.spine_angle.is_none() && gap.head.is_none());
    let back = filter.filter(32 * 8_333_333, &impact);
    assert_eq!(
        back.spine_angle, impact.spine_angle,
        "restarted after the gap"
    );
}

/// A track from per-frame keypoints at [`FPS`], starting at 0.
fn track_of(poses: Vec<Vec<Keypoint>>) -> PoseTrack {
    let frames = poses
        .into_iter()
        .enumerate()
        .map(|(i, k)| {
            let pose = Pose::new(Skeleton::Halpe26, k, BBox::new(0.0, 0.0, 1280.0, 800.0));
            PoseFrame::with_pose(t_ns(i as f64 / FPS), &pose)
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

/// The figure's keypoints per frame from `from` to `to` seconds.
fn poses(figure: &Figure, from: f64, to: f64) -> Vec<Vec<Keypoint>> {
    let (a, b) = ((from * FPS).round() as usize, (to * FPS).round() as usize);
    (a..b).map(|i| figure.keypoints(i as f64 / FPS)).collect()
}

/// `secs` of frames easing from pose `a` to pose `b`.
fn transition(a: &[Keypoint], b: &[Keypoint], secs: f64) -> Vec<Vec<Keypoint>> {
    let n = (secs * FPS).round() as usize;
    (1..=n)
        .map(|i| {
            let u = ramp(i as f64 / (n + 1) as f64, 0.0, 1.0) as f32;
            a.iter()
                .zip(b)
                .map(|(p, q)| Keypoint {
                    x: p.x + u * (q.x - p.x),
                    y: p.y + u * (q.y - p.y),
                    score: p.score,
                })
                .collect()
        })
        .collect()
}

fn given(figure: &Figure, track: &PoseTrack, impact: f64) -> SwingReport {
    analyze_swing(track, &options(figure.view, Handed::Right, Some(impact))).unwrap()
}

#[test]
fn a_given_impact_is_refined_against_the_hands() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        let track = figure.track();
        let exact = given(&figure, &track, IMPACT_S);
        let (top, tempo) = (exact.events.top.unwrap(), exact.tempo.unwrap().ratio);
        // The launch monitor's time is off by a few frames, or by a lot.
        for off in [-4.0 / FPS, -2.0 / FPS, 2.0 / FPS, 4.0 / FPS, -0.25, 0.25] {
            let r = given(&figure, &track, IMPACT_S + off);
            let e = r.events;
            assert_eq!(e.impact_source, ImpactSource::Refined, "offset {off}");
            let (i, i0) = (e.impact.unwrap(), exact.events.impact.unwrap());
            assert!(
                i.t_ns.abs_diff(i0.t_ns) < t_ns(0.5 / FPS),
                "offset {off}: {i:?} vs {i0:?}"
            );
            assert_eq!(e.top.unwrap().frame, top.frame, "offset {off}");
            let t = r.tempo.unwrap().ratio;
            assert!(
                (t / tempo - 1.0).abs() < 0.1,
                "offset {off}: tempo {t} vs {tempo}"
            );
            assert!(e.top.unwrap().confidence > 0.8);
            assert_eq!(r.handed_guess.unwrap().value, Handed::Right);
        }
    }
    // Nowhere near a swing: the given time stands, unconfirmed.
    let figure = Figure::face_on();
    let r = given(&figure, &figure.track(), 0.5);
    assert_eq!(r.events.impact_source, ImpactSource::Given);
    assert_eq!(r.events.impact.unwrap().t_ns, t_ns(0.5));
    assert_eq!(r.events.impact.unwrap().confidence, 0.5);
}

#[test]
fn a_practice_swing_or_lowering_the_hands_is_not_impact() {
    let figure = Figure::face_on();
    let single = given(&figure, &figure.track(), IMPACT_S);
    let swing = poses(&figure, 0.0, DURATION_S);
    let (finish, address) = (swing[swing.len() - 1].clone(), swing[0].clone());

    // A practice swing, the hands lowered back to address over a second,
    // then the shot 4.5 s in.
    let mut both = swing.clone();
    both.extend(transition(&finish, &address, 1.0));
    both.extend(swing.clone());
    let both = track_of(both);
    let shot = DURATION_S + 1.0;
    let est = analyze_swing(&both, &options(View::FaceOn, Handed::Right, None)).unwrap();
    let impact = secs(est.events.impact);
    assert!(
        [IMPACT_S, shot + IMPACT_S]
            .iter()
            .any(|t| (impact - t - IMPACT_LAG_S).abs() <= 1.5 / FPS),
        "estimated impact {impact}"
    );
    let r = given(&figure, &both, shot + IMPACT_S);
    assert!((secs(r.events.top) - shot - secs(single.events.top)).abs() < 1.5 / FPS);
    let (t, t1) = (r.tempo.unwrap().ratio, single.tempo.unwrap().ratio);
    assert!((t / t1 - 1.0).abs() < 0.1, "tempo {t} vs {t1}");

    // The clip starts with the hands high (after the previous swing), and
    // they come down to address before the swing.
    let mut high = vec![finish.clone(); (0.3 * FPS) as usize];
    high.extend(transition(&finish, &address, 0.8));
    let lead = high.len() as f64 / FPS;
    high.extend(swing);
    let high = track_of(high);
    let est = analyze_swing(&high, &options(View::FaceOn, Handed::Right, None)).unwrap();
    let impact = secs(est.events.impact);
    assert!(
        (impact - lead - IMPACT_S - IMPACT_LAG_S).abs() <= 1.5 / FPS,
        "estimated impact {impact}"
    );
    assert!((secs(est.events.address) - lead - secs(single.events.address)).abs() < 2.0 / FPS);
    let t = est.tempo.unwrap().ratio;
    assert!((t / t1 - 1.0).abs() < 0.1, "tempo {t} vs {t1}");
}

#[test]
fn a_drifting_hold_at_the_top_ends_at_the_downswing() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        let clean = secs(given(&figure, &figure.track(), IMPACT_S).events.top);
        // 1.5 s at the top while the hands sag 0.2 torso lengths.
        let hold = 1.5;
        let sag = 0.2 * TORSO;
        let mut frames = poses(&figure, 0.0, TOP_S);
        let top_pose = figure.keypoints(TOP_S);
        let n = (hold * FPS) as usize;
        let lower = |k: &[Keypoint], dy: f64| -> Vec<Keypoint> {
            let mut k = k.to_vec();
            for j in [
                Joint::LeadWrist,
                Joint::TrailWrist,
                Joint::LeadElbow,
                Joint::TrailElbow,
            ] {
                k[j.index(Handed::Right)].y += dy as f32;
            }
            k
        };
        for i in 0..n {
            frames.push(lower(&top_pose, sag * i as f64 / n as f64));
        }
        for k in poses(&figure, TOP_S, DURATION_S) {
            frames.push(lower(&k, sag));
        }
        let r = given(&figure, &track_of(frames), IMPACT_S + hold);
        let top = secs(r.events.top);
        assert!(
            (top - clean - hold).abs() <= 3.0 / FPS,
            "{:?}: top {top}, {clean} without the hold",
            figure.view
        );
        let tempo = r.tempo.unwrap();
        assert!(tempo.downswing_ns < 450_000_000, "{tempo:?}");
        assert!(r.events.top.unwrap().confidence > 0.8);
    }
}

#[test]
fn a_downswing_out_of_range_is_not_trusted() {
    // A top held for 2 s with the hands sagging is a top; a "downswing"
    // much longer than any real one is not.
    let figure = Figure::face_on();
    let track = figure.track();
    let r = given(&figure, &track, IMPACT_S);
    assert!(r.events.top.unwrap().confidence > 0.8);
    // Stretch the downswing to 1.2 s by slowing time after the top.
    let mut slow = track.clone();
    let top = t_ns(TOP_S);
    for f in &mut slow.frames {
        if f.t_ns > top {
            f.t_ns = top + (f.t_ns - top) * 3;
        }
    }
    let slow_impact = TOP_S + (IMPACT_S - TOP_S) * 3.0;
    let r = given(&figure, &slow, slow_impact);
    assert!(r.events.top.unwrap().confidence < 0.1, "{:?}", r.events.top);
    // ...and a doubtful top doesn't drive the handedness check.
    assert!(r.handed_guess.is_none());
    // An impossible one gives no tempo.
    assert_eq!(r.events.top.unwrap().confidence, 0.0);
    assert!(r.tempo.is_none());
}

#[test]
fn a_clip_starting_at_address_keeps_address_and_tempo() {
    let figure = Figure::face_on();
    let full = given(&figure, &figure.track(), IMPACT_S);
    let start = TAKEAWAY_S - 0.05;
    let trimmed = track_of(poses(&figure, start, DURATION_S));
    let r = given(&figure, &trimmed, IMPACT_S - start);
    let (a, k) = (r.events.address.unwrap(), r.events.takeaway.unwrap());
    let want = secs(full.events.address) - start;
    assert!((secs(Some(a)) - want).abs() <= 1.0 / FPS, "address {a:?}");
    assert!(a.confidence < full.events.address.unwrap().confidence);
    assert!(a.confidence > 0.4);
    assert_eq!(k.frame, a.frame + 1);
    let (t, t1) = (r.tempo.unwrap().ratio, full.tempo.unwrap().ratio);
    assert!((t / t1 - 1.0).abs() < 0.1, "tempo {t} vs {t1}");
    let sway = value(&r, MetricId::HipSway);
    assert!(sway.top.is_some() && sway.impact.is_some());
}

#[test]
fn a_gap_over_the_top_lowers_its_confidence() {
    let figure = Figure::face_on();
    let clean = given(&figure, &figure.track(), IMPACT_S);
    let clean_top = secs(clean.events.top);
    let mut track = figure.track();
    for f in &mut track.frames {
        if (f.t_ns as f64 * 1e-9 - clean_top).abs() <= 0.05 {
            for j in [Joint::LeadWrist, Joint::TrailWrist] {
                f.keypoints[j.index(Handed::Right)].score = 0.05;
            }
        }
    }
    let r = given(&figure, &track, IMPACT_S);
    let top = r.events.top.unwrap();
    assert!((secs(Some(top)) - clean_top).abs() <= 0.07, "{top:?}");
    assert!(top.confidence <= 0.5, "{top:?}");
    assert!(clean.events.top.unwrap().confidence > 0.8);
    let t = r.tempo.unwrap().ratio;
    assert!(
        (t / clean.tempo.unwrap().ratio - 1.0).abs() < 0.25,
        "tempo {t}"
    );
}

#[test]
fn a_mirrored_image_reads_as_the_true_hand() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        let right = figure.track();
        // What a selfie-style camera records of the same right-hander.
        let selfie = mirrored(&right);
        let truth = given(&figure, &right, IMPACT_S);
        let mut opts = options(figure.view, Handed::Right, Some(IMPACT_S));
        opts.mirrored = true;
        let r = analyze_swing(&selfie, &opts).unwrap();
        assert!(r.mirrored);
        assert_eq!(r.handed, Handed::Right);
        assert_eq!(r.handed_guess.unwrap().value, Handed::Right);
        assert_eq!(r.events, truth.events);
        assert_eq!(r.metrics.len(), truth.metrics.len());
        for (a, b) in r.metrics.iter().zip(&truth.metrics) {
            assert_eq!(a.id, b.id);
            for (x, y) in [
                (a.values.address, b.values.address),
                (a.values.top, b.values.top),
                (a.values.impact, b.values.impact),
                (a.values.finish, b.values.finish),
            ] {
                assert_eq!(x.is_some(), y.is_some());
                if let (Some(x), Some(y)) = (x, y) {
                    assert!((x - y).abs() < 1e-3, "{:?}: {x} vs {y}", a.id);
                }
            }
        }
        assert!(r.to_json().unwrap().contains("\"mirrored\":true"));
        // Undeclared, the mirror image looks left-handed.
        let plain = given(&figure, &selfie, IMPACT_S);
        assert_eq!(plain.handed_guess.unwrap().value, Handed::Left);

        // Live: the unmirrored pose gives the true posture.
        let pose = figure.pose(IMPACT_S);
        let mut flipped = pose.clone();
        flipped.keypoints = selfie.frames[(IMPACT_S * FPS) as usize].keypoints.clone();
        let live = posture(
            &unmirror(&flipped, WIDTH as f32),
            figure.view,
            Handed::Right,
        );
        let want = posture(&pose, figure.view, Handed::Right);
        assert!(near(live.spine_angle, want.spine_angle.unwrap(), 1e-3));
        assert!(near(
            live.lead_knee_flex,
            want.lead_knee_flex.unwrap(),
            1e-3
        ));
    }
}

#[test]
fn an_ambiguous_view_is_unknown() {
    // Squeezing a face-on address pose sideways turns it into a
    // down-the-line one; in between, the guess is unknown.
    let pose = Figure::face_on().pose(0.5);
    let cx = 640.0;
    let mut seen = Vec::new();
    for step in 0..=100 {
        let k = step as f32 / 100.0;
        let mut squeezed = pose.clone();
        for p in &mut squeezed.keypoints {
            p.x = cx + k * (p.x - cx);
        }
        seen.push(guess_view(&squeezed).map(|g| g.value));
    }
    assert_eq!(seen[0], Some(View::DownTheLine));
    assert_eq!(seen[100], Some(View::FaceOn));
    assert!(seen.contains(&None));
    // Without a view, only the either-view metrics are measured.
    let mut track = Figure::face_on().track();
    let squeeze = (0..=100)
        .find(|&s| seen[s].is_none())
        .map(|s| s as f32 / 100.0)
        .unwrap();
    for f in &mut track.frames {
        for p in &mut f.keypoints {
            p.x = cx + squeeze * (p.x - cx);
        }
    }
    let r = analyze_swing(&track, &SwingOptions::default()).unwrap();
    if r.view_guess.is_none() {
        assert_eq!(r.view, None);
        assert!(r.setup().is_none());
        assert!(r.metrics.iter().all(|m| m.view.is_none()));
        assert!(r.handed_guess.is_none());
    }
}

#[test]
fn scores_above_one_are_clamped() {
    let figure = Figure::down_the_line();
    let mut track = figure.track();
    for f in &mut track.frames {
        for k in &mut f.keypoints {
            k.score = 1.3;
        }
    }
    let r = given(&figure, &track, IMPACT_S);
    assert!(r.metrics.iter().all(|m| m.confidence <= 1.0));
    let e = r.events;
    for ev in [e.address, e.takeaway, e.top, e.impact, e.finish] {
        assert!(ev.unwrap().confidence <= 1.0);
    }
}

#[test]
fn an_unconfirmed_given_impact_derives_no_swing() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        let track = figure.track();
        // At address, and in the held finish past where the hands still
        // fall back to the swing's low point: nothing to refine.
        for t in [0.5, FINISH_S + 0.4] {
            let r = given(&figure, &track, t);
            let e = r.events;
            assert_eq!(e.impact_source, ImpactSource::Given, "{t}");
            let impact = e.impact.unwrap();
            assert_eq!((impact.t_ns, impact.confidence), (t_ns(t), 0.5));
            assert!(
                e.address.is_none()
                    && e.takeaway.is_none()
                    && e.top.is_none()
                    && e.finish.is_none(),
                "{t}: {e:?}"
            );
            assert!(r.tempo.is_none(), "{t}");
            assert!(r.handed_guess.is_none(), "{t}");
            assert_eq!(SwingReport::from_json(&r.to_json().unwrap()).unwrap(), r);
        }
    }
}

#[test]
fn swing_options_json_defaults_missing_fields() {
    let opts: SwingOptions = serde_json::from_str("{\"handed\":\"left\"}").unwrap();
    assert_eq!(
        opts,
        SwingOptions {
            handed: Handed::Left,
            ..SwingOptions::default()
        }
    );
    let mut full = options(View::DownTheLine, Handed::Right, Some(IMPACT_S));
    full.mirrored = true;
    let json = serde_json::to_string(&full).unwrap();
    assert_eq!(serde_json::from_str::<SwingOptions>(&json).unwrap(), full);
}

#[test]
fn reading_a_report_checks_its_length_before_its_text() {
    let dir = TempDir::new("swing-report-read");
    let report = given(&Figure::face_on(), &Figure::face_on().track(), IMPACT_S);
    let json = report.to_json().unwrap();
    // One byte past the cap, with the cut inside a two-byte character.
    let mut long = json.into_bytes();
    long.resize(MAX_SWING_REPORT_BYTES as usize, b' ');
    long.extend_from_slice("é".as_bytes());
    let path = dir.path().join("long.swing.json");
    std::fs::write(&path, &long).unwrap();
    let err = SwingReport::read(&path).unwrap_err();
    assert!(matches!(err, crate::Error::InvalidArgument(_)), "{err}");
    assert!(err.to_string().contains("longer than"), "{err}");
    // Not UTF-8 within the cap.
    let path = dir.path().join("latin1.swing.json");
    std::fs::write(&path, b"{\"version\":1,\"x\":\"\xe9\"}").unwrap();
    let err = SwingReport::read(&path).unwrap_err();
    assert!(matches!(err, crate::Error::InvalidArgument(_)), "{err}");
}

#[test]
fn the_line_solve_recovers_tilt_and_turn() {
    use super::metrics::solve_line;
    let w0 = 170.0;
    for tilt in [-40.0_f64, -8.0, 0.0, 10.0, 25.0] {
        for turn in [0.0_f64, 30.0, 70.0, 89.0] {
            let dx = w0 * tilt.to_radians().cos() * turn.to_radians().cos();
            let dy = w0 * tilt.to_radians().sin();
            for dx in [dx, -dx] {
                let (t, r) = solve_line(dx, dy, w0).unwrap();
                let r = r.unwrap();
                assert!(
                    (t - tilt).abs() < 1e-9 && (r - turn).abs() < 1e-5,
                    "{tilt} {turn}: {t} {r}"
                );
            }
        }
    }
    // The line at address solves to its own angle and no turn.
    let (dx0, dy0) = (150.0_f64, 26.0_f64);
    let (t, r) = solve_line(dx0, dy0, dx0.hypot(dy0)).unwrap();
    assert!((t - dy0.atan2(dx0).to_degrees()).abs() < 1e-9 && r.unwrap() < 1e-5);
    // Seen end-on: 90°.
    assert_eq!(solve_line(0.0, 0.0, 10.0), Some((0.0, Some(90.0))));
    // Longer than at address (|dy| > w0, or dx > w0 cos φ): no turn, and the
    // tilt is the line's own angle, continuous with the solve at the
    // boundary.
    let angle = |dx: f64, dy: f64| dy.atan2(dx).to_degrees();
    assert_eq!(solve_line(0.0, 11.0, 10.0), Some((90.0, None)));
    assert_eq!(
        solve_line(-3.0, -11.0, 10.0),
        Some((angle(3.0, -11.0), None))
    );
    assert_eq!(solve_line(9.0, 5.0, 10.0), Some((angle(9.0, 5.0), None)));
    assert_eq!(solve_line(10.5, 0.0, 10.0), Some((0.0, None)));
    let edge = 75f64.sqrt();
    let (inside, turn) = solve_line(edge - 1e-6, 5.0, 10.0).unwrap();
    let (outside, none) = solve_line(edge + 1e-6, 5.0, 10.0).unwrap();
    assert!((inside - 30.0).abs() < 1e-4 && (outside - 30.0).abs() < 1e-4);
    assert!(turn.unwrap() < 0.1 && none.is_none());
    // Vertical: a tilt, no turn.
    assert_eq!(solve_line(0.0, 10.0, 10.0), Some((90.0, None)));
    // Undefined: no value, never NaN.
    assert_eq!(solve_line(10.0, 0.0, 0.0), None, "no length at address");
    assert_eq!(solve_line(10.0, 0.0, f64::NAN), None);
    assert_eq!(solve_line(f64::NAN, 0.0, 10.0), None);
    assert_eq!(solve_line(0.0, f64::INFINITY, 10.0), None);
    assert_eq!(solve_line(1.0, 0.0, f64::INFINITY), None);
}

#[test]
fn hip_and_shoulder_tilt_from_the_line_solve() {
    let mut figure = Figure::face_on();
    figure.hip_tilt = Phases {
        address: 3.0,
        top: -8.0,
        impact: 10.0,
        finish: 0.0,
    };
    let right = figure.track();
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let r = analyze_swing(&right, &opts).unwrap();
    opts.handed = Handed::Left;
    let l = analyze_swing(&mirrored(&right), &opts).unwrap();
    for report in [&r, &l] {
        let hips = report.metric(MetricId::HipTilt).unwrap();
        assert_eq!((hips.unit, hips.view), (Unit::Degrees, Some(View::FaceOn)));
        let hips = hips.values;
        assert!(
            near(hips.address, 3.0, 1e-3)
                && near(hips.top, -8.0, 1e-3)
                && near(hips.impact, 10.0, 1e-3),
            "{hips:?}"
        );
        // The lead hip lower at the top, higher at impact, for either hand.
        assert!(hips.top.unwrap() < 0.0 && hips.impact.unwrap() > 0.0);
        assert!(hips.arm_parallel_back.is_none() && hips.finish.is_none());
        let shoulders = value(report, MetricId::ShoulderTilt);
        assert!(
            near(shoulders.address, 10.0, 1e-3)
                && near(shoulders.top, -40.0, 1e-3)
                && near(shoulders.impact, 25.0, 1e-3),
            "{shoulders:?}"
        );
        assert!(shoulders.top.unwrap() < 0.0);
        // The turns are unchanged by the tilt.
        let turn = value(report, MetricId::HipTurn);
        assert!(near(turn.top, 0.8f64.acos().to_degrees(), 0.01), "{turn:?}");
        assert!(near(turn.impact, 0.9f64.acos().to_degrees(), 0.01));
        // Every frame: the figure's tilts.
        for (frame, &t_ns) in report.series.t_ns.iter().enumerate() {
            let t = t_ns as f64 * 1e-9;
            let hip = series_at(report, MetricId::HipTilt, frame);
            assert!(near(hip, figure.hip_tilt.at(t), 1e-3), "{t}: {hip:?}");
            let sh = series_at(report, MetricId::ShoulderTilt, frame);
            assert!(near(sh, figure.shoulder_tilt.at(t), 1e-3), "{t}: {sh:?}");
        }
    }
    // At address the tilt is the shoulder line's angle from horizontal (as
    // the live posture reads it); with the shoulders turned at impact, that
    // angle overstates the tilt.
    let at = |e: Option<SwingEvent>| e.unwrap().frame as f64 / FPS;
    let plain = |t: f64| {
        posture(&figure.pose(t), View::FaceOn, Handed::Right)
            .shoulder_tilt
            .unwrap()
    };
    let tilt = value(&r, MetricId::ShoulderTilt);
    assert!(near(tilt.address, plain(at(r.events.address)), 1e-3));
    let overstated = plain(at(r.events.impact));
    let expected = (25f64.to_radians().tan() / 30f64.to_radians().cos())
        .atan()
        .to_degrees();
    assert!((overstated - expected).abs() < 1e-3, "{overstated}");
    assert!(overstated > tilt.impact.unwrap() + 3.0);

    // Square and a little longer than at address at impact (keypoint noise,
    // a step toward the camera): no turn, and the tilt is the line's angle.
    let mut long = figure;
    (long.shoulder_half.impact, long.shoulder_turn.impact) = (88.0, 0.0);
    opts.handed = Handed::Right;
    let r = analyze_swing(&long.track(), &opts).unwrap();
    assert!(near(value(&r, MetricId::ShoulderTilt).impact, 25.0, 1e-3));
    assert!(value(&r, MetricId::ShoulderTurn).impact.is_none());
}

#[test]
fn chest_sway_sway_gap_and_pelvis_lift_face_on() {
    let mut figure = Figure::face_on();
    figure.chest_dx = Phases {
        address: 0.0,
        top: -30.0,
        impact: -10.0,
        finish: 0.0,
    };
    figure.pelvis_dy = Phases {
        address: 0.0,
        top: 8.0,
        impact: -6.0,
        finish: 0.0,
    };
    let right = figure.track();
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let r = analyze_swing(&right, &opts).unwrap();
    opts.handed = Handed::Left;
    let l = analyze_swing(&mirrored(&right), &opts).unwrap();
    let e = r.events;
    assert_eq!(l.events, e);
    let at = |ev: Option<SwingEvent>| ev.unwrap().frame as f64 / FPS;
    let t0 = at(e.address);
    let chest = |t: f64| (figure.shoulders(t).0 - figure.shoulders(t0).0) / TORSO;
    let gap = |t: f64| (figure.shoulders(t).0 - figure.pelvis(t).0) / TORSO;
    for report in [&r, &l] {
        let sway = report.metric(MetricId::ChestSway).unwrap();
        assert_eq!((sway.unit, sway.view), (Unit::Torso, Some(View::FaceOn)));
        let sway = sway.values;
        assert!(sway.address.is_none() && sway.finish.is_none(), "{sway:?}");
        for (v, ev) in [
            (sway.arm_parallel_back, e.arm_parallel_back),
            (sway.top, e.top),
            (sway.arm_parallel_down, e.arm_parallel_down),
            (sway.impact, e.impact),
        ] {
            assert!(near(v, chest(at(ev)), 1e-3), "{v:?} vs {}", chest(at(ev)));
        }
        // The chest moves away from the target in the backswing, further
        // than the pelvis.
        let hips = value(report, MetricId::HipSway);
        assert!(sway.top.unwrap() < hips.top.unwrap() && hips.top.unwrap() < 0.0);
        let gaps = report.metric(MetricId::SwayGap).unwrap();
        assert_eq!((gaps.unit, gaps.view), (Unit::Torso, Some(View::FaceOn)));
        let gaps = gaps.values;
        assert!(gaps.finish.is_none());
        for (v, ev) in [
            (gaps.address, e.address),
            (gaps.arm_parallel_back, e.arm_parallel_back),
            (gaps.top, e.top),
            (gaps.arm_parallel_down, e.arm_parallel_down),
            (gaps.impact, e.impact),
        ] {
            assert!(near(v, gap(at(ev)), 1e-3), "{v:?} vs {}", gap(at(ev)));
        }
        // The chest behind the pelvis (away from the target) at the top and
        // at impact.
        assert!(
            gaps.top.unwrap() < 0.0 && gaps.impact.unwrap() < 0.0,
            "{gaps:?}"
        );
        let lift = report.metric(MetricId::PelvisLift).unwrap();
        assert_eq!((lift.unit, lift.view), (Unit::Torso, None));
        let lift = lift.values;
        // Down 8 px at the top, up 6 px at impact.
        assert!(near(lift.top, -8.0 / TORSO, 1e-3) && near(lift.impact, 6.0 / TORSO, 1e-3));
        assert!(
            lift.address.is_none() && lift.arm_parallel_back.is_none() && lift.finish.is_none()
        );
    }
    // The series is the formula at every frame; the pelvis lift is from
    // address.
    let address = e.address.unwrap().frame;
    for frame in [address, 150, 230, 290, 400] {
        let t = frame as f64 / FPS;
        assert!(near(series_at(&r, MetricId::SwayGap, frame), gap(t), 1e-3));
        assert!(near(
            series_at(&r, MetricId::ChestSway, frame),
            chest(t),
            1e-3
        ));
        let rise = -figure.pelvis_dy.at(t) / TORSO;
        assert!(near(series_at(&r, MetricId::PelvisLift, frame), rise, 1e-3));
    }
    // Not relative: the gap has a value before the address is known.
    assert!(near(series_at(&r, MetricId::SwayGap, 0), gap(0.0), 1e-3));
}

#[test]
fn hand_depth_and_pelvis_lift_down_the_line() {
    let mut figure = Figure::down_the_line();
    // Bent back at the finish: down the line sees the golfer's back then,
    // so the bend is in depth and not measured.
    figure.spine.finish = -20.0;
    figure.pelvis_dy = Phases {
        address: 0.0,
        top: 6.0,
        impact: -4.0,
        finish: 0.0,
    };
    let right = figure.track();
    let mut opts = options(View::DownTheLine, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let r = analyze_swing(&right, &opts).unwrap();
    opts.handed = Handed::Left;
    let l = analyze_swing(&mirrored(&right), &opts).unwrap();
    let e = r.events;
    assert_eq!(l.events, e);
    let at = |ev: Option<SwingEvent>| ev.unwrap().frame as f64 / FPS;
    let top_t = at(e.top);
    let (_, trail) = figure.shoulder_points(top_t);
    let depth = -(figure.hands(top_t).0 - trail.0) / TORSO;
    // The figure's hands are behind the trail shoulder at the top.
    assert!(depth > 0.3, "{depth}");
    for report in [&r, &l] {
        let hand = report.metric(MetricId::HandDepth).unwrap();
        assert_eq!(
            (hand.unit, hand.view),
            (Unit::Torso, Some(View::DownTheLine))
        );
        assert!(near(hand.values.top, depth, 1e-3), "{hand:?}");
        assert!(hand.values.address.is_none() && hand.values.impact.is_none());
        let lift = value(report, MetricId::PelvisLift);
        assert!(near(lift.top, -6.0 / TORSO, 1e-3) && near(lift.impact, 4.0 / TORSO, 1e-3));
        let bend = value(report, MetricId::SpineForwardBend);
        assert!(near(bend.address, 35.0, 0.05), "{bend:?}");
        assert!(e.finish.is_some() && bend.finish.is_none(), "{bend:?}");
        for (v, ev) in [
            (bend.arm_parallel_back, e.arm_parallel_back),
            (bend.arm_parallel_down, e.arm_parallel_down),
        ] {
            assert!(near(v, figure.spine.at(at(ev)), 0.05), "{v:?}");
        }
        for id in [MetricId::LossOfPosture, MetricId::EarlyExtension] {
            let v = value(report, id);
            assert!(v.arm_parallel_back.is_some() && v.arm_parallel_down.is_some());
        }
    }
}

/// The first time in `(a, b)` where `f` changes sign (to 1 µs).
fn crossing(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let step = 1e-3;
    let mut t = a;
    while t + step <= b {
        if (f(t) < 0.0) != (f(t + step) < 0.0) {
            let (mut lo, mut hi) = (t, t + step);
            while hi - lo > 1e-6 {
                let mid = f64::midpoint(lo, hi);
                if (f(lo) < 0.0) == (f(mid) < 0.0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return f64::midpoint(lo, hi);
        }
        t += step;
    }
    panic!("no crossing in ({a}, {b})");
}

#[test]
fn lead_arm_parallel_events() {
    for figure in [Figure::face_on(), Figure::down_the_line()] {
        // The lead wrist's height above the lead shoulder.
        let height = |t: f64| figure.shoulder_points(t).0.1 - figure.hands(t).1;
        let t3 = crossing(height, TAKEAWAY_S, TOP_S);
        let t5 = crossing(height, TOP_S, IMPACT_S);
        if figure.view == View::DownTheLine {
            // A level shoulder line: the arm is level at 90° of hand swing,
            // where 150·(1 − cos πu)/2 = 90 in the backswing and
            // 150·(1 − u²) = 90 in the downswing.
            let u3 = (-0.2f64).acos() / PI;
            assert!((t3 - (TAKEAWAY_S + u3 * (TOP_S - TAKEAWAY_S))).abs() < 1e-5);
            let u5 = 0.4f64.sqrt();
            assert!((t5 - (TOP_S + u5 * (IMPACT_S - TOP_S))).abs() < 1e-5);
        }
        let right = figure.track();
        for smoothing in [raw(), Smoothing::default()] {
            for (track, handed) in [
                (right.clone(), Handed::Right),
                (mirrored(&right), Handed::Left),
            ] {
                let mut opts = options(figure.view, handed, Some(IMPACT_S));
                opts.smoothing = smoothing;
                let e = analyze_swing(&track, &opts).unwrap().events;
                let (p3, p5) = (e.arm_parallel_back.unwrap(), e.arm_parallel_down.unwrap());
                for (event, expected) in [(p3, t3), (p5, t5)] {
                    let t = event.t_ns as f64 * 1e-9;
                    assert!(
                        (t - expected).abs() <= 1.0 / FPS,
                        "{:?} {handed:?}: {t} vs {expected}",
                        figure.view
                    );
                    assert!((event.frame as f64 / FPS - t).abs() <= 0.5 / FPS + 1e-9);
                    assert!(event.confidence > 0.8, "{event:?}");
                }
                let order = [
                    e.takeaway,
                    e.arm_parallel_back,
                    e.top,
                    e.arm_parallel_down,
                    e.impact,
                ]
                .map(|ev| ev.unwrap().t_ns);
                assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
            }
        }
    }
    // Hands that never reach shoulder height (60° each way, a chip): no P3
    // or P5, though the top and impact are found.
    for base in [Figure::face_on(), Figure::down_the_line()] {
        let mut figure = base;
        figure.hand_swing = 60.0;
        let r = analyze_swing(
            &figure.track(),
            &options(figure.view, Handed::Right, Some(IMPACT_S)),
        )
        .unwrap();
        let e = r.events;
        assert!(e.top.is_some() && e.impact.is_some(), "{e:?}");
        assert!(e.arm_parallel_back.is_none() && e.arm_parallel_down.is_none());
        assert!(
            r.metrics
                .iter()
                .all(|m| m.values.arm_parallel_back.is_none()
                    && m.values.arm_parallel_down.is_none())
        );
    }
}

#[test]
fn p3_is_the_last_rise_before_the_top() {
    let figure = Figure::face_on();
    let height = |t: f64| figure.shoulder_points(t).0.1 - figure.hands(t).1;
    let (t3, t5) = (
        crossing(height, TAKEAWAY_S, TOP_S),
        crossing(height, TOP_S, IMPACT_S),
    );
    let mut track = figure.track();
    // One frame early in the backswing where the pose model puts the lead
    // wrist above the lead shoulder.
    let glitch = (1.25 * FPS) as usize;
    let lead_wrist = Joint::LeadWrist.index(Handed::Right);
    let lead_shoulder = Joint::LeadShoulder.index(Handed::Right);
    let k = &mut track.frames[glitch].keypoints;
    k[lead_wrist].y = k[lead_shoulder].y - 60.0;
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let e = analyze_swing(&track, &opts).unwrap().events;
    assert!(secs(e.takeaway) < 1.25, "{e:?}");
    assert!((secs(e.arm_parallel_back) - t3).abs() <= 1.0 / FPS, "{e:?}");
    assert!((secs(e.arm_parallel_down) - t5).abs() <= 1.0 / FPS, "{e:?}");
}

#[test]
fn hands_hovering_at_shoulder_height_keep_p3_and_p5_off_the_top() {
    // Down the line the shoulders are level, so at 90° of hand swing the
    // lead wrist tops out at the shoulder's height; a little more and it
    // crosses within a frame or two of the top (at 90.4°, the first fall
    // through it would be on the top frame).
    let mut off_the_top = 0;
    for swing in [90.0, 90.2, 90.4, 90.46, 91.0] {
        for smoothing in [raw(), Smoothing::default()] {
            let mut figure = Figure::down_the_line();
            figure.hand_swing = swing;
            let mut opts = options(View::DownTheLine, Handed::Right, Some(IMPACT_S));
            opts.smoothing = smoothing;
            let e = analyze_swing(&figure.track(), &opts).unwrap().events;
            let top = e.top.expect("top found").frame;
            for (event, after) in [(e.arm_parallel_back, false), (e.arm_parallel_down, true)] {
                let Some(event) = event else { continue };
                if after {
                    assert!(event.frame > top, "{swing}: P5 {event:?}, top {top}");
                } else {
                    assert!(event.frame < top, "{swing}: P3 {event:?}, top {top}");
                }
                // A wrist drifting along the shoulder's height crosses it
                // slowly, which shows in the confidence.
                if event.frame.abs_diff(top) <= 3 {
                    off_the_top += 1;
                    assert!(event.confidence < 0.5, "{swing}: {event:?}");
                }
            }
        }
    }
    // Some crossings came within three frames of the top.
    assert!(off_the_top > 0);
}

/// Meters per figure pixel for the pinhole figure: a 0.5 m torso.
const PINHOLE_M_PER_PX: f64 = 0.5 / TORSO;

/// The face-on figure as a pinhole camera with a 55° horizontal field of
/// view across [`WIDTH`] sees it from `distance` meters, centered on the
/// image: each turned body line's ends move in depth by
/// `half·cos(tilt)·sin(turn)` (the lead end toward the camera), every
/// other keypoint stays in the body's plane.
fn pinhole_track(figure: &Figure, distance: f64) -> PoseTrack {
    let focal = f64::from(WIDTH) / 2.0 / 27.5f64.to_radians().tan();
    let (cx, cy) = (f64::from(WIDTH) / 2.0, 400.0);
    let mut track = figure.track();
    for frame in &mut track.frames {
        let t = frame.t_ns as f64 * 1e-9;
        let depth = |half: &Phases, tilt: &Phases, turn: &Phases| {
            half.at(t) * tilt.at(t).to_radians().cos() * turn.at(t).to_radians().sin()
        };
        let hip = depth(&figure.hip_half, &figure.hip_tilt, &figure.hip_turn);
        let shoulder = depth(
            &figure.shoulder_half,
            &figure.shoulder_tilt,
            &figure.shoulder_turn,
        );
        for (i, k) in frame.keypoints.iter_mut().enumerate() {
            let dz = [
                (Joint::LeadHip, -hip),
                (Joint::TrailHip, hip),
                (Joint::LeadShoulder, -shoulder),
                (Joint::TrailShoulder, shoulder),
            ]
            .iter()
            .find(|(j, _)| j.index(Handed::Right) == i)
            .map_or(0.0, |(_, d)| *d);
            let z = distance + dz * PINHOLE_M_PER_PX;
            let x = (f64::from(k.x) - cx) * PINHOLE_M_PER_PX;
            let y = (f64::from(k.y) - cy) * PINHOLE_M_PER_PX;
            k.x = (cx + focal * x / z) as f32;
            k.y = (cy + focal * y / z) as f32;
        }
    }
    track
}

/// The face-on figure with a 0.35 m hip line turned 45° at the top.
fn pinhole_figure() -> Figure {
    let mut figure = Figure::face_on();
    figure.hip_half = Phases::flat(0.175 / PINHOLE_M_PER_PX);
    figure.hip_turn.top = 45.0;
    figure
}

/// The line solve against a pinhole camera (55° horizontal field of view,
/// at chest height, the golfer centered at address; 0.5 m torso, 0.35 m hip
/// line turned 45° at the top). Measured: the hip turn at the top reads
/// 45.5° at 5 m, 45.7° at 3 m and 45.9° at 2 m; the hip sway at the top is
/// off by 0.3 cm at 5 m and 0.75 cm at 2 m; the shoulder tilt at the top
/// (−40°) reads −37.9° at 5 m and −35.0° at 2 m. The impact hip turn
/// (25.8°, the pelvis 7.5 cm toward the target) reads 23.1° at 2 m: a body
/// off the camera's axis by `x` at distance `D` is seen turned by about
/// `x / D` radians.
#[test]
fn the_line_solve_through_a_pinhole_camera() {
    let figure = pinhole_figure();
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    for (distance, turn_error) in [(5.0, 1.0), (2.0, 2.0)] {
        let r = analyze_swing(&pinhole_track(&figure, distance), &opts).unwrap();
        let top = r.events.top.unwrap().frame as f64 / FPS;
        assert_eq!(figure.hip_turn.at(top), 45.0);
        let turn = value(&r, MetricId::HipTurn).top.unwrap();
        assert!((turn - 45.0).abs() < turn_error, "{distance} m: {turn}");
        // Torso units to centimeters with the 0.5 m torso.
        let sway = value(&r, MetricId::HipSway).top.unwrap() * 50.0;
        let bias = sway - figure.pelvis_dx.at(top) / TORSO * 50.0;
        assert!(bias.abs() < 1.0, "{distance} m: hip sway off by {bias} cm");
        let tilt = value(&r, MetricId::ShoulderTilt).top.unwrap();
        assert!((-40.0..-34.0).contains(&tilt), "{distance} m: {tilt}");
    }
}

#[test]
fn a_portrait_camera_reads_the_same_through_rotation() {
    // The pinhole figure at 2 m in an upright 800 × 1280 portrait frame.
    let mut upright = pinhole_track(&pinhole_figure(), 2.0);
    upright.width = 800;
    upright.height = 1280;
    for f in &mut upright.frames {
        for k in &mut f.keypoints {
            k.x -= 240.0;
            k.y += 240.0;
        }
    }
    // A camera on its side records the picture turned counterclockwise: a
    // landscape frame. The pose pass's `Rotation::Cw90` turns it back.
    let turn = |track: &PoseTrack, r: Rotation| {
        let mut out = track.clone();
        (out.width, out.height) = r.dimensions(track.width, track.height);
        out.rotation = r;
        for f in &mut out.frames {
            for k in &mut f.keypoints {
                let (x, y) = r.point(f64::from(k.x), f64::from(k.y), track.width, track.height);
                (k.x, k.y) = (x as f32, y as f32);
            }
        }
        out
    };
    let recorded = turn(&upright, Rotation::Ccw90);
    assert_eq!((recorded.width, recorded.height), (1280, 800));
    let analyzed = turn(&recorded, Rotation::Cw90);
    assert_eq!((analyzed.width, analyzed.height), (800, 1280));
    let mut opts = options(View::FaceOn, Handed::Right, Some(IMPACT_S));
    opts.smoothing = raw();
    let a = analyze_swing(&upright, &opts).unwrap();
    let b = analyze_swing(&analyzed, &opts).unwrap();
    assert_eq!(a.events, b.events);
    assert_eq!(a.metrics.len(), b.metrics.len());
    for (m, n) in a.metrics.iter().zip(&b.metrics) {
        assert_eq!(m.id, n.id);
        for (x, y) in [
            (m.values.address, n.values.address),
            (m.values.arm_parallel_back, n.values.arm_parallel_back),
            (m.values.top, n.values.top),
            (m.values.arm_parallel_down, n.values.arm_parallel_down),
            (m.values.impact, n.values.impact),
            (m.values.finish, n.values.finish),
        ] {
            assert_eq!(x.is_some(), y.is_some(), "{:?}", m.id);
            if let (Some(x), Some(y)) = (x, y) {
                assert!((x - y).abs() <= 2e-4, "{:?}: {x} vs {y}", m.id);
            }
        }
    }
    // Unturned, the sideways golfer gives no usable swing.
    let sideways = analyze_swing(&recorded, &opts).unwrap();
    assert_ne!(
        value(&a, MetricId::HipTurn).top.map(f64::round),
        sideways
            .metric(MetricId::HipTurn)
            .and_then(|m| m.values.top)
            .map(f64::round)
    );
}

// ---- norms ----

use super::norms::{
    Band, Confidence, Direction, FADE_OFFSET_HIP_SWAY_IN, FADE_OFFSET_HIP_TURN_DEG,
    FADE_OFFSET_SHOULDER_TILT_DEG, FADE_OFFSET_SHOULDER_TURN_DEG, Measure, Miss, NormUnit, Query,
    Range, Scale, SdSource, TempoMeasure,
};

fn tour_driver() -> Context {
    Context::new(ClubClass::Driver, Sex::Male, Reference::Tour)
}

/// A table of one band.
fn table(band: Band) -> Norms {
    Norms::from_bands(1, std::collections::BTreeMap::new(), vec![band])
}

/// The hip turn at impact as the built-in table has it: 38 ± 5, fade and
/// draw bands 3° either side.
fn hip_turn_impact() -> Band {
    let mut b = Band::new(
        MetricId::HipTurn,
        Position::Impact,
        Reference::Tour,
        38.0,
        5.0,
        NormUnit::Degrees,
    );
    b.shape_offset = Some(3.0);
    b.low = Some(Miss::new(Direction::Closed, "hang-back"));
    b.high = Some(Miss::new(Direction::Open, "slice"));
    b
}

fn grade_deg(norms: &Norms, id: MetricId, position: Position, v: f64) -> Option<Grade> {
    norms.grade_value(id, position, v, Unit::Degrees, None, 0.9, &tour_driver())
}

#[test]
fn the_builtin_table_loads_and_names_only_known_ids() {
    let norms = Norms::builtin().unwrap();
    assert_eq!(norms.skipped(), 0);
    assert_eq!(norms.version, 1);
    assert!(!norms.revision.is_empty());
    assert!(norms.bands().len() > 100, "{}", norms.bands().len());
    let mut seen = std::collections::HashSet::new();
    let offsets = [
        (MetricId::HipTurn, FADE_OFFSET_HIP_TURN_DEG),
        (MetricId::ShoulderTurn, FADE_OFFSET_SHOULDER_TURN_DEG),
        (MetricId::ShoulderTilt, FADE_OFFSET_SHOULDER_TILT_DEG),
        (MetricId::HipSway, FADE_OFFSET_HIP_SWAY_IN),
    ];
    for b in norms.bands() {
        if let Measure::Metric(id) = b.metric {
            assert!(MetricId::ALL.contains(&id));
            let unit = match id.unit() {
                Unit::Degrees => NormUnit::Degrees,
                Unit::Torso => NormUnit::Inches,
                Unit::Ratio => NormUnit::Ratio,
            };
            assert_eq!(b.unit, unit, "{b:?}");
            // Every metric with a band is measured at that position: one
            // way to tell is the stick figure below; here, the turns.
            if matches!(id, MetricId::HipTurn | MetricId::ShoulderTurn) {
                assert_ne!(b.position, Position::Address, "{b:?}");
            }
            if id == MetricId::SpineForwardBend {
                assert_ne!(b.position, Position::Finish);
            }
            let linked = offsets
                .iter()
                .find(|(o, _)| *o == id)
                .filter(|_| b.position == Position::Impact);
            match linked {
                Some((_, o)) => assert_eq!(b.shape_offset, Some(*o), "{b:?}"),
                None => assert_eq!(b.shape_offset, None, "{b:?}"),
            }
        } else {
            assert_eq!(b.position, Position::Swing);
            assert!(matches!(b.unit, NormUnit::Ratio | NormUnit::Seconds));
        }
        assert!(b.sd > 0.0 && b.center.is_finite(), "{b:?}");
        assert!(!b.sources.is_empty(), "{b:?}");
        for s in &b.sources {
            assert!(norms.sources.contains_key(s), "source {s}");
        }
        assert!(
            b.low.is_some() || b.high.is_some(),
            "a band with no miss: {b:?}"
        );
        assert!(
            seen.insert((b.metric, b.position, b.club, b.sex, b.reference)),
            "duplicate {b:?}"
        );
        // High confidence needs several sources (NORMS.md's rule).
        if b.confidence == Confidence::High {
            assert!(b.sources.len() > 1, "{b:?}");
        }
        // Turns are signed positive open, so their words name closed below
        // the center and open above it (`Grade::phrase` relies on it).
        for (miss, want) in [(&b.low, Direction::Closed), (&b.high, Direction::Open)] {
            if let Some(m) = miss
                && matches!(m.direction, Direction::Open | Direction::Closed)
            {
                assert_eq!(m.direction, want, "{b:?}");
            }
        }
        assert_ne!(b.reference, Reference::Personal);
    }
    // Each shape-linked quantity has rows (club, sex and reference
    // variants).
    for (id, _) in offsets {
        assert!(
            norms
                .bands()
                .iter()
                .any(|b| b.metric == Measure::Metric(id) && b.shape_linked())
        );
    }
    // The finish back bend is a reference, not graded; the top shoulder
    // tilt is graded with low confidence.
    for b in norms.bands() {
        let finish_bend = b.metric == Measure::Metric(MetricId::LateralSpineTilt)
            && b.position == Position::Finish;
        assert_eq!(b.graded, !finish_bend, "{b:?}");
        if b.metric == Measure::Metric(MetricId::ShoulderTilt) && b.position == Position::Top {
            assert_eq!(b.confidence, Confidence::Low);
        }
    }
    // Every source key is used.
    for key in norms.sources.keys() {
        assert!(
            norms.bands().iter().any(|b| b.sources.contains(key)),
            "{key}"
        );
    }
}

#[test]
fn turn_words_follow_the_values_sign() {
    let norms = Norms::builtin().unwrap();
    let shoulders = |v| grade_deg(norms, MetricId::ShoulderTurn, Position::Impact, v).unwrap();
    // 15° open against the tour's 30° open: short of the center on its
    // side, so less open, not more closed.
    let g = shoulders(15.0);
    assert_eq!(
        (g.fit, g.side, g.direction),
        (Fit::Far, Side::Low, Some(Direction::Closed))
    );
    assert_eq!(g.phrase(), Some("less open"));
    assert_eq!(
        g.fit_word().unwrap(),
        "Shoulder turn less open than tour: shoulders behind, push or hook"
    );
    // One camera reads a size: −15 is 15° open too.
    assert_eq!(shoulders(-15.0).fit_word(), g.fit_word());
    // In the draw band, short of the center.
    assert_eq!(
        shoulders(26.0).fit_word().unwrap(),
        "Shoulder turn less open: draw side"
    );
    // Beyond the center: more open.
    assert_eq!(
        shoulders(45.0).fit_word().unwrap(),
        "Shoulder turn more open than tour: out-to-in, pull or slice pattern"
    );
    assert_eq!(
        shoulders(34.0).fit_word().unwrap(),
        "Shoulder turn more open: fade side"
    );
    // On the far side of zero: more closed.
    let mut far = g.clone();
    far.value = -5.0;
    far.z = (far.value - far.center) / far.sd;
    assert_eq!(far.phrase(), Some("more closed"));
    assert_eq!(
        far.fit_word().unwrap(),
        "Shoulder turn more closed than tour: shoulders behind, push or hook"
    );
    // Zero is not short of the center: it has no direction of its own.
    far.value = 0.0;
    assert_eq!(far.phrase(), Some("more closed"));
    // A closed center (the top, 98° closed): 75° closed is less closed,
    // 110° closed more closed.
    let ctx = tour_driver();
    let top = |v| {
        norms
            .grade_value_any_position(
                MetricId::ShoulderTurn,
                Position::Top,
                v,
                Unit::Degrees,
                None,
                0.9,
                &ctx,
            )
            .unwrap()
    };
    let g = top(75.0);
    assert_eq!((g.value, g.direction), (-75.0, Some(Direction::Open)));
    assert_eq!(
        g.fit_word().unwrap(),
        "Shoulder turn less closed than tour: short backswing, arms-only, loss of speed"
    );
    assert_eq!(
        top(110.0).fit_word().unwrap(),
        "Shoulder turn more closed than tour: over-swing, loss of control"
    );
    // Against amateurs (hips at the top, 43.8° closed).
    let amateur = Context::new(ClubClass::Driver, Sex::Male, Reference::Amateur);
    let g = norms
        .grade_value_any_position(
            MetricId::HipTurn,
            Position::Top,
            25.0,
            Unit::Degrees,
            None,
            0.9,
            &amateur,
        )
        .unwrap();
    assert_eq!(
        g.fit_word().unwrap(),
        "Hip turn less closed than amateurs: restricted hip turn, arms-only backswing, loss of coil"
    );
    // Positional and magnitude words are unchanged short of the center:
    // hips 1 in toward the target at impact against 4 in toward.
    let sway = norms
        .grade_value(
            MetricId::HipSway,
            Position::Impact,
            2.54,
            Unit::Torso,
            Some(2.54),
            0.9,
            &ctx,
        )
        .unwrap();
    assert_eq!(sway.direction, Some(Direction::Away));
    assert_eq!(sway.phrase(), Some(Direction::Away.phrase()));
    // A left-handed swing reads the same words.
    let mut f = tour_face_on();
    f.shoulder_turn.impact = 15.0;
    let words = |handed| {
        let grades = norms.grade_report(&tour_report(&f, handed), &ctx);
        find_grade(&grades, MetricId::ShoulderTurn, Position::Impact)
            .unwrap()
            .fit_word()
    };
    let right = words(Handed::Right).unwrap();
    assert!(
        right.starts_with("Shoulder turn less open than tour"),
        "{right}"
    );
    assert_eq!(words(Handed::Left).unwrap(), right);
}

#[test]
fn a_shape_linked_band_fits_both_fade_draw_near_far() {
    let norms = table(hip_turn_impact());
    let cases = [
        (38.0, Fit::Both, Side::On),
        (40.0, Fit::Both, Side::On),
        (44.0, Fit::Fade, Side::High),
        (33.0, Fit::Draw, Side::Low),
        (48.0, Fit::Near, Side::High),
        (52.0, Fit::Far, Side::High),
        (24.0, Fit::Far, Side::Low),
        (46.0, Fit::Fade, Side::High),
        (46.1, Fit::Near, Side::High),
    ];
    for (v, fit, side) in cases {
        let g = grade_deg(&norms, MetricId::HipTurn, Position::Impact, v).unwrap();
        assert_eq!((g.fit, g.side), (fit, side), "{v}: {g:?}");
        assert_eq!(g.value, v);
        assert!((g.z - (v - 38.0) / 5.0).abs() < 1e-12);
        assert_eq!(
            g.fade,
            Some(Range {
                center: 41.0,
                sd: 5.0
            })
        );
        assert_eq!(
            g.draw,
            Some(Range {
                center: 35.0,
                sd: 5.0
            })
        );
        assert!((g.z_fade.unwrap() - (v - 41.0) / 5.0).abs() < 1e-12);
        let miss = g.miss.as_ref().map(|m| m.direction);
        match fit {
            Fit::Near | Fit::Far => {
                let want = if side == Side::High {
                    Direction::Open
                } else {
                    Direction::Closed
                };
                assert_eq!(miss, Some(want));
                assert_eq!(g.direction, Some(want));
            }
            _ => assert_eq!(miss, None),
        }
    }
    let word = |v| {
        grade_deg(&norms, MetricId::HipTurn, Position::Impact, v)
            .unwrap()
            .fit_word()
    };
    assert_eq!(word(38.0), None);
    assert_eq!(word(44.0).unwrap(), "Hip turn more open: fade side");
    assert_eq!(word(33.0).unwrap(), "Hip turn less open: draw side");
    assert_eq!(word(52.0).unwrap(), "Hip turn more open than tour: slice");
    assert_eq!(
        word(24.0).unwrap(),
        "Hip turn less open than tour: hang-back"
    );
}

#[test]
fn touching_bands_keep_an_overlap_with_offset_3() {
    let mut b = Band::new(
        MetricId::ShoulderTilt,
        Position::Impact,
        Reference::Tour,
        39.0,
        4.5,
        NormUnit::Degrees,
    );
    b.shape_offset = Some(FADE_OFFSET_SHOULDER_TILT_DEG);
    b.low = Some(Miss::new(Direction::Less, "level"));
    b.high = Some(Miss::new(Direction::More, "hanging back"));
    let norms = table(b);
    for (v, fit) in [
        (39.0, Fit::Both),
        (37.6, Fit::Both),
        (40.4, Fit::Both),
        (37.4, Fit::Fade),
        (40.6, Fit::Draw),
    ] {
        let g = grade_deg(&norms, MetricId::ShoulderTilt, Position::Impact, v).unwrap();
        assert_eq!(g.fit, fit, "{v}");
    }
}

#[test]
fn shape_neutral_is_never_fade_or_draw() {
    let mut b = Band::new(
        MetricId::SpineForwardBend,
        Position::Address,
        Reference::Tour,
        39.0,
        5.0,
        NormUnit::Degrees,
    );
    b.low = Some(Miss::new(Direction::Less, "upright"));
    b.high = Some(Miss::new(Direction::More, "bent"));
    let norms = table(b);
    let mut fits = std::collections::HashSet::new();
    for i in 0..=400 {
        let v = 9.0 + f64::from(i) * 0.15;
        let g = grade_deg(&norms, MetricId::SpineForwardBend, Position::Address, v).unwrap();
        assert!(g.fade.is_none() && g.z_fade.is_none() && g.z_draw.is_none());
        let want = match g.z.abs() {
            z if z <= 1.0 + 1e-9 => Fit::Both,
            z if z <= 2.0 + 1e-9 => Fit::Near,
            _ => Fit::Far,
        };
        assert_eq!(g.fit, want, "{v}");
        assert_eq!(g.side == Side::On, g.fit == Fit::Both);
        fits.insert(g.fit);
    }
    assert_eq!(fits.len(), 3);
}

#[test]
fn a_personal_band_with_one_shape_never_fits_both() {
    let mut b = hip_turn_impact();
    b.reference = Reference::Personal;
    b.fade = Some(Range {
        center: 45.0,
        sd: 4.0,
    });
    let norms = table(b.clone());
    let ctx = Context::new(ClubClass::Driver, Sex::Male, Reference::Personal);
    let mut fits = std::collections::HashSet::new();
    for i in 0..=300 {
        let v = 20.0 + f64::from(i) * 0.15;
        let g = norms
            .grade_value(
                MetricId::HipTurn,
                Position::Impact,
                v,
                Unit::Degrees,
                None,
                0.9,
                &ctx,
            )
            .unwrap();
        assert!(
            matches!(g.fit, Fit::Fade | Fit::Near | Fit::Far),
            "{v}: {g:?}"
        );
        assert!(g.draw.is_none() && g.z_draw.is_none());
        fits.insert(g.fit);
    }
    assert_eq!(fits.len(), 3);
    // Neither sub-band: no grade.
    b.fade = None;
    let norms = table(b);
    assert!(
        norms
            .grade_value(
                MetricId::HipTurn,
                Position::Impact,
                38.0,
                Unit::Degrees,
                None,
                0.9,
                &ctx
            )
            .is_none()
    );
    // The built-in table has no personal bands.
    assert!(
        Norms::builtin()
            .unwrap()
            .grade_value(
                MetricId::HipTurn,
                Position::Impact,
                38.0,
                Unit::Degrees,
                None,
                0.9,
                &ctx
            )
            .is_none()
    );
}

#[test]
fn one_sided_bands_clamp_to_both() {
    let norms = Norms::builtin().unwrap();
    // The finish hip turn (113 ± 8) has no high miss. Graded with the rule
    // lifted: the one-camera turn is impact-only.
    let at = |v| {
        norms
            .grade_value_any_position(
                MetricId::HipTurn,
                Position::Finish,
                v,
                Unit::Degrees,
                None,
                0.9,
                &tour_driver(),
            )
            .unwrap()
    };
    let g = at(140.0);
    assert_eq!(
        (g.fit, g.side, g.miss.as_ref()),
        (Fit::Both, Side::High, None)
    );
    assert_eq!(g.fit_word(), None);
    let g = at(90.0);
    assert_eq!((g.fit, g.side), (Fit::Far, Side::Low));
    assert_eq!(g.miss.unwrap().direction, Direction::Closed);
    // Downswing time: slow is a miss, fast never is.
    let tempo = |down: f64| Tempo {
        backswing_ns: 750_000_000,
        downswing_ns: (down * 1e9) as u64,
        ratio: 0.75 / down,
    };
    let grades = norms.grade_tempo(&tempo(0.1), &tour_driver());
    let down = grades
        .iter()
        .find(|g| g.metric == Measure::Tempo(TempoMeasure::Downswing))
        .unwrap();
    assert_eq!((down.fit, down.side), (Fit::Both, Side::Low));
}

#[test]
fn unknown_rows_are_skipped_not_errors() {
    let row = |metric: &str, position: &str, club: &str, direction: &str| {
        format!(
            r#"{{ "metric": "{metric}", "position": "{position}", "club": {club}, "sex": null,
                 "reference": "tour", "center": 1, "sd": 1, "unit": "degrees",
                 "sd_source": "proxy", "confidence": "low", "future_field": [1, 2],
                 "low": {{ "direction": "{direction}", "tends_to": "x" }} }}"#
        )
    };
    let json = format!(
        r#"{{ "version": 1, "revision": "r", "sources": {{}}, "bands": [{}, {}, {}, {}, {}] }}"#,
        row("wrist_hinge", "top", "null", "less"),
        row("hip_tilt", "p9", "null", "less"),
        row("hip_tilt", "top", r#""putter""#, "less"),
        row("hip_tilt", "top", "null", "sideways"),
        row("hip_tilt", "top", r#""driver""#, "less"),
    );
    let norms = Norms::from_json(&json).unwrap();
    assert_eq!(norms.skipped(), 4);
    assert_eq!(norms.bands().len(), 1);
    assert_eq!(norms.bands()[0].club, Some(ClubClass::Driver));
    // A known row without a center is an error, as is version 2.
    let broken = json.replacen(r#""center": 1, "#, "", 5);
    assert!(matches!(
        Norms::from_json(&broken),
        Err(crate::Error::InvalidArgument(_))
    ));
    let v2 = json.replace(r#""version": 1"#, r#""version": 2"#);
    assert!(Norms::from_json(&v2).is_err());
    // A unit that doesn't match the metric, and a zero SD, are errors.
    let inches = json.replace(r#""unit": "degrees""#, r#""unit": "inches""#);
    assert!(Norms::from_json(&inches).is_err());
    let zero = json.replace(r#""sd": 1"#, r#""sd": 0"#);
    assert!(Norms::from_json(&zero).is_err());
}

#[test]
fn the_builtin_table_never_panics_on_a_bad_embed() {
    // Trimmed so the last cut always lands inside the closing brace, whatever
    // line endings the checkout gave the embedded file.
    let json = Norms::builtin_json().trim_end();
    for cut in [0, 1, json.len() / 3, json.len() / 2, json.len() - 1] {
        assert!(Norms::from_json(&json[..cut]).is_err(), "cut at {cut}");
    }
    assert!(Norms::from_json("{}").is_err());
    assert!(Norms::from_json("[]").is_err());
}

#[test]
fn lookup_falls_back_by_sex_then_club_and_null_sex_wins() {
    let band = |club: Option<ClubClass>, sex: Option<Sex>, center: f64| {
        let mut b = Band::new(
            MetricId::HipTilt,
            Position::Impact,
            Reference::Tour,
            center,
            5.0,
            NormUnit::Degrees,
        );
        b.club = club;
        b.sex = sex;
        b
    };
    let q = |club, sex| {
        Query::new(
            MetricId::HipTilt,
            Position::Impact,
            club,
            sex,
            Reference::Tour,
        )
    };
    let norms = Norms::from_bands(
        1,
        std::collections::BTreeMap::new(),
        vec![
            band(Some(ClubClass::Driver), Some(Sex::Male), 8.0),
            band(Some(ClubClass::MidIron), Some(Sex::Male), 7.0),
        ],
    );
    // A woman with a wedge: the mid-iron male band, both flags.
    let l = norms.band(&q(ClubClass::Wedge, Sex::Female)).unwrap();
    assert_eq!(l.band.center, 7.0);
    assert!(l.club_fallback && l.sex_fallback);
    // The flags reach the grade.
    let ctx = Context::new(ClubClass::Wedge, Sex::Female, Reference::Tour);
    let g = norms
        .grade_value(
            MetricId::HipTilt,
            Position::Impact,
            7.0,
            Unit::Degrees,
            None,
            0.9,
            &ctx,
        )
        .unwrap();
    assert!(g.club_fallback && g.sex_fallback);
    let l = norms.band(&q(ClubClass::Driver, Sex::Male)).unwrap();
    assert!(!l.club_fallback && !l.sex_fallback && l.band.center == 8.0);
    // A `club: null` row wins over the wedge fallback.
    let mut with_any = norms.bands().to_vec();
    with_any.push(band(None, Some(Sex::Male), 6.0));
    let norms = Norms::from_bands(1, std::collections::BTreeMap::new(), with_any);
    let l = norms.band(&q(ClubClass::Wedge, Sex::Male)).unwrap();
    assert_eq!(l.band.center, 6.0);
    assert!(!l.club_fallback);
    // A `sex: null` row wins over the male row for a woman.
    let mut with_either = norms.bands().to_vec();
    with_either.push(band(Some(ClubClass::Driver), None, 5.0));
    let norms = Norms::from_bands(1, std::collections::BTreeMap::new(), with_either);
    let l = norms.band(&q(ClubClass::Driver, Sex::Female)).unwrap();
    assert_eq!(l.band.center, 5.0);
    assert!(!l.sex_fallback);
    // A man keeps his own row.
    assert_eq!(
        norms
            .band(&q(ClubClass::Driver, Sex::Male))
            .unwrap()
            .band
            .center,
        8.0
    );
    // No row at all: none.
    assert!(
        norms
            .band(&Query::new(
                MetricId::HipTilt,
                Position::Top,
                ClubClass::Driver,
                Sex::Male,
                Reference::Tour
            ))
            .is_none()
    );
    // The built-in table: a woman's wedge hip sway at impact takes the
    // female mid-iron row.
    let l = Norms::builtin()
        .unwrap()
        .band(&Query::new(
            MetricId::HipSway,
            Position::Impact,
            ClubClass::Wedge,
            Sex::Female,
            Reference::Tour,
        ))
        .unwrap();
    assert_eq!(l.band.center, 3.7);
    assert!(l.club_fallback && !l.sex_fallback);
}

#[test]
fn inches_bands_use_the_height_or_an_assumed_one() {
    let norms = Norms::builtin().unwrap();
    // Hip sway at the top: −0.6 ± 1.5 in.
    let grade = |v: f64, cm: Option<f64>, scale: Option<BodyScale>| {
        let mut ctx = tour_driver();
        ctx.scale = scale;
        norms.grade_value(
            MetricId::HipSway,
            Position::Top,
            v,
            Unit::Torso,
            cm,
            0.9,
            &ctx,
        )
    };
    let g = grade(-0.02, Some(-0.6 * 2.54), None).unwrap();
    assert_eq!(g.scale, Scale::Measured);
    assert!((g.value + 0.6).abs() < 1e-12 && g.fit == Fit::Both);
    assert_eq!(g.unit, NormUnit::Inches);
    // No centimeters: the body scale and an assumed 177 cm.
    let scale = BodyScale {
        torso_px: 200.0,
        upright_px: Some(220.0),
        cm_per_px: None,
    };
    let g = grade(-0.1, None, Some(scale)).unwrap();
    assert_eq!(g.scale, Scale::Assumed);
    let cm = -0.1 * 200.0 * TORSO_PER_HEIGHT * 177.0 / 220.0;
    assert!((g.value - cm / 2.54).abs() < 1e-9, "{g:?}");
    // A known height in the scale counts as measured.
    let measured = BodyScale {
        cm_per_px: Some(0.25),
        ..scale
    };
    let g = grade(-0.1, None, Some(measured)).unwrap();
    assert_eq!(g.scale, Scale::Measured);
    assert!((g.value - (-0.1 * 200.0 * 0.25 / 2.54)).abs() < 1e-9);
    // Neither: no grade, and a scale without the upright torso can't
    // assume a height either.
    assert!(grade(-0.1, None, None).is_none());
    let old = BodyScale {
        upright_px: None,
        ..scale
    };
    assert!(grade(-0.1, None, Some(old)).is_none());
    // A woman's assumed height is 164 cm.
    let mut ctx = Context::new(ClubClass::Driver, Sex::Female, Reference::Tour);
    ctx.scale = Some(scale);
    let g = norms
        .grade_value(
            MetricId::HipSway,
            Position::Top,
            -0.1,
            Unit::Torso,
            None,
            0.9,
            &ctx,
        )
        .unwrap();
    let cm = -0.1 * 200.0 * TORSO_PER_HEIGHT * 164.0 / 220.0;
    assert!((g.value - cm / 2.54).abs() < 1e-9);
    // Degree bands need no scale; a unit that isn't the metric's is
    // refused.
    assert!(grade_deg(norms, MetricId::HipTilt, Position::Top, -8.0).is_some());
    assert!(
        norms
            .grade_value(
                MetricId::HipTilt,
                Position::Top,
                -8.0,
                Unit::Torso,
                None,
                0.9,
                &tour_driver()
            )
            .is_none()
    );
}

#[test]
fn one_camera_turns_take_the_bands_sign() {
    let norms = Norms::builtin().unwrap();
    let ctx = tour_driver();
    let any = |id, position, v| {
        norms.grade_value_any_position(id, position, v, Unit::Degrees, None, 0.9, &ctx)
    };
    // 40° at the top takes the band's sign: −40 against −40.
    let g = any(MetricId::HipTurn, Position::Top, 40.0).unwrap();
    assert_eq!((g.value, g.fit, g.side), (-40.0, Fit::Both, Side::On));
    // Over-turned at the top: more closed.
    let g = any(MetricId::HipTurn, Position::Top, 55.0).unwrap();
    assert_eq!((g.value, g.side), (-55.0, Side::Low));
    assert_eq!(g.miss.unwrap().direction, Direction::Closed);
    // A negative input is a size too.
    assert_eq!(
        any(MetricId::HipTurn, Position::Top, -40.0).unwrap().value,
        -40.0
    );
    // 38 at impact: +38, Both.
    let g = grade_deg(norms, MetricId::HipTurn, Position::Impact, 38.0).unwrap();
    assert_eq!((g.value, g.fit), (38.0, Fit::Both));
    // At P5 the hips' center (−11) is within 2 SD (6) of zero: no grade.
    assert!(any(MetricId::HipTurn, Position::ArmParallelDown, 11.0).is_none());
    // The impact-only rule: no grade at the top, at P3 or at P5, whatever
    // the rows.
    for position in [
        Position::ArmParallelBack,
        Position::Top,
        Position::ArmParallelDown,
        Position::Finish,
    ] {
        assert!(grade_deg(norms, MetricId::ShoulderTurn, position, 61.0).is_none());
        assert!(grade_deg(norms, MetricId::HipTurn, position, 40.0).is_none());
    }
    // Other metrics keep their own sign.
    let g = grade_deg(norms, MetricId::ShoulderTilt, Position::Top, 38.0).unwrap();
    assert_eq!((g.value, g.fit), (38.0, Fit::Far));
}

#[test]
fn low_confidence_is_not_graded() {
    let norms = Norms::builtin().unwrap();
    let at = |confidence, min: Option<f32>| {
        let mut ctx = tour_driver();
        if let Some(m) = min {
            ctx.min_confidence = m;
        }
        norms.grade_value(
            MetricId::HipTilt,
            Position::Top,
            -8.0,
            Unit::Degrees,
            None,
            confidence,
            &ctx,
        )
    };
    assert!(at(0.3, None).is_none());
    assert!(at(0.5, None).is_some());
    assert_eq!(at(0.5, None).unwrap().confidence, 0.5);
    assert!(at(0.3, Some(0.2)).is_some());
    assert!(at(0.6, Some(0.7)).is_none());
}

#[test]
fn ungraded_rows_are_found_but_not_graded() {
    let norms = Norms::builtin().unwrap();
    let q = Query::new(
        MetricId::LateralSpineTilt,
        Position::Finish,
        ClubClass::Driver,
        Sex::Male,
        Reference::Tour,
    );
    let l = norms.band(&q).unwrap();
    assert!(!l.band.graded && l.band.center == 33.0);
    assert!(grade_deg(norms, MetricId::LateralSpineTilt, Position::Finish, 33.0).is_none());
    // The flag round-trips through JSON as `"grade": false`.
    let json = serde_json::to_string(l.band).unwrap();
    assert!(json.contains(r#""grade":false"#), "{json}");
    let graded = serde_json::to_string(&hip_turn_impact()).unwrap();
    assert!(!graded.contains("grade"), "{graded}");
}

#[test]
fn tempo_grades_ratio_and_downswing() {
    let norms = Norms::builtin().unwrap();
    let tempo = |back: f64, down: f64| Tempo {
        backswing_ns: (back * 1e9) as u64,
        downswing_ns: (down * 1e9) as u64,
        ratio: back / down,
    };
    let find = |grades: &[Grade], m: TempoMeasure| {
        grades
            .iter()
            .find(|g| g.metric == Measure::Tempo(m))
            .cloned()
            .unwrap()
    };
    let g = norms.grade_tempo(&tempo(0.75, 0.25), &tour_driver());
    assert_eq!(g.len(), 2);
    let ratio = find(&g, TempoMeasure::Ratio);
    assert_eq!(
        (ratio.fit, ratio.unit, ratio.position),
        (Fit::Both, NormUnit::Ratio, Position::Swing)
    );
    assert!((ratio.value - 3.0).abs() < 1e-9);
    let down = find(&g, TempoMeasure::Downswing);
    assert_eq!((down.fit, down.unit), (Fit::Both, NormUnit::Seconds));
    assert!((down.value - 0.25).abs() < 1e-9);
    // A quick backswing (1.75:1) and a slow downswing (0.4 s).
    let g = norms.grade_tempo(&tempo(0.7, 0.4), &tour_driver());
    let ratio = find(&g, TempoMeasure::Ratio);
    assert_eq!((ratio.fit, ratio.side), (Fit::Far, Side::Low));
    assert_eq!(
        ratio.fit_word().unwrap(),
        "Tempo less than tour: quick backswing"
    );
    let down = find(&g, TempoMeasure::Downswing);
    assert_eq!((down.fit, down.side), (Fit::Far, Side::High));
    // A woman's downswing band is 0.30 s.
    let woman = Context::new(ClubClass::Driver, Sex::Female, Reference::Tour);
    let down = find(
        &norms.grade_tempo(&tempo(0.9, 0.3), &woman),
        TempoMeasure::Downswing,
    );
    assert!(down.z.abs() < 1e-9 && !down.sex_fallback);
    let ratio = find(
        &norms.grade_tempo(&tempo(0.9, 0.3), &woman),
        TempoMeasure::Ratio,
    );
    assert!(ratio.sex_fallback);
}

/// The golfer's height in the tour-figure tests, and the figure's pixels
/// per inch with it (its upright torso is [`TORSO`]).
const TOUR_HEIGHT_CM: f64 = 177.0;

fn inches(v: f64) -> f64 {
    v * 2.54 / (TORSO_PER_HEIGHT * TOUR_HEIGHT_CM / TORSO)
}

/// The face-on figure at the driver tour bands (male): hip sway −0.6 in at
/// the top and +4 in at impact, lateral spine tilt 4, 4.5 and 18° (the
/// chest sway and sway gap follow from the two and land inside their
/// bands), shoulder tilt 15 / −38 / 25°, hip tilt 1 / −8 / 8°, impact turns
/// 38° (hips) and 30° (shoulders), pelvis lift −1.6 / +0.9 in, head shift
/// −1 / −0.5 in, head height −0.25 / −0.75 in.
fn tour_face_on() -> Figure {
    let mut f = Figure::face_on();
    f.pelvis_dx = Phases {
        address: 0.0,
        top: inches(-0.6),
        impact: inches(4.0),
        finish: inches(5.0),
    };
    f.pelvis_dy = Phases {
        address: 0.0,
        top: inches(1.6),
        impact: inches(-0.9),
        finish: inches(-1.0),
    };
    f.spine = Phases {
        address: -4.0,
        top: -4.5,
        impact: -18.0,
        finish: -FINISH_LEAN,
    };
    f.shoulder_tilt = Phases {
        address: 15.0,
        top: -38.0,
        impact: 25.0,
        finish: 0.0,
    };
    f.hip_tilt = Phases {
        address: 1.0,
        top: -8.0,
        impact: 8.0,
        finish: 0.0,
    };
    f.hip_turn.impact = 38.0;
    f.shoulder_turn.impact = 30.0;
    f.head_dx = Phases {
        address: 0.0,
        top: inches(-1.0),
        impact: inches(-0.5),
        finish: 40.0,
    };
    f.head_dy = Phases {
        address: 0.0,
        top: inches(0.25),
        impact: inches(0.75),
        finish: -10.0,
    };
    f
}

/// The down-the-line figure at the driver tour bands (male): forward bend
/// 39° at address and 35° at impact (so 4° of lost posture), early
/// extension 0.5 / 0.4 in, hands 3 in deep at the top, pelvis lift and
/// head height as face-on.
fn tour_down_the_line() -> Figure {
    let mut f = Figure::down_the_line();
    f.spine = Phases {
        address: 39.0,
        top: 37.0,
        impact: 35.0,
        finish: 10.0,
    };
    f.pelvis_dx = Phases {
        address: 0.0,
        top: inches(0.5),
        impact: inches(0.4),
        finish: 10.0,
    };
    f.pelvis_dy = tour_face_on().pelvis_dy;
    f.head_dy = tour_face_on().head_dy;
    // The shoulders turned side-on at the top show their width, with the
    // trail shoulder 3 in in front of the hands (which are 95 px behind
    // the shoulders' midpoint).
    f.shoulder_half.top = top_hands_behind() - inches(3.0);
    f
}

/// How far the down-the-line figure's hands are behind its shoulders'
/// midpoint at the top (pixels).
fn top_hands_behind() -> f64 {
    -ARM * (-HAND_SWING.to_radians()).sin()
}

fn tour_report(figure: &Figure, handed: Handed) -> SwingReport {
    let track = match handed {
        Handed::Right => figure.track(),
        Handed::Left => mirrored(&figure.track()),
    };
    let mut opts = options(figure.view, handed, Some(IMPACT_S));
    opts.smoothing = raw();
    opts.height_cm = Some(TOUR_HEIGHT_CM);
    analyze_swing(&track, &opts).unwrap()
}

/// The P3 or P5 slot of a `Positions`.
fn slot(pos: &mut Positions, p: Position) -> &mut Option<f64> {
    match p {
        Position::ArmParallelBack => &mut pos.arm_parallel_back,
        _ => &mut pos.arm_parallel_down,
    }
}

fn find_grade(grades: &[Grade], id: MetricId, position: Position) -> Option<&Grade> {
    grades
        .iter()
        .find(|g| g.metric == Measure::Metric(id) && g.position == position)
}

#[test]
fn the_tour_figure_grades_every_band_both() {
    let norms = Norms::builtin().unwrap();
    let ctx = tour_driver();
    for figure in [tour_face_on(), tour_down_the_line()] {
        let report = tour_report(&figure, Handed::Right);
        assert!(near(report.scale.unwrap().upright_px, TORSO, 0.01));
        let grades = norms.grade_report(&report, &ctx);
        // The cells with a graded band, the one-camera turns at impact only.
        let mut expected = 0;
        for m in &report.metrics {
            for &p in Position::KEY {
                let turn = matches!(m.id, MetricId::HipTurn | MetricId::ShoulderTurn);
                let band = norms.band(&Query::new(m.id, p, ctx.club, ctx.sex, ctx.reference));
                if m.values.at(p).is_some()
                    && band.is_some_and(|l| l.band.graded)
                    && (!turn || p == Position::Impact)
                {
                    expected += 1;
                }
            }
        }
        assert_eq!(grades.len(), expected, "{:?}", figure.view);
        assert!(grades.len() >= 8, "{}", grades.len());
        for g in &grades {
            let Measure::Metric(id) = g.metric else {
                panic!("tempo in a report grade")
            };
            let m = report.metric(id).unwrap();
            let want = match m.unit {
                Unit::Torso => m.cm.unwrap().at(g.position).unwrap() / 2.54,
                _ => m.values.at(g.position).unwrap(),
            };
            assert!((g.value - want).abs() < 1e-9, "{g:?}");
            assert!((g.z - (g.value - g.center) / g.sd).abs() < 1e-9);
            assert_eq!(
                g.scale,
                if m.unit == Unit::Torso {
                    Scale::Measured
                } else {
                    Scale::Native
                }
            );
            // P3 and P5 fall on the figure's ramps: graded, but the figure
            // sets only address, top, impact and finish.
            if matches!(
                g.position,
                Position::ArmParallelBack | Position::ArmParallelDown
            ) {
                continue;
            }
            assert_eq!(g.fit, Fit::Both, "{g:?}");
            assert!(g.miss.is_none() && g.fit_word().is_none());
        }
        // The same report with P3 and P5 set to their band centers grades
        // Both there too.
        let mut at_centers = report.clone();
        for m in &mut at_centers.metrics {
            for p in [Position::ArmParallelBack, Position::ArmParallelDown] {
                let Some(l) = norms.band(&Query::new(m.id, p, ctx.club, ctx.sex, ctx.reference))
                else {
                    continue;
                };
                let c = l.band.center;
                if slot(&mut m.values, p).is_none() {
                    continue;
                }
                if let Some(cm) = &mut m.cm {
                    *slot(cm, p) = Some(c * 2.54);
                } else {
                    *slot(&mut m.values, p) = Some(c.abs());
                }
            }
        }
        for g in norms.grade_report(&at_centers, &ctx) {
            assert_eq!(g.fit, Fit::Both, "{g:?}");
        }
        // Left-handed: the mirror image grades identically.
        let left = norms.grade_report(&tour_report(&figure, Handed::Left), &ctx);
        assert_eq!(left.len(), grades.len());
        for (l, r) in left.iter().zip(&grades) {
            assert_eq!(
                (l.metric, l.position, l.fit, l.side),
                (r.metric, r.position, r.fit, r.side)
            );
            assert!((l.value - r.value).abs() < 1e-3, "{l:?} vs {r:?}");
        }
    }
    // Tempo 3:1 with a 0.25 s downswing.
    let tempo = Tempo {
        backswing_ns: 750_000_000,
        downswing_ns: 250_000_000,
        ratio: 3.0,
    };
    assert!(
        norms
            .grade_tempo(&tempo, &ctx)
            .iter()
            .all(|g| g.fit == Fit::Both)
    );
}

#[test]
fn each_fault_figure_names_its_miss() {
    let norms = Norms::builtin().unwrap();
    let ctx = tour_driver();
    let check = |figure: Figure, id, position, fit: Fit, side, direction| {
        let grades = norms.grade_report(&tour_report(&figure, Handed::Right), &ctx);
        let g = find_grade(&grades, id, position)
            .unwrap_or_else(|| panic!("{id:?} at {position:?} graded"));
        assert_eq!(
            (g.fit, g.side),
            (fit, side),
            "{id:?} at {position:?}: {g:?}"
        );
        assert_eq!(g.direction, Some(direction), "{g:?}");
        if matches!(fit, Fit::Near | Fit::Far) {
            assert_eq!(g.miss.as_ref().unwrap().direction, direction);
            let word = g.fit_word().unwrap();
            assert!(
                word.starts_with(g.metric.name()) && word.contains(" than tour: "),
                "{word}"
            );
        }
        // Left-handed, the same fault reads the same.
        let left = norms.grade_report(&tour_report(&figure, Handed::Left), &ctx);
        let l = find_grade(&left, id, position).unwrap();
        assert_eq!((l.fit, l.side, l.direction), (g.fit, g.side, g.direction));
    };
    // Sway off the ball at the top.
    let mut f = tour_face_on();
    f.pelvis_dx.top = inches(-4.0);
    check(
        f,
        MetricId::HipSway,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Away,
    );
    // A slide at impact.
    let mut f = tour_face_on();
    f.pelvis_dx.impact = inches(9.0);
    check(
        f,
        MetricId::HipSway,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::Toward,
    );
    // A reverse pivot: hips toward the target and the spine leaning
    // toward it at the top.
    let mut f = tour_face_on();
    f.pelvis_dx.top = inches(3.0);
    check(
        f,
        MetricId::HipSway,
        Position::Top,
        Fit::Far,
        Side::High,
        Direction::Toward,
    );
    let mut f = tour_face_on();
    f.spine.top = 8.0;
    check(
        f,
        MetricId::LateralSpineTilt,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Less,
    );
    // Head up at impact, and a dip at the top.
    let mut f = tour_face_on();
    f.head_dy.impact = inches(-3.0);
    check(
        f,
        MetricId::HeadVertical,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::Up,
    );
    let mut f = tour_face_on();
    f.head_dy.top = inches(2.0);
    check(
        f,
        MetricId::HeadVertical,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Down,
    );
    // Hips 20° less open than tour at impact, and far more open.
    let mut f = tour_face_on();
    f.hip_turn.impact = 18.0;
    check(
        f,
        MetricId::HipTurn,
        Position::Impact,
        Fit::Far,
        Side::Low,
        Direction::Closed,
    );
    let mut f = tour_face_on();
    f.hip_turn.impact = 55.0;
    check(
        f,
        MetricId::HipTurn,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::Open,
    );
    // Hips a little more open than the neutral band: the fade side.
    let mut f = tour_face_on();
    f.hip_turn.impact = 44.0;
    check(
        f,
        MetricId::HipTurn,
        Position::Impact,
        Fit::Fade,
        Side::High,
        Direction::Open,
    );
    // Steep and flat shoulders at the top.
    let mut f = tour_face_on();
    f.shoulder_tilt.top = -50.0;
    check(
        f,
        MetricId::ShoulderTilt,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::More,
    );
    // Down the line: standing up at impact.
    let mut f = tour_down_the_line();
    f.spine.impact = 24.0;
    check(
        f,
        MetricId::SpineForwardBend,
        Position::Impact,
        Fit::Far,
        Side::Low,
        Direction::Less,
    );
    // Early extension, and falling back.
    let mut f = tour_down_the_line();
    f.pelvis_dx.impact = inches(4.0);
    check(
        f,
        MetricId::EarlyExtension,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::TowardBall,
    );
    let mut f = tour_down_the_line();
    f.pelvis_dx.impact = inches(-3.0);
    check(
        f,
        MetricId::EarlyExtension,
        Position::Impact,
        Fit::Far,
        Side::Low,
        Direction::Back,
    );
    // Hands in front of the trail shoulder at the top, and deep.
    let mut f = tour_down_the_line();
    f.shoulder_half.top = top_hands_behind() + inches(1.0);
    check(
        f,
        MetricId::HandDepth,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Less,
    );
    let mut f = tour_down_the_line();
    f.shoulder_half.top = top_hands_behind() - inches(7.0);
    check(
        f,
        MetricId::HandDepth,
        Position::Top,
        Fit::Far,
        Side::High,
        Direction::More,
    );
}

#[test]
fn grades_serialize() {
    let norms = Norms::builtin().unwrap();
    let g = grade_deg(norms, MetricId::HipTurn, Position::Impact, 52.0).unwrap();
    let json = serde_json::to_string(&g).unwrap();
    assert!(
        json.contains(r#""metric":"hip_turn""#) && json.contains(r#""fit":"far""#),
        "{json}"
    );
    let back: Grade = serde_json::from_str(&json).unwrap();
    assert_eq!(back, g);
    assert!(g.proxy_sd());
    let t: Measure = serde_json::from_str(r#""tempo_ratio""#).unwrap();
    assert_eq!(t, Measure::Tempo(TempoMeasure::Ratio));
    assert_eq!(serde_json::to_string(&t).unwrap(), r#""tempo_ratio""#);
    let _ = SdSource::Published;
}

#[test]
fn the_norms_doc_lists_every_band() {
    let doc = include_str!("../../docs/NORMS.md");
    let norms = Norms::builtin().unwrap();
    let position = |p: Position| match p {
        Position::Address => "address",
        Position::ArmParallelBack => "P3",
        Position::Top => "top",
        Position::ArmParallelDown => "P5",
        Position::Impact => "impact",
        Position::Finish => "finish",
        _ => "swing",
    };
    for b in norms.bands().iter().filter(|b| b.sex != Some(Sex::Female)) {
        let metric = serde_json::to_value(b.metric).unwrap();
        let club = match b.club {
            Some(ClubClass::Driver) => "driver",
            Some(ClubClass::MidIron) => "mid-iron",
            Some(_) => "wedge",
            None => "any",
        };
        let center = format!("{}", b.center).replace('-', "−");
        let row = format!(
            "| `{}` | {} | {club} | {center}",
            metric.as_str().unwrap(),
            position(b.position)
        );
        assert!(doc.contains(&row), "docs/NORMS.md lacks {row}");
    }
    for key in norms.sources.keys() {
        assert!(doc.contains(&format!("| {key} | ")), "source {key}");
    }
}
