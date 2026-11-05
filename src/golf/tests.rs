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
use crate::pose::{BBox, Keypoint, POSE_TRACK_VERSION, Pose, PoseFrame, PoseTrack, Skeleton};
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
/// towards the right-hander's lead side, +x): still, a cosine backswing to
/// the trail side, an accelerating downswing that is fastest at impact, and
/// a decelerating follow-through to the lead side, then still.
fn hand_angle(t: f64) -> f64 {
    let a = HAND_SWING.to_radians();
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
    /// Hip half-width.
    hip_half: Phases,
    /// Shoulder half-width.
    shoulder_half: Phases,
    /// Spine tilt from vertical, positive towards +x.
    spine: Phases,
    /// Shoulder line, positive with the lead shoulder higher.
    shoulder_tilt: Phases,
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
            hip_half: Phases {
                address: 45.0,
                top: 36.0,
                impact: 40.5,
                finish: 22.5,
            },
            // A 70° turn at the top, 30° at impact, 60° at the finish.
            shoulder_half: Phases {
                address: 85.0,
                top: 85.0 * 70f64.to_radians().cos(),
                impact: 85.0 * 30f64.to_radians().cos(),
                finish: 85.0 * 60f64.to_radians().cos(),
            },
            spine: Phases {
                address: -6.0,
                top: -2.0,
                impact: -12.0,
                finish: 0.0,
            },
            shoulder_tilt: Phases {
                address: 10.0,
                top: -40.0,
                impact: 25.0,
                finish: 0.0,
            },
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
            hip_half: Phases::flat(4.0),
            shoulder_half: Phases::flat(5.0),
            spine: Phases {
                address: 35.0,
                top: 33.0,
                impact: 30.0,
                finish: 10.0,
            },
            shoulder_tilt: Phases::flat(0.0),
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
        (x + self.pelvis_dx.at(t), y)
    }

    fn shoulders(&self, t: f64) -> (f64, f64) {
        let (px, py) = self.pelvis(t);
        let s = self.spine.at(t).to_radians();
        (px + TORSO * s.sin(), py - TORSO * s.cos())
    }

    fn hands(&self, t: f64) -> (f64, f64) {
        let (sx, sy) = self.shoulders(t);
        let a = hand_angle(t);
        (sx + ARM * a.sin(), sy + ARM * a.cos())
    }

    /// The right-hander's 26 keypoints at `t`.
    fn keypoints(&self, t: f64) -> Vec<Keypoint> {
        let mut k = vec![(0.0, 0.0); 26];
        let set = |k: &mut Vec<(f64, f64)>, j: Joint, p: (f64, f64)| k[j.index(Handed::Right)] = p;
        let (px, py) = self.pelvis(t);
        let (sx, sy) = self.shoulders(t);
        let hip = self.hip_half.at(t);
        let (lead_sign, half_shoulder) = (1.0, self.shoulder_half.at(t));
        set(&mut k, Joint::Hip, (px, py));
        set(&mut k, Joint::LeadHip, (px + lead_sign * hip, py));
        set(&mut k, Joint::TrailHip, (px - lead_sign * hip, py));
        let st = self.shoulder_tilt.at(t).to_radians();
        let (dx, dy) = (half_shoulder * st.cos(), half_shoulder * st.sin());
        let lead_shoulder = (sx + dx, sy - dy);
        let trail_shoulder = (sx - dx, sy + dy);
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
        let lead_hip = (px + lead_sign * hip, py);
        let trail_hip = (px - lead_sign * hip, py);
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
    // 1.0 s up and 0.4 s down (2.5:1): the takeaway's first millimetres,
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
        top: Some(event(1.9)),
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
    let cm = scale.cm_per_px.unwrap();
    assert!((cm - TORSO_PER_HEIGHT * 180.0 / TORSO).abs() < 1e-6);
    let tol = 1e-3;
    // Hip sway: 20 px away from the target at the top, 30 px towards it at
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
    let shoulders = value(&r, MetricId::ShoulderTilt);
    assert!(near(shoulders.address, 10.0, 0.05), "{shoulders:?}");
    // The figure holds 25° from 2.25 to 2.55 s, around the impact.
    assert!(near(shoulders.impact, 25.0, 0.05));
    assert!(shoulders.top.is_none() && shoulders.finish.is_none());
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
                    (e.top, m.values.top),
                    (e.impact, m.values.impact),
                    (e.finish, m.values.finish),
                ] {
                    if let Some(v) = value {
                        let frame = event.unwrap().frame;
                        assert_eq!(series.values[frame], Some(v), "{:?} at {frame}", m.id);
                        assert_eq!(s.value_at(m.id, e_t(event)), Some(v));
                    }
                }
                // Centimetres follow the body scale as the positions' do.
                if let (Some(cm), Some(top)) = (m.cm, m.values.top) {
                    assert_eq!(r.scale.unwrap().to_cm(top), cm.top);
                }
            }
            // Shifts are from the address frame.
            let address = e.address.unwrap().frame;
            for id in [
                MetricId::HipSway,
                MetricId::EarlyExtension,
                MetricId::HeadVertical,
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
    // Hip half-widths 36 and 40.5 against 45 at address.
    assert!(
        near(hips.values.top, 0.8f64.acos().to_degrees(), 0.01),
        "{hips:?}"
    );
    assert!(near(hips.values.impact, 0.9f64.acos().to_degrees(), 0.01));
    assert!(hips.values.address.is_none() && hips.values.finish.is_none());
    let shoulders = value(&r, MetricId::ShoulderTurn);
    assert!(near(shoulders.top, 70.0, 0.01), "{shoulders:?}");
    assert!(near(shoulders.impact, 30.0, 0.01));
    assert!(shoulders.address.is_none() && shoulders.finish.is_none());
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
        let expect = |half: f64, half0: f64| (half / half0).min(1.0).acos().to_degrees();
        let hip = series_at(&r, MetricId::HipTurn, frame);
        assert!(
            near(hip, expect(figure.hip_half.at(t), 45.0), 0.1),
            "{t}: {hip:?}"
        );
        let sh = series_at(&r, MetricId::ShoulderTurn, frame);
        assert!(
            near(sh, expect(figure.shoulder_half.at(t), 85.0), 0.1),
            "{t}: {sh:?}"
        );
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
    wide.hip_half.impact = 50.0;
    wide.shoulder_half.top = 90.0;
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
    // 420 frames, 10 face-on metrics at 4 decimals: under 100 bytes per
    // frame.
    let frames = report.series.t_ns.len();
    assert_eq!(frames, 420);
    assert!(json.len() < frames * 100, "{} bytes", json.len());

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
                    (a.values.top, b.values.top),
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
                // it at the top, towards it at impact, for either hand.
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
            // The right-hander's mirror image analysed as right-handed is
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
            "interpolated: half the neighbours' score"
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

    // Too little to analyse.
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
