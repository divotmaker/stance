//! Live posture: cheap per-frame angles for a display beside the camera.

use serde::{Deserialize, Serialize};

use super::body::{Body, Handed, Setup, View};
use crate::pose::{Keypoint, OneEuro, OneEuroParams, Pose, Skeleton};

/// Keypoints scoring below this are ignored in [`posture`].
const MIN_SCORE: f32 = 0.3;

/// One frame's posture: 2D angles in the image plane, for a live readout.
///
/// Angles follow the [`MetricId`](super::MetricId) definitions. Each is
/// `None` when a keypoint it needs scores below 0.3.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Posture {
    /// The view and handedness the angles were measured for.
    pub setup: Setup,
    /// Spine (pelvis → shoulders) from vertical, degrees. Face-on: lateral
    /// tilt, positive with the shoulders away from the target. Down the
    /// line: forward bend, positive leaning toward the ball.
    pub spine_angle: Option<f64>,
    /// Shoulder line from horizontal, degrees, positive with the lead
    /// shoulder higher (meaningful face-on).
    pub shoulder_tilt: Option<f64>,
    /// Lead knee flexion, degrees (0 = straight).
    pub lead_knee_flex: Option<f64>,
    /// Trail knee flexion, degrees.
    pub trail_knee_flex: Option<f64>,
    /// Lead shoulder–elbow–wrist angle, degrees (180 = straight).
    pub lead_elbow_angle: Option<f64>,
    /// Trail shoulder–elbow–wrist angle, degrees.
    pub trail_elbow_angle: Option<f64>,
    /// The head (head-top keypoint, else the nose), image pixels `[x, y]`.
    pub head: Option<[f64; 2]>,
    /// Torso length (pelvis to shoulders midpoint), pixels.
    pub torso_px: Option<f64>,
}

/// The head's shift from a reference posture (typically address), in that
/// posture's torso lengths.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HeadShift {
    /// Along the image's x: toward the target face-on, toward the ball
    /// down the line.
    pub forward: f64,
    /// Upwards.
    pub up: f64,
}

/// Keypoints from a mirrored image as an unmirrored camera would see them:
/// `x` reflected about the image's center line and left/right labels
/// swapped (a pose model labels the mirror image of a left side as a right
/// side).
pub(crate) fn unmirror_keypoints(
    skeleton: Skeleton,
    keypoints: &[Keypoint],
    image_width: f32,
) -> Vec<Keypoint> {
    let mirror = skeleton.mirror();
    let mut out = keypoints.to_vec();
    if keypoints.len() == mirror.len() {
        for (i, k) in keypoints.iter().enumerate() {
            out[mirror[i]] = Keypoint {
                x: image_width - k.x,
                ..*k
            };
        }
    }
    out
}

/// A pose from a mirrored image (a selfie-style preview `image_width`
/// pixels wide) as an unmirrored camera would see it, so [`posture`] can be
/// given the golfer's true hand: `posture(&unmirror(&pose, w), view,
/// handed)`.
#[must_use]
pub fn unmirror(pose: &Pose, image_width: f32) -> Pose {
    let mut out = pose.clone();
    out.keypoints = unmirror_keypoints(pose.skeleton, &pose.keypoints, image_width);
    let (x1, x2) = (pose.bbox.x1, pose.bbox.x2);
    out.bbox.x1 = image_width - x2;
    out.bbox.x2 = image_width - x1;
    out
}

/// Measures a pose's posture. Cheap: a few angles, no allocation. For a
/// mirrored image, measure [`unmirror`]ed poses.
#[must_use]
pub fn posture(pose: &Pose, view: View, handed: Handed) -> Posture {
    let setup = Setup { view, handed };
    let Some(body) = Body::from_keypoints(&pose.keypoints, handed, MIN_SCORE) else {
        return Posture {
            setup,
            spine_angle: None,
            shoulder_tilt: None,
            lead_knee_flex: None,
            trail_knee_flex: None,
            lead_elbow_angle: None,
            trail_elbow_angle: None,
            head: None,
            torso_px: None,
        };
    };
    let spine_sign = match view {
        View::FaceOn => -setup.forward_x(),
        View::DownTheLine => setup.forward_x(),
    };
    Posture {
        setup,
        spine_angle: body.spine_tilt(spine_sign),
        shoulder_tilt: body.shoulder_tilt(),
        lead_knee_flex: body.knee_flex(true),
        trail_knee_flex: body.knee_flex(false),
        lead_elbow_angle: body.elbow(true),
        trail_elbow_angle: body.elbow(false),
        head: body.head().map(|p| [p.x, p.y]),
        torso_px: body.torso(),
    }
}

impl Posture {
    /// The head's shift from `reference` (the posture captured at address),
    /// in the reference's torso lengths.
    #[must_use]
    pub fn head_shift(&self, reference: &Posture) -> Option<HeadShift> {
        let ([x, y], [x0, y0]) = (self.head?, reference.head?);
        let l = reference.torso_px.filter(|l| *l > 1e-6)?;
        Some(HeadShift {
            forward: self.setup.forward_x() * (x - x0) / l,
            up: (y0 - y) / l,
        })
    }
}

/// One Euro smoothing of a live [`Posture`] stream, each field filtered on
/// its own (angles in degrees, positions in pixels). A missing field resets
/// its filter.
#[derive(Debug, Clone)]
pub struct PostureFilter {
    filters: [OneEuro; 9],
}

impl PostureFilter {
    /// A filter with no history.
    #[must_use]
    pub fn new(params: OneEuroParams) -> Self {
        Self {
            filters: std::array::from_fn(|_| OneEuro::new(params)),
        }
    }

    /// Filters a posture measured at `t_ns`.
    #[must_use]
    pub fn filter(&mut self, t_ns: u64, posture: &Posture) -> Posture {
        // Seconds for the filter; precision is ample.
        #[allow(clippy::cast_precision_loss, reason = "ns times fit f64")]
        let t = t_ns as f64 * 1e-9;
        let [spine, shoulder, lk, tk, le, te, hx, hy, torso] = &mut self.filters;
        let f = |filter: &mut OneEuro, v: Option<f64>| {
            if let Some(v) = v {
                Some(filter.filter(t, v))
            } else {
                filter.reset();
                None
            }
        };
        let head = if let Some([x, y]) = posture.head {
            Some([hx.filter(t, x), hy.filter(t, y)])
        } else {
            hx.reset();
            hy.reset();
            None
        };
        Posture {
            setup: posture.setup,
            spine_angle: f(spine, posture.spine_angle),
            shoulder_tilt: f(shoulder, posture.shoulder_tilt),
            lead_knee_flex: f(lk, posture.lead_knee_flex),
            trail_knee_flex: f(tk, posture.trail_knee_flex),
            lead_elbow_angle: f(le, posture.lead_elbow_angle),
            trail_elbow_angle: f(te, posture.trail_elbow_angle),
            head,
            torso_px: f(torso, posture.torso_px),
        }
    }

    /// Forgets the history.
    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
    }
}
