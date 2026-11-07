//! Camera view, handedness, lead/trail keypoints and the 2D geometry the
//! metrics share.

use serde::{Deserialize, Serialize};

use crate::pose::{Keypoint, Skeleton};

/// Where the camera stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum View {
    /// Facing the golfer's chest, square to the target line.
    FaceOn,
    /// Behind the golfer's hands, looking down the target line.
    DownTheLine,
}

/// Which way the golfer swings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Handed {
    /// Right-handed: the left side leads (faces the target).
    #[default]
    Right,
    /// Left-handed: the right side leads.
    Left,
}

/// A Halpe-26 keypoint named by its role in the swing.
///
/// "Lead" is the side nearer the target: the left side of a right-handed
/// golfer, the right side of a left-hander. [`Joint::index`] maps a joint to
/// the Halpe-26 index for a [`Handed`]; a left-hander's joints are the
/// mirror images of a right-hander's, so every metric mirrors cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs, reason = "the names are the keypoints")]
pub enum Joint {
    Nose,
    LeadEye,
    TrailEye,
    LeadEar,
    TrailEar,
    LeadShoulder,
    TrailShoulder,
    LeadElbow,
    TrailElbow,
    LeadWrist,
    TrailWrist,
    LeadHip,
    TrailHip,
    LeadKnee,
    TrailKnee,
    LeadAnkle,
    TrailAnkle,
    /// Top of the head.
    Head,
    Neck,
    /// Hip center.
    Hip,
    LeadBigToe,
    TrailBigToe,
    LeadSmallToe,
    TrailSmallToe,
    LeadHeel,
    TrailHeel,
}

impl Joint {
    /// The Halpe-26 keypoint index of this joint for a golfer of the given
    /// handedness.
    #[must_use]
    pub fn index(self, handed: Handed) -> usize {
        // The variants follow Halpe-26 order with "lead" in the left slots,
        // so a right-hander's index is the discriminant.
        let right_handed = self as usize;
        match handed {
            Handed::Right => right_handed,
            Handed::Left => Skeleton::Halpe26.mirror()[right_handed],
        }
    }
}

/// Camera view and handedness together: what the image directions mean.
///
/// Image coordinates have `x` growing to the right and `y` growing
/// downwards. Keypoint labels are the person's own sides, so:
///
/// - **Face-on**, the golfer faces the camera and their left side appears on
///   the image's right. A right-hander's target (on their left) is therefore
///   on the image's **right** (+x); a left-hander's is on the left (−x).
/// - **Down-the-line**, the camera looks along the target line from behind
///   the golfer, so the target is into the image. A right-hander faces the
///   ball on the image's **right** (+x); a left-hander faces left (−x).
///
/// A mirrored image (a selfie-style preview) swaps both: a right-hander then
/// looks left-handed. [`SwingOptions::mirrored`](super::SwingOptions::mirrored)
/// and [`unmirror`](super::unmirror) undo it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Setup {
    /// The camera view.
    pub view: View,
    /// The golfer's handedness.
    pub handed: Handed,
}

impl Setup {
    /// `+1.0` when the target is toward +x in the image, `−1.0` when
    /// toward −x; `None` down the line (the target is into the image).
    #[must_use]
    pub fn target_x(self) -> Option<f64> {
        (self.view == View::FaceOn).then(|| self.sign())
    }

    /// `+1.0` when the golfer faces +x (toward the ball), `−1.0` when −x;
    /// `None` face-on (the golfer faces the camera).
    #[must_use]
    pub fn ball_x(self) -> Option<f64> {
        (self.view == View::DownTheLine).then(|| self.sign())
    }

    /// The image direction "forward" means in this view: toward the target
    /// face-on, toward the ball down the line.
    #[must_use]
    pub fn forward_x(self) -> f64 {
        self.sign()
    }

    fn sign(self) -> f64 {
        match self.handed {
            Handed::Right => 1.0,
            Handed::Left => -1.0,
        }
    }
}

/// A point in image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Pt {
    pub x: f64,
    pub y: f64,
}

impl Pt {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn mid(self, other: Self) -> Self {
        Self::new(
            f64::midpoint(self.x, other.x),
            f64::midpoint(self.y, other.y),
        )
    }

    pub fn dist(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

/// The angle at `b` between `b→a` and `b→c`, degrees in `[0, 180]`; `None`
/// when a segment has no length.
pub(crate) fn angle_at(a: Pt, b: Pt, c: Pt) -> Option<f64> {
    let (ux, uy) = (a.x - b.x, a.y - b.y);
    let (vx, vy) = (c.x - b.x, c.y - b.y);
    let (lu, lv) = (ux.hypot(uy), vx.hypot(vy));
    if lu < 1e-9 || lv < 1e-9 {
        return None;
    }
    let cos = ((ux * vx + uy * vy) / (lu * lv)).clamp(-1.0, 1.0);
    Some(cos.acos().to_degrees())
}

/// The tilt of the segment `lower → upper` from vertical, degrees; positive
/// when `upper` lies toward `sign` (±1 in x) of `lower`. Range ±180.
pub(crate) fn tilt_from_vertical(lower: Pt, upper: Pt, sign: f64) -> Option<f64> {
    let (dx, up) = (sign * (upper.x - lower.x), lower.y - upper.y);
    (dx.hypot(up) > 1e-9).then(|| dx.atan2(up).to_degrees())
}

/// One frame's keypoints indexed by Halpe-26 index: a position where the
/// keypoint is usable, and its score (0 where it isn't).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Body {
    pub pts: [Option<Pt>; 26],
    pub scores: [f32; 26],
    pub handed: Handed,
}

impl Body {
    /// Keypoints scoring at least `min_score` (and finite) are usable;
    /// scores are clamped to `0..=1`. `None` unless there are exactly 26
    /// keypoints.
    pub fn from_keypoints(keypoints: &[Keypoint], handed: Handed, min_score: f32) -> Option<Self> {
        if keypoints.len() != Skeleton::Halpe26.len() {
            return None;
        }
        let mut pts = [None; 26];
        let mut scores = [0.0; 26];
        for (i, k) in keypoints.iter().enumerate() {
            if k.score >= min_score && k.x.is_finite() && k.y.is_finite() {
                pts[i] = Some(Pt::new(f64::from(k.x), f64::from(k.y)));
                scores[i] = k.score.min(1.0);
            }
        }
        Some(Self {
            pts,
            scores,
            handed,
        })
    }

    pub fn at(&self, j: Joint) -> Option<Pt> {
        self.pts[j.index(self.handed)]
    }

    pub fn score(&self, j: Joint) -> f32 {
        self.scores[j.index(self.handed)]
    }

    fn pair(&self, a: Joint, b: Joint) -> Option<Pt> {
        Some(self.at(a)?.mid(self.at(b)?))
    }

    pub fn shoulders(&self) -> Option<Pt> {
        self.pair(Joint::LeadShoulder, Joint::TrailShoulder)
    }

    /// The pelvis center: the midpoint of the hips.
    pub fn pelvis(&self) -> Option<Pt> {
        self.pair(Joint::LeadHip, Joint::TrailHip)
    }

    /// The hands: the midpoint of the wrists, or the one wrist found.
    pub fn hands(&self) -> Option<Pt> {
        match (self.at(Joint::LeadWrist), self.at(Joint::TrailWrist)) {
            (Some(a), Some(b)) => Some(a.mid(b)),
            (a, b) => a.or(b),
        }
    }

    /// The head: the head-top keypoint, else the nose.
    pub fn head(&self) -> Option<Pt> {
        self.at(Joint::Head).or_else(|| self.at(Joint::Nose))
    }

    /// The torso length: shoulder midpoint to hip midpoint, pixels.
    pub fn torso(&self) -> Option<f64> {
        let d = self.shoulders()?.dist(self.pelvis()?);
        (d > 1e-6).then_some(d)
    }

    pub fn width(&self, a: Joint, b: Joint) -> Option<f64> {
        Some(self.at(a)?.dist(self.at(b)?))
    }

    /// The angle at `b` in the chain `a`–`b`–`c`.
    pub fn angle(&self, a: Joint, b: Joint, c: Joint) -> Option<f64> {
        angle_at(self.at(a)?, self.at(b)?, self.at(c)?)
    }

    /// Spine (pelvis → shoulder midpoint) tilt from vertical toward `sign`.
    pub fn spine_tilt(&self, sign: f64) -> Option<f64> {
        tilt_from_vertical(self.pelvis()?, self.shoulders()?, sign)
    }

    /// Shoulder line from horizontal, positive when the lead shoulder is
    /// higher; ±90°.
    pub fn shoulder_tilt(&self) -> Option<f64> {
        let (lead, trail) = (
            self.at(Joint::LeadShoulder)?,
            self.at(Joint::TrailShoulder)?,
        );
        let (rise, run) = (trail.y - lead.y, (lead.x - trail.x).abs());
        (rise.hypot(run) > 1e-9).then(|| rise.atan2(run).to_degrees())
    }

    /// Knee flexion: 180° minus the hip–knee–ankle angle (0 = straight).
    pub fn knee_flex(&self, lead: bool) -> Option<f64> {
        let (h, k, a) = if lead {
            (Joint::LeadHip, Joint::LeadKnee, Joint::LeadAnkle)
        } else {
            (Joint::TrailHip, Joint::TrailKnee, Joint::TrailAnkle)
        };
        self.angle(h, k, a).map(|v| 180.0 - v)
    }

    /// Elbow angle: shoulder–elbow–wrist (180° = straight).
    pub fn elbow(&self, lead: bool) -> Option<f64> {
        if lead {
            self.angle(Joint::LeadShoulder, Joint::LeadElbow, Joint::LeadWrist)
        } else {
            self.angle(Joint::TrailShoulder, Joint::TrailElbow, Joint::TrailWrist)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lead_and_trail_mirror_by_handedness() {
        let s = Skeleton::Halpe26;
        assert_eq!(
            s.names()[Joint::LeadShoulder.index(Handed::Right)],
            "left_shoulder"
        );
        assert_eq!(
            s.names()[Joint::LeadShoulder.index(Handed::Left)],
            "right_shoulder"
        );
        assert_eq!(
            s.names()[Joint::TrailHeel.index(Handed::Right)],
            "right_heel"
        );
        assert_eq!(s.names()[Joint::TrailHeel.index(Handed::Left)], "left_heel");
        assert_eq!(s.names()[Joint::Head.index(Handed::Left)], "head");
        assert_eq!(s.names()[Joint::Hip.index(Handed::Right)], "hip");
    }

    #[test]
    fn directions_follow_view_and_hand() {
        let fo = |handed| Setup {
            view: View::FaceOn,
            handed,
        };
        let dtl = |handed| Setup {
            view: View::DownTheLine,
            handed,
        };
        assert_eq!(fo(Handed::Right).target_x(), Some(1.0));
        assert_eq!(fo(Handed::Left).target_x(), Some(-1.0));
        assert_eq!(fo(Handed::Right).ball_x(), None);
        assert_eq!(dtl(Handed::Right).ball_x(), Some(1.0));
        assert_eq!(dtl(Handed::Left).ball_x(), Some(-1.0));
        assert_eq!(dtl(Handed::Left).target_x(), None);
    }

    #[test]
    fn angles() {
        let o = Pt::new(0.0, 0.0);
        let near = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 1e-9;
        assert!(near(
            angle_at(Pt::new(1.0, 0.0), o, Pt::new(0.0, 1.0)),
            90.0
        ));
        assert!(near(
            angle_at(Pt::new(1.0, 0.0), o, Pt::new(-1.0, 0.0)),
            180.0
        ));
        assert!(angle_at(o, o, Pt::new(1.0, 0.0)).is_none());
        // Up in the image is −y.
        assert!(near(tilt_from_vertical(o, Pt::new(0.0, -1.0), 1.0), 0.0));
        assert!(near(tilt_from_vertical(o, Pt::new(1.0, -1.0), 1.0), 45.0));
        assert!(near(tilt_from_vertical(o, Pt::new(1.0, -1.0), -1.0), -45.0));
    }
}
