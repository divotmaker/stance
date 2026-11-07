//! Per-view body metrics at the swing's key positions.

use serde::{Deserialize, Deserializer, Serialize};

use super::body::{Body, Joint, Setup, View};
use super::smooth::Frames;

/// A body metric. Every metric is a 2D proxy measured in the image plane of
/// one camera, not a 3D joint angle; each needs the view it names.
///
/// Notation: image `x` grows to the right, `y` downwards. *Pelvis* is the
/// hips' midpoint, *shoulders* the shoulders' midpoint, *spine* the line
/// pelvis → shoulders, *hands* the wrists' midpoint, *head* the head-top
/// keypoint (the nose when it is missing). `fwd` is +1 or −1: towards the
/// target face-on, towards the ball down the line ([`Setup::forward_x`]).
/// `L` is the torso length (pelvis to shoulders) at address in pixels:
/// distances are fractions of it ([`Unit::Torso`]) and, given the golfer's
/// height, centimetres. Shifts are measured from address. *P3* and *P5*
/// are the lead arm parallel to the ground in the backswing and the
/// downswing ([`SwingEvents::arm_parallel_back`](super::SwingEvents::arm_parallel_back),
/// [`SwingEvents::arm_parallel_down`](super::SwingEvents::arm_parallel_down)).
///
/// | Metric | View | Unit | Positions | Formula |
/// |---|---|---|---|---|
/// | `hip_sway` | face-on | torso | P3, top, P5, impact | pelvis shift towards the target (negative: away from it): `fwd·(pelvis.x − pelvis₀.x) / L` |
/// | `chest_sway` | face-on | torso | P3, top, P5, impact | shoulders' shift towards the target (negative: away from it): `fwd·(shoulders.x − shoulders₀.x) / L` |
/// | `sway_gap` | face-on | torso | address, P3, top, P5, impact | shoulders ahead of the pelvis towards the target: `fwd·(shoulders.x − pelvis.x) / L`; negative with the chest behind the pelvis, as at the top and impact of most full swings |
/// | `head_lateral` | face-on | torso | P3, top, P5, impact | head shift towards the target: `fwd·(head.x − head₀.x) / L` |
/// | `head_vertical` | either | torso | top, impact | head rise: `(head₀.y − head.y) / L` |
/// | `pelvis_lift` | either | torso | top, impact | pelvis rise (negative: it dropped): `(pelvis₀.y − pelvis.y) / L` |
/// | `lateral_spine_tilt` | face-on | deg | address, P3, top, P5, impact, finish | spine from vertical, positive with the shoulders away from the target; at the finish, where the golfer faces the target and the camera sees them side-on, it is the bend back (the "reverse C") |
/// | `shoulder_tilt` | face-on | deg | address, top, impact | shoulder-line tilt φ from the line solve below, positive with the lead shoulder higher (so negative at the top); at address, the line's angle from horizontal. The top value is beta (see the limits below) |
/// | `hip_tilt` | face-on | deg | address, top, impact | hip-line tilt φ from the line solve below, positive with the lead hip higher; at address, the line's angle from horizontal |
/// | `lead_arm_angle` | face-on | deg | address, top, impact | lead shoulder–elbow–wrist angle (180 = straight) |
/// | `stance_width` | face-on | ratio | address | ankle-to-ankle distance / shoulder-to-shoulder distance |
/// | `hip_width_proxy` | face-on | ratio | top, impact, finish | hip-to-hip distance / its address value: a pelvis-turn proxy, ≈ cos of the turn only if the hips are square to the camera at address, level, and the hip keypoints stay on the pelvis edges; absent when above 1 (not measurable: a turn can't widen the hips) |
/// | `hip_turn` | face-on | deg | P3, top, P5, impact | pelvis-turn estimate θ from the line solve below on the hip line |
/// | `shoulder_turn` | face-on | deg | P3, top, P5, impact | shoulder-turn estimate θ from the line solve below on the shoulder line |
/// | `spine_forward_bend` | down the line | deg | address, P3, top, P5, impact | spine from vertical, positive leaning towards the ball |
/// | `loss_of_posture` | down the line | deg | P3, top, P5, impact | `spine_forward_bend` at address minus now: positive = more upright |
/// | `early_extension` | down the line | torso | P3, top, P5, impact | pelvis shift towards the ball: `fwd·(pelvis.x − pelvis₀.x) / L` |
/// | `lead_knee_flex` | down the line | deg | address, top, impact, finish | 180 − lead hip–knee–ankle angle (0 = straight) |
/// | `trail_knee_flex` | down the line | deg | address, top, impact, finish | 180 − trail hip–knee–ankle angle |
/// | `trail_elbow_angle` | down the line | deg | address, top, impact | trail shoulder–elbow–wrist angle |
/// | `hand_height` | down the line | torso | top | hands above the trail shoulder: `(trail_shoulder.y − hands.y) / L` |
/// | `hand_depth` | down the line | torso | top | hands behind the trail shoulder, away from the ball (negative: nearer the ball than it): `−fwd·(hands.x − trail_shoulder.x) / L` |
///
/// Face-on, the golfer bends towards the camera, so `L` is foreshortened
/// by the cosine of the forward bend (about 0.82 for a typical 35° bend)
/// and face-on fractions read about 20 % larger than the same distance down
/// the line; centimetres don't have this bias (see [`BodyScale`]). Relative
/// metrics (shifts and changes from address) take their confidence from the
/// keypoints at address too. Values are rounded to 4 decimal places of
/// their unit.
///
/// # The line solve: tilt and turn from one camera
///
/// `hip_tilt`, `hip_turn`, `shoulder_tilt` and `shoulder_turn` come from
/// one solve per body line (hip to hip, shoulder to shoulder). A line of
/// true length `w₀`, tilted by φ (lead end higher) and turned by θ about
/// the vertical, shows face-on as `dx = w₀ cos φ cos θ` across and
/// `dy = w₀ sin φ` up, so
///
/// ```text
/// φ = asin(dy / w₀)
/// θ = acos(dx / (w₀ cos φ))
/// ```
///
/// with `w₀` the line's length at address (where it is taken as square to
/// the camera, so θ = 0 and φ is the line's angle from horizontal). Where
/// the line shows longer than `w₀` (`hypot(dx, dy) > w₀`: keypoint noise,
/// a body that wasn't square at address, a golfer nearer the camera) the
/// turn is absent and the tilt is the line's angle from horizontal, as for
/// a square line (the two meet where the line shows exactly `w₀`). Without
/// an address, the frame's own line stands in for it: the tilt is then the
/// line's angle from horizontal and there is no turn. The solve runs at
/// every key position and every frame, address and impact included, so a
/// tilt is reported wherever both keypoints of its line are found.
///
/// For any line of length `w₀` at elevation φ and azimuth θ, a face-on
/// camera far away sees exactly `dx = w₀ cos φ cos θ` and `dy = w₀ sin φ`,
/// so the solve's φ is the line's true angle to the horizontal: what a
/// sensor's tilt or a 3D model's side bend reports. The lead shoulder
/// dropping as the bent-over golfer turns is part of that angle, not an
/// error of the solve. What does limit it, in order:
///
/// 1. **Perspective.** The solve assumes a distant camera. For a golfer
///    centred in the picture of a 55° lens at chest height, a 45° hip turn
///    reads about 0.5° high at 5 m and 1° high at 2 m (a portrait camera in
///    a small room), the pelvis midpoint drifts with the turn by under 1 cm
///    at 2 m (0.3 cm at 5 m), and a −40° shoulder tilt at the top reads
///    about 2° less at 5 m and 5° less at 2 m. A body off the camera's axis
///    by `x` at distance `D` is seen turned by about `x / D` radians, so a
///    10 cm slide at 2 m moves a turn by about 3°.
/// 2. **Keypoint drift and saturation.** The far hip or shoulder is hidden
///    as the body turns and the model places it, and near 90° `acos` is
///    steep, so a few pixels move the turn by many degrees; a turn past 90°
///    reads as less, which is why neither turn is reported at the finish.
///    On real swings the shoulder tilt at the top reads less lead-side-down
///    than published tour averages (the hidden trail shoulder, and item 1
///    up close): treat it as beta.
/// 3. **Camera roll.** Not modelled: a camera rolled by 2° adds 2° to every
///    tilt, address included. Level the camera; a non-zero address
///    shoulder tilt where you expect level is roll.
/// 4. **The address is taken as square and level in the image plane.**
///    Hips or shoulders 10° open at address make `w₀` 1.5 % short, so the
///    turn reads about 0 for small turns and up to about 10° low later.
/// 5. **Depth and direction.** A line seen end-on still shows the body's
///    thickness, the golfer is assumed to stay at the same distance, and
///    one camera can't tell which way the line turned: a turn back past
///    square reads as a turn.
///
/// X-factor (shoulder minus hip turn) and the kinematic sequence need 3D
/// (two calibrated views) and are not measured.
///
/// [`norms`](super::norms) grades the metrics against published tour and
/// amateur averages. Its turns are signed (positive open), so a one-camera
/// `hip_turn` or `shoulder_turn`, an unsigned size, takes the sign of the
/// band's centre and is graded at impact only.
///
/// [`SwingReport::series`](super::SwingReport::series) holds every
/// measured metric's value at every frame, with the same formulas and
/// keypoints; shifts, the line solve and the turns are against the address
/// frame there too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[allow(missing_docs, reason = "documented in the table above")]
pub enum MetricId {
    HipSway,
    ChestSway,
    SwayGap,
    HeadLateral,
    HeadVertical,
    PelvisLift,
    LateralSpineTilt,
    ShoulderTilt,
    HipTilt,
    LeadArmAngle,
    StanceWidth,
    HipWidthProxy,
    HipTurn,
    ShoulderTurn,
    SpineForwardBend,
    LossOfPosture,
    EarlyExtension,
    LeadKneeFlex,
    TrailKneeFlex,
    TrailElbowAngle,
    HandHeight,
    HandDepth,
}

/// A metric's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    /// Degrees in the image plane.
    Degrees,
    /// Fractions of the torso length at address (with centimetres in
    /// [`Metric::cm`] when the golfer's height is known).
    Torso,
    /// A dimensionless ratio of two image distances.
    Ratio,
}

impl MetricId {
    /// Every metric.
    pub const ALL: &'static [Self] = &[
        Self::HipSway,
        Self::ChestSway,
        Self::SwayGap,
        Self::HeadLateral,
        Self::HeadVertical,
        Self::PelvisLift,
        Self::LateralSpineTilt,
        Self::ShoulderTilt,
        Self::HipTilt,
        Self::LeadArmAngle,
        Self::StanceWidth,
        Self::HipWidthProxy,
        Self::HipTurn,
        Self::ShoulderTurn,
        Self::SpineForwardBend,
        Self::LossOfPosture,
        Self::EarlyExtension,
        Self::LeadKneeFlex,
        Self::TrailKneeFlex,
        Self::TrailElbowAngle,
        Self::HandHeight,
        Self::HandDepth,
    ];

    /// The view the metric needs (`None`: either).
    #[must_use]
    pub fn view(self) -> Option<View> {
        match self {
            Self::HeadVertical | Self::PelvisLift => None,
            Self::HipSway
            | Self::ChestSway
            | Self::SwayGap
            | Self::HeadLateral
            | Self::LateralSpineTilt
            | Self::ShoulderTilt
            | Self::HipTilt
            | Self::LeadArmAngle
            | Self::StanceWidth
            | Self::HipWidthProxy
            | Self::HipTurn
            | Self::ShoulderTurn => Some(View::FaceOn),
            Self::SpineForwardBend
            | Self::LossOfPosture
            | Self::EarlyExtension
            | Self::LeadKneeFlex
            | Self::TrailKneeFlex
            | Self::TrailElbowAngle
            | Self::HandHeight
            | Self::HandDepth => Some(View::DownTheLine),
        }
    }

    /// The metric's unit.
    #[must_use]
    pub fn unit(self) -> Unit {
        match self {
            Self::HipSway
            | Self::ChestSway
            | Self::SwayGap
            | Self::HeadLateral
            | Self::HeadVertical
            | Self::PelvisLift
            | Self::EarlyExtension
            | Self::HandHeight
            | Self::HandDepth => Unit::Torso,
            Self::StanceWidth | Self::HipWidthProxy => Unit::Ratio,
            Self::LateralSpineTilt
            | Self::ShoulderTilt
            | Self::HipTilt
            | Self::LeadArmAngle
            | Self::HipTurn
            | Self::ShoulderTurn
            | Self::SpineForwardBend
            | Self::LossOfPosture
            | Self::LeadKneeFlex
            | Self::TrailKneeFlex
            | Self::TrailElbowAngle => Unit::Degrees,
        }
    }

    /// Whether the metric applies to a view.
    #[must_use]
    pub fn applies_to(self, view: View) -> bool {
        self.view().is_none_or(|v| v == view)
    }

    /// The metric whose bands ([`norms`](super::norms)) grade this one:
    /// every metric grades against its own rows.
    #[must_use]
    pub const fn band_key(self) -> Self {
        self
    }
}

/// A metric's value at each key position (absent where it doesn't apply or
/// the keypoints it needs weren't found).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[allow(missing_docs, reason = "the swing positions")]
#[non_exhaustive]
pub struct Positions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<f64>,
    /// P3: the lead arm parallel to the ground in the backswing
    /// ([`SwingEvents::arm_parallel_back`](super::SwingEvents::arm_parallel_back)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arm_parallel_back: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top: Option<f64>,
    /// P5: the lead arm parallel to the ground in the downswing
    /// ([`SwingEvents::arm_parallel_down`](super::SwingEvents::arm_parallel_down)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arm_parallel_down: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impact: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<f64>,
}

impl Positions {
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            address: self.address.map(&f),
            arm_parallel_back: self.arm_parallel_back.map(&f),
            top: self.top.map(&f),
            arm_parallel_down: self.arm_parallel_down.map(&f),
            impact: self.impact.map(&f),
            finish: self.finish.map(&f),
        }
    }

    fn is_empty(&self) -> bool {
        self.address.is_none()
            && self.arm_parallel_back.is_none()
            && self.top.is_none()
            && self.arm_parallel_down.is_none()
            && self.impact.is_none()
            && self.finish.is_none()
    }
}

/// One measured metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Metric {
    /// Which metric (definitions on [`MetricId`]).
    pub id: MetricId,
    /// The view it needs (`None`: either).
    pub view: Option<View>,
    /// The unit of [`Metric::values`].
    pub unit: Unit,
    /// The value at each key position.
    pub values: Positions,
    /// For [`Unit::Torso`] metrics with a known golfer height: the values in
    /// centimetres.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cm: Option<Positions>,
    /// `0..=1`: the mean score of the keypoints the values use (at the
    /// measured frames, and at address for a relative metric).
    pub confidence: f32,
}

/// One metric's value at every frame of the smoothed track, aligned to
/// [`Series::t_ns`]: `None` where the keypoints it needs weren't found, or
/// (for a metric measured from address) where there is no address.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MetricSeries {
    /// Which metric (definitions on [`MetricId`]).
    pub id: MetricId,
    /// The unit of [`MetricSeries::values`]. For [`Unit::Torso`],
    /// [`BodyScale::to_cm`] gives centimetres, as in [`Metric::cm`].
    pub unit: Unit,
    /// The value at each frame.
    pub values: Vec<Option<f64>>,
}

/// Every measured metric at every frame: the swing as a time series, for a
/// display that follows a replay frame by frame.
///
/// The values come from the same formulas and keypoints as
/// [`Metric::values`], so at an event's frame a series holds that
/// metric's value at the position (where the position has one). A metric
/// with no value at any frame is left out; with no metric, the series is
/// empty (and left out of the JSON).
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[non_exhaustive]
pub struct Series {
    /// The frame times on the track's time base (ns from the clip's first
    /// frame), one per track frame.
    pub t_ns: Vec<u64>,
    /// The view's metrics, in [`MetricId::ALL`] order, each with one value
    /// per entry of [`Series::t_ns`].
    pub metrics: Vec<MetricSeries>,
}

impl Series {
    /// Whether no metric has a series.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// A metric's series by id.
    #[must_use]
    pub fn metric(&self, id: MetricId) -> Option<&MetricSeries> {
        self.metrics.iter().find(|m| m.id == id)
    }

    /// The index of the frame nearest `t_ns` (ties go to the earlier
    /// frame); `None` for an empty series.
    #[must_use]
    pub fn frame_at(&self, t_ns: u64) -> Option<usize> {
        let i = self.t_ns.partition_point(|&t| t < t_ns);
        match (i.checked_sub(1), (i < self.t_ns.len()).then_some(i)) {
            (Some(a), Some(b)) => Some(if t_ns - self.t_ns[a] <= self.t_ns[b] - t_ns {
                a
            } else {
                b
            }),
            (a, b) => a.or(b),
        }
    }

    /// A metric's value at the frame nearest `t_ns`.
    #[must_use]
    pub fn value_at(&self, id: MetricId, t_ns: u64) -> Option<f64> {
        let i = self.frame_at(t_ns)?;
        self.metric(id)?.values.get(i).copied().flatten()
    }
}

/// The series as stored: metrics with an unknown id are skipped (written by
/// a later version); a known metric whose length differs from the frame
/// times is an error.
impl<'de> Deserialize<'de> for Series {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            t_ns: Vec<u64>,
            #[serde(default)]
            metrics: Vec<serde_json::Value>,
        }
        let raw = Raw::deserialize(d)?;
        let mut metrics = Vec::with_capacity(raw.metrics.len());
        for value in raw.metrics {
            let known = value
                .get("id")
                .is_some_and(|id| MetricId::deserialize(id).is_ok());
            if !known {
                continue;
            }
            let m = MetricSeries::deserialize(value).map_err(serde::de::Error::custom)?;
            if m.values.len() != raw.t_ns.len() {
                return Err(serde::de::Error::custom(format!(
                    "series {:?} has {} values for {} frames",
                    m.id,
                    m.values.len(),
                    raw.t_ns.len()
                )));
            }
            metrics.push(m);
        }
        Ok(Self {
            t_ns: raw.t_ns,
            metrics,
        })
    }
}

/// Torso length per unit of body height: shoulder to hip joint height in
/// the anthropometric segment table of Drillis and Contini (as tabulated by
/// Winter, *Biomechanics and Motor Control of Human Movement*): 0.818 H −
/// 0.530 H.
pub const TORSO_PER_HEIGHT: f64 = 0.288;

/// The body scale distances are measured in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BodyScale {
    /// Torso length at address (pelvis to shoulders midpoint), pixels: the
    /// unit of [`Unit::Torso`].
    pub torso_px: f64,
    /// The upright torso, pixels: the torso's longest 2D length in the clip
    /// (the 90th percentile, and at least [`BodyScale::torso_px`]; the
    /// golfer stands upright before address and at the finish, where the
    /// torso isn't foreshortened). [`BodyScale::cm_per_px`] is matched to
    /// it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upright_px: Option<f64>,
    /// Centimetres per pixel, from the golfer's height: the upright torso
    /// ([`BodyScale::upright_px`]) is taken as [`TORSO_PER_HEIGHT`] ×
    /// height.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cm_per_px: Option<f64>,
}

impl BodyScale {
    /// A [`Unit::Torso`] value in centimetres, when the golfer's height is
    /// known.
    #[must_use]
    pub fn to_cm(&self, torso: f64) -> Option<f64> {
        self.cm_per_px.map(|c| torso * self.torso_px * c)
    }

    pub(crate) fn measure(
        frames: &Frames,
        address: Option<usize>,
        height_cm: Option<f64>,
    ) -> Option<Self> {
        let mut torsos: Vec<f64> = frames.bodies.iter().filter_map(Body::torso).collect();
        if torsos.is_empty() {
            return None;
        }
        torsos.sort_by(f64::total_cmp);
        let torso_px = address
            .and_then(|a| frames.bodies[a].torso())
            .unwrap_or(torsos[torsos.len() / 2]);
        let upright = torsos[(torsos.len() * 9 / 10).min(torsos.len() - 1)].max(torso_px);
        let cm_per_px = height_cm
            .filter(|h| h.is_finite() && *h > 0.0)
            .map(|h| TORSO_PER_HEIGHT * h / upright);
        Some(Self {
            torso_px,
            upright_px: Some(upright),
            cm_per_px,
        })
    }
}

/// The frames of the key positions.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KeyFrames {
    pub address: Option<usize>,
    pub arm_parallel_back: Option<usize>,
    pub top: Option<usize>,
    pub arm_parallel_down: Option<usize>,
    pub impact: Option<usize>,
    pub finish: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pos {
    Address,
    ArmParallelBack,
    Top,
    ArmParallelDown,
    Impact,
    Finish,
}

const ADDRESS_TOP_IMPACT: &[Pos] = &[Pos::Address, Pos::Top, Pos::Impact];
const ADDRESS_TOP_IMPACT_FINISH: &[Pos] = &[Pos::Address, Pos::Top, Pos::Impact, Pos::Finish];
const TOP_IMPACT: &[Pos] = &[Pos::Top, Pos::Impact];
/// P3, top, P5 and impact.
const SWING: &[Pos] = &[
    Pos::ArmParallelBack,
    Pos::Top,
    Pos::ArmParallelDown,
    Pos::Impact,
];
/// Address, P3, top, P5 and impact.
const ADDRESS_TO_IMPACT: &[Pos] = &[
    Pos::Address,
    Pos::ArmParallelBack,
    Pos::Top,
    Pos::ArmParallelDown,
    Pos::Impact,
];
const EVERY_POSITION: &[Pos] = &[
    Pos::Address,
    Pos::ArmParallelBack,
    Pos::Top,
    Pos::ArmParallelDown,
    Pos::Impact,
    Pos::Finish,
];

const PELVIS: &[Joint] = &[Joint::LeadHip, Joint::TrailHip];
const SHOULDERS: &[Joint] = &[Joint::LeadShoulder, Joint::TrailShoulder];
const SPINE: &[Joint] = &[
    Joint::LeadHip,
    Joint::TrailHip,
    Joint::LeadShoulder,
    Joint::TrailShoulder,
];
const HEAD: &[Joint] = &[Joint::Head];
const HAND_AND_TRAIL_SHOULDER: &[Joint] =
    &[Joint::TrailShoulder, Joint::LeadWrist, Joint::TrailWrist];

/// Metric values are rounded to 4 decimal places: far below keypoint
/// noise, and it keeps a series short in JSON.
const DECIMALS: f64 = 1e4;

/// A metric's formula: `f(now, address)` (the frame itself stands in for
/// the address pose of a metric that doesn't use it).
type Formula<'a> = Box<dyn Fn(&Body, &Body) -> Option<f64> + 'a>;

/// One metric's definition, shared by its key positions and its series.
struct Def<'a> {
    id: MetricId,
    /// Where it is reported.
    positions: &'static [Pos],
    /// The keypoints its confidence comes from.
    joints: &'static [Joint],
    /// Measured from the address pose: none without an address, never at
    /// address, and the address frame's scores count towards the
    /// confidence.
    relative: bool,
    f: Formula<'a>,
}

impl<'a> Def<'a> {
    fn metric(
        id: MetricId,
        positions: &'static [Pos],
        joints: &'static [Joint],
        f: impl Fn(&Body, &Body) -> Option<f64> + 'a,
    ) -> Self {
        Self {
            id,
            positions,
            joints,
            relative: false,
            f: Box::new(f),
        }
    }

    fn relative(
        id: MetricId,
        positions: &'static [Pos],
        joints: &'static [Joint],
        f: impl Fn(&Body, &Body) -> Option<f64> + 'a,
    ) -> Self {
        Self {
            relative: true,
            ..Self::metric(id, positions, joints, f)
        }
    }
}

struct Ctx<'a> {
    frames: &'a Frames,
    keys: KeyFrames,
    scale: BodyScale,
}

impl Ctx<'_> {
    fn frame(&self, p: Pos) -> Option<usize> {
        match p {
            Pos::Address => self.keys.address,
            Pos::ArmParallelBack => self.keys.arm_parallel_back,
            Pos::Top => self.keys.top,
            Pos::ArmParallelDown => self.keys.arm_parallel_down,
            Pos::Impact => self.keys.impact,
            Pos::Finish => self.keys.finish,
        }
    }

    fn address(&self) -> Option<&Body> {
        self.keys.address.map(|a| &self.frames.bodies[a])
    }

    /// The definition's formula on frame `i`, rounded to [`DECIMALS`];
    /// `None` for a relative metric without an address, and for a
    /// non-finite value.
    fn value(&self, def: &Def, i: usize) -> Option<f64> {
        let address = self.address();
        if def.relative && address.is_none() {
            return None;
        }
        let body = &self.frames.bodies[i];
        (def.f)(body, address.unwrap_or(body))
            .filter(|v| v.is_finite())
            // `+ 0.0` turns a negative zero (no shift at address) into 0.
            .map(|v| (v * DECIMALS).round() / DECIMALS + 0.0)
    }

    /// The metric at its positions, with the mean score of its joints at
    /// the frames used as its confidence.
    fn measure(&self, def: &Def) -> Option<Metric> {
        let address = self.address();
        let mut values = Positions::default();
        let (mut score_sum, mut score_n) = (0.0_f64, 0_u32);
        for &p in def.positions {
            if def.relative && p == Pos::Address {
                continue;
            }
            let Some(i) = self.frame(p) else { continue };
            let Some(v) = self.value(def, i) else {
                continue;
            };
            let slot = match p {
                Pos::Address => &mut values.address,
                Pos::ArmParallelBack => &mut values.arm_parallel_back,
                Pos::Top => &mut values.top,
                Pos::ArmParallelDown => &mut values.arm_parallel_down,
                Pos::Impact => &mut values.impact,
                Pos::Finish => &mut values.finish,
            };
            *slot = Some(v);
            let body = &self.frames.bodies[i];
            for &j in def.joints {
                score_sum += f64::from(body.score(j));
                score_n += 1;
                if def.relative
                    && let Some(a) = address
                {
                    score_sum += f64::from(a.score(j));
                    score_n += 1;
                }
            }
        }
        if values.is_empty() {
            return None;
        }
        let unit = def.id.unit();
        let cm = match (unit, self.scale.cm_per_px) {
            (Unit::Torso, Some(c)) => Some(values.map(|v| v * self.scale.torso_px * c)),
            _ => None,
        };
        // A mean of 0..=1 scores.
        #[allow(clippy::cast_possible_truncation, reason = "a 0..=1 value")]
        let confidence = (score_sum / f64::from(score_n.max(1))) as f32;
        Some(Metric {
            id: def.id,
            view: def.id.view(),
            unit,
            values,
            cm,
            confidence,
        })
    }

    /// The metric at every frame; `None` when it has no value at any.
    fn series(&self, def: &Def) -> Option<MetricSeries> {
        let values: Vec<Option<f64>> = (0..self.frames.len()).map(|i| self.value(def, i)).collect();
        values.iter().any(Option::is_some).then(|| MetricSeries {
            id: def.id,
            unit: def.id.unit(),
            values,
        })
    }

    /// A point's shift from address along x towards `sign`, in torso lengths.
    fn shift_x(
        &self,
        now: Option<super::body::Pt>,
        then: Option<super::body::Pt>,
        sign: f64,
    ) -> Option<f64> {
        Some(sign * (now?.x - then?.x) / self.scale.torso_px)
    }
}

/// Measures every metric of the setup's view at the key positions and at
/// every frame; with the view unknown, the metrics that work in either
/// view.
pub(crate) fn measure(
    frames: &Frames,
    keys: KeyFrames,
    setup: Option<Setup>,
    scale: BodyScale,
) -> (Vec<Metric>, Series) {
    let ctx = Ctx {
        frames,
        keys,
        scale,
    };
    let defs = match setup {
        Some(s) if s.view == View::FaceOn => face_on(&ctx, s.forward_x()),
        Some(s) => down_the_line(&ctx, s.forward_x()),
        None => vec![head_vertical(&ctx), pelvis_lift(&ctx)],
    };
    let metrics = defs.iter().filter_map(|d| ctx.measure(d)).collect();
    let series: Vec<MetricSeries> = defs.iter().filter_map(|d| ctx.series(d)).collect();
    let series = Series {
        t_ns: if series.is_empty() {
            Vec::new()
        } else {
            frames.t_ns.clone()
        },
        metrics: series,
    };
    (metrics, series)
}

fn head_vertical<'a>(ctx: &'a Ctx) -> Def<'a> {
    let l = ctx.scale.torso_px;
    Def::relative(MetricId::HeadVertical, TOP_IMPACT, HEAD, move |b, a| {
        Some((a.head()?.y - b.head()?.y) / l)
    })
}

fn pelvis_lift<'a>(ctx: &'a Ctx) -> Def<'a> {
    let l = ctx.scale.torso_px;
    Def::relative(MetricId::PelvisLift, TOP_IMPACT, PELVIS, move |b, a| {
        Some((a.pelvis()?.y - b.pelvis()?.y) / l)
    })
}

/// A width ratio `a`–`b` over `c`–`d`.
fn ratio(body: &Body, (a, b): (Joint, Joint), over: &Body, (c, d): (Joint, Joint)) -> Option<f64> {
    let base = over.width(c, d)?;
    (base > 1e-6)
        .then(|| body.width(a, b).map(|w| w / base))
        .flatten()
}

/// How far (as a fraction) a line may show longer than its address length
/// before the line solve gives up: rounding only, so the address frame
/// itself always solves.
const LINE_SLACK: f64 = 1e-9;

/// The line solve ([`MetricId`]): the tilt φ and turn θ (degrees) of a line
/// of true length `w0` that shows `dx` across (either sign) and `dy` up
/// (positive with the lead end higher). `None` when `w0` has no length or
/// an input isn't finite.
///
/// A line showing longer than `w0` (`hypot(dx, dy) > w0`, the same as
/// `|dy| > w0` or `dx > w0 cos φ`) has no turn, and its tilt is its angle
/// from horizontal: the solve for a square line of its own length, which
/// meets the solve where the line shows exactly `w0`. A vertical line has
/// no turn either.
pub(crate) fn solve_line(dx: f64, dy: f64, w0: f64) -> Option<(f64, Option<f64>)> {
    if !(w0 >= 1e-6 && w0.is_finite() && dx.is_finite() && dy.is_finite()) {
        return None;
    }
    let dx = dx.abs();
    if dx.hypot(dy) > w0 * (1.0 + LINE_SLACK) {
        return Some((dy.atan2(dx).to_degrees(), None));
    }
    let tilt = (dy / w0).clamp(-1.0, 1.0).asin();
    let base = w0 * tilt.cos();
    let turn = (base >= 1e-6).then(|| (dx / base).min(1.0).acos().to_degrees());
    Some((tilt.to_degrees(), turn))
}

/// The line solve on the body line `pair` (lead, trail) against its length
/// at address: the tilt and, where measurable, the turn, in degrees.
fn line_pose(
    body: &Body,
    address: &Body,
    (lead, trail): (Joint, Joint),
) -> Option<(f64, Option<f64>)> {
    let w0 = address.width(lead, trail)?;
    let (l, t) = (body.at(lead)?, body.at(trail)?);
    solve_line(l.x - t.x, t.y - l.y, w0)
}

fn face_on<'a>(ctx: &'a Ctx, fwd: f64) -> Vec<Def<'a>> {
    const HIPS: (Joint, Joint) = (Joint::LeadHip, Joint::TrailHip);
    const SHOULDER_LINE: (Joint, Joint) = (Joint::LeadShoulder, Joint::TrailShoulder);
    let l = ctx.scale.torso_px;
    vec![
        Def::relative(MetricId::HipSway, SWING, PELVIS, move |b, a| {
            ctx.shift_x(b.pelvis(), a.pelvis(), fwd)
        }),
        Def::relative(MetricId::ChestSway, SWING, SHOULDERS, move |b, a| {
            ctx.shift_x(b.shoulders(), a.shoulders(), fwd)
        }),
        Def::metric(MetricId::SwayGap, ADDRESS_TO_IMPACT, SPINE, move |b, _| {
            Some(fwd * (b.shoulders()?.x - b.pelvis()?.x) / l)
        }),
        Def::relative(MetricId::HeadLateral, SWING, HEAD, move |b, a| {
            ctx.shift_x(b.head(), a.head(), fwd)
        }),
        head_vertical(ctx),
        pelvis_lift(ctx),
        Def::metric(
            MetricId::LateralSpineTilt,
            EVERY_POSITION,
            SPINE,
            move |b, _| b.spine_tilt(-fwd),
        ),
        Def::metric(
            MetricId::ShoulderTilt,
            ADDRESS_TOP_IMPACT,
            SHOULDERS,
            |b, a| line_pose(b, a, SHOULDER_LINE).map(|(tilt, _)| tilt),
        ),
        Def::metric(MetricId::HipTilt, ADDRESS_TOP_IMPACT, PELVIS, |b, a| {
            line_pose(b, a, HIPS).map(|(tilt, _)| tilt)
        }),
        Def::metric(
            MetricId::LeadArmAngle,
            ADDRESS_TOP_IMPACT,
            &[Joint::LeadShoulder, Joint::LeadElbow, Joint::LeadWrist],
            |b, _| b.elbow(true),
        ),
        Def::metric(
            MetricId::StanceWidth,
            &[Pos::Address],
            &[
                Joint::LeadAnkle,
                Joint::TrailAnkle,
                Joint::LeadShoulder,
                Joint::TrailShoulder,
            ],
            |b, _| ratio(b, (Joint::LeadAnkle, Joint::TrailAnkle), b, SHOULDER_LINE),
        ),
        Def::relative(
            MetricId::HipWidthProxy,
            &[Pos::Top, Pos::Impact, Pos::Finish],
            PELVIS,
            |b, a| ratio(b, HIPS, a, HIPS).filter(|&r| r <= 1.0),
        ),
        Def::relative(MetricId::HipTurn, SWING, PELVIS, |b, a| {
            line_pose(b, a, HIPS)?.1
        }),
        Def::relative(MetricId::ShoulderTurn, SWING, SHOULDERS, |b, a| {
            line_pose(b, a, SHOULDER_LINE)?.1
        }),
    ]
}

fn down_the_line<'a>(ctx: &'a Ctx, fwd: f64) -> Vec<Def<'a>> {
    let l = ctx.scale.torso_px;
    vec![
        head_vertical(ctx),
        pelvis_lift(ctx),
        Def::metric(
            MetricId::SpineForwardBend,
            ADDRESS_TO_IMPACT,
            SPINE,
            move |b, _| b.spine_tilt(fwd),
        ),
        Def::relative(MetricId::LossOfPosture, SWING, SPINE, move |b, a| {
            Some(a.spine_tilt(fwd)? - b.spine_tilt(fwd)?)
        }),
        Def::relative(MetricId::EarlyExtension, SWING, PELVIS, move |b, a| {
            ctx.shift_x(b.pelvis(), a.pelvis(), fwd)
        }),
        Def::metric(
            MetricId::LeadKneeFlex,
            ADDRESS_TOP_IMPACT_FINISH,
            &[Joint::LeadHip, Joint::LeadKnee, Joint::LeadAnkle],
            |b, _| b.knee_flex(true),
        ),
        Def::metric(
            MetricId::TrailKneeFlex,
            ADDRESS_TOP_IMPACT_FINISH,
            &[Joint::TrailHip, Joint::TrailKnee, Joint::TrailAnkle],
            |b, _| b.knee_flex(false),
        ),
        Def::metric(
            MetricId::TrailElbowAngle,
            ADDRESS_TOP_IMPACT,
            &[Joint::TrailShoulder, Joint::TrailElbow, Joint::TrailWrist],
            |b, _| b.elbow(false),
        ),
        Def::metric(
            MetricId::HandHeight,
            &[Pos::Top],
            HAND_AND_TRAIL_SHOULDER,
            move |b, _| Some((b.at(Joint::TrailShoulder)?.y - b.hands()?.y) / l),
        ),
        Def::metric(
            MetricId::HandDepth,
            &[Pos::Top],
            HAND_AND_TRAIL_SHOULDER,
            move |b, _| Some(-fwd * (b.hands()?.x - b.at(Joint::TrailShoulder)?.x) / l),
        ),
    ]
}
