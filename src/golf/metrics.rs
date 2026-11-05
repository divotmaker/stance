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
/// height, centimetres. Shifts are measured from address.
///
/// | Metric | View | Unit | Positions | Formula |
/// |---|---|---|---|---|
/// | `hip_sway` | face-on | torso | top, impact | pelvis shift towards the target (negative: away from it): `fwd·(pelvis.x − pelvis₀.x) / L` |
/// | `head_lateral` | face-on | torso | top, impact | head shift towards the target: `fwd·(head.x − head₀.x) / L` |
/// | `head_vertical` | either | torso | top, impact | head rise: `(head₀.y − head.y) / L` |
/// | `lateral_spine_tilt` | face-on | deg | address, top, impact | spine from vertical, positive with the shoulders away from the target |
/// | `shoulder_tilt` | face-on | deg | address, impact | shoulder line from horizontal, positive with the lead shoulder higher (at the top and finish the shoulders turn end-on and the line is foreshortening, not tilt) |
/// | `lead_arm_angle` | face-on | deg | address, top, impact | lead shoulder–elbow–wrist angle (180 = straight) |
/// | `stance_width` | face-on | ratio | address | ankle-to-ankle distance / shoulder-to-shoulder distance |
/// | `hip_width_proxy` | face-on | ratio | top, impact, finish | hip-to-hip distance / its address value: a pelvis-turn proxy, ≈ cos of the turn only if the hips are square to the camera at address and the hip keypoints stay on the pelvis edges; absent when above 1 (not measurable: a turn can't widen the hips) |
/// | `hip_turn` | face-on | deg | top, impact | pelvis-turn estimate: `acos(w / w₀)` of the hip-to-hip distance `w` against address; absent when `w / w₀` is above 1 |
/// | `shoulder_turn` | face-on | deg | top, impact | shoulder-turn estimate: `acos(w / w₀)` of the shoulder-to-shoulder distance against address; absent when the ratio is above 1 |
/// | `spine_forward_bend` | down the line | deg | address, top, impact | spine from vertical, positive leaning towards the ball |
/// | `loss_of_posture` | down the line | deg | top, impact | `spine_forward_bend` at address minus now: positive = more upright |
/// | `early_extension` | down the line | torso | top, impact | pelvis shift towards the ball: `fwd·(pelvis.x − pelvis₀.x) / L` |
/// | `lead_knee_flex` | down the line | deg | address, top, impact, finish | 180 − lead hip–knee–ankle angle (0 = straight) |
/// | `trail_knee_flex` | down the line | deg | address, top, impact, finish | 180 − trail hip–knee–ankle angle |
/// | `trail_elbow_angle` | down the line | deg | address, top, impact | trail shoulder–elbow–wrist angle |
/// | `hand_height` | down the line | torso | top | hands above the trail shoulder: `(trail_shoulder.y − hands.y) / L` |
///
/// Face-on, the golfer bends towards the camera, so `L` is foreshortened
/// by the cosine of the forward bend (about 0.82 for a typical 35° bend)
/// and face-on fractions read about 20 % larger than the same distance down
/// the line; centimetres don't have this bias (see [`BodyScale`]). Relative
/// metrics (shifts and changes from address) take their confidence from the
/// keypoints at address too. Values are rounded to 4 decimal places of
/// their unit.
///
/// `hip_turn` and `shoulder_turn` are one-camera estimates: a body segment
/// square to the camera at address that turns by θ about the vertical shows
/// `cos θ` of its width. They assume the hips and shoulders are square to
/// the camera at address (any angle there is read as part of the turn), a
/// vertical image plane, and the golfer staying at the same distance from
/// the camera. They can't tell which way the body turned, so a turn back
/// past square reads as a turn. Foreshortening and keypoint drift grow
/// with the turn (the far hip or shoulder is hidden and the model places
/// it). A body seen end-on still shows its depth, and with the golfer bent
/// forwards a turning shoulder line partly tilts in the image instead of
/// shortening, so large turns read low. They saturate towards 90°, where `acos` is steep and a few pixels
/// move them by many degrees, and a turn past 90° reads as less (which is
/// why neither is reported at the finish). X-factor (shoulder minus hip
/// turn) and the kinematic sequence need 3D (two calibrated views) and are
/// not measured.
///
/// [`SwingReport::series`](super::SwingReport::series) holds every
/// measured metric's value at every frame, with the same formulas and
/// keypoints; shifts and the turns are against the address frame there
/// too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[allow(missing_docs, reason = "documented in the table above")]
pub enum MetricId {
    HipSway,
    HeadLateral,
    HeadVertical,
    LateralSpineTilt,
    ShoulderTilt,
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
        Self::HeadLateral,
        Self::HeadVertical,
        Self::LateralSpineTilt,
        Self::ShoulderTilt,
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
    ];

    /// The view the metric needs (`None`: either).
    #[must_use]
    pub fn view(self) -> Option<View> {
        match self {
            Self::HeadVertical => None,
            Self::HipSway
            | Self::HeadLateral
            | Self::LateralSpineTilt
            | Self::ShoulderTilt
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
            | Self::HandHeight => Some(View::DownTheLine),
        }
    }

    /// The metric's unit.
    #[must_use]
    pub fn unit(self) -> Unit {
        match self {
            Self::HipSway
            | Self::HeadLateral
            | Self::HeadVertical
            | Self::EarlyExtension
            | Self::HandHeight => Unit::Torso,
            Self::StanceWidth | Self::HipWidthProxy => Unit::Ratio,
            Self::LateralSpineTilt
            | Self::ShoulderTilt
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
}

/// A metric's value at each key position (absent where it doesn't apply or
/// the keypoints it needs weren't found).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[allow(missing_docs, reason = "the swing positions")]
#[non_exhaustive]
pub struct Positions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impact: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<f64>,
}

impl Positions {
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            address: self.address.map(&f),
            top: self.top.map(&f),
            impact: self.impact.map(&f),
            finish: self.finish.map(&f),
        }
    }

    fn is_empty(&self) -> bool {
        self.address.is_none()
            && self.top.is_none()
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
    /// Centimetres per pixel, from the golfer's height: the torso is taken
    /// as [`TORSO_PER_HEIGHT`] × height, matched to its longest 2D length in
    /// the clip (the 90th percentile; the golfer stands upright before
    /// address and at the finish, where the torso isn't foreshortened).
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
            cm_per_px,
        })
    }
}

/// The frames of the key positions.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KeyFrames {
    pub address: Option<usize>,
    pub top: Option<usize>,
    pub impact: Option<usize>,
    pub finish: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pos {
    Address,
    Top,
    Impact,
    Finish,
}

const ADDRESS_TOP_IMPACT: &[Pos] = &[Pos::Address, Pos::Top, Pos::Impact];
const ALL_POSITIONS: &[Pos] = &[Pos::Address, Pos::Top, Pos::Impact, Pos::Finish];
const TOP_IMPACT: &[Pos] = &[Pos::Top, Pos::Impact];

const PELVIS: &[Joint] = &[Joint::LeadHip, Joint::TrailHip];
const SHOULDERS: &[Joint] = &[Joint::LeadShoulder, Joint::TrailShoulder];
const SPINE: &[Joint] = &[
    Joint::LeadHip,
    Joint::TrailHip,
    Joint::LeadShoulder,
    Joint::TrailShoulder,
];
const HEAD: &[Joint] = &[Joint::Head];

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
            Pos::Top => self.keys.top,
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
                Pos::Top => &mut values.top,
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
        None => vec![head_vertical(&ctx)],
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

/// A width ratio `a`–`b` over `c`–`d`.
fn ratio(body: &Body, (a, b): (Joint, Joint), over: &Body, (c, d): (Joint, Joint)) -> Option<f64> {
    let base = over.width(c, d)?;
    (base > 1e-6)
        .then(|| body.width(a, b).map(|w| w / base))
        .flatten()
}

/// The turn (degrees) a segment square to the camera at address shows as
/// its width ratio now: `acos(ratio)`, `None` above 1 (a turn can't widen
/// it).
fn turn(body: &Body, address: &Body, pair: (Joint, Joint)) -> Option<f64> {
    ratio(body, pair, address, pair)
        .filter(|&r| r <= 1.0)
        .map(|r| r.acos().to_degrees())
}

fn face_on<'a>(ctx: &'a Ctx, fwd: f64) -> Vec<Def<'a>> {
    const HIPS: (Joint, Joint) = (Joint::LeadHip, Joint::TrailHip);
    const SHOULDER_LINE: (Joint, Joint) = (Joint::LeadShoulder, Joint::TrailShoulder);
    vec![
        Def::relative(MetricId::HipSway, TOP_IMPACT, PELVIS, move |b, a| {
            ctx.shift_x(b.pelvis(), a.pelvis(), fwd)
        }),
        Def::relative(MetricId::HeadLateral, TOP_IMPACT, HEAD, move |b, a| {
            ctx.shift_x(b.head(), a.head(), fwd)
        }),
        head_vertical(ctx),
        Def::metric(
            MetricId::LateralSpineTilt,
            ADDRESS_TOP_IMPACT,
            SPINE,
            move |b, _| b.spine_tilt(-fwd),
        ),
        Def::metric(
            MetricId::ShoulderTilt,
            &[Pos::Address, Pos::Impact],
            SHOULDERS,
            |b, _| b.shoulder_tilt(),
        ),
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
        Def::relative(MetricId::HipTurn, TOP_IMPACT, PELVIS, |b, a| {
            turn(b, a, HIPS)
        }),
        Def::relative(MetricId::ShoulderTurn, TOP_IMPACT, SHOULDERS, |b, a| {
            turn(b, a, SHOULDER_LINE)
        }),
    ]
}

fn down_the_line<'a>(ctx: &'a Ctx, fwd: f64) -> Vec<Def<'a>> {
    let l = ctx.scale.torso_px;
    vec![
        head_vertical(ctx),
        Def::metric(
            MetricId::SpineForwardBend,
            ADDRESS_TOP_IMPACT,
            SPINE,
            move |b, _| b.spine_tilt(fwd),
        ),
        Def::relative(MetricId::LossOfPosture, TOP_IMPACT, SPINE, move |b, a| {
            Some(a.spine_tilt(fwd)? - b.spine_tilt(fwd)?)
        }),
        Def::relative(MetricId::EarlyExtension, TOP_IMPACT, PELVIS, move |b, a| {
            ctx.shift_x(b.pelvis(), a.pelvis(), fwd)
        }),
        Def::metric(
            MetricId::LeadKneeFlex,
            ALL_POSITIONS,
            &[Joint::LeadHip, Joint::LeadKnee, Joint::LeadAnkle],
            |b, _| b.knee_flex(true),
        ),
        Def::metric(
            MetricId::TrailKneeFlex,
            ALL_POSITIONS,
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
            &[Joint::TrailShoulder, Joint::LeadWrist, Joint::TrailWrist],
            move |b, _| Some((b.at(Joint::TrailShoulder)?.y - b.hands()?.y) / l),
        ),
    ]
}
