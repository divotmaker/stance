//! Bands for the body metrics: where a measured value sits against
//! published tour (or amateur) averages, which way it is off, and what that
//! miss tends to produce.
//!
//! A [`Band`] is a centre and a standard deviation for one metric at one
//! [`Position`], for a [`ClubClass`], a [`Sex`] and a [`Reference`]. Most
//! bands are **shape-neutral**: one band, `centre ± sd`. Four quantities at
//! impact are **shape-linked** (hip turn, shoulder turn, shoulder tilt and
//! hip sway): they carry a fade band at `centre + offset` and a draw band at
//! `centre − offset`, both `± sd`, because a fade and a draw are both good
//! swings and sit a little apart there. The data file, not the caller, says
//! which bands are shape-linked.
//!
//! [`Norms::grade`] places a value: a [`Fit`] ([`Fit::Both`] inside both
//! sub-bands, or inside a neutral band; [`Fit::Fade`] or [`Fit::Draw`]
//! inside one; [`Fit::Near`] within 2 SD of the nearer; [`Fit::Far`]
//! beyond), the [`Side`] of the neutral centre it is on, and for a miss the
//! direction word and the pattern it tends to produce ([`Miss`]). The
//! engine never decides colours and never needs the player's target shape.
//!
//! The built-in table ([`Norms::builtin`]) is a versioned JSON file embedded
//! in the crate, from published tour averages (Sportsbox AI, GEARS, K-Vest
//! norms, biomechanics papers); `docs/NORMS.md` lists every band with its
//! sources. [`Norms::from_json`] reads another table and
//! [`Norms::from_bands`] builds one in code (e.g. a player's own bands).
//!
//! # Values and signs
//!
//! Bands use stance's signs ([`MetricId`]): shifts positive towards the
//! target, the ball or up; tilts positive with the lead side higher; lateral
//! spine tilt positive with the shoulders away from the target. Turns are
//! signed: **positive open** (towards the target), negative closed, so a
//! tour top is about −40° for the hips and impact about +38°. One face-on
//! camera measures a turn's size, not its direction, so a one-camera
//! `hip_turn` or `shoulder_turn` takes the sign of the band's centre, is not
//! graded where the centre is within 2 SD of zero, and is graded at impact
//! only (at the top it reads 25–40° low). Distances grade in inches: from
//! [`Metric::cm`] when the golfer's height was given ([`Scale::Measured`]),
//! otherwise from an assumed height ([`Scale::Assumed`]). Every metric is
//! target-relative, so one table serves both hands.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::OnceLock;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::events::Tempo;
use super::metrics::{BodyScale, Metric, MetricId, Positions, TORSO_PER_HEIGHT, Unit};
use super::report::SwingReport;
use crate::{Error, Result};

/// The band-table format version [`Norms::from_json`] reads.
pub const NORMS_VERSION: u32 = 1;

/// The fade-side centre offset of the hip turn at impact (degrees; the draw
/// side is the negation). The four offsets are inferences, not
/// measurements: the built-in table carries them on its shape-linked rows.
pub const FADE_OFFSET_HIP_TURN_DEG: f64 = 3.0;
/// The fade-side centre offset of the shoulder turn at impact (degrees).
pub const FADE_OFFSET_SHOULDER_TURN_DEG: f64 = 3.0;
/// The fade-side centre offset of the shoulder tilt at impact (degrees):
/// 3, not 4, so with its SD of 4.5 the fade and draw bands overlap by
/// ±1.5° around the centre rather than ±0.5°.
pub const FADE_OFFSET_SHOULDER_TILT_DEG: f64 = -3.0;
/// The fade-side centre offset of the hip sway at impact (inches).
pub const FADE_OFFSET_HIP_SWAY_IN: f64 = -0.75;

/// The height (cm) inch bands assume for a man when the golfer's height is
/// unknown: the US adult mean.
pub const ASSUMED_HEIGHT_MALE_CM: f64 = 177.0;
/// The height (cm) inch bands assume for a woman when the golfer's height
/// is unknown: the US adult mean.
pub const ASSUMED_HEIGHT_FEMALE_CM: f64 = 164.0;

/// [`Context::min_confidence`]'s default.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.5;

/// The built-in table.
const TOUR_JSON: &str = include_str!("norms/tour.json");

/// Distances compare against band edges with this slack, so a value on an
/// edge stays inside after rounding.
const EDGE: f64 = 1e-9;

/// Where in the swing a band applies: the key positions of a
/// [`SwingReport`], and [`Position::Swing`] for whole-swing quantities
/// (tempo).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Position {
    /// Address.
    Address,
    /// P3: the lead arm parallel to the ground in the backswing.
    ArmParallelBack,
    /// The top of the backswing.
    Top,
    /// P5: the lead arm parallel to the ground in the downswing.
    ArmParallelDown,
    /// Impact.
    Impact,
    /// The finish.
    Finish,
    /// The whole swing (tempo).
    Swing,
}

impl Position {
    /// The key positions of a [`SwingReport`], in swing order.
    pub const KEY: &'static [Self] = &[
        Self::Address,
        Self::ArmParallelBack,
        Self::Top,
        Self::ArmParallelDown,
        Self::Impact,
        Self::Finish,
    ];
}

impl Positions {
    /// The value at a key position (`None` for [`Position::Swing`]).
    #[must_use]
    pub fn at(&self, position: Position) -> Option<f64> {
        match position {
            Position::Address => self.address,
            Position::ArmParallelBack => self.arm_parallel_back,
            Position::Top => self.top,
            Position::ArmParallelDown => self.arm_parallel_down,
            Position::Impact => self.impact,
            Position::Finish => self.finish,
            Position::Swing => None,
        }
    }
}

/// A whole-swing quantity with bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TempoMeasure {
    /// [`Tempo::ratio`]: backswing over downswing.
    #[serde(rename = "tempo_ratio")]
    Ratio,
    /// [`Tempo::downswing_ns`], in seconds.
    #[serde(rename = "downswing_time")]
    Downswing,
}

/// What a band measures: a body metric, or a tempo quantity. In JSON, the
/// metric's id (`"hip_turn"`) or `"tempo_ratio"` / `"downswing_time"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum Measure {
    /// A body metric.
    Metric(MetricId),
    /// A tempo quantity.
    Tempo(TempoMeasure),
}

impl From<MetricId> for Measure {
    fn from(id: MetricId) -> Self {
        Self::Metric(id)
    }
}

impl From<TempoMeasure> for Measure {
    fn from(t: TempoMeasure) -> Self {
        Self::Tempo(t)
    }
}

impl Measure {
    /// The quantity's name for display ("Hip turn").
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Metric(id) => match id {
                MetricId::HipSway => "Hip sway",
                MetricId::ChestSway => "Chest sway",
                MetricId::SwayGap => "Sway gap",
                MetricId::HeadLateral => "Head shift",
                MetricId::HeadVertical => "Head height",
                MetricId::PelvisLift => "Pelvis lift",
                MetricId::LateralSpineTilt => "Lateral spine tilt",
                MetricId::ShoulderTilt => "Shoulder tilt",
                MetricId::HipTilt => "Hip tilt",
                MetricId::LeadArmAngle => "Lead arm",
                MetricId::StanceWidth => "Stance width",
                MetricId::HipWidthProxy => "Hip width",
                MetricId::HipTurn => "Hip turn",
                MetricId::ShoulderTurn => "Shoulder turn",
                MetricId::SpineForwardBend => "Forward bend",
                MetricId::LossOfPosture => "Loss of posture",
                MetricId::EarlyExtension => "Early extension",
                MetricId::LeadKneeFlex => "Lead knee flex",
                MetricId::TrailKneeFlex => "Trail knee flex",
                MetricId::TrailElbowAngle => "Trail elbow",
                MetricId::HandHeight => "Hand height",
                MetricId::HandDepth => "Hand depth",
            },
            Self::Tempo(TempoMeasure::Ratio) => "Tempo",
            Self::Tempo(TempoMeasure::Downswing) => "Downswing time",
        }
    }

    /// The unit its bands are in.
    #[must_use]
    pub fn norm_unit(self) -> NormUnit {
        match self {
            Self::Metric(id) => match id.unit() {
                Unit::Degrees => NormUnit::Degrees,
                Unit::Torso => NormUnit::Inches,
                Unit::Ratio => NormUnit::Ratio,
            },
            Self::Tempo(TempoMeasure::Ratio) => NormUnit::Ratio,
            Self::Tempo(TempoMeasure::Downswing) => NormUnit::Seconds,
        }
    }

    /// The quantity whose bands grade this one ([`MetricId::band_key`]).
    #[must_use]
    pub fn band_key(self) -> Self {
        match self {
            Self::Metric(id) => Self::Metric(id.band_key()),
            Self::Tempo(_) => self,
        }
    }

    /// A one-camera turn: an unsigned size that takes its band's sign.
    fn one_camera_turn(self) -> bool {
        matches!(
            self,
            Self::Metric(MetricId::HipTurn | MetricId::ShoulderTurn)
        )
    }
}

/// The club a band is for. A caller maps its club to a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ClubClass {
    /// Driver (and mini driver).
    Driver,
    /// Mid-irons, and the clubs graded like them: fairway woods and hybrids; also
    /// used for wedges where no wedge band exists.
    MidIron,
    /// Wedges.
    Wedge,
}

/// The golfer's sex, for the bands that differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Sex {
    /// Male.
    Male,
    /// Female.
    Female,
}

/// The population a band describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Reference {
    /// Tour players.
    Tour,
    /// Amateur golfers.
    Amateur,
    /// The player's own swings: tables a caller builds
    /// ([`Norms::from_bands`]); the built-in table has none.
    Personal,
}

/// Where a band's SD comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SdSource {
    /// Published (or measured) for this cell.
    Published,
    /// Borrowed from the nearest published spread.
    Proxy,
}

/// How far a band's centre can be trusted for this measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Confidence {
    /// Little support, or a known gap between the source's definition and
    /// stance's measurement.
    Low,
    /// One good source, or two that agree loosely.
    Medium,
    /// Several sources that agree, on a matching definition.
    High,
}

/// A band's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum NormUnit {
    /// Degrees ([`Unit::Degrees`] metrics).
    Degrees,
    /// Inches ([`Unit::Torso`] metrics, through centimetres).
    Inches,
    /// A ratio ([`Unit::Ratio`] metrics, tempo).
    Ratio,
    /// Seconds (downswing time).
    Seconds,
}

/// The direction of a miss, in stance's target-relative words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[allow(missing_docs, reason = "the words are the docs")]
pub enum Direction {
    Open,
    Closed,
    Toward,
    Away,
    Up,
    Down,
    TowardBall,
    Back,
    More,
    Less,
}

impl Direction {
    /// The comparative phrase after a quantity's name ("Hip turn *more
    /// open*").
    #[must_use]
    pub fn phrase(self) -> &'static str {
        match self {
            Self::Open => "more open",
            Self::Closed => "more closed",
            Self::Toward => "further toward the target",
            Self::Away => "further away from the target",
            Self::Up => "higher",
            Self::Down => "lower",
            Self::TowardBall => "further toward the ball",
            Self::Back => "further back",
            Self::More => "more",
            Self::Less => "less",
        }
    }

    /// The phrase for a value short of a centre in this direction ("Shoulder
    /// turn *less open*"): `Some` for the words a signed value reads as
    /// (open, closed), `None` for the positional and magnitude words, whose
    /// [`Direction::phrase`] already reads right on either side of zero.
    #[must_use]
    pub fn short_phrase(self) -> Option<&'static str> {
        match self {
            Self::Open => Some("less open"),
            Self::Closed => Some("less closed"),
            _ => None,
        }
    }

    /// The other side's word: open and closed, toward and away, up and down,
    /// toward the ball and back, more and less.
    #[must_use]
    pub fn opposite(self) -> Self {
        match self {
            Self::Open => Self::Closed,
            Self::Closed => Self::Open,
            Self::Toward => Self::Away,
            Self::Away => Self::Toward,
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::TowardBall => Self::Back,
            Self::Back => Self::TowardBall,
            Self::More => Self::Less,
            Self::Less => Self::More,
        }
    }
}

/// One side of a band's miss: the direction word and the pattern it tends
/// to produce.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Miss {
    /// Which way the value is off.
    pub direction: Direction,
    /// The pattern it tends to produce ("hang-back, flip, hook or push
    /// pattern").
    pub tends_to: String,
}

impl Miss {
    /// A miss.
    #[must_use]
    pub fn new(direction: Direction, tends_to: impl Into<String>) -> Self {
        Self {
            direction,
            tends_to: tends_to.into(),
        }
    }
}

/// A sub-band: `centre ± sd`, in the band's unit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Range {
    /// The centre.
    pub centre: f64,
    /// The half-width (one SD).
    pub sd: f64,
}

impl Range {
    /// The distance of `value` from the centre in SDs, signed.
    #[must_use]
    pub fn z(&self, value: f64) -> f64 {
        (value - self.centre) / self.sd
    }
}

fn yes() -> bool {
    true
}

#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde's signature")]
fn is_true(b: &bool) -> bool {
    *b
}

/// One band: a metric at a position, for a club class, a sex and a
/// reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Band {
    /// What it measures.
    pub metric: Measure,
    /// Where in the swing.
    pub position: Position,
    /// The club class; `None`: any club.
    #[serde(default)]
    pub club: Option<ClubClass>,
    /// The sex; `None`: either.
    #[serde(default)]
    pub sex: Option<Sex>,
    /// The population.
    pub reference: Reference,
    /// The neutral band's centre, in [`Band::unit`].
    pub centre: f64,
    /// The neutral band's half-width (one SD), and each shape sub-band's.
    pub sd: f64,
    /// The unit of the centre and SD.
    pub unit: NormUnit,
    /// `Some(o)`: a shape-linked band, the fade sub-band centred at
    /// `centre + o` and the draw sub-band at `centre − o`, both `± sd`
    /// (or the explicit [`Band::fade`] / [`Band::draw`]). `None`:
    /// shape-neutral.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_offset: Option<f64>,
    /// An explicit fade sub-band (a player's own bands); either may be
    /// absent. Used only with [`Band::shape_offset`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fade: Option<Range>,
    /// An explicit draw sub-band (a player's own bands).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw: Option<Range>,
    /// Where the SD comes from.
    pub sd_source: SdSource,
    /// How far the centre can be trusted for this measurement.
    pub confidence: Confidence,
    /// Keys into [`Norms::sources`].
    #[serde(default)]
    pub sources: Vec<String>,
    /// The miss below the centre; `None`: a low value is never a miss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<Miss>,
    /// The miss above the centre; `None`: a high value is never a miss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<Miss>,
    /// Whether values are graded against it. An ungraded band (`"grade":
    /// false` in JSON) is a reference not yet validated for stance's
    /// measurement: [`Norms::band`] finds it, [`Norms::grade`] doesn't.
    #[serde(default = "yes", rename = "grade", skip_serializing_if = "is_true")]
    pub graded: bool,
    /// A remark on the band.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

impl Band {
    /// A shape-neutral, two-sided band for any club and either sex, with no
    /// miss words, an SD measured for it, medium confidence and no sources.
    #[must_use]
    pub fn new(
        metric: impl Into<Measure>,
        position: Position,
        reference: Reference,
        centre: f64,
        sd: f64,
        unit: NormUnit,
    ) -> Self {
        Self {
            metric: metric.into(),
            position,
            club: None,
            sex: None,
            reference,
            centre,
            sd,
            unit,
            shape_offset: None,
            fade: None,
            draw: None,
            sd_source: SdSource::Published,
            confidence: Confidence::Medium,
            sources: Vec::new(),
            low: None,
            high: None,
            graded: true,
            note: String::new(),
        }
    }

    /// Whether the band is shape-linked.
    #[must_use]
    pub fn shape_linked(&self) -> bool {
        self.shape_offset.is_some()
    }

    /// The fade and draw sub-bands of a shape-linked band: explicit ones
    /// when either is set, otherwise `centre ± offset` with the band's SD
    /// (none for a [`Reference::Personal`] band). `(None, None)` for a
    /// shape-neutral band.
    #[must_use]
    pub fn sub_bands(&self) -> (Option<Range>, Option<Range>) {
        let Some(o) = self.shape_offset else {
            return (None, None);
        };
        if self.fade.is_some() || self.draw.is_some() {
            return (self.fade, self.draw);
        }
        if self.reference == Reference::Personal {
            return (None, None);
        }
        let at = |centre| {
            Some(Range {
                centre,
                sd: self.sd,
            })
        };
        (at(self.centre + o), at(self.centre - o))
    }

    fn miss(&self, side: Side) -> Option<&Miss> {
        match side {
            Side::Low => self.low.as_ref(),
            Side::High => self.high.as_ref(),
            Side::On => None,
        }
    }

    /// Errors for a band no value can be graded against.
    fn check(&self) -> Result<()> {
        let bad = |what: &str| {
            Err(Error::InvalidArgument(format!(
                "norms: {:?} at {:?}: {what}",
                self.metric, self.position
            )))
        };
        let positive = |sd: f64| sd.is_finite() && sd > 0.0;
        if !self.centre.is_finite() {
            return bad("centre is not finite");
        }
        if !positive(self.sd) {
            return bad("sd is not positive");
        }
        if self.shape_offset.is_some_and(|o| !o.is_finite()) {
            return bad("shape_offset is not finite");
        }
        if [self.fade, self.draw]
            .iter()
            .flatten()
            .any(|r| !r.centre.is_finite() || !positive(r.sd))
        {
            return bad("a sub-band is not finite and positive");
        }
        if self.unit != self.metric.norm_unit() {
            return bad("unit does not match the metric");
        }
        Ok(())
    }
}

/// A band table.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Norms {
    /// The format version ([`NORMS_VERSION`]).
    pub version: u32,
    /// The data's revision (a date), for a log line.
    pub revision: String,
    /// What each source key in [`Band::sources`] refers to.
    pub sources: BTreeMap<String, String>,
    bands: Vec<Band>,
    skipped: usize,
}

/// How a band was found ([`Norms::band`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Lookup<'a> {
    /// The band.
    pub band: &'a Band,
    /// The club class had no band: a wedge took the mid-iron band.
    pub club_fallback: bool,
    /// The sex had no band: a woman took the male band.
    pub sex_fallback: bool,
}

/// A band lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Query {
    /// What is measured (looked up by [`Measure::band_key`]).
    pub metric: Measure,
    /// Where in the swing.
    pub position: Position,
    /// The club class.
    pub club: ClubClass,
    /// The golfer's sex.
    pub sex: Sex,
    /// The population.
    pub reference: Reference,
}

impl Query {
    /// A lookup.
    #[must_use]
    pub fn new(
        metric: impl Into<Measure>,
        position: Position,
        club: ClubClass,
        sex: Sex,
        reference: Reference,
    ) -> Self {
        Self {
            metric: metric.into(),
            position,
            club,
            sex,
            reference,
        }
    }
}

/// What grading needs to know about the golfer and the shot.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Context {
    /// The club class.
    pub club: ClubClass,
    /// The golfer's sex.
    pub sex: Sex,
    /// The population to compare with.
    pub reference: Reference,
    /// The body scale, for inch bands against a torso-unit value without
    /// centimetres: with [`BodyScale::cm_per_px`] the value is measured,
    /// otherwise an assumed height is used. [`Norms::grade_report`] takes
    /// the report's scale when this is `None`.
    pub scale: Option<BodyScale>,
    /// A metric whose keypoint confidence is below this is not graded.
    pub min_confidence: f32,
}

impl Context {
    /// A context with no body scale and the default minimum confidence
    /// ([`DEFAULT_MIN_CONFIDENCE`]).
    #[must_use]
    pub fn new(club: ClubClass, sex: Sex, reference: Reference) -> Self {
        Self {
            club,
            sex,
            reference,
            scale: None,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }
}

/// Where a value sits against a band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Fit {
    /// Inside the neutral band, or inside both the fade and the draw band.
    Both,
    /// Inside the fade band only.
    Fade,
    /// Inside the draw band only.
    Draw,
    /// Outside, within 2 SD of the nearer band.
    Near,
    /// Further out.
    Far,
}

/// Which side of the neutral centre a value is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Side {
    /// Below: inside neither band's overlap, under the centre.
    Low,
    /// Above.
    High,
    /// Inside the neutral band, or inside both sub-bands of a shape-linked
    /// band.
    On,
}

/// How a distance was put into inches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Scale {
    /// No conversion (degrees, ratios, seconds).
    Native,
    /// From the golfer's height.
    Measured,
    /// From an assumed height ([`ASSUMED_HEIGHT_MALE_CM`],
    /// [`ASSUMED_HEIGHT_FEMALE_CM`]).
    Assumed,
}

/// A graded value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Grade {
    /// What was graded.
    pub metric: Measure,
    /// Where in the swing.
    pub position: Position,
    /// The value in the band's unit, signed as the band (a one-camera turn
    /// takes the band's sign).
    pub value: f64,
    /// The neutral band's centre.
    pub centre: f64,
    /// The neutral band's SD.
    pub sd: f64,
    /// The band's unit.
    pub unit: NormUnit,
    /// `(value − centre) / sd`.
    pub z: f64,
    /// Where it sits.
    pub fit: Fit,
    /// Which side of the neutral centre.
    pub side: Side,
    /// The direction word of [`Grade::side`] (`None` on the centre, or on
    /// a side with no miss).
    pub direction: Option<Direction>,
    /// The side's miss for [`Fit::Near`] and [`Fit::Far`].
    pub miss: Option<Miss>,
    /// The fade sub-band (shape-linked bands).
    pub fade: Option<Range>,
    /// The draw sub-band (shape-linked bands).
    pub draw: Option<Range>,
    /// The distance from the fade sub-band's centre in SDs.
    pub z_fade: Option<f64>,
    /// The distance from the draw sub-band's centre in SDs.
    pub z_draw: Option<f64>,
    /// The metric's keypoint confidence ([`Metric::confidence`]; 1 for
    /// tempo).
    pub confidence: f32,
    /// The band's population.
    pub reference: Reference,
    /// Where the band's SD comes from.
    pub sd_source: SdSource,
    /// The band's own confidence.
    pub band_confidence: Confidence,
    /// How a distance was put into inches.
    pub scale: Scale,
    /// A wedge graded against the mid-iron band.
    pub club_fallback: bool,
    /// A woman graded against the male band.
    pub sex_fallback: bool,
}

impl Grade {
    /// Whether the band's SD is borrowed ([`SdSource::Proxy`]).
    #[must_use]
    pub fn proxy_sd(&self) -> bool {
        self.sd_source == SdSource::Proxy
    }

    /// The comparative phrase for [`Grade::direction`]: "more open" or
    /// "more closed" ([`Direction::phrase`]), except that a value between
    /// zero and the centre reads "less" and the centre's own word, so it
    /// never contradicts the value's sign: a 15° open shoulder turn against
    /// a 30° open centre is "less open", not "more closed". Positional and
    /// magnitude words ("further toward the target", "higher", "more")
    /// always use [`Direction::phrase`]. `None` without a direction.
    #[must_use]
    pub fn phrase(&self) -> Option<&'static str> {
        let d = self.direction?;
        // Strictly between zero and the centre (NaN or infinite for a zero
        // centre, so never).
        let r = self.value / self.centre;
        let short = r > 0.0 && r < 1.0;
        Some(
            short
                .then(|| d.opposite().short_phrase())
                .flatten()
                .unwrap_or_else(|| d.phrase()),
        )
    }

    /// The words for a value off the band: `None` for [`Fit::Both`] (a
    /// display shows the metric's own direction word); "Hip turn more open:
    /// fade side" for [`Fit::Fade`] and [`Fit::Draw`]; "Hip turn more open
    /// than tour: fade or slice pattern when the arms lag" for
    /// [`Fit::Near`] and [`Fit::Far`]. The comparative is
    /// [`Grade::phrase`]: "Shoulder turn less open than tour: shoulders
    /// behind, push or hook" for a 15° open turn against a 30° open centre.
    #[must_use]
    pub fn fit_word(&self) -> Option<String> {
        let mut s = self.metric.name().to_owned();
        if let Some(p) = self.phrase() {
            s.push(' ');
            s.push_str(p);
        }
        match self.fit {
            Fit::Both => return None,
            Fit::Fade => s.push_str(": fade side"),
            Fit::Draw => s.push_str(": draw side"),
            Fit::Near | Fit::Far => {
                let reference = match self.reference {
                    Reference::Tour => "tour",
                    Reference::Amateur => "amateurs",
                    Reference::Personal => "usual",
                };
                let _ = write!(s, " than {reference}");
                if let Some(m) = &self.miss {
                    let _ = write!(s, ": {}", m.tends_to);
                }
            }
        }
        Some(s)
    }
}

/// A value to grade.
struct Input {
    metric: Measure,
    position: Position,
    value: f64,
    /// The value's unit as measured (`None`: already in the band's unit).
    unit: Option<Unit>,
    cm: Option<f64>,
    confidence: f32,
}

/// Whether a row's optional field, when present, parses as `T`.
fn parses<T: DeserializeOwned>(row: &serde_json::Value, key: &str) -> bool {
    row.get(key)
        .is_none_or(|v| v.is_null() || T::deserialize(v).is_ok())
}

/// Whether a row names only things this crate knows.
fn known_row(row: &serde_json::Value) -> bool {
    let miss_known = |key: &str| {
        row.get(key)
            .is_none_or(|m| parses::<Direction>(m, "direction"))
    };
    parses::<Measure>(row, "metric")
        && parses::<Position>(row, "position")
        && parses::<ClubClass>(row, "club")
        && parses::<Sex>(row, "sex")
        && parses::<Reference>(row, "reference")
        && parses::<NormUnit>(row, "unit")
        && miss_known("low")
        && miss_known("high")
}

impl Norms {
    /// The built-in table (`src/golf/norms/tour.json`), parsed once.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the embedded table doesn't parse (a
    /// build defect; the crate's tests load it).
    pub fn builtin() -> Result<&'static Self> {
        static BUILTIN: OnceLock<std::result::Result<Norms, String>> = OnceLock::new();
        match BUILTIN.get_or_init(|| Self::from_json(TOUR_JSON).map_err(|e| e.to_string())) {
            Ok(n) => Ok(n),
            Err(e) => Err(Error::InvalidArgument(format!("built-in norms: {e}"))),
        }
    }

    /// The built-in table's JSON, as a starting point for another table.
    #[must_use]
    pub fn builtin_json() -> &'static str {
        TOUR_JSON
    }

    /// Reads a band table.
    ///
    /// ```json
    /// {
    ///   "version": 1,
    ///   "revision": "2026-11-06",
    ///   "sources": { "SB": "Sportsbox AI tour averages" },
    ///   "bands": [
    ///     { "metric": "hip_turn", "position": "impact", "club": "driver", "sex": "male",
    ///       "reference": "tour", "centre": 38, "sd": 5, "unit": "degrees", "shape_offset": 3,
    ///       "sd_source": "proxy", "confidence": "high", "sources": ["SB"],
    ///       "low": { "direction": "closed", "tends_to": "hang-back, flip, hook or push pattern" },
    ///       "high": { "direction": "open", "tends_to": "fade or slice pattern when the arms lag" } }
    ///   ]
    /// }
    /// ```
    ///
    /// A row naming a metric, position, club class, sex, reference, unit or
    /// direction this crate doesn't know is skipped and counted
    /// ([`Norms::skipped`]), so a table written for a later version loads;
    /// unknown fields are ignored. `club` and `sex` may be `null` (any).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for malformed JSON, a version other than
    /// [`NORMS_VERSION`], or a known row that is malformed (a missing or
    /// non-finite centre, an SD that isn't positive, a unit that doesn't
    /// match the metric).
    pub fn from_json(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Raw {
            version: u32,
            #[serde(default)]
            revision: String,
            #[serde(default)]
            sources: BTreeMap<String, String>,
            #[serde(default)]
            bands: Vec<serde_json::Value>,
        }
        let invalid = |e: serde_json::Error| Error::InvalidArgument(format!("norms: {e}"));
        let raw: Raw = serde_json::from_str(json).map_err(invalid)?;
        if raw.version != NORMS_VERSION {
            return Err(Error::InvalidArgument(format!(
                "norms version {} (this crate reads {NORMS_VERSION})",
                raw.version
            )));
        }
        let mut bands = Vec::with_capacity(raw.bands.len());
        let mut skipped = 0;
        for row in raw.bands {
            if !known_row(&row) {
                skipped += 1;
                continue;
            }
            let band = Band::deserialize(row).map_err(invalid)?;
            band.check()?;
            bands.push(band);
        }
        Ok(Self {
            version: raw.version,
            revision: raw.revision,
            sources: raw.sources,
            bands,
            skipped,
        })
    }

    /// A table built in code, e.g. a player's own bands
    /// ([`Reference::Personal`]). Bands no value can be graded against (a
    /// non-positive SD, a unit that doesn't match the metric) are kept but
    /// never grade.
    #[must_use]
    pub fn from_bands(version: u32, sources: BTreeMap<String, String>, bands: Vec<Band>) -> Self {
        Self {
            version,
            revision: String::new(),
            sources,
            bands,
            skipped: 0,
        }
    }

    /// Rows [`Norms::from_json`] skipped.
    #[must_use]
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Every band.
    #[must_use]
    pub fn bands(&self) -> &[Band] {
        &self.bands
    }

    fn find(
        &self,
        metric: Measure,
        position: Position,
        club: Option<ClubClass>,
        sex: Option<Sex>,
        reference: Reference,
    ) -> Option<&Band> {
        self.bands.iter().find(|b| {
            b.metric == metric
                && b.position == position
                && b.club == club
                && b.sex == sex
                && b.reference == reference
        })
    }

    /// The band for a query, and how it was found:
    ///
    /// 1. the exact club class and sex;
    /// 2. no band for the sex: the band for either sex (`sex: null`), else
    ///    for a woman the male band (`sex_fallback`);
    /// 3. no band for the club class: the band for any club
    ///    (`club: null`), else for a wedge the mid-iron band
    ///    (`club_fallback`), each with rule 2.
    ///
    /// The metric is looked up by [`Measure::band_key`].
    #[must_use]
    pub fn band(&self, q: &Query) -> Option<Lookup<'_>> {
        let metric = q.metric.band_key();
        let mut clubs = vec![(Some(q.club), false), (None, false)];
        if q.club == ClubClass::Wedge {
            clubs.push((Some(ClubClass::MidIron), true));
        }
        let mut sexes = vec![(Some(q.sex), false), (None, false)];
        if q.sex == Sex::Female {
            sexes.push((Some(Sex::Male), true));
        }
        clubs.iter().find_map(|&(club, club_fallback)| {
            sexes.iter().find_map(|&(sex, sex_fallback)| {
                self.find(metric, q.position, club, sex, q.reference)
                    .map(|band| Lookup {
                        band,
                        club_fallback,
                        sex_fallback,
                    })
            })
        })
    }

    /// Grades a metric of a [`SwingReport`] at a position (see
    /// [`Norms::grade_value`]).
    #[must_use]
    pub fn grade(&self, metric: &Metric, position: Position, ctx: &Context) -> Option<Grade> {
        self.grade_value(
            metric.id,
            position,
            metric.values.at(position)?,
            metric.unit,
            metric.cm.and_then(|c| c.at(position)),
            metric.confidence,
            ctx,
        )
    }

    /// Grades a value of metric `id` at `position`, in `unit` (the metric's
    /// own), with its value in centimetres for a torso-unit metric when the
    /// golfer's height is known, and its keypoint confidence. `None`
    /// (ungraded, the plain number shows) when:
    ///
    /// - no band applies ([`Norms::band`]), or the band is ungraded
    ///   ([`Band::graded`]);
    /// - `confidence` is below [`Context::min_confidence`];
    /// - `unit` isn't the metric's, or an inch band has neither
    ///   centimetres nor a body scale with [`BodyScale::upright_px`];
    /// - it is a one-camera `hip_turn` or `shoulder_turn` anywhere but
    ///   impact, or its band's centre is within 2 SD of zero (the turn's
    ///   direction is unknown);
    /// - a shape-linked [`Reference::Personal`] band has neither sub-band.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "one value and its measurement facts"
    )]
    pub fn grade_value(
        &self,
        id: MetricId,
        position: Position,
        value: f64,
        unit: Unit,
        cm: Option<f64>,
        confidence: f32,
        ctx: &Context,
    ) -> Option<Grade> {
        self.grade_input(
            &Input {
                metric: id.into(),
                position,
                value,
                unit: Some(unit),
                cm,
                confidence,
            },
            ctx,
            true,
        )
    }

    /// Every metric of the report at every position with a band, in the
    /// report's metric order and swing order. The report's body scale is
    /// used when `ctx` has none. Tempo is graded by [`Norms::grade_tempo`].
    #[must_use]
    pub fn grade_report(&self, report: &SwingReport, ctx: &Context) -> Vec<Grade> {
        let ctx = Context {
            scale: ctx.scale.or(report.scale),
            ..*ctx
        };
        report
            .metrics
            .iter()
            .flat_map(|m| {
                Position::KEY
                    .iter()
                    .filter_map(move |&p| self.grade(m, p, &ctx))
            })
            .collect()
    }

    /// The tempo ratio and the downswing time (seconds), where they have
    /// bands.
    #[must_use]
    pub fn grade_tempo(&self, tempo: &Tempo, ctx: &Context) -> Vec<Grade> {
        // Downswings are well under 2^52 ns.
        #[allow(clippy::cast_precision_loss, reason = "durations fit f64")]
        let downswing_s = tempo.downswing_ns as f64 / 1e9;
        [
            (TempoMeasure::Ratio, tempo.ratio),
            (TempoMeasure::Downswing, downswing_s),
        ]
        .into_iter()
        .filter_map(|(m, value)| {
            self.grade_input(
                &Input {
                    metric: m.into(),
                    position: Position::Swing,
                    value,
                    unit: None,
                    cm: None,
                    confidence: 1.0,
                },
                ctx,
                true,
            )
        })
        .collect()
    }

    /// The value in the band's unit, and how it was scaled.
    fn in_band_unit(input: &Input, band: &Band, ctx: &Context) -> Option<(f64, Scale)> {
        let native = input.metric.norm_unit();
        if band.unit != native {
            return None;
        }
        if let (Measure::Metric(id), Some(unit)) = (input.metric, input.unit)
            && unit != id.unit()
        {
            return None;
        }
        if native != NormUnit::Inches {
            return Some((input.value, Scale::Native));
        }
        if let Some(cm) = input.cm {
            return Some((cm / 2.54, Scale::Measured));
        }
        let scale = ctx.scale?;
        if let Some(cm) = scale.to_cm(input.value) {
            return Some((cm / 2.54, Scale::Measured));
        }
        let upright = scale.upright_px.filter(|u| *u > 0.0)?;
        let height = match ctx.sex {
            Sex::Female => ASSUMED_HEIGHT_FEMALE_CM,
            Sex::Male => ASSUMED_HEIGHT_MALE_CM,
        };
        let cm = input.value * scale.torso_px * (TORSO_PER_HEIGHT * height / upright);
        Some((cm / 2.54, Scale::Assumed))
    }

    /// Grades one value; `impact_only_turns` applies the rule that a
    /// one-camera turn is graded at impact only.
    fn grade_input(&self, input: &Input, ctx: &Context, impact_only_turns: bool) -> Option<Grade> {
        if !input.value.is_finite() || input.confidence < ctx.min_confidence {
            return None;
        }
        let one_camera_turn = input.metric.one_camera_turn();
        if one_camera_turn && impact_only_turns && input.position != Position::Impact {
            return None;
        }
        let found = self.band(&Query {
            metric: input.metric,
            position: input.position,
            club: ctx.club,
            sex: ctx.sex,
            reference: ctx.reference,
        })?;
        let band = found.band;
        if !band.graded || band.check().is_err() {
            return None;
        }
        let (mut v, scale) = Self::in_band_unit(input, band, ctx)?;
        if !v.is_finite() {
            return None;
        }
        let (c, s) = (band.centre, band.sd);
        if one_camera_turn {
            if c.abs() < 2.0 * s {
                return None;
            }
            v = v.abs() * c.signum();
        }
        let z = (v - c) / s;
        let inside = |z: f64| z.abs() <= 1.0 + EDGE;
        let (fade, draw) = band.sub_bands();
        let (z_fade, z_draw) = (fade.map(|r| r.z(v)), draw.map(|r| r.z(v)));
        let (mut fit, side) = if band.shape_linked() {
            if fade.is_none() && draw.is_none() {
                return None;
            }
            let (in_f, in_d) = (z_fade.is_some_and(inside), z_draw.is_some_and(inside));
            let nearest = z_fade
                .into_iter()
                .chain(z_draw)
                .map(f64::abs)
                .fold(f64::INFINITY, f64::min);
            let fit = match (in_f, in_d) {
                (true, true) => Fit::Both,
                (true, false) => Fit::Fade,
                (false, true) => Fit::Draw,
                _ if nearest <= 2.0 + EDGE => Fit::Near,
                _ => Fit::Far,
            };
            let side = if fit == Fit::Both {
                Side::On
            } else {
                side_of(v, c)
            };
            (fit, side)
        } else if inside(z) {
            (Fit::Both, Side::On)
        } else if z.abs() <= 2.0 + EDGE {
            (Fit::Near, side_of(v, c))
        } else {
            (Fit::Far, side_of(v, c))
        };
        let side_miss = band.miss(side);
        if matches!(fit, Fit::Near | Fit::Far) && side != Side::On && side_miss.is_none() {
            // A one-sided band: no miss on this side.
            fit = Fit::Both;
        }
        let miss = matches!(fit, Fit::Near | Fit::Far)
            .then(|| side_miss.cloned())
            .flatten();
        Some(Grade {
            metric: input.metric,
            position: input.position,
            value: v,
            centre: c,
            sd: s,
            unit: band.unit,
            z,
            fit,
            side,
            direction: side_miss.map(|m| m.direction),
            miss,
            fade,
            draw,
            z_fade,
            z_draw,
            confidence: input.confidence,
            reference: band.reference,
            sd_source: band.sd_source,
            band_confidence: band.confidence,
            scale,
            club_fallback: found.club_fallback,
            sex_fallback: found.sex_fallback,
        })
    }

    /// [`Norms::grade_value`] with one-camera turns graded at every
    /// position (the sign rule still applies).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments, reason = "mirrors grade_value")]
    pub(crate) fn grade_value_any_position(
        &self,
        id: MetricId,
        position: Position,
        value: f64,
        unit: Unit,
        cm: Option<f64>,
        confidence: f32,
        ctx: &Context,
    ) -> Option<Grade> {
        self.grade_input(
            &Input {
                metric: id.into(),
                position,
                value,
                unit: Some(unit),
                cm,
                confidence,
            },
            ctx,
            false,
        )
    }
}

fn side_of(v: f64, centre: f64) -> Side {
    if v > centre {
        Side::High
    } else if v < centre {
        Side::Low
    } else {
        Side::On
    }
}
