//! Guessing the camera view and the golfer's handedness.

use serde::{Deserialize, Serialize};

use super::body::{Body, Handed, Joint, Pt, View, tilt_from_vertical};
use crate::pose::Pose;

/// A heuristic guess and how sure it is (`0..=1`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Guess<T> {
    /// The guess.
    pub value: T,
    /// `0..=1`.
    pub confidence: f32,
}

/// Keypoints scoring below this are ignored by the guesses.
const MIN_SCORE: f32 = 0.3;

/// Face-on, the shoulders, hips and feet spread across the image; down the
/// line they overlap. Each feature is a width in torso lengths with the
/// value that separates the views and its weight.
const VIEW_FEATURES: [(Joint, Joint, f64, f64); 3] = [
    (Joint::LeadShoulder, Joint::TrailShoulder, 0.45, 1.0),
    (Joint::LeadHip, Joint::TrailHip, 0.3, 1.0),
    (Joint::LeadAnkle, Joint::TrailAnkle, 0.35, 1.0),
];

/// Down the line, the spine leans toward the ball by about 30–45°;
/// face-on it tilts sideways by 15° or less. Degrees that separate them.
const VIEW_SPINE_DEG: f64 = 22.0;

/// A view guess less sure than this is no guess.
const VIEW_MIN_CONFIDENCE: f64 = 0.1;

/// The handedness check ignores a face-on top less sure than this.
const HANDED_MIN_TOP_CONFIDENCE: f32 = 0.5;

/// Guesses the view from an address pose (any handedness).
///
/// Face-on, the shoulders, hips and ankles are spread across the image (the
/// shoulders about one torso length apart) and the spine is near vertical.
/// Down the line, they overlap and the spine leans 30–45° toward the ball.
/// Each of the four cues votes by how far it is from the value that
/// separates the views, scaled to ±1; the confidence is the size of the
/// mean vote. `None` when the shoulders and hips aren't visible, or the
/// votes nearly cancel (confidence under 0.1): the view is unknown.
#[must_use]
pub fn guess_view(address: &Pose) -> Option<Guess<View>> {
    let body = Body::from_keypoints(&address.keypoints, Handed::Right, MIN_SCORE)?;
    guess_view_body(&body)
}

pub(crate) fn guess_view_body(body: &Body) -> Option<Guess<View>> {
    let torso = body.torso()?;
    // Positive votes say face-on.
    let mut votes = Vec::new();
    for (a, b, split, weight) in VIEW_FEATURES {
        if let Some(w) = body.width(a, b) {
            votes.push(weight * ((w / torso - split) / split).clamp(-1.0, 1.0));
        }
    }
    let spine = body.spine_tilt(1.0)?.abs();
    votes.push(((VIEW_SPINE_DEG - spine) / VIEW_SPINE_DEG).clamp(-1.0, 1.0));
    // At most four votes.
    #[allow(clippy::cast_precision_loss, reason = "a handful of votes")]
    let mean = votes.iter().sum::<f64>() / votes.len() as f64;
    if mean.abs() < VIEW_MIN_CONFIDENCE {
        return None;
    }
    let view = if mean > 0.0 {
        View::FaceOn
    } else {
        View::DownTheLine
    };
    Some(Guess {
        value: view,
        confidence: confidence(mean.abs()),
    })
}

/// Down the line, a right-hander leans toward +x at address; face-on, a
/// right-hander's hands are on the −x (trail) side of the pelvis at the
/// top. Degrees of lean, and torso lengths of hand offset, for full
/// confidence.
const HANDED_LEAN_DEG: f64 = 20.0;
const HANDED_HANDS_TORSO: f64 = 0.5;

/// Guesses handedness from the address and top poses for a known view: down
/// the line from which way the spine leans (toward the ball), face-on from
/// which side of the pelvis the hands are at the top (the trail side; only
/// for a top with confidence 0.5 or more). A mirrored camera image makes a
/// right-hander look left-handed.
pub(crate) fn guess_handed(
    view: View,
    address: Option<&Body>,
    top: Option<(&Body, f32)>,
) -> Option<Guess<Handed>> {
    // Signed so that positive says right-handed.
    let (signed, full) = match view {
        View::DownTheLine => {
            let a = address?;
            (
                tilt_from_vertical(a.pelvis()?, a.shoulders()?, 1.0)?,
                HANDED_LEAN_DEG,
            )
        }
        View::FaceOn => {
            let (t, sure) = top?;
            if sure < HANDED_MIN_TOP_CONFIDENCE {
                return None;
            }
            let torso = address.and_then(Body::torso).or_else(|| t.torso())?;
            let (hands, pelvis): (Pt, Pt) = (t.hands()?, t.pelvis()?);
            ((pelvis.x - hands.x) / torso, HANDED_HANDS_TORSO)
        }
    };
    let value = if signed >= 0.0 {
        Handed::Right
    } else {
        Handed::Left
    };
    Some(Guess {
        value,
        confidence: confidence((signed.abs() / full).min(1.0)),
    })
}

fn confidence(v: f64) -> f32 {
    // Clamped to 0..=1.
    #[allow(clippy::cast_possible_truncation, reason = "a 0..=1 value")]
    let c = v.clamp(0.0, 1.0) as f32;
    c
}
