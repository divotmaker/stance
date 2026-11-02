//! Flight Relay Protocol triggers (`frp` feature).
//!
//! Maps an FRP `shot_trigger` event onto a clip [`Trigger`] tagged with the
//! shot's [`ShotKey`]. The trigger's source is the envelope's `device`, so
//! two launch monitors seeing one swing join one clip, and a monitor's next
//! shot cuts its previous clip short if that clip's window is still running.
//! FRP events carry no timestamp: stamp them with
//! [`clock::now_ns`](crate::clock::now_ns) when they arrive. Pass each shot
//! key once: a repeated `shot_trigger` for the same key would start a second
//! clip.

use std::time::Duration;

pub use flightrelay::ShotKey;
use flightrelay::{FrpEnvelope, FrpEvent, FrpMessage};

use crate::Trigger;

/// The trigger for a `shot_trigger` envelope, or `None` for any other event.
///
/// `impact_lead` is how long before its trigger the device's impact happens
/// (zero for monitors that trigger at impact).
///
/// ```
/// use std::time::Duration;
/// use flightrelay::{FrpEnvelope, FrpEvent, ShotKey};
/// use stance::frp::trigger_from_envelope;
///
/// let env = FrpEnvelope::new("radar-1", FrpEvent::shot_trigger(ShotKey::new("a1b2", 7)));
/// let t = trigger_from_envelope(&env, 5_000_000_000, Duration::from_millis(800)).unwrap();
/// assert_eq!(t.source, "radar-1");
/// assert_eq!(t.tag.shot_number, 7);
/// assert_eq!(t.impact_ns(), 4_200_000_000);
/// ```
#[must_use]
pub fn trigger_from_envelope(
    envelope: &FrpEnvelope,
    at_ns: u64,
    impact_lead: Duration,
) -> Option<Trigger<ShotKey>> {
    match &envelope.event {
        FrpEvent::ShotTrigger { key } => Some(
            Trigger::new(at_ns, envelope.device.clone(), key.clone()).with_impact_lead(impact_lead),
        ),
        _ => None,
    }
}

/// As [`trigger_from_envelope`], for any FRP message.
#[must_use]
pub fn trigger_from_message(
    message: &FrpMessage,
    at_ns: u64,
    impact_lead: Duration,
) -> Option<Trigger<ShotKey>> {
    match message {
        FrpMessage::Envelope(env) => trigger_from_envelope(env, at_ns, impact_lead),
        FrpMessage::Protocol(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_shot_triggers_trigger() {
        let key = ShotKey::new("id", 1);
        let shot = FrpMessage::Envelope(FrpEnvelope::new("m", FrpEvent::shot_trigger(key.clone())));
        let t = trigger_from_message(&shot, 10, Duration::ZERO).unwrap();
        assert_eq!((t.at_ns, t.impact_lead_ns, t.tag), (10, 0, key.clone()));
        let done = FrpMessage::Envelope(FrpEnvelope::new("m", FrpEvent::shot_finished(key)));
        assert!(trigger_from_message(&done, 10, Duration::ZERO).is_none());
        let parsed = FrpMessage::parse(
            r#"{"device":"x","event":{"kind":"shot_trigger","key":{"shot_id":"s","shot_number":3}}}"#,
        )
        .unwrap();
        assert_eq!(
            trigger_from_message(&parsed, 0, Duration::ZERO)
                .unwrap()
                .source,
            "x"
        );
        let alert =
            FrpMessage::parse(r#"{"kind":"alert","severity":"warn","message":"m"}"#).unwrap();
        assert!(trigger_from_message(&alert, 0, Duration::ZERO).is_none());
    }
}
