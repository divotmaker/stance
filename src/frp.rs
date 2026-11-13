//! Flight Relay Protocol triggers (`frp` feature).
//!
//! Maps an FRP `shot.trigger` message onto a clip [`Trigger`] tagged with
//! the shot's [`ShotKey`]. The trigger's source is the message's `from`, the
//! launch monitor actor that saw the strike, so two launch monitors seeing
//! one swing join one clip, and a monitor's next shot cuts its previous clip
//! short if that clip's window is still running. A `shot.trigger`'s optional
//! `timestamp` is the device's wall clock in epoch milliseconds, not this
//! crate's monotonic clock: stamp each trigger with
//! [`clock::now_ns`](crate::clock::now_ns) when it arrives. Pass each shot
//! key once: a repeated `shot.trigger` for the same key would start a second
//! clip.

use std::time::Duration;

pub use flightrelay::ShotKey;
use flightrelay::{FrpBody, FrpMessage};

use crate::Trigger;

/// The trigger for a `shot.trigger` message, or `None` for any other kind.
///
/// `impact_lead` is how long before its trigger the device's impact happens
/// (zero for monitors that trigger at impact).
///
/// ```
/// use std::time::Duration;
/// use flightrelay::{FrpMessage, ShotKey, ShotTrigger};
/// use stance::frp::trigger_from_message;
///
/// let msg = FrpMessage::new("7d2f9a41c03be856", ShotTrigger::new(ShotKey::new("a1b2", 7)));
/// let t = trigger_from_message(&msg, 5_000_000_000, Duration::from_millis(800)).unwrap();
/// assert_eq!(t.source, "7d2f9a41c03be856");
/// assert_eq!(t.tag.shot_number, 7);
/// assert_eq!(t.impact_ns(), 4_200_000_000);
/// ```
#[must_use]
pub fn trigger_from_message(
    message: &FrpMessage,
    at_ns: u64,
    impact_lead: Duration,
) -> Option<Trigger<ShotKey>> {
    match &message.body {
        FrpBody::ShotTrigger(trigger) => Some(
            Trigger::new(at_ns, message.from.clone(), trigger.key.clone())
                .with_impact_lead(impact_lead),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use flightrelay::{BusAlert, Severity, ShotFinished, ShotTrigger};

    use super::*;

    #[test]
    fn only_shot_triggers_trigger() {
        let key = ShotKey::new("id", 1);
        let shot = FrpMessage::new("m", ShotTrigger::new(key.clone()));
        let t = trigger_from_message(&shot, 10, Duration::ZERO).unwrap();
        assert_eq!((t.at_ns, t.impact_lead_ns, t.tag), (10, 0, key.clone()));
        let done = FrpMessage::new("m", ShotFinished { key });
        assert!(trigger_from_message(&done, 10, Duration::ZERO).is_none());
        let parsed = FrpMessage::parse(
            r#"{"kind":"shot.trigger","from":"x","key":{"shot_id":"s","shot_number":3}}"#,
        )
        .unwrap();
        assert_eq!(
            trigger_from_message(&parsed, 0, Duration::ZERO)
                .unwrap()
                .source,
            "x"
        );
        let stamped = FrpMessage::new(
            "m",
            ShotTrigger::at(ShotKey::new("id", 2), 1_794_063_845_123),
        );
        let t = trigger_from_message(&stamped, 77, Duration::ZERO).unwrap();
        assert_eq!((t.at_ns, t.tag.shot_number), (77, 2));
        let alert = FrpMessage::new("m", BusAlert::new(Severity::Warn, "m"));
        assert!(trigger_from_message(&alert, 0, Duration::ZERO).is_none());
    }
}
