//! Golf swing analysis (`pose` feature, alpha): swing events, tempo and
//! per-view body metrics from a clip's [`PoseTrack`](crate::pose::PoseTrack),
//! and a live posture readout from single poses. Pure math on keypoints, no
//! models.
//!
//! - [`analyze_swing`] turns a pose track (and, ideally, the launch
//!   monitor's impact time, which it refines against the hands) into a
//!   [`SwingReport`]: [`SwingEvents`]
//!   (address, takeaway, top, impact, finish), [`Tempo`], the view's
//!   [`Metric`]s at each key position and their [`Series`] over every
//!   frame. [`SwingReport::write_beside`] stores it as `<clip>.swing.json`.
//! - [`posture`] measures one [`Pose`](crate::pose::Pose) for a live display
//!   (spine angle, shoulder tilt, knee flex, elbow angles, head position);
//!   [`Posture::head_shift`] compares it with the posture at address, and
//!   [`PostureFilter`] smooths the stream.
//! - [`guess_view`] tells face-on from down-the-line from an address pose.
//! - [`smooth_track`] gives the cleaned keypoints the analysis measures on.
//!
//! # Coordinates and sides
//!
//! Keypoints are image pixels, `x` to the right and `y` down. [`Joint`]
//! names keypoints by role (lead/trail) for a [`Handed`] golfer, so a
//! left-hander mirrors a right-hander exactly. [`Setup`] says which image
//! direction is towards the target (face-on: +x for a right-hander) or the
//! ball (down the line: +x for a right-hander). A mirrored image (a
//! selfie-style preview) makes a right-hander look left-handed:
//! [`SwingOptions::mirrored`] and [`unmirror`] analyse it with the golfer's
//! true hand.
//!
//! # 2D proxies
//!
//! Every metric is measured in one camera's image plane. They are proxies
//! for coaching feedback, not 3D joint angles: a lean towards or away from
//! the camera is invisible, and rotations show only as foreshortening.
//! The hip and shoulder turns are estimated from that foreshortening, face-on
//! only. [`MetricId`] defines each one (formula, view, unit, limits). Distances are
//! fractions of the golfer's torso length at address, and centimetres when
//! the golfer's height is given ([`BodyScale`]).
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use stance::golf::{Handed, SwingOptions, View, analyze_swing};
//! use stance::pose::PoseTrack;
//!
//! let track = PoseTrack::read(Path::new("clips/shot-0001.pose.json"))?;
//! let report = analyze_swing(&track, &SwingOptions {
//!     view: Some(View::FaceOn),
//!     handed: Handed::Right,
//!     impact_ns: Some(2_035_366_667), // from the sidecar / launch monitor
//!     height_cm: Some(180.0),
//!     ..SwingOptions::default()
//! })?;
//! if let Some(tempo) = report.tempo {
//!     println!("tempo {:.1}:1", tempo.ratio);
//! }
//! report.write_beside(Path::new("clips/shot-0001.avi"))?; // shot-0001.swing.json
//! # Ok::<(), stance::Error>(())
//! ```

mod body;
mod events;
mod metrics;
mod posture;
mod report;
mod smooth;
mod view;

pub use body::{Handed, Joint, Setup, View};
pub use events::{ImpactSource, SwingEvent, SwingEvents, Tempo};
pub use metrics::{
    BodyScale, Metric, MetricId, MetricSeries, Positions, Series, TORSO_PER_HEIGHT, Unit,
};
pub use posture::{HeadShift, Posture, PostureFilter, posture, unmirror};
pub use report::{
    MAX_SWING_REPORT_BYTES, SWING_REPORT_VERSION, SwingOptions, SwingReport, analyze_swing,
};
pub use smooth::{Smoothing, smooth_track};
pub use view::{Guess, guess_view};

#[cfg(test)]
mod tests;
