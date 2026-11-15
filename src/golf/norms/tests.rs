//! `golf::norms` tests: the built-in table, band fits, lookups, scaling,
//! and grading synthetic tour and fault swings.

// Test fixtures use small exact values.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "synthetic fixtures"
)]

use super::{
    Band, Confidence, Direction, Measure, Miss, NormUnit, Query, Range, Scale, SdSource,
    TempoMeasure,
};
use crate::golf::testfigure::*;
use crate::golf::*;

fn tour_driver() -> Context {
    Context::new(ClubClass::Driver, Sex::Male, Reference::Tour)
}

/// A table of one band.
fn table(band: Band) -> Norms {
    Norms::from_bands(1, std::collections::BTreeMap::new(), vec![band])
}

/// The hip turn at impact as the built-in table has it: 38 ± 5, fade and
/// draw bands 3° either side.
fn hip_turn_impact() -> Band {
    let mut b = Band::new(
        MetricId::HipTurn,
        Position::Impact,
        Reference::Tour,
        38.0,
        5.0,
        NormUnit::Degrees,
    );
    b.shape_offset = Some(3.0);
    b.low = Some(Miss::new(Direction::Closed, "hang-back"));
    b.high = Some(Miss::new(Direction::Open, "slice"));
    b
}

fn grade_deg(norms: &Norms, id: MetricId, position: Position, v: f64) -> Option<Grade> {
    norms.grade_value(id, position, v, Unit::Degrees, None, 0.9, &tour_driver())
}

#[test]
fn the_builtin_table_loads_and_names_only_known_ids() {
    let norms = Norms::builtin().unwrap();
    assert_eq!(norms.skipped(), 0);
    assert_eq!(norms.version, 1);
    assert!(!norms.revision.is_empty());
    assert!(norms.bands().len() > 100, "{}", norms.bands().len());
    let mut seen = std::collections::HashSet::new();
    // The fade-side center offsets of the shape-linked rows (the draw side
    // is the negation): degrees, and inches for the sway.
    let offsets = [
        (MetricId::HipTurn, 3.0),
        (MetricId::ShoulderTurn, 3.0),
        (MetricId::ShoulderTilt, -3.0),
        (MetricId::HipSway, -0.75),
    ];
    for b in norms.bands() {
        if let Measure::Metric(id) = b.metric {
            assert!(MetricId::ALL.contains(&id));
            let unit = match id.unit() {
                Unit::Degrees => NormUnit::Degrees,
                Unit::Torso => NormUnit::Inches,
                Unit::Ratio => NormUnit::Ratio,
            };
            assert_eq!(b.unit, unit, "{b:?}");
            // Every metric with a band is measured at that position: one
            // way to tell is the stick figure below; here, the turns.
            if matches!(id, MetricId::HipTurn | MetricId::ShoulderTurn) {
                assert_ne!(b.position, Position::Address, "{b:?}");
            }
            if id == MetricId::SpineForwardBend {
                assert_ne!(b.position, Position::Finish);
            }
            let linked = offsets
                .iter()
                .find(|(o, _)| *o == id)
                .filter(|_| b.position == Position::Impact);
            match linked {
                Some((_, o)) => assert_eq!(b.shape_offset, Some(*o), "{b:?}"),
                None => assert_eq!(b.shape_offset, None, "{b:?}"),
            }
        } else {
            assert_eq!(b.position, Position::Swing);
            assert!(matches!(b.unit, NormUnit::Ratio | NormUnit::Seconds));
        }
        assert!(b.sd > 0.0 && b.center.is_finite(), "{b:?}");
        assert!(!b.sources.is_empty(), "{b:?}");
        for s in &b.sources {
            assert!(norms.sources.contains_key(s), "source {s}");
        }
        assert!(
            b.low.is_some() || b.high.is_some(),
            "a band with no miss: {b:?}"
        );
        assert!(
            seen.insert((b.metric, b.position, b.club, b.sex, b.reference)),
            "duplicate {b:?}"
        );
        // High confidence needs several sources (NORMS.md's rule).
        if b.confidence == Confidence::High {
            assert!(b.sources.len() > 1, "{b:?}");
        }
        // Turns are signed positive open, so their words name closed below
        // the center and open above it (`Grade::phrase` relies on it).
        for (miss, want) in [(&b.low, Direction::Closed), (&b.high, Direction::Open)] {
            if let Some(m) = miss
                && matches!(m.direction, Direction::Open | Direction::Closed)
            {
                assert_eq!(m.direction, want, "{b:?}");
            }
        }
        assert_ne!(b.reference, Reference::Personal);
    }
    // Each shape-linked quantity has rows (club, sex and reference
    // variants).
    for (id, _) in offsets {
        assert!(
            norms
                .bands()
                .iter()
                .any(|b| b.metric == Measure::Metric(id) && b.shape_linked())
        );
    }
    // The finish back bend is a reference, not graded; the top shoulder
    // tilt is graded with low confidence.
    for b in norms.bands() {
        let finish_bend = b.metric == Measure::Metric(MetricId::LateralSpineTilt)
            && b.position == Position::Finish;
        assert_eq!(b.graded, !finish_bend, "{b:?}");
        if b.metric == Measure::Metric(MetricId::ShoulderTilt) && b.position == Position::Top {
            assert_eq!(b.confidence, Confidence::Low);
        }
    }
    // Every source key is used.
    for key in norms.sources.keys() {
        assert!(
            norms.bands().iter().any(|b| b.sources.contains(key)),
            "{key}"
        );
    }
}

#[test]
fn turn_words_follow_the_values_sign() {
    let norms = Norms::builtin().unwrap();
    let shoulders = |v| grade_deg(norms, MetricId::ShoulderTurn, Position::Impact, v).unwrap();
    // 15° open against the tour's 30° open: short of the center on its
    // side, so less open, not more closed.
    let g = shoulders(15.0);
    assert_eq!(
        (g.fit, g.side, g.direction),
        (Fit::Far, Side::Low, Some(Direction::Closed))
    );
    assert_eq!(g.phrase(), Some("less open"));
    assert_eq!(
        g.fit_word().unwrap(),
        "Shoulder turn less open than tour: shoulders behind, push or hook"
    );
    // One camera reads a size: −15 is 15° open too.
    assert_eq!(shoulders(-15.0).fit_word(), g.fit_word());
    // In the draw band, short of the center.
    assert_eq!(
        shoulders(26.0).fit_word().unwrap(),
        "Shoulder turn less open: draw side"
    );
    // Beyond the center: more open.
    assert_eq!(
        shoulders(45.0).fit_word().unwrap(),
        "Shoulder turn more open than tour: out-to-in, pull or slice pattern"
    );
    assert_eq!(
        shoulders(34.0).fit_word().unwrap(),
        "Shoulder turn more open: fade side"
    );
    // On the far side of zero: more closed.
    let mut far = g.clone();
    far.value = -5.0;
    far.z = (far.value - far.center) / far.sd;
    assert_eq!(far.phrase(), Some("more closed"));
    assert_eq!(
        far.fit_word().unwrap(),
        "Shoulder turn more closed than tour: shoulders behind, push or hook"
    );
    // Zero is not short of the center: it has no direction of its own.
    far.value = 0.0;
    assert_eq!(far.phrase(), Some("more closed"));
    // A closed center (the top, 98° closed): 75° closed is less closed,
    // 110° closed more closed.
    let ctx = tour_driver();
    let top = |v| {
        norms
            .grade_value_any_position(
                MetricId::ShoulderTurn,
                Position::Top,
                v,
                Unit::Degrees,
                None,
                0.9,
                &ctx,
            )
            .unwrap()
    };
    let g = top(75.0);
    assert_eq!((g.value, g.direction), (-75.0, Some(Direction::Open)));
    assert_eq!(
        g.fit_word().unwrap(),
        "Shoulder turn less closed than tour: short backswing, arms-only, loss of speed"
    );
    assert_eq!(
        top(110.0).fit_word().unwrap(),
        "Shoulder turn more closed than tour: over-swing, loss of control"
    );
    // Against amateurs (hips at the top, 43.8° closed).
    let amateur = Context::new(ClubClass::Driver, Sex::Male, Reference::Amateur);
    let g = norms
        .grade_value_any_position(
            MetricId::HipTurn,
            Position::Top,
            25.0,
            Unit::Degrees,
            None,
            0.9,
            &amateur,
        )
        .unwrap();
    assert_eq!(
        g.fit_word().unwrap(),
        "Hip turn less closed than amateurs: restricted hip turn, arms-only backswing, loss of coil"
    );
    // Positional and magnitude words are unchanged short of the center:
    // hips 1 in toward the target at impact against 4 in toward.
    let sway = norms
        .grade_value(
            MetricId::HipSway,
            Position::Impact,
            2.54,
            Unit::Torso,
            Some(2.54),
            0.9,
            &ctx,
        )
        .unwrap();
    assert_eq!(sway.direction, Some(Direction::Away));
    assert_eq!(sway.phrase(), Some(Direction::Away.phrase()));
    // A left-handed swing reads the same words.
    let mut f = tour_face_on();
    f.shoulder_turn.impact = 15.0;
    let words = |handed| {
        let grades = norms.grade_report(&tour_report(&f, handed), &ctx);
        find_grade(&grades, MetricId::ShoulderTurn, Position::Impact)
            .unwrap()
            .fit_word()
    };
    let right = words(Handed::Right).unwrap();
    assert!(
        right.starts_with("Shoulder turn less open than tour"),
        "{right}"
    );
    assert_eq!(words(Handed::Left).unwrap(), right);
}

#[test]
fn a_shape_linked_band_fits_both_fade_draw_near_far() {
    let norms = table(hip_turn_impact());
    let cases = [
        (38.0, Fit::Both, Side::On),
        (40.0, Fit::Both, Side::On),
        (44.0, Fit::Fade, Side::High),
        (33.0, Fit::Draw, Side::Low),
        (48.0, Fit::Near, Side::High),
        (52.0, Fit::Far, Side::High),
        (24.0, Fit::Far, Side::Low),
        (46.0, Fit::Fade, Side::High),
        (46.1, Fit::Near, Side::High),
    ];
    for (v, fit, side) in cases {
        let g = grade_deg(&norms, MetricId::HipTurn, Position::Impact, v).unwrap();
        assert_eq!((g.fit, g.side), (fit, side), "{v}: {g:?}");
        assert_eq!(g.value, v);
        assert!((g.z - (v - 38.0) / 5.0).abs() < 1e-12);
        assert_eq!(
            g.fade,
            Some(Range {
                center: 41.0,
                sd: 5.0
            })
        );
        assert_eq!(
            g.draw,
            Some(Range {
                center: 35.0,
                sd: 5.0
            })
        );
        assert!((g.z_fade.unwrap() - (v - 41.0) / 5.0).abs() < 1e-12);
        let miss = g.miss.as_ref().map(|m| m.direction);
        match fit {
            Fit::Near | Fit::Far => {
                let want = if side == Side::High {
                    Direction::Open
                } else {
                    Direction::Closed
                };
                assert_eq!(miss, Some(want));
                assert_eq!(g.direction, Some(want));
            }
            _ => assert_eq!(miss, None),
        }
    }
    let word = |v| {
        grade_deg(&norms, MetricId::HipTurn, Position::Impact, v)
            .unwrap()
            .fit_word()
    };
    assert_eq!(word(38.0), None);
    assert_eq!(word(44.0).unwrap(), "Hip turn more open: fade side");
    assert_eq!(word(33.0).unwrap(), "Hip turn less open: draw side");
    assert_eq!(word(52.0).unwrap(), "Hip turn more open than tour: slice");
    assert_eq!(
        word(24.0).unwrap(),
        "Hip turn less open than tour: hang-back"
    );
}

#[test]
fn touching_bands_keep_an_overlap_with_offset_3() {
    let mut b = Band::new(
        MetricId::ShoulderTilt,
        Position::Impact,
        Reference::Tour,
        39.0,
        4.5,
        NormUnit::Degrees,
    );
    // 3, not 4: with an SD of 4.5 the fade and draw bands overlap by ±1.5°
    // around the center rather than ±0.5°.
    b.shape_offset = Some(-3.0);
    b.low = Some(Miss::new(Direction::Less, "level"));
    b.high = Some(Miss::new(Direction::More, "hanging back"));
    let norms = table(b);
    for (v, fit) in [
        (39.0, Fit::Both),
        (37.6, Fit::Both),
        (40.4, Fit::Both),
        (37.4, Fit::Fade),
        (40.6, Fit::Draw),
    ] {
        let g = grade_deg(&norms, MetricId::ShoulderTilt, Position::Impact, v).unwrap();
        assert_eq!(g.fit, fit, "{v}");
    }
}

#[test]
fn shape_neutral_is_never_fade_or_draw() {
    let mut b = Band::new(
        MetricId::SpineForwardBend,
        Position::Address,
        Reference::Tour,
        39.0,
        5.0,
        NormUnit::Degrees,
    );
    b.low = Some(Miss::new(Direction::Less, "upright"));
    b.high = Some(Miss::new(Direction::More, "bent"));
    let norms = table(b);
    let mut fits = std::collections::HashSet::new();
    for i in 0..=400 {
        let v = 9.0 + f64::from(i) * 0.15;
        let g = grade_deg(&norms, MetricId::SpineForwardBend, Position::Address, v).unwrap();
        assert!(g.fade.is_none() && g.z_fade.is_none() && g.z_draw.is_none());
        let want = match g.z.abs() {
            z if z <= 1.0 + 1e-9 => Fit::Both,
            z if z <= 2.0 + 1e-9 => Fit::Near,
            _ => Fit::Far,
        };
        assert_eq!(g.fit, want, "{v}");
        assert_eq!(g.side == Side::On, g.fit == Fit::Both);
        fits.insert(g.fit);
    }
    assert_eq!(fits.len(), 3);
}

#[test]
fn a_personal_band_with_one_shape_never_fits_both() {
    let mut b = hip_turn_impact();
    b.reference = Reference::Personal;
    b.fade = Some(Range {
        center: 45.0,
        sd: 4.0,
    });
    let norms = table(b.clone());
    let ctx = Context::new(ClubClass::Driver, Sex::Male, Reference::Personal);
    let mut fits = std::collections::HashSet::new();
    for i in 0..=300 {
        let v = 20.0 + f64::from(i) * 0.15;
        let g = norms
            .grade_value(
                MetricId::HipTurn,
                Position::Impact,
                v,
                Unit::Degrees,
                None,
                0.9,
                &ctx,
            )
            .unwrap();
        assert!(
            matches!(g.fit, Fit::Fade | Fit::Near | Fit::Far),
            "{v}: {g:?}"
        );
        assert!(g.draw.is_none() && g.z_draw.is_none());
        fits.insert(g.fit);
    }
    assert_eq!(fits.len(), 3);
    // Neither sub-band: no grade.
    b.fade = None;
    let norms = table(b);
    assert!(
        norms
            .grade_value(
                MetricId::HipTurn,
                Position::Impact,
                38.0,
                Unit::Degrees,
                None,
                0.9,
                &ctx
            )
            .is_none()
    );
    // The built-in table has no personal bands.
    assert!(
        Norms::builtin()
            .unwrap()
            .grade_value(
                MetricId::HipTurn,
                Position::Impact,
                38.0,
                Unit::Degrees,
                None,
                0.9,
                &ctx
            )
            .is_none()
    );
}

#[test]
fn one_sided_bands_clamp_to_both() {
    let norms = Norms::builtin().unwrap();
    // The finish hip turn (113 ± 8) has no high miss. Graded with the rule
    // lifted: the one-camera turn is impact-only.
    let at = |v| {
        norms
            .grade_value_any_position(
                MetricId::HipTurn,
                Position::Finish,
                v,
                Unit::Degrees,
                None,
                0.9,
                &tour_driver(),
            )
            .unwrap()
    };
    let g = at(140.0);
    assert_eq!(
        (g.fit, g.side, g.miss.as_ref()),
        (Fit::Both, Side::High, None)
    );
    assert_eq!(g.fit_word(), None);
    let g = at(90.0);
    assert_eq!((g.fit, g.side), (Fit::Far, Side::Low));
    assert_eq!(g.miss.unwrap().direction, Direction::Closed);
    // Downswing time: slow is a miss, fast never is.
    let tempo = |down: f64| Tempo {
        backswing_ns: 750_000_000,
        downswing_ns: (down * 1e9) as u64,
        ratio: 0.75 / down,
    };
    let grades = norms.grade_tempo(&tempo(0.1), &tour_driver());
    let down = grades
        .iter()
        .find(|g| g.metric == Measure::Tempo(TempoMeasure::Downswing))
        .unwrap();
    assert_eq!((down.fit, down.side), (Fit::Both, Side::Low));
}

#[test]
fn unknown_rows_are_skipped_not_errors() {
    let row = |metric: &str, position: &str, club: &str, direction: &str| {
        format!(
            r#"{{ "metric": "{metric}", "position": "{position}", "club": {club}, "sex": null,
                 "reference": "tour", "center": 1, "sd": 1, "unit": "degrees",
                 "sd_source": "proxy", "confidence": "low", "future_field": [1, 2],
                 "low": {{ "direction": "{direction}", "tends_to": "x" }} }}"#
        )
    };
    let json = format!(
        r#"{{ "version": 1, "revision": "r", "sources": {{}}, "bands": [{}, {}, {}, {}, {}] }}"#,
        row("wrist_hinge", "top", "null", "less"),
        row("hip_tilt", "p9", "null", "less"),
        row("hip_tilt", "top", r#""putter""#, "less"),
        row("hip_tilt", "top", "null", "sideways"),
        row("hip_tilt", "top", r#""driver""#, "less"),
    );
    let norms = Norms::from_json(&json).unwrap();
    assert_eq!(norms.skipped(), 4);
    assert_eq!(norms.bands().len(), 1);
    assert_eq!(norms.bands()[0].club, Some(ClubClass::Driver));
    // A known row without a center is an error, as is version 2.
    let broken = json.replacen(r#""center": 1, "#, "", 5);
    assert!(matches!(
        Norms::from_json(&broken),
        Err(crate::Error::InvalidArgument(_))
    ));
    let v2 = json.replace(r#""version": 1"#, r#""version": 2"#);
    assert!(Norms::from_json(&v2).is_err());
    // A unit that doesn't match the metric, and a zero SD, are errors.
    let inches = json.replace(r#""unit": "degrees""#, r#""unit": "inches""#);
    assert!(Norms::from_json(&inches).is_err());
    let zero = json.replace(r#""sd": 1"#, r#""sd": 0"#);
    assert!(Norms::from_json(&zero).is_err());
}

#[test]
fn the_builtin_table_never_panics_on_a_bad_embed() {
    // Trimmed so the last cut always lands inside the closing brace, whatever
    // line endings the checkout gave the embedded file.
    let json = Norms::builtin_json().trim_end();
    for cut in [0, 1, json.len() / 3, json.len() / 2, json.len() - 1] {
        assert!(Norms::from_json(&json[..cut]).is_err(), "cut at {cut}");
    }
    assert!(Norms::from_json("{}").is_err());
    assert!(Norms::from_json("[]").is_err());
}

#[test]
fn lookup_falls_back_by_sex_then_club_and_null_sex_wins() {
    let band = |club: Option<ClubClass>, sex: Option<Sex>, center: f64| {
        let mut b = Band::new(
            MetricId::HipTilt,
            Position::Impact,
            Reference::Tour,
            center,
            5.0,
            NormUnit::Degrees,
        );
        b.club = club;
        b.sex = sex;
        b
    };
    let q = |club, sex| {
        Query::new(
            MetricId::HipTilt,
            Position::Impact,
            club,
            sex,
            Reference::Tour,
        )
    };
    let norms = Norms::from_bands(
        1,
        std::collections::BTreeMap::new(),
        vec![
            band(Some(ClubClass::Driver), Some(Sex::Male), 8.0),
            band(Some(ClubClass::MidIron), Some(Sex::Male), 7.0),
        ],
    );
    // A woman with a wedge: the mid-iron male band, both flags.
    let l = norms.band(&q(ClubClass::Wedge, Sex::Female)).unwrap();
    assert_eq!(l.band.center, 7.0);
    assert!(l.club_fallback && l.sex_fallback);
    // The flags reach the grade.
    let ctx = Context::new(ClubClass::Wedge, Sex::Female, Reference::Tour);
    let g = norms
        .grade_value(
            MetricId::HipTilt,
            Position::Impact,
            7.0,
            Unit::Degrees,
            None,
            0.9,
            &ctx,
        )
        .unwrap();
    assert!(g.club_fallback && g.sex_fallback);
    let l = norms.band(&q(ClubClass::Driver, Sex::Male)).unwrap();
    assert!(!l.club_fallback && !l.sex_fallback && l.band.center == 8.0);
    // A `club: null` row wins over the wedge fallback.
    let mut with_any = norms.bands().to_vec();
    with_any.push(band(None, Some(Sex::Male), 6.0));
    let norms = Norms::from_bands(1, std::collections::BTreeMap::new(), with_any);
    let l = norms.band(&q(ClubClass::Wedge, Sex::Male)).unwrap();
    assert_eq!(l.band.center, 6.0);
    assert!(!l.club_fallback);
    // A `sex: null` row wins over the male row for a woman.
    let mut with_either = norms.bands().to_vec();
    with_either.push(band(Some(ClubClass::Driver), None, 5.0));
    let norms = Norms::from_bands(1, std::collections::BTreeMap::new(), with_either);
    let l = norms.band(&q(ClubClass::Driver, Sex::Female)).unwrap();
    assert_eq!(l.band.center, 5.0);
    assert!(!l.sex_fallback);
    // A man keeps his own row.
    assert_eq!(
        norms
            .band(&q(ClubClass::Driver, Sex::Male))
            .unwrap()
            .band
            .center,
        8.0
    );
    // No row at all: none.
    assert!(
        norms
            .band(&Query::new(
                MetricId::HipTilt,
                Position::Top,
                ClubClass::Driver,
                Sex::Male,
                Reference::Tour
            ))
            .is_none()
    );
    // The built-in table: a woman's wedge hip sway at impact takes the
    // female mid-iron row.
    let l = Norms::builtin()
        .unwrap()
        .band(&Query::new(
            MetricId::HipSway,
            Position::Impact,
            ClubClass::Wedge,
            Sex::Female,
            Reference::Tour,
        ))
        .unwrap();
    assert_eq!(l.band.center, 3.7);
    assert!(l.club_fallback && !l.sex_fallback);
}

#[test]
fn inches_bands_use_the_height_or_an_assumed_one() {
    let norms = Norms::builtin().unwrap();
    // Hip sway at the top: −0.6 ± 1.5 in.
    let grade = |v: f64, cm: Option<f64>, scale: Option<BodyScale>| {
        let mut ctx = tour_driver();
        ctx.scale = scale;
        norms.grade_value(
            MetricId::HipSway,
            Position::Top,
            v,
            Unit::Torso,
            cm,
            0.9,
            &ctx,
        )
    };
    let g = grade(-0.02, Some(-0.6 * 2.54), None).unwrap();
    assert_eq!(g.scale, Scale::Measured);
    assert!((g.value + 0.6).abs() < 1e-12 && g.fit == Fit::Both);
    assert_eq!(g.unit, NormUnit::Inches);
    // No centimeters: the body scale and an assumed 177 cm.
    let scale = BodyScale {
        torso_px: 200.0,
        upright_px: Some(220.0),
        cm_per_px: None,
    };
    let g = grade(-0.1, None, Some(scale)).unwrap();
    assert_eq!(g.scale, Scale::Assumed);
    let cm = -0.1 * 200.0 * TORSO_PER_HEIGHT * 177.0 / 220.0;
    assert!((g.value - cm / 2.54).abs() < 1e-9, "{g:?}");
    // A known height in the scale counts as measured.
    let measured = BodyScale {
        cm_per_px: Some(0.25),
        ..scale
    };
    let g = grade(-0.1, None, Some(measured)).unwrap();
    assert_eq!(g.scale, Scale::Measured);
    assert!((g.value - (-0.1 * 200.0 * 0.25 / 2.54)).abs() < 1e-9);
    // Neither: no grade, and a scale without the upright torso can't
    // assume a height either.
    assert!(grade(-0.1, None, None).is_none());
    let old = BodyScale {
        upright_px: None,
        ..scale
    };
    assert!(grade(-0.1, None, Some(old)).is_none());
    // A woman's assumed height is 164 cm.
    let mut ctx = Context::new(ClubClass::Driver, Sex::Female, Reference::Tour);
    ctx.scale = Some(scale);
    let g = norms
        .grade_value(
            MetricId::HipSway,
            Position::Top,
            -0.1,
            Unit::Torso,
            None,
            0.9,
            &ctx,
        )
        .unwrap();
    let cm = -0.1 * 200.0 * TORSO_PER_HEIGHT * 164.0 / 220.0;
    assert!((g.value - cm / 2.54).abs() < 1e-9);
    // Degree bands need no scale; a unit that isn't the metric's is
    // refused.
    assert!(grade_deg(norms, MetricId::HipTilt, Position::Top, -8.0).is_some());
    assert!(
        norms
            .grade_value(
                MetricId::HipTilt,
                Position::Top,
                -8.0,
                Unit::Torso,
                None,
                0.9,
                &tour_driver()
            )
            .is_none()
    );
}

#[test]
fn one_camera_turns_take_the_bands_sign() {
    let norms = Norms::builtin().unwrap();
    let ctx = tour_driver();
    let any = |id, position, v| {
        norms.grade_value_any_position(id, position, v, Unit::Degrees, None, 0.9, &ctx)
    };
    // 40° at the top takes the band's sign: −40 against −40.
    let g = any(MetricId::HipTurn, Position::Top, 40.0).unwrap();
    assert_eq!((g.value, g.fit, g.side), (-40.0, Fit::Both, Side::On));
    // Over-turned at the top: more closed.
    let g = any(MetricId::HipTurn, Position::Top, 55.0).unwrap();
    assert_eq!((g.value, g.side), (-55.0, Side::Low));
    assert_eq!(g.miss.unwrap().direction, Direction::Closed);
    // A negative input is a size too.
    assert_eq!(
        any(MetricId::HipTurn, Position::Top, -40.0).unwrap().value,
        -40.0
    );
    // 38 at impact: +38, Both.
    let g = grade_deg(norms, MetricId::HipTurn, Position::Impact, 38.0).unwrap();
    assert_eq!((g.value, g.fit), (38.0, Fit::Both));
    // At P5 the hips' center (−11) is within 2 SD (6) of zero: no grade.
    assert!(any(MetricId::HipTurn, Position::ArmParallelDown, 11.0).is_none());
    // The impact-only rule: no grade at the top, at P3 or at P5, whatever
    // the rows.
    for position in [
        Position::ArmParallelBack,
        Position::Top,
        Position::ArmParallelDown,
        Position::Finish,
    ] {
        assert!(grade_deg(norms, MetricId::ShoulderTurn, position, 61.0).is_none());
        assert!(grade_deg(norms, MetricId::HipTurn, position, 40.0).is_none());
    }
    // Other metrics keep their own sign.
    let g = grade_deg(norms, MetricId::ShoulderTilt, Position::Top, 38.0).unwrap();
    assert_eq!((g.value, g.fit), (38.0, Fit::Far));
}

#[test]
fn low_confidence_is_not_graded() {
    let norms = Norms::builtin().unwrap();
    let at = |confidence, min: Option<f32>| {
        let mut ctx = tour_driver();
        if let Some(m) = min {
            ctx.min_confidence = m;
        }
        norms.grade_value(
            MetricId::HipTilt,
            Position::Top,
            -8.0,
            Unit::Degrees,
            None,
            confidence,
            &ctx,
        )
    };
    assert!(at(0.3, None).is_none());
    assert!(at(0.5, None).is_some());
    assert_eq!(at(0.5, None).unwrap().confidence, 0.5);
    assert!(at(0.3, Some(0.2)).is_some());
    assert!(at(0.6, Some(0.7)).is_none());
}

#[test]
fn ungraded_rows_are_found_but_not_graded() {
    let norms = Norms::builtin().unwrap();
    let q = Query::new(
        MetricId::LateralSpineTilt,
        Position::Finish,
        ClubClass::Driver,
        Sex::Male,
        Reference::Tour,
    );
    let l = norms.band(&q).unwrap();
    assert!(!l.band.graded && l.band.center == 33.0);
    assert!(grade_deg(norms, MetricId::LateralSpineTilt, Position::Finish, 33.0).is_none());
    // The flag round-trips through JSON as `"grade": false`.
    let json = serde_json::to_string(l.band).unwrap();
    assert!(json.contains(r#""grade":false"#), "{json}");
    let graded = serde_json::to_string(&hip_turn_impact()).unwrap();
    assert!(!graded.contains("grade"), "{graded}");
}

#[test]
fn tempo_grades_ratio_and_downswing() {
    let norms = Norms::builtin().unwrap();
    let tempo = |back: f64, down: f64| Tempo {
        backswing_ns: (back * 1e9) as u64,
        downswing_ns: (down * 1e9) as u64,
        ratio: back / down,
    };
    let find = |grades: &[Grade], m: TempoMeasure| {
        grades
            .iter()
            .find(|g| g.metric == Measure::Tempo(m))
            .cloned()
            .unwrap()
    };
    let g = norms.grade_tempo(&tempo(0.75, 0.25), &tour_driver());
    assert_eq!(g.len(), 2);
    let ratio = find(&g, TempoMeasure::Ratio);
    assert_eq!(
        (ratio.fit, ratio.unit, ratio.position),
        (Fit::Both, NormUnit::Ratio, Position::Swing)
    );
    assert!((ratio.value - 3.0).abs() < 1e-9);
    let down = find(&g, TempoMeasure::Downswing);
    assert_eq!((down.fit, down.unit), (Fit::Both, NormUnit::Seconds));
    assert!((down.value - 0.25).abs() < 1e-9);
    // A quick backswing (1.75:1) and a slow downswing (0.4 s).
    let g = norms.grade_tempo(&tempo(0.7, 0.4), &tour_driver());
    let ratio = find(&g, TempoMeasure::Ratio);
    assert_eq!((ratio.fit, ratio.side), (Fit::Far, Side::Low));
    assert_eq!(
        ratio.fit_word().unwrap(),
        "Tempo less than tour: quick backswing"
    );
    let down = find(&g, TempoMeasure::Downswing);
    assert_eq!((down.fit, down.side), (Fit::Far, Side::High));
    // A woman's downswing band is 0.30 s.
    let woman = Context::new(ClubClass::Driver, Sex::Female, Reference::Tour);
    let down = find(
        &norms.grade_tempo(&tempo(0.9, 0.3), &woman),
        TempoMeasure::Downswing,
    );
    assert!(down.z.abs() < 1e-9 && !down.sex_fallback);
    let ratio = find(
        &norms.grade_tempo(&tempo(0.9, 0.3), &woman),
        TempoMeasure::Ratio,
    );
    assert!(ratio.sex_fallback);
}

/// The golfer's height in the tour-figure tests, and the figure's pixels
/// per inch with it (its upright torso is [`TORSO`]).
const TOUR_HEIGHT_CM: f64 = 177.0;

fn inches(v: f64) -> f64 {
    v * 2.54 / (TORSO_PER_HEIGHT * TOUR_HEIGHT_CM / TORSO)
}

/// The face-on figure at the driver tour bands (male): hip sway −0.6 in at
/// the top and +4 in at impact, lateral spine tilt 4, 4.5 and 18° (the
/// chest sway and sway gap follow from the two and land inside their
/// bands), shoulder tilt 15 / −38 / 25°, hip tilt 1 / −8 / 8°, impact turns
/// 38° (hips) and 30° (shoulders), pelvis lift −1.6 / +0.9 in, head shift
/// −1 / −0.5 in, head height −0.25 / −0.75 in.
fn tour_face_on() -> Figure {
    let mut f = Figure::face_on();
    f.pelvis_dx = Phases {
        address: 0.0,
        top: inches(-0.6),
        impact: inches(4.0),
        finish: inches(5.0),
    };
    f.pelvis_dy = Phases {
        address: 0.0,
        top: inches(1.6),
        impact: inches(-0.9),
        finish: inches(-1.0),
    };
    f.spine = Phases {
        address: -4.0,
        top: -4.5,
        impact: -18.0,
        finish: -FINISH_LEAN,
    };
    f.shoulder_tilt = Phases {
        address: 15.0,
        top: -38.0,
        impact: 25.0,
        finish: 0.0,
    };
    f.hip_tilt = Phases {
        address: 1.0,
        top: -8.0,
        impact: 8.0,
        finish: 0.0,
    };
    f.hip_turn.impact = 38.0;
    f.shoulder_turn.impact = 30.0;
    f.head_dx = Phases {
        address: 0.0,
        top: inches(-1.0),
        impact: inches(-0.5),
        finish: 40.0,
    };
    f.head_dy = Phases {
        address: 0.0,
        top: inches(0.25),
        impact: inches(0.75),
        finish: -10.0,
    };
    f
}

/// The down-the-line figure at the driver tour bands (male): forward bend
/// 39° at address and 35° at impact (so 4° of lost posture), early
/// extension 0.5 / 0.4 in, hands 3 in deep at the top, pelvis lift and
/// head height as face-on.
fn tour_down_the_line() -> Figure {
    let mut f = Figure::down_the_line();
    f.spine = Phases {
        address: 39.0,
        top: 37.0,
        impact: 35.0,
        finish: 10.0,
    };
    f.pelvis_dx = Phases {
        address: 0.0,
        top: inches(0.5),
        impact: inches(0.4),
        finish: 10.0,
    };
    f.pelvis_dy = tour_face_on().pelvis_dy;
    f.head_dy = tour_face_on().head_dy;
    // The shoulders turned side-on at the top show their width, with the
    // trail shoulder 3 in in front of the hands (which are 95 px behind
    // the shoulders' midpoint).
    f.shoulder_half.top = top_hands_behind() - inches(3.0);
    f
}

/// How far the down-the-line figure's hands are behind its shoulders'
/// midpoint at the top (pixels).
fn top_hands_behind() -> f64 {
    -ARM * (-HAND_SWING.to_radians()).sin()
}

fn tour_report(figure: &Figure, handed: Handed) -> SwingReport {
    let track = match handed {
        Handed::Right => figure.track(),
        Handed::Left => mirrored(&figure.track()),
    };
    let mut opts = options(figure.view, handed, Some(IMPACT_S));
    opts.smoothing = raw();
    opts.height_cm = Some(TOUR_HEIGHT_CM);
    analyze_swing(&track, &opts).unwrap()
}

/// The P3 or P5 slot of a `Positions`.
fn slot(pos: &mut Positions, p: Position) -> &mut Option<f64> {
    match p {
        Position::ArmParallelBack => &mut pos.arm_parallel_back,
        _ => &mut pos.arm_parallel_down,
    }
}

fn find_grade(grades: &[Grade], id: MetricId, position: Position) -> Option<&Grade> {
    grades
        .iter()
        .find(|g| g.metric == Measure::Metric(id) && g.position == position)
}

#[test]
fn the_tour_figure_grades_every_band_both() {
    let norms = Norms::builtin().unwrap();
    let ctx = tour_driver();
    for figure in [tour_face_on(), tour_down_the_line()] {
        let report = tour_report(&figure, Handed::Right);
        assert!(near(report.scale.unwrap().upright_px, TORSO, 0.01));
        let grades = norms.grade_report(&report, &ctx);
        // The cells with a graded band, the one-camera turns at impact only.
        let mut expected = 0;
        for m in &report.metrics {
            for &p in Position::KEY {
                let turn = matches!(m.id, MetricId::HipTurn | MetricId::ShoulderTurn);
                let band = norms.band(&Query::new(m.id, p, ctx.club, ctx.sex, ctx.reference));
                if m.values.at(p).is_some()
                    && band.is_some_and(|l| l.band.graded)
                    && (!turn || p == Position::Impact)
                {
                    expected += 1;
                }
            }
        }
        assert_eq!(grades.len(), expected, "{:?}", figure.view);
        assert!(grades.len() >= 8, "{}", grades.len());
        for g in &grades {
            let Measure::Metric(id) = g.metric else {
                panic!("tempo in a report grade")
            };
            let m = report.metric(id).unwrap();
            let want = match m.unit {
                Unit::Torso => m.cm.unwrap().at(g.position).unwrap() / 2.54,
                _ => m.values.at(g.position).unwrap(),
            };
            assert!((g.value - want).abs() < 1e-9, "{g:?}");
            assert!((g.z - (g.value - g.center) / g.sd).abs() < 1e-9);
            assert_eq!(
                g.scale,
                if m.unit == Unit::Torso {
                    Scale::Measured
                } else {
                    Scale::Native
                }
            );
            // P3 and P5 fall on the figure's ramps: graded, but the figure
            // sets only address, top, impact and finish.
            if matches!(
                g.position,
                Position::ArmParallelBack | Position::ArmParallelDown
            ) {
                continue;
            }
            assert_eq!(g.fit, Fit::Both, "{g:?}");
            assert!(g.miss.is_none() && g.fit_word().is_none());
        }
        // The same report with P3 and P5 set to their band centers grades
        // Both there too.
        let mut at_centers = report.clone();
        for m in &mut at_centers.metrics {
            for p in [Position::ArmParallelBack, Position::ArmParallelDown] {
                let Some(l) = norms.band(&Query::new(m.id, p, ctx.club, ctx.sex, ctx.reference))
                else {
                    continue;
                };
                let c = l.band.center;
                if slot(&mut m.values, p).is_none() {
                    continue;
                }
                if let Some(cm) = &mut m.cm {
                    *slot(cm, p) = Some(c * 2.54);
                } else {
                    *slot(&mut m.values, p) = Some(c.abs());
                }
            }
        }
        for g in norms.grade_report(&at_centers, &ctx) {
            assert_eq!(g.fit, Fit::Both, "{g:?}");
        }
        // Left-handed: the mirror image grades identically.
        let left = norms.grade_report(&tour_report(&figure, Handed::Left), &ctx);
        assert_eq!(left.len(), grades.len());
        for (l, r) in left.iter().zip(&grades) {
            assert_eq!(
                (l.metric, l.position, l.fit, l.side),
                (r.metric, r.position, r.fit, r.side)
            );
            assert!((l.value - r.value).abs() < 1e-3, "{l:?} vs {r:?}");
        }
    }
    // Tempo 3:1 with a 0.25 s downswing.
    let tempo = Tempo {
        backswing_ns: 750_000_000,
        downswing_ns: 250_000_000,
        ratio: 3.0,
    };
    assert!(
        norms
            .grade_tempo(&tempo, &ctx)
            .iter()
            .all(|g| g.fit == Fit::Both)
    );
}

#[test]
fn each_fault_figure_names_its_miss() {
    let norms = Norms::builtin().unwrap();
    let ctx = tour_driver();
    let check = |figure: Figure, id, position, fit: Fit, side, direction| {
        let grades = norms.grade_report(&tour_report(&figure, Handed::Right), &ctx);
        let g = find_grade(&grades, id, position)
            .unwrap_or_else(|| panic!("{id:?} at {position:?} graded"));
        assert_eq!(
            (g.fit, g.side),
            (fit, side),
            "{id:?} at {position:?}: {g:?}"
        );
        assert_eq!(g.direction, Some(direction), "{g:?}");
        if matches!(fit, Fit::Near | Fit::Far) {
            assert_eq!(g.miss.as_ref().unwrap().direction, direction);
            let word = g.fit_word().unwrap();
            assert!(
                word.starts_with(g.metric.name()) && word.contains(" than tour: "),
                "{word}"
            );
        }
        // Left-handed, the same fault reads the same.
        let left = norms.grade_report(&tour_report(&figure, Handed::Left), &ctx);
        let l = find_grade(&left, id, position).unwrap();
        assert_eq!((l.fit, l.side, l.direction), (g.fit, g.side, g.direction));
    };
    // Sway off the ball at the top.
    let mut f = tour_face_on();
    f.pelvis_dx.top = inches(-4.0);
    check(
        f,
        MetricId::HipSway,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Away,
    );
    // A slide at impact.
    let mut f = tour_face_on();
    f.pelvis_dx.impact = inches(9.0);
    check(
        f,
        MetricId::HipSway,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::Toward,
    );
    // A reverse pivot: hips toward the target and the spine leaning
    // toward it at the top.
    let mut f = tour_face_on();
    f.pelvis_dx.top = inches(3.0);
    check(
        f,
        MetricId::HipSway,
        Position::Top,
        Fit::Far,
        Side::High,
        Direction::Toward,
    );
    let mut f = tour_face_on();
    f.spine.top = 8.0;
    check(
        f,
        MetricId::LateralSpineTilt,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Less,
    );
    // Head up at impact, and a dip at the top.
    let mut f = tour_face_on();
    f.head_dy.impact = inches(-3.0);
    check(
        f,
        MetricId::HeadVertical,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::Up,
    );
    let mut f = tour_face_on();
    f.head_dy.top = inches(2.0);
    check(
        f,
        MetricId::HeadVertical,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Down,
    );
    // Hips 20° less open than tour at impact, and far more open.
    let mut f = tour_face_on();
    f.hip_turn.impact = 18.0;
    check(
        f,
        MetricId::HipTurn,
        Position::Impact,
        Fit::Far,
        Side::Low,
        Direction::Closed,
    );
    let mut f = tour_face_on();
    f.hip_turn.impact = 55.0;
    check(
        f,
        MetricId::HipTurn,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::Open,
    );
    // Hips a little more open than the neutral band: the fade side.
    let mut f = tour_face_on();
    f.hip_turn.impact = 44.0;
    check(
        f,
        MetricId::HipTurn,
        Position::Impact,
        Fit::Fade,
        Side::High,
        Direction::Open,
    );
    // Steep and flat shoulders at the top.
    let mut f = tour_face_on();
    f.shoulder_tilt.top = -50.0;
    check(
        f,
        MetricId::ShoulderTilt,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::More,
    );
    // Down the line: standing up at impact.
    let mut f = tour_down_the_line();
    f.spine.impact = 24.0;
    check(
        f,
        MetricId::SpineForwardBend,
        Position::Impact,
        Fit::Far,
        Side::Low,
        Direction::Less,
    );
    // Early extension, and falling back.
    let mut f = tour_down_the_line();
    f.pelvis_dx.impact = inches(4.0);
    check(
        f,
        MetricId::EarlyExtension,
        Position::Impact,
        Fit::Far,
        Side::High,
        Direction::TowardBall,
    );
    let mut f = tour_down_the_line();
    f.pelvis_dx.impact = inches(-3.0);
    check(
        f,
        MetricId::EarlyExtension,
        Position::Impact,
        Fit::Far,
        Side::Low,
        Direction::Back,
    );
    // Hands in front of the trail shoulder at the top, and deep.
    let mut f = tour_down_the_line();
    f.shoulder_half.top = top_hands_behind() + inches(1.0);
    check(
        f,
        MetricId::HandDepth,
        Position::Top,
        Fit::Far,
        Side::Low,
        Direction::Less,
    );
    let mut f = tour_down_the_line();
    f.shoulder_half.top = top_hands_behind() - inches(7.0);
    check(
        f,
        MetricId::HandDepth,
        Position::Top,
        Fit::Far,
        Side::High,
        Direction::More,
    );
}

#[test]
fn grades_serialize() {
    let norms = Norms::builtin().unwrap();
    let g = grade_deg(norms, MetricId::HipTurn, Position::Impact, 52.0).unwrap();
    let json = serde_json::to_string(&g).unwrap();
    assert!(
        json.contains(r#""metric":"hip_turn""#) && json.contains(r#""fit":"far""#),
        "{json}"
    );
    let back: Grade = serde_json::from_str(&json).unwrap();
    assert_eq!(back, g);
    assert!(g.proxy_sd());
    let t: Measure = serde_json::from_str(r#""tempo_ratio""#).unwrap();
    assert_eq!(t, Measure::Tempo(TempoMeasure::Ratio));
    assert_eq!(serde_json::to_string(&t).unwrap(), r#""tempo_ratio""#);
    let _ = SdSource::Published;
}

#[test]
fn the_norms_doc_lists_every_band() {
    let doc = include_str!("../../../docs/NORMS.md");
    let norms = Norms::builtin().unwrap();
    let position = |p: Position| match p {
        Position::Address => "address",
        Position::ArmParallelBack => "P3",
        Position::Top => "top",
        Position::ArmParallelDown => "P5",
        Position::Impact => "impact",
        Position::Finish => "finish",
        _ => "swing",
    };
    for b in norms.bands().iter().filter(|b| b.sex != Some(Sex::Female)) {
        let metric = serde_json::to_value(b.metric).unwrap();
        let club = match b.club {
            Some(ClubClass::Driver) => "driver",
            Some(ClubClass::MidIron) => "mid-iron",
            Some(_) => "wedge",
            None => "any",
        };
        let center = format!("{}", b.center).replace('-', "−");
        let row = format!(
            "| `{}` | {} | {club} | {center}",
            metric.as_str().unwrap(),
            position(b.position)
        );
        assert!(doc.contains(&row), "docs/NORMS.md lacks {row}");
    }
    for key in norms.sources.keys() {
        assert!(doc.contains(&format!("| {key} | ")), "source {key}");
    }
}
