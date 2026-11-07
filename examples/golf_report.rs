//! Swing events, tempo and body metrics for a recorded clip.
//!
//! ```text
//! cargo run --release --example golf_report --features pose -- <models dir> <clip.avi> \
//!     [--view fo|dtl] [--handed r|l] [--impact-ns N | --estimate-impact] [--height-cm H] \
//!     [--mirrored] [--rotate 0|90|180|270] [--track <clip.pose.json>] [--render <out dir>] \
//!     [--suffix TEXT] [--grade driver|iron|wedge [male|female] [tour|amateur]] [--json]
//! ```
//!
//! Runs the offline pose pass on the clip (or reads `--track`), then
//! `stance::golf::analyze_swing` and prints the report. The impact comes
//! from the clip's sidecar unless `--impact-ns` (ns from the first frame)
//! or `--estimate-impact` says otherwise; the view is guessed unless
//! `--view` gives it. `--mirrored` says the clip is a mirrored image
//! (`--handed` is then the golfer's true hand). `--rotate` turns every frame
//! clockwise by that many degrees before the pose pass, for a camera on its
//! side (a `--track` carries its own rotation). `--grade` grades every
//! metric and the tempo against the built-in bands (`stance::golf::norms`)
//! for the club class, sex (default male) and reference (default tour), and
//! prints each graded number with its band, z, fit and words. `--json`
//! prints the report as JSON instead (with `--grade`, an object with the
//! `report` and its `grades`).
//!
//! `--render <out dir>` writes `<clip><suffix>.swing.json` there and a JPEG
//! still at each event (`<clip><suffix>-<event>.jpg`, on the turned frame) with the smoothed
//! skeleton (the person's left side cyan, right orange), the hands' path so
//! far in yellow and the event's numbers.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use stance::golf::norms::NormUnit;
use stance::golf::{
    ClubClass, Context, Grade, Handed, ImpactSource, MetricId, Norms, Positions, Reference, Sex,
    Smoothing, SwingEvent, SwingOptions, SwingReport, Unit, View, analyze_swing, smooth_track,
};
use stance::pose::{
    Image, PoseConfig, PoseFrame, PoseModels, PoseTrack, Rotation, analyze_clip, decode_jpeg, draw,
};

/// The hands' path.
const PATH: draw::Rgb = [255, 230, 0];

/// A handedness guess at least this sure is worth a warning.
const HANDED_WARN: f32 = 0.5;

/// Prints a line; a closed pipe (`| head`) is not an error.
fn say(line: &str) {
    let _ = writeln!(std::io::stdout().lock(), "{line}");
}

struct Args {
    models: PathBuf,
    clip: PathBuf,
    view: Option<View>,
    handed: Handed,
    impact_ns: Option<u64>,
    estimate_impact: bool,
    height_cm: Option<f64>,
    mirrored: bool,
    rotation: Rotation,
    track: Option<PathBuf>,
    render: Option<PathBuf>,
    suffix: String,
    grade: Option<Context>,
    json: bool,
}

fn usage() -> String {
    "usage: golf_report <models dir> <clip.avi> [--view fo|dtl] [--handed r|l] \
     [--impact-ns N | --estimate-impact] [--height-cm H] [--mirrored] [--rotate 0|90|180|270] \
     [--track FILE] [--render DIR] [--suffix TEXT] \
     [--grade driver|iron|wedge [male|female] [tour|amateur]] [--json]"
        .into()
}

fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut args = Args {
        models: PathBuf::new(),
        clip: PathBuf::new(),
        view: None,
        handed: Handed::Right,
        impact_ns: None,
        estimate_impact: false,
        height_cm: None,
        mirrored: false,
        rotation: Rotation::None,
        track: None,
        render: None,
        suffix: String::new(),
        grade: None,
        json: false,
    };
    let mut it = std::env::args().skip(1).peekable();
    while let Some(a) = it.next() {
        let mut value = || it.next().ok_or_else(usage);
        match a.as_str() {
            "--view" => {
                args.view = Some(match value()?.as_str() {
                    "fo" | "face-on" | "face_on" => View::FaceOn,
                    "dtl" | "down-the-line" | "down_the_line" => View::DownTheLine,
                    other => return Err(format!("unknown view `{other}`")),
                });
            }
            "--handed" => {
                args.handed = match value()?.as_str() {
                    "r" | "right" => Handed::Right,
                    "l" | "left" => Handed::Left,
                    other => return Err(format!("unknown hand `{other}`")),
                };
            }
            "--impact-ns" => args.impact_ns = Some(value()?.parse().map_err(|e| format!("{e}"))?),
            "--estimate-impact" => args.estimate_impact = true,
            "--mirrored" => args.mirrored = true,
            "--rotate" => {
                let degrees = value()?;
                args.rotation = degrees
                    .parse()
                    .ok()
                    .and_then(Rotation::from_degrees)
                    .ok_or_else(|| format!("rotation `{degrees}`: 0, 90, 180 or 270"))?;
            }
            "--height-cm" => args.height_cm = Some(value()?.parse().map_err(|e| format!("{e}"))?),
            "--track" => args.track = Some(PathBuf::from(value()?)),
            "--render" => args.render = Some(PathBuf::from(value()?)),
            "--suffix" => args.suffix = value()?,
            "--grade" => {
                let club = match value()?.as_str() {
                    "driver" => ClubClass::Driver,
                    "iron" | "mid_iron" | "mid-iron" => ClubClass::MidIron,
                    "wedge" => ClubClass::Wedge,
                    other => return Err(format!("unknown club class `{other}`")),
                };
                let mut ctx = Context::new(club, Sex::Male, Reference::Tour);
                while let Some(word) = it.peek() {
                    match word.as_str() {
                        "male" => ctx.sex = Sex::Male,
                        "female" => ctx.sex = Sex::Female,
                        "tour" => ctx.reference = Reference::Tour,
                        "amateur" => ctx.reference = Reference::Amateur,
                        _ => break,
                    }
                    it.next();
                }
                args.grade = Some(ctx);
            }
            "--json" => args.json = true,
            "-h" | "--help" => return Err(usage()),
            _ => positional.push(PathBuf::from(a)),
        }
    }
    let [models, clip] = <[PathBuf; 2]>::try_from(positional).map_err(|_| usage())?;
    args.models = models;
    args.clip = clip;
    Ok(args)
}

/// A metric id as written in the report (`lead_arm_angle`).
fn metric_name(id: MetricId) -> String {
    serde_json::to_value(id)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| format!("{id:?}"))
}

fn view_name(view: Option<View>) -> &'static str {
    match view {
        Some(View::FaceOn) => "face-on",
        Some(View::DownTheLine) => "down-the-line",
        _ => "unknown view",
    }
}

fn hand_name(handed: Handed) -> &'static str {
    match handed {
        Handed::Right => "right",
        Handed::Left => "left",
    }
}

fn format_value(unit: Unit, v: f64, cm: Option<f64>) -> String {
    match (unit, cm) {
        (Unit::Degrees, _) => format!("{v:+.1}°"),
        (Unit::Torso, Some(cm)) => format!("{v:+.2} torso ({cm:+.1} cm)"),
        (Unit::Torso, None) => format!("{v:+.2} torso"),
        (Unit::Ratio, _) => format!("{v:.2}"),
    }
}

fn event_line(name: &str, e: Option<SwingEvent>) -> String {
    e.map_or_else(
        || format!("{name:9} -"),
        |e| {
            #[allow(clippy::cast_precision_loss, reason = "ns times fit f64")]
            let s = e.t_ns as f64 * 1e-9;
            format!(
                "{name:9} {s:7.3} s  frame {:4}  confidence {:.2}",
                e.frame, e.confidence
            )
        },
    )
}

fn print_report(stem: &str, r: &SwingReport) {
    say(&format!(
        "{stem}: {} {}-handed{}",
        view_name(r.view),
        hand_name(r.handed),
        if r.mirrored { " (mirrored image)" } else { "" }
    ));
    match r.view_guess {
        Some(g) => say(&format!(
            "  view guess {} ({:.2})",
            view_name(Some(g.value)),
            g.confidence
        )),
        None => say("  view guess unknown"),
    }
    if let Some(g) = r.handed_guess
        && g.value != r.handed
    {
        if g.confidence >= HANDED_WARN {
            say(&format!(
                "  the swing looks {}-handed ({:.2}): check the hand or a mirrored image",
                hand_name(g.value),
                g.confidence
            ));
        } else {
            say(&format!(
                "  weak hint of a {}-handed swing ({:.2})",
                hand_name(g.value),
                g.confidence
            ));
        }
    }
    let e = &r.events;
    for (name, ev) in [
        ("address", e.address),
        ("takeaway", e.takeaway),
        ("P3", e.arm_parallel_back),
        ("top", e.top),
        ("P5", e.arm_parallel_down),
        ("impact", e.impact),
        ("finish", e.finish),
    ] {
        say(&format!("  {}", event_line(name, ev)));
    }
    say(&format!("  impact source {:?}", e.impact_source));
    if e.impact_source == ImpactSource::Given {
        say("  no swing found near the given impact time: check it");
    }
    if let Some(t) = r.tempo {
        #[allow(clippy::cast_precision_loss, reason = "ns durations fit f64")]
        let (b, d) = (t.backswing_ns as f64 / 1e6, t.downswing_ns as f64 / 1e6);
        say(&format!(
            "  tempo {:.2}:1 (backswing {b:.0} ms, downswing {d:.0} ms)",
            t.ratio
        ));
    }
    if let Some(s) = r.scale {
        say(&format!("  torso {:.1} px at address", s.torso_px));
    }
    if !r.series.is_empty() {
        say(&format!(
            "  series: {} metrics over {} frames",
            r.series.metrics.len(),
            r.series.t_ns.len()
        ));
    }
    let mut header = format!("  {:20}", "metric");
    for name in POSITIONS {
        let _ = write!(header, " {name:>22}");
    }
    say(&format!("{header}  conf"));
    for m in &r.metrics {
        let cm = m.cm.unwrap_or_default();
        let mut line = format!("  {:20}", metric_name(m.id));
        for name in POSITIONS {
            let cell = pick(m.values, name).map_or_else(
                || "-".to_owned(),
                |v| format_value(m.unit, v, pick(cm, name)),
            );
            let _ = write!(line, " {cell:>22}");
        }
        say(&format!("{line}  {:.2}", m.confidence));
    }
}

fn position_name(p: stance::golf::Position) -> &'static str {
    use stance::golf::Position;
    match p {
        Position::Address => "address",
        Position::ArmParallelBack => "P3",
        Position::Top => "top",
        Position::ArmParallelDown => "P5",
        Position::Impact => "impact",
        Position::Finish => "finish",
        _ => "swing",
    }
}

fn unit_suffix(unit: NormUnit) -> &'static str {
    match unit {
        NormUnit::Degrees => "°",
        NormUnit::Inches => " in",
        NormUnit::Seconds => " s",
        _ => "",
    }
}

fn print_grades(ctx: &Context, grades: &[Grade]) {
    say(&format!(
        "  grades against {:?} {:?} {:?} bands:",
        ctx.reference, ctx.club, ctx.sex
    ));
    say(&format!(
        "  {:20} {:8} {:>9} {:>20} {:>7} {:>13}  {:5}  words",
        "metric", "at", "value", "band", "z", "z fade/draw", "fit"
    ));
    for g in grades {
        let u = unit_suffix(g.unit);
        let mut band = format!("{:.1} ± {:.1}{u}", g.centre, g.sd);
        if let (Some(f), Some(d)) = (g.fade, g.draw) {
            band = format!("{:.1}/{:.1} ± {:.1}{u}", f.centre, d.centre, g.sd);
        }
        let shape = match (g.z_fade, g.z_draw) {
            (Some(f), Some(d)) => format!("{f:+.2}/{d:+.2}"),
            _ => "-".to_owned(),
        };
        let mut flags = String::new();
        if g.club_fallback {
            flags.push_str(" (mid-iron band)");
        }
        if g.sex_fallback {
            flags.push_str(" (male band)");
        }
        say(&format!(
            "  {:20} {:8} {:>9} {band:>20} {:>+7.2} {shape:>13}  {:5}  {}{flags}",
            g.metric.name(),
            position_name(g.position),
            format!("{:+.1}{u}", g.value),
            g.z,
            format!("{:?}", g.fit),
            g.fit_word().unwrap_or_default(),
        ));
    }
}

fn encode(img: &Image) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let w = u16::try_from(img.width()).map_err(|e| e.to_string())?;
    let h = u16::try_from(img.height()).map_err(|e| e.to_string())?;
    jpeg_encoder::Encoder::new(&mut out, 88)
        .encode(img.data(), w, h, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// The hands (the wrists' midpoint) per frame, where both wrists scored.
fn hands(frame: &PoseFrame, handed: Handed) -> Option<(f32, f32)> {
    use stance::golf::Joint;
    let (a, b) = (
        frame.keypoints.get(Joint::LeadWrist.index(handed))?,
        frame.keypoints.get(Joint::TrailWrist.index(handed))?,
    );
    (a.score > 0.0 && b.score > 0.0).then(|| (f32::midpoint(a.x, b.x), f32::midpoint(a.y, b.y)))
}

/// The key positions, in swing order.
const POSITIONS: [&str; 6] = ["address", "P3", "top", "P5", "impact", "finish"];

fn pick(values: Positions, event: &str) -> Option<f64> {
    match event {
        "address" => values.address,
        "P3" => values.arm_parallel_back,
        "top" => values.top,
        "P5" => values.arm_parallel_down,
        "impact" => values.impact,
        "finish" => values.finish,
        _ => None,
    }
}

fn render(
    args: &Args,
    stem: &str,
    track: &PoseTrack,
    report: &SwingReport,
    out: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let name = format!("{stem}{}", args.suffix);
    let path = report
        .write_beside(&out.join(format!("{name}.avi")))
        .map_err(|e| e.to_string())?;
    say(&path.display().to_string());
    let smoothed = smooth_track(
        track,
        &report.smoothing,
        report.events.impact.map(|e| e.t_ns),
    )
    .map_err(|e| e.to_string())?;
    let dir = args.clip.parent().unwrap_or(Path::new("."));
    let (mut avi, _) = stance::open_clip(dir, stem).map_err(|e| e.to_string())?;
    let e = &report.events;
    let mut header = format!("{} {}", view_name(report.view), hand_name(report.handed));
    if let Some(t) = report.tempo {
        let _ = write!(header, "  tempo {:.1}:1", t.ratio);
    }
    for (event, ev) in [
        ("address", e.address),
        ("P3", e.arm_parallel_back),
        ("top", e.top),
        ("P5", e.arm_parallel_down),
        ("impact", e.impact),
        ("finish", e.finish),
    ] {
        let Some(ev) = ev else { continue };
        let jpeg = avi.read_frame(ev.frame).map_err(|e| e.to_string())?;
        let mut img = decode_jpeg(&jpeg)
            .map_err(|e| e.to_string())?
            .rotated(track.rotation);
        let style = draw::SkeletonStyle {
            fade: false,
            ..draw::SkeletonStyle::for_image(&img)
        };
        let path: Vec<(f32, f32)> = smoothed.frames[..=ev.frame]
            .iter()
            .filter_map(|f| hands(f, report.handed))
            .collect();
        for w in path.windows(2) {
            draw::line(&mut img, w[0], w[1], style.scale, PATH, 0.8);
        }
        draw::skeleton(
            &mut img,
            smoothed.skeleton,
            &smoothed.frames[ev.frame].keypoints,
            style,
        );
        #[allow(clippy::cast_precision_loss, reason = "ns times fit f64")]
        let t = ev.t_ns as f64 * 1e-9;
        let mut lines = vec![
            format!("{event} {t:.3} s  conf {:.2}", ev.confidence),
            header.clone(),
        ];
        for m in &report.metrics {
            if let Some(v) = pick(m.values, event) {
                let cm = m.cm.and_then(|c| pick(c, event));
                lines.push(format!(
                    "{} {}",
                    metric_name(m.id).replace('_', " "),
                    format_value(m.unit, v, cm)
                ));
            }
        }
        let text_scale = (img.height() / 360).max(1);
        draw::text_panel(&mut img, &lines, text_scale);
        let file = out.join(format!("{name}-{event}.jpg"));
        std::fs::write(&file, encode(&img)?).map_err(|e| format!("{}: {e}", file.display()))?;
        say(&file.display().to_string());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let dir = args.clip.parent().unwrap_or(Path::new("."));
    let stem = args
        .clip
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("clip name")?
        .to_owned();
    let sidecar =
        stance::read_sidecar(&dir.join(format!("{stem}.json"))).map_err(|e| e.to_string())?;
    let track = if let Some(path) = &args.track {
        PoseTrack::read(path).map_err(|e| e.to_string())?
    } else {
        let config = PoseConfig {
            rotation: args.rotation,
            ..PoseConfig::default()
        };
        let models = PoseModels::load(&args.models, &config).map_err(|e| e.to_string())?;
        analyze_clip(&models, &args.clip, &config).map_err(|e| e.to_string())?
    };
    let impact_ns = if args.estimate_impact {
        None
    } else {
        args.impact_ns
            .or_else(|| u64::try_from(sidecar.impact_ns).ok())
    };
    let report = analyze_swing(
        &track,
        &SwingOptions {
            view: args.view,
            handed: args.handed,
            impact_ns,
            height_cm: args.height_cm,
            mirrored: args.mirrored,
            smoothing: Smoothing::default(),
        },
    )
    .map_err(|e| e.to_string())?;
    let grades = match &args.grade {
        Some(ctx) => {
            let norms = Norms::builtin().map_err(|e| e.to_string())?;
            let mut grades = norms.grade_report(&report, ctx);
            if let Some(t) = &report.tempo {
                grades.extend(norms.grade_tempo(t, ctx));
            }
            Some((ctx, grades))
        }
        None => None,
    };
    match (&grades, args.json) {
        (None, true) => say(&report.to_json().map_err(|e| e.to_string())?),
        (Some((_, grades)), true) => {
            let value = serde_json::json!({ "report": report, "grades": grades });
            say(&serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?);
        }
        (_, false) => {
            print_report(&stem, &report);
            if let Some((ctx, grades)) = &grades {
                print_grades(ctx, grades);
            }
        }
    }
    if let Some(out) = &args.render {
        render(&args, &stem, &track, &report, out)?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
