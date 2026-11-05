//! Swing events, tempo and body metrics for a recorded clip.
//!
//! ```text
//! cargo run --release --example golf_report --features pose -- <models dir> <clip.avi> \
//!     [--view fo|dtl] [--handed r|l] [--impact-ns N | --estimate-impact] [--height-cm H] \
//!     [--mirrored] [--track <clip.pose.json>] [--render <out dir>] [--suffix TEXT] [--json]
//! ```
//!
//! Runs the offline pose pass on the clip (or reads `--track`), then
//! `stance::golf::analyze_swing` and prints the report. The impact comes
//! from the clip's sidecar unless `--impact-ns` (ns from the first frame)
//! or `--estimate-impact` says otherwise; the view is guessed unless
//! `--view` gives it. `--mirrored` says the clip is a mirrored image
//! (`--handed` is then the golfer's true hand). `--json` prints the report
//! as JSON instead.
//!
//! `--render <out dir>` writes `<clip><suffix>.swing.json` there and a JPEG
//! still at each event (`<clip><suffix>-<event>.jpg`) with the smoothed
//! skeleton (the person's left side cyan, right orange), the hands' path so
//! far in yellow and the event's numbers.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use stance::golf::{
    Handed, ImpactSource, MetricId, Positions, Smoothing, SwingEvent, SwingOptions, SwingReport,
    Unit, View, analyze_swing, smooth_track,
};
use stance::pose::{
    Image, PoseConfig, PoseFrame, PoseModels, PoseTrack, analyze_clip, decode_jpeg, draw,
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
    track: Option<PathBuf>,
    render: Option<PathBuf>,
    suffix: String,
    json: bool,
}

fn usage() -> String {
    "usage: golf_report <models dir> <clip.avi> [--view fo|dtl] [--handed r|l] \
     [--impact-ns N | --estimate-impact] [--height-cm H] [--mirrored] [--track FILE] [--render DIR] \
     [--suffix TEXT] [--json]"
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
        track: None,
        render: None,
        suffix: String::new(),
        json: false,
    };
    let mut it = std::env::args().skip(1);
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
            "--height-cm" => args.height_cm = Some(value()?.parse().map_err(|e| format!("{e}"))?),
            "--track" => args.track = Some(PathBuf::from(value()?)),
            "--render" => args.render = Some(PathBuf::from(value()?)),
            "--suffix" => args.suffix = value()?,
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
        ("top", e.top),
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
    say(&format!(
        "  {:20} {:>22} {:>22} {:>22} {:>22}  conf",
        "metric", "address", "top", "impact", "finish"
    ));
    for m in &r.metrics {
        let cell = |v: Option<f64>, cm: Option<f64>| {
            v.map_or_else(|| "-".to_owned(), |v| format_value(m.unit, v, cm))
        };
        let cm = m.cm.unwrap_or_default();
        say(&format!(
            "  {:20} {:>22} {:>22} {:>22} {:>22}  {:.2}",
            metric_name(m.id),
            cell(m.values.address, cm.address),
            cell(m.values.top, cm.top),
            cell(m.values.impact, cm.impact),
            cell(m.values.finish, cm.finish),
            m.confidence
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

fn pick(values: Positions, event: &str) -> Option<f64> {
    match event {
        "address" => values.address,
        "top" => values.top,
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
        ("top", e.top),
        ("impact", e.impact),
        ("finish", e.finish),
    ] {
        let Some(ev) = ev else { continue };
        let jpeg = avi.read_frame(ev.frame).map_err(|e| e.to_string())?;
        let mut img = decode_jpeg(&jpeg).map_err(|e| e.to_string())?;
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
        let config = PoseConfig::default();
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
    if args.json {
        say(&report.to_json().map_err(|e| e.to_string())?);
    } else {
        print_report(&stem, &report);
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
