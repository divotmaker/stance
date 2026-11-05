//! Benchmarks pose estimation on recorded clips.
//!
//! ```text
//! cargo run --release --example pose_bench --features pose -- <models dir> <clips dir | clip.avi>... \
//!     [--pose rtmpose-m-halpe26] [--detector yolox-tiny] [--threads N] [--inference-threads N] \
//!     [--samples 40] [--detect-every 10] [--min-pose-score 0.3] [--json out.json]
//! ```
//!
//! For each clip (an AVI with its stance sidecar) and overall it reports:
//! detector and pose latency per frame (p50/p95) with one inference on one
//! core and on `--inference-threads` cores (default: the live tracker's,
//! half the cores and at most 4), the offline clip pass's wall time
//! (`--threads` workers), mean keypoint scores
//! per body part, the mean wrist and elbow score within ±50 ms of the
//! sidecar's impact, and a left/right swap count. When a `manifest.json`
//! with per-clip `events` (times in ns from the clip start) sits beside the
//! clips, the mean keypoint score at each event is reported too. Results are
//! grouped by the manifest's `view` and `quality` when present.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde_json::{Value, json};
use stance::pose::{
    BBox, Image, InferencePool, ModelSpec, PersonDetector, PoseConfig, PoseEngine, PoseModels,
    PoseTrack, RTMPOSE_M_HALPE26, Skeleton, YOLOX_TINY, analyze_frames, decode_jpeg,
};

/// Body-part groups of the Halpe-26 keypoints.
const GROUPS: [(&str, &[&str]); 6] = [
    (
        "head",
        &[
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "head",
        ],
    ),
    (
        "torso",
        &[
            "neck",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "hip",
        ],
    ),
    ("arms", &["left_elbow", "right_elbow"]),
    ("wrists", &["left_wrist", "right_wrist"]),
    (
        "legs",
        &["left_knee", "right_knee", "left_ankle", "right_ankle"],
    ),
    (
        "feet",
        &[
            "left_big_toe",
            "right_big_toe",
            "left_small_toe",
            "right_small_toe",
            "left_heel",
            "right_heel",
        ],
    ),
];

/// Half-width of the impact window (ns).
const IMPACT_WINDOW_NS: i64 = 50_000_000;

struct Args {
    models: PathBuf,
    inputs: Vec<PathBuf>,
    json: Option<PathBuf>,
    config: PoseConfig,
    samples: usize,
}

fn usage() -> String {
    "usage: pose_bench <models dir> <clips dir | clip.avi>... [--pose NAME] [--detector NAME] \
     [--threads N] [--inference-threads N] [--samples N] [--detect-every N] \
     [--min-pose-score S] [--json FILE]"
        .into()
}

fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut json = None;
    let mut config = PoseConfig {
        pose_model: RTMPOSE_M_HALPE26,
        detector: YOLOX_TINY,
        ..PoseConfig::default()
    };
    let mut samples = 40;
    let mut it = std::env::args().skip(1);
    let spec = |name: Option<String>| -> Result<ModelSpec, String> {
        let name = name.ok_or_else(usage)?;
        ModelSpec::find(&name)
            .copied()
            .ok_or_else(|| format!("unknown model `{name}`"))
    };
    let number = |v: Option<String>| -> Result<usize, String> {
        v.ok_or_else(usage)?.parse().map_err(|e| format!("{e}"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--json" => json = Some(PathBuf::from(it.next().ok_or_else(usage)?)),
            "--pose" => config.pose_model = spec(it.next())?,
            "--detector" => config.detector = spec(it.next())?,
            "--threads" => config.threads = number(it.next())?,
            "--inference-threads" => config.inference_threads = number(it.next())?,
            "--samples" => samples = number(it.next())?.max(1),
            "--min-pose-score" => {
                config.min_pose_score = it
                    .next()
                    .ok_or_else(usage)?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--detect-every" => {
                config.detect_every =
                    u32::try_from(number(it.next())?).map_err(|e| e.to_string())?;
            }
            "-h" | "--help" => return Err(usage()),
            _ => positional.push(PathBuf::from(a)),
        }
    }
    if positional.len() < 2 {
        return Err(usage());
    }
    let models = positional.remove(0);
    Ok(Args {
        models,
        inputs: positional,
        json,
        config,
        samples,
    })
}

/// `(avi path, manifest entry)` for every clip named on the command line.
fn find_clips(inputs: &[PathBuf]) -> Result<Vec<(PathBuf, Value)>, String> {
    let mut clips = Vec::new();
    for input in inputs {
        let (dir, files): (PathBuf, Vec<PathBuf>) = if input.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(input)
                .map_err(|e| format!("{}: {e}", input.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "avi"))
                .filter(|p| p.with_extension("json").exists())
                .collect();
            files.sort();
            (input.clone(), files)
        } else {
            let dir = input.parent().unwrap_or(Path::new(".")).to_path_buf();
            (dir, vec![input.clone()])
        };
        let manifest = read_manifest(&dir);
        for f in files {
            let name = f.file_name().map(|n| n.to_string_lossy().into_owned());
            let entry = manifest
                .iter()
                .find(|e| e.get("file").and_then(Value::as_str) == name.as_deref())
                .cloned()
                .unwrap_or(Value::Null);
            clips.push((f, entry));
        }
    }
    Ok(clips)
}

fn read_manifest(dir: &Path) -> Vec<Value> {
    let Ok(text) = std::fs::read_to_string(dir.join("manifest.json")) else {
        return Vec::new();
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Array(a)) => a,
        Ok(Value::Object(o)) => o
            .get("clips")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let i = ((s.len() - 1) as f64 * p).round() as usize;
    s[i]
}

fn mean(v: impl IntoIterator<Item = f64>) -> f64 {
    let (sum, n) = v
        .into_iter()
        .fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    #[allow(clippy::cast_precision_loss)]
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

/// Per-call latencies (ms) of the detector and the pose model.
#[derive(Default)]
struct Timings {
    detect: Vec<f64>,
    pose: Vec<f64>,
}

/// Times detection and pose on each image, one call after another.
fn time_calls(m: &mut PoseModels, work: &[(&Image, BBox)]) -> Result<Timings, stance::Error> {
    let mut t = Timings::default();
    for (image, bbox) in work {
        let t0 = Instant::now();
        let _ = m.detector.detect(image)?;
        t.detect.push(t0.elapsed().as_secs_f64() * 1e3);
        let t0 = Instant::now();
        let _ = m.pose.estimate(image, bbox)?;
        t.pose.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(t)
}

/// Times the same calls with each inference on `pool`.
fn time_threaded(
    models: &PoseModels,
    work: &[(&Image, BBox)],
    pool: &InferencePool,
) -> Result<Timings, stance::Error> {
    let mut m = models.fork()?;
    m.pose.set_inference_pool(pool)?;
    m.detector.set_inference_pool(pool)?;
    time_calls(&mut m, work)
}

/// Left/right swaps: a keypoint lands within `tol` of where its partner was
/// on the previous frame while the partner lands within `tol` of the
/// keypoint's old spot (and the two were apart).
fn count_swaps(track: &PoseTrack) -> usize {
    let pairs = track.skeleton.pairs();
    let mut swaps = 0;
    for w in track.frames.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if a.keypoints.is_empty() || b.keypoints.is_empty() {
            continue;
        }
        let height = b.bbox.map_or(100.0, |bb| bb.height());
        let tol = (0.02 * height).max(3.0);
        let d = |p: &stance::pose::Keypoint, q: &stance::pose::Keypoint| {
            ((p.x - q.x).powi(2) + (p.y - q.y).powi(2)).sqrt()
        };
        for &(l, r) in pairs {
            let (l0, r0, l1, r1) = (
                &a.keypoints[l],
                &a.keypoints[r],
                &b.keypoints[l],
                &b.keypoints[r],
            );
            if d(l0, r0) > 2.0 * tol && d(l1, r0) < tol && d(r1, l0) < tol {
                swaps += 1;
            }
        }
    }
    swaps
}

fn group_means(track: &PoseTrack, frames: &[usize], names: &[&str]) -> f64 {
    let idx: Vec<usize> = names
        .iter()
        .filter_map(|n| track.skeleton.index(n))
        .collect();
    mean(frames.iter().flat_map(|&i| {
        let f = &track.frames[i];
        idx.iter()
            .filter(|_| !f.keypoints.is_empty())
            .map(move |&k| f64::from(f.keypoints[k].score))
    }))
}

struct ClipResult {
    json: Value,
    view: String,
    quality: String,
    timings_1: Timings,
    timings_n: Timings,
    frames: usize,
    pass_ms: f64,
}

// One pass over the clip, top to bottom: splitting it would only scatter the
// steps.
#[allow(clippy::too_many_lines)]
fn bench_clip(
    models: &PoseModels,
    pool: &InferencePool,
    avi: &Path,
    entry: &Value,
    args: &Args,
) -> Result<ClipResult, String> {
    let dir = avi.parent().unwrap_or(Path::new("."));
    let name = avi
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("clip name")?;
    let (mut reader, sidecar) = stance::open_clip(dir, name).map_err(|e| e.to_string())?;
    let frames = sidecar
        .frames
        .iter()
        .enumerate()
        .map(|(i, f)| Ok((f.t_ns, reader.read_frame(i)?)))
        .collect::<Result<Vec<_>, stance::Error>>()
        .map_err(|e| e.to_string())?;

    let t0 = Instant::now();
    let track = analyze_frames(models, &frames, &args.config).map_err(|e| e.to_string())?;
    let pass_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Latency on evenly spaced frames, posed in the clip pass's boxes.
    let n = frames.len();
    let step = (n / args.samples).max(1);
    let images: Vec<(Image, BBox)> = (0..n)
        .step_by(step)
        .take(args.samples)
        .filter_map(|i| {
            let image = decode_jpeg(&frames[i].1).ok()?;
            // Frame sizes are far below f32's exact-integer range.
            #[allow(clippy::cast_precision_loss)]
            let whole = BBox::new(0.0, 0.0, image.width() as f32, image.height() as f32);
            let bbox = track.frames[i].bbox.unwrap_or(whole);
            Some((image, bbox))
        })
        .collect();
    let work: Vec<(&Image, BBox)> = images.iter().map(|(i, b)| (i, *b)).collect();
    let mut one = models.fork().map_err(|e| e.to_string())?;
    let timings_1 = time_calls(&mut one, &work).map_err(|e| e.to_string())?;
    let threads = pool.threads();
    let timings_n = time_threaded(models, &work, pool).map_err(|e| e.to_string())?;

    let all: Vec<usize> = (0..n).collect();
    let groups: BTreeMap<&str, f64> = GROUPS
        .iter()
        .map(|(g, names)| (*g, group_means(&track, &all, names)))
        .collect();
    let impact_frames: Vec<usize> = (0..n)
        .filter(|&i| {
            let t = i64::try_from(track.frames[i].t_ns).unwrap_or(i64::MAX);
            (t - sidecar.impact_ns).abs() <= IMPACT_WINDOW_NS
        })
        .collect();
    let impact_arms = group_means(
        &track,
        &impact_frames,
        &["left_wrist", "right_wrist", "left_elbow", "right_elbow"],
    );
    let swaps = count_swaps(&track);
    let detected = track
        .frames
        .iter()
        .filter(|f| !f.keypoints.is_empty())
        .count();

    let mut events = BTreeMap::new();
    if let Some(ev) = entry.get("events").and_then(Value::as_object) {
        for (event, t) in ev {
            let Some(t) = t.as_i64() else { continue };
            let nearest = (0..n).min_by_key(|&i| {
                (i64::try_from(track.frames[i].t_ns).unwrap_or(i64::MAX) - t).abs()
            });
            if let Some(i) = nearest {
                let all_names: Vec<&str> = Skeleton::Halpe26.names().to_vec();
                events.insert(
                    event.clone(),
                    json!({
                        "frame": i,
                        "mean": group_means(&track, &[i], &all_names),
                        "wrists": group_means(&track, &[i], &["left_wrist", "right_wrist"]),
                    }),
                );
            }
        }
    }

    let s = |key: &str| {
        entry
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("-")
            .to_owned()
    };
    let (view, quality) = (s("view"), s("quality"));
    #[allow(clippy::cast_precision_loss)]
    let json = json!({
        "clip": name,
        "view": view,
        "quality": quality,
        "width": sidecar.width,
        "height": sidecar.height,
        "fps": sidecar.fps.measured,
        "frames": n,
        "frames_with_pose": detected,
        "detect_ms_1": { "p50": percentile(&timings_1.detect, 0.5), "p95": percentile(&timings_1.detect, 0.95) },
        "pose_ms_1": { "p50": percentile(&timings_1.pose, 0.5), "p95": percentile(&timings_1.pose, 0.95) },
        "detect_ms_n": { "p50": percentile(&timings_n.detect, 0.5), "p95": percentile(&timings_n.detect, 0.95) },
        "pose_ms_n": { "p50": percentile(&timings_n.pose, 0.5), "p95": percentile(&timings_n.pose, 0.95) },
        "workers": args.config.threads,
        "inference_threads": threads,
        "pass_ms": pass_ms,
        "pass_ms_per_frame": pass_ms / n.max(1) as f64,
        "scores": groups,
        "impact_wrist_elbow": impact_arms,
        "swaps": swaps,
        "events": events,
    });
    Ok(ClipResult {
        json,
        view,
        quality,
        timings_1,
        timings_n,
        frames: n,
        pass_ms,
    })
}

fn f2(v: &Value) -> String {
    v.as_f64()
        .filter(|x| x.is_finite())
        .map_or_else(|| "-".into(), |x| format!("{x:.2}"))
}

fn f1(v: &Value) -> String {
    v.as_f64()
        .filter(|x| x.is_finite())
        .map_or_else(|| "-".into(), |x| format!("{x:.1}"))
}

fn print_table(results: &[ClipResult]) {
    println!(
        "{:<34} {:<13} {:<11} {:>5} {:>5} {:>13} {:>13} {:>13} {:>13} {:>8} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} {:>5}",
        "clip",
        "view",
        "quality",
        "fps",
        "frm",
        "det1 p50/95",
        "pose1 p50/95",
        "detN p50/95",
        "poseN p50/95",
        "pass ms/f",
        "head",
        "torso",
        "arms",
        "wrist",
        "legs",
        "feet",
        "imp±50",
        "swaps"
    );
    for r in results {
        let j = &r.json;
        let pair = |k: &str| format!("{}/{}", f1(&j[k]["p50"]), f1(&j[k]["p95"]));
        println!(
            "{:<34} {:<13} {:<11} {:>5} {:>5} {:>13} {:>13} {:>13} {:>13} {:>8} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} {:>5}",
            j["clip"].as_str().unwrap_or("-"),
            r.view,
            r.quality,
            f1(&j["fps"]),
            j["frames"],
            pair("detect_ms_1"),
            pair("pose_ms_1"),
            pair("detect_ms_n"),
            pair("pose_ms_n"),
            f1(&j["pass_ms_per_frame"]),
            f2(&j["scores"]["head"]),
            f2(&j["scores"]["torso"]),
            f2(&j["scores"]["arms"]),
            f2(&j["scores"]["wrists"]),
            f2(&j["scores"]["legs"]),
            f2(&j["scores"]["feet"]),
            f2(&j["impact_wrist_elbow"]),
            j["swaps"],
        );
    }
}

/// Mean of a numeric field over clips.
fn field_mean(rs: &[&ClipResult], path: &[&str]) -> f64 {
    mean(rs.iter().filter_map(|r| {
        let mut v = &r.json;
        for p in path {
            v = &v[*p];
        }
        v.as_f64().filter(|x| x.is_finite())
    }))
}

fn summary(rs: &[&ClipResult]) -> Value {
    let mut t1 = Timings::default();
    let mut tn = Timings::default();
    for r in rs {
        t1.detect.extend(&r.timings_1.detect);
        t1.pose.extend(&r.timings_1.pose);
        tn.detect.extend(&r.timings_n.detect);
        tn.pose.extend(&r.timings_n.pose);
    }
    let frames: usize = rs.iter().map(|r| r.frames).sum();
    let pass: f64 = rs.iter().map(|r| r.pass_ms).sum();
    let mut events: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in rs {
        if let Some(ev) = r.json["events"].as_object() {
            for (k, v) in ev {
                if let Some(m) = v["mean"].as_f64().filter(|x| x.is_finite()) {
                    events.entry(k.clone()).or_default().push(m);
                }
            }
        }
    }
    let scores: BTreeMap<&str, f64> = GROUPS
        .iter()
        .map(|(g, _)| (*g, field_mean(rs, &["scores", g])))
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let json = json!({
        "clips": rs.len(),
        "frames": frames,
        "detect_ms_1": { "p50": percentile(&t1.detect, 0.5), "p95": percentile(&t1.detect, 0.95) },
        "pose_ms_1": { "p50": percentile(&t1.pose, 0.5), "p95": percentile(&t1.pose, 0.95) },
        "detect_ms_n": { "p50": percentile(&tn.detect, 0.5), "p95": percentile(&tn.detect, 0.95) },
        "pose_ms_n": { "p50": percentile(&tn.pose, 0.5), "p95": percentile(&tn.pose, 0.95) },
        "pass_ms_total": pass,
        "pass_ms_per_frame": pass / frames.max(1) as f64,
        "pass_ms_per_clip": pass / rs.len().max(1) as f64,
        "scores": scores,
        "impact_wrist_elbow": field_mean(rs, &["impact_wrist_elbow"]),
        "swaps": rs.iter().map(|r| r.json["swaps"].as_u64().unwrap_or(0)).sum::<u64>(),
        "event_means": events.into_iter().map(|(k, v)| (k, mean(v))).collect::<BTreeMap<_, _>>(),
    });
    json
}

fn print_summary(label: &str, s: &Value) {
    let pair = |k: &str| format!("{}/{}", f1(&s[k]["p50"]), f1(&s[k]["p95"]));
    println!(
        "{label:<26} clips {:>3} frames {:>6} | det1 {:>11} pose1 {:>11} detN {:>11} poseN {:>11} | pass {:>6} ms/clip {:>5} ms/frame | head {} torso {} arms {} wrists {} legs {} feet {} | impact±50 {} | swaps {}",
        s["clips"],
        s["frames"],
        pair("detect_ms_1"),
        pair("pose_ms_1"),
        pair("detect_ms_n"),
        pair("pose_ms_n"),
        f1(&s["pass_ms_per_clip"]),
        f1(&s["pass_ms_per_frame"]),
        f2(&s["scores"]["head"]),
        f2(&s["scores"]["torso"]),
        f2(&s["scores"]["arms"]),
        f2(&s["scores"]["wrists"]),
        f2(&s["scores"]["legs"]),
        f2(&s["scores"]["feet"]),
        f2(&s["impact_wrist_elbow"]),
        s["swaps"],
    );
    if let Some(ev) = s["event_means"].as_object().filter(|e| !e.is_empty()) {
        let parts: Vec<String> = ev.iter().map(|(k, v)| format!("{k} {}", f2(v))).collect();
        println!("{:<26} event scores: {}", "", parts.join(", "));
    }
}

fn run() -> Result<(), String> {
    let mut args = parse_args()?;
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    if args.config.threads == 0 {
        args.config.threads = cores.min(8);
    }
    if args.config.inference_threads == 0 {
        args.config.inference_threads = (cores / 2).clamp(1, 4);
    }
    let pool = InferencePool::new(args.config.inference_threads).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let models = PoseModels::load(&args.models, &args.config).map_err(|e| e.to_string())?;
    eprintln!(
        "loaded {} + {} in {:.0} ms; {} workers, {} inference threads",
        args.config.pose_model.name,
        args.config.detector.name,
        t0.elapsed().as_secs_f64() * 1e3,
        args.config.threads,
        args.config.inference_threads
    );
    let clips = find_clips(&args.inputs)?;
    if clips.is_empty() {
        return Err("no clips (an .avi with its .json sidecar) found".into());
    }
    let mut results = Vec::new();
    for (avi, entry) in &clips {
        eprintln!("{} ...", avi.display());
        match bench_clip(&models, &pool, avi, entry, &args) {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("{}: {e}", avi.display()),
        }
    }
    println!(
        "model {} + {}, detect every {} frames, {} workers, {} inference threads",
        args.config.pose_model.name,
        args.config.detector.name,
        args.config.detect_every,
        args.config.threads,
        args.config.inference_threads
    );
    print_table(&results);
    println!();
    let mut groups: BTreeMap<(String, String), Vec<&ClipResult>> = BTreeMap::new();
    for r in &results {
        groups
            .entry((r.view.clone(), r.quality.clone()))
            .or_default()
            .push(r);
    }
    let mut group_json = Vec::new();
    for ((view, quality), rs) in &groups {
        let s = summary(rs);
        print_summary(&format!("{view} / {quality}"), &s);
        group_json.push(json!({ "view": view, "quality": quality, "summary": s }));
    }
    let all: Vec<&ClipResult> = results.iter().collect();
    let overall = summary(&all);
    print_summary("overall", &overall);
    if let Some(path) = &args.json {
        let out = json!({
            "pose_model": args.config.pose_model.name,
            "detector": args.config.detector.name,
            "detect_every": args.config.detect_every,
            "min_pose_score": args.config.min_pose_score,
            "threads": args.config.threads,
            "inference_threads": args.config.inference_threads,
            "clips": results.iter().map(|r| r.json.clone()).collect::<Vec<_>>(),
            "groups": group_json,
            "overall": overall,
        });
        let text = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        eprintln!("wrote {}", path.display());
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
