//! Draws the pose track of a clip onto its frames.
//!
//! ```text
//! cargo run --release --example pose_render --features pose -- <models dir> <clip.avi> <out dir> \
//!     [--pose rtmpose-m-halpe26] [--detector yolox-tiny] [--threads N] [--suffix -pose]
//! ```
//!
//! Runs the offline clip pass, then writes `<out>/<clip><suffix>.avi` (MJPEG,
//! through stance's own clip writer, with its sidecar), the track as
//! `<clip><suffix>.pose.json`, and JPEG stills at the impact frame and at
//! every event a `manifest.json` beside the clip lists for it. The skeleton
//! is drawn with `stance::pose::draw`: the person's left side in cyan, the
//! right in orange, the midline in white; low-score keypoints fade out.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use serde_json::Value;
use stance::pose::{
    Image, ModelSpec, PoseConfig, PoseModels, RTMPOSE_M_HALPE26, YOLOX_TINY, analyze_clip,
    decode_jpeg, draw,
};
use stance::{Clip, ClipId, Frame};

struct Args {
    models: PathBuf,
    clip: PathBuf,
    out: PathBuf,
    suffix: String,
    config: PoseConfig,
}

fn usage() -> String {
    "usage: pose_render <models dir> <clip.avi> <out dir> [--pose NAME] [--detector NAME] \
     [--threads N] [--suffix TEXT]"
        .into()
}

fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut suffix = "-pose".to_owned();
    let mut config = PoseConfig {
        pose_model: RTMPOSE_M_HALPE26,
        detector: YOLOX_TINY,
        ..PoseConfig::default()
    };
    let spec = |name: Option<String>| -> Result<ModelSpec, String> {
        let name = name.ok_or_else(usage)?;
        ModelSpec::find(&name)
            .copied()
            .ok_or_else(|| format!("unknown model `{name}`"))
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--pose" => config.pose_model = spec(it.next())?,
            "--detector" => config.detector = spec(it.next())?,
            "--threads" => {
                config.threads = it
                    .next()
                    .ok_or_else(usage)?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--suffix" => suffix = it.next().ok_or_else(usage)?,
            "-h" | "--help" => return Err(usage()),
            _ => positional.push(PathBuf::from(a)),
        }
    }
    let [models, clip, out] = <[PathBuf; 3]>::try_from(positional).map_err(|_| usage())?;
    Ok(Args {
        models,
        clip,
        out,
        suffix,
        config,
    })
}

fn encode(img: &Image) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let w = u16::try_from(img.width()).map_err(|e| e.to_string())?;
    let h = u16::try_from(img.height()).map_err(|e| e.to_string())?;
    jpeg_encoder::Encoder::new(&mut out, 85)
        .encode(img.data(), w, h, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

fn manifest_events(clip: &Path) -> Vec<(String, i64)> {
    let dir = clip.parent().unwrap_or(Path::new("."));
    let name = clip.file_name().map(|n| n.to_string_lossy().into_owned());
    let Ok(text) = std::fs::read_to_string(dir.join("manifest.json")) else {
        return Vec::new();
    };
    let entries = match serde_json::from_str::<Value>(&text) {
        Ok(Value::Array(a)) => a,
        Ok(Value::Object(o)) => o
            .get("clips")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    entries
        .iter()
        .find(|e| e.get("file").and_then(Value::as_str) == name.as_deref())
        .and_then(|e| e.get("events").and_then(Value::as_object))
        .map(|ev| {
            ev.iter()
                .filter_map(|(k, v)| v.as_i64().map(|t| (k.clone(), t)))
                .collect()
        })
        .unwrap_or_default()
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let models = PoseModels::load(&args.models, &args.config).map_err(|e| e.to_string())?;
    let dir = args.clip.parent().unwrap_or(Path::new("."));
    let stem = args
        .clip
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("clip name")?
        .to_owned();
    let (mut reader, sidecar) = stance::open_clip(dir, &stem).map_err(|e| e.to_string())?;
    let t0 = std::time::Instant::now();
    let track = analyze_clip(&models, &args.clip, &args.config).map_err(|e| e.to_string())?;
    eprintln!(
        "{stem}: {} frames posed in {:.2} s",
        track.frames.len(),
        t0.elapsed().as_secs_f64()
    );

    let name = format!("{stem}{}", args.suffix);
    let mut events = vec![(
        "impact".to_owned(),
        sidecar
            .impact_frame()
            .map_or(0, |f| i64::try_from(f.t_ns).unwrap_or(0)),
    )];
    events.extend(
        manifest_events(&args.clip)
            .into_iter()
            .filter(|(k, _)| k != "impact"),
    );
    let still_at: Vec<(String, usize)> = events
        .iter()
        .filter_map(|(k, t)| {
            (0..sidecar.frames.len())
                .min_by_key(|&i| {
                    (i64::try_from(sidecar.frames[i].t_ns).unwrap_or(i64::MAX) - t).abs()
                })
                .map(|i| (k.clone(), i))
        })
        .collect();

    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let mut frames = Vec::with_capacity(sidecar.frames.len());
    for (i, f) in sidecar.frames.iter().enumerate() {
        let jpeg = reader.read_frame(i).map_err(|e| e.to_string())?;
        let (out, width, height) = match decode_jpeg(&jpeg) {
            Ok(mut img) => {
                let style = draw::SkeletonStyle::for_image(&img);
                draw::skeleton(&mut img, track.skeleton, &track.frames[i].keypoints, style);
                (encode(&img)?, img.width(), img.height())
            }
            Err(e) => {
                eprintln!("frame {i}: {e}; kept as is");
                (jpeg, sidecar.width, sidecar.height)
            }
        };
        for (event, _) in still_at.iter().filter(|(_, at)| *at == i) {
            let path = args.out.join(format!("{name}-{event}.jpg"));
            std::fs::write(&path, &out).map_err(|e| format!("{}: {e}", path.display()))?;
            println!("{}", path.display());
        }
        frames.push(Frame {
            jpeg: Arc::from(out),
            host_ns: sidecar.start_ns.saturating_add(f.t_ns),
            device_ns: None,
            seq: i as u64,
            width,
            height,
        });
    }
    let abs = |rel: i64| sidecar.start_ns.saturating_add_signed(rel);
    let clip = Clip {
        id: ClipId(0),
        tags: vec![stem.clone()],
        frames,
        trigger_index: sidecar.trigger_index,
        impact_index: sidecar.impact_index,
        trigger_ns: abs(sidecar.trigger_ns),
        impact_ns: abs(sidecar.impact_ns),
        complete: sidecar.complete,
        dropped: sidecar.dropped,
        truncated: sidecar.truncated,
        short_buffer: false,
        fps: sidecar.fps.nominal,
    };
    let files = stance::write_clip(&args.out, &name, &clip).map_err(|e| e.to_string())?;
    println!("{}", files.avi.display());
    let pose_path = track.write_beside(&files.avi).map_err(|e| e.to_string())?;
    println!("{}", pose_path.display());
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
