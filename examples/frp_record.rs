//! Records a clip per shot from a Flight Relay Protocol bus.
//!
//! Joins an FRP bus (a hub or a launch monitor) as a collector actor,
//! captures one camera (or the test pattern) and writes
//! `<shot number>-<shot id>.avi` + `.json` for every `shot.trigger`. Its actor
//! id is new on every run. With `--keyboard`, pressing Enter triggers a shot
//! instead.
//!
//! ```text
//! cargo run --example frp_record --features frp -- --list
//! cargo run --example frp_record --features frp -- --camera test-pattern --url ws://localhost:5880/frp
//! cargo run --example frp_record --features frp -- --camera <id> --mode 1280x720@120 --lead-ms 0 --out clips
//! ```

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use flightrelay::types::roles;
use flightrelay::{
    BusActor, DEFAULT_HEARTBEAT, EndpointConfig, FrpBody, FrpEndpoint, FrpError, FrpMessage,
    ShotKey, Transport,
};
use stance::frp::trigger_from_message;
use stance::{
    Capture, CaptureConfig, CaptureStatus, ClipRecorder, FormatRequest, RecorderConfig,
    RecorderEvent, Source, TEST_PATTERN_ID, TestPattern, Trigger, clock,
};

struct Args {
    list: bool,
    url: String,
    camera: String,
    mode: FormatRequest,
    out: PathBuf,
    pre: Duration,
    post: Duration,
    lead: Duration,
    keyboard: bool,
}

const USAGE: &str = "usage: frp_record [--list] [--url URL] [--camera ID|test-pattern] \
[--mode auto|WxH@FPS] [--out DIR] [--pre SECS] [--post SECS] [--lead-ms MS] [--keyboard]";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        list: false,
        url: flightrelay::DEFAULT_URL.to_owned(),
        camera: TEST_PATTERN_ID.to_owned(),
        mode: FormatRequest::Auto,
        out: PathBuf::from("clips"),
        pre: Duration::from_secs(4),
        post: Duration::from_millis(1500),
        lead: Duration::ZERO,
        keyboard: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
        };
        let secs = |v: String| {
            v.parse::<f64>()
                .ok()
                .filter(|s| s.is_finite() && *s >= 0.0)
                .map(Duration::from_secs_f64)
                .ok_or_else(|| format!("{flag}: `{v}` is not a number of seconds"))
        };
        match flag.as_str() {
            "--list" => a.list = true,
            "--keyboard" => a.keyboard = true,
            "--url" => a.url = value()?,
            "--camera" => a.camera = value()?,
            "--mode" => a.mode = value()?.parse().map_err(|e| format!("{e}"))?,
            "--out" => a.out = PathBuf::from(value()?),
            "--pre" => a.pre = secs(value()?)?,
            "--post" => a.post = secs(value()?)?,
            "--lead-ms" => {
                let v = value()?;
                let ms: u64 = v.parse().map_err(|_| format!("--lead-ms: `{v}`"))?;
                a.lead = Duration::from_millis(ms);
            }
            "-h" | "--help" => return Err(USAGE.to_owned()),
            other => return Err(format!("unknown argument `{other}`\n{USAGE}")),
        }
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    if args.list {
        for cam in stance::list().map_err(|e| e.to_string())? {
            println!("{}\n  id: {}", cam.name, cam.id);
            if let Some(usb) = cam.usb {
                println!("  usb: {usb}");
            }
            if let Some(e) = &cam.probe_error {
                println!("  (could not read modes: {e})");
            }
            for f in cam.mjpeg_formats() {
                println!("  {f}");
            }
        }
        return Ok(());
    }

    let camera = args.camera.clone();
    let mode = args.mode;
    let capture = Capture::start(
        CaptureConfig::default(),
        move || -> stance::Result<Box<dyn Source>> {
            if camera == TEST_PATTERN_ID {
                let config = TestPattern::config_for(&mode);
                Ok(Box::new(TestPattern::new(config)?))
            } else {
                Ok(Box::new(stance::open(&camera, &mode)?))
            }
        },
    )
    .map_err(|e| e.to_string())?;
    println!("capturing {} ({})", capture.info().name, capture.format());

    let (tx, rx) = mpsc::sync_channel::<Trigger<ShotKey>>(16);
    if args.keyboard {
        println!("press Enter to trigger a shot");
        thread::spawn(move || keyboard(&tx));
    } else {
        let (url, lead) = (args.url.clone(), args.lead);
        thread::spawn(move || {
            if let Err(e) = frp(&url, lead, &tx) {
                eprintln!("FRP: {e}");
            }
        });
    }
    record(args, &capture, &rx)
}

/// Pulls frames into the recorder, applies triggers, writes finished clips.
fn record(args: &Args, capture: &Capture, rx: &Receiver<Trigger<ShotKey>>) -> Result<(), String> {
    let mut recorder = ClipRecorder::new(RecorderConfig {
        pre_roll: args.pre,
        post_roll: args.post,
        buffer: args.pre + args.lead + Duration::from_secs(2),
        fps: Some(capture.format().fps()),
        ..RecorderConfig::default()
    });
    let mut cursor: Option<u64> = None;
    let mut triggers_open = true;
    loop {
        let frames = capture.wait_frames_after(cursor, Duration::from_millis(50));
        if let (Some(c), Some(first)) = (cursor, frames.first())
            && first.seq > c + 1
        {
            eprintln!("warning: fell behind, {} frames skipped", first.seq - c - 1);
        }
        let mut events = Vec::new();
        for frame in frames {
            cursor = Some(frame.seq);
            events.extend(recorder.push(frame));
        }
        while triggers_open {
            match rx.try_recv() {
                Ok(t) => {
                    let what =
                        format!("#{} {} from {}", t.tag.shot_number, t.tag.shot_id, t.source);
                    println!("{what}: {:?}", recorder.trigger(t));
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => triggers_open = false,
            }
        }
        events.extend(recorder.poll(clock::now_ns()));
        if let CaptureStatus::Failed(e) = capture.status() {
            events.extend(recorder.flush());
            save(args, events);
            return Err(format!("capture stopped: {e}"));
        }
        save(args, events);
        if !triggers_open && recorder.open_clips() == 0 {
            return Ok(());
        }
    }
}

fn save(args: &Args, events: Vec<RecorderEvent<ShotKey>>) {
    for event in events {
        match event {
            RecorderEvent::Finished(clip) => {
                let Some(key) = clip.tags.first() else {
                    continue;
                };
                let name = format!("{:04}-{}", key.shot_number, key.shot_id);
                match stance::write_clip(&args.out, &name, &clip) {
                    Ok(files) => println!(
                        "wrote {} ({} frames, {:.1} s{}{})",
                        files.avi.display(),
                        clip.frames.len(),
                        clip.duration().as_secs_f64(),
                        if clip.complete { "" } else { ", incomplete" },
                        if clip.truncated { ", truncated" } else { "" },
                    ),
                    Err(e) => eprintln!("writing {name}: {e}"),
                }
            }
            RecorderEvent::Discarded(d) => {
                eprintln!("discarded clip {:?} ({:?})", d.id, d.reason);
            }
        }
    }
}

/// Joins the bus as a collector, keeps its `bus.actor` alive and forwards
/// shot triggers, stamped on arrival.
fn frp(url: &str, lead: Duration, tx: &SyncSender<Trigger<ShotKey>>) -> Result<(), String> {
    let me = FrpMessage::new(
        flightrelay::new_actor_id(),
        BusActor::new("stance frp_record", DEFAULT_HEARTBEAT.as_secs_f64())
            .with_role(roles::COLLECTOR),
    );
    let config = EndpointConfig::new(Transport::connect(url));
    let mut endpoint = FrpEndpoint::open(config).map_err(|e| e.to_string())?;
    let mut conn = endpoint.establish().map_err(|e| e.to_string())?;
    conn.greet(std::slice::from_ref(&me), &[])
        .map_err(|e| e.to_string())?;
    println!(
        "connected to {url} (FRP {}) as {}",
        flightrelay::SPEC_VERSION,
        me.from
    );
    let mut beat = Instant::now();
    loop {
        let left = DEFAULT_HEARTBEAT.saturating_sub(beat.elapsed());
        match conn.recv_timeout(left) {
            Ok(Some(msg)) => {
                if matches!(msg.body, FrpBody::BusSync(_)) {
                    conn.send(&me).map_err(|e| e.to_string())?;
                } else if let Some(t) = trigger_from_message(&msg, clock::now_ns(), lead)
                    && tx.send(t).is_err()
                {
                    return Ok(());
                }
            }
            // Nothing arrived in time, or one message that is not FRP: keep reading.
            Ok(None) | Err(FrpError::Json(_)) => {}
            Err(e) => return Err(e.to_string()),
        }
        if beat.elapsed() >= DEFAULT_HEARTBEAT {
            beat = Instant::now();
            conn.send(&me).map_err(|e| e.to_string())?;
        }
    }
}

/// Every line on stdin is a shot.
fn keyboard(tx: &SyncSender<Trigger<ShotKey>>) {
    let mut line = String::new();
    for n in 1.. {
        line.clear();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {
                let key = ShotKey::new(format!("keyboard-{n}"), n);
                if tx
                    .send(Trigger::new(clock::now_ns(), "keyboard", key))
                    .is_err()
                {
                    return;
                }
            }
        }
    }
}
