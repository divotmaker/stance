# stance

Camera library for golf swing replay, in pure Rust: USB capture, a pre-roll
buffer and shot-triggered clips.

`stance` reads MJPEG frames from USB (UVC) webcams and machine-vision
cameras, keeps the last few seconds in a pre-roll buffer, cuts a clip around
each shot trigger, and writes it as an MJPEG AVI with a JSON sidecar. Frames
are stored exactly as the camera compressed them: nothing is decoded or
re-encoded on the capture path, so a 120 fps camera costs almost no CPU.

With the opt-in `pose` feature (alpha) it also estimates body keypoints:
live on camera frames, or over a recorded clip after the shot. Applications
that only capture never pull in the ML dependencies.

Status: **beta** (0.0.x). The API may still change.

## How it works

```text
camera ──► Capture thread ──► Ring (bounded) ──► your loop ──► ClipRecorder ──► Clip ──► write_clip
                                                    ▲
                     shot triggers (FRP, buttons) ──┘
```

1. **Sources.** `list` enumerates cameras with their modes and controls;
   `open` starts one in MJPEG. `TestPattern` (generated frames with a moving
   marker and a frame counter) and `AviFile` (playback of a recorded clip)
   implement the same `Source` trait, so everything downstream can be tested
   without hardware.
2. **Capture.** `Capture` runs a source on its own thread and pushes each
   frame straight into a `Ring` bounded by duration and bytes. The ring is the
   only buffer: memory stays bounded, and a consumer that falls behind loses
   the oldest frames instead of growing a queue. It also keeps the latest
   frame (for a still image), counts achieved fps and dropped frames, and
   reports a failed, unplugged or silent source as an error instead of
   spinning. `Capture::start` blocks while the camera opens and `stop` waits
   for the thread; async callers start it in a blocking task and stop it
   with the non-blocking `request_stop`. Frame sequence numbers keep rising
   across captures, so a consumer's cursor survives a reconnect.
3. **Recording.** `ClipRecorder` is fed frames and triggers and hands back
   finished clips. It is pure and deterministic (no threads, no clock of its
   own) and keeps its own pre-roll ring. Call its `poll` on a timer: when a
   camera stops, nothing else finishes the open clips.
4. **Storage.** `write_clip` writes `<name>.avi` (MJPEG in a plain AVI 1.0
   file) and `<name>.json`, a sidecar listing every frame's time and byte
   offset inside the AVI. The reader treats files as untrusted: corrupt
   sizes are clamped to the file, never trusted for an allocation.

All times are host monotonic nanoseconds (`clock::now_ns`): QPC on Windows,
`CLOCK_MONOTONIC` on Linux and macOS. Frames are stamped when they arrive;
callers stamp triggers with the same clock.

## Clip rules

Each trigger has a time `at_ns`, a `source` (the launch monitor that
triggered), a tag (e.g. the shot's id) and an optional `impact_lead_ns`.

- **Impact estimate** = `at_ns − impact_lead_ns`. Launch monitors that report
  after the ball has left trigger late; the lead moves the clip back to the
  swing.
- **Window** = `[impact − pre_roll, latest trigger + post_roll]`. Defaults:
  4 s pre-roll, 1.5 s post-roll. Rolls can be set per trigger or changed at
  any time for later triggers.
- **Same source again** → if the open clip with a trigger from that source is
  still inside its window, it ends at the new trigger's time
  (`truncated: true`) and a new clip starts. A clip whose window already
  ended finishes normally.
- **Different source within the merge window** (default 500 ms between impact
  estimates) → the trigger joins the newest open clip that matches (every
  open clip is checked); the clip carries both tags.
- **Otherwise** a new clip starts. Overlapping clips share frames (they are
  reference-counted, not copied).
- A clip **finishes** when a frame past its window arrives, or through
  `poll(now)` / `flush()` when frames stop (camera unplugged). It is kept
  when it holds its **impact frame** and discarded (reported as `Discarded`)
  when it doesn't.
- `complete` is true when frames past the window arrived and none are
  missing across the trigger or the impact. A clip may end between the
  impact and a late trigger: `trigger_index` is then `frames.len()`.
- `dropped` counts frames missing inside the clip: gaps longer than 1.5 frame
  periods on device timestamps, or 2.5 periods on arrival times (which
  jitter).
- `short_buffer` flags a clip whose pre-roll plus impact lead didn't fit the
  recorder's `buffer` (`RecorderConfig::buffer_covers` checks a lead up
  front).
- **Duplicates are the caller's job.** Every trigger counts: the same source
  triggering twice for one shot cuts the first clip and starts another.

## Sidecar

```json
{
  "version": 1,
  "fourcc": "MJPG",
  "width": 1280,
  "height": 720,
  "fps": { "nominal": 120.0, "measured": 119.62 },
  "start_ns": 81234567890123,
  "frames": [
    { "t_ns": 0, "offset": 232, "len": 41022 },
    { "t_ns": 8358000, "offset": 41262, "len": 40876 }
  ],
  "trigger_index": 480,
  "impact_index": 480,
  "trigger_ns": 4012000000,
  "impact_ns": 4012000000,
  "complete": true,
  "dropped": 0,
  "truncated": false
}
```

- `trigger_index` may be `frames.len()` (the clip ends before its trigger);
  `impact_index` always points at a frame. `Sidecar::trigger_frame` and
  `impact_frame` index safely.
- `t_ns` is each frame's host time relative to the first frame; `start_ns` is
  the first frame's absolute host time, so clips from several cameras on one
  host line up.
- `offset`/`len` locate each JPEG inside the AVI file, so a browser can fetch
  single frames with HTTP `Range` requests instead of the whole clip.
- `trigger_ns`/`impact_ns` are relative to the first frame (`impact_ns` is
  negative if the estimate precedes the clip).
- The AVI header carries the measured frame rate, so players show the clip at
  real speed even when the camera ran below its nominal rate.
- Files are written as `.part`, synced and renamed; the sidecar is written
  last, so a clip whose sidecar exists is complete on disk.

The sidecar holds capture facts only. What a clip shows (camera position,
which shot) belongs to the application.

## Frames without Huffman tables

Many webcams send MJPEG frames without a DHT segment and rely on the decoder
assuming the standard tables. `jpeg::ensure_huffman` inserts them (without
touching the image data) so strict JPEG decoders accept the frame. The AVI
keeps the bytes as sent; call it before decoding.

## Platforms

| Platform | Camera backend | Notes |
|---|---|---|
| Windows | Media Foundation | Native MJPEG modes, converters disabled. Reads are synchronous and ignore the read timeout; stopping (or a stall) shuts the camera down, which ends a pending read. `list` doesn't probe cameras this process has open (it returns their info from when they were opened); cameras open in other applications are listed with `probe_error`. |
| Linux (64-bit) | V4L2 (raw ioctls, MMAP buffers) | Driver timestamps are used as the device time when monotonic. Buffers the driver marks as failed or empty are skipped. `list` doesn't open cameras this process has open either. |
| macOS | none yet | `list`/`open` return `Error::Unsupported`; everything else works. |
| wasm32 | none | Frame and format types, `Ring`, `ClipRecorder`, the sidecar and `jpeg` build (e.g. for a browser replay UI). |

`camera::SUPPORTED` says whether this platform has a camera backend.

Only MJPEG is captured: a camera without an MJPEG mode returns a clear error.
`FormatRequest::Auto` picks the highest MJPEG frame rate, then the largest
size; `"1280x720@120"` asks for an exact mode. For the test pattern,
`TestPattern::config_for` maps a request the same way over
`TestPattern::modes`, and `TestPattern::camera_info` describes it as a camera
list entry.

Camera ids are stable across replugs and reboots: the Media Foundation
symbolic link on Windows; on Linux the `/dev/v4l/by-id` path, or
`<bus_info>/<card>` (unique per USB port) for identical cameras without serial
numbers. Names are for display and need not be unique.

## Example

```rust,no_run
use std::time::Duration;
use stance::{
    Capture, CaptureConfig, ClipRecorder, Error, FormatRequest, RecorderConfig, RecorderEvent,
    Trigger, clock,
};

fn main() -> stance::Result<()> {
    let cameras = stance::list()?;
    let id = cameras.first().ok_or(Error::NotFound("any camera".into()))?.id.clone();
    let capture = Capture::start(CaptureConfig::default(), move || {
        stance::open(&id, &FormatRequest::Auto)
    })?;
    let mut recorder = ClipRecorder::new(RecorderConfig::default());
    let mut cursor = None;
    let mut shots = 0;
    loop {
        let mut events = Vec::new();
        for frame in capture.wait_frames_after(cursor, Duration::from_millis(50)) {
            cursor = Some(frame.seq);
            events.extend(recorder.push(frame));
        }
        if shot_detected() {
            shots += 1;
            recorder.trigger(Trigger::new(clock::now_ns(), "monitor-1", shots));
        }
        // Every pass, frames or not: poll finishes clips when frames stop.
        events.extend(recorder.poll(clock::now_ns()));
        for event in events {
            if let RecorderEvent::Finished(clip) = event {
                let name = format!("shot-{}", clip.tags[0]);
                stance::write_clip("clips".as_ref(), &name, &clip)?;
            }
        }
    }
}

fn shot_detected() -> bool {
    false // your launch monitor here
}
```

### Flight Relay Protocol

With the `frp` feature, `frp::trigger_from_message` turns a Flight Relay
Protocol `shot_trigger` into a `Trigger<ShotKey>` whose source is the
envelope's device. The `frp_record` example connects to any FRP server and
writes a clip per shot:

```text
cargo run --example frp_record --features frp -- --list
cargo run --example frp_record --features frp -- --camera <id> --url ws://localhost:5880/frp --out clips
cargo run --example frp_record --features frp -- --camera test-pattern --keyboard
```

## Pose estimation (`pose` feature, alpha)

`stance::pose` finds the golfer's body keypoints in frames and clips. It is
a top-down pipeline in pure Rust: a YOLOX person detector finds the golfer,
then RTMPose finds 26 Halpe keypoints inside that box (head, neck, shoulders,
elbows, wrists, hips, knees, ankles, heels and toes). Inference runs on the
CPU through [tract](https://github.com/sonos/tract), so an application stays a
single executable with no native libraries, on Windows, Linux and macOS.

- **Live:** `PoseTracker` follows one golfer frame by frame. It detects on
  the first frame, every `detect_every` frames (default 10) and whenever the
  pose score drops, and otherwise crops around the previous frame's
  keypoints, falling back to the detector's box when those collapse onto
  part of the body (e.g. only the head scoring at the top of the
  backswing). The golfer is the most confident large person near the frame
  centre; later detections stay on the same person by box overlap. Once
  tracking is lost, the next detection picks the golfer afresh, which may
  be someone else if the golfer has left. Optional One Euro smoothing;
  `inference_threads` lets one inference use several cores (one thread pool
  per tracker, shared by its two models) for lower latency.
- **After the shot:** `analyze_clip` runs a recorded clip on several threads
  (detection on key frames, interpolated boxes, pose on every frame, then a
  sequential pass that redoes weak frames from their neighbours' keypoints)
  and returns a `PoseTrack`: raw keypoints per frame on the sidecar's time
  base. `PoseTrack::write_beside` stores it as `<clip>.pose.json`. Decoded
  frames held at once stay under 256 MiB across all workers. `threads` sets
  the worker count, and `on_worker_start` (a `WorkerHook`) runs at the start
  of every worker thread, on that thread, e.g. to lower its priority so a
  simulator on the same computer keeps the CPU it needs.
- **Absent, not garbage:** a pose whose mean score is below
  `min_pose_score` (default 0.3) is `None` live and an empty frame in a
  `PoseTrack`, never stored.
- **Any engine:** `PoseModels`, `PoseTracker` and `analyze_clip` work over
  the `PoseEngine` and `PersonDetector` traits; RTMPose and YOLOX are the
  defaults (`PoseModels::load`), and another pair plugs in through
  `PoseModels::new`.
- **Filters:** `OneEuro` for live smoothing and a zero-phase `Butterworth`
  for whole clips. Tracks store raw keypoints; filtering is derived.

The image pipeline is a port of [rtmlib](https://github.com/Tau-J/rtmlib)'s
RTMPose path (Apache-2.0, itself from MMPose and YOLOX; see `NOTICE`): box → crop with 1.25 padding at the model's 3:4
aspect → bilinear affine warp to 192×256 → mean/std normalisation → SimCC
argmax → back through the inverse warp. On the test image, keypoints match
rtmlib's code fed RGB (as the model was trained; stock rtmlib feeds BGR, so
it differs slightly and systematically) to a mean of 0.1–0.2 px, at most
about one SimCC bin (≤ 1.3 px).

### Models

Models are not bundled. `ModelSpec` lists each one's source, SHA-256, size
and input size. `RtmPose::load`, `Yolox::load` and `PoseModels::load` read
`<dir>/<name>.onnx` and refuse a file whose SHA-256 doesn't match. With the
`pose-download` feature, `fetch_model` (or `PoseModels::download`) downloads
a model over HTTPS only (rustls, redirects included) into a cache directory,
extracts it from its zip archive when the source is one, verifies it and
renames it into place; concurrent fetches never see a partial file. A
download has bounds: 30 s to connect, 60 s for any one read, 15 minutes in
all. `fetch_model_with` takes an optional progress callback (bytes read,
and the total when the server sends one) and an optional cancel flag, which
stops the download and removes its partial files.

| Name | Model | Size | Licence |
|---|---|---|---|
| `rtmpose-m-halpe26` (default) | RTMPose-m, Halpe-26, 256×192 | 56 MB | Apache-2.0 |
| `rtmpose-s-halpe26` | RTMPose-s, Halpe-26, 256×192 | 23 MB | Apache-2.0 |
| `yolox-tiny` (default) | YOLOX-Tiny, COCO, 416×416 | 20 MB | Apache-2.0 |
| `yolox-nano` | YOLOX-Nano, COCO, 416×416 | 4 MB | Apache-2.0 |

The RTMPose weights are OpenMMLab's official ONNX exports, labelled
Apache-2.0; they were trained on public pose datasets (COCO, AI Challenger,
CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18) whose own terms are mixed,
some research-only. The YOLOX weights are Megvii's release, Apache-2.0,
trained on COCO.

### Usage

```rust,ignore
use std::path::Path;
use stance::pose::{PoseConfig, PoseModels, decode_jpeg};

let config = PoseConfig::default();
let models = PoseModels::load(Path::new("models"), &config)?; // or PoseModels::download

// Live: one frame at a time.
let mut tracker = models.tracker(config.clone())?;
let image = decode_jpeg(&frame.jpeg)?;
if let Some(pose) = tracker.track(&image, frame.host_ns)? {
    if let Some(wrist) = pose.keypoint("left_wrist") {
        println!("left wrist at ({:.0}, {:.0}), score {:.2}", wrist.x, wrist.y, wrist.score);
    }
}

// After the shot: the whole clip on several threads.
let track = stance::pose::analyze_clip(&models, &files.avi, &config)?;
track.write_beside(&files.avi)?; // <clip>.pose.json
```

### Examples

```text
cargo run --release --example pose_bench --features pose -- <models dir> <clips dir | clip.avi>... [--json out.json]
cargo run --release --example pose_render --features pose -- <models dir> <clip.avi> <out dir>
```

`pose_bench` reports detector and pose latency, the clip pass's wall time,
mean keypoint scores per body part, scores around the impact and left/right
swaps per clip. `pose_render` writes the clip with the skeleton drawn on it
(the person's left side cyan, right orange; faint where the score is low),
its `.pose.json`, and stills at the impact frame.

Both draw with `stance::pose::draw`: blended lines and discs, a skeleton
and a 5×7 pixel font for labels, in pure Rust, for any application that
wants to annotate frames.

### Golf metrics (alpha)

`stance::golf` turns a clip's `PoseTrack` into a `SwingReport`: swing
events, tempo and body metrics for the camera's view, at the key positions
and at every frame. It is pure math on the
keypoints (no models), behind the same `pose` feature.

- **Cleaning:** keypoints scoring below 0.3 are gaps, never zeros. Gaps up
  to 150 ms are interpolated, longer ones stay empty. Each keypoint is
  smoothed with a zero-phase 8 Hz Butterworth, filtered separately before
  and after impact so the fastest frames don't smear.
- **Events** from the hands (the wrists' midpoint): address (hands at
  rest), takeaway (they start moving), top (the start of the downswing, so
  a pause there counts as backswing), impact, and finish (the hands settle
  high after the follow-through). Each has a time on the track's time base
  and a confidence.
- **Impact** is 25 ms after the hands' low point in the swing's clearest
  down-and-up (a fast downswing of plausible length into a follow-through,
  so a practice swing's lowering or hands brought down to address don't
  count). A launch monitor's shot time is a prior, not the answer: the low
  point is searched within ±0.3 s of it (`ImpactSource::Refined`), and only
  with no clear low point there does the given time stand
  (`ImpactSource::Given`). That means no swing was found near the time:
  the report then has the impact alone, with no other event, tempo or
  handedness check. Without a time, impact is estimated from the whole clip
  (`ImpactSource::Estimated`). The top's confidence drops for a downswing
  outside 120–700 ms, to 0 below 60 ms and from 1 s. The 25 ms is on the
  track's time base, so it is only right for a clip on capture time.
- **Tempo:** backswing (takeaway → top) against downswing (top → impact);
  none when the top's confidence is 0.
- **Limits:** an estimated impact trusts the keypoints, so a pose model
  that latches onto something else for a stretch can make a false swing
  that outscores the real one; pass the launch monitor's time whenever it
  is known. Two similar low points near a given time are told apart by
  their distance from it, so on a hard clip (golfer partly out of frame)
  the impact can move between them as the time moves.
- **Metrics** at address, top, impact and finish, each with its view, unit
  and a confidence from the keypoint scores:

| Face-on | Down the line | Either view |
|---|---|---|
| hip sway (top, impact), head shift (lateral), lateral spine tilt, shoulder tilt (address, impact), lead-arm angle, stance width, hip width proxy (a pelvis-turn proxy; absent when above 1), hip turn and shoulder turn (estimates; top, impact) | spine forward bend, loss of posture, early extension, lead and trail knee flex, trail-elbow angle, hand height at the top | head shift (vertical) |

- **Per frame:** `SwingReport::series` holds every measured metric's value
  at every frame of the smoothed track (`t_ns` plus one value list per
  metric, `null` where the keypoints weren't found), with the same formulas
  and keypoints as the key positions, so a display can follow a replay
  frame by frame. Shifts and turns are against the address frame. Values
  are rounded to 4 decimal places; a face-on report is about 100 bytes per
  frame (about 45 kB for a 460-frame clip). Reports without a series still
  read.

All metrics are **2D proxies** measured in one camera's image plane, not 3D
joint angles: motion towards or away from the camera is invisible, and
rotations only show as foreshortening.
**Hip and shoulder turn** are one-camera estimates from that
foreshortening: `acos(w / w₀)` of the hip-to-hip (shoulder-to-shoulder)
distance against address, in degrees, absent when the ratio is above 1.
They need the body square to the camera at address (any angle there reads
as turn) and can't tell which way the body turned. Keypoint drift grows with
the turn (the far hip or shoulder is hidden), a body seen end-on still shows
its depth and the forward bend tilts the shoulder line rather than
shortening it, so large turns read low; near 90° they saturate (a few
pixels move them by many degrees) and a turn past 90° reads as less, so
they aren't reported at the finish. X-factor and the kinematic sequence
need two calibrated views and are not reported.
Distances are fractions of the golfer's torso length at address, and
centimetres when the golfer's height is given. Face-on, the forward bend
foreshortens the torso (by about 0.82 at 35°), so face-on fractions read
about 20 % larger than down the line; centimetres are unaffected. The
lateral shifts face-on (hip sway: the pelvis, head shift: the head, both
from address) are positive towards the target and negative away from it.
The rustdoc of `MetricId` gives every formula.

Handedness is the caller's (`Handed`); a left-hander's metrics are the exact
mirror image of a right-hander's. Face-on, a right-hander's target is on the
image's right; down the line, a right-hander faces the image's right. The
report also guesses the view from the address pose (unknown when the cues
disagree; only the either-view metrics are measured then) and checks the
handedness against the swing. A mirrored camera image (a selfie-style
preview) makes a right-hander look left-handed: set `SwingOptions::mirrored`
(live: measure `unmirror(&pose, width)`) and give the golfer's true hand.
The report reader skips metrics it doesn't know, so reports from a later
crate version still load.

```rust,ignore
use stance::golf::{Handed, SwingOptions, View, analyze_swing, posture};

let report = analyze_swing(&track, &SwingOptions {
    handed: Handed::Right,
    impact_ns: Some(impact_ns), // launch monitor (refined); None estimates it
    height_cm: Some(180.0),
    ..SwingOptions::default()   // view: guessed from the address pose
})?;
report.write_beside(&files.avi)?; // <clip>.swing.json

// Live, beside the camera view: a few angles per frame.
let view = report.view.unwrap_or(View::FaceOn);
let now = posture(&pose, view, report.handed);
println!("spine {:?}°, head shift {:?}", now.spine_angle, now.head_shift(&at_address));
```

```text
cargo run --release --example golf_report --features pose -- <models dir> <clip.avi> \
    [--view fo|dtl] [--handed r|l] [--impact-ns N | --estimate-impact] [--height-cm H] \
    [--mirrored] [--render <out dir>]
```

`golf_report` prints the report and, with `--render`, writes the
`.swing.json` and a still at each event with the skeleton, the hands' path
and the numbers drawn on it.

## Resource use

MJPEG at 1280×720 and 120 fps is roughly 10 MB/s, so a 5.5 s clip is about
55 MB per camera and a 10 s pre-roll ring about 100 MB of memory. The ring
byte caps bound memory regardless of the camera.

## License

MIT OR Apache-2.0.
