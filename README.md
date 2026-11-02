# stance

Camera library for golf swing replay, in pure Rust: USB capture, a pre-roll
buffer and shot-triggered clips.

`stance` reads MJPEG frames from USB (UVC) webcams and machine-vision
cameras, keeps the last few seconds in a pre-roll buffer, cuts a clip around
each shot trigger, and writes it as an MJPEG AVI with a JSON sidecar. Frames
are stored exactly as the camera compressed them: nothing is decoded or
re-encoded on the capture path, so a 120 fps camera costs almost no CPU.

Pose estimation is planned in this same crate later, behind an opt-in cargo
feature, so applications that only capture never pull in ML dependencies.

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

`camera::SUPPORTED` says whether this platform has a camera backend.
| wasm32 | none | Frame and format types, `Ring`, `ClipRecorder`, the sidecar and `jpeg` build (e.g. for a browser replay UI). |

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

## Resource use

MJPEG at 1280×720 and 120 fps is roughly 10 MB/s, so a 5.5 s clip is about
55 MB per camera and a 10 s pre-roll ring about 100 MB of memory. The ring
byte caps bound memory regardless of the camera.

## License

MIT OR Apache-2.0.
