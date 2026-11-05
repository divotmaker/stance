# CLAUDE.md — stance conventions

`stance` is a single crate, the camera library for golf swing replay: USB
camera capture, pre-roll ring, trigger-cut clips, MJPEG AVI + JSON sidecar.
Pose estimation (RTMPose keypoints, YOLOX person detection, tract inference)
lives behind the opt-in `pose` feature (`pose-download` adds fetching models
over HTTPS), and FRP behind `frp`, so consumers that only capture never pull
in ML or protocol dependencies.

## Layout

| Path | Contents |
|---|---|
| `src/lib.rs` | Crate docs, module list, re-exports, target gating |
| `src/error.rs` | `Error` (thiserror), `Result` |
| `src/frame.rs` | `Frame`, `Format` |
| `src/camera/mod.rs` | `CameraInfo`, `UsbId`, controls, `FormatRequest`, `choose_format`, `list`, `open`, `Camera`, the registry of cameras open in this process |
| `src/camera/mf.rs` | Windows backend (Media Foundation) |
| `src/camera/v4l2.rs` | Linux 64-bit backend (raw V4L2 ioctls, MMAP) |
| `src/camera/unsupported.rs` | Every other target: `Error::Unsupported` |
| `src/source.rs` | `Source` trait, `Interrupter` |
| `src/test_pattern.rs` | `TestPattern` synthetic source |
| `src/avi.rs` | AVI writer (crate-private), `AviReader`, `AviFile` source |
| `src/ring.rs` | `Ring` (duration + byte bounded) |
| `src/capture.rs` | `Capture` thread |
| `src/recorder.rs` | `ClipRecorder`, `Trigger`, `Clip` and the clip rules |
| `src/sidecar.rs` | `Sidecar` JSON types (all targets) |
| `src/storage.rs` | `write_clip`, `read_sidecar`, `open_clip` |
| `src/jpeg.rs` | `ensure_huffman` (all targets) |
| `src/clock.rs` | `now_ns` (QPC / `CLOCK_MONOTONIC`) |
| `src/frp.rs` | `frp` feature: FRP `shot_trigger` → `Trigger<ShotKey>` |
| `src/pose/mod.rs` | `pose` feature: module docs, `Keypoint`, `Pose`, `PoseEngine`/`PersonDetector` traits (with `fork`), `PoseConfig`, `PoseModels<P, D>`, the crop-collapse rule |
| `src/pose/image.rs` | `Image`, `decode_jpeg`, the decode budget, the affine-warp and letterbox tensor builders |
| `src/pose/geometry.rs` | `BBox`, `Detection`, affine transforms, rtmlib's box → crop, NMS |
| `src/pose/skeleton.rs` | `Skeleton` (Halpe-26 tables: names, bones, left/right pairs, mirror), `Side` |
| `src/pose/model.rs` | `ModelSpec` table (URL, zip member, SHA-256, size, input), hash verification |
| `src/pose/runtime.rs` | The inference seam: the only file that uses tract (`Network`, `Session`, `InferencePool`) |
| `src/pose/rtmpose.rs` | `RtmPose` (crop, normalise, SimCC decode, inverse warp) |
| `src/pose/yolox.rs` | `Yolox` (letterbox, grid decode, person NMS) |
| `src/pose/tracker.rs` | `PoseTracker` (single golfer, detect every N frames, IoU identity, One Euro) |
| `src/pose/clip.rs` | `analyze_clip`/`analyze_frames` (parallel key-frame detection and pose, sequential fix-up, low-score poses dropped), `WorkerHook` (`PoseConfig::on_worker_start`, run first on every worker thread) |
| `src/pose/track.rs` | `PoseTrack` JSON (`<clip>.pose.json`) |
| `src/pose/filter.rs` | `OneEuro`, zero-phase `Butterworth` |
| `src/pose/download.rs` | `pose-download` feature: `fetch_model` / `fetch_model_with` (optional `FetchProgress` callback and cancel flag, `Error::Cancelled`) (ureq, HTTPS only, timeouts, per-process temporary files, zip member extraction) |
| `src/pose/draw.rs` | `pose::draw`: blended lines and discs, skeletons, the 5×7 label font (used by the examples) |
| `src/golf/mod.rs` | `pose` feature: golf swing analysis, module docs and re-exports |
| `src/golf/body.rs` | `View`, `Handed`, `Joint` (lead/trail over Halpe-26), `Setup` (image directions), 2D geometry |
| `src/golf/smooth.rs` | `Smoothing`, gap filling and impact-split Butterworth, `smooth_track` |
| `src/golf/events.rs` | `SwingEvents` from the hands (impact refined from a given time or estimated, top at the downswing onset, address, takeaway, finish), `Tempo` |
| `src/golf/metrics.rs` | `MetricId` (definitions table, incl. the one-camera turn estimates), `Metric`, `Positions`, `Series`/`MetricSeries` (every metric per frame), `Unit`, `BodyScale`; one definition per metric serves its positions and its series |
| `src/golf/posture.rs` | Live `posture`, `HeadShift`, `PostureFilter` (One Euro), `unmirror` |
| `src/golf/view.rs` | `guess_view`, the handedness check, `Guess` |
| `src/golf/report.rs` | `analyze_swing`, `SwingOptions` (incl. mirrored images), `SwingReport` JSON (`<clip>.swing.json`, per-frame `series`, unknown metrics skipped, bounded read) |
| `src/golf/tests.rs` | Synthetic swings (a parametric stick figure): events, tempo, metric formulas, per-frame series, turn estimates, mirror symmetry, gaps, JSON; robustness: offset impact, unconfirmed given impact, practice swing, hands high at the start, drifting hold at the top, clip starting at address, gap over the top, mirrored image |
| `examples/frp_record.rs` | FRP WebSocket client writing a clip per shot |
| `examples/pose_bench.rs` | Pose latency, clip-pass time, scores per body part, impact scores, L/R swaps over clips |
| `examples/pose_render.rs` | Draws a clip's pose track onto its frames (AVI + sidecar, `.pose.json`, stills) |
| `examples/golf_report.rs` | Swing report for a clip; `--render` writes `.swing.json` and annotated stills per event |
| `tests/end_to_end.rs` | Test pattern → capture → recorder → files → playback |
| `tests/hostile_files.rs` | Corrupt/truncated AVIs and sidecars; a tracking allocator proves no allocation follows an untrusted size |
| `tests/pose_models.rs` | `pose` with real models (only when `STANCE_MODEL_DIR` is set): rtmlib parity, detector, clip pass, live tracker |
| `tests/golf_models.rs` | `golf` posture on real keypoints (only when `STANCE_MODEL_DIR` is set): mirror symmetry |
| `tests/data/` | `golfer.jpg` (public domain) and rtmlib's reference keypoints for it; provenance in its README |

## Target gating

- All targets: errors, frame/format types, camera types + `list`/`open`,
  `Source`, `Ring`, `ClipRecorder`, `sidecar`, `jpeg`, `frp`.
- Not wasm: `clock`, `TestPattern`, `avi`, storage, `Capture` (files, threads,
  a host clock), `pose` and `golf`.
- Backends: `mf` on Windows, `v4l2` on Linux with 64-bit pointers (the ioctl
  structs mirror the 64-bit kernel layout), `unsupported` elsewhere.

There are no default features. The wasm build is
`cargo build --target wasm32-unknown-unknown --no-default-features`.

## Design rules

- Frames are stored as the camera sent them. No decode or re-encode on the
  capture path; `ensure_huffman` is for consumers that decode.
- The ring is the only buffer between the capture thread and consumers. No
  unbounded channels anywhere.
- `ClipRecorder` stays pure: no threads, no clock reads, deterministic tests.
- All times are host monotonic ns from `clock::now_ns`.
- The sidecar holds capture facts only; application data (camera position,
  shot details) belongs to the caller.
- A `Camera` is not `Send`: Media Foundation objects belong to the thread that
  opened them, so `Capture::start` opens the source on its own thread.
- Stopping never depends on a read returning: sources whose reads can block
  past their timeout (Media Foundation) hand out an `Interrupter`, which
  `Capture::request_stop` and stall detection call from another thread.
- `Capture` re-stamps `Frame::seq` from per-capture blocks of a process-wide
  counter: consecutive within a capture, above every earlier capture, so
  cursors survive reconnects.
- File readers treat input as untrusted: clamp every size to the bytes the
  file holds before allocating, saturate offset arithmetic, validate sidecar
  offsets.
- ML lives only behind `pose`: without the feature, the dependency tree
  (`cargo tree -e normal`) is the capture crate's alone. Model files are
  never committed or packaged; `ModelSpec` pins each by SHA-256 and loading
  verifies it.
- Only `src/pose/runtime.rs` touches the inference runtime, so another
  backend (e.g. `ort` for GPUs) slots in without changing the public API.
  `PoseModels`, `PoseTracker` and `analyze_frames` are generic over
  `PoseEngine`/`PersonDetector`, so another model family (e.g. BlazePose)
  plugs in the same way.
- A pose scoring below `PoseConfig::min_pose_score` is absent (`None` live,
  an empty `PoseFrame`), never stored.
- Decoding is bounded: `MAX_DECODE_DIMENSION` per frame, and the clip pass
  holds at most 256 MiB of decoded frames across its workers.
- Pose numerics follow rtmlib (`tests/pose_models.rs` checks parity against
  `tests/data/golfer.rtmlib.json`). RTMPose input is RGB (its mmdeploy
  pipeline says `to_rgb`); YOLOX input is raw BGR.
- `PoseTrack` stores raw keypoints; smoothing is derived, never stored.
- `golf` is pure math on keypoints: no models, no I/O beyond its JSON.
  Metrics are 2D, per view, and documented as proxies; lead/trail mapping
  keeps left-handers an exact mirror of right-handers.

## Rust conventions

- Rust 2024 edition, toolchain pinned in `rust-toolchain.toml`.
- `clippy::pedantic` and `missing_docs` via `[lints]`; every allow
  carries a reason. `clippy::undocumented_unsafe_blocks` is on: every `unsafe`
  block has a `// SAFETY:` comment.
- Library errors are `Error` (thiserror). No `unwrap`/`expect`/panics outside
  tests. `#[must_use]` on value-returning functions.
- Minimal dependencies; justify each in `Cargo.toml` with a comment.
- Never modify `version` in `Cargo.toml`; release CI bumps it.
- Comments and docs describe current behaviour only — no history.

## Docs policy

Public docs explain how the software works. UVC, Media Foundation and V4L2
are public standards and APIs. No machine names, user names or local paths in
this repository.

## Gate

All must pass, with zero warnings, from the repository root:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo clippy --all-targets --all-features --target x86_64-pc-windows-gnu -- -D warnings
cargo build --examples --all-features --target x86_64-pc-windows-gnu
cargo build --target wasm32-unknown-unknown --no-default-features
```

The Windows cross-build needs mingw-w64. No test needs a camera; tests use
`TestPattern` and temporary files. No test needs the network or a model:
`tests/pose_models.rs` runs its model tests only with
`STANCE_MODEL_DIR=<dir holding <name>.onnx>` (run them with `--release`;
tract is slow in debug builds). Before a release also check
`cargo package --list --allow-dirty` (no model files) and that
`cargo tree -e normal` without features shows no ML crates. CI also runs
clippy, the build and the tests on `windows-latest` and `macos-latest`.

## Releases

Push a `vX.Y.Z` tag. CI runs the gate, publishes `stance` to
crates.io, creates a GitHub Release, then commits the next patch version to
`main`.
