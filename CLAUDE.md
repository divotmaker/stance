# CLAUDE.md — stance conventions

`stance` is a single crate, the camera library for golf swing replay: USB
camera capture, pre-roll ring, trigger-cut clips, MJPEG AVI + JSON sidecar. No
ML and no hard FRP dependency (`frp` is an optional feature). Pose estimation
is planned in this same crate later behind an opt-in cargo feature, so
consumers that only capture never pull in ML dependencies.

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
| `examples/frp_record.rs` | FRP WebSocket client writing a clip per shot |
| `tests/end_to_end.rs` | Test pattern → capture → recorder → files → playback |
| `tests/hostile_files.rs` | Corrupt/truncated AVIs and sidecars; a tracking allocator proves no allocation follows an untrusted size |

## Target gating

- All targets: errors, frame/format types, camera types + `list`/`open`,
  `Source`, `Ring`, `ClipRecorder`, `sidecar`, `jpeg`, `frp`.
- Not wasm: `clock`, `TestPattern`, `avi`, storage, `Capture` (files, threads,
  a host clock).
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
`TestPattern` and temporary files. CI also runs clippy, the build and the
tests on `windows-latest` and `macos-latest`.

## Releases

Push a `vX.Y.Z` tag. CI runs the gate, publishes `stance` to
crates.io, creates a GitHub Release, then commits the next patch version to
`main`.
