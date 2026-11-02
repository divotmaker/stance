//! Test pattern → capture thread → recorder → AVI + sidecar → playback.

use std::path::{Path, PathBuf};
use std::time::Duration;

use stance::jpeg::ensure_huffman;
use stance::{
    AviFile, Capture, CaptureConfig, CaptureStatus, Clip, ClipRecorder, Error, Frame,
    RecorderConfig, RecorderEvent, Source, TestPattern, TestPatternConfig, Trigger, TriggerOutcome,
    open_clip, write_clip,
};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;

const MS: u64 = 1_000_000;
const ORIGIN: u64 = 10_000 * MS;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("stance-e2e-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The pattern's frame number, from the frame's time (`seq` is re-stamped by
/// the capture).
fn n(f: &Frame) -> u64 {
    (f.host_ns - ORIGIN) / (10 * MS)
}

fn decode(jpeg: &[u8]) -> (usize, usize) {
    let fixed = ensure_huffman(jpeg);
    let mut dec = JpegDecoder::new(ZCursor::new(fixed.as_ref()));
    dec.decode().unwrap();
    dec.dimensions().unwrap()
}

/// Runs 3 s of a 100 fps test pattern (unpaced, frame `n` at ORIGIN + n·10 ms)
/// through a capture thread and a recorder. Each `(at_ms, source, tag)`
/// trigger is applied once the first frame at or after `at_ms` arrives, as a
/// live trigger would be.
fn record(triggers: &[(u64, &'static str, &'static str)]) -> Vec<RecorderEvent<&'static str>> {
    let pattern = TestPatternConfig {
        width: 96,
        height: 64,
        fps: 100,
        paced: false,
        frames: Some(300),
        origin_ns: Some(ORIGIN),
        quality: 70,
    };
    // The ring holds the whole run, so the unpaced source can't outrun us.
    let config = CaptureConfig {
        ring_duration: Duration::from_secs(10),
        ..CaptureConfig::default()
    };
    let capture = Capture::start(config, move || TestPattern::new(pattern)).unwrap();
    let mut recorder = ClipRecorder::new(RecorderConfig {
        pre_roll: Duration::from_millis(500),
        post_roll: Duration::from_millis(500),
        fps: Some(capture.format().fps()),
        ..RecorderConfig::default()
    });
    let mut pending = triggers.to_vec();
    let mut events = Vec::new();
    let mut cursor = None;
    loop {
        let frames = capture.wait_frames_after(cursor, Duration::from_secs(5));
        if frames.is_empty() {
            break;
        }
        for frame in frames {
            if let Some(&(at, source, tag)) = pending.first()
                && frame.host_ns >= ORIGIN + at * MS
            {
                recorder.trigger(Trigger::new(ORIGIN + at * MS, source, tag));
                pending.remove(0);
            }
            if let Some(c) = cursor {
                assert_eq!(c + 1, frame.seq, "no frame skipped");
            }
            cursor = Some(frame.seq);
            events.extend(recorder.push(frame));
        }
    }
    assert!(
        matches!(capture.status(), CaptureStatus::Failed(e) if matches!(*e, Error::Ended)),
        "the source ran out"
    );
    events.extend(recorder.flush());
    events
}

fn clips(events: Vec<RecorderEvent<&'static str>>) -> Vec<Clip<&'static str>> {
    events
        .into_iter()
        .filter_map(|e| match e {
            RecorderEvent::Finished(c) => Some(c),
            RecorderEvent::Discarded(_) => None,
        })
        .collect()
}

#[test]
fn records_writes_and_plays_back_clips() {
    // Shot 1 at 1.0 s; a second monitor sees it 100 ms later (joins); the
    // first monitor's next shot at 2.8 s can't get its full post-roll.
    let events = record(&[(1000, "a", "a1"), (1100, "b", "b1"), (2800, "a", "a2")]);
    let clips = clips(events);
    assert_eq!(clips.len(), 2);

    let shot = &clips[0];
    assert_eq!(shot.tags, ["a1", "b1"]);
    assert!(shot.complete && !shot.truncated);
    assert_eq!(shot.dropped, 0);
    // 500..=1600 ms.
    assert_eq!(n(&shot.frames[0]), 50);
    assert_eq!(n(shot.frames.last().unwrap()), 160);
    assert_eq!(n(&shot.frames[shot.trigger_index]), 100);

    let last = &clips[1];
    assert_eq!(last.tags, ["a2"]);
    assert!(!last.complete, "the source ended inside the post-roll");
    assert_eq!(n(last.frames.last().unwrap()), 299);

    let dir = TempDir::new("clips");
    for (name, clip) in [("shot-1", shot), ("shot-2", last)] {
        let files = write_clip(dir.path(), name, clip).unwrap();
        let (mut avi, sidecar) = open_clip(dir.path(), name).unwrap();
        assert_eq!(sidecar, files.sidecar);
        assert_eq!(sidecar.complete, clip.complete);
        assert_eq!(avi.len(), clip.frames.len());
        let trigger = avi.read_frame(sidecar.trigger_index).unwrap();
        assert_eq!(trigger.as_slice(), &*clip.frames[clip.trigger_index].jpeg);
        assert_eq!(decode(&trigger), (96, 64));

        // Paced-free playback reproduces the frames and their spacing.
        let mut playback = AviFile::open(&files.avi, false).unwrap().with_origin(0);
        for (i, want) in clip.frames.iter().enumerate() {
            let f = playback.next_frame(Duration::ZERO).unwrap().unwrap();
            assert_eq!(f.jpeg, want.jpeg, "frame {i}");
            assert_eq!(f.host_ns, want.host_ns - clip.frames[0].host_ns);
        }
        assert!(matches!(
            playback.next_frame(Duration::ZERO),
            Err(Error::Ended)
        ));
    }
}

#[test]
fn same_source_shots_in_quick_succession() {
    let events = record(&[(1000, "putter", "p1"), (1300, "putter", "p2")]);
    let clips = clips(events);
    assert_eq!(clips.len(), 2);
    assert!(clips[0].truncated);
    assert_eq!(n(clips[0].frames.last().unwrap()), 130);
    assert_eq!(n(&clips[1].frames[0]), 80, "overlaps the first clip");
    assert!(clips[1].complete && !clips[1].truncated);
}

#[test]
fn trigger_outcomes() {
    let mut rec: ClipRecorder<u32> = ClipRecorder::new(RecorderConfig::default());
    assert!(matches!(
        rec.trigger(Trigger::new(ORIGIN, "a", 1)),
        TriggerOutcome::Started(_)
    ));
    assert!(matches!(
        rec.trigger(Trigger::new(ORIGIN, "b", 2)),
        TriggerOutcome::Joined(_)
    ));
    assert!(matches!(
        rec.trigger(Trigger::new(ORIGIN + 100 * MS, "a", 3)),
        TriggerOutcome::Truncated { .. }
    ));
}

#[test]
fn capture_types_cross_threads() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Capture>();
    send_sync::<stance::Frame>();
    send_sync::<Error>();
    send_sync::<Clip<String>>();
    send_sync::<ClipRecorder<String>>();
    send_sync::<TestPattern>();
    send_sync::<AviFile>();
    send_sync::<stance::Interrupter>();
}
