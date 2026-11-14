use std::sync::Arc;

use super::*;

const MS: u64 = 1_000_000;

/// 100 fps frames: frame `i` at `i * 10` ms.
fn frame(seq: u64) -> Frame {
    frame_at(seq, seq * 10 * MS)
}

fn frame_at(seq: u64, host_ns: u64) -> Frame {
    Frame {
        jpeg: Arc::from(vec![u8::try_from(seq % 256).unwrap(); 8]),
        host_ns,
        device_ns: None,
        seq,
        width: 8,
        height: 8,
    }
}

/// As `frame`, with the device time equal to the arrival time.
fn device_frame(seq: u64) -> Frame {
    Frame {
        device_ns: Some(seq * 10 * MS),
        ..frame(seq)
    }
}

fn config() -> RecorderConfig {
    RecorderConfig {
        pre_roll: Duration::from_millis(200),
        post_roll: Duration::from_millis(100),
        merge_window: Duration::from_millis(500),
        buffer: Duration::from_secs(5),
        max_bytes: usize::MAX,
        stall: Duration::from_millis(300),
        fps: Some(100.0),
    }
}

fn feed(
    rec: &mut ClipRecorder<&'static str>,
    seqs: std::ops::Range<u64>,
) -> Vec<RecorderEvent<&'static str>> {
    seqs.flat_map(|s| rec.push(frame(s))).collect()
}

fn finished(events: Vec<RecorderEvent<&'static str>>) -> Vec<Clip<&'static str>> {
    events
        .into_iter()
        .map(|e| match e {
            RecorderEvent::Finished(c) => c,
            RecorderEvent::Discarded(d) => panic!("unexpected discard {d:?}"),
        })
        .collect()
}

fn seqs(c: &Clip<&str>) -> (u64, u64) {
    (c.frames[0].seq, c.frames[c.frames.len() - 1].seq)
}

#[test]
fn starts_and_finishes_on_frames_past_the_window() {
    let mut rec = ClipRecorder::new(config());
    assert!(feed(&mut rec, 0..100).is_empty());
    // Trigger at 1000 ms: window 800..=1100 ms.
    assert_eq!(
        rec.trigger(Trigger::new(1000 * MS, "a", "s1")),
        TriggerOutcome::Started(ClipId(0))
    );
    assert!(feed(&mut rec, 100..111).is_empty(), "1100 ms is inside");
    assert_eq!(rec.open_clips(), 1);
    let clips = finished(rec.push(frame(111)));
    assert_eq!(clips.len(), 1);
    let c = &clips[0];
    assert_eq!(seqs(c), (80, 110));
    assert_eq!(c.frames.len(), 31);
    assert_eq!(c.frames[c.trigger_index].seq, 100);
    assert_eq!(c.impact_index, c.trigger_index);
    assert!(c.complete);
    assert!(!c.truncated);
    assert_eq!(c.dropped, 0);
    assert_eq!(c.tags, ["s1"]);
    assert_eq!(rec.open_clips(), 0);
    assert!((c.measured_fps().unwrap() - 100.0).abs() < 1e-9);
}

#[test]
fn impact_lead_shifts_the_window() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..200);
    // Trigger arrives at 1990 ms, impact 1.5 s earlier: window 290..=2090 ms.
    let t = Trigger::new(1990 * MS, "a", "s1").with_impact_lead(Duration::from_millis(1500));
    assert_eq!(t.impact_ns(), 490 * MS);
    rec.trigger(t);
    let c = finished(feed(&mut rec, 200..300)).remove(0);
    assert_eq!(seqs(&c), (29, 209));
    assert_eq!(c.frames[c.impact_index].seq, 49);
    assert_eq!(c.frames[c.trigger_index].seq, 199);
    assert_eq!(c.impact_ns, 490 * MS);
}

#[test]
fn per_trigger_and_live_rolls() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(
        Trigger::new(990 * MS, "a", "s1")
            .with_pre_roll(Duration::from_millis(50))
            .with_post_roll(Duration::from_millis(20)),
    );
    let c = finished(feed(&mut rec, 100..110)).remove(0);
    assert_eq!(seqs(&c), (94, 101));
    rec.set_pre_roll(Duration::from_millis(30));
    rec.set_post_roll(Duration::from_millis(30));
    rec.trigger(Trigger::new(1090 * MS, "a", "s2"));
    let c = finished(feed(&mut rec, 110..120)).remove(0);
    assert_eq!(seqs(&c), (106, 112));
}

#[test]
fn different_source_inside_the_merge_window_joins() {
    let mut rec = ClipRecorder::new(config());
    let post = Duration::from_millis(1500);
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "radar", "r1").with_post_roll(post));
    assert!(feed(&mut rec, 100..130).is_empty());
    // The other monitor reports 1.3 s late but sees the same impact (40 ms apart).
    let late = Trigger::new(2250 * MS, "camera", "c1")
        .with_impact_lead(Duration::from_millis(1300))
        .with_post_roll(post);
    assert_eq!(rec.trigger(late), TriggerOutcome::Joined(ClipId(0)));
    // The window now runs from the earlier impact estimate (950 ms) minus
    // the pre-roll to 1.5 s after the late trigger: 750..=3750 ms.
    assert!(feed(&mut rec, 130..376).is_empty());
    let c = finished(rec.push(frame(376))).remove(0);
    assert_eq!(c.tags, ["r1", "c1"]);
    assert_eq!(seqs(&c), (75, 375));
    assert_eq!(c.frames.len(), 301, "every frame once");
    assert_eq!(
        c.frames[c.trigger_index].seq, 99,
        "anchored on the first trigger"
    );
}

#[test]
fn different_source_outside_the_merge_window_starts_a_new_clip() {
    let mut rec = ClipRecorder::new(config());
    let post = Duration::from_millis(1500);
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "radar", "r1").with_post_roll(post));
    feed(&mut rec, 100..130);
    // 1.29 s apart: another swing, seen by the other monitor only.
    let other = Trigger::new(2280 * MS, "camera", "c1").with_post_roll(post);
    assert_eq!(rec.trigger(other), TriggerOutcome::Started(ClipId(1)));
    assert_eq!(rec.open_clips(), 2);
    let clips = finished(feed(&mut rec, 130..400));
    assert_eq!(clips.len(), 2);
    assert_eq!(
        (clips[0].tags.as_slice(), seqs(&clips[0])),
        (&["r1"][..], (79, 249))
    );
    assert_eq!(
        (clips[1].tags.as_slice(), seqs(&clips[1])),
        (&["c1"][..], (208, 378))
    );
}

#[test]
fn overlapping_clips_share_frames() {
    let mut cfg = config();
    cfg.merge_window = Duration::from_millis(100);
    let mut rec = ClipRecorder::new(cfg);
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1")); // 790..=1090 ms
    rec.trigger(Trigger::new(1190 * MS, "b", "b1")); // 990..=1290 ms
    let clips = finished(feed(&mut rec, 100..200));
    assert_eq!(seqs(&clips[0]), (79, 109));
    assert_eq!(seqs(&clips[1]), (99, 129));
    let in_both = |c: &Clip<&str>| c.frames.iter().find(|f| f.seq == 105).unwrap().jpeg.clone();
    assert!(Arc::ptr_eq(&in_both(&clips[0]), &in_both(&clips[1])));
}

#[test]
fn joining_with_an_earlier_impact_extends_the_start() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1"));
    // Impact estimate 300 ms earlier: start moves from 790 to 490 ms.
    let t = Trigger::new(990 * MS, "b", "b1").with_impact_lead(Duration::from_millis(300));
    assert_eq!(rec.trigger(t), TriggerOutcome::Joined(ClipId(0)));
    let c = finished(feed(&mut rec, 100..120)).remove(0);
    assert_eq!(seqs(&c), (49, 109));
    assert_eq!(c.frames.len(), 61);
}

#[test]
fn same_source_truncates_and_starts_a_new_clip() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "putter", "p1"));
    feed(&mut rec, 100..105); // up to 1040 ms
    // The next putt 50 ms later: p1 ends at 1040 ms, p2 starts.
    assert_eq!(
        rec.trigger(Trigger::new(1040 * MS, "putter", "p2")),
        TriggerOutcome::Truncated {
            previous: ClipId(0),
            clip: ClipId(1)
        }
    );
    let clips = finished(feed(&mut rec, 105..200));
    assert_eq!(clips.len(), 2);
    let (p1, p2) = (&clips[0], &clips[1]);
    assert_eq!(p1.tags, ["p1"]);
    assert!(p1.truncated && p1.complete);
    assert_eq!(seqs(p1), (79, 104));
    assert_eq!(p2.tags, ["p2"]);
    assert!(!p2.truncated);
    assert_eq!(seqs(p2), (84, 114), "overlaps p1");
}

#[test]
fn same_source_truncation_drops_frames_already_past_the_cut() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1"));
    feed(&mut rec, 100..108); // frames up to 1070 ms are in the clip
    rec.trigger(Trigger::new(1020 * MS, "a", "a2")); // arrived late
    let clips = finished(feed(&mut rec, 108..200));
    assert_eq!(seqs(&clips[0]), (79, 102));
}

#[test]
fn poll_finishes_incomplete_clips_when_frames_stop() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1"));
    feed(&mut rec, 100..103); // camera dies at 1020 ms; window ends 1090 ms
    assert!(rec.poll(1300 * MS).is_empty(), "stall not reached");
    let clips = finished(rec.poll(1390 * MS));
    let c = &clips[0];
    assert!(!c.complete);
    assert_eq!(seqs(c), (79, 102));
    assert_eq!(c.frames[c.trigger_index].seq, 99);
    assert_eq!(rec.open_clips(), 0);
}

#[test]
fn clips_without_their_impact_frame_are_discarded() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100); // last frame at 990 ms
    rec.trigger(Trigger::new(995 * MS, "a", "a1"));
    let events = rec.flush();
    assert_eq!(
        events,
        vec![RecorderEvent::Discarded(Discarded {
            id: ClipId(0),
            tags: vec!["a1"],
            reason: DiscardReason::NoImpactFrame
        })]
    );
    // No frames at all.
    let mut rec: ClipRecorder<&str> = ClipRecorder::new(config());
    rec.trigger(Trigger::new(MS, "a", "a1"));
    assert_eq!(
        rec.poll(u64::MAX),
        vec![RecorderEvent::Discarded(Discarded {
            id: ClipId(0),
            tags: vec!["a1"],
            reason: DiscardReason::NoFrames
        })]
    );
}

#[test]
fn counts_dropped_frames() {
    let mut rec = ClipRecorder::new(config());
    let mut events = Vec::new();
    for s in (0..120).filter(|s| !(95..98).contains(s) && *s != 104) {
        events.extend(rec.push(device_frame(s)));
        if s == 99 {
            rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        }
    }
    let c = finished(events).remove(0);
    assert_eq!(c.dropped, 4, "3 at 950-970 ms, 1 at 1040 ms");
    assert!(c.complete);
    assert_eq!(c.fps, Some(100.0));
    // Without a nominal rate the clip's own median spacing is used.
    let mut cfg = config();
    cfg.fps = None;
    let mut rec = ClipRecorder::new(cfg);
    for s in (0..105).filter(|s| *s != 102) {
        rec.push(device_frame(s));
    }
    rec.trigger(Trigger::new(990 * MS, "a", "a1"));
    let c = finished(rec.flush()).remove(0);
    assert_eq!(c.dropped, 1);
    assert!(!c.complete, "flushed before the window ended");
    assert_eq!(c.fps, None);
}

#[test]
fn arrival_times_tolerate_delivery_jitter() {
    let mut rec = ClipRecorder::new(config());
    let mut events = Vec::new();
    // No device times. 1040 ms is missing (a two-period gap, which
    // delivery jitter also produces) and 950..=970 ms (four periods).
    for s in (0..120).filter(|s| !(95..98).contains(s) && *s != 104) {
        events.extend(rec.push(frame(s)));
        if s == 99 {
            rec.trigger(Trigger::new(990 * MS, "a", "a1"));
        }
    }
    let c = finished(events).remove(0);
    assert_eq!(c.dropped, 3);
}

#[test]
fn a_trigger_entirely_in_the_past_finishes_on_poll() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..200);
    rec.trigger(Trigger::new(500 * MS, "a", "a1"));
    let c = finished(rec.poll(0)).remove(0);
    assert!(c.complete);
    assert_eq!(seqs(&c), (30, 60));
}

#[test]
fn ignores_out_of_order_and_repeated_frames() {
    let mut rec: ClipRecorder<&str> = ClipRecorder::new(config());
    assert!(rec.push(frame(10)).is_empty());
    rec.push(frame(10));
    rec.push(frame_at(11, 50 * MS));
    assert_eq!(rec.ring().len(), 1);
}

#[test]
fn same_source_after_the_window_ended_starts_a_new_clip() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..101); // frames to 1000 ms
    rec.trigger(Trigger::new(990 * MS, "a", "a1")); // window 790..=1090 ms
    // Frames stall; the same monitor fires again at 1500 ms, after the
    // first window ended.
    assert_eq!(
        rec.trigger(Trigger::new(1500 * MS, "a", "a2")),
        TriggerOutcome::Started(ClipId(1))
    );
    // Frames resume at 1010 ms.
    let clips = finished(feed(&mut rec, 101..200));
    assert_eq!(clips.len(), 2);
    let (a1, a2) = (&clips[0], &clips[1]);
    assert_eq!(a1.tags, ["a1"]);
    assert_eq!(seqs(a1), (79, 109), "ends at its own window");
    assert!(a1.complete && !a1.truncated);
    assert_eq!(a2.tags, ["a2"]);
    assert_eq!(seqs(a2), (130, 160));
}

#[test]
fn same_source_at_the_window_end_does_not_truncate() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1")); // ends 1090 ms
    assert_eq!(
        rec.trigger(Trigger::new(1090 * MS, "a", "a2")),
        TriggerOutcome::Started(ClipId(1))
    );
    // One nanosecond earlier cuts it.
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1"));
    assert!(matches!(
        rec.trigger(Trigger::new(1090 * MS - 1, "a", "a2")),
        TriggerOutcome::Truncated { .. }
    ));
}

#[test]
fn duplicate_triggers_are_the_callers_to_drop() {
    // The same source twice at the same instant: the recorder can't tell
    // a duplicate from a second shot, so the first clip is cut at the
    // trigger and a second clip starts.
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1"));
    assert_eq!(
        rec.trigger(Trigger::new(990 * MS, "a", "a1-again")),
        TriggerOutcome::Truncated {
            previous: ClipId(0),
            clip: ClipId(1)
        }
    );
    let clips = finished(feed(&mut rec, 100..200));
    assert_eq!(clips.len(), 2);
    assert!(clips[0].truncated);
    assert_eq!(seqs(&clips[0]), (79, 99));
    assert_eq!(seqs(&clips[1]), (79, 109));
}

#[test]
fn join_considers_every_open_clip() {
    let mut cfg = config();
    cfg.post_roll = Duration::from_millis(1500);
    let mut rec = ClipRecorder::new(cfg);
    feed(&mut rec, 0..100);
    rec.trigger(Trigger::new(990 * MS, "a", "a1")); // impact 990 ms
    feed(&mut rec, 100..200);
    rec.trigger(Trigger::new(1990 * MS, "b", "b1")); // another swing
    // A third monitor reports the first swing 1.1 s late: impact 990 ms.
    let late = Trigger::new(2090 * MS, "c", "c1").with_impact_lead(Duration::from_millis(1100));
    assert_eq!(rec.trigger(late), TriggerOutcome::Joined(ClipId(0)));
    // The newest matching clip wins when several match.
    let near = Trigger::new(1700 * MS, "d", "d1").with_impact_lead(Duration::ZERO);
    assert_eq!(rec.trigger(near), TriggerOutcome::Joined(ClipId(1)));
    // A clip that already has the source is skipped.
    let again = Trigger::new(1995 * MS, "b", "b2");
    assert!(matches!(
        rec.trigger(again),
        TriggerOutcome::Truncated { .. }
    ));
}

#[test]
fn a_clip_ending_after_its_impact_is_kept_incomplete() {
    // A monitor reporting 2 s after impact; the camera dies 0.5 s after
    // the impact, before the trigger arrives.
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..150); // to 1490 ms
    let t = Trigger::new(3000 * MS, "r10", "s1").with_impact_lead(Duration::from_millis(2000));
    rec.trigger(t);
    let c = finished(rec.poll(5000 * MS)).remove(0);
    assert!(!c.complete);
    assert_eq!(seqs(&c), (80, 149));
    assert_eq!(c.frames[c.impact_index].seq, 100);
    assert_eq!(c.trigger_index, c.frames.len(), "no trigger frame");
    assert!(c.trigger_frame().is_none());
    assert_eq!(c.impact_frame().map(|f| f.seq), Some(100));
}

#[test]
fn a_gap_across_the_trigger_is_incomplete() {
    // Frames to 940 ms, trigger at 990 ms, frames resume at 1200 ms:
    // inside a 1.5 s post-roll, the clip is kept but not complete.
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..95);
    let impact = Trigger::new(990 * MS, "a", "a1")
        .with_impact_lead(Duration::from_millis(100))
        .with_post_roll(Duration::from_millis(1500));
    rec.trigger(impact);
    let c = finished(feed(&mut rec, 120..300)).remove(0);
    assert!(!c.complete, "frames are missing across the trigger");
    assert_eq!(c.frames[c.impact_index].seq, 89);
    assert_eq!(c.frames[c.trigger_index].seq, 120, "first frame after it");
    assert_eq!(c.dropped, 25);

    // The same gap past a short window: the clip has its impact frame
    // (880 ms) but no trigger frame.
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..95);
    rec.trigger(Trigger::new(990 * MS, "a", "a1").with_impact_lead(Duration::from_millis(100)));
    let c = finished(feed(&mut rec, 120..130)).remove(0);
    assert!(!c.complete);
    assert_eq!(c.trigger_index, c.frames.len());

    // Without a lead the impact frame is missing too: discarded.
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..95);
    rec.trigger(Trigger::new(990 * MS, "a", "a1"));
    let events = feed(&mut rec, 120..130);
    assert!(matches!(
        events.as_slice(),
        [RecorderEvent::Discarded(Discarded {
            reason: DiscardReason::NoImpactFrame,
            ..
        })]
    ));
}

#[test]
fn open_clips_wait_for_poll_when_frames_stop() {
    let mut rec = ClipRecorder::new(config());
    feed(&mut rec, 0..100);
    for i in 0..1000u64 {
        rec.trigger(Trigger::new(990 * MS + i * 600 * MS, "a", "x"));
    }
    assert_eq!(
        rec.open_clips(),
        1000,
        "nothing finishes them without frames"
    );
    let events = rec.poll(u64::MAX);
    assert_eq!(events.len(), 1000);
    assert_eq!(rec.open_clips(), 0);
    assert!(matches!(events[0], RecorderEvent::Finished(_)));
    assert!(
        events[1..]
            .iter()
            .all(|e| matches!(e, RecorderEvent::Discarded(_)))
    );
}

#[test]
fn flags_a_buffer_shorter_than_pre_roll_plus_lead() {
    let mut cfg = config();
    cfg.buffer = Duration::from_millis(1000);
    assert!(cfg.buffer_covers(Duration::from_millis(800)));
    assert!(!cfg.buffer_covers(Duration::from_millis(801)));
    let mut rec = ClipRecorder::new(cfg);
    feed(&mut rec, 0..300);
    // 200 ms pre-roll + 900 ms lead > 1 s buffer.
    rec.trigger(Trigger::new(2990 * MS, "a", "a1").with_impact_lead(Duration::from_millis(900)));
    let c = finished(feed(&mut rec, 300..320)).remove(0);
    assert!(c.short_buffer);
    assert_eq!(c.frames[0].seq, 199, "starts at the ring's oldest frame");
    // A lead the buffer covers.
    rec.trigger(Trigger::new(3190 * MS, "a", "a2").with_impact_lead(Duration::from_millis(500)));
    let c = finished(feed(&mut rec, 320..340)).remove(0);
    assert!(!c.short_buffer);
    assert_eq!(c.frames[0].seq, 249);
}
