//! A pre-roll buffer of recent frames, bounded by duration and bytes.

use std::collections::VecDeque;
use std::time::Duration;

use crate::Frame;

/// Recent frames, oldest first, bounded by the time span they cover and by
/// the total size of their JPEG bytes.
///
/// Pushing a frame evicts the oldest frames until both bounds hold again; the
/// newest frame is always kept. Frames must be pushed in capture order.
///
/// ```
/// # use std::sync::Arc;
/// # use std::time::Duration;
/// use stance::{Frame, Ring};
///
/// let mut ring = Ring::new(Duration::from_millis(100), 1 << 20);
/// for seq in 0..20 {
///     let host_ns = seq * 10_000_000; // 100 fps
///     ring.push(Frame { jpeg: Arc::from(vec![0u8; 1000]), host_ns, device_ns: None, seq, width: 8, height: 8 });
/// }
/// assert_eq!(ring.len(), 11); // 100 ms of frames, both ends included
/// assert_eq!(ring.oldest().map(|f| f.seq), Some(9));
/// ```
#[derive(Debug, Clone)]
pub struct Ring {
    frames: VecDeque<Frame>,
    bytes: usize,
    max_duration_ns: u64,
    max_bytes: usize,
}

impl Ring {
    /// An empty ring keeping at most `max_duration` of frames and `max_bytes`
    /// of JPEG data.
    #[must_use]
    pub fn new(max_duration: Duration, max_bytes: usize) -> Self {
        Self {
            frames: VecDeque::new(),
            bytes: 0,
            max_duration_ns: duration_ns(max_duration),
            max_bytes,
        }
    }

    /// Changes the bounds; frames beyond them are evicted now.
    pub fn set_limits(&mut self, max_duration: Duration, max_bytes: usize) {
        self.max_duration_ns = duration_ns(max_duration);
        self.max_bytes = max_bytes;
        self.evict();
    }

    /// The duration bound.
    #[must_use]
    pub fn max_duration(&self) -> Duration {
        Duration::from_nanos(self.max_duration_ns)
    }

    /// The byte bound.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Appends a frame and evicts what no longer fits.
    pub fn push(&mut self, frame: Frame) {
        self.bytes += frame.jpeg.len();
        self.frames.push_back(frame);
        self.evict();
    }

    fn evict(&mut self) {
        let Some(newest) = self.frames.back().map(|f| f.host_ns) else {
            return;
        };
        while self.frames.len() > 1 {
            let Some(oldest) = self.frames.front() else {
                break;
            };
            let too_old = newest.saturating_sub(oldest.host_ns) > self.max_duration_ns;
            if !too_old && self.bytes <= self.max_bytes {
                break;
            }
            self.bytes -= oldest.jpeg.len();
            self.frames.pop_front();
        }
    }

    /// Number of frames held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// True if no frames are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Total JPEG bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Time between the oldest and the newest frame.
    #[must_use]
    pub fn span(&self) -> Duration {
        match (self.frames.front(), self.frames.back()) {
            (Some(a), Some(b)) => Duration::from_nanos(b.host_ns.saturating_sub(a.host_ns)),
            _ => Duration::ZERO,
        }
    }

    /// The oldest frame.
    #[must_use]
    pub fn oldest(&self) -> Option<&Frame> {
        self.frames.front()
    }

    /// The newest frame.
    #[must_use]
    pub fn newest(&self) -> Option<&Frame> {
        self.frames.back()
    }

    /// Frames oldest first.
    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Frame> + ExactSizeIterator {
        self.frames.iter()
    }

    /// Frames with `from_ns <= host_ns <= to_ns`, oldest first.
    #[must_use]
    pub fn range(&self, from_ns: u64, to_ns: u64) -> Vec<Frame> {
        let start = self.frames.partition_point(|f| f.host_ns < from_ns);
        self.frames
            .range(start..)
            .take_while(|f| f.host_ns <= to_ns)
            .cloned()
            .collect()
    }

    /// Frames with a sequence number above `seq` (all frames for `None`), oldest first.
    #[must_use]
    pub fn after(&self, seq: Option<u64>) -> Vec<Frame> {
        let start = seq.map_or(0, |s| self.frames.partition_point(|f| f.seq <= s));
        self.frames.range(start..).cloned().collect()
    }

    /// Removes every frame.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.bytes = 0;
    }
}

pub(crate) fn duration_ns(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn frame(seq: u64, ms: u64, len: usize) -> Frame {
        Frame {
            jpeg: Arc::from(vec![0u8; len]),
            host_ns: ms * 1_000_000,
            device_ns: None,
            seq,
            width: 4,
            height: 4,
        }
    }

    #[test]
    fn bounded_by_duration() {
        let mut r = Ring::new(Duration::from_millis(1000), usize::MAX);
        for i in 0..300 {
            r.push(frame(i, i * 10, 10));
        }
        // Newest at 2990 ms; everything from 1990 ms on is kept.
        assert_eq!(r.newest().unwrap().seq, 299);
        assert_eq!(r.oldest().unwrap().host_ns, 1_990_000_000);
        assert_eq!(r.len(), 101);
        assert_eq!(r.span(), Duration::from_secs(1));
        assert_eq!(r.bytes(), 1010);
    }

    #[test]
    fn bounded_by_bytes() {
        let mut r = Ring::new(Duration::from_secs(3600), 1000);
        for i in 0..50 {
            r.push(frame(i, i, 100));
        }
        assert_eq!(r.len(), 10);
        assert_eq!(r.bytes(), 1000);
        assert_eq!(r.oldest().unwrap().seq, 40);
        // A single frame larger than the cap is still kept, alone.
        r.push(frame(50, 50, 5000));
        assert_eq!(r.len(), 1);
        assert_eq!(r.bytes(), 5000);
        r.push(frame(51, 51, 10));
        assert_eq!(r.len(), 1);
        assert_eq!(r.newest().unwrap().seq, 51);
    }

    #[test]
    fn shrinking_limits_evicts() {
        let mut r = Ring::new(Duration::from_secs(10), usize::MAX);
        for i in 0..100 {
            r.push(frame(i, i * 10, 1));
        }
        r.set_limits(Duration::from_millis(100), usize::MAX);
        assert_eq!(r.len(), 11);
        assert_eq!(r.max_duration(), Duration::from_millis(100));
        r.clear();
        assert!(r.is_empty());
        assert_eq!(r.bytes(), 0);
        assert_eq!(r.span(), Duration::ZERO);
    }

    #[test]
    fn range_and_after() {
        let mut r = Ring::new(Duration::from_secs(10), usize::MAX);
        for i in 0..10 {
            r.push(frame(i, i * 10, 1));
        }
        let seqs = |v: Vec<Frame>| v.into_iter().map(|f| f.seq).collect::<Vec<_>>();
        assert_eq!(seqs(r.range(20_000_000, 50_000_000)), vec![2, 3, 4, 5]);
        assert_eq!(seqs(r.range(21_000_000, 49_000_000)), vec![3, 4]);
        assert!(r.range(200_000_000, 300_000_000).is_empty());
        assert_eq!(seqs(r.after(Some(7))), vec![8, 9]);
        assert_eq!(r.after(None).len(), 10);
        assert!(r.after(Some(9)).is_empty());
    }
}
