//! A synthetic camera: JPEG frames with a moving marker and a frame counter.

use std::sync::Arc;
use std::time::Duration;

use crate::avi::wait_until;
use crate::{
    CameraInfo, Error, Format, FormatRequest, Frame, Result, Source, choose_format, clock,
};

/// The [`CameraInfo::id`] of test pattern sources.
pub const TEST_PATTERN_ID: &str = "test-pattern";

/// The [`CameraInfo::name`] of test pattern sources.
pub const TEST_PATTERN_NAME: &str = "Test pattern";

/// Fastest frame rate [`TestPattern::config_for`] asks for.
const MAX_FPS: u32 = 1000;

/// Test pattern settings.
#[derive(Debug, Clone, PartialEq)]
pub struct TestPatternConfig {
    /// Frame width (16..=65535).
    pub width: u32,
    /// Frame height (16..=65535).
    pub height: u32,
    /// Frames per second.
    pub fps: u32,
    /// Deliver frames in real time (`true`), or as fast as they are asked for
    /// with timestamps spaced as if they were (`false`, for tests).
    pub paced: bool,
    /// Stop after this many frames ([`Error::Ended`] afterwards).
    pub frames: Option<u64>,
    /// Host time of frame 0 (default: when the source is created).
    pub origin_ns: Option<u64>,
    /// JPEG quality, 1..=100.
    pub quality: u8,
}

impl Default for TestPatternConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            fps: 30,
            paced: true,
            frames: None,
            origin_ns: None,
            quality: 80,
        }
    }
}

/// A [`Source`] of generated MJPEG frames: a white square sweeping across a
/// dark gradient, and the frame number in large digits at the top left.
/// Pure Rust, on every native platform.
///
/// Frame `n` is stamped `origin + n / fps`. Paced patterns wait for that time
/// before returning the frame; one that can't encode at the frame rate skips
/// ahead like a camera dropping frames, so stamps stay on real time.
///
/// ```
/// use std::time::Duration;
/// use stance::{Source, TestPattern, TestPatternConfig};
///
/// let mut pattern = TestPattern::new(TestPatternConfig {
///     width: 160,
///     height: 120,
///     fps: 60,
///     paced: false,
///     ..TestPatternConfig::default()
/// })?;
/// let frame = pattern.next_frame(Duration::ZERO)?.expect("unpaced patterns never wait");
/// assert_eq!((frame.width, frame.height), (160, 120));
/// assert_eq!(&frame.jpeg[..2], &[0xFF, 0xD8]);
/// # Ok::<(), stance::Error>(())
/// ```
#[derive(Debug)]
pub struct TestPattern {
    config: TestPatternConfig,
    info: CameraInfo,
    format: Format,
    origin_ns: u64,
    period_ns: u64,
    background: Vec<u8>,
    rgb: Vec<u8>,
    seq: u64,
    /// Frame slot on the timeline: `origin + slot / fps`. Equals `seq` unless
    /// a paced pattern fell behind real time and skipped slots.
    slot: u64,
}

impl TestPattern {
    /// A pattern with these settings.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for a size outside 16..=65535, a zero frame
    /// rate, or a quality outside 1..=100.
    pub fn new(config: TestPatternConfig) -> Result<Self> {
        let size_ok = |v: u32| (16..=u32::from(u16::MAX)).contains(&v);
        if !size_ok(config.width) || !size_ok(config.height) {
            return Err(Error::InvalidArgument(format!(
                "test pattern size {}x{} (16..=65535)",
                config.width, config.height
            )));
        }
        if config.fps == 0 {
            return Err(Error::InvalidArgument("test pattern frame rate 0".into()));
        }
        if !(1..=100).contains(&config.quality) {
            return Err(Error::InvalidArgument(format!(
                "JPEG quality {} (1..=100)",
                config.quality
            )));
        }
        let format = Format::mjpeg(config.width, config.height, config.fps);
        let info = CameraInfo {
            formats: vec![format.clone()],
            ..Self::camera_info()
        };
        let (w, h) = (config.width as usize, config.height as usize);
        let mut background = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                // A dim diagonal gradient, so motion and compression are visible.
                let v = u8::try_from((x * 48 / w + y * 48 / h) % 256).unwrap_or(0);
                background.extend_from_slice(&[16 + v / 2, 24 + v / 2, 40 + v]);
            }
        }
        Ok(Self {
            origin_ns: config.origin_ns.unwrap_or_else(clock::now_ns),
            period_ns: format.frame_period_ns(),
            rgb: background.clone(),
            background,
            info,
            format,
            config,
            seq: 0,
            slot: 0,
        })
    }

    /// The modes a camera list offers for the test pattern, fastest first.
    /// The pattern renders any size from 16 to 65535 and any whole frame
    /// rate; these are the suggestions.
    #[must_use]
    pub fn modes() -> Vec<Format> {
        vec![Format::mjpeg(1280, 720, 60), Format::mjpeg(640, 360, 30)]
    }

    /// The test pattern as a camera list entry: [`TEST_PATTERN_ID`],
    /// [`TEST_PATTERN_NAME`] and [`TestPattern::modes`].
    ///
    /// ```
    /// use stance::{TEST_PATTERN_ID, TestPattern};
    /// let info = TestPattern::camera_info();
    /// assert_eq!(info.id, TEST_PATTERN_ID);
    /// assert_eq!(info.mjpeg_formats().count(), TestPattern::modes().len());
    /// ```
    #[must_use]
    pub fn camera_info() -> CameraInfo {
        CameraInfo {
            id: TEST_PATTERN_ID.to_owned(),
            name: TEST_PATTERN_NAME.to_owned(),
            usb: None,
            formats: Self::modes(),
            controls: Vec::new(),
            exposure_unit: None,
            probe_error: None,
        }
    }

    /// Settings for a mode request, the rest at their defaults:
    /// [`FormatRequest::Auto`] picks from [`TestPattern::modes`] as it does
    /// for a camera (the highest frame rate, then the largest size); an
    /// explicit mode is taken as asked, its rate rounded to a whole number
    /// from 1 to 1000. [`TestPattern::new`] checks the size.
    ///
    /// ```
    /// use stance::{FormatRequest, TestPattern};
    /// let auto = TestPattern::config_for(&FormatRequest::Auto);
    /// assert_eq!((auto.width, auto.height, auto.fps), (1280, 720, 60));
    /// let small = TestPattern::config_for(&"160x120@29.97".parse()?);
    /// assert_eq!((small.width, small.height, small.fps), (160, 120, 30));
    /// # Ok::<(), stance::Error>(())
    /// ```
    #[must_use]
    pub fn config_for(request: &FormatRequest) -> TestPatternConfig {
        let (width, height, fps) = match *request {
            FormatRequest::Auto => {
                let modes = Self::modes();
                let format = choose_format(TEST_PATTERN_NAME, &modes, request)
                    .unwrap_or_else(|_| Format::mjpeg(640, 360, 30));
                (
                    format.width,
                    format.height,
                    format.fps_num / format.fps_den.max(1),
                )
            }
            // Clamped to 1..=1000 first, so the cast is exact.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            FormatRequest::Mode { width, height, fps } => (
                width,
                height,
                fps.round().clamp(1.0, f64::from(MAX_FPS)) as u32,
            ),
        };
        TestPatternConfig {
            width,
            height,
            fps,
            ..TestPatternConfig::default()
        }
    }

    /// Renders and encodes frame `seq`, with the marker where it is at `slot`.
    fn render(&mut self, seq: u64, slot: u64) -> Result<Vec<u8>> {
        let (w, h) = (self.config.width as usize, self.config.height as usize);
        self.rgb.copy_from_slice(&self.background);
        // The marker sweeps left to right and back once every 2 s.
        let side = (h / 6).max(4);
        let travel = w.saturating_sub(side).max(1);
        let steps = u64::from(self.config.fps) * 2;
        let phase = usize::try_from(slot % steps).unwrap_or(0);
        let half = usize::try_from(steps / 2).unwrap_or(1).max(1);
        let pos = if phase < half {
            phase
        } else {
            2 * half - phase
        };
        let x0 = pos * travel / half;
        let y0 = h / 2 - side / 2;
        fill(&mut self.rgb, w, x0, y0, side, side, [255, 255, 255]);
        // Frame counter.
        let scale = (h / 60).max(1);
        let digits = seq.to_string();
        for (i, d) in digits.bytes().enumerate() {
            let x = scale * 2 + i * scale * 4;
            draw_digit(&mut self.rgb, w, h, (x, scale * 2), scale, d - b'0');
        }
        let mut out = Vec::with_capacity(w * h / 4);
        let (ew, eh) = (
            u16::try_from(w).map_err(|_| Error::Encode("width".into()))?,
            u16::try_from(h).map_err(|_| Error::Encode("height".into()))?,
        );
        jpeg_encoder::Encoder::new(&mut out, self.config.quality)
            .encode(&self.rgb, ew, eh, jpeg_encoder::ColorType::Rgb)
            .map_err(|e| Error::Encode(e.to_string()))?;
        Ok(out)
    }
}

/// 3×5 digit glyphs, one row per entry, bit 2 = left column.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b010, 0b010, 0b010],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

/// Draws one digit with its top left corner at `at`, each glyph cell
/// `scale` pixels square, clipped to the `width`×`height` image.
fn draw_digit(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    at: (usize, usize),
    scale: usize,
    digit: u8,
) {
    let Some(glyph) = DIGITS.get(usize::from(digit)) else {
        return;
    };
    for (row, bits) in glyph.iter().enumerate() {
        for col in 0..3 {
            if bits & (0b100 >> col) != 0 {
                let (px, py) = (at.0 + col * scale, at.1 + row * scale);
                if px + scale <= width && py + scale <= height {
                    fill(rgb, width, px, py, scale, scale, [255, 220, 0]);
                }
            }
        }
    }
}

fn fill(rgb: &mut [u8], w: usize, x0: usize, y0: usize, cw: usize, ch: usize, c: [u8; 3]) {
    for y in y0..y0 + ch {
        let start = (y * w + x0) * 3;
        if let Some(row) = rgb.get_mut(start..start + cw * 3) {
            for px in row.chunks_exact_mut(3) {
                px.copy_from_slice(&c);
            }
        }
    }
}

impl Source for TestPattern {
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>> {
        if self.config.frames.is_some_and(|n| self.seq >= n) {
            return Err(Error::Ended);
        }
        if self.config.paced {
            // Fell more than a frame behind (encoding slower than the frame
            // rate): skip to the current slot, as a camera drops frames.
            let late = clock::now_ns().saturating_sub(self.origin_ns) / self.period_ns.max(1);
            if late > self.slot + 1 {
                self.slot = late;
            }
        }
        let due = self
            .origin_ns
            .saturating_add(self.slot.saturating_mul(self.period_ns));
        if self.config.paced && !wait_until(due, timeout) {
            return Ok(None);
        }
        let jpeg = self.render(self.seq, self.slot)?;
        let frame = Frame {
            jpeg: Arc::from(jpeg),
            host_ns: due,
            device_ns: None,
            seq: self.seq,
            width: self.config.width,
            height: self.config.height,
        };
        self.seq += 1;
        self.slot += 1;
        Ok(Some(frame))
    }

    fn info(&self) -> &CameraInfo {
        &self.info
    }

    fn format(&self) -> &Format {
        &self.format
    }
}

#[cfg(test)]
mod tests {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;

    use super::*;

    fn unpaced(width: u32, height: u32) -> TestPatternConfig {
        TestPatternConfig {
            width,
            height,
            fps: 120,
            paced: false,
            frames: Some(3),
            origin_ns: Some(1_000),
            quality: 75,
        }
    }

    #[test]
    fn the_listed_modes_and_requests_make_valid_patterns() {
        let info = TestPattern::camera_info();
        assert_eq!(info.id, TEST_PATTERN_ID);
        assert_eq!(info.name, TEST_PATTERN_NAME);
        assert_eq!(info.formats, TestPattern::modes());
        for f in TestPattern::modes() {
            let request = FormatRequest::Mode {
                width: f.width,
                height: f.height,
                fps: f.fps(),
            };
            let config = TestPattern::config_for(&request);
            let pattern = TestPattern::new(config).unwrap();
            assert_eq!(pattern.format(), &f);
            assert_eq!(pattern.info().formats, std::slice::from_ref(&f));
        }
        let auto = TestPattern::config_for(&FormatRequest::Auto);
        let best = choose_format("t", &TestPattern::modes(), &FormatRequest::Auto).unwrap();
        assert_eq!(
            (auto.width, auto.height, auto.fps),
            (best.width, best.height, 60)
        );
        let slow = FormatRequest::Mode {
            width: 64,
            height: 48,
            fps: 0.2,
        };
        assert_eq!(TestPattern::config_for(&slow).fps, 1);
        assert_eq!(TestPattern::config_for(&slow).quality, 80);
    }

    #[test]
    fn frames_decode_at_the_configured_size() {
        let mut p = TestPattern::new(unpaced(320, 180)).unwrap();
        assert_eq!(p.format(), &Format::mjpeg(320, 180, 120));
        assert_eq!(p.info().id, TEST_PATTERN_ID);
        let mut previous: Option<Vec<u8>> = None;
        for seq in 0..3 {
            let f = p.next_frame(Duration::ZERO).unwrap().unwrap();
            assert_eq!(f.seq, seq);
            assert_eq!(f.host_ns, 1_000 + seq * 8_333_333);
            let mut dec = JpegDecoder::new(ZCursor::new(&*f.jpeg));
            let pixels = dec.decode().unwrap();
            assert_eq!(dec.dimensions(), Some((320, 180)));
            assert_eq!(pixels.len(), 320 * 180 * 3);
            assert_ne!(previous.as_ref(), Some(&pixels), "the marker moves");
            previous = Some(pixels);
        }
        assert!(matches!(p.next_frame(Duration::ZERO), Err(Error::Ended)));
    }

    #[test]
    fn odd_sizes_encode() {
        let mut p = TestPattern::new(unpaced(17, 31)).unwrap();
        let f = p.next_frame(Duration::ZERO).unwrap().unwrap();
        let mut dec = JpegDecoder::new(ZCursor::new(&*f.jpeg));
        dec.decode().unwrap();
        assert_eq!(dec.dimensions(), Some((17, 31)));
    }

    #[test]
    fn rejects_bad_settings() {
        assert!(TestPattern::new(unpaced(8, 100)).is_err());
        assert!(TestPattern::new(unpaced(100, 70_000)).is_err());
        assert!(
            TestPattern::new(TestPatternConfig {
                fps: 0,
                ..unpaced(64, 64)
            })
            .is_err()
        );
        assert!(
            TestPattern::new(TestPatternConfig {
                quality: 0,
                ..unpaced(64, 64)
            })
            .is_err()
        );
    }

    #[test]
    fn paced_patterns_honour_the_timeout() {
        let cfg = TestPatternConfig {
            paced: true,
            fps: 1,
            origin_ns: Some(clock::now_ns()),
            ..unpaced(32, 32)
        };
        let mut p = TestPattern::new(cfg).unwrap();
        assert!(
            p.next_frame(Duration::from_millis(10)).unwrap().is_some(),
            "frame 0 is due"
        );
        assert!(
            p.next_frame(Duration::from_millis(10)).unwrap().is_none(),
            "frame 1 is 1 s away"
        );
    }

    #[test]
    fn paced_patterns_that_fall_behind_skip_ahead() {
        let now = clock::now_ns();
        let cfg = TestPatternConfig {
            paced: true,
            fps: 100,
            origin_ns: Some(now - 1_000_000_000),
            frames: None,
            ..unpaced(32, 32)
        };
        let mut p = TestPattern::new(cfg).unwrap();
        let f = p.next_frame(Duration::from_millis(100)).unwrap().unwrap();
        assert_eq!(f.seq, 0);
        assert!(
            f.host_ns >= now - 20_000_000,
            "stamped near now, not at the origin"
        );
        let g = p.next_frame(Duration::from_millis(100)).unwrap().unwrap();
        assert_eq!(g.seq, 1);
        assert!(g.host_ns > f.host_ns);
    }
}
