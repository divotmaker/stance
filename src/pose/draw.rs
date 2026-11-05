//! Drawing on [`Image`]s, for previews and reports: blended lines and discs,
//! a pose skeleton, and text in a 5×7 pixel font. Pure Rust, no
//! dependencies.
//!
//! Coordinates are image pixels; anything outside the image is clipped.

use super::Keypoint;
use super::image::Image;
use super::skeleton::{Side, Skeleton};

/// An RGB colour.
pub type Rgb = [u8; 3];

/// The person's left side (cyan).
pub const LEFT: Rgb = [0, 220, 255];
/// The person's right side (orange).
pub const RIGHT: Rgb = [255, 140, 0];
/// The midline (white).
pub const CENTER: Rgb = [255, 255, 255];
/// The background of [`text_panel`] (black, at 60 % opacity).
pub const PANEL: Rgb = [0, 0, 0];

/// The colour of a side of the body.
#[must_use]
pub fn side_color(side: Side) -> Rgb {
    match side {
        Side::Left => LEFT,
        Side::Right => RIGHT,
        Side::Center => CENTER,
    }
}

/// Blends `color` into pixel `(x, y)` with opacity `alpha` (0 to 1).
pub fn blend(img: &mut Image, x: i64, y: i64, color: Rgb, alpha: f32) {
    let (width, height) = (i64::from(img.width()), i64::from(img.height()));
    if x < 0 || y < 0 || x >= width || y >= height {
        return;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "inside the image, so the index fits usize"
    )]
    let at = ((y * width + x) * 3) as usize;
    let Some(pixel) = img.data_mut().get_mut(at..at + 3) else {
        return;
    };
    for (channel, target) in pixel.iter_mut().zip(color) {
        let mixed = f32::from(*channel) * (1.0 - alpha) + f32::from(target) * alpha;
        #[allow(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "clamped to the u8 range"
        )]
        {
            *channel = mixed.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// The pixel range `[lo - r, hi + r]`, clipped to `0..size`.
fn span(lo: f32, hi: f32, r: f32, size: u32) -> std::ops::RangeInclusive<i64> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "image sizes fit f32; clamped to the image before the cast"
    )]
    let clip = |v: f32| v.clamp(-1.0, size as f32) as i64;
    clip((lo - r).floor()).max(0)..=clip((hi + r).ceil()).min(i64::from(size) - 1)
}

/// A filled disc of radius `r` centred on `c`.
pub fn disc(img: &mut Image, c: (f32, f32), r: f32, color: Rgb, alpha: f32) {
    if !(c.0.is_finite() && c.1.is_finite() && r.is_finite()) {
        return;
    }
    for y in span(c.1, c.1, r, img.height()) {
        for x in span(c.0, c.0, r, img.width()) {
            #[allow(clippy::cast_precision_loss, reason = "pixel coordinates")]
            let (dx, dy) = (x as f32 - c.0, y as f32 - c.1);
            if dx * dx + dy * dy <= r * r {
                blend(img, x, y, color, alpha);
            }
        }
    }
}

/// A line from `a` to `b`, `width` pixels wide: every pixel within
/// `width / 2` of the segment.
pub fn line(img: &mut Image, a: (f32, f32), b: (f32, f32), width: f32, color: Rgb, alpha: f32) {
    if ![a.0, a.1, b.0, b.1, width].iter().all(|v| v.is_finite()) {
        return;
    }
    let r = width / 2.0;
    let (vx, vy) = (b.0 - a.0, b.1 - a.1);
    let len2 = (vx * vx + vy * vy).max(1e-6);
    for y in span(a.1.min(b.1), a.1.max(b.1), r, img.height()) {
        for x in span(a.0.min(b.0), a.0.max(b.0), r, img.width()) {
            #[allow(clippy::cast_precision_loss, reason = "pixel coordinates")]
            let (px, py) = (x as f32 - a.0, y as f32 - a.1);
            let t = ((px * vx + py * vy) / len2).clamp(0.0, 1.0);
            let (dx, dy) = (px - t * vx, py - t * vy);
            if dx * dx + dy * dy <= r * r {
                blend(img, x, y, color, alpha);
            }
        }
    }
}

/// How [`skeleton`] draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkeletonStyle {
    /// Pixels per unit: bones are 1.5 units wide, keypoints 2 units in
    /// radius.
    pub scale: f32,
    /// Fade keypoints and bones by score (faint below 0.2, solid from 0.7);
    /// otherwise draw them at 90 % opacity.
    pub fade: bool,
}

impl SkeletonStyle {
    /// A unit of one pixel per 240 image rows (at least one pixel), faded
    /// by score.
    #[must_use]
    pub fn for_image(img: &Image) -> Self {
        #[allow(clippy::cast_precision_loss, reason = "image heights fit f32")]
        let scale = (img.height() as f32 / 240.0).max(1.0);
        Self { scale, fade: true }
    }

    fn opacity(self, score: f32) -> f32 {
        if self.fade {
            ((score - 0.2) / 0.5).clamp(0.12, 1.0)
        } else {
            0.9
        }
    }
}

/// Draws a pose: bones, then keypoints, the person's left side in
/// [`LEFT`], the right in [`RIGHT`], the midline in [`CENTER`]. Keypoints
/// with a score of zero or less (no location) and their bones are skipped;
/// nothing is drawn when `keypoints` doesn't match the skeleton.
pub fn skeleton(img: &mut Image, skeleton: Skeleton, keypoints: &[Keypoint], style: SkeletonStyle) {
    if keypoints.len() != skeleton.len() {
        return;
    }
    let k = keypoints;
    for &(a, b) in skeleton.bones() {
        let (Some(ka), Some(kb)) = (k.get(a), k.get(b)) else {
            continue;
        };
        if ka.score <= 0.0 || kb.score <= 0.0 {
            continue;
        }
        let side = match (skeleton.side(a), skeleton.side(b)) {
            (Side::Center, s) | (s, _) => s,
        };
        line(
            img,
            (ka.x, ka.y),
            (kb.x, kb.y),
            1.5 * style.scale,
            side_color(side),
            style.opacity(ka.score.min(kb.score)),
        );
    }
    for (i, kp) in k.iter().enumerate().filter(|(_, kp)| kp.score > 0.0) {
        disc(
            img,
            (kp.x, kp.y),
            2.0 * style.scale,
            side_color(skeleton.side(i)),
            style.opacity(kp.score),
        );
    }
}

/// Width and height of a character cell of the text font at `scale`
/// pixels per font dot.
#[must_use]
pub fn cell_size(scale: u32) -> (u32, u32) {
    (6 * scale, 9 * scale)
}

/// Draws one line of text with its top left at `(x, y)`, `scale` pixels per
/// font dot. The font has digits, upper-case letters (lower case is drawn
/// as upper case) and `. , - + : / ( ) = _ °`; other characters are blank.
pub fn text(img: &mut Image, (x, y): (i64, i64), text: &str, scale: u32, color: Rgb) {
    let s = i64::from(scale);
    let cell = i64::from(cell_size(scale).0);
    for (col, c) in (0_i64..).zip(text.chars()) {
        let x0 = x + col * cell;
        for (gy, bits) in (0_i64..).zip(glyph(c)) {
            for gx in 0..5_i64 {
                if bits & (0x10 >> gx) == 0 {
                    continue;
                }
                for dy in 0..s {
                    for dx in 0..s {
                        blend(img, x0 + gx * s + dx, y + gy * s + dy, color, 1.0);
                    }
                }
            }
        }
    }
}

/// Lines of white text on a translucent [`PANEL`] at the top left of the
/// image, `scale` pixels per font dot.
pub fn text_panel(img: &mut Image, lines: &[String], scale: u32) {
    let (cell_w, cell_h) = cell_size(scale);
    let (cell_w, cell_h, pad) = (i64::from(cell_w), i64::from(cell_h), 2 * i64::from(scale));
    let columns = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let w = i64::try_from(columns).unwrap_or(0) * cell_w + pad;
    let h = i64::try_from(lines.len()).unwrap_or(0) * cell_h + pad;
    let (w, h) = (
        w.min(i64::from(img.width())),
        h.min(i64::from(img.height())),
    );
    for y in 0..h {
        for x in 0..w {
            blend(img, x, y, PANEL, 0.6);
        }
    }
    for (row, line) in (0_i64..).zip(lines) {
        text(img, (pad, row * cell_h + pad), line, scale, CENTER);
    }
}

/// 5×7 glyphs, one byte per row, bit 4 leftmost.
fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0, 0, 0, 0, 0, 0x0C, 0x0C],
        ',' => [0, 0, 0, 0, 0x0C, 0x04, 0x08],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        '+' => [0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0],
        ':' => [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0],
        '/' => [0, 0x01, 0x02, 0x04, 0x08, 0x10, 0],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '=' => [0, 0, 0x1F, 0, 0x1F, 0, 0],
        '_' => [0, 0, 0, 0, 0, 0, 0x1F],
        '°' => [0x0C, 0x12, 0x12, 0x0C, 0, 0, 0],
        _ => [0; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::Keypoint;

    fn black(w: u32, h: u32) -> Image {
        Image::new(w, h, vec![0; (w * h * 3) as usize]).unwrap()
    }

    fn lit(img: &Image) -> usize {
        img.data()
            .chunks(3)
            .filter(|p| p.iter().any(|&c| c > 0))
            .count()
    }

    #[test]
    fn blend_mixes_and_clips() {
        let mut img = black(4, 4);
        blend(&mut img, 1, 2, [200, 100, 0], 0.5);
        assert_eq!(img.pixel(1, 2), Some([100, 50, 0]));
        // Outside the image: nothing happens, nothing panics.
        blend(&mut img, -1, 0, CENTER, 1.0);
        blend(&mut img, 4, 0, CENTER, 1.0);
        blend(&mut img, 0, i64::MAX, CENTER, 1.0);
        assert_eq!(lit(&img), 1);
    }

    #[test]
    fn line_and_disc_cover_the_expected_pixels() {
        let mut img = black(20, 20);
        line(&mut img, (2.0, 10.0), (17.0, 10.0), 1.0, CENTER, 1.0);
        assert_eq!(lit(&img), 16);
        assert_eq!(img.pixel(2, 10), Some(CENTER));
        assert_eq!(img.pixel(17, 10), Some(CENTER));
        assert_eq!(img.pixel(10, 11), Some([0, 0, 0]));
        let mut img = black(20, 20);
        disc(&mut img, (10.0, 10.0), 2.0, LEFT, 1.0);
        assert_eq!(lit(&img), 13);
        // Huge, off-image and non-finite shapes are clipped or skipped.
        disc(&mut img, (1e30, -1e30), 1e30, LEFT, 1.0);
        line(&mut img, (f32::NAN, 0.0), (5.0, 5.0), 2.0, LEFT, 1.0);
        line(&mut img, (-1e9, -1e9), (-1e9, 1e9), 3.0, LEFT, 1.0);
    }

    #[test]
    fn skeleton_skips_unlocated_keypoints() {
        let n = Skeleton::Halpe26.len();
        let kps: Vec<Keypoint> = (0..n)
            .map(|i| Keypoint {
                #[allow(clippy::cast_precision_loss, reason = "tiny index")]
                x: 10.0 + 3.0 * i as f32,
                y: 50.0,
                score: if i == 0 { 0.0 } else { 0.9 },
            })
            .collect();
        let mut img = black(120, 100);
        let style = SkeletonStyle {
            scale: 1.0,
            fade: false,
        };
        skeleton(&mut img, Skeleton::Halpe26, &kps, style);
        assert!(lit(&img) > 0);
        assert_eq!(img.pixel(10, 48), Some([0, 0, 0]), "keypoint 0 is skipped");
        let mut img = black(120, 100);
        skeleton(&mut img, Skeleton::Halpe26, &kps[1..], style);
        assert_eq!(lit(&img), 0, "wrong keypoint count draws nothing");
        assert!(SkeletonStyle::for_image(&black(10, 720)).scale > 2.9);
    }

    #[test]
    fn text_draws_glyphs_on_a_panel() {
        let mut img = black(64, 20);
        text(&mut img, (0, 0), "1", 1, CENTER);
        // The `1` glyph has 10 dots.
        assert_eq!(lit(&img), 10);
        let mut img = Image::new(64, 32, vec![255; 64 * 32 * 3]).unwrap();
        text_panel(&mut img, &["AB 1.5°".to_owned(), "x".to_owned()], 1);
        // The panel darkens its area; the text stays white.
        assert_eq!(img.pixel(1, 1), Some([102, 102, 102]));
        assert_eq!(img.pixel(63, 31), Some([255, 255, 255]));
        assert_eq!(cell_size(2), (12, 18));
    }
}
