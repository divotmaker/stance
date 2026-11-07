//! RGB images, JPEG decoding and the resampling the models need.

use std::sync::{Condvar, Mutex, PoisonError};

use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use super::geometry::Affine;
use crate::{Error, Result};

/// Largest width or height [`decode_jpeg`] accepts, which bounds the memory
/// a hostile frame can make it allocate (48 MiB of RGB at most; 4K UHD
/// frames are 24 MiB).
pub const MAX_DECODE_DIMENSION: u32 = 4096;

/// An RGB image, 8 bits per channel, rows top to bottom with no padding.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl Image {
    /// Wraps RGB bytes (`width × height × 3`, row-major).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if a dimension is zero or the length
    /// doesn't match.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3));
        if width == 0 || height == 0 || expected != Some(data.len()) {
            return Err(Error::InvalidArgument(format!(
                "{width}x{height} RGB image needs {expected:?} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The RGB bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The RGB bytes, mutable (for drawing).
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consumes the image, returning its RGB bytes.
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// The pixel at `(x, y)`, or `None` outside the image.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 3;
        self.data.get(i..i + 3).map(|p| [p[0], p[1], p[2]])
    }

    /// Channel `c` of the pixel at integer `(x, y)`, `None` outside the
    /// image.
    #[inline]
    fn at(&self, x: i64, y: i64, c: usize) -> Option<f32> {
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return None;
        }
        // Both coordinates are inside the image, so they fit usize.
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let i = (y as usize * self.width as usize + x as usize) * 3 + c;
        self.data.get(i).map(|&v| f32::from(v))
    }

    /// The image turned by `rotation` (a copy; [`Rotation::None`] returns
    /// an equal image). Quarter turns swap the width and the height.
    #[must_use]
    pub fn rotated(&self, rotation: Rotation) -> Image {
        if rotation == Rotation::None {
            return self.clone();
        }
        let (width, height) = (self.width, self.height);
        let (out_w, out_h) = rotation.dimensions(width, height);
        let mut data = vec![0; self.data.len()];
        for (i, src) in self.data.chunks_exact(3).enumerate() {
            // `i` indexes a width × height image, so both coordinates fit u32.
            #[allow(clippy::cast_possible_truncation, reason = "inside the image")]
            let (col, row) = ((i % width as usize) as u32, (i / width as usize) as u32);
            let (to_col, to_row) = rotation.pixel(col, row, width, height);
            let at = (to_row as usize * out_w as usize + to_col as usize) * 3;
            data[at..at + 3].copy_from_slice(src);
        }
        Image {
            width: out_w,
            height: out_h,
            data,
        }
    }

    /// Turns the image by a half turn in place.
    fn rotate_half(&mut self) {
        let pixels = self.data.len() / 3;
        for i in 0..pixels / 2 {
            let j = pixels - 1 - i;
            for c in 0..3 {
                self.data.swap(i * 3 + c, j * 3 + c);
            }
        }
    }
}

/// Quarter turns applied to every decoded frame before detection and pose
/// ([`PoseConfig::rotation`](super::PoseConfig::rotation)), so a camera
/// mounted on its side gives an upright picture. Clips stay as the camera
/// sent them; keypoints, [`PoseTrack::width`](super::PoseTrack::width) and
/// [`PoseTrack::height`](super::PoseTrack::height) are in the turned frame.
///
/// In JSON it is the clockwise angle in degrees: `0`, `90`, `180` or `270`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Rotation {
    /// As the camera sent it.
    #[default]
    None,
    /// A quarter turn clockwise: the frame's left edge becomes its top.
    Cw90,
    /// A half turn.
    Half,
    /// A quarter turn anticlockwise: the frame's right edge becomes its
    /// top.
    Ccw90,
}

impl Rotation {
    /// The rotation of `degrees` clockwise: `0`, `90`, `180` or `270`
    /// (anything else is `None`).
    #[must_use]
    pub fn from_degrees(degrees: u16) -> Option<Self> {
        match degrees {
            0 => Some(Self::None),
            90 => Some(Self::Cw90),
            180 => Some(Self::Half),
            270 => Some(Self::Ccw90),
            _ => None,
        }
    }

    /// The clockwise angle in degrees.
    #[must_use]
    pub fn degrees(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Cw90 => 90,
            Self::Half => 180,
            Self::Ccw90 => 270,
        }
    }

    /// Whether this is [`Rotation::None`].
    #[must_use]
    pub fn is_none(&self) -> bool {
        *self == Self::None
    }

    /// The turned frame's width and height for a `width × height` frame.
    #[must_use]
    pub fn dimensions(self, width: u32, height: u32) -> (u32, u32) {
        match self {
            Self::None | Self::Half => (width, height),
            Self::Cw90 | Self::Ccw90 => (height, width),
        }
    }

    /// Where pixel `(x, y)` of a `width × height` frame lands in the turned
    /// frame.
    fn pixel(self, x: u32, y: u32, width: u32, height: u32) -> (u32, u32) {
        match self {
            Self::None => (x, y),
            Self::Cw90 => (height - 1 - y, x),
            Self::Half => (width - 1 - x, height - 1 - y),
            Self::Ccw90 => (y, width - 1 - x),
        }
    }

    /// Where the point `(x, y)` of a `width × height` frame lands in the
    /// turned frame (pixel centres at integer coordinates, as keypoints
    /// are).
    #[cfg(test)]
    pub(crate) fn point(self, x: f64, y: f64, width: u32, height: u32) -> (f64, f64) {
        let (w, h) = (f64::from(width) - 1.0, f64::from(height) - 1.0);
        match self {
            Self::None => (x, y),
            Self::Cw90 => (h - y, x),
            Self::Half => (w - x, h - y),
            Self::Ccw90 => (y, w - x),
        }
    }
}

impl serde::Serialize for Rotation {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_u16(self.degrees())
    }
}

impl<'de> serde::Deserialize<'de> for Rotation {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let degrees = u16::deserialize(d)?;
        Self::from_degrees(degrees).ok_or_else(|| {
            serde::de::Error::custom(format!("rotation {degrees}: expected 0, 90, 180 or 270"))
        })
    }
}

/// Decodes a JPEG (an MJPEG frame as a camera sent it, with or without
/// Huffman tables) to RGB.
///
/// # Errors
///
/// [`Error::Decode`] for data that isn't a decodable JPEG or is larger than
/// [`MAX_DECODE_DIMENSION`] in either direction.
pub fn decode_jpeg(jpeg: &[u8]) -> Result<Image> {
    decode_reserving(jpeg, |_| ()).map(|(image, ())| image)
}

/// Decodes a JPEG as [`decode_jpeg`], calling `reserve` with the decoded
/// size in bytes after reading the headers and before allocating the image.
fn decode_reserving<G>(jpeg: &[u8], reserve: impl FnOnce(usize) -> G) -> Result<(Image, G)> {
    let jpeg = crate::jpeg::ensure_huffman(jpeg);
    let max = MAX_DECODE_DIMENSION as usize;
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(max)
        .set_max_height(max);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(&*jpeg), options);
    decoder
        .decode_headers()
        .map_err(|e| Error::Decode(e.to_string()))?;
    let (Some((w, h)), Some(size)) = (decoder.dimensions(), decoder.output_buffer_size()) else {
        return Err(Error::Decode("no frame header".into()));
    };
    let grant = reserve(size);
    let mut data = vec![0; size];
    decoder
        .decode_into(&mut data)
        .map_err(|e| Error::Decode(e.to_string()))?;
    let (w, h) = (
        u32::try_from(w).map_err(|_| Error::Decode("width".into()))?,
        u32::try_from(h).map_err(|_| Error::Decode("height".into()))?,
    );
    let image = Image::new(w, h, data).map_err(|e| Error::Decode(e.to_string()))?;
    Ok((image, grant))
}

/// A cap on the bytes of decoded images alive at once across threads: a
/// decode waits until its image fits, and the bytes return when the image
/// is dropped.
#[derive(Debug)]
pub(crate) struct DecodeBudget {
    total: usize,
    free: Mutex<usize>,
    freed: Condvar,
}

impl DecodeBudget {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            total,
            free: Mutex::new(total),
            freed: Condvar::new(),
        }
    }

    /// Decodes `jpeg` (as [`decode_jpeg`]) and turns it by `rotation` once
    /// its image fits the budget. A quarter turn copies the image, so it
    /// reserves room for two.
    pub(crate) fn decode(&self, jpeg: &[u8], rotation: Rotation) -> Result<Budgeted<'_>> {
        let copies = match rotation {
            Rotation::Cw90 | Rotation::Ccw90 => 2,
            Rotation::None | Rotation::Half => 1,
        };
        let (mut image, grant) =
            decode_reserving(jpeg, |bytes| self.take(bytes.saturating_mul(copies)))?;
        match rotation {
            Rotation::None => {}
            Rotation::Half => image.rotate_half(),
            Rotation::Cw90 | Rotation::Ccw90 => image = image.rotated(rotation),
        }
        Ok(Budgeted {
            image,
            _grant: grant,
        })
    }

    /// Waits for `bytes` (at most the whole budget, so one image larger than
    /// it still decodes, alone).
    fn take(&self, bytes: usize) -> Grant<'_> {
        let bytes = bytes.min(self.total);
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        while *free < bytes {
            free = self
                .freed
                .wait(free)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *free -= bytes;
        Grant {
            budget: self,
            bytes,
        }
    }
}

/// Bytes taken from a [`DecodeBudget`], returned on drop.
#[derive(Debug)]
struct Grant<'a> {
    budget: &'a DecodeBudget,
    bytes: usize,
}

impl Drop for Grant<'_> {
    fn drop(&mut self) {
        let mut free = self
            .budget
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *free += self.bytes;
        self.budget.freed.notify_all();
    }
}

/// An image decoded within a [`DecodeBudget`].
#[derive(Debug)]
pub(crate) struct Budgeted<'a> {
    image: Image,
    _grant: Grant<'a>,
}

impl std::ops::Deref for Budgeted<'_> {
    type Target = Image;

    fn deref(&self) -> &Image {
        &self.image
    }
}

/// Channel order of a model input tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelOrder {
    Rgb,
    Bgr,
}

impl ChannelOrder {
    /// The source (RGB) channel feeding tensor plane `plane`.
    fn source(self, plane: usize) -> usize {
        match self {
            Self::Rgb => plane,
            Self::Bgr => 2 - plane,
        }
    }
}

/// Per-channel normalisation `(v − mean) / std`, in tensor plane order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Normalize {
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
}

impl Normalize {
    /// Raw 0–255 values.
    #[cfg(test)]
    pub(crate) const NONE: Self = Self {
        mean: [0.0; 3],
        std: [1.0; 3],
    };
}

/// Warps the image through `to_crop` (image → crop pixels) into an NCHW
/// float tensor of `width × height`, bilinear, with zero outside the image,
/// then normalises.
///
/// Matches OpenCV's `warpAffine(INTER_LINEAR, BORDER_CONSTANT 0)`: crop pixel
/// `(u, v)` samples the image at `to_crop⁻¹(u, v)`, pixel centres at integer
/// coordinates.
pub(crate) fn warp_to_tensor(
    image: &Image,
    to_crop: &Affine,
    width: usize,
    height: usize,
    order: ChannelOrder,
    norm: Normalize,
) -> Vec<f32> {
    let inv = to_crop.inverse().unwrap_or(Affine::IDENTITY);
    let plane = width * height;
    let mut out = vec![0.0f32; plane * 3];
    for v in 0..height {
        for u in 0..width {
            // Crop sizes are a few hundred pixels: exact in f64.
            #[allow(clippy::cast_precision_loss)]
            let (sx, sy) = inv.apply(u as f64, v as f64);
            let px = bilinear(image, sx, sy);
            for p in 0..3 {
                let c = order.source(p);
                out[p * plane + v * width + u] = (px[c] - norm.mean[p]) / norm.std[p];
            }
        }
    }
    out
}

/// Bilinear sample with zero outside the image.
fn bilinear(image: &Image, x: f64, y: f64) -> [f32; 3] {
    if !x.is_finite() || !y.is_finite() {
        return [0.0; 3];
    }
    let (x0, y0) = (x.floor(), y.floor());
    // Coordinates were checked finite; far-away values saturate and read as
    // outside the image.
    #[allow(clippy::cast_possible_truncation)]
    let (ix, iy) = (x0 as i64, y0 as i64);
    #[allow(clippy::cast_possible_truncation)]
    let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
    let mut px = [0.0f32; 3];
    for (c, out) in px.iter_mut().enumerate() {
        let p00 = image.at(ix, iy, c).unwrap_or(0.0);
        let p10 = image.at(ix.saturating_add(1), iy, c).unwrap_or(0.0);
        let p01 = image.at(ix, iy.saturating_add(1), c).unwrap_or(0.0);
        let p11 = image
            .at(ix.saturating_add(1), iy.saturating_add(1), c)
            .unwrap_or(0.0);
        let top = p00 + (p10 - p00) * fx;
        let bottom = p01 + (p11 - p01) * fx;
        *out = top + (bottom - top) * fy;
    }
    px
}

/// Resizes the image by `scale` (bilinear, half-pixel centres, edges
/// clamped, as OpenCV's `resize(INTER_LINEAR)`), places it at the top left
/// of a `size × size` canvas filled with `pad`, and returns it as an NCHW
/// float tensor (raw 0–255 values) with the scale used.
///
/// The resized size is `floor(width × scale) × floor(height × scale)`,
/// where `scale = min(size / width, size / height)`.
pub(crate) fn letterbox_to_tensor(
    image: &Image,
    size: usize,
    pad: f32,
    order: ChannelOrder,
) -> (Vec<f32>, f64) {
    let (src_w, src_h) = (f64::from(image.width), f64::from(image.height));
    // Model input sizes are small integers: exact in f64.
    #[allow(clippy::cast_precision_loss)]
    let square = size as f64;
    let scale = (square / src_w).min(square / src_h);
    // Bounded by `size`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (rw, rh) = (
        ((src_w * scale) as usize).clamp(1, size),
        ((src_h * scale) as usize).clamp(1, size),
    );
    let plane = size * size;
    let mut out = vec![pad; plane * 3];
    // OpenCV derives the per-axis factor from the integer sizes.
    #[allow(clippy::cast_precision_loss)]
    let (fx, fy) = (src_w / rw as f64, src_h / rh as f64);
    let max_x = i64::from(image.width) - 1;
    let max_y = i64::from(image.height) - 1;
    for v in 0..rh {
        #[allow(clippy::cast_precision_loss)]
        let sy = ((v as f64 + 0.5) * fy - 0.5).max(0.0);
        // `sy` is finite, non-negative and below the image height.
        #[allow(clippy::cast_possible_truncation)]
        let y0 = (sy.floor() as i64).min(max_y);
        let y1 = (y0 + 1).min(max_y);
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let ty = (sy - y0 as f64).clamp(0.0, 1.0) as f32;
        for u in 0..rw {
            #[allow(clippy::cast_precision_loss)]
            let sx = ((u as f64 + 0.5) * fx - 0.5).max(0.0);
            #[allow(clippy::cast_possible_truncation)]
            let x0 = (sx.floor() as i64).min(max_x);
            let x1 = (x0 + 1).min(max_x);
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let tx = (sx - x0 as f64).clamp(0.0, 1.0) as f32;
            for p in 0..3 {
                let c = order.source(p);
                let g = |x, y| image.at(x, y, c).unwrap_or(0.0);
                let top = g(x0, y0) + (g(x1, y0) - g(x0, y0)) * tx;
                let bottom = g(x0, y1) + (g(x1, y1) - g(x0, y1)) * tx;
                out[p * plane + v * size + u] = top + (bottom - top) * ty;
            }
        }
    }
    (out, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> Image {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[
                    u8::try_from(x * 255 / (w - 1)).unwrap(),
                    u8::try_from(y * 255 / (h - 1)).unwrap(),
                    77,
                ]);
            }
        }
        Image::new(w, h, data).unwrap()
    }

    #[test]
    fn image_checks_its_length() {
        assert!(Image::new(2, 2, vec![0; 12]).is_ok());
        assert!(Image::new(2, 2, vec![0; 11]).is_err());
        assert!(Image::new(0, 2, Vec::new()).is_err());
        assert!(Image::new(u32::MAX, u32::MAX, Vec::new()).is_err());
    }

    #[test]
    fn identity_warp_copies_pixels_and_zero_fills_outside() {
        let img = gradient(8, 6);
        // Shift by (+2, +1): crop pixel (u, v) shows image pixel (u-2, v-1).
        let to_crop = Affine::new([1.0, 0.0, 2.0, 0.0, 1.0, 1.0]);
        let t = warp_to_tensor(&img, &to_crop, 8, 6, ChannelOrder::Rgb, Normalize::NONE);
        let plane = 48;
        for v in 0..6usize {
            for u in 0..8usize {
                let expect = if u >= 2 && v >= 1 {
                    let p = img
                        .pixel(u32::try_from(u - 2).unwrap(), u32::try_from(v - 1).unwrap())
                        .unwrap();
                    [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])]
                } else {
                    [0.0; 3]
                };
                for c in 0..3 {
                    assert!(
                        (t[c * plane + v * 8 + u] - expect[c]).abs() < 1e-4,
                        "{u},{v},{c}"
                    );
                }
            }
        }
    }

    #[test]
    fn bgr_order_swaps_planes_and_normalises() {
        let img = Image::new(1, 1, vec![10, 20, 30]).unwrap();
        let norm = Normalize {
            mean: [1.0, 2.0, 3.0],
            std: [2.0, 2.0, 2.0],
        };
        let t = warp_to_tensor(&img, &Affine::IDENTITY, 1, 1, ChannelOrder::Bgr, norm);
        assert_eq!(
            t,
            vec![(30.0 - 1.0) / 2.0, (20.0 - 2.0) / 2.0, (10.0 - 3.0) / 2.0]
        );
    }

    #[test]
    fn letterbox_keeps_aspect_and_pads() {
        let img = gradient(64, 32);
        let (t, scale) = letterbox_to_tensor(&img, 32, 114.0, ChannelOrder::Rgb);
        assert!((scale - 0.5).abs() < 1e-12);
        // Top half holds the image, bottom half the padding.
        assert!((t[0] - 0.0).abs() < 4.0, "{}", t[0]);
        assert!((t[31] - 255.0).abs() < 8.0, "{}", t[31]);
        assert!(
            t[16 * 32..32 * 32]
                .iter()
                .all(|&v| (v - 114.0).abs() < 1e-6)
        );
    }

    fn jpeg(w: u16, h: u16) -> Vec<u8> {
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 80)
            .encode(
                &vec![128u8; usize::from(w) * usize::from(h) * 3],
                w,
                h,
                jpeg_encoder::ColorType::Rgb,
            )
            .unwrap();
        out
    }

    #[test]
    fn decode_budget_bounds_images_alive_at_once() {
        let frame = jpeg(16, 16);
        let one = 16 * 16 * 3;
        let budget = DecodeBudget::new(one + 10);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..20 {
                        let img = budget.decode(&frame, Rotation::None).unwrap();
                        assert_eq!(img.width(), 16);
                        assert!(*budget.free.lock().unwrap() <= 10);
                    }
                });
            }
        });
        assert_eq!(*budget.free.lock().unwrap(), one + 10);
        // Truncated frames (decoded or not) return their bytes; an image
        // larger than the whole budget still decodes, alone.
        let _ = budget.decode(&frame[..frame.len() / 2], Rotation::None);
        assert_eq!(*budget.free.lock().unwrap(), one + 10);
        let small = DecodeBudget::new(8);
        assert_eq!(small.decode(&frame, Rotation::None).unwrap().height(), 16);
        assert_eq!(*small.free.lock().unwrap(), 8);
    }

    /// A 4 × 3 image whose pixel `(x, y)` is `[x, y, 10x + y]`.
    fn numbered() -> Image {
        let mut data = Vec::new();
        for y in 0..3u8 {
            for x in 0..4u8 {
                data.extend_from_slice(&[x, y, 10 * x + y]);
            }
        }
        Image::new(4, 3, data).unwrap()
    }

    /// Where a source pixel lands.
    type Landing = fn(u32, u32) -> (u32, u32);

    #[test]
    fn rotation_moves_every_pixel() {
        let img = numbered();
        // Where the source pixel (x, y) lands, written out per turn.
        let cases: [(Rotation, (u32, u32), Landing); 4] = [
            (Rotation::None, (4, 3), |x, y| (x, y)),
            (Rotation::Cw90, (3, 4), |x, y| (2 - y, x)),
            (Rotation::Half, (4, 3), |x, y| (3 - x, 2 - y)),
            (Rotation::Ccw90, (3, 4), |x, y| (y, 3 - x)),
        ];
        for (rotation, dims, to) in cases {
            let out = img.rotated(rotation);
            assert_eq!((out.width(), out.height()), dims, "{rotation:?}");
            assert_eq!(rotation.dimensions(4, 3), dims);
            for y in 0..3 {
                for x in 0..4 {
                    let (u, v) = to(x, y);
                    assert_eq!(out.pixel(u, v), img.pixel(x, y), "{rotation:?} ({x}, {y})");
                    let p = rotation.point(f64::from(x), f64::from(y), 4, 3);
                    assert_eq!(p, (f64::from(u), f64::from(v)));
                }
            }
        }
        // The top-left pixel goes to the top-right corner on a clockwise
        // quarter turn.
        assert_eq!(img.rotated(Rotation::Cw90).pixel(2, 0), Some([0, 0, 0]));
        // Four quarter turns, or two half turns, are the identity.
        let mut round = img.clone();
        for _ in 0..4 {
            round = round.rotated(Rotation::Cw90);
        }
        assert_eq!(round, img);
        assert_eq!(img.rotated(Rotation::Cw90).rotated(Rotation::Ccw90), img);
        let mut half = img.clone();
        half.rotate_half();
        assert_eq!(half, img.rotated(Rotation::Half));
    }

    #[test]
    fn rotation_degrees_and_json() {
        for (r, d) in [
            (Rotation::None, 0),
            (Rotation::Cw90, 90),
            (Rotation::Half, 180),
            (Rotation::Ccw90, 270),
        ] {
            assert_eq!(r.degrees(), d);
            assert_eq!(Rotation::from_degrees(d), Some(r));
            assert_eq!(serde_json::to_string(&r).unwrap(), d.to_string());
            assert_eq!(serde_json::from_str::<Rotation>(&d.to_string()).unwrap(), r);
        }
        assert_eq!(Rotation::default(), Rotation::None);
        assert_eq!(Rotation::from_degrees(45), None);
        assert!(serde_json::from_str::<Rotation>("45").is_err());
        assert!(serde_json::from_str::<Rotation>("\"cw90\"").is_err());
    }

    #[test]
    fn a_quarter_turn_decodes_within_the_budget() {
        let frame = jpeg(16, 8);
        let one = 16 * 8 * 3;
        let budget = DecodeBudget::new(4 * one);
        let img = budget.decode(&frame, Rotation::Cw90).unwrap();
        assert_eq!((img.width(), img.height()), (8, 16));
        // Room for the decoded frame and its turned copy.
        assert_eq!(*budget.free.lock().unwrap(), 2 * one);
        let half = budget.decode(&frame, Rotation::Half).unwrap();
        assert_eq!((half.width(), half.height()), (16, 8));
        assert_eq!(*budget.free.lock().unwrap(), one);
        drop((img, half));
        assert_eq!(*budget.free.lock().unwrap(), 4 * one);
    }

    #[test]
    fn oversized_frames_are_refused() {
        let wide = u16::try_from(MAX_DECODE_DIMENSION + 8).unwrap();
        assert!(matches!(decode_jpeg(&jpeg(wide, 8)), Err(Error::Decode(_))));
        let edge = u16::try_from(MAX_DECODE_DIMENSION).unwrap();
        assert!(decode_jpeg(&jpeg(edge, 8)).is_ok());
    }

    #[test]
    fn corrupt_jpeg_is_an_error() {
        assert!(matches!(decode_jpeg(&[]), Err(Error::Decode(_))));
        assert!(matches!(
            decode_jpeg(&[0xFF, 0xD8, 0xFF]),
            Err(Error::Decode(_))
        ));
        let mut out = Vec::new();
        jpeg_encoder::Encoder::new(&mut out, 80)
            .encode(&[128u8; 16 * 16 * 3], 16, 16, jpeg_encoder::ColorType::Rgb)
            .unwrap();
        let img = decode_jpeg(&out).unwrap();
        assert_eq!((img.width(), img.height()), (16, 16));
        // Truncations and bit flips never panic.
        for cut in (0..out.len()).step_by(7) {
            let _ = decode_jpeg(&out[..cut]);
        }
        for i in (2..out.len()).step_by(5) {
            let mut bad = out.clone();
            bad[i] ^= 0x5A;
            let _ = decode_jpeg(&bad);
        }
    }
}
