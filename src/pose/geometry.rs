//! Boxes, affine transforms and non-maximum suppression.

use serde::{Deserialize, Serialize};

/// An axis-aligned box in image pixels: `(x1, y1)` top left, `(x2, y2)`
/// bottom right.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BBox {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge.
    pub x2: f32,
    /// Bottom edge.
    pub y2: f32,
}

impl BBox {
    /// A box from its corners.
    #[must_use]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Width (zero for an inverted box).
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    /// Height (zero for an inverted box).
    #[must_use]
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    /// Area.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Center point.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Intersection over union, in `[0, 1]` (0 when either box is empty or
    /// not finite).
    #[must_use]
    pub fn iou(&self, other: &BBox) -> f32 {
        if !self.is_valid() || !other.is_valid() {
            return 0.0;
        }
        let iw = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let ih = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union > 0.0 && union.is_finite() {
            (inter / union).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The box scaled about its center by `factor`.
    #[must_use]
    pub fn expand(&self, factor: f32) -> BBox {
        let (cx, cy) = self.center();
        let (hw, hh) = (
            (self.x2 - self.x1) * 0.5 * factor,
            (self.y2 - self.y1) * 0.5 * factor,
        );
        BBox::new(cx - hw, cy - hh, cx + hw, cy + hh)
    }

    /// The tightest box around the points, `None` for no (finite) points.
    #[must_use]
    pub fn around(points: impl IntoIterator<Item = (f32, f32)>) -> Option<BBox> {
        let mut b: Option<BBox> = None;
        for (x, y) in points {
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            b = Some(match b {
                None => BBox::new(x, y, x, y),
                Some(b) => BBox::new(b.x1.min(x), b.y1.min(y), b.x2.max(x), b.y2.max(y)),
            });
        }
        b
    }

    /// Linear interpolation from `self` (`t = 0`) to `other` (`t = 1`).
    #[must_use]
    pub fn lerp(&self, other: &BBox, t: f32) -> BBox {
        let l = |a: f32, b: f32| a + (b - a) * t;
        BBox::new(
            l(self.x1, other.x1),
            l(self.y1, other.y1),
            l(self.x2, other.x2),
            l(self.y2, other.y2),
        )
    }

    /// All four coordinates are finite and the box has a positive area.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        [self.x1, self.y1, self.x2, self.y2]
            .iter()
            .all(|v| v.is_finite())
            && self.x2 > self.x1
            && self.y2 > self.y1
    }
}

/// A 2D affine transform `(x, y) → (a·x + b·y + c, d·x + e·y + f)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Affine {
    m: [f64; 6],
}

impl Affine {
    pub(crate) const IDENTITY: Self = Self {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    };

    /// The transform `[a, b, c, d, e, f]`.
    pub(crate) fn new(m: [f64; 6]) -> Self {
        Self { m }
    }

    pub(crate) fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.m;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }

    /// The inverse transform, `None` when singular.
    pub(crate) fn inverse(&self) -> Option<Self> {
        let m = &self.m;
        let det = m[0] * m[4] - m[1] * m[3];
        if det.abs() < 1e-12 || !det.is_finite() {
            return None;
        }
        let (ia, ib, id, ie) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
        Some(Self::new([
            ia,
            ib,
            -(ia * m[2] + ib * m[5]),
            id,
            ie,
            -(id * m[2] + ie * m[5]),
        ]))
    }
}

/// The crop a top-down pose model sees for a box: center and size in image
/// pixels, with the size padded and widened or heightened to the model's
/// aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Crop {
    pub(crate) center: (f64, f64),
    /// Width and height in image pixels.
    pub(crate) scale: (f64, f64),
}

impl Crop {
    /// rtmlib's `bbox_xyxy2cs` (padding) followed by `top_down_affine`'s
    /// aspect fix for an `input_w × input_h` model.
    pub(crate) fn for_box(bbox: &BBox, padding: f64, input_w: u32, input_h: u32) -> Self {
        let (x1, y1, x2, y2) = (
            f64::from(bbox.x1),
            f64::from(bbox.y1),
            f64::from(bbox.x2),
            f64::from(bbox.y2),
        );
        let center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5);
        let (w, h) = ((x2 - x1) * padding, (y2 - y1) * padding);
        let aspect = f64::from(input_w) / f64::from(input_h);
        let scale = if w > h * aspect {
            (w, w / aspect)
        } else {
            (h * aspect, h)
        };
        Self { center, scale }
    }

    /// Image → crop pixels for an `input_w × input_h` model (no rotation):
    /// the crop's center lands on the model input's center, its width on the
    /// input's width.
    pub(crate) fn input_transform(&self, input_w: u32, input_h: u32) -> Affine {
        let k = if self.scale.0 > 0.0 {
            f64::from(input_w) / self.scale.0
        } else {
            1.0
        };
        Affine::new([
            k,
            0.0,
            f64::from(input_w) * 0.5 - k * self.center.0,
            0.0,
            k,
            f64::from(input_h) * 0.5 - k * self.center.1,
        ])
    }
}

/// A scored box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// The box in image pixels.
    pub bbox: BBox,
    /// Detector confidence in `[0, 1]`.
    pub score: f32,
}

/// Greedy non-maximum suppression (rtmlib's `nms`, which counts box sizes
/// inclusively, `x2 − x1 + 1`): keeps the best box, drops every box
/// overlapping it by more than `iou_threshold`, repeats. Returns the kept
/// detections, best first.
#[must_use]
pub(crate) fn nms(mut dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    dets.retain(|d| d.score.is_finite() && d.bbox.is_valid());
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let area = |b: &BBox| (b.x2 - b.x1 + 1.0) * (b.y2 - b.y1 + 1.0);
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        let suppressed = keep.iter().any(|k| {
            let w = (k.bbox.x2.min(d.bbox.x2) - k.bbox.x1.max(d.bbox.x1) + 1.0).max(0.0);
            let h = (k.bbox.y2.min(d.bbox.y2) - k.bbox.y1.max(d.bbox.y1) + 1.0).max(0.0);
            let inter = w * h;
            inter / (area(&k.bbox) + area(&d.bbox) - inter) > iou_threshold
        });
        if !suppressed {
            keep.push(d);
        }
    }
    keep
}

#[cfg(test)]
// Disjoint and empty boxes have an IoU of exactly zero.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn affine_inverse_round_trips() {
        let crop = Crop::for_box(&BBox::new(250.0, 113.0, 399.0, 436.0), 1.25, 192, 256);
        let to = crop.input_transform(192, 256);
        let back = to.inverse().unwrap();
        for (x, y) in [(0.0, 0.0), (250.0, 113.0), (399.5, 436.25), (-30.0, 900.0)] {
            let (u, v) = to.apply(x, y);
            let (x2, y2) = back.apply(u, v);
            assert!((x - x2).abs() < 1e-9 && (y - y2).abs() < 1e-9);
        }
        // The box center maps to the input center, and the padded crop spans
        // the input.
        let (u, v) = to.apply(crop.center.0, crop.center.1);
        assert!((u - 96.0).abs() < 1e-9 && (v - 128.0).abs() < 1e-9);
        let (u0, v0) = to.apply(
            crop.center.0 - crop.scale.0 / 2.0,
            crop.center.1 - crop.scale.1 / 2.0,
        );
        assert!(u0.abs() < 1e-9 && v0.abs() < 1e-9);
        assert!(
            Affine::new([1.0, 2.0, 0.0, 2.0, 4.0, 0.0])
                .inverse()
                .is_none()
        );
    }

    #[test]
    fn crop_fixes_the_aspect_ratio() {
        // Tall box: height wins, width grows to 3:4.
        let c = Crop::for_box(&BBox::new(0.0, 0.0, 100.0, 400.0), 1.25, 192, 256);
        assert_eq!(c.center, (50.0, 200.0));
        assert!((c.scale.1 - 500.0).abs() < 1e-9 && (c.scale.0 - 375.0).abs() < 1e-9);
        // Wide box: width wins.
        let c = Crop::for_box(&BBox::new(0.0, 0.0, 400.0, 100.0), 1.0, 192, 256);
        assert!((c.scale.0 - 400.0).abs() < 1e-9);
        assert!((c.scale.1 - 400.0 * 256.0 / 192.0).abs() < 1e-9);
    }

    #[test]
    fn iou_of_known_boxes() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
        assert!((a.iou(&BBox::new(5.0, 0.0, 15.0, 10.0)) - 50.0 / 150.0).abs() < 1e-6);
        assert_eq!(a.iou(&BBox::new(20.0, 20.0, 30.0, 30.0)), 0.0);
        assert_eq!(a.iou(&BBox::default()), 0.0);
        assert_eq!(BBox::new(f32::NAN, 0.0, 1.0, 1.0).iou(&a), 0.0);
    }

    #[test]
    fn box_helpers() {
        let b = BBox::around([(1.0, 5.0), (3.0, 2.0), (f32::NAN, 0.0)]).unwrap();
        assert_eq!(b, BBox::new(1.0, 2.0, 3.0, 5.0));
        assert!(BBox::around(std::iter::empty()).is_none());
        assert_eq!(b.expand(2.0), BBox::new(0.0, 0.5, 4.0, 6.5));
        let c = b.lerp(&BBox::new(3.0, 4.0, 5.0, 7.0), 0.5);
        assert_eq!(c, BBox::new(2.0, 3.0, 4.0, 6.0));
    }

    #[test]
    fn nms_keeps_the_best_of_each_cluster() {
        let d = |x: f32, s: f32| Detection {
            bbox: BBox::new(x, 0.0, x + 100.0, 200.0),
            score: s,
        };
        let kept = nms(
            vec![
                d(0.0, 0.8),
                d(5.0, 0.9),
                d(300.0, 0.5),
                d(10.0, 0.7),
                d(305.0, 0.6),
            ],
            0.45,
        );
        assert_eq!(kept, vec![d(5.0, 0.9), d(305.0, 0.6)]);
        // Invalid boxes and NaN scores are dropped, not compared.
        let bad = Detection {
            bbox: BBox::new(0.0, 0.0, f32::NAN, 1.0),
            score: 0.99,
        };
        let nan = Detection {
            score: f32::NAN,
            ..d(0.0, 0.0)
        };
        assert_eq!(nms(vec![bad, nan, d(0.0, 0.3)], 0.45), vec![d(0.0, 0.3)]);
        assert!(nms(Vec::new(), 0.45).is_empty());
    }
}
