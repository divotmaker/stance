//! YOLOX person detection with raw head outputs.
//!
//! The model takes a letterboxed square image (resized to fit, top left,
//! padded with grey 114), raw 0–255 BGR values. Its output is one row per
//! anchor point on the stride 8, 16 and 32 grids: `cx, cy, log w, log h`
//! relative to the grid cell, objectness, then 80 COCO class scores. This
//! module decodes the grid, scores each box as objectness × person score,
//! drops boxes below the threshold and suppresses overlaps (rtmlib's
//! `YOLOX` post-processing for raw exports, person class only).

use std::path::Path;

use super::PersonDetector;
use super::geometry::{BBox, Detection, nms};
use super::image::{ChannelOrder, Image, letterbox_to_tensor};
use super::model::ModelSpec;
use super::runtime::{InferencePool, Network, Session};
use crate::{Error, Result};

/// Grey used to pad the letterbox.
const PAD: f32 = 114.0;
/// Grid strides of the YOLOX heads.
const STRIDES: [usize; 3] = [8, 16, 32];
/// Columns per anchor: box (4), objectness, 80 classes.
const COLUMNS: usize = 85;

/// A YOLOX detector reporting people.
///
/// One instance runs one inference at a time; [`PersonDetector::fork`]
/// makes another that shares the loaded weights, for a worker thread.
#[derive(Debug)]
pub struct Yolox {
    spec: ModelSpec,
    session: Session,
    score_threshold: f32,
    nms_threshold: f32,
}

impl Yolox {
    /// Default minimum person score (objectness × class score).
    pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.3;
    /// Default IoU above which a lower-scored box is suppressed.
    pub const DEFAULT_NMS_THRESHOLD: f32 = 0.45;

    /// Loads `<dir>/<name>.onnx` for a detector of the table, checking its
    /// SHA-256.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] if the file is missing or doesn't verify, the spec
    /// isn't a person detector, or the runtime can't load it.
    pub fn load(spec: &ModelSpec, dir: &Path) -> Result<Self> {
        let bytes = spec.read_verified(dir)?;
        Self::from_verified_bytes(spec, &bytes)
    }

    /// Loads a model from bytes after checking them against the spec's
    /// SHA-256.
    ///
    /// # Errors
    ///
    /// As [`Yolox::load`].
    pub fn from_bytes(spec: &ModelSpec, onnx: &[u8]) -> Result<Self> {
        spec.verify(onnx)?;
        Self::from_verified_bytes(spec, onnx)
    }

    fn from_verified_bytes(spec: &ModelSpec, onnx: &[u8]) -> Result<Self> {
        spec.expect_detector()?;
        if spec.input_width != spec.input_height {
            return Err(Error::model(spec.name, "YOLOX input must be square"));
        }
        let size = spec.input_width as usize;
        let session = Network::load(spec.name, onnx, [1, 3, size, size])?.session()?;
        Ok(Self {
            spec: *spec,
            session,
            score_threshold: Self::DEFAULT_SCORE_THRESHOLD,
            nms_threshold: Self::DEFAULT_NMS_THRESHOLD,
        })
    }

    /// Sets the minimum person score and the suppression IoU.
    pub fn set_thresholds(&mut self, score: f32, nms: f32) {
        self.score_threshold = score;
        self.nms_threshold = nms;
    }

    /// The model's spec.
    #[must_use]
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
}

impl PersonDetector for Yolox {
    fn model_name(&self) -> &str {
        self.spec.name
    }

    fn fork(&self) -> Result<Self> {
        Ok(Self {
            spec: self.spec,
            session: self.session.network().session()?,
            score_threshold: self.score_threshold,
            nms_threshold: self.nms_threshold,
        })
    }

    fn set_inference_pool(&mut self, pool: &InferencePool) -> Result<()> {
        self.session.set_pool(pool);
        Ok(())
    }

    fn detect(&mut self, image: &Image) -> Result<Vec<Detection>> {
        let size = self.spec.input_width as usize;
        let (tensor, scale) = letterbox_to_tensor(image, size, PAD, ChannelOrder::Bgr);
        let outputs = self.session.run(&tensor)?;
        let out = outputs
            .first()
            .ok_or_else(|| Error::model(self.spec.name, "no output"))?;
        if out.data.len() != anchors(size) * COLUMNS {
            return Err(Error::model(
                self.spec.name,
                format!(
                    "output {:?}, expected [1, {}, {COLUMNS}]",
                    out.shape,
                    anchors(size)
                ),
            ));
        }
        Ok(decode(
            &out.data,
            size,
            scale,
            self.score_threshold,
            self.nms_threshold,
        ))
    }
}

/// Anchor points for a square input of `size`.
fn anchors(size: usize) -> usize {
    STRIDES.iter().map(|s| (size / s) * (size / s)).sum()
}

/// Decodes raw YOLOX rows into person detections in image pixels (input
/// pixels divided by `scale`), best first.
pub(crate) fn decode(
    rows: &[f32],
    size: usize,
    scale: f64,
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<Detection> {
    // The letterbox scale is a positive ratio of small integers.
    #[allow(clippy::cast_possible_truncation)]
    let inv = if scale > 0.0 {
        (1.0 / scale) as f32
    } else {
        1.0
    };
    let mut dets = Vec::new();
    let mut row = 0usize;
    for stride in STRIDES {
        let cells = size / stride;
        // Grid coordinates and strides are small integers: exact in f32.
        #[allow(clippy::cast_precision_loss)]
        let step = stride as f32;
        for gy in 0..cells {
            for gx in 0..cells {
                let Some(r) = rows.get(row * COLUMNS..(row + 1) * COLUMNS) else {
                    return nms(dets, nms_threshold);
                };
                row += 1;
                let score = r[4] * r[5];
                if score.is_nan() || score <= score_threshold {
                    continue;
                }
                #[allow(clippy::cast_precision_loss)]
                let (cx, cy) = ((r[0] + gx as f32) * step, (r[1] + gy as f32) * step);
                let (w, h) = (r[2].exp() * step, r[3].exp() * step);
                dets.push(Detection {
                    bbox: BBox::new(
                        (cx - w / 2.0) * inv,
                        (cy - h / 2.0) * inv,
                        (cx + w / 2.0) * inv,
                        (cy + h / 2.0) * inv,
                    ),
                    score,
                });
            }
        }
    }
    nms(dets, nms_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_grid_cells_and_keeps_people_only() {
        let size = 64; // grids 8×8, 4×4, 2×2
        assert_eq!(anchors(size), 84);
        let mut rows = vec![0.0f32; 84 * COLUMNS];
        // A person in stride-8 cell (3, 2): centre offset (0.5, 0.5), size e^1·8.
        let r = &mut rows[(2 * 8 + 3) * COLUMNS..][..COLUMNS];
        r[..6].copy_from_slice(&[0.5, 0.5, 1.0, 1.0, 0.9, 0.8]);
        // A confident car in the first stride-32 cell: not a person.
        let r = &mut rows[(64 + 16) * COLUMNS..][..COLUMNS];
        r[..5].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.99]);
        r[5 + 2] = 0.99;
        let dets = decode(&rows, size, 0.5, 0.3, 0.45);
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert!((d.score - 0.72).abs() < 1e-6);
        let half = 1.0f32.exp() * 8.0 / 2.0;
        let (cx, cy) = (3.5 * 8.0, 2.5 * 8.0);
        let expect = BBox::new(
            (cx - half) * 2.0,
            (cy - half) * 2.0,
            (cx + half) * 2.0,
            (cy + half) * 2.0,
        );
        for (a, b) in [
            (d.bbox.x1, expect.x1),
            (d.bbox.y1, expect.y1),
            (d.bbox.x2, expect.x2),
            (d.bbox.y2, expect.y2),
        ] {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn short_or_hostile_output_does_not_panic() {
        assert!(decode(&[], 64, 1.0, 0.3, 0.45).is_empty());
        let rows = vec![f32::NAN; 84 * COLUMNS];
        assert!(decode(&rows, 64, 1.0, 0.3, 0.45).is_empty());
        let mut rows = vec![0.0f32; 84 * COLUMNS];
        rows[2] = f32::INFINITY;
        rows[4] = 1.0;
        rows[5] = 1.0;
        assert!(decode(&rows, 64, 1.0, 0.3, 0.45).is_empty());
    }
}
