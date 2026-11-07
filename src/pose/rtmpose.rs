//! RTMPose: top-down keypoints with a SimCC head.
//!
//! A port of rtmlib's `RTMPose` (Apache-2.0, <https://github.com/Tau-J/rtmlib>):
//! box → center and size with 1.25 padding, widened or heightened to the
//! model's 3:4 aspect → affine warp to 192×256 (bilinear) → per-channel
//! mean/std normalization of the RGB pixels → SimCC: per keypoint, the argmax
//! of the x and y vectors divided by the split ratio (2.0), the score the
//! mean of the two maxima → back to image pixels through the inverse warp.

use std::path::Path;

use super::geometry::{BBox, Crop};
use super::image::{ChannelOrder, Image, Normalize, warp_to_tensor};
use super::model::ModelSpec;
use super::runtime::{InferencePool, Network, Session};
use super::skeleton::Skeleton;
use super::{Keypoint, Pose, PoseEngine};
use crate::{Error, Result};

/// Padding applied around the box before cropping.
const PADDING: f64 = 1.25;
/// SimCC bins per input pixel.
const SIMCC_SPLIT_RATIO: f32 = 2.0;
/// `ImageNet` mean and standard deviation in RGB order, 0–255 scale.
const NORMALIZE: Normalize = Normalize {
    mean: [123.675, 116.28, 103.53],
    std: [58.395, 57.12, 57.375],
};

/// RTMPose on the CPU.
///
/// One instance runs one inference at a time; [`PoseEngine::fork`] makes
/// another that shares the loaded weights, for a worker thread.
#[derive(Debug)]
pub struct RtmPose {
    spec: ModelSpec,
    skeleton: Skeleton,
    session: Session,
}

impl RtmPose {
    /// Loads `<dir>/<name>.onnx` for a pose model of the table, checking its
    /// SHA-256.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] if the file is missing or doesn't verify, the spec
    /// isn't a pose model, or the runtime can't load it.
    pub fn load(spec: &ModelSpec, dir: &Path) -> Result<Self> {
        let bytes = spec.read_verified(dir)?;
        Self::from_verified_bytes(spec, &bytes)
    }

    /// Loads a model from bytes after checking them against the spec's
    /// SHA-256.
    ///
    /// # Errors
    ///
    /// As [`RtmPose::load`].
    pub fn from_bytes(spec: &ModelSpec, onnx: &[u8]) -> Result<Self> {
        spec.verify(onnx)?;
        Self::from_verified_bytes(spec, onnx)
    }

    fn from_verified_bytes(spec: &ModelSpec, onnx: &[u8]) -> Result<Self> {
        let skeleton = spec.skeleton()?;
        let input = [1, 3, spec.input_height as usize, spec.input_width as usize];
        let session = Network::load(spec.name, onnx, input)?.session()?;
        Ok(Self {
            spec: *spec,
            skeleton,
            session,
        })
    }

    /// The model's spec.
    #[must_use]
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
}

impl PoseEngine for RtmPose {
    fn skeleton(&self) -> Skeleton {
        self.skeleton
    }

    fn model_name(&self) -> &str {
        self.spec.name
    }

    fn fork(&self) -> Result<Self> {
        Ok(Self {
            spec: self.spec,
            skeleton: self.skeleton,
            session: self.session.network().session()?,
        })
    }

    fn set_inference_pool(&mut self, pool: &InferencePool) -> Result<()> {
        self.session.set_pool(pool);
        Ok(())
    }

    fn estimate(&mut self, image: &Image, bbox: &BBox) -> Result<Pose> {
        if !bbox.is_valid() {
            return Err(Error::InvalidArgument(format!("pose box {bbox:?}")));
        }
        let (in_w, in_h) = (self.spec.input_width, self.spec.input_height);
        let crop = Crop::for_box(bbox, PADDING, in_w, in_h);
        let to_input = crop.input_transform(in_w, in_h);
        let tensor = warp_to_tensor(
            image,
            &to_input,
            in_w as usize,
            in_h as usize,
            ChannelOrder::Rgb,
            NORMALIZE,
        );
        let outputs = self.session.run(&tensor)?;
        let k = self.skeleton.len();
        let (Some(x), Some(y)) = (outputs.first(), outputs.get(1)) else {
            return Err(Error::model(self.spec.name, "expected two SimCC outputs"));
        };
        let (wx, wy) = (x.data.len() / k.max(1), y.data.len() / k.max(1));
        if wx == 0 || wy == 0 || x.data.len() != k * wx || y.data.len() != k * wy {
            return Err(Error::model(
                self.spec.name,
                format!(
                    "SimCC outputs {:?} and {:?} for {k} keypoints",
                    x.shape, y.shape
                ),
            ));
        }
        let from_input = to_input
            .inverse()
            .ok_or_else(|| Error::InvalidArgument(format!("degenerate pose box {bbox:?}")))?;
        let keypoints = simcc_decode(&x.data, &y.data, k, SIMCC_SPLIT_RATIO)
            .into_iter()
            .map(|(u, v, score)| {
                let (px, py) = from_input.apply(f64::from(u), f64::from(v));
                // Image coordinates are small; f32 keeps sub-pixel precision.
                #[allow(clippy::cast_possible_truncation)]
                Keypoint {
                    x: px as f32,
                    y: py as f32,
                    score,
                }
            })
            .collect();
        Ok(Pose::new(self.skeleton, keypoints, *bbox))
    }
}

/// Decodes SimCC vectors (`k × wx` and `k × wy`, row-major) into
/// `(x, y, score)` per keypoint in model-input pixels: the argmax of each
/// vector divided by `split_ratio`, the score the mean of the two maxima.
/// Keypoints whose score is not positive get bin −1 on both axes, divided
/// by `split_ratio` like any other bin (`(-0.5, -0.5)` at ratio 2), as
/// rtmlib marks them.
pub(crate) fn simcc_decode(
    simcc_x: &[f32],
    simcc_y: &[f32],
    k: usize,
    split_ratio: f32,
) -> Vec<(f32, f32, f32)> {
    let (wx, wy) = (simcc_x.len() / k.max(1), simcc_y.len() / k.max(1));
    (0..k)
        .map(|i| {
            let (ix, vx) = argmax(&simcc_x[i * wx..(i + 1) * wx]);
            let (iy, vy) = argmax(&simcc_y[i * wy..(i + 1) * wy]);
            let score = 0.5 * (vx + vy);
            if score > 0.0 {
                // Bin indices are below a few thousand: exact in f32.
                #[allow(clippy::cast_precision_loss)]
                (ix as f32 / split_ratio, iy as f32 / split_ratio, score)
            } else {
                let unlocated = -1.0 / split_ratio;
                (unlocated, unlocated, score)
            }
        })
        .collect()
}

/// Index and value of the first maximum (`NumPy`'s `argmax`); NaNs never win.
fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = (0, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    if best.1 == f32::NEG_INFINITY {
        (0, 0.0)
    } else {
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simcc_decode_finds_peaks() {
        // Two keypoints, 8 x bins and 6 y bins each.
        let mut x = vec![0.0f32; 16];
        let mut y = vec![0.0f32; 12];
        x[5] = 0.9; // kp 0 at bin 5
        y[2] = 0.7; // kp 0 at bin 2
        x[8 + 1] = 0.4;
        x[8 + 6] = 0.4; // tie: the first wins
        y[6 + 5] = 0.2;
        let d = simcc_decode(&x, &y, 2, 2.0);
        assert_eq!((d[0].0, d[0].1), (2.5, 1.0));
        assert!((d[0].2 - 0.8).abs() < 1e-6);
        assert_eq!((d[1].0, d[1].1), (0.5, 2.5));
        assert!((d[1].2 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn simcc_decode_marks_non_positive_scores() {
        let x = vec![-1.0f32; 4];
        let y = vec![-2.0f32, -0.5];
        let d = simcc_decode(&x, &y, 1, 2.0);
        assert_eq!((d[0].0, d[0].1), (-0.5, -0.5));
        assert!(d[0].2 < 0.0);
        let nan = vec![f32::NAN; 4];
        let d = simcc_decode(&nan, &nan, 1, 2.0);
        assert_eq!(d[0], (-0.5, -0.5, 0.0));
    }
}
