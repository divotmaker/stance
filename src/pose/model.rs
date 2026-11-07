//! The model table: where each model comes from and how to check it.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::skeleton::Skeleton;
use crate::{Error, Result};

/// What a model does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelKind {
    /// A top-down pose model (RTMPose with a SimCC head): keypoints of the
    /// person in a box.
    Pose(Skeleton),
    /// A YOLOX detector with raw head outputs (decoded and suppressed in
    /// Rust); only its person class is used.
    PersonDetector,
}

/// One downloadable ONNX model.
///
/// Models are never bundled. They live in a directory as
/// `<name>.onnx` ([`ModelSpec::file_name`]); loading reads the file and
/// checks its SHA-256 before handing it to the runtime. With the
/// `pose-download` feature, [`fetch_model`](crate::pose::fetch_model)
/// downloads the file (extracting it from its zip archive when the source is
/// one) into such a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// Short name; the file is `<name>.onnx`.
    pub name: &'static str,
    /// What the model does.
    pub kind: ModelKind,
    /// HTTPS source: the `.onnx` file itself, or a zip archive holding it.
    pub url: &'static str,
    /// Path of the `.onnx` file inside the zip archive at `url`, `None` when
    /// `url` is the file itself.
    pub zip_member: Option<&'static str>,
    /// SHA-256 of the `.onnx` file, lowercase hex.
    pub sha256: &'static str,
    /// Size of the `.onnx` file in bytes.
    pub size: u64,
    /// Model input width in pixels.
    pub input_width: u32,
    /// Model input height in pixels.
    pub input_height: u32,
    /// License of the weights.
    pub license: &'static str,
}

/// RTMPose-s, Halpe-26, 256×192 (OpenMMLab, trained on the Body7 datasets).
pub const RTMPOSE_S_HALPE26: ModelSpec = ModelSpec {
    name: "rtmpose-s-halpe26",
    kind: ModelKind::Pose(Skeleton::Halpe26),
    url: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
    zip_member: Some(
        "20230831/rtmpose_onnx/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605/end2end.onnx",
    ),
    sha256: "85c5d669aa0b0c7969799ea9e0620d3b8c1d3c2b6878243e98a1ef8351457986",
    size: 22_793_379,
    input_width: 192,
    input_height: 256,
    license: "Apache-2.0",
};

/// RTMPose-m, Halpe-26, 256×192 (OpenMMLab, trained on the Body7 datasets).
pub const RTMPOSE_M_HALPE26: ModelSpec = ModelSpec {
    name: "rtmpose-m-halpe26",
    kind: ModelKind::Pose(Skeleton::Halpe26),
    url: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
    zip_member: Some(
        "20230831/rtmpose_onnx/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605/end2end.onnx",
    ),
    sha256: "26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf",
    size: 55_685_444,
    input_width: 192,
    input_height: 256,
    license: "Apache-2.0",
};

/// YOLOX-Nano, 416×416, COCO (Megvii release 0.1.1rc0).
pub const YOLOX_NANO: ModelSpec = ModelSpec {
    name: "yolox-nano",
    kind: ModelKind::PersonDetector,
    url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
    zip_member: None,
    sha256: "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
    size: 3_659_407,
    input_width: 416,
    input_height: 416,
    license: "Apache-2.0",
};

/// YOLOX-Tiny, 416×416, COCO (Megvii release 0.1.1rc0).
pub const YOLOX_TINY: ModelSpec = ModelSpec {
    name: "yolox-tiny",
    kind: ModelKind::PersonDetector,
    url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
    zip_member: None,
    sha256: "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
    size: 20_219_662,
    input_width: 416,
    input_height: 416,
    license: "Apache-2.0",
};

/// Every model this crate knows.
pub const MODELS: &[ModelSpec] = &[RTMPOSE_S_HALPE26, RTMPOSE_M_HALPE26, YOLOX_NANO, YOLOX_TINY];

impl ModelSpec {
    /// A model of [`MODELS`] by name.
    #[must_use]
    pub fn find(name: &str) -> Option<&'static ModelSpec> {
        MODELS.iter().find(|m| m.name == name)
    }

    /// `<name>.onnx`.
    #[must_use]
    pub fn file_name(&self) -> String {
        format!("{}.onnx", self.name)
    }

    /// `<dir>/<name>.onnx`.
    #[must_use]
    pub fn path_in(&self, dir: &Path) -> PathBuf {
        dir.join(self.file_name())
    }

    /// Checks bytes against the model's SHA-256.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when the hash differs.
    pub fn verify(&self, onnx: &[u8]) -> Result<()> {
        self.check_sha256(&sha256_hex(onnx), onnx.len() as u64)
    }

    /// Checks a SHA-256 (hex) computed over `len` bytes against the model's.
    pub(crate) fn check_sha256(&self, got: &str, len: u64) -> Result<()> {
        if got.eq_ignore_ascii_case(self.sha256) {
            Ok(())
        } else {
            Err(Error::model(
                self.name,
                format!(
                    "SHA-256 mismatch: expected {}, the file has {got} ({len} bytes)",
                    self.sha256
                ),
            ))
        }
    }

    /// Reads `<dir>/<name>.onnx` and verifies it.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when the file is missing, unreadable, of the wrong
    /// size or doesn't match the hash.
    pub fn read_verified(&self, dir: &Path) -> Result<Vec<u8>> {
        let path = self.path_in(dir);
        let len = std::fs::metadata(&path)
            .map_err(|e| Error::model(self.name, format!("{}: {e}", path.display())))?
            .len();
        if len != self.size {
            return Err(Error::model(
                self.name,
                format!("{} is {len} bytes, expected {}", path.display(), self.size),
            ));
        }
        let bytes = std::fs::read(&path)
            .map_err(|e| Error::model(self.name, format!("{}: {e}", path.display())))?;
        self.verify(&bytes)?;
        Ok(bytes)
    }

    /// The skeleton of a pose model.
    pub(crate) fn skeleton(&self) -> Result<Skeleton> {
        match self.kind {
            ModelKind::Pose(s) => Ok(s),
            ModelKind::PersonDetector => Err(Error::model(self.name, "not a pose model")),
        }
    }

    /// Fails unless the model is a person detector.
    pub(crate) fn expect_detector(&self) -> Result<()> {
        match self.kind {
            ModelKind::PersonDetector => Ok(()),
            ModelKind::Pose(_) => Err(Error::model(self.name, "not a person detector")),
        }
    }
}

/// Lowercase hex SHA-256.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Lowercase hex.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn spec_for(bytes: &[u8]) -> ModelSpec {
        ModelSpec {
            name: "test-model",
            sha256: Box::leak(sha256_hex(bytes).into_boxed_str()),
            size: bytes.len() as u64,
            ..YOLOX_NANO
        }
    }

    #[test]
    fn sha256_of_known_input() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn verify_accepts_good_and_rejects_bad_bytes() {
        let good = b"pretend onnx".to_vec();
        let spec = spec_for(&good);
        spec.verify(&good).unwrap();
        let mut bad = good.clone();
        bad[0] ^= 1;
        assert!(matches!(spec.verify(&bad), Err(Error::Model { .. })));

        let dir = TempDir::new("model-verify");
        assert!(spec.read_verified(dir.path()).is_err(), "missing file");
        std::fs::write(spec.path_in(dir.path()), &bad).unwrap();
        assert!(spec.read_verified(dir.path()).is_err(), "wrong hash");
        std::fs::write(spec.path_in(dir.path()), b"short").unwrap();
        assert!(spec.read_verified(dir.path()).is_err(), "wrong size");
        std::fs::write(spec.path_in(dir.path()), &good).unwrap();
        assert_eq!(spec.read_verified(dir.path()).unwrap(), good);
    }

    #[test]
    fn table_is_well_formed() {
        for m in MODELS {
            assert_eq!(ModelSpec::find(m.name), Some(m));
            assert_eq!(m.sha256.len(), 64);
            assert!(
                m.sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            );
            assert!(m.url.starts_with("https://"));
            assert_eq!(
                m.zip_member.is_some(),
                std::path::Path::new(m.url)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("zip"))
            );
            assert!(m.size > 0 && m.input_width > 0 && m.input_height > 0);
        }
        assert!(ModelSpec::find("nope").is_none());
        assert!(RTMPOSE_M_HALPE26.skeleton().is_ok());
        assert!(YOLOX_TINY.skeleton().is_err());
        assert!(YOLOX_TINY.expect_detector().is_ok());
        assert!(RTMPOSE_S_HALPE26.expect_detector().is_err());
    }
}
