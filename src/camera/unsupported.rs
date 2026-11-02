//! Backend for platforms without camera support (macOS and wasm for now).

use super::{CameraInfo, Device};
use crate::{Error, Result};

pub(super) const NAME: &str = "unsupported";
pub(super) const SUPPORTED: bool = false;

pub(super) fn list() -> Result<Vec<CameraInfo>> {
    Err(Error::Unsupported)
}

pub(super) fn open(_id: &str) -> Result<(Box<dyn Device>, CameraInfo)> {
    Err(Error::Unsupported)
}
