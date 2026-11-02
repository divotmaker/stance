//! The crate's error type.

use std::io;

/// Everything that can go wrong in this crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Camera capture is not available on this platform (macOS and wasm for now).
    #[error("camera capture is not supported on this platform")]
    Unsupported,

    /// No camera with this id is connected.
    #[error("no camera with id `{0}`")]
    NotFound(String),

    /// The camera offers no MJPEG mode.
    #[error("camera `{camera}` has no MJPEG mode (it offers {offered})")]
    NoMjpeg {
        /// Friendly name of the camera.
        camera: String,
        /// The pixel formats the camera does offer.
        offered: String,
    },

    /// No MJPEG mode matches the requested size and frame rate.
    #[error("camera `{camera}` has no MJPEG mode {request}; available: {available}")]
    NoMatchingFormat {
        /// Friendly name of the camera.
        camera: String,
        /// The request, e.g. `1280x720@120`.
        request: String,
        /// The MJPEG modes the camera offers.
        available: String,
    },

    /// The platform camera API failed.
    #[error("{context}: {message}")]
    Device {
        /// What was being done.
        context: String,
        /// The platform's error message.
        message: String,
    },

    /// The source delivered no frame for this long (a camera that stopped
    /// sending without reporting an error). See `CaptureConfig::stall`.
    #[error("no frame from the source for {0:?}")]
    Stalled(std::time::Duration),

    /// The source has no more frames (end of a file, frame limit reached).
    #[error("the source has no more frames")]
    Ended,

    /// A file or thread operation failed.
    #[error("{context}: {source}")]
    Io {
        /// What was being done.
        context: String,
        /// The underlying error.
        #[source]
        source: io::Error,
    },

    /// A file is not an AVI this crate can read.
    #[error("invalid AVI file: {0}")]
    InvalidAvi(String),

    /// A clip sidecar could not be parsed or does not match its AVI.
    #[error("invalid clip sidecar: {0}")]
    InvalidSidecar(String),

    /// An argument is out of range or malformed.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// JPEG encoding failed.
    #[error("JPEG encoding failed: {0}")]
    Encode(String),
}

/// `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// An [`Error::Io`] with context.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// An [`Error::Device`] with context.
    #[cfg_attr(
        not(any(windows, all(target_os = "linux", target_pointer_width = "64"))),
        allow(dead_code)
    )]
    // Takes the error by value so call sites read `map_err(|e| Error::device(context, e))`.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn device(context: impl Into<String>, message: impl ToString) -> Self {
        Self::Device {
            context: context.into(),
            message: message.to_string(),
        }
    }
}
