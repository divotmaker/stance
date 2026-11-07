//! Anything that delivers frames: a camera, a test pattern, an AVI file.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::{CameraInfo, Format, Frame, Result};

/// Ends a [`Source::next_frame`] call that is blocked on another thread.
///
/// Sources whose reads can block past their timeout hand one out through
/// [`Source::interrupter`]; [`Capture`](crate::Capture) uses it to stop
/// promptly. After an interrupt the source delivers no more frames: the
/// blocked read and every later one return an error. Interrupting a source
/// that is already closed does nothing.
#[derive(Clone)]
pub struct Interrupter(Arc<dyn Fn() + Send + Sync>);

impl Interrupter {
    /// An interrupter that runs `f`. `f` may be called from any thread, more
    /// than once, and after the source is gone.
    pub fn new(f: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    /// Interrupts the source.
    pub fn interrupt(&self) {
        (self.0)();
    }
}

impl fmt::Debug for Interrupter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Interrupter")
    }
}

/// A blocking frame source.
///
/// Implemented by [`Camera`](crate::Camera), [`TestPattern`](crate::TestPattern)
/// and [`AviFile`](crate::AviFile), so capture code, tests and replays can't
/// tell them apart.
pub trait Source {
    /// Waits up to `timeout` for the next frame. `Ok(None)` on timeout.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`](crate::Error::Ended) when a finite source has no more
    /// frames; a [`Error::Device`](crate::Error::Device) when a camera fails or
    /// is unplugged. A source that returned an error delivers no more frames.
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>>;

    /// What this source is: the camera's info, or a description of the
    /// synthetic or file source.
    fn info(&self) -> &CameraInfo;

    /// The mode frames arrive in.
    fn format(&self) -> &Format;

    /// A handle that ends a blocked [`Source::next_frame`] from another
    /// thread, for sources whose reads can outlast their timeout (Media
    /// Foundation cameras). `None` (the default) for sources that honor the
    /// timeout.
    fn interrupter(&self) -> Option<Interrupter> {
        None
    }
}

impl<S: Source + ?Sized> Source for Box<S> {
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>> {
        (**self).next_frame(timeout)
    }

    fn info(&self) -> &CameraInfo {
        (**self).info()
    }

    fn format(&self) -> &Format {
        (**self).format()
    }

    fn interrupter(&self) -> Option<Interrupter> {
        (**self).interrupter()
    }
}
