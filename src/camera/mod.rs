//! USB (UVC) cameras: enumeration, capture modes and controls, and opening a
//! camera as a frame [`Source`].
//!
//! Backends: Media Foundation on Windows, V4L2 on 64-bit Linux. On every other
//! target (macOS, wasm, …) [`list`] and [`open`] return [`Error::Unsupported`].

use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::frame::fmt_fps;
use crate::source::Interrupter;
use crate::{Error, Format, Frame, Result, Source};

#[cfg(windows)]
mod mf;
#[cfg(windows)]
use mf as backend;

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
mod v4l2;
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
use v4l2 as backend;

#[cfg(not(any(windows, all(target_os = "linux", target_pointer_width = "64"))))]
mod unsupported;
#[cfg(not(any(windows, all(target_os = "linux", target_pointer_width = "64"))))]
use unsupported as backend;

/// Name of this platform's camera API: `Media Foundation`, `V4L2` or `unsupported`.
pub const BACKEND: &str = backend::NAME;

/// Whether this platform has a camera backend: [`list`] and [`open`] work
/// on Windows and 64-bit Linux, and return [`Error::Unsupported`] everywhere
/// else. Sources that are not cameras ([`TestPattern`](crate::TestPattern),
/// [`AviFile`](crate::AviFile)) work regardless.
///
/// ```
/// let supported = cfg!(any(windows, all(target_os = "linux", target_pointer_width = "64")));
/// assert_eq!(stance::camera::SUPPORTED, supported);
/// ```
pub const SUPPORTED: bool = backend::SUPPORTED;

/// A camera found by [`list`], or the description of any other [`Source`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraInfo {
    /// Stable identifier, used by [`open`]: the Media Foundation symbolic link
    /// on Windows; on Linux the `/dev/v4l/by-id/…` path, or `<bus_info>/<card>`
    /// when the camera has no unique by-id link (identical cameras without
    /// serial numbers). It survives replugging into the same port and reboots.
    pub id: String,
    /// Friendly name for display, e.g. `HD Pro Webcam C920`. Not unique.
    pub name: String,
    /// USB vendor and product id, when the OS exposes them.
    pub usb: Option<UsbId>,
    /// Every capture mode the camera offers (all pixel formats).
    pub formats: Vec<Format>,
    /// The camera's controls with their ranges and current state.
    pub controls: Vec<ControlInfo>,
    /// Unit of the exposure control's raw values, for cameras.
    pub exposure_unit: Option<ExposureUnit>,
    /// Why `formats` and `controls` could not be read (camera in use by
    /// another application, access blocked), if they couldn't.
    pub probe_error: Option<String>,
}

impl CameraInfo {
    /// The camera's MJPEG modes.
    pub fn mjpeg_formats(&self) -> impl Iterator<Item = &Format> {
        self.formats.iter().filter(|f| f.is_mjpeg())
    }

    /// The control with this id, if the camera has it.
    #[must_use]
    pub fn control(&self, id: ControlId) -> Option<&ControlInfo> {
        self.controls.iter().find(|c| c.id == id)
    }
}

/// A USB vendor and product id. Displays as `046d:085e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UsbId {
    /// Vendor id.
    pub vid: u16,
    /// Product id.
    pub pid: u16,
}

impl fmt::Display for UsbId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}:{:04x}", self.vid, self.pid)
    }
}

impl UsbId {
    /// `vid_046d&pid_085e` inside a device path → `046d:085e`.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub(crate) fn from_device_path(path: &str) -> Option<Self> {
        let l = path.to_ascii_lowercase();
        let hex4 = |key: &str| {
            let at = l.find(key)? + key.len();
            l.get(at..at + 4)
                .and_then(|h| u16::from_str_radix(h, 16).ok())
        };
        Some(Self {
            vid: hex4("vid_")?,
            pid: hex4("pid_")?,
        })
    }
}

/// Which control a [`ControlInfo`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlId {
    /// Exposure time (see [`ExposureUnit`]).
    Exposure,
    /// Sensor gain.
    Gain,
    /// White balance temperature.
    WhiteBalance,
    /// Focus.
    Focus,
    /// Brightness.
    Brightness,
    /// Contrast.
    Contrast,
    /// Saturation.
    Saturation,
    /// Sharpness.
    Sharpness,
    /// Gamma.
    Gamma,
    /// Hue.
    Hue,
    /// Backlight compensation.
    BacklightCompensation,
    /// Power line frequency (anti-flicker).
    PowerLineFrequency,
    /// Colour on/off.
    ColorEnable,
    /// Zoom.
    Zoom,
    /// Pan.
    Pan,
    /// Tilt.
    Tilt,
    /// Roll.
    Roll,
    /// Iris.
    Iris,
    /// A backend-specific control without a common name (V4L2 control id).
    Other(u32),
}

impl ControlId {
    /// Canonical name for the common controls: `exposure`, `white_balance`, ….
    #[must_use]
    pub fn common_name(self) -> Option<&'static str> {
        Some(match self {
            Self::Exposure => "exposure",
            Self::Gain => "gain",
            Self::WhiteBalance => "white_balance",
            Self::Focus => "focus",
            Self::Brightness => "brightness",
            Self::Contrast => "contrast",
            Self::Saturation => "saturation",
            Self::Sharpness => "sharpness",
            Self::Gamma => "gamma",
            Self::Hue => "hue",
            Self::BacklightCompensation => "backlight_compensation",
            Self::PowerLineFrequency => "power_line_frequency",
            Self::ColorEnable => "color_enable",
            Self::Zoom => "zoom",
            Self::Pan => "pan",
            Self::Tilt => "tilt",
            Self::Roll => "roll",
            Self::Iris => "iris",
            Self::Other(_) => return None,
        })
    }
}

/// A control's range and state, read from the camera.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlInfo {
    /// Which control.
    pub id: ControlId,
    /// Canonical name, or a slug of the driver's name for backend-specific controls.
    pub name: String,
    /// Smallest raw value.
    pub min: i64,
    /// Largest raw value.
    pub max: i64,
    /// Raw value step.
    pub step: i64,
    /// Default raw value.
    pub default: i64,
    /// Current raw value, if the camera reports one.
    pub current: Option<i64>,
    /// The control has an automatic mode.
    pub auto_supported: bool,
    /// The control is currently automatic.
    pub auto: Option<bool>,
    /// Menu entries (value, label) for menu controls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub menu: Vec<(i64, String)>,
}

/// The unit of a camera's raw exposure values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureUnit {
    /// Raw value `v` means `2^v` seconds (UVC through `DirectShow` controls on
    /// Windows), so −7 is 1/128 s.
    Log2Seconds,
    /// Raw value `v` means v × 100 µs (V4L2 `EXPOSURE_ABSOLUTE`), so 78 is 7.8 ms.
    HundredMicros,
}

impl ExposureUnit {
    /// Seconds for a raw exposure value.
    ///
    /// ```
    /// use stance::camera::ExposureUnit;
    /// assert_eq!(ExposureUnit::Log2Seconds.to_seconds(-7), 1.0 / 128.0);
    /// assert!((ExposureUnit::HundredMicros.to_seconds(78) - 0.0078).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn to_seconds(self, raw: i64) -> f64 {
        match self {
            // Clamped to a tiny range, so the cast is exact.
            #[allow(clippy::cast_possible_truncation)]
            Self::Log2Seconds => 2f64.powi(raw.clamp(-64, 64) as i32),
            // Exposure values are far below 2^52.
            #[allow(clippy::cast_precision_loss)]
            Self::HundredMicros => raw as f64 * 100e-6,
        }
    }
}

/// Which capture mode to open. MJPEG only.
///
/// Parses from and displays as `auto` or `WIDTHxHEIGHT@FPS` (`1280x720@120`,
/// `1920x1080@29.97`), and serialises the same way.
///
/// ```
/// use stance::FormatRequest;
/// let req: FormatRequest = "1280x720@120".parse()?;
/// assert_eq!(req, FormatRequest::Mode { width: 1280, height: 720, fps: 120.0 });
/// assert_eq!(req.to_string(), "1280x720@120");
/// assert_eq!("auto".parse::<FormatRequest>()?, FormatRequest::Auto);
/// # Ok::<(), stance::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FormatRequest {
    /// The highest MJPEG frame rate, then the largest size at that rate.
    #[default]
    Auto,
    /// An exact MJPEG size, at the frame rate closest to `fps` (within 0.5).
    Mode {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Frames per second.
        fps: f64,
    },
}

impl fmt::Display for FormatRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Mode { width, height, fps } => {
                write!(f, "{width}x{height}@{}", fmt_fps(*fps))
            }
        }
    }
}

impl FromStr for FormatRequest {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let bad = || {
            Error::InvalidArgument(format!(
                "video mode `{s}`: expected `auto` or WIDTHxHEIGHT@FPS, e.g. 1280x720@120"
            ))
        };
        let (size, fps) = t.split_once('@').ok_or_else(bad)?;
        let (w, h) = size.trim().split_once(['x', 'X', '×']).ok_or_else(bad)?;
        let width: u32 = w.trim().parse().map_err(|_| bad())?;
        let height: u32 = h.trim().parse().map_err(|_| bad())?;
        let fps: f64 = fps.trim().parse().map_err(|_| bad())?;
        if width == 0 || height == 0 || !(fps.is_finite() && fps > 0.0) {
            return Err(bad());
        }
        Ok(Self::Mode { width, height, fps })
    }
}

impl Serialize for FormatRequest {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for FormatRequest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Picks the MJPEG mode for a request (see [`FormatRequest`]).
///
/// # Errors
///
/// [`Error::NoMjpeg`] if the camera has no MJPEG mode, [`Error::NoMatchingFormat`]
/// if none matches the requested size and rate.
///
/// ```
/// use stance::{Format, FormatRequest, choose_format};
/// let modes = [Format::mjpeg(1920, 1080, 30), Format::mjpeg(1280, 720, 120)];
/// assert_eq!(choose_format("cam", &modes, &FormatRequest::Auto)?, modes[1]);
/// # Ok::<(), stance::Error>(())
/// ```
pub fn choose_format(camera: &str, formats: &[Format], request: &FormatRequest) -> Result<Format> {
    let mjpeg: Vec<&Format> = formats.iter().filter(|f| f.is_mjpeg()).collect();
    if mjpeg.is_empty() {
        let mut offered: Vec<&str> = formats.iter().map(|f| f.fourcc.as_str()).collect();
        offered.sort_unstable();
        offered.dedup();
        return Err(Error::NoMjpeg {
            camera: camera.to_owned(),
            offered: if offered.is_empty() {
                "no modes".to_owned()
            } else {
                offered.join(", ")
            },
        });
    }
    let best = match *request {
        FormatRequest::Auto => mjpeg
            .iter()
            .copied()
            .max_by(|a, b| a.fps().total_cmp(&b.fps()).then(a.area().cmp(&b.area()))),
        FormatRequest::Mode { width, height, fps } => mjpeg
            .iter()
            .copied()
            .filter(|f| f.width == width && f.height == height && (f.fps() - fps).abs() < 0.5)
            .min_by(|a, b| (a.fps() - fps).abs().total_cmp(&(b.fps() - fps).abs())),
    };
    best.cloned().ok_or_else(|| {
        let available: Vec<String> = mjpeg
            .iter()
            .map(|f| format!("{}x{}@{}", f.width, f.height, fmt_fps(f.fps())))
            .collect();
        Error::NoMatchingFormat {
            camera: camera.to_owned(),
            request: request.to_string(),
            available: available.join(", "),
        }
    })
}

/// Lists the connected cameras with their capture modes and controls.
///
/// Each camera is opened briefly to read its modes and controls, then closed:
/// no device stays open. A camera that can't be opened (in use by another
/// application, access blocked) is still listed, with empty `formats` and
/// `controls` and `probe_error` set.
///
/// A camera that is open in this process (a streaming [`Camera`], e.g.
/// inside a [`Capture`](crate::Capture)) is never opened again to probe it:
/// its entry is the info read when it was opened. On Windows a second
/// activation would disturb the stream; on Linux the same rule keeps `list`
/// from touching a device this process is streaming from.
///
/// # Errors
///
/// [`Error::Unsupported`] on platforms without a camera backend; a
/// [`Error::Device`] if the platform's device enumeration fails.
///
/// ```no_run
/// for cam in stance::list()? {
///     println!("{} ({})", cam.name, cam.id);
/// }
/// # Ok::<(), stance::Error>(())
/// ```
pub fn list() -> Result<Vec<CameraInfo>> {
    backend::list()
}

/// Opens a camera by its [`CameraInfo::id`] and starts streaming MJPEG in the
/// requested mode.
///
/// A `Camera` belongs to the thread that opened it (Media Foundation objects
/// are per thread); use [`Capture::start`](crate::Capture::start) to run one
/// on its own thread.
///
/// # Errors
///
/// [`Error::NotFound`] for an unknown id, [`Error::NoMjpeg`] /
/// [`Error::NoMatchingFormat`] if the mode isn't available, [`Error::Device`]
/// if the camera can't be opened (in use, access blocked),
/// [`Error::Unsupported`] on platforms without a camera backend.
///
/// ```no_run
/// use std::time::Duration;
/// use stance::{FormatRequest, Source};
///
/// let mut cam = stance::open("/dev/v4l/by-id/usb-cam-video-index0", &FormatRequest::Auto)?;
/// println!("streaming {}", cam.format());
/// if let Some(frame) = cam.next_frame(Duration::from_secs(1))? {
///     println!("{} bytes", frame.jpeg.len());
/// }
/// # Ok::<(), stance::Error>(())
/// ```
pub fn open(id: &str, request: &FormatRequest) -> Result<Camera> {
    let (mut device, info) = backend::open(id)?;
    let format = choose_format(&info.name, &info.formats, request)?;
    device.start(&format)?;
    Ok(Camera {
        device,
        _held: Held::new(&info),
        info,
        format,
        seq: 0,
    })
}

/// Cameras open in this process, with the info read when they were opened.
static HELD: Mutex<Vec<CameraInfo>> = Mutex::new(Vec::new());

/// The info of a camera this process has open, if it has. The camera
/// backends consult it; the unsupported one has nothing to list.
#[cfg_attr(
    not(any(windows, all(target_os = "linux", target_pointer_width = "64"))),
    allow(dead_code)
)]
pub(crate) fn held_info(id: &str) -> Option<CameraInfo> {
    HELD.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .find(|c| c.id == id)
        .cloned()
}

/// Registers an open camera in [`HELD`] until dropped.
struct Held(String);

impl Held {
    fn new(info: &CameraInfo) -> Self {
        HELD.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(info.clone());
        Self(info.id.clone())
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = held.iter().position(|c| c.id == self.0) {
            held.remove(i);
        }
    }
}

/// A frame as a backend delivers it.
pub(crate) struct RawFrame {
    pub data: Arc<[u8]>,
    pub host_ns: u64,
    pub device_ns: Option<u64>,
}

/// An opened platform camera.
pub(crate) trait Device {
    /// Selects a capture mode and starts streaming.
    fn start(&mut self, format: &Format) -> Result<()>;
    /// The next frame, or `None` if none arrived within `timeout`.
    fn read_frame(&mut self, timeout: Duration) -> Result<Option<RawFrame>>;
    /// Ends a blocked `read_frame` from another thread, for backends whose
    /// reads ignore the timeout.
    fn interrupter(&self) -> Option<Interrupter> {
        None
    }
}

/// A streaming camera. See [`open`].
pub struct Camera {
    device: Box<dyn Device>,
    // Dropped after `device`: the camera is closed before it leaves the
    // registry, so `list` never probes a camera that is still closing.
    _held: Held,
    info: CameraInfo,
    format: Format,
    seq: u64,
}

impl fmt::Debug for Camera {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Camera")
            .field("info", &self.info)
            .field("format", &self.format)
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}

impl Source for Camera {
    /// The next frame. On Windows the read waits for the camera regardless of
    /// `timeout` (Media Foundation's synchronous reader has none); a camera
    /// that is unplugged, or an [`Interrupter`] call, ends the wait with an
    /// error.
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>> {
        let Some(raw) = self.device.read_frame(timeout)? else {
            return Ok(None);
        };
        let frame = Frame {
            jpeg: raw.data,
            host_ns: raw.host_ns,
            device_ns: raw.device_ns,
            seq: self.seq,
            width: self.format.width,
            height: self.format.height,
        };
        self.seq += 1;
        Ok(Some(frame))
    }

    fn info(&self) -> &CameraInfo {
        &self.info
    }

    fn format(&self) -> &Format {
        &self.format
    }

    /// On Windows, shuts the camera down: a read blocked in Media Foundation
    /// returns an error and the camera is released.
    fn interrupter(&self) -> Option<Interrupter> {
        self.device.interrupter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(fourcc: &str, w: u32, h: u32, fps: u32) -> Format {
        Format {
            fourcc: fourcc.into(),
            width: w,
            height: h,
            fps_num: fps,
            fps_den: 1,
        }
    }

    fn formats() -> Vec<Format> {
        vec![
            fmt("YUY2", 640, 480, 120),
            fmt("MJPG", 1920, 1080, 30),
            fmt("MJPG", 1280, 720, 60),
            fmt("MJPG", 640, 480, 60),
            fmt("MJPG", 1920, 1080, 60),
            fmt("YUY2", 1920, 1080, 5),
        ]
    }

    #[test]
    fn auto_is_mjpeg_fastest_then_largest() {
        let f = choose_format("c", &formats(), &FormatRequest::Auto).unwrap();
        assert_eq!(f, fmt("MJPG", 1920, 1080, 60));
    }

    #[test]
    fn explicit_mode() {
        let req = "1920x1080@30".parse().unwrap();
        assert_eq!(
            choose_format("c", &formats(), &req).unwrap(),
            fmt("MJPG", 1920, 1080, 30)
        );
        let ntsc = Format {
            fps_num: 30_000,
            fps_den: 1001,
            ..fmt("MJPG", 1280, 720, 0)
        };
        let req = "1280x720@30".parse().unwrap();
        assert_eq!(
            choose_format("c", std::slice::from_ref(&ntsc), &req).unwrap(),
            ntsc
        );
        // YUY2 only at that size: not offered.
        let req = "640x480@120".parse().unwrap();
        let err = choose_format("c", &formats(), &req).unwrap_err();
        assert!(matches!(err, Error::NoMatchingFormat { .. }), "{err}");
        assert!(err.to_string().contains("1280x720@60"), "{err}");
    }

    #[test]
    fn no_mjpeg_is_a_clear_error() {
        let only_yuy2 = vec![fmt("YUY2", 640, 480, 30), fmt("NV12", 640, 480, 30)];
        let err = choose_format("Cam", &only_yuy2, &FormatRequest::Auto).unwrap_err();
        assert_eq!(
            err.to_string(),
            "camera `Cam` has no MJPEG mode (it offers NV12, YUY2)"
        );
        assert!(matches!(
            choose_format("Cam", &[], &FormatRequest::Auto),
            Err(Error::NoMjpeg { .. })
        ));
    }

    #[test]
    fn parses_requests() {
        assert_eq!(
            " Auto ".parse::<FormatRequest>().unwrap(),
            FormatRequest::Auto
        );
        assert_eq!(
            "1920 × 1080 @ 29.97".parse::<FormatRequest>().unwrap(),
            FormatRequest::Mode {
                width: 1920,
                height: 1080,
                fps: 29.97
            }
        );
        for bad in ["", "1280x720", "1280@60", "0x720@60", "1280x720@0", "axb@c"] {
            assert!(bad.parse::<FormatRequest>().is_err(), "{bad}");
        }
        let json = serde_json::to_string(&FormatRequest::Mode {
            width: 1280,
            height: 720,
            fps: 120.0,
        })
        .unwrap();
        assert_eq!(json, r#""1280x720@120""#);
        assert_eq!(
            serde_json::from_str::<FormatRequest>(r#""auto""#).unwrap(),
            FormatRequest::Auto
        );
    }

    #[test]
    fn usb_ids_from_device_paths() {
        let link = r"\\?\usb#vid_046D&pid_085E&mi_00#7&1a2b3c4d&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global";
        let id = UsbId::from_device_path(link).unwrap();
        assert_eq!(
            id,
            UsbId {
                vid: 0x046d,
                pid: 0x085e
            }
        );
        assert_eq!(id.to_string(), "046d:085e");
        assert_eq!(UsbId::from_device_path(r"\\?\root#camera#0000"), None);
        assert_eq!(UsbId::from_device_path("vid_zz12&pid_0001"), None);
    }

    #[test]
    fn control_names() {
        assert_eq!(ControlId::WhiteBalance.common_name(), Some("white_balance"));
        assert_eq!(ControlId::Other(7).common_name(), None);
        assert_eq!(
            serde_json::to_string(&ControlId::WhiteBalance).unwrap(),
            r#""white_balance""#
        );
    }

    #[test]
    fn list_and_open_never_panic() {
        // CI and build machines have no cameras: either an empty list or Unsupported.
        match list() {
            Ok(cams) => {
                for c in cams {
                    assert!(!c.id.is_empty());
                }
            }
            Err(e) => assert!(
                matches!(e, Error::Unsupported | Error::Device { .. }),
                "{e}"
            ),
        }
        assert!(open("no-such-camera", &FormatRequest::Auto).is_err());
    }

    #[test]
    fn open_cameras_are_registered_until_dropped() {
        let info = |id: &str| CameraInfo {
            id: id.into(),
            name: "cam".into(),
            usb: None,
            formats: vec![Format::mjpeg(640, 480, 30)],
            controls: Vec::new(),
            exposure_unit: None,
            probe_error: None,
        };
        let id = "registry-test-camera";
        assert!(held_info(id).is_none());
        let a = Held::new(&info(id));
        let b = Held::new(&info(id));
        assert_eq!(held_info(id).unwrap().formats.len(), 1);
        drop(a);
        assert!(held_info(id).is_some(), "still held once");
        drop(b);
        assert!(held_info(id).is_none());
    }
}
