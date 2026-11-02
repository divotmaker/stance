//! Windows backend: Media Foundation through the `windows` crate.
//!
//! - Enumeration: `MFEnumDeviceSources` with the video-capture source type.
//!   A camera's id is its symbolic link.
//! - Frames: an `IMFSourceReader` on the device's media source, set to one of
//!   the device's *native* media types with converters disabled, so MJPEG
//!   arrives as the camera's own compressed bytes. Reads are synchronous.
//! - Controls: `IAMCameraControl` (exposure, focus, zoom, …) and
//!   `IAMVideoProcAmp` (gain, white balance, brightness, …) queried from the
//!   media source. Exposure is in log2 seconds.
//! - Stopping: a read waits for the camera whatever the timeout. The
//!   [`Interrupter`] shuts the media source down from the stopping thread,
//!   which ends a pending read with an error and releases the camera.
//! - Enumeration skips cameras this process has open (probing would disturb
//!   them) and lists them with the info read when they were opened.

use std::ptr;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use windows::Win32::Media::DirectShow::{IAMCameraControl, IAMVideoProcAmp};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFAttributes, IMFMediaSource, IMFMediaType, IMFSample, IMFSourceReader,
    MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, MF_E_NO_MORE_TYPES, MF_LOW_LATENCY,
    MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_SUBTYPE, MF_READWRITE_DISABLE_CONVERTERS,
    MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_DISABLE_DXVA,
    MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED,
    MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR, MF_VERSION, MFCreateAttributes,
    MFCreateSourceReaderFromMediaSource, MFEnumDeviceSources, MFSTARTUP_LITE,
    MFSampleExtension_DeviceReferenceSystemTime, MFShutdown, MFStartup,
};
use windows::Win32::System::Com::{
    COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree, CoUninitialize,
};
use windows::core::{GUID, Interface, PWSTR};

use super::{CameraInfo, ControlId, ControlInfo, Device, ExposureUnit, RawFrame, UsbId};
use crate::source::Interrupter;
use crate::{Error, Format, Result, clock};

pub(super) const NAME: &str = "Media Foundation";
pub(super) const SUPPORTED: bool = true;

// MF_SOURCE_READER_FIRST_VIDEO_STREAM is the constant 0xFFFFFFFC; the API
// takes it as a DWORD.
#[allow(clippy::cast_sign_loss)]
const VIDEO_STREAM: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
#[allow(clippy::cast_sign_loss)]
const ALL_STREAMS: u32 = MF_SOURCE_READER_ALL_STREAMS.0 as u32;

/// `CameraControl_Flags_Auto` / `VideoProcAmp_Flags_Auto`.
const FLAG_AUTO: i32 = 1;

/// `IAMCameraControl` properties (`CameraControlProperty`).
const CAMERA_PROPS: [(i32, ControlId); 7] = [
    (0, ControlId::Pan),
    (1, ControlId::Tilt),
    (2, ControlId::Roll),
    (3, ControlId::Zoom),
    (4, ControlId::Exposure),
    (5, ControlId::Iris),
    (6, ControlId::Focus),
];

/// `IAMVideoProcAmp` properties (`VideoProcAmpProperty`, plus the KS power-line
/// frequency property 13, which UVC drivers expose through the same interface).
const PROCAMP_PROPS: [(i32, ControlId); 11] = [
    (0, ControlId::Brightness),
    (1, ControlId::Contrast),
    (2, ControlId::Hue),
    (3, ControlId::Saturation),
    (4, ControlId::Sharpness),
    (5, ControlId::Gamma),
    (6, ControlId::ColorEnable),
    (7, ControlId::WhiteBalance),
    (8, ControlId::BacklightCompensation),
    (9, ControlId::Gain),
    (13, ControlId::PowerLineFrequency),
];

fn mf_error(context: &str) -> impl FnOnce(windows::core::Error) -> Error + '_ {
    move |e| Error::device(context, e)
}

/// COM + Media Foundation initialisation for the current thread.
struct Runtime {
    com: bool,
}

impl Runtime {
    fn new() -> Result<Self> {
        // SAFETY: plain initialisation calls; balanced in Drop. COM may
        // already be initialised in another mode on this thread, in which case
        // it is left alone.
        unsafe {
            let com = CoInitializeEx(None, COINIT_MULTITHREADED).is_ok();
            if let Err(e) = MFStartup(MF_VERSION, MFSTARTUP_LITE) {
                if com {
                    CoUninitialize();
                }
                return Err(Error::device(
                    "starting Media Foundation (is it installed?)",
                    e,
                ));
            }
            Ok(Self { com })
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: balances the successful calls in `Runtime::new`.
        unsafe {
            let _ = MFShutdown();
            if self.com {
                CoUninitialize();
            }
        }
    }
}

fn video_sources() -> Result<Vec<IMFActivate>> {
    // SAFETY: standard MF enumeration. The returned array is CoTaskMemAlloc'd;
    // each element is moved out with ptr::read exactly once, then the array is freed.
    unsafe {
        let mut attrs: Option<IMFAttributes> = None;
        MFCreateAttributes(&raw mut attrs, 1).map_err(mf_error("MFCreateAttributes"))?;
        let attrs = attrs
            .ok_or_else(|| Error::device("MFCreateAttributes", "no attribute store returned"))?;
        attrs
            .SetGUID(
                &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
            )
            .map_err(mf_error("setting the video-capture source type"))?;
        let mut array: *mut Option<IMFActivate> = ptr::null_mut();
        let mut count = 0u32;
        MFEnumDeviceSources(&attrs, &raw mut array, &raw mut count)
            .map_err(mf_error("listing cameras (MFEnumDeviceSources)"))?;
        let mut out = Vec::with_capacity(count as usize);
        if !array.is_null() {
            for i in 0..count as usize {
                if let Some(a) = ptr::read(array.add(i)) {
                    out.push(a);
                }
            }
            CoTaskMemFree(Some(array.cast_const().cast()));
        }
        Ok(out)
    }
}

fn string_attr(attrs: &IMFAttributes, key: &GUID) -> Option<String> {
    // SAFETY: GetAllocatedString returns a CoTaskMemAlloc'd, NUL-terminated
    // string which is copied and then freed.
    unsafe {
        let mut p = PWSTR::null();
        let mut len = 0u32;
        attrs
            .GetAllocatedString(key, &raw mut p, &raw mut len)
            .ok()?;
        let s = p.to_string().ok();
        CoTaskMemFree(Some(p.0.cast_const().cast()));
        s
    }
}

/// (id, friendly name, USB id) of an enumerated camera.
fn identify(index: usize, a: &IMFActivate) -> (String, String, Option<UsbId>) {
    let name = string_attr(a, &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME)
        .unwrap_or_else(|| format!("camera {index}"));
    let link = string_attr(a, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
    let usb = link.as_deref().and_then(UsbId::from_device_path);
    let id = link.unwrap_or_else(|| name.clone());
    (id, name, usb)
}

fn info(id: String, name: String, usb: Option<UsbId>) -> CameraInfo {
    CameraInfo {
        id,
        name,
        usb,
        formats: Vec::new(),
        controls: Vec::new(),
        exposure_unit: Some(ExposureUnit::Log2Seconds),
        probe_error: None,
    }
}

pub(super) fn list() -> Result<Vec<CameraInfo>> {
    let _rt = Runtime::new()?;
    let mut out = Vec::new();
    for (i, a) in video_sources()?.iter().enumerate() {
        let (id, name, usb) = identify(i, a);
        // Open in this process: activating it again would disturb the stream.
        if let Some(held) = super::held_info(&id) {
            out.push(held);
            continue;
        }
        let mut cam = info(id, name, usb);
        match activate(a, &cam.name) {
            Ok(dev) => {
                cam.formats = dev.formats();
                cam.controls = dev.controls();
            }
            Err(e) => cam.probe_error = Some(e.to_string()),
        }
        out.push(cam);
    }
    Ok(out)
}

pub(super) fn open(id: &str) -> Result<(Box<dyn Device>, CameraInfo)> {
    let rt = Runtime::new()?;
    let sources = video_sources()?;
    for (i, a) in sources.iter().enumerate() {
        let (cid, name, usb) = identify(i, a);
        if cid != id {
            continue;
        }
        let mut dev = activate(a, &name)?;
        let mut cam = info(cid, name, usb);
        cam.formats = dev.formats();
        cam.controls = dev.controls();
        dev.rt = Some(rt);
        return Ok((Box::new(dev), cam));
    }
    Err(Error::NotFound(id.to_owned()))
}

/// Creates the media source and a source reader for one camera.
fn activate(a: &IMFActivate, name: &str) -> Result<MfDevice> {
    let context = format!("opening `{name}` (in use by another app, or camera access blocked?)");
    // SAFETY: COM calls on valid interfaces created above.
    let (source, reader) = unsafe {
        let source: IMFMediaSource = a.ActivateObject().map_err(mf_error(&context))?;
        let mut attrs: Option<IMFAttributes> = None;
        MFCreateAttributes(&raw mut attrs, 3).map_err(mf_error("MFCreateAttributes"))?;
        let attrs = attrs
            .ok_or_else(|| Error::device("MFCreateAttributes", "no attribute store returned"))?;
        let set = |key, v| {
            attrs
                .SetUINT32(key, v)
                .map_err(mf_error("configuring the reader"))
        };
        set(&MF_READWRITE_DISABLE_CONVERTERS, 1)?;
        set(&MF_SOURCE_READER_DISABLE_DXVA, 1)?;
        set(&MF_LOW_LATENCY, 1)?;
        let reader = match MFCreateSourceReaderFromMediaSource(&source, &attrs) {
            Ok(r) => r,
            Err(e) => {
                let _ = source.Shutdown();
                return Err(Error::device("creating the source reader", e));
            }
        };
        (source, reader)
    };
    // Absence of either interface is not an error: the camera just has fewer controls.
    let (camera_control, procamp) = (
        source.cast::<IAMCameraControl>().ok(),
        source.cast::<IAMVideoProcAmp>().ok(),
    );
    let mut dev = MfDevice {
        types: Vec::new(),
        reader,
        interrupt_slot: Arc::new(Mutex::new(Some(SharedSource(source.clone())))),
        source,
        camera_control,
        procamp,
        rt: None,
    };
    dev.types = native_types(&dev.reader)?;
    Ok(dev)
}

fn fourcc_of(subtype: &GUID) -> String {
    let b = subtype.data1.to_le_bytes();
    if b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b' ') {
        return String::from_utf8_lossy(&b).trim_end().to_owned();
    }
    match subtype.data1 {
        20 => "RGB24".to_owned(),
        21 => "ARGB32".to_owned(),
        22 => "RGB32".to_owned(),
        23 => "RGB565".to_owned(),
        24 => "RGB555".to_owned(),
        50 => "L8".to_owned(),
        n => format!("0x{n:08x}"),
    }
}

fn split_u64(v: u64) -> (u32, u32) {
    // High and low halves of an MF packed attribute; truncation is the intent.
    #[allow(clippy::cast_possible_truncation)]
    ((v >> 32) as u32, v as u32)
}

fn native_types(reader: &IMFSourceReader) -> Result<Vec<(Format, IMFMediaType)>> {
    let mut out = Vec::new();
    for i in 0.. {
        // SAFETY: COM calls on a valid source reader.
        let mt = match unsafe { reader.GetNativeMediaType(VIDEO_STREAM, i) } {
            Ok(mt) => mt,
            Err(e) if e.code() == MF_E_NO_MORE_TYPES => break,
            Err(e) => return Err(Error::device("listing capture modes", e)),
        };
        // SAFETY: attribute reads on a valid media type.
        let (subtype, size, rate) = unsafe {
            (
                mt.GetGUID(&MF_MT_SUBTYPE),
                mt.GetUINT64(&MF_MT_FRAME_SIZE),
                mt.GetUINT64(&MF_MT_FRAME_RATE),
            )
        };
        let (Ok(subtype), Ok(size)) = (subtype, size) else {
            continue;
        };
        let (width, height) = split_u64(size);
        let (fps_num, fps_den) = rate.map_or((0, 1), split_u64);
        out.push((
            Format {
                fourcc: fourcc_of(&subtype),
                width,
                height,
                fps_num,
                fps_den: fps_den.max(1),
            },
            mt,
        ));
    }
    Ok(out)
}

/// The camera's media source, shared with the thread that stops a capture.
struct SharedSource(IMFMediaSource);

// SAFETY: Media Foundation requires every MF component to be thread-safe and
// media sources to be free-threaded or neutral ("Media Foundation and COM"):
// their methods may be called from any thread without marshaling, as MF's own
// work-queue threads do. The pointer is AddRef'd and released by the
// `windows` wrapper, and only `Shutdown` is called through it (see
// `MfDevice::interrupter`).
unsafe impl Send for SharedSource {}
// SAFETY: as for `Send`; calls through a shared reference are serialised by
// the `interrupt_slot` mutex anyway.
unsafe impl Sync for SharedSource {}

struct MfDevice {
    types: Vec<(Format, IMFMediaType)>,
    reader: IMFSourceReader,
    /// The source for the interrupter. Emptied in `Drop`, on the capture
    /// thread, before the source is released and MF/COM shut down there.
    interrupt_slot: Arc<Mutex<Option<SharedSource>>>,
    source: IMFMediaSource,
    camera_control: Option<IAMCameraControl>,
    procamp: Option<IAMVideoProcAmp>,
    // Declared last so COM/MF shut down after the interfaces above are released.
    rt: Option<Runtime>,
}

impl Drop for MfDevice {
    fn drop(&mut self) {
        // From here on an interrupt is a no-op: no call through the shared
        // source can outlive this device or its runtime.
        drop(
            self.interrupt_slot
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
        // SAFETY: shutting down a media source we own; errors are irrelevant here.
        unsafe {
            let _ = self.source.Shutdown();
        }
    }
}

/// A control interface: `IAMCameraControl` and `IAMVideoProcAmp` share a shape.
enum Iface<'a> {
    Camera(&'a IAMCameraControl),
    ProcAmp(&'a IAMVideoProcAmp),
}

impl Iface<'_> {
    fn range(&self, prop: i32) -> Option<[i32; 5]> {
        let mut r = [0i32; 5];
        let [min, max, step, default, caps] = &mut r;
        // SAFETY: GetRange writes five i32 outputs.
        let ok = unsafe {
            match self {
                Self::Camera(i) => i.GetRange(prop, min, max, step, default, caps),
                Self::ProcAmp(i) => i.GetRange(prop, min, max, step, default, caps),
            }
        };
        ok.ok().map(|()| r)
    }

    fn get(&self, prop: i32) -> Option<(i32, i32)> {
        let (mut v, mut flags) = (0i32, 0i32);
        // SAFETY: Get writes two i32 outputs.
        let ok = unsafe {
            match self {
                Self::Camera(i) => i.Get(prop, &raw mut v, &raw mut flags),
                Self::ProcAmp(i) => i.Get(prop, &raw mut v, &raw mut flags),
            }
        };
        ok.ok().map(|()| (v, flags))
    }
}

impl MfDevice {
    /// Every native mode, without duplicates.
    fn formats(&self) -> Vec<Format> {
        let mut formats: Vec<Format> = Vec::new();
        for (f, _) in &self.types {
            if !formats.contains(f) {
                formats.push(f.clone());
            }
        }
        formats
    }

    fn controls(&self) -> Vec<ControlInfo> {
        CAMERA_PROPS
            .iter()
            .chain(PROCAMP_PROPS.iter())
            .filter_map(|&(_, id)| self.read(id))
            .collect()
    }

    fn locate(&self, id: ControlId) -> Option<(Iface<'_>, i32)> {
        if let Some(&(p, _)) = CAMERA_PROPS.iter().find(|(_, c)| *c == id) {
            return self.camera_control.as_ref().map(|i| (Iface::Camera(i), p));
        }
        if let Some(&(p, _)) = PROCAMP_PROPS.iter().find(|(_, c)| *c == id) {
            return self.procamp.as_ref().map(|i| (Iface::ProcAmp(i), p));
        }
        None
    }

    fn read(&self, id: ControlId) -> Option<ControlInfo> {
        let (iface, prop) = self.locate(id)?;
        let [min, max, step, default, caps] = iface.range(prop)?;
        let current = iface.get(prop);
        let auto_supported = caps & FLAG_AUTO != 0;
        Some(ControlInfo {
            id,
            name: id.common_name().unwrap_or("unknown").to_owned(),
            min: min.into(),
            max: max.into(),
            step: step.into(),
            default: default.into(),
            current: current.map(|(v, _)| v.into()),
            auto_supported,
            auto: current
                .filter(|_| auto_supported)
                .map(|(_, f)| f & FLAG_AUTO != 0),
            menu: Vec::new(),
        })
    }
}

impl Device for MfDevice {
    fn start(&mut self, format: &Format) -> Result<()> {
        let mt = self
            .types
            .iter()
            .find(|(f, _)| f == format)
            .map(|(_, mt)| mt.clone())
            .ok_or_else(|| {
                Error::InvalidArgument(format!("{format} is not a native mode of this camera"))
            })?;
        // SAFETY: COM calls on a valid source reader and media type.
        unsafe {
            let _ = self.reader.SetStreamSelection(ALL_STREAMS, false);
            self.reader
                .SetStreamSelection(VIDEO_STREAM, true)
                .map_err(mf_error("selecting the video stream"))?;
            self.reader
                .SetCurrentMediaType(VIDEO_STREAM, None, &mt)
                .map_err(|e| Error::device(format!("selecting {format}"), e))?;
        }
        Ok(())
    }

    /// Shuts the media source down from any thread. A `ReadSample` pending on
    /// the capture thread then fails (`MF_E_SHUTDOWN`), ending the read, and
    /// the camera is released at once.
    fn interrupter(&self) -> Option<Interrupter> {
        let slot = Arc::clone(&self.interrupt_slot);
        Some(Interrupter::new(move || {
            // Held during the call, so `MfDevice::drop` (which empties the
            // slot first) can't release the source or shut MF down meanwhile.
            let source = slot.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(source) = source.as_ref() else {
                return;
            };
            // SAFETY: `SharedSource` explains why this call may come from any
            // thread. COM is joined to the MTA for the call if the thread has
            // no apartment yet, and left balanced; a thread that is already in
            // an apartment (even an STA) calls the free-threaded source
            // directly.
            unsafe {
                let com = CoInitializeEx(None, COINIT_MULTITHREADED).is_ok();
                let _ = source.0.Shutdown();
                if com {
                    CoUninitialize();
                }
            }
        }))
    }

    /// Synchronous read: waits for the camera whatever the timeout. A gap in
    /// the stream (no sample) returns `None`.
    fn read_frame(&mut self, _timeout: Duration) -> Result<Option<RawFrame>> {
        let mut flags = 0u32;
        let mut ts = 0i64;
        let mut sample: Option<IMFSample> = None;
        // SAFETY: synchronous ReadSample with valid out-pointers.
        unsafe {
            self.reader
                .ReadSample(
                    VIDEO_STREAM,
                    0,
                    None,
                    Some(&raw mut flags),
                    Some(&raw mut ts),
                    Some(&raw mut sample),
                )
                .map_err(mf_error("reading a frame"))?;
        }
        let host_ns = clock::now_ns();
        #[allow(clippy::cast_sign_loss)]
        {
            if flags & MF_SOURCE_READERF_ERROR.0 as u32 != 0 {
                return Err(Error::device(
                    "reading a frame",
                    "the camera reported a stream error",
                ));
            }
            if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                return Err(Error::device(
                    "reading a frame",
                    "the camera stopped streaming",
                ));
            }
            // Frames would carry the old size; reopen the camera instead.
            if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
                return Err(Error::device(
                    "reading a frame",
                    "the camera changed its video mode",
                ));
            }
        }
        let Some(sample) = sample else {
            return Ok(None);
        };
        // SAFETY: the buffer is locked while its bytes are copied, then unlocked.
        let data: Arc<[u8]> = unsafe {
            let buffer = sample
                .ConvertToContiguousBuffer()
                .map_err(mf_error("reading the sample buffer"))?;
            let mut p: *mut u8 = ptr::null_mut();
            let mut len = 0u32;
            buffer
                .Lock(&raw mut p, None, Some(&raw mut len))
                .map_err(mf_error("locking the sample buffer"))?;
            let data: Arc<[u8]> = if p.is_null() {
                Arc::from(Vec::new())
            } else {
                Arc::from(std::slice::from_raw_parts(p, len as usize))
            };
            let _ = buffer.Unlock();
            data
        };
        // SAFETY: attribute read on a valid sample; absent on many drivers.
        let device_ns = unsafe { sample.GetUINT64(&MFSampleExtension_DeviceReferenceSystemTime) }
            .ok()
            .map(|v| v.saturating_mul(100));
        Ok(Some(RawFrame {
            data,
            host_ns,
            device_ns,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourccs() {
        let mjpg = GUID::from_u128(0x4750_4a4d_0000_0010_8000_00aa_0038_9b71);
        assert_eq!(fourcc_of(&mjpg), "MJPG");
        let rgb24 = GUID::from_u128(0x0000_0014_0000_0010_8000_00aa_0038_9b71);
        assert_eq!(fourcc_of(&rgb24), "RGB24");
    }
}
