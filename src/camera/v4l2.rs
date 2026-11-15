//! Linux backend: V4L2 through raw ioctls (no bindgen, no libv4l).
//!
//! Streaming uses MMAP buffers. Each frame is copied out of the driver buffer
//! untouched and stamped with `CLOCK_MONOTONIC` right after `VIDIOC_DQBUF`;
//! buffers the driver marks as failed (`V4L2_BUF_FLAG_ERROR`, a USB transfer
//! error) or empty are re-queued and skipped;
//! the driver's buffer timestamp becomes the device time when the driver says
//! it is monotonic.

// The ioctl structs mirror <linux/videodev2.h> for 64-bit targets (this
// backend is only built there); their sizes are checked in the tests.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::ffi::CString;
use std::fs;
use std::io;
use std::os::raw::{c_int, c_ulong, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::{CameraInfo, Device, RawFrame, UsbId};
use crate::{Error, Format, Result, clock};

pub(super) const SUPPORTED: bool = true;

const BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const MEMORY_MMAP: u32 = 1;
const CAP_VIDEO_CAPTURE: u32 = 0x1;
const CAP_STREAMING: u32 = 0x0400_0000;
const CAP_DEVICE_CAPS: u32 = 0x8000_0000;
const FRMSIZE_TYPE_DISCRETE: u32 = 1;
const FRMIVAL_TYPE_DISCRETE: u32 = 1;
const BUFFER_COUNT: u32 = 6;
const BUF_FLAG_ERROR: u32 = 0x0000_0040;
const BUF_FLAG_TIMESTAMP_MASK: u32 = 0x0000_e000;
const BUF_FLAG_TIMESTAMP_MONOTONIC: u32 = 0x0000_2000;
const FIELD_NONE: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FmtDesc {
    index: u32,
    type_: u32,
    flags: u32,
    description: [u8; 32],
    pixelformat: u32,
    mbus_code: u32,
    reserved: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrmSizeEnum {
    index: u32,
    pixel_format: u32,
    type_: u32,
    /// Discrete: `[width, height]`; stepwise: `[min_w, max_w, step_w, min_h, max_h, step_h]`.
    u: [u32; 6],
    reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrmIvalEnum {
    index: u32,
    pixel_format: u32,
    width: u32,
    height: u32,
    type_: u32,
    /// Discrete: `[num, den]`; stepwise: `[min_num, min_den, max_num, max_den, step_num, step_den]`.
    u: [u32; 6],
    reserved: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
union FmtUnion {
    pix: PixFormat,
    raw: [u8; 200],
    // The kernel union contains pointers, which makes it 8-byte aligned.
    _align: [u64; 25],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2Format {
    type_: u32,
    fmt: FmtUnion,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CaptureParm {
    capability: u32,
    capturemode: u32,
    timeperframe_num: u32,
    timeperframe_den: u32,
    extendedmode: u32,
    readbuffers: u32,
    reserved: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
union ParmUnion {
    capture: CaptureParm,
    raw: [u8; 200],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StreamParm {
    type_: u32,
    parm: ParmUnion,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    reserved: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Timecode {
    type_: u32,
    flags: u32,
    frames: u8,
    seconds: u8,
    minutes: u8,
    hours: u8,
    userbits: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
union BufM {
    offset: u32,
    userptr: c_ulong,
    planes: *mut c_void,
    fd: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    timestamp: libc::timeval,
    timecode: Timecode,
    sequence: u32,
    memory: u32,
    m: BufM,
    length: u32,
    reserved2: u32,
    request_fd: i32,
}

const fn ioc(dir: u32, nr: u32, size: usize) -> c_ulong {
    ((dir << 30) | ((size as u32) << 16) | ((b'V' as u32) << 8) | nr) as c_ulong
}
const R: u32 = 2;
const W: u32 = 1;
const RW: u32 = 3;
const VIDIOC_QUERYCAP: c_ulong = ioc(R, 0, size_of::<Capability>());
const VIDIOC_ENUM_FMT: c_ulong = ioc(RW, 2, size_of::<FmtDesc>());
const VIDIOC_S_FMT: c_ulong = ioc(RW, 5, size_of::<V4l2Format>());
const VIDIOC_REQBUFS: c_ulong = ioc(RW, 8, size_of::<RequestBuffers>());
const VIDIOC_QUERYBUF: c_ulong = ioc(RW, 9, size_of::<Buffer>());
const VIDIOC_QBUF: c_ulong = ioc(RW, 15, size_of::<Buffer>());
const VIDIOC_DQBUF: c_ulong = ioc(RW, 17, size_of::<Buffer>());
const VIDIOC_STREAMON: c_ulong = ioc(W, 18, size_of::<c_int>());
const VIDIOC_STREAMOFF: c_ulong = ioc(W, 19, size_of::<c_int>());
const VIDIOC_S_PARM: c_ulong = ioc(RW, 22, size_of::<StreamParm>());
const VIDIOC_ENUM_FRAMESIZES: c_ulong = ioc(RW, 74, size_of::<FrmSizeEnum>());
const VIDIOC_ENUM_FRAMEINTERVALS: c_ulong = ioc(RW, 75, size_of::<FrmIvalEnum>());

/// An open device node.
struct Fd(c_int);

impl Fd {
    fn open(path: &Path) -> io::Result<Self> {
        let c = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "device path contains NUL"))?;
        // SAFETY: valid NUL-terminated path.
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(fd))
    }

    /// Issues an ioctl, retrying on EINTR.
    fn ioctl<T>(&self, req: c_ulong, arg: &mut T) -> io::Result<()> {
        loop {
            // SAFETY: `arg` is a correctly sized struct for `req` (sizes are
            // encoded in the request number and checked by the kernel).
            let r =
                unsafe { libc::ioctl(self.0, req as _, std::ptr::from_mut(arg).cast::<c_void>()) };
            if r >= 0 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            if e.raw_os_error() != Some(libc::EINTR) {
                return Err(e);
            }
        }
    }

    /// Waits until a frame can be dequeued. `false` on timeout.
    fn wait_readable(&self, timeout: Duration) -> io::Result<bool> {
        let ms = c_int::try_from(timeout.as_millis()).unwrap_or(c_int::MAX);
        let mut pfd = libc::pollfd {
            fd: self.0,
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            // SAFETY: one valid pollfd, count 1.
            let r = unsafe { libc::poll(&raw mut pfd, 1, ms) };
            if r > 0 {
                if pfd.revents & libc::POLLIN != 0 {
                    return Ok(true);
                }
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "the camera was disconnected",
                ));
            }
            if r == 0 {
                return Ok(false);
            }
            let e = io::Error::last_os_error();
            if e.raw_os_error() != Some(libc::EINTR) {
                return Err(e);
            }
        }
    }
}

impl Drop for Fd {
    fn drop(&mut self) {
        // SAFETY: closing a descriptor we own.
        unsafe {
            libc::close(self.0);
        }
    }
}

fn zeroed<T: Copy>() -> T {
    // SAFETY: only used for the plain-old-data ioctl structs above, for which
    // all-zero bytes are valid.
    unsafe { std::mem::zeroed() }
}

fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).trim().to_owned()
}

fn fourcc_str(v: u32) -> String {
    String::from_utf8_lossy(&v.to_le_bytes())
        .trim_end()
        .to_owned()
}

fn fourcc_code(s: &str) -> u32 {
    let mut b = *b"    ";
    for (o, c) in b.iter_mut().zip(s.bytes()) {
        *o = c;
    }
    u32::from_le_bytes(b)
}

/// A `/dev/video*` node that can stream video.
struct Node {
    path: PathBuf,
    card: String,
    bus_info: String,
}

/// Capture nodes in numeric order.
fn capture_nodes() -> Vec<Node> {
    let mut nodes: Vec<(u32, PathBuf)> = fs::read_dir("/dev")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let n: u32 = name.strip_prefix("video")?.parse().ok()?;
            Some((n, e.path()))
        })
        .collect();
    nodes.sort();
    nodes
        .into_iter()
        .filter_map(|(_, path)| {
            let fd = Fd::open(&path).ok()?;
            let mut cap: Capability = zeroed();
            fd.ioctl(VIDIOC_QUERYCAP, &mut cap).ok()?;
            let caps = if cap.capabilities & CAP_DEVICE_CAPS != 0 {
                cap.device_caps
            } else {
                cap.capabilities
            };
            (caps & CAP_VIDEO_CAPTURE != 0 && caps & CAP_STREAMING != 0).then(|| Node {
                path,
                card: cstr(&cap.card),
                bus_info: cstr(&cap.bus_info),
            })
        })
        .collect()
}

/// `/dev/v4l/by-id` links as (resolved node, link).
fn by_id_links() -> Vec<(PathBuf, PathBuf)> {
    let mut links: Vec<(PathBuf, PathBuf)> = fs::read_dir("/dev/v4l/by-id")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| Some((fs::canonicalize(e.path()).ok()?, e.path())))
        .collect();
    links.sort();
    links
}

/// The stable id of each node: its by-id link, unless another capture node
/// has the same card name (identical cameras without serial numbers share one
/// by-id name, so the link can point at either); then `<bus_info>/<card>`,
/// which is unique per USB port.
fn node_ids(nodes: &[Node]) -> Vec<String> {
    let links = by_id_links();
    nodes
        .iter()
        .map(|n| {
            let twin = nodes.iter().filter(|o| o.card == n.card).nth(1).is_some();
            let link = fs::canonicalize(&n.path)
                .ok()
                .and_then(|real| links.iter().find(|(target, _)| *target == real))
                .map(|(_, link)| link.to_string_lossy().into_owned());
            match link {
                Some(link) if !twin => link,
                _ => format!("{}/{}", n.bus_info, n.card),
            }
        })
        .collect()
}

/// USB vid:pid from sysfs: /sys/class/video4linux/videoN/device is the USB
/// interface; its parent directory holds idVendor/idProduct.
fn usb_id(node: &Path) -> Option<UsbId> {
    let name = node.file_name()?.to_str()?;
    let dev = fs::canonicalize(format!("/sys/class/video4linux/{name}/device")).ok()?;
    let parent = dev.parent()?;
    let read = |f: &str| {
        fs::read_to_string(parent.join(f))
            .ok()
            .and_then(|s| u16::from_str_radix(s.trim(), 16).ok())
    };
    Some(UsbId {
        vid: read("idVendor")?,
        pid: read("idProduct")?,
    })
}

/// Reads a node's modes through an open descriptor.
fn describe(node: &Node, id: String, fd: Result<&Fd, String>) -> CameraInfo {
    let mut info = CameraInfo {
        id,
        name: node.card.clone(),
        usb: usb_id(&node.path),
        formats: Vec::new(),
        probe_error: None,
    };
    match fd {
        Ok(fd) => info.formats = list_formats(fd),
        Err(e) => info.probe_error = Some(e),
    }
    info
}

// Shares the backend signature; V4L2 enumeration itself can't fail.
#[allow(clippy::unnecessary_wraps)]
pub(super) fn list() -> Result<Vec<CameraInfo>> {
    let nodes = capture_nodes();
    let ids = node_ids(&nodes);
    Ok(nodes
        .iter()
        .zip(ids)
        .map(|(n, id)| {
            // Open in this process: its entry is the info read at open.
            if let Some(held) = super::held_info(&id) {
                return held;
            }
            let fd = Fd::open(&n.path).map_err(|e| format!("opening {}: {e}", n.path.display()));
            describe(n, id, fd.as_ref().map_err(Clone::clone))
        })
        .collect())
}

pub(super) fn open(id: &str) -> Result<(Box<dyn Device>, CameraInfo)> {
    let nodes = capture_nodes();
    let ids = node_ids(&nodes);
    // The id itself, or any path that resolves to the node (`/dev/video0`).
    let wanted = fs::canonicalize(id).ok();
    let (node, node_id) = nodes
        .iter()
        .zip(ids)
        .find(|(n, nid)| {
            nid == id
                || wanted
                    .as_ref()
                    .is_some_and(|w| fs::canonicalize(&n.path).ok().as_ref() == Some(w))
        })
        .ok_or_else(|| Error::NotFound(id.to_owned()))?;
    let fd = Fd::open(&node.path).map_err(|e| {
        Error::device(
            format!("opening `{}` ({})", node.card, node.path.display()),
            e,
        )
    })?;
    let info = describe(node, node_id, Ok(&fd));
    Ok((
        Box::new(V4l2Device {
            fd,
            buffers: Vec::new(),
            streaming: false,
        }),
        info,
    ))
}

fn list_formats(fd: &Fd) -> Vec<Format> {
    let mut out = Vec::new();
    for i in 0.. {
        let mut d: FmtDesc = zeroed();
        d.index = i;
        d.type_ = BUF_TYPE_VIDEO_CAPTURE;
        if fd.ioctl(VIDIOC_ENUM_FMT, &mut d).is_err() {
            break;
        }
        let fourcc = fourcc_str(d.pixelformat);
        for (w, h) in frame_sizes(fd, d.pixelformat) {
            for (num, den) in frame_rates(fd, d.pixelformat, w, h) {
                out.push(Format {
                    fourcc: fourcc.clone(),
                    width: w,
                    height: h,
                    fps_num: num,
                    fps_den: den,
                });
            }
        }
    }
    out
}

fn frame_sizes(fd: &Fd, pixfmt: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for i in 0.. {
        let mut s: FrmSizeEnum = zeroed();
        s.index = i;
        s.pixel_format = pixfmt;
        if fd.ioctl(VIDIOC_ENUM_FRAMESIZES, &mut s).is_err() {
            break;
        }
        if s.type_ == FRMSIZE_TYPE_DISCRETE {
            out.push((s.u[0], s.u[1]));
        } else {
            // Stepwise/continuous: offer the largest size.
            out.push((s.u[1], s.u[4]));
            break;
        }
    }
    out
}

/// Frame rates as `(fps_num, fps_den)`, i.e. the inverse of the frame interval.
fn frame_rates(fd: &Fd, pixfmt: u32, w: u32, h: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for i in 0.. {
        let mut v: FrmIvalEnum = zeroed();
        v.index = i;
        v.pixel_format = pixfmt;
        v.width = w;
        v.height = h;
        if fd.ioctl(VIDIOC_ENUM_FRAMEINTERVALS, &mut v).is_err() {
            break;
        }
        // Discrete: the interval. Stepwise/continuous: the shortest interval,
        // i.e. the highest rate.
        let (num, den) = (v.u[0], v.u[1]);
        if num > 0 && den > 0 {
            out.push((den, num));
        }
        if v.type_ != FRMIVAL_TYPE_DISCRETE {
            break;
        }
    }
    out
}

struct MappedBuffer {
    ptr: *mut c_void,
    len: usize,
}

struct V4l2Device {
    fd: Fd,
    buffers: Vec<MappedBuffer>,
    streaming: bool,
}

impl Drop for V4l2Device {
    fn drop(&mut self) {
        if self.streaming {
            let mut t = BUF_TYPE_VIDEO_CAPTURE as c_int;
            let _ = self.fd.ioctl(VIDIOC_STREAMOFF, &mut t);
        }
        for b in &self.buffers {
            // SAFETY: unmapping regions returned by mmap in `start`.
            unsafe {
                libc::munmap(b.ptr, b.len);
            }
        }
    }
}

impl Device for V4l2Device {
    fn start(&mut self, format: &Format) -> Result<()> {
        if self.streaming {
            return Err(Error::InvalidArgument(
                "the camera is already streaming".into(),
            ));
        }
        let mut fmt: V4l2Format = zeroed();
        fmt.type_ = BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix = PixFormat {
            width: format.width,
            height: format.height,
            pixelformat: fourcc_code(&format.fourcc),
            field: FIELD_NONE,
            ..zeroed()
        };
        self.fd
            .ioctl(VIDIOC_S_FMT, &mut fmt)
            .map_err(|e| Error::device(format!("selecting {format}"), e))?;
        // SAFETY: the kernel filled `pix` for a video-capture format.
        let (w, h, pf) = unsafe {
            (
                fmt.fmt.pix.width,
                fmt.fmt.pix.height,
                fmt.fmt.pix.pixelformat,
            )
        };
        if (w, h, pf) != (format.width, format.height, fourcc_code(&format.fourcc)) {
            return Err(Error::device(
                format!("selecting {format}"),
                format!("the driver chose {} {w}x{h}", fourcc_str(pf)),
            ));
        }
        let mut parm: StreamParm = zeroed();
        parm.type_ = BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture = CaptureParm {
            timeperframe_num: format.fps_den,
            timeperframe_den: format.fps_num,
            ..zeroed()
        };
        self.fd
            .ioctl(VIDIOC_S_PARM, &mut parm)
            .map_err(|e| Error::device("setting the frame rate", e))?;

        let mut req = RequestBuffers {
            count: BUFFER_COUNT,
            type_: BUF_TYPE_VIDEO_CAPTURE,
            memory: MEMORY_MMAP,
            capabilities: 0,
            flags: 0,
            reserved: [0; 3],
        };
        self.fd
            .ioctl(VIDIOC_REQBUFS, &mut req)
            .map_err(|e| Error::device("requesting capture buffers", e))?;
        for i in 0..req.count {
            let mut b: Buffer = zeroed();
            b.index = i;
            b.type_ = BUF_TYPE_VIDEO_CAPTURE;
            b.memory = MEMORY_MMAP;
            self.fd
                .ioctl(VIDIOC_QUERYBUF, &mut b)
                .map_err(|e| Error::device("querying a capture buffer", e))?;
            // SAFETY: `m.offset` is the active union member for MMAP buffers.
            let offset = unsafe { b.m.offset };
            // SAFETY: mapping the driver buffer at the offset it reported.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    b.length as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd.0,
                    libc::off_t::from(offset),
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(Error::device(
                    "mapping a capture buffer",
                    io::Error::last_os_error(),
                ));
            }
            self.buffers.push(MappedBuffer {
                ptr,
                len: b.length as usize,
            });
            self.fd
                .ioctl(VIDIOC_QBUF, &mut b)
                .map_err(|e| Error::device("queueing a capture buffer", e))?;
        }
        let mut t = BUF_TYPE_VIDEO_CAPTURE as c_int;
        self.fd
            .ioctl(VIDIOC_STREAMON, &mut t)
            .map_err(|e| Error::device("starting the stream", e))?;
        self.streaming = true;
        Ok(())
    }

    fn read_frame(&mut self, timeout: Duration) -> Result<Option<RawFrame>> {
        if !self.streaming {
            return Err(Error::InvalidArgument("the stream is not started".into()));
        }
        if !self
            .fd
            .wait_readable(timeout)
            .map_err(|e| Error::device("waiting for a frame", e))?
        {
            return Ok(None);
        }
        let mut b: Buffer = zeroed();
        b.type_ = BUF_TYPE_VIDEO_CAPTURE;
        b.memory = MEMORY_MMAP;
        self.fd
            .ioctl(VIDIOC_DQBUF, &mut b)
            .map_err(|e| Error::device("reading a frame", e))?;
        let host_ns = clock::now_ns();
        let Some(buf) = self.buffers.get(b.index as usize) else {
            return Err(Error::device(
                "reading a frame",
                "the driver returned an unknown buffer",
            ));
        };
        if b.flags & BUF_FLAG_ERROR != 0 || b.bytesused == 0 {
            // A failed transfer or an empty buffer: not a frame.
            self.fd
                .ioctl(VIDIOC_QBUF, &mut b)
                .map_err(|e| Error::device("re-queueing a capture buffer", e))?;
            return Ok(None);
        }
        let len = (b.bytesused as usize).min(buf.len);
        // SAFETY: the mapped buffer is `buf.len` bytes and owned by us until re-queued.
        let data: Arc<[u8]> =
            Arc::from(unsafe { std::slice::from_raw_parts(buf.ptr.cast::<u8>(), len) });
        let device_ns = (b.flags & BUF_FLAG_TIMESTAMP_MASK == BUF_FLAG_TIMESTAMP_MONOTONIC)
            .then(|| clock::timespec_ns(b.timestamp.tv_sec, b.timestamp.tv_usec * 1000));
        self.fd
            .ioctl(VIDIOC_QBUF, &mut b)
            .map_err(|e| Error::device("re-queueing a capture buffer", e))?;
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
    fn struct_sizes_match_the_kernel() {
        assert_eq!(size_of::<Capability>(), 104);
        assert_eq!(size_of::<FmtDesc>(), 64);
        assert_eq!(size_of::<FrmSizeEnum>(), 44);
        assert_eq!(size_of::<FrmIvalEnum>(), 52);
        assert_eq!(size_of::<V4l2Format>(), 208);
        assert_eq!(size_of::<StreamParm>(), 204);
        assert_eq!(size_of::<RequestBuffers>(), 20);
        assert_eq!(size_of::<Buffer>(), 88);
    }

    #[test]
    fn ioctl_numbers_match_the_kernel() {
        // Values from <linux/videodev2.h> on 64-bit targets.
        assert_eq!(VIDIOC_QUERYCAP, 0x8068_5600);
        assert_eq!(VIDIOC_ENUM_FMT, 0xc040_5602);
        assert_eq!(VIDIOC_S_FMT, 0xc0d0_5605);
        assert_eq!(VIDIOC_REQBUFS, 0xc014_5608);
        assert_eq!(VIDIOC_QUERYBUF, 0xc058_5609);
        assert_eq!(VIDIOC_QBUF, 0xc058_560f);
        assert_eq!(VIDIOC_DQBUF, 0xc058_5611);
        assert_eq!(VIDIOC_STREAMON, 0x4004_5612);
        assert_eq!(VIDIOC_S_PARM, 0xc0cc_5616);
        assert_eq!(VIDIOC_ENUM_FRAMESIZES, 0xc02c_564a);
        assert_eq!(VIDIOC_ENUM_FRAMEINTERVALS, 0xc034_564b);
    }

    #[test]
    fn fourcc_round_trip() {
        assert_eq!(fourcc_str(fourcc_code("MJPG")), "MJPG");
        assert_eq!(fourcc_code("YUYV"), 0x5659_5559);
    }

    #[test]
    fn enumeration_without_cameras_is_ok() {
        // CI and the build container have no cameras; this must not fail.
        assert!(list().is_ok());
        assert!(matches!(open("missing"), Err(Error::NotFound(_))));
    }
}
