//! MJPEG AVI files: the clip writer, a reader, and playback as a [`Source`].
//!
//! The writer stores each frame's bytes untouched as a `00dc` chunk, then
//! writes a legacy `idx1` index and patches the header counts on finish: MJPEG
//! in a plain AVI 1.0 file (one video stream, every frame a key frame).
//!
//! The reader treats files as untrusted: every size read from a file is
//! clamped to the bytes the file actually holds before anything is
//! allocated, so a truncated, half-written or hostile file returns an error
//! (or fewer frames) instead of exhausting memory or panicking.
//!
//! Layout:
//! ```text
//! RIFF 'AVI '
//!   LIST 'hdrl'
//!     avih (MainAVIHeader, 56 bytes)
//!     LIST 'strl'
//!       strh (AVIStreamHeader, 56 bytes)
//!       strf (BITMAPINFOHEADER, 40 bytes)
//!   LIST 'movi'
//!     00dc <frame> ...
//!   idx1 (16 bytes per frame)
//! ```

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::sidecar::Sidecar;
use crate::{CameraInfo, Error, Format, Frame, Result, Source, clock};

/// Largest file the writer produces: inside the 32-bit RIFF sizes and the
/// 2 GiB limit of older players.
const FILE_LIMIT: u64 = (1 << 31) - 1;

const AVIF_HASINDEX: u32 = 0x10;
const AVIIF_KEYFRAME: u32 = 0x10;

// Byte offsets of the fields patched on finish. The header layout is fixed.
const OFF_RIFF_SIZE: u64 = 4;
const OFF_AVIH: u64 = 32;
const OFF_AVIH_MAX_BYTES_PER_SEC: u64 = OFF_AVIH + 4;
const OFF_AVIH_TOTAL_FRAMES: u64 = OFF_AVIH + 16;
const OFF_AVIH_SUGGESTED_BUFFER: u64 = OFF_AVIH + 28;
const OFF_STRH: u64 = 108;
const OFF_STRH_LENGTH: u64 = OFF_STRH + 32;
const OFF_STRH_SUGGESTED_BUFFER: u64 = OFF_STRH + 36;
const OFF_MOVI_SIZE: u64 = 216;
/// Position of the `movi` fourcc; `idx1` offsets are relative to it.
const OFF_MOVI_FOURCC: u64 = 220;
const HEADER_LEN: u64 = 224;
/// Most `hdrl` bytes the reader parses. Real headers are a few hundred bytes;
/// the cap bounds the allocation for a corrupt size field.
const HDRL_MAX: u64 = 1 << 20;

/// Stream parameters for a new AVI file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AviParams {
    pub fourcc: [u8; 4],
    pub width: u32,
    pub height: u32,
    /// Frame rate as `fps_num / fps_den`.
    pub fps_num: u32,
    pub fps_den: u32,
}

/// Writes frames to an AVI file.
pub(crate) struct AviWriter {
    out: BufWriter<File>,
    path: PathBuf,
    params: AviParams,
    /// (offset relative to the `movi` fourcc, payload size) per frame.
    index: Vec<(u32, u32)>,
    /// Current end of the `movi` data (absolute file offset).
    pos: u64,
    max_frame: u32,
}

impl AviWriter {
    /// Creates `path` and writes the headers.
    pub fn create(path: &Path, params: &AviParams) -> Result<Self> {
        if params.width == 0 || params.height == 0 {
            return Err(Error::InvalidArgument(
                "AVI frame size must be non-zero".into(),
            ));
        }
        if params.fps_num == 0 || params.fps_den == 0 {
            return Err(Error::InvalidArgument(
                "AVI frame rate must be non-zero".into(),
            ));
        }
        let file =
            File::create(path).map_err(|e| Error::io(format!("creating {}", path.display()), e))?;
        let mut w = Self {
            out: BufWriter::with_capacity(1 << 20, file),
            path: path.to_owned(),
            params: params.clone(),
            index: Vec::new(),
            pos: HEADER_LEN,
            max_frame: 0,
        };
        let header = w.header();
        w.out
            .write_all(&header)
            .map_err(|e| Error::io(format!("writing {}", path.display()), e))?;
        Ok(w)
    }

    fn header(&self) -> Vec<u8> {
        let p = &self.params;
        let micros_per_frame =
            u32::try_from(u64::from(p.fps_den) * 1_000_000 / u64::from(p.fps_num))
                .unwrap_or(u32::MAX);
        let width = i32::try_from(p.width).unwrap_or(i32::MAX);
        let height = i32::try_from(p.height).unwrap_or(i32::MAX);
        let image_size =
            u32::try_from(u64::from(p.width) * u64::from(p.height) * 3).unwrap_or(u32::MAX);
        let rect_w = i16::try_from(p.width).unwrap_or(i16::MAX);
        let rect_h = i16::try_from(p.height).unwrap_or(i16::MAX);

        let mut h: Vec<u8> = Vec::with_capacity(256);
        h.extend_from_slice(b"RIFF");
        put_u32(&mut h, 0); // patched
        h.extend_from_slice(b"AVI ");
        h.extend_from_slice(b"LIST");
        put_u32(&mut h, 192);
        h.extend_from_slice(b"hdrl");
        // avih
        h.extend_from_slice(b"avih");
        put_u32(&mut h, 56);
        put_u32(&mut h, micros_per_frame);
        put_u32(&mut h, 0); // max bytes/sec, patched
        put_u32(&mut h, 0); // padding granularity
        put_u32(&mut h, AVIF_HASINDEX);
        put_u32(&mut h, 0); // total frames, patched
        put_u32(&mut h, 0); // initial frames
        put_u32(&mut h, 1); // streams
        put_u32(&mut h, 0); // suggested buffer, patched
        put_u32(&mut h, p.width);
        put_u32(&mut h, p.height);
        h.extend_from_slice(&[0; 16]); // reserved
        // strl
        h.extend_from_slice(b"LIST");
        put_u32(&mut h, 116);
        h.extend_from_slice(b"strl");
        h.extend_from_slice(b"strh");
        put_u32(&mut h, 56);
        h.extend_from_slice(b"vids");
        h.extend_from_slice(&p.fourcc);
        put_u32(&mut h, 0); // flags
        h.extend_from_slice(&0u16.to_le_bytes()); // priority
        h.extend_from_slice(&0u16.to_le_bytes()); // language
        put_u32(&mut h, 0); // initial frames
        put_u32(&mut h, p.fps_den); // scale
        put_u32(&mut h, p.fps_num); // rate
        put_u32(&mut h, 0); // start
        put_u32(&mut h, 0); // length, patched
        put_u32(&mut h, 0); // suggested buffer, patched
        put_u32(&mut h, u32::MAX); // quality: default
        put_u32(&mut h, 0); // sample size: variable
        for v in [0, 0, rect_w, rect_h] {
            h.extend_from_slice(&v.to_le_bytes());
        }
        h.extend_from_slice(b"strf");
        put_u32(&mut h, 40);
        put_u32(&mut h, 40);
        h.extend_from_slice(&width.to_le_bytes());
        h.extend_from_slice(&height.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes()); // planes
        h.extend_from_slice(&24u16.to_le_bytes()); // bit count
        h.extend_from_slice(&p.fourcc);
        put_u32(&mut h, image_size);
        h.extend_from_slice(&[0; 16]); // pels/meter x/y, colors used/important
        h.extend_from_slice(b"LIST");
        put_u32(&mut h, 0); // patched
        h.extend_from_slice(b"movi");
        h
    }

    /// Appends one frame, bytes untouched. Returns the absolute file offset
    /// of the frame's data.
    pub fn write_frame(&mut self, data: &[u8]) -> Result<u64> {
        let len = u32::try_from(data.len())
            .map_err(|_| Error::InvalidArgument("frame larger than 4 GiB".into()))?;
        let chunk = 8 + u64::from(len) + u64::from(len & 1);
        let index_after = 8 + 16 * (self.index.len() as u64 + 1);
        if self.pos + chunk + index_after > FILE_LIMIT {
            return Err(Error::InvalidArgument(
                "clip too large for an AVI 1.0 file (2 GiB)".into(),
            ));
        }
        let rel = u32::try_from(self.pos - OFF_MOVI_FOURCC)
            .map_err(|_| Error::InvalidArgument("AVI offset overflow".into()))?;
        let offset = self.pos + 8;
        let io = |e| Error::io(format!("writing {}", self.path.display()), e);
        self.out.write_all(b"00dc").map_err(io)?;
        self.out.write_all(&len.to_le_bytes()).map_err(io)?;
        self.out.write_all(data).map_err(io)?;
        if len & 1 == 1 {
            self.out.write_all(&[0]).map_err(io)?;
        }
        self.index.push((rel, len));
        self.pos += chunk;
        self.max_frame = self.max_frame.max(len);
        Ok(offset)
    }

    /// Writes the index, patches the headers, flushes and syncs the file.
    pub fn finish(mut self) -> Result<()> {
        let path = self.path.clone();
        self.finish_inner()
            .map_err(|e| Error::io(format!("finishing {}", path.display()), e))
    }

    fn finish_inner(&mut self) -> std::io::Result<()> {
        let frames = self.index.len() as u64;
        let mut idx = Vec::with_capacity(8 + 16 * self.index.len());
        idx.extend_from_slice(b"idx1");
        put_u32(&mut idx, u32::try_from(16 * frames).unwrap_or(u32::MAX));
        for &(off, len) in &self.index {
            idx.extend_from_slice(b"00dc");
            put_u32(&mut idx, AVIIF_KEYFRAME);
            put_u32(&mut idx, off);
            put_u32(&mut idx, len);
        }
        let movi_end = self.pos;
        self.out.write_all(&idx)?;
        let file_len = movi_end + idx.len() as u64;

        let p = &self.params;
        // Peak rate if every frame were the largest one.
        let max_bps = u64::from(self.max_frame) * u64::from(p.fps_num) / u64::from(p.fps_den);
        let patches = [
            (OFF_RIFF_SIZE, file_len - 8),
            (OFF_MOVI_SIZE, movi_end - OFF_MOVI_FOURCC),
            (OFF_AVIH_MAX_BYTES_PER_SEC, max_bps),
            (OFF_AVIH_TOTAL_FRAMES, frames),
            // The largest chunk, header included: enough for readers that
            // read whole chunks.
            (OFF_AVIH_SUGGESTED_BUFFER, u64::from(self.max_frame) + 8),
            (OFF_STRH_LENGTH, frames),
            (OFF_STRH_SUGGESTED_BUFFER, u64::from(self.max_frame) + 8),
        ];
        for (off, value) in patches {
            self.out.seek(SeekFrom::Start(off))?;
            self.out
                .write_all(&u32::try_from(value).unwrap_or(u32::MAX).to_le_bytes())?;
        }
        self.out.flush()?;
        self.out.get_ref().sync_all()
    }
}

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn le_u32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4)
        .and_then(|s| s.try_into().ok())
        .map_or(0, u32::from_le_bytes)
}

/// Where one frame's data sits in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLocation {
    /// Absolute byte offset of the frame data.
    pub offset: u64,
    /// Length in bytes.
    pub len: u32,
}

/// Reads frames from an AVI file with one video stream.
///
/// Frame locations come from the clip's sidecar ([`AviReader::with_sidecar`]),
/// the file's `idx1` index, or, for a file whose writer never finished, a
/// walk over the `movi` list.
///
/// ```no_run
/// use std::path::Path;
/// use stance::avi::AviReader;
///
/// let mut avi = AviReader::open(Path::new("clips/shot.avi"))?;
/// println!("{} frames, {}x{}", avi.len(), avi.width(), avi.height());
/// let first = avi.read_frame(0)?;
/// # Ok::<(), stance::Error>(())
/// ```
#[derive(Debug)]
pub struct AviReader {
    file: File,
    path: PathBuf,
    fourcc: [u8; 4],
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    frames: Vec<FrameLocation>,
}

impl AviReader {
    /// Opens a file and locates its frames through its index (or by walking
    /// the `movi` list when there is none).
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file can't be read, [`Error::InvalidAvi`] if it
    /// isn't an AVI with a video stream.
    pub fn open(path: &Path) -> Result<Self> {
        let mut r = Self::open_headers(path)?;
        let (movi, idx1) = r.top_chunks()?;
        let from_index = idx1.and_then(|(pos, size)| r.read_index(pos, size, movi).ok());
        r.frames = match from_index {
            Some(f) if !f.is_empty() || movi.is_none() => f,
            _ => match movi {
                Some((body, end)) => r.walk_movi(body, end)?,
                None => Vec::new(),
            },
        };
        Ok(r)
    }

    /// Opens a clip using the frame locations in its sidecar.
    ///
    /// # Errors
    ///
    /// As [`AviReader::open`], plus [`Error::InvalidSidecar`] if the sidecar
    /// doesn't match the file.
    pub fn with_sidecar(path: &Path, sidecar: &Sidecar) -> Result<Self> {
        sidecar.validate()?;
        let mut r = Self::open_headers(path)?;
        let file_len = r.file_len()?;
        if (r.width, r.height) != (sidecar.width, sidecar.height) {
            return Err(Error::InvalidSidecar(format!(
                "sidecar says {}x{}, {} is {}x{}",
                sidecar.width,
                sidecar.height,
                path.display(),
                r.width,
                r.height
            )));
        }
        if sidecar.frames.iter().any(|f| f.end() > file_len) {
            return Err(Error::InvalidSidecar(format!(
                "frame offsets run past the end of {}",
                path.display()
            )));
        }
        r.frames = sidecar
            .frames
            .iter()
            .map(|f| FrameLocation {
                offset: f.offset,
                len: f.len,
            })
            .collect();
        Ok(r)
    }

    fn io(&self, e: std::io::Error) -> Error {
        Error::io(format!("reading {}", self.path.display()), e)
    }

    fn file_len(&self) -> Result<u64> {
        self.file
            .metadata()
            .map(|m| m.len())
            .map_err(|e| self.io(e))
    }

    fn open_headers(path: &Path) -> Result<Self> {
        let file =
            File::open(path).map_err(|e| Error::io(format!("opening {}", path.display()), e))?;
        let mut r = Self {
            file,
            path: path.to_owned(),
            fourcc: [0; 4],
            width: 0,
            height: 0,
            fps_num: 0,
            fps_den: 1,
            frames: Vec::new(),
        };
        let mut head = [0u8; 12];
        if r.file.read_exact(&mut head).is_err()
            || &head[0..4] != b"RIFF"
            || &head[8..12] != b"AVI "
        {
            return Err(Error::InvalidAvi(format!(
                "{} is not an AVI file",
                path.display()
            )));
        }
        let file_len = r.file_len()?;
        let mut pos = 12u64;
        while pos + 12 <= file_len {
            let (id, size) = r.chunk_header(pos)?;
            if &id == b"LIST" && &r.list_kind(pos)? == b"hdrl" {
                // The list body after its type, clamped to the file and the cap.
                let len = u64::from(size.saturating_sub(4))
                    .min(file_len - (pos + 12))
                    .min(HDRL_MAX);
                let mut buf = Vec::new();
                (&mut r.file)
                    .take(len)
                    .read_to_end(&mut buf)
                    .map_err(|e| r.io(e))?;
                r.parse_hdrl(&buf);
                break;
            }
            pos = pos.saturating_add(8 + u64::from(size) + u64::from(size & 1));
        }
        if r.width == 0 || r.height == 0 {
            return Err(Error::InvalidAvi(format!(
                "{} has no video stream header",
                path.display()
            )));
        }
        Ok(r)
    }

    /// (`movi` body start, end) and (`idx1` body start, size), if present.
    #[allow(clippy::type_complexity)]
    fn top_chunks(&mut self) -> Result<(Option<(u64, u64)>, Option<(u64, u32)>)> {
        let file_len = self.file_len()?;
        let (mut movi, mut idx1) = (None, None);
        let mut pos = 12u64;
        while pos + 8 <= file_len {
            let (id, size) = self.chunk_header(pos)?;
            let body = pos + 8;
            if &id == b"LIST" && pos + 12 <= file_len && &self.list_kind(pos)? == b"movi" {
                // An unfinished writer leaves the movi size at 0: walk to EOF then.
                let end = if size >= 4 {
                    (body + u64::from(size)).min(file_len)
                } else {
                    file_len
                };
                movi = Some((body + 4, end));
                if size < 4 {
                    break;
                }
            } else if &id == b"idx1" {
                idx1 = Some((body, size));
            }
            pos = body.saturating_add(u64::from(size) + u64::from(size & 1));
        }
        Ok((movi, idx1))
    }

    fn chunk_header(&mut self, pos: u64) -> Result<([u8; 4], u32)> {
        let mut h = [0u8; 8];
        self.file
            .seek(SeekFrom::Start(pos))
            .and_then(|_| self.file.read_exact(&mut h))
            .map_err(|e| self.io(e))?;
        Ok(([h[0], h[1], h[2], h[3]], le_u32(&h, 4)))
    }

    /// The list type of the LIST chunk at `pos` (leaves the file after it).
    fn list_kind(&mut self, pos: u64) -> Result<[u8; 4]> {
        let mut kind = [0u8; 4];
        self.file
            .seek(SeekFrom::Start(pos + 8))
            .and_then(|_| self.file.read_exact(&mut kind))
            .map_err(|e| self.io(e))?;
        Ok(kind)
    }

    fn parse_hdrl(&mut self, buf: &[u8]) {
        let mut i = 0usize;
        while i + 8 <= buf.len() {
            let id = &buf[i..i + 4];
            let size = le_u32(buf, i + 4) as usize;
            if id == b"LIST" {
                i += 12;
                continue;
            }
            let body = i + 8;
            if id == b"avih" {
                self.width = le_u32(buf, body + 32);
                self.height = le_u32(buf, body + 36);
            } else if id == b"strh" && buf.get(body..body + 4) == Some(b"vids") {
                if let Some(h) = buf.get(body + 4..body + 8) {
                    self.fourcc.copy_from_slice(h);
                }
                self.fps_den = le_u32(buf, body + 20).max(1);
                self.fps_num = le_u32(buf, body + 24);
            } else if id == b"strf" {
                if let Some(c) = buf.get(body + 16..body + 20)
                    && c != [0; 4]
                {
                    self.fourcc.copy_from_slice(c);
                }
                let w = le_u32(buf, body + 4);
                let h = le_u32(buf, body + 8);
                if w > 0 {
                    self.width = w;
                    // Bottom-up RGB stores a negative height.
                    self.height = h.cast_signed().unsigned_abs();
                }
            }
            i = body + size + (size & 1);
        }
    }

    /// Frame locations from `idx1`. Offsets are relative to the `movi`
    /// fourcc (the norm) or absolute (some writers); the first entry decides.
    fn read_index(
        &mut self,
        pos: u64,
        size: u32,
        movi: Option<(u64, u64)>,
    ) -> Result<Vec<FrameLocation>> {
        let file_len = self.file_len()?;
        // Clamped to the bytes the file holds: a corrupt size can't drive the
        // allocation.
        let len = u64::from(size).min(file_len.saturating_sub(pos));
        let mut buf = Vec::new();
        self.file
            .seek(SeekFrom::Start(pos))
            .and_then(|_| (&mut self.file).take(len).read_to_end(&mut buf))
            .map_err(|e| self.io(e))?;
        let entries: Vec<([u8; 4], u64, u32)> = buf
            .chunks_exact(16)
            .map(|e| {
                (
                    [e[0], e[1], e[2], e[3]],
                    u64::from(le_u32(e, 8)),
                    le_u32(e, 12),
                )
            })
            .filter(|(id, _, _)| id[2..4] == *b"dc" || id[2..4] == *b"db")
            .collect();
        let Some(&(first_id, first_off, _)) = entries.first() else {
            return Ok(Vec::new());
        };
        let movi_fourcc = movi.map_or(OFF_MOVI_FOURCC, |(body, _)| body - 4);
        let base = if self
            .chunk_header(movi_fourcc.saturating_add(first_off))
            .ok()
            .map(|h| h.0)
            == Some(first_id)
        {
            movi_fourcc
        } else {
            0
        };
        let frames: Vec<FrameLocation> = entries
            .iter()
            .map(|&(_, off, len)| FrameLocation {
                offset: base.saturating_add(off).saturating_add(8),
                len,
            })
            .collect();
        if frames
            .iter()
            .any(|f| f.offset.saturating_add(u64::from(f.len)) > file_len)
        {
            return Err(Error::InvalidAvi(
                "index points past the end of the file".into(),
            ));
        }
        Ok(frames)
    }

    fn walk_movi(&mut self, mut pos: u64, end: u64) -> Result<Vec<FrameLocation>> {
        let mut out = Vec::new();
        while pos + 8 <= end {
            let (id, size) = self.chunk_header(pos)?;
            let body = pos + 8;
            if &id == b"LIST" {
                // `rec ` lists group chunks; descend into them.
                pos = body + 4;
                continue;
            }
            if body.saturating_add(u64::from(size)) > end {
                break; // truncated final frame
            }
            if id[2..4] == *b"dc" || id[2..4] == *b"db" {
                out.push(FrameLocation {
                    offset: body,
                    len: size,
                });
            }
            pos = body.saturating_add(u64::from(size) + u64::from(size & 1));
        }
        Ok(out)
    }

    /// Number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// True if the file has no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Stream fourcc, e.g. `MJPG`.
    #[must_use]
    pub fn fourcc(&self) -> String {
        String::from_utf8_lossy(&self.fourcc).trim_end().to_owned()
    }

    /// Frame width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The stream's frame rate from its header.
    #[must_use]
    pub fn fps(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den.max(1))
    }

    /// The stream as a [`Format`].
    #[must_use]
    pub fn format(&self) -> Format {
        Format {
            fourcc: self.fourcc(),
            width: self.width,
            height: self.height,
            fps_num: self.fps_num,
            fps_den: self.fps_den.max(1),
        }
    }

    /// Every frame's location, in file order.
    #[must_use]
    pub fn frames(&self) -> &[FrameLocation] {
        &self.frames
    }

    /// Reads one frame's bytes.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for an index out of range, [`Error::Io`] if
    /// the read fails or the file is shorter than the frame's location says
    /// (it changed since it was opened).
    pub fn read_frame(&mut self, i: usize) -> Result<Vec<u8>> {
        let loc = *self
            .frames
            .get(i)
            .ok_or_else(|| Error::InvalidArgument(format!("frame {i} of {}", self.frames.len())))?;
        // Locations were checked against the file's length when it was opened.
        let mut buf = Vec::with_capacity(loc.len as usize);
        self.file
            .seek(SeekFrom::Start(loc.offset))
            .and_then(|_| {
                (&mut self.file)
                    .take(u64::from(loc.len))
                    .read_to_end(&mut buf)
            })
            .map_err(|e| self.io(e))?;
        if buf.len() != loc.len as usize {
            return Err(self.io(std::io::ErrorKind::UnexpectedEof.into()));
        }
        Ok(buf)
    }
}

/// Plays back an MJPEG AVI file (a clip written by this crate, or any
/// single-stream MJPEG AVI) as a [`Source`].
///
/// Frame times come from the clip's sidecar `<name>.json` when one sits next
/// to the file, otherwise from the stream's frame rate. Each frame's
/// `host_ns` is the start time plus the frame's offset into the file;
/// `paced` playback also waits for that time, unpaced playback returns frames
/// immediately. After the last frame [`Source::next_frame`] returns
/// [`Error::Ended`].
///
/// ```no_run
/// use std::path::Path;
/// use std::time::Duration;
/// use stance::{AviFile, Source};
///
/// let mut clip = AviFile::open(Path::new("clips/shot.avi"), true)?;
/// loop {
///     match clip.next_frame(Duration::from_secs(1)) {
///         Ok(Some(frame)) => println!("frame {} at {} ns", frame.seq, frame.host_ns),
///         Ok(None) => {}
///         Err(stance::Error::Ended) => break,
///         Err(e) => return Err(e),
///     }
/// }
/// # Ok::<(), stance::Error>(())
/// ```
#[derive(Debug)]
pub struct AviFile {
    reader: AviReader,
    info: CameraInfo,
    format: Format,
    /// Each frame's time relative to the first.
    times: Vec<u64>,
    origin_ns: u64,
    paced: bool,
    next: usize,
}

impl AviFile {
    /// Opens a file for playback, starting now.
    ///
    /// # Errors
    ///
    /// As [`AviReader::open`], and [`Error::InvalidAvi`] for a stream that
    /// isn't MJPEG.
    pub fn open(path: &Path, paced: bool) -> Result<Self> {
        let sidecar_path = path.with_extension("json");
        let sidecar = std::fs::read_to_string(&sidecar_path)
            .ok()
            .and_then(|json| Sidecar::from_json(&json).ok());
        let (reader, times) = if let Some(s) = sidecar {
            let times = s.frames.iter().map(|f| f.t_ns).collect();
            (AviReader::with_sidecar(path, &s)?, times)
        } else {
            let r = AviReader::open(path)?;
            let period = r.format().frame_period_ns();
            let times = (0..r.len() as u64).map(|i| i * period).collect();
            (r, times)
        };
        let format = reader.format();
        if !format.is_mjpeg() {
            return Err(Error::InvalidAvi(format!(
                "{} holds {} video; only MJPEG is supported",
                path.display(),
                format.fourcc
            )));
        }
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        );
        let info = CameraInfo {
            id: path.display().to_string(),
            name,
            usb: None,
            formats: vec![format.clone()],
            controls: Vec::new(),
            exposure_unit: None,
            probe_error: None,
        };
        Ok(Self {
            reader,
            info,
            format,
            times,
            origin_ns: clock::now_ns(),
            paced,
            next: 0,
        })
    }

    /// Sets the host time of the first frame (default: when the file was opened).
    #[must_use]
    pub fn with_origin(mut self, origin_ns: u64) -> Self {
        self.origin_ns = origin_ns;
        self
    }

    /// Number of frames in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// True if the file has no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

impl Source for AviFile {
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>> {
        let Some(&t) = self.times.get(self.next) else {
            return Err(Error::Ended);
        };
        let due = self.origin_ns.saturating_add(t);
        if self.paced && !wait_until(due, timeout) {
            return Ok(None);
        }
        let i = self.next;
        let jpeg: Arc<[u8]> = Arc::from(self.reader.read_frame(i)?);
        self.next += 1;
        Ok(Some(Frame {
            jpeg,
            host_ns: due,
            device_ns: None,
            seq: i as u64,
            width: self.format.width,
            height: self.format.height,
        }))
    }

    fn info(&self) -> &CameraInfo {
        &self.info
    }

    fn format(&self) -> &Format {
        &self.format
    }
}

/// Sleeps until the host clock reaches `due_ns`, but no longer than
/// `timeout`. True if `due_ns` was reached.
pub(crate) fn wait_until(due_ns: u64, timeout: Duration) -> bool {
    let now = clock::now_ns();
    if now >= due_ns {
        return true;
    }
    let wait = Duration::from_nanos(due_ns - now);
    if wait > timeout {
        thread::sleep(timeout);
        return false;
    }
    thread::sleep(wait);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    /// Independent minimal RIFF parser: a tree of chunks over a byte slice.
    #[derive(Debug)]
    struct Chunk<'a> {
        id: [u8; 4],
        kind: [u8; 4],
        data: &'a [u8],
        offset: usize,
        children: Vec<Chunk<'a>>,
    }

    fn parse_chunks(buf: &[u8], base: usize) -> Vec<Chunk<'_>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 8 <= buf.len() {
            let id: [u8; 4] = buf[i..i + 4].try_into().unwrap();
            let size = u32::from_le_bytes(buf[i + 4..i + 8].try_into().unwrap()) as usize;
            let data = &buf[i + 8..i + 8 + size];
            let (kind, children) = if &id == b"RIFF" || &id == b"LIST" {
                let kind: [u8; 4] = data[..4].try_into().unwrap();
                (kind, parse_chunks(&data[4..], base + i + 12))
            } else {
                ([0; 4], Vec::new())
            };
            out.push(Chunk {
                id,
                kind,
                data,
                offset: base + i,
                children,
            });
            i += 8 + size + (size & 1);
        }
        assert_eq!(i, buf.len(), "chunks must tile their parent exactly");
        out
    }

    fn find<'a, 'b>(chunks: &'b [Chunk<'a>], id: [u8; 4], kind: Option<[u8; 4]>) -> &'b Chunk<'a> {
        chunks
            .iter()
            .find(|c| c.id == id && kind.is_none_or(|k| c.kind == k))
            .unwrap()
    }

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }

    fn params() -> AviParams {
        AviParams {
            fourcc: *b"MJPG",
            width: 1280,
            height: 720,
            fps_num: 120,
            fps_den: 1,
        }
    }

    fn frame(i: usize) -> Vec<u8> {
        // Varying sizes, including odd ones to exercise padding.
        let len = 100 + i * 37 % 91;
        let mut v: Vec<u8> = (0..len)
            .map(|j| u8::try_from((i * 7 + j) % 251).unwrap())
            .collect();
        v[0] = 0xFF;
        v[1] = 0xD8;
        v
    }

    fn write(path: &Path, frames: &[Vec<u8>]) -> Vec<u64> {
        let mut w = AviWriter::create(path, &params()).unwrap();
        let offsets = frames.iter().map(|f| w.write_frame(f).unwrap()).collect();
        w.finish().unwrap();
        offsets
    }

    #[test]
    fn writes_parseable_avi() {
        let dir = TempDir::new("avi");
        let path = dir.path().join("t.avi");
        let frames: Vec<Vec<u8>> = (0..25).map(frame).collect();
        let offsets = write(&path, &frames);
        assert_eq!(offsets[0], HEADER_LEN + 8, "fixed header length");

        let bytes = std::fs::read(&path).unwrap();
        let top = parse_chunks(&bytes, 0);
        assert_eq!(top.len(), 1);
        let riff = &top[0];
        assert_eq!((&riff.id, &riff.kind), (b"RIFF", b"AVI "));
        assert_eq!(riff.data.len() + 8, bytes.len());

        let hdrl = find(&riff.children, *b"LIST", Some(*b"hdrl"));
        let avih = find(&hdrl.children, *b"avih", None);
        assert_eq!(u32_at(avih.data, 0), 1_000_000 / 120); // µs per frame
        assert_eq!(u32_at(avih.data, 12) & AVIF_HASINDEX, AVIF_HASINDEX);
        assert_eq!(u32_at(avih.data, 16), 25); // total frames
        let max = u32::try_from(frames.iter().map(Vec::len).max().unwrap()).unwrap();
        assert_eq!(
            u32_at(avih.data, 28),
            max + 8,
            "largest chunk with its header"
        );
        assert_eq!((u32_at(avih.data, 32), u32_at(avih.data, 36)), (1280, 720));
        let strl = find(&hdrl.children, *b"LIST", Some(*b"strl"));
        let stream_header = find(&strl.children, *b"strh", None);
        let sh = stream_header.data;
        assert_eq!(&sh[0..8], b"vidsMJPG");
        assert_eq!((u32_at(sh, 20), u32_at(sh, 24)), (1, 120));
        assert_eq!(u32_at(sh, 32), 25); // length
        let stream_format = find(&strl.children, *b"strf", None);
        assert_eq!(&stream_format.data[16..20], b"MJPG");

        let movi = find(&riff.children, *b"LIST", Some(*b"movi"));
        assert_eq!(movi.children.len(), 25);
        for ((c, f), off) in movi.children.iter().zip(&frames).zip(&offsets) {
            assert_eq!(&c.id, b"00dc");
            assert_eq!(c.data, f.as_slice(), "frame bytes are stored untouched");
            assert_eq!(
                usize::try_from(*off).unwrap(),
                c.offset + 8,
                "returned offset is the payload"
            );
        }
        let idx1 = find(&riff.children, *b"idx1", None);
        let movi_fourcc = movi.offset + 8;
        for (k, c) in movi.children.iter().enumerate() {
            let e = &idx1.data[k * 16..k * 16 + 16];
            assert_eq!(&e[0..4], b"00dc");
            assert_eq!(u32_at(e, 4), AVIIF_KEYFRAME);
            assert_eq!(movi_fourcc + u32_at(e, 8) as usize, c.offset);
            assert_eq!(u32_at(e, 12) as usize, c.data.len());
        }
    }

    #[test]
    fn reader_round_trips_through_the_index() {
        let dir = TempDir::new("avi-read");
        let path = dir.path().join("r.avi");
        let frames: Vec<Vec<u8>> = (0..10).map(frame).collect();
        let offsets = write(&path, &frames);
        let mut avi = AviReader::open(&path).unwrap();
        assert_eq!(avi.fourcc(), "MJPG");
        assert_eq!((avi.width(), avi.height()), (1280, 720));
        assert!((avi.fps() - 120.0).abs() < 1e-9);
        assert_eq!(avi.len(), 10);
        for (i, want) in frames.iter().enumerate() {
            assert_eq!(&avi.read_frame(i).unwrap(), want);
            assert_eq!(avi.frames()[i].offset, offsets[i]);
        }
        assert!(avi.read_frame(10).is_err());
    }

    #[test]
    fn reader_recovers_an_unfinished_file() {
        let dir = TempDir::new("avi-crash");
        let path = dir.path().join("c.avi");
        let mut w = AviWriter::create(&path, &params()).unwrap();
        for i in 0..5 {
            w.write_frame(&frame(i)).unwrap();
        }
        w.out.flush().unwrap();
        // A killed writer: no index, header sizes still zero.
        drop(w);
        let mut avi = AviReader::open(&path).unwrap();
        assert_eq!(avi.len(), 5);
        assert_eq!(avi.read_frame(4).unwrap(), frame(4));
    }

    #[test]
    fn empty_file_is_valid() {
        let dir = TempDir::new("avi-empty");
        let path = dir.path().join("e.avi");
        write(&path, &[]);
        let avi = AviReader::open(&path).unwrap();
        assert!(avi.is_empty());
    }

    #[test]
    fn rejects_non_avi() {
        let dir = TempDir::new("avi-bad");
        let path = dir.path().join("x.avi");
        std::fs::write(&path, b"RIFF\0\0\0\0WAVEfmt ").unwrap();
        assert!(matches!(AviReader::open(&path), Err(Error::InvalidAvi(_))));
        assert!(matches!(
            AviReader::open(&dir.path().join("missing.avi")),
            Err(Error::Io { .. })
        ));
        assert!(
            AviWriter::create(
                &path,
                &AviParams {
                    width: 0,
                    ..params()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn plays_back_unpaced_without_a_sidecar() {
        let dir = TempDir::new("avi-play");
        let path = dir.path().join("p.avi");
        let frames: Vec<Vec<u8>> = (0..3).map(frame).collect();
        write(&path, &frames);
        let mut src = AviFile::open(&path, false).unwrap().with_origin(1000);
        assert_eq!(src.len(), 3);
        assert_eq!(src.format(), &Format::mjpeg(1280, 720, 120));
        assert_eq!(src.info().name, "p.avi");
        for (i, want) in frames.iter().enumerate() {
            let f = src.next_frame(Duration::ZERO).unwrap().unwrap();
            assert_eq!(&*f.jpeg, want.as_slice());
            assert_eq!(f.seq, i as u64);
            assert_eq!(f.host_ns, 1000 + i as u64 * 8_333_333);
        }
        assert!(matches!(src.next_frame(Duration::ZERO), Err(Error::Ended)));
    }

    #[test]
    fn paced_playback_waits_for_each_frame() {
        let dir = TempDir::new("avi-paced");
        let path = dir.path().join("p.avi");
        write(&path, &[frame(0), frame(1)]);
        let start = clock::now_ns() + 2_000_000_000;
        let mut src = AviFile::open(&path, true).unwrap().with_origin(start);
        // Due in 2 s; a 10 ms timeout returns nothing.
        assert!(src.next_frame(Duration::from_millis(10)).unwrap().is_none());
        let mut src = src.with_origin(clock::now_ns());
        assert!(
            src.next_frame(Duration::from_millis(100))
                .unwrap()
                .is_some()
        );
        let f = src.next_frame(Duration::from_millis(100)).unwrap().unwrap();
        assert!(clock::now_ns() >= f.host_ns);
    }
}
