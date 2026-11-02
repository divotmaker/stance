//! Truncated, corrupt and hostile AVI files and sidecars: errors or fewer
//! frames, never a panic or an allocation sized by an untrusted field.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use stance::avi::AviReader;
use stance::{AviFile, Clip, ClipId, Error, Frame, write_clip};

/// The system allocator, recording the largest single allocation.
struct Tracking;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator unchanged; only records sizes.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract is passed through.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        // SAFETY: the caller's contract is passed through.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Tracking = Tracking;

/// No test here legitimately allocates more than this.
const SANE: usize = 16 << 20;

const MS: u64 = 1_000_000;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("stance-hostile-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn clip() -> Clip<&'static str> {
    let frames: Vec<Frame> = (0..5u64)
        .map(|i| Frame {
            jpeg: Arc::from(vec![
                u8::try_from(i).unwrap();
                8 + usize::try_from(i).unwrap()
            ]),
            host_ns: 1_000 * MS + i * 10 * MS,
            device_ns: None,
            seq: i,
            width: 8,
            height: 8,
        })
        .collect();
    Clip {
        id: ClipId(0),
        tags: vec!["t"],
        frames,
        trigger_index: 2,
        impact_index: 2,
        trigger_ns: 1_020 * MS,
        impact_ns: 1_020 * MS,
        complete: true,
        dropped: 0,
        truncated: false,
        short_buffer: false,
        fps: Some(100.0),
    }
}

/// Opens `path` every way there is and reads every frame, ignoring errors.
fn exercise(path: &Path) {
    if let Ok(mut r) = AviReader::open(path) {
        for i in 0..r.len() {
            let _ = r.read_frame(i);
        }
    }
    let _ = AviFile::open(path, false);
}

/// Overwrites the size field of the first chunk with this fourcc (or LIST
/// with this list type).
fn patch_size(bytes: &mut [u8], tag: [u8; 4], size: u32) {
    let at = bytes.windows(4).position(|w| w == tag).unwrap();
    // A list type sits 8 bytes after the LIST fourcc whose size we want.
    let size_at = if &tag == b"hdrl" || &tag == b"movi" {
        at - 4
    } else {
        at + 4
    };
    bytes[size_at..size_at + 4].copy_from_slice(&size.to_le_bytes());
}

#[test]
fn huge_chunk_sizes_allocate_nothing_large() {
    let dir = TempDir::new("sizes");
    let files = write_clip(dir.path(), "c", &clip()).unwrap();
    let good = std::fs::read(&files.avi).unwrap();
    let path = dir.path().join("x.avi");
    for tag in [*b"idx1", *b"hdrl", *b"movi", *b"00dc", *b"strh", *b"avih"] {
        for size in [0xFFFF_FFF0, u32::MAX, 0x7FFF_FFFF] {
            let mut bytes = good.clone();
            patch_size(&mut bytes, tag, size);
            std::fs::write(&path, &bytes).unwrap();
            exercise(&path);
        }
    }
    // A 4 GiB idx1 still yields the real index.
    let mut bytes = good;
    patch_size(&mut bytes, *b"idx1", 0xFFFF_FFF0);
    std::fs::write(&path, &bytes).unwrap();
    let mut r = AviReader::open(&path).unwrap();
    assert_eq!(r.len(), 5);
    assert_eq!(r.read_frame(4).unwrap(), vec![4u8; 12]);
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(largest < SANE, "largest allocation {largest} bytes");
}

#[test]
fn truncated_and_corrupted_files_never_panic() {
    let dir = TempDir::new("trunc");
    let files = write_clip(dir.path(), "c", &clip()).unwrap();
    let bytes = std::fs::read(&files.avi).unwrap();
    let path = dir.path().join("x.avi");
    for n in 0..bytes.len() {
        std::fs::write(&path, &bytes[..n]).unwrap();
        exercise(&path);
    }
    for i in 0..bytes.len() {
        for v in [0x00, 0xFF] {
            let mut b = bytes.clone();
            b[i] = v;
            std::fs::write(&path, &b).unwrap();
            exercise(&path);
        }
    }
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(largest < SANE, "largest allocation {largest} bytes");
}

#[test]
fn hostile_sidecars_are_rejected() {
    let dir = TempDir::new("sidecar");
    let files = write_clip(dir.path(), "c", &clip()).unwrap();
    let reject = |edit: &dyn Fn(&mut stance::sidecar::Sidecar)| {
        let mut s = files.sidecar.clone();
        edit(&mut s);
        assert!(matches!(
            AviReader::with_sidecar(&files.avi, &s),
            Err(Error::InvalidSidecar(_))
        ));
    };
    reject(&|s| s.frames[0].offset = u64::MAX);
    reject(&|s| s.frames[4].offset = u64::MAX - 3);
    reject(&|s| s.frames[4].len = u32::MAX);
    reject(&|s| s.frames.swap(1, 2));
    reject(&|s| s.impact_index = 5);
    // A sidecar beside the AVI that lies about it: playback refuses it and
    // falls back to the index only when the sidecar doesn't parse.
    let mut s = files.sidecar.clone();
    s.frames[4].len = u32::MAX;
    std::fs::write(&files.sidecar_path, s.to_json().unwrap()).unwrap();
    assert!(matches!(
        AviFile::open(&files.avi, false),
        Err(Error::InvalidSidecar(_))
    ));
    std::fs::write(&files.sidecar_path, "{ not json").unwrap();
    assert_eq!(AviFile::open(&files.avi, false).unwrap().len(), 5);
}
