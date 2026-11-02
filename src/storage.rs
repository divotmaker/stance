//! Writing clips to disk: an MJPEG AVI plus its JSON sidecar.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::avi::{AviParams, AviReader, AviWriter};
use crate::recorder::Clip;
use crate::sidecar::{SIDECAR_VERSION, Sidecar, SidecarFps, SidecarFrame};
use crate::{Error, Result};

/// The files of a written clip.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipFiles {
    /// `<dir>/<name>.avi`.
    pub avi: PathBuf,
    /// `<dir>/<name>.json`.
    pub sidecar_path: PathBuf,
    /// The sidecar's contents.
    pub sidecar: Sidecar,
}

/// Writes a clip as `<dir>/<name>.avi` and `<dir>/<name>.json`.
///
/// The AVI is written as `<name>.avi.part`, synced and renamed when
/// complete; the sidecar is written last (also through a synced `.part`
/// file). A clip whose sidecar exists is therefore complete on disk. `dir` is
/// created if needed.
///
/// The AVI header carries the clip's measured frame rate, so players show
/// the clip at real speed even when the camera ran below its nominal rate.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for an empty clip or a `name` that isn't a
/// plain file name (empty, starting with `.`, or containing a path separator,
/// a control character or one of `:*?"<>|`); [`Error::Io`] if writing fails
/// (disk full, permissions, a name the file system refuses).
///
/// ```no_run
/// # fn example(clip: stance::Clip<String>) -> stance::Result<()> {
/// let files = stance::write_clip(std::path::Path::new("clips"), "shot-0001", &clip)?;
/// println!("{} frames in {}", files.sidecar.frames.len(), files.avi.display());
/// # Ok(()) }
/// ```
pub fn write_clip<T>(dir: &Path, name: &str, clip: &Clip<T>) -> Result<ClipFiles> {
    if name.is_empty()
        || name.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|'])
        || name.chars().any(char::is_control)
        || name.starts_with('.')
        || Path::new(name).file_name() != Some(name.as_ref())
    {
        return Err(Error::InvalidArgument(format!(
            "clip name `{name}` must be a plain file name"
        )));
    }
    let (Some(first), Some(_)) = (clip.frames.first(), clip.frames.last()) else {
        return Err(Error::InvalidArgument("the clip has no frames".into()));
    };
    fs::create_dir_all(dir).map_err(|e| Error::io(format!("creating {}", dir.display()), e))?;
    let avi_path = dir.join(format!("{name}.avi"));
    let avi_part = dir.join(format!("{name}.avi.part"));
    let sidecar_path = dir.join(format!("{name}.json"));
    let sidecar_part = dir.join(format!("{name}.json.part"));

    let measured = clip.measured_fps();
    let header_fps = measured.or(clip.fps).filter(|f| *f > 0.0).unwrap_or(30.0);
    // Rates are small positive numbers, so the millifps value fits a u32.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let fps_num = (header_fps * 1000.0).round().max(1.0) as u32;
    let params = AviParams {
        fourcc: *b"MJPG",
        width: first.width,
        height: first.height,
        fps_num,
        fps_den: 1000,
    };

    let mut frames = Vec::with_capacity(clip.frames.len());
    let written = (|| {
        let mut w = AviWriter::create(&avi_part, &params)?;
        for f in &clip.frames {
            let offset = w.write_frame(&f.jpeg)?;
            frames.push(SidecarFrame {
                t_ns: f.host_ns.saturating_sub(first.host_ns),
                offset,
                len: u32::try_from(f.jpeg.len()).unwrap_or(u32::MAX),
            });
        }
        w.finish()
    })();
    if let Err(e) = written {
        let _ = fs::remove_file(&avi_part);
        return Err(e);
    }
    fs::rename(&avi_part, &avi_path)
        .map_err(|e| Error::io(format!("renaming {}", avi_part.display()), e))?;

    let rel = |ns: u64| -> i64 {
        let d = i128::from(ns) - i128::from(first.host_ns);
        i64::try_from(d).unwrap_or(if d < 0 { i64::MIN } else { i64::MAX })
    };
    let sidecar = Sidecar {
        version: SIDECAR_VERSION,
        fourcc: "MJPG".into(),
        width: first.width,
        height: first.height,
        fps: SidecarFps {
            nominal: clip.fps,
            // A measurement: millifps resolution is plenty.
            measured: measured.map_or(0.0, |f| (f * 1000.0).round() / 1000.0),
        },
        start_ns: first.host_ns,
        frames,
        trigger_index: clip.trigger_index,
        impact_index: clip.impact_index,
        trigger_ns: rel(clip.trigger_ns),
        impact_ns: rel(clip.impact_ns),
        complete: clip.complete,
        dropped: clip.dropped,
        truncated: clip.truncated,
    };
    let json = sidecar.to_json()?;
    write_synced(&sidecar_part, json.as_bytes())
        .and_then(|()| fs::rename(&sidecar_part, &sidecar_path))
        .map_err(|e| {
            let _ = fs::remove_file(&sidecar_part);
            Error::io(format!("writing {}", sidecar_path.display()), e)
        })?;
    Ok(ClipFiles {
        avi: avi_path,
        sidecar_path,
        sidecar,
    })
}

/// Writes `bytes` to a new file and syncs it to disk.
fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Reads a clip sidecar.
///
/// # Errors
///
/// [`Error::Io`] if the file can't be read, [`Error::InvalidSidecar`] if it
/// doesn't parse.
///
/// ```no_run
/// let sidecar = stance::read_sidecar(std::path::Path::new("clips/shot-0001.json"))?;
/// println!("{} frames at {:.1} fps", sidecar.frames.len(), sidecar.fps.measured);
/// # Ok::<(), stance::Error>(())
/// ```
pub fn read_sidecar(path: &Path) -> Result<Sidecar> {
    let json = fs::read_to_string(path)
        .map_err(|e| Error::io(format!("reading {}", path.display()), e))?;
    Sidecar::from_json(&json)
}

/// Opens `<dir>/<name>.avi` with the frame locations of `<dir>/<name>.json`.
///
/// # Errors
///
/// As [`read_sidecar`] and [`AviReader::with_sidecar`].
///
/// ```no_run
/// let (mut avi, sidecar) = stance::open_clip(std::path::Path::new("clips"), "shot-0001")?;
/// let impact_frame = avi.read_frame(sidecar.impact_index)?;
/// # Ok::<(), stance::Error>(())
/// ```
pub fn open_clip(dir: &Path, name: &str) -> Result<(AviReader, Sidecar)> {
    let sidecar = read_sidecar(&dir.join(format!("{name}.json")))?;
    let avi = AviReader::with_sidecar(&dir.join(format!("{name}.avi")), &sidecar)?;
    Ok((avi, sidecar))
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::Arc;

    use super::*;
    use crate::Frame;
    use crate::recorder::ClipId;
    use crate::testutil::TempDir;

    fn clip() -> Clip<&'static str> {
        let frames = (0..12u64)
            .map(|i| Frame {
                // Odd and even sizes, distinct bytes per frame.
                jpeg: Arc::from(vec![
                    u8::try_from(i).unwrap();
                    50 + usize::try_from(i).unwrap()
                ]),
                host_ns: 5_000_000_000 + i * 10_000_000,
                device_ns: None,
                seq: i,
                width: 64,
                height: 48,
            })
            .collect();
        Clip {
            id: ClipId(3),
            tags: vec!["s1"],
            frames,
            trigger_index: 8,
            impact_index: 2,
            trigger_ns: 5_080_000_000,
            impact_ns: 5_020_000_000,
            complete: true,
            dropped: 0,
            truncated: false,
            short_buffer: false,
            fps: Some(100.0),
        }
    }

    #[test]
    fn round_trip_with_byte_offsets() {
        let dir = TempDir::new("clip");
        let clip = clip();
        let files = write_clip(dir.path(), "shot", &clip).unwrap();
        assert!(files.avi.ends_with("shot.avi"));
        assert!(!dir.path().join("shot.avi.part").exists());
        assert!(!dir.path().join("shot.json.part").exists());
        let s = read_sidecar(&files.sidecar_path).unwrap();
        assert_eq!(s, files.sidecar);
        assert_eq!((s.width, s.height), (64, 48));
        assert_eq!(s.frames.len(), 12);
        assert_eq!(s.start_ns, 5_000_000_000);
        assert_eq!(s.frames[11].t_ns, 110_000_000);
        assert_eq!((s.trigger_index, s.impact_index), (8, 2));
        assert_eq!((s.trigger_ns, s.impact_ns), (80_000_000, 20_000_000));
        assert_eq!(s.fps.nominal, Some(100.0));
        assert!((s.fps.measured - 100.0).abs() < 1e-9);
        assert!(s.complete && !s.truncated);

        // Every frame read back by its sidecar offset, straight from the file
        // (what an HTTP Range request does).
        let mut file = std::fs::File::open(&files.avi).unwrap();
        for (f, want) in s.frames.iter().zip(&clip.frames) {
            let mut buf = vec![0u8; f.len as usize];
            file.seek(SeekFrom::Start(f.offset)).unwrap();
            file.read_exact(&mut buf).unwrap();
            assert_eq!(buf.as_slice(), &*want.jpeg);
        }

        // The same frames through the AVI index and through the sidecar.
        let mut by_index = AviReader::open(&files.avi).unwrap();
        let (mut by_sidecar, _) = open_clip(dir.path(), "shot").unwrap();
        assert_eq!(by_index.frames(), by_sidecar.frames());
        assert_eq!(by_index.read_frame(7).unwrap(), &*clip.frames[7].jpeg);
        assert_eq!(by_sidecar.read_frame(7).unwrap(), &*clip.frames[7].jpeg);
        assert!(
            (by_index.fps() - 100.0).abs() < 1e-9,
            "measured rate in the header"
        );
    }

    #[test]
    fn rejects_bad_names_and_empty_clips() {
        let dir = TempDir::new("clip-bad");
        for bad in [
            "", "a/b", "..", ".hidden", "c:d", r"a\b", "a*", "a?", "a\"b", "a<b", "a>b", "a|b",
            "a\nb",
        ] {
            assert!(
                matches!(
                    write_clip(dir.path(), bad, &clip()),
                    Err(Error::InvalidArgument(_))
                ),
                "{bad}"
            );
        }
        let mut empty = clip();
        empty.frames.clear();
        assert!(write_clip(dir.path(), "e", &empty).is_err());
    }

    #[test]
    fn a_clip_ending_before_its_trigger_round_trips() {
        let dir = TempDir::new("clip-no-trigger");
        let mut clip = clip();
        clip.trigger_index = clip.frames.len();
        clip.trigger_ns = 6_000_000_000;
        clip.complete = false;
        write_clip(dir.path(), "late", &clip).unwrap();
        let (_, s) = open_clip(dir.path(), "late").unwrap();
        assert_eq!(s.trigger_index, 12);
        assert!(s.trigger_frame().is_none());
        assert!(!s.complete);
    }

    #[test]
    fn a_mismatched_sidecar_is_rejected() {
        let dir = TempDir::new("clip-mismatch");
        let files = write_clip(dir.path(), "a", &clip()).unwrap();
        let mut s = files.sidecar.clone();
        s.width = 640;
        assert!(AviReader::with_sidecar(&files.avi, &s).is_err());
        let mut s = files.sidecar.clone();
        s.frames[11].offset = 1 << 40;
        assert!(AviReader::with_sidecar(&files.avi, &s).is_err());
        // Offsets that overflow: an error, not a panic.
        let mut s = files.sidecar;
        s.frames[0].offset = u64::MAX;
        assert!(matches!(
            AviReader::with_sidecar(&files.avi, &s),
            Err(Error::InvalidSidecar(_))
        ));
    }
}
