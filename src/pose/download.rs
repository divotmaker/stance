//! Fetching models over HTTPS (`pose-download` feature).

use std::fs;
use std::io::{self, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::model::{ModelSpec, hex};
use crate::{Error, Result};

/// The largest download accepted, archive overhead included.
const MAX_DOWNLOAD: u64 = 512 * 1024 * 1024;
/// Longest wait for name resolution and the TCP connection, per request.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Longest wait for any one read from the connection (the TLS handshake,
/// the response headers, each piece of the body): a stalled transfer fails
/// after this.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
/// Upper bound on the whole request, redirects included.
const GLOBAL_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// Redirects followed (the release hosts redirect to their storage).
const MAX_REDIRECTS: u32 = 5;

/// Makes sure `<cache_dir>/<name>.onnx` holds the verified model and returns
/// its path.
///
/// A file that is already there and matches the SHA-256 is used as is.
/// Otherwise the model is downloaded from [`ModelSpec::url`], over HTTPS
/// only (rustls), redirects included. The response streams into a temporary
/// file in `cache_dir` named for this process; when the source is a zip
/// archive, its `.onnx` member is extracted into a second one. The result
/// is renamed to `<name>.onnx` only once its SHA-256 matches, so a
/// concurrent fetch by another process or thread never sees a partial file.
/// Temporary files are removed on failure. `cache_dir` is created if
/// needed.
///
/// This blocks for the whole download, with bounds: 30 s to connect, 60 s
/// for any one read (a stalled connection fails rather than hangs), 15
/// minutes for the whole request, and 512 MiB of response.
///
/// # Errors
///
/// [`Error::Download`] if the URL isn't `https://`, a request fails or
/// times out, a redirect leaves HTTPS, the response is too large, or the
/// archive lacks the member; [`Error::Model`] if the bytes don't match the
/// hash; [`Error::Io`] if a file can't be written.
pub fn fetch_model(spec: &ModelSpec, cache_dir: &Path) -> Result<PathBuf> {
    fetch_model_with(spec, cache_dir, None, None)
}

/// How far a download has got: bytes of the response read so far, and the
/// response's length when the server sent one. For a zip source these are
/// the archive's bytes, not the extracted model's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchProgress {
    /// Bytes read so far.
    pub done: u64,
    /// The response's `Content-Length`, when known.
    pub total: Option<u64>,
}

/// [`fetch_model`] with an optional progress callback and an optional
/// cancel flag.
///
/// `progress` is called once when the response starts (`done` 0) and after
/// every piece of the body read, on the calling thread. `cancel` is checked
/// before the request and after every piece read; once it is set the fetch
/// stops, removes its temporary files and returns [`Error::Cancelled`]. A
/// cancel is noticed between reads, so it can wait for one read (at most the
/// 60 s read timeout on a stalled connection). A file already verified in
/// `cache_dir` is returned without calling either.
///
/// # Errors
///
/// As [`fetch_model`], and [`Error::Cancelled`] once `cancel` is set.
pub fn fetch_model_with(
    spec: &ModelSpec,
    cache_dir: &Path,
    mut progress: Option<&mut dyn FnMut(FetchProgress)>,
    cancel: Option<&AtomicBool>,
) -> Result<PathBuf> {
    let path = spec.path_in(cache_dir);
    if spec.read_verified(cache_dir).is_ok() {
        return Ok(path);
    }
    let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let download_err = |message: String| Error::Download {
        url: spec.url.to_owned(),
        message,
    };
    if !spec.url.starts_with("https://") {
        return Err(download_err("not an https:// URL".into()));
    }
    fs::create_dir_all(cache_dir)
        .map_err(|e| Error::io(format!("creating {}", cache_dir.display()), e))?;

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .https_only(true)
        .max_redirects(MAX_REDIRECTS)
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_per_read(Some(READ_TIMEOUT))
        .timeout_global(Some(GLOBAL_TIMEOUT))
        .build()
        .into();
    let response = agent
        .get(spec.url)
        .call()
        .map_err(|e| download_err(e.to_string()))?;
    let total = response.body().content_length();
    let mut body = response
        .into_body()
        .into_with_config()
        .limit(MAX_DOWNLOAD)
        .reader();
    let mut download = TempFile::create(&path)?;
    let mut report = |done| {
        if let Some(progress) = progress.as_mut() {
            progress(FetchProgress { done, total });
        }
        !cancelled()
    };
    if !report(0) {
        return Err(Error::Cancelled);
    }
    let downloaded = copy_hashing(&mut body, &mut download, &mut report).map_err(|e| match e {
        Copy::Read(e) => download_err(e.to_string()),
        Copy::Write(e) => Error::io(format!("writing {}", download.path.display()), e),
        Copy::Cancelled => Error::Cancelled,
    })?;

    let (onnx, (digest, len)) = match spec.zip_member {
        Some(member) => {
            download
                .file
                .seek(io::SeekFrom::Start(0))
                .map_err(|e| Error::io(format!("reading {}", download.path.display()), e))?;
            let mut onnx = TempFile::create(&path)?;
            let hashed =
                extract(&mut download.file, member, spec.size, &mut onnx).map_err(|e| match e {
                    Copy::Read(e) => download_err(e.to_string()),
                    Copy::Write(e) => Error::io(format!("writing {}", onnx.path.display()), e),
                    Copy::Cancelled => Error::Cancelled,
                })?;
            drop(download);
            (onnx, hashed)
        }
        None => (download, downloaded),
    };
    spec.check_sha256(&digest, len)?;
    if cancelled() {
        return Err(Error::Cancelled);
    }
    onnx.persist(&path)?;
    Ok(path)
}

/// Why a copy failed: reading the source, writing the file, or the
/// progress callback asked to stop.
enum Copy {
    Read(io::Error),
    Write(io::Error),
    Cancelled,
}

/// Copies `from` into `to`, returning the SHA-256 (hex) and length.
/// `progress` gets the bytes copied so far after every piece and returns
/// whether to go on.
fn copy_hashing(
    from: &mut impl Read,
    to: &mut TempFile,
    progress: &mut dyn FnMut(u64) -> bool,
) -> std::result::Result<(String, u64), Copy> {
    let mut hasher = Sha256::new();
    let mut len = 0u64;
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = match from.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Copy::Read(e)),
        };
        let chunk = &buf[..n];
        hasher.update(chunk);
        to.file.write_all(chunk).map_err(Copy::Write)?;
        len += n as u64;
        if !progress(len) {
            return Err(Copy::Cancelled);
        }
    }
    Ok((hex(&hasher.finalize()), len))
}

/// Copies one member of a zip archive into `to`, refusing more than `max`
/// bytes; returns its SHA-256 (hex) and length.
fn extract<R: Read + Seek>(
    archive: R,
    member: &str,
    max: u64,
    to: &mut TempFile,
) -> std::result::Result<(String, u64), Copy> {
    let bad = |message: String| Copy::Read(io::Error::new(io::ErrorKind::InvalidData, message));
    let mut zip = zip::ZipArchive::new(archive).map_err(|e| bad(e.to_string()))?;
    let file = zip
        .by_name(member)
        .map_err(|e| bad(format!("archive member `{member}`: {e}")))?;
    // Read one byte past the limit to notice an oversized member.
    let (digest, len) = copy_hashing(&mut file.take(max.saturating_add(1)), to, &mut |_| true)
        .map_err(|e| match e {
            Copy::Read(e) => bad(format!("archive member `{member}`: {e}")),
            other => other,
        })?;
    if len > max {
        return Err(bad(format!(
            "archive member `{member}` is larger than {max} bytes"
        )));
    }
    Ok((digest, len))
}

/// A file beside the destination with a name no other process or thread
/// uses (`<file>.part-<pid>-<n>`), removed on drop unless persisted.
struct TempFile {
    path: PathBuf,
    file: fs::File,
    keep: bool,
}

impl TempFile {
    fn create(dest: &Path) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = dest
            .file_name()
            .map_or_else(|| "model".into(), |n| n.to_string_lossy().into_owned());
        let path = dest.with_file_name(format!(
            "{name}.part-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| Error::io(format!("creating {}", path.display()), e))?;
        Ok(Self {
            path,
            file,
            keep: false,
        })
    }

    /// Syncs the file and renames it to `dest`.
    fn persist(mut self, dest: &Path) -> Result<()> {
        self.file
            .sync_all()
            .map_err(|e| Error::io(format!("writing {}", self.path.display()), e))?;
        fs::rename(&self.path, dest)
            .map_err(|e| Error::io(format!("renaming {}", self.path.display()), e))?;
        self.keep = true;
        Ok(())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn archive(name: &str, data: &[u8]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        let mut w = zip::ZipWriter::new(&mut buf);
        w.start_file(
            name,
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored),
        )
        .unwrap();
        w.write_all(data).unwrap();
        w.finish().unwrap();
        buf.into_inner()
    }

    fn read_back(t: &mut TempFile) -> Vec<u8> {
        let mut out = Vec::new();
        t.file.seek(io::SeekFrom::Start(0)).unwrap();
        t.file.read_to_end(&mut out).unwrap();
        out
    }

    #[test]
    fn extracts_the_member() {
        let dir = crate::testutil::TempDir::new("fetch-extract");
        let dest = dir.path().join("m.onnx");
        let a = archive("dir/end2end.onnx", b"model bytes");
        let mut t = TempFile::create(&dest).unwrap();
        let (digest, len) = extract(Cursor::new(&a), "dir/end2end.onnx", 100, &mut t)
            .ok()
            .unwrap();
        assert_eq!(read_back(&mut t), b"model bytes");
        assert_eq!(len, 11);
        assert_eq!(digest, super::super::model::sha256_hex(b"model bytes"));
        for (archive, member, max) in [
            (&a[..], "other.onnx", 100),
            (&a[..], "dir/end2end.onnx", 4),
            (&b"not a zip"[..], "x", 100),
        ] {
            let mut t = TempFile::create(&dest).unwrap();
            assert!(matches!(
                extract(Cursor::new(archive), member, max, &mut t),
                Err(Copy::Read(_))
            ));
        }
    }

    #[test]
    fn temporary_files_are_unique_and_cleaned_up() {
        let dir = crate::testutil::TempDir::new("fetch-temp");
        let dest = dir.path().join("m.onnx");
        let a = TempFile::create(&dest).unwrap();
        let mut b = TempFile::create(&dest).unwrap();
        assert_ne!(a.path, b.path);
        let pid = format!("m.onnx.part-{}-", std::process::id());
        assert!(
            a.path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(&pid)
        );
        drop(a);
        b.file.write_all(b"done").unwrap();
        b.persist(&dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"done");
        // Only the destination is left.
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_verified_file_in_the_cache_is_used_without_network() {
        let dir = crate::testutil::TempDir::new("fetch-cached");
        let bytes = b"cached model".to_vec();
        let spec = ModelSpec {
            name: "cached",
            url: "https://invalid.example/never-fetched.onnx",
            sha256: Box::leak(super::super::model::sha256_hex(&bytes).into_boxed_str()),
            size: bytes.len() as u64,
            ..super::super::YOLOX_NANO
        };
        std::fs::write(spec.path_in(dir.path()), &bytes).unwrap();
        assert_eq!(
            fetch_model(&spec, dir.path()).unwrap(),
            spec.path_in(dir.path())
        );
    }

    #[test]
    fn a_copy_reports_its_progress_and_stops_when_told() {
        let dir = crate::testutil::TempDir::new("fetch-progress");
        let dest = dir.path().join("m.onnx");
        let data = vec![7u8; 200 * 1024];
        let mut t = TempFile::create(&dest).unwrap();
        let mut seen = Vec::new();
        let (digest, len) = copy_hashing(&mut Cursor::new(&data), &mut t, &mut |done| {
            seen.push(done);
            true
        })
        .ok()
        .unwrap();
        assert_eq!(len, data.len() as u64);
        assert_eq!(digest, super::super::model::sha256_hex(&data));
        assert_eq!(seen.last(), Some(&len));
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");
        // Told to stop after the first piece: no more is read.
        let mut t = TempFile::create(&dest).unwrap();
        let mut calls = 0;
        let stopped = copy_hashing(&mut Cursor::new(&data), &mut t, &mut |_| {
            calls += 1;
            false
        });
        assert!(matches!(stopped, Err(Copy::Cancelled)));
        assert_eq!(calls, 1);
        assert!(read_back(&mut t).len() < data.len());
    }

    #[test]
    fn a_cancelled_fetch_stops_before_any_request_and_writes_nothing() {
        let dir = crate::testutil::TempDir::new("fetch-cancel");
        let spec = ModelSpec {
            name: "cancelled",
            url: "https://invalid.example/never-fetched.onnx",
            ..super::super::YOLOX_NANO
        };
        let cancel = AtomicBool::new(true);
        let mut calls = 0;
        let mut progress = |_: FetchProgress| calls += 1;
        let e =
            fetch_model_with(&spec, dir.path(), Some(&mut progress), Some(&cancel)).unwrap_err();
        assert!(matches!(e, Error::Cancelled), "{e}");
        assert_eq!(calls, 0);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
        // A verified file is still returned, without a callback.
        let bytes = b"cached model".to_vec();
        let cached = ModelSpec {
            sha256: Box::leak(super::super::model::sha256_hex(&bytes).into_boxed_str()),
            size: bytes.len() as u64,
            ..spec
        };
        std::fs::write(cached.path_in(dir.path()), &bytes).unwrap();
        let mut progress = |_: FetchProgress| calls += 1;
        assert_eq!(
            fetch_model_with(&cached, dir.path(), Some(&mut progress), Some(&cancel)).unwrap(),
            cached.path_in(dir.path())
        );
        assert_eq!(calls, 0);
    }

    #[test]
    fn plain_http_is_refused_before_any_request() {
        let dir = crate::testutil::TempDir::new("fetch-http");
        let spec = ModelSpec {
            name: "insecure",
            url: "http://invalid.example/model.onnx",
            ..super::super::YOLOX_NANO
        };
        let e = fetch_model(&spec, dir.path()).unwrap_err();
        assert!(matches!(e, Error::Download { .. }), "{e}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
