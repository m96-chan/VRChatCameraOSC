//! Auto-download pretrained model weights (the MediaPipe YuNet / FaceMesh V2
//! / Blendshape V2 ONNX stack, issue #17) on first run (issue #12), so end
//! users don't need a Python/PyTorch toolchain.
//!
//! Weights are hosted as GitHub Release assets (`FACE_DETECTION_MODEL_URL` /
//! `FACE_LANDMARKS_MODEL_URL` / `FACE_BLENDSHAPES_MODEL_URL`) and fetched
//! over plain HTTPS.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Default local path for the YuNet face-detection ONNX model (issue #17).
/// Auto-downloaded from [`FACE_DETECTION_MODEL_URL`] on first run.
pub fn default_face_detection_path() -> PathBuf {
    PathBuf::from("models/face_detection.onnx")
}

/// Default local path for the MediaPipe FaceMesh V2 ONNX model (issue #17).
/// See [`default_face_detection_path`].
pub fn default_face_landmarks_path() -> PathBuf {
    PathBuf::from("models/face_landmarks.onnx")
}

/// Default local path for the MediaPipe Blendshape V2 ONNX model (issue
/// #17). See [`default_face_detection_path`].
pub fn default_face_blendshapes_path() -> PathBuf {
    PathBuf::from("models/face_blendshapes.onnx")
}

/// GitHub Release URL for the YuNet face-detection ONNX model, hosted on
/// this repo's `models-v1` release.
pub const FACE_DETECTION_MODEL_URL: &str =
    "https://github.com/m96-chan/VRChatCameraOSC/releases/download/models-v1/face_detection.onnx";
/// GitHub Release URL for the MediaPipe FaceMesh V2 ONNX model.
pub const FACE_LANDMARKS_MODEL_URL: &str =
    "https://github.com/m96-chan/VRChatCameraOSC/releases/download/models-v1/face_landmarks.onnx";
/// GitHub Release URL for the MediaPipe Blendshape V2 ONNX model.
pub const FACE_BLENDSHAPES_MODEL_URL: &str =
    "https://github.com/m96-chan/VRChatCameraOSC/releases/download/models-v1/face_blendshapes.onnx";

/// Download `url` to `path` if `path` doesn't already exist.
///
/// Writes to a sibling `.part` file first and renames into place on success,
/// so a failed or interrupted download never leaves a corrupt file at `path`.
///
/// # Errors
/// Returns a clear error (never panics) on a network failure or non-200
/// response; callers decide how to degrade (see `main.rs::build_tracker`).
pub fn ensure_present(path: &Path, url: &str) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let tmp_path = path.with_extension("part");
    download_to_file(url, &tmp_path)?;
    std::fs::rename(&tmp_path, path)
        .with_context(|| format!("renaming {} -> {}", tmp_path.display(), path.display()))?;
    Ok(())
}

fn download_to_file(url: &str, tmp_path: &Path) -> Result<()> {
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("downloading {url}"))?;
    let status = resp.status();
    if status != 200 {
        bail!("downloading {url}: unexpected HTTP status {status}");
    }
    let mut file = std::fs::File::create(tmp_path)
        .with_context(|| format!("creating {}", tmp_path.display()))?;
    std::io::copy(&mut resp.into_reader(), &mut file)
        .with_context(|| format!("writing {}", tmp_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Serve a single HTTP/1.1 response with `body` on an ephemeral local
    /// port and return its `http://127.0.0.1:PORT/...` base URL.
    fn serve_once(status_line: &str, body: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let status_line = status_line.to_string();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf); // drain the request, ignore it
            let header = format!(
                "{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).unwrap();
            stream.write_all(body).unwrap();
        });
        format!("http://{addr}/model.bin")
    }

    fn unique_temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vrchat_camera_osc_test_{name}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn ensure_present_downloads_when_missing() {
        let body: &'static [u8] = b"fake model bytes";
        let url = serve_once("HTTP/1.1 200 OK", body);
        let path = unique_temp_path("download");
        let _ = std::fs::remove_file(&path);

        ensure_present(&path, &url).unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), body);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn ensure_present_skips_when_already_present() {
        let path = unique_temp_path("skip");
        std::fs::write(&path, b"already here").unwrap();

        // A port nothing listens on: if this were actually requested, the
        // connection would fail and the test would error out.
        ensure_present(&path, "http://127.0.0.1:1/should-not-be-fetched").unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"already here");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn ensure_present_errors_on_non_200_and_leaves_no_file() {
        let url = serve_once("HTTP/1.1 404 Not Found", b"nope");
        let path = unique_temp_path("error");
        let _ = std::fs::remove_file(&path);

        let err = ensure_present(&path, &url).unwrap_err();

        // anyhow's `{}` only shows the top-level context; the underlying
        // ureq status is further down the chain, shown by `{:#}`.
        assert!(format!("{err:#}").contains("404"), "{err:#}");
        assert!(!path.exists());
        assert!(!path.with_extension("part").exists());
    }
}
