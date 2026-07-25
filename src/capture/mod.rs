//! Camera capture pipeline.
//!
//! The pipeline is split into:
//! - [`Frame`] / [`CameraSource`] — the platform-agnostic capture contract every
//!   downstream stage (face mesh, hand tracking) consumes.
//! - [`fake`] — a deterministic synthetic source for tests and headless runs.
//! - [`convert`] — pure pixel-buffer -> [`Frame`] helpers (unit-tested with
//!   synthetic buffers; no camera required).
//! - [`fps`] — an injectable-time FPS counter for the realtime loop / TUI.
//! - `native` — the real backend (AVFoundation on macOS, V4L2 on Linux;
//!   `camera` feature only).
//!
//! Platform-specific code stays behind [`CameraSource`] so other OS backends
//! slot in without churn.

use anyhow::{bail, Result};

pub mod convert;
pub mod fake;
pub mod fps;

#[cfg(all(any(target_os = "macos", target_os = "linux"), feature = "camera"))]
pub mod native;

pub use convert::{rgb8_to_frame, rgba8_to_frame};
pub use fake::{FakeCamera, Pattern};
pub use fps::FpsCounter;

/// A single captured video frame: packed 8-bit RGB, row-major, no padding.
///
/// `data.len()` is always exactly `width * height * 3`.
#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Packed RGB8 pixels (`R,G,B,R,G,B,...`), length `width * height * 3`.
    pub data: Vec<u8>,
}

impl Frame {
    /// Build a [`Frame`], validating that `data` holds exactly one RGB8 pixel
    /// per `(x, y)`.
    ///
    /// # Errors
    /// Returns an error if `data.len() != width * height * 3`.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected = Self::expected_len(width, height);
        if data.len() != expected {
            bail!(
                "frame buffer size mismatch: expected {expected} bytes for {width}x{height} RGB8, got {}",
                data.len()
            );
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Number of bytes a packed RGB8 buffer of `width * height` pixels needs.
    #[inline]
    pub fn expected_len(width: u32, height: u32) -> usize {
        width as usize * height as usize * 3
    }

    /// Total pixel count (`width * height`).
    #[inline]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Avoid dumping the whole pixel buffer.
        f.debug_struct("Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.data.len())
            .finish()
    }
}

/// Metadata for an enumerated capture device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraInfo {
    /// Backend device index, usable with the backend's `open(index, ..)`.
    pub index: u32,
    /// Human-readable device name.
    pub name: String,
}

/// A pull-based source of [`Frame`]s.
///
/// Implementors hide the platform capture backend (AVFoundation, a synthetic
/// generator, ...). The realtime loop calls [`CameraSource::next_frame`] to get
/// the latest frame and [`CameraSource::resolution`] to size downstream stages.
pub trait CameraSource {
    /// Pull the next frame. Blocks only as long as the backend needs to produce
    /// one; returns a clear error rather than panicking on failure.
    fn next_frame(&mut self) -> Result<Frame>;

    /// The `(width, height)` of frames this source produces.
    fn resolution(&self) -> (u32, u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_new_accepts_correct_length() {
        let frame = Frame::new(2, 3, vec![0u8; 2 * 3 * 3]).unwrap();
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 3);
        assert_eq!(frame.data.len(), 18);
        assert_eq!(frame.pixel_count(), 6);
    }

    #[test]
    fn frame_new_rejects_too_short() {
        let err = Frame::new(2, 2, vec![0u8; 11]).unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
    }

    #[test]
    fn frame_new_rejects_too_long() {
        assert!(Frame::new(2, 2, vec![0u8; 13]).is_err());
    }

    #[test]
    fn frame_new_accepts_zero_sized() {
        // Degenerate but valid: 0 pixels => 0 bytes.
        let frame = Frame::new(0, 0, Vec::new()).unwrap();
        assert_eq!(frame.pixel_count(), 0);
    }

    #[test]
    fn frame_debug_is_compact() {
        let frame = Frame::new(2, 2, vec![7u8; 12]).unwrap();
        let s = format!("{frame:?}");
        // Reports byte count, not the raw buffer contents.
        assert!(s.contains("bytes: 12"), "{s}");
        assert!(s.contains("width: 2"), "{s}");
    }

    #[test]
    fn camera_info_is_constructible() {
        let info = CameraInfo {
            index: 3,
            name: "cam".to_string(),
        };
        assert_eq!(info.index, 3);
        assert_eq!(info, info.clone());
    }

    #[test]
    fn expected_len_matches() {
        assert_eq!(Frame::expected_len(4, 5), 60);
        assert_eq!(Frame::expected_len(0, 100), 0);
    }
}
