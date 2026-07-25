//! Camera capture behind a platform-agnostic [`CameraSource`] trait.
//!
//! macOS (AVFoundation via `nokhwa`) is the primary target; the trait keeps the
//! platform backend swappable and lets tests run against a synthetic source.

use anyhow::Result;

/// A single captured frame in packed RGB8 (row-major, 3 bytes/pixel).
#[derive(Clone, PartialEq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// `width * height * 3` bytes, RGB order.
    pub data: Vec<u8>,
}

impl Frame {
    /// Construct a frame, validating the buffer length matches `width*height*3`.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected = width as usize * height as usize * 3;
        anyhow::ensure!(
            data.len() == expected,
            "frame buffer length {} != expected {} for {}x{} RGB8",
            data.len(),
            expected,
            width,
            height
        );
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.data.len())
            .finish()
    }
}

/// A source of camera frames. Implementations must be non-blocking-friendly:
/// `next_frame` returns the latest available frame.
pub trait CameraSource {
    /// Pull the next frame. May block briefly waiting for the camera.
    fn next_frame(&mut self) -> Result<Frame>;
    /// Current capture resolution `(width, height)`.
    fn resolution(&self) -> (u32, u32);
}

pub mod fake;
#[cfg(all(target_os = "macos", feature = "camera"))]
pub mod macos;
