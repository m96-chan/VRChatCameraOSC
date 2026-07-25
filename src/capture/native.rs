//! Native webcam backend — AVFoundation on macOS, V4L2 on Linux (gated on
//! `target_os` + the `camera` feature; see `Cargo.toml`'s per-target
//! `nokhwa` dependencies).
//!
//! This is intentionally thin: it wires nokhwa's per-platform backend (picked
//! automatically via [`nokhwa::utils::ApiBackend::Auto`]) to the
//! [`CameraSource`] contract and pushes all the testable logic (pixel
//! packing, validation) into [`crate::capture::convert`]. It cannot be
//! unit-tested in CI (no webcam), so correctness of the moving parts lives in
//! those pure helpers.
//!
//! Failure modes — no camera present and permission denied — surface as clear
//! [`anyhow`] errors; this module never panics on them.

use crate::capture::{convert, CameraInfo, CameraSource, Frame};
use anyhow::{anyhow, Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use nokhwa::Camera;

/// A live webcam opened through the platform's native backend.
pub struct NativeCamera {
    camera: Camera,
    width: u32,
    height: u32,
}

impl NativeCamera {
    /// Enumerate available capture devices on the native backend.
    ///
    /// # Errors
    /// Returns an error if the platform query fails (e.g. permission not yet
    /// granted).
    pub fn list_devices() -> Result<Vec<CameraInfo>> {
        request_permission();
        let devices = nokhwa::query(ApiBackend::Auto)
            .map_err(|e| anyhow!("failed to enumerate cameras: {e}"))?;
        Ok(devices
            .into_iter()
            .map(|info| CameraInfo {
                index: info.index().as_index().unwrap_or(0),
                name: info.human_name(),
            })
            .collect())
    }

    /// Open the camera at `index`, requesting the closest available format to
    /// `width`x`height` that can decode to RGB8, and start streaming.
    ///
    /// # Errors
    /// Returns a clear error (never panics) when there is no camera at `index`,
    /// when camera permission is denied, or when the stream cannot start.
    pub fn open(index: u32, width: u32, height: u32) -> Result<Self> {
        request_permission();

        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestResolution(
            Resolution::new(width, height),
        ));

        let mut camera = Camera::new(CameraIndex::Index(index), requested).map_err(|e| {
            anyhow!(
                "failed to open camera {index} (no such camera, or camera permission denied): {e}"
            )
        })?;

        camera
            .open_stream()
            .map_err(|e| anyhow!("failed to start camera {index} stream: {e}"))?;

        let res = camera.resolution();
        Ok(Self {
            camera,
            width: res.width(),
            height: res.height(),
        })
    }
}

impl CameraSource for NativeCamera {
    fn next_frame(&mut self) -> Result<Frame> {
        let buffer = self
            .camera
            .frame()
            .map_err(|e| anyhow!("failed to capture frame: {e}"))?;
        let decoded = buffer
            .decode_image::<RgbFormat>()
            .map_err(|e| anyhow!("failed to decode frame to RGB8: {e}"))?;
        let (w, h) = (decoded.width(), decoded.height());
        // Keep the reported resolution in sync with what actually arrives.
        self.width = w;
        self.height = h;
        convert::rgb8_to_frame(w, h, decoded.as_raw())
            .context("decoded RGB8 buffer did not match its reported dimensions")
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Request camera permission where the platform needs one (macOS). nokhwa
/// resolves the prompt via a callback; a denied/pending grant surfaces later
/// as an open/query error, which we translate into a clear message. On
/// backends without a permission model (e.g. V4L2 on Linux) this is a no-op.
fn request_permission() {
    if !nokhwa::nokhwa_check() {
        nokhwa::nokhwa_initialize(|_granted| {});
    }
}
