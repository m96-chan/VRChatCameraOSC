//! Face detection — find the face bounding box(es) before FAN crops around them
//! (issue #9). The default backend is [`sfd`] — a pure-Rust (candle) port of the
//! S3FD detector used by the `face_alignment` reference.

use image::RgbImage;

pub mod sfd;

/// An axis-aligned face bounding box in image pixels, with a detection score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl BoundingBox {
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }
}

/// A face detector: given an RGB image, return the detected face boxes
/// (already thresholded and NMS-filtered), strongest first is not guaranteed —
/// callers pick by score.
pub trait FaceDetector {
    fn detect(&self, image: &RgbImage) -> anyhow::Result<Vec<BoundingBox>>;
}
