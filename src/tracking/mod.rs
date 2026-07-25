//! Face tracking behind a pluggable [`FaceTracker`] backend.
//!
//! The default backend is [`fan`] — a pure-Rust (candle) port of the
//! face-alignment FAN network, validated for numeric parity against the PyTorch
//! `face_alignment` reference (issue #4).

use crate::capture::Frame;
use anyhow::Result;

/// The FAN 2D model predicts 68 landmarks.
pub const NUM_LANDMARKS: usize = 68;

/// A single 2D facial landmark in the coordinate frame of the input image
/// (pixels), plus the heatmap confidence at that point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// The full set of facial landmarks for one detected face.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceLandmarks {
    /// Exactly [`NUM_LANDMARKS`] points, in the standard iBUG-68 ordering.
    pub points: Vec<Landmark>,
}

impl FaceLandmarks {
    pub fn new(points: Vec<Landmark>) -> Result<Self> {
        anyhow::ensure!(
            points.len() == NUM_LANDMARKS,
            "expected {} landmarks, got {}",
            NUM_LANDMARKS,
            points.len()
        );
        Ok(Self { points })
    }

    /// Mean confidence across all points (0..1-ish; FAN scores are heatmap peaks).
    pub fn mean_score(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.points.iter().map(|p| p.score).sum::<f32>() / self.points.len() as f32
    }
}

/// A face tracker: given a frame, produce landmarks (or `None` if no face).
pub trait FaceTracker {
    fn track(&mut self, frame: &Frame) -> Result<Option<FaceLandmarks>>;
}

pub mod fan;
