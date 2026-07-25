//! Landmarks → VRChat avatar OSC parameters.
//!
//! Turns iBUG-68 facial landmarks into normalised avatar parameters (mouth
//! open, blink, brows, head yaw/pitch/roll), with clamping and smoothing.
//! Implemented by issue #5.

use crate::osc::OscParam;
use crate::tracking::FaceLandmarks;

/// Maps landmarks to OSC parameters. Holds smoothing state between frames.
#[derive(Debug, Default)]
pub struct Mapper {
    // smoothing state added in issue #5
}

impl Mapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Produce the OSC parameter updates for one frame of landmarks.
    pub fn map(&mut self, _landmarks: &FaceLandmarks) -> Vec<OscParam> {
        // Implemented by issue #5.
        Vec::new()
    }
}
