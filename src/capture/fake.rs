//! Synthetic camera source for tests and headless/CI runs (no real webcam).

use super::{CameraSource, Frame};
use anyhow::Result;

/// A deterministic synthetic camera. Each `next_frame` returns a solid-colour
/// frame whose colour advances by `step` per call, so tests can assert motion.
pub struct FakeCamera {
    width: u32,
    height: u32,
    tick: u8,
    step: u8,
}

impl FakeCamera {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            tick: 0,
            step: 1,
        }
    }
}

impl CameraSource for FakeCamera {
    fn next_frame(&mut self) -> Result<Frame> {
        let v = self.tick;
        self.tick = self.tick.wrapping_add(self.step);
        let data = vec![v; self.width as usize * self.height as usize * 3];
        Frame::new(self.width, self.height, data)
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
