//! Deterministic synthetic [`CameraSource`] for tests and headless runs.
//!
//! [`FakeCamera`] never touches hardware: it generates reproducible frames from
//! a frame counter, so downstream stages and the TUI can be exercised without a
//! webcam. It is the default source used by tests.

use crate::capture::{CameraSource, Frame};
use anyhow::Result;

/// How a [`FakeCamera`] paints each synthetic frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    /// Every pixel is the same value, cycling per frame — cheapest, and easy to
    /// assert on.
    Solid,
    /// A horizontal/vertical gradient that shifts by the frame counter, so
    /// successive frames differ.
    Gradient,
}

/// A synthetic camera producing deterministic [`Frame`]s.
#[derive(Debug, Clone)]
pub struct FakeCamera {
    width: u32,
    height: u32,
    pattern: Pattern,
    frame_index: u64,
}

impl FakeCamera {
    /// Create a fake camera of the given size using [`Pattern::Gradient`].
    pub fn new(width: u32, height: u32) -> Self {
        Self::with_pattern(width, height, Pattern::Gradient)
    }

    /// Create a fake camera with an explicit [`Pattern`].
    pub fn with_pattern(width: u32, height: u32, pattern: Pattern) -> Self {
        Self {
            width,
            height,
            pattern,
            frame_index: 0,
        }
    }

    /// Number of frames produced so far.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    fn render(&self, index: u64) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut data = Vec::with_capacity(w * h * 3);
        let phase = (index & 0xff) as u8;
        match self.pattern {
            Pattern::Solid => {
                data.resize(w * h * 3, phase);
            }
            Pattern::Gradient => {
                for y in 0..h {
                    for x in 0..w {
                        let r = (x as u8).wrapping_add(phase);
                        let g = (y as u8).wrapping_add(phase);
                        let b = phase;
                        data.push(r);
                        data.push(g);
                        data.push(b);
                    }
                }
            }
        }
        data
    }
}

impl CameraSource for FakeCamera {
    fn next_frame(&mut self) -> Result<Frame> {
        let data = self.render(self.frame_index);
        self.frame_index += 1;
        Frame::new(self.width, self.height, data)
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_matches_construction() {
        let cam = FakeCamera::new(640, 480);
        assert_eq!(cam.resolution(), (640, 480));
    }

    #[test]
    fn frames_are_correctly_sized() {
        let mut cam = FakeCamera::new(8, 4);
        let frame = cam.next_frame().unwrap();
        assert_eq!(frame.width, 8);
        assert_eq!(frame.height, 4);
        assert_eq!(frame.data.len(), Frame::expected_len(8, 4));
    }

    #[test]
    fn advances_and_is_deterministic() {
        let mut a = FakeCamera::new(4, 4);
        let mut b = FakeCamera::new(4, 4);
        assert_eq!(a.frame_index(), 0);

        let a0 = a.next_frame().unwrap();
        let a1 = a.next_frame().unwrap();
        let b0 = b.next_frame().unwrap();
        let b1 = b.next_frame().unwrap();

        // Deterministic across independent instances.
        assert_eq!(a0, b0);
        assert_eq!(a1, b1);
        // Successive frames differ (gradient shifts by frame index).
        assert_ne!(a0, a1);
        assert_eq!(a.frame_index(), 2);
    }

    #[test]
    fn solid_pattern_is_uniform_and_cycles() {
        let mut cam = FakeCamera::with_pattern(3, 2, Pattern::Solid);
        let f0 = cam.next_frame().unwrap();
        assert!(f0.data.iter().all(|&b| b == 0));
        let f1 = cam.next_frame().unwrap();
        assert!(f1.data.iter().all(|&b| b == 1));
        assert_ne!(f0, f1);
    }

    #[test]
    fn is_usable_as_trait_object() {
        let mut cam: Box<dyn CameraSource> = Box::new(FakeCamera::new(2, 2));
        assert_eq!(cam.resolution(), (2, 2));
        assert!(cam.next_frame().is_ok());
    }
}
