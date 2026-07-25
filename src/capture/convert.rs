//! Pure pixel-buffer conversion helpers.
//!
//! These take raw byte slices + dimensions and return a validated [`Frame`], so
//! they are unit-testable with synthetic buffers and share the exact validation
//! the real camera backends rely on. Keeping the conversion here (rather than in
//! a backend) is what lets the untestable camera code stay thin.

use crate::capture::Frame;
use anyhow::{bail, Result};

/// Wrap a packed RGB8 buffer (`R,G,B,...`) into a [`Frame`], validating length.
///
/// # Errors
/// Returns an error if `data.len() != width * height * 3`.
pub fn rgb8_to_frame(width: u32, height: u32, data: &[u8]) -> Result<Frame> {
    Frame::new(width, height, data.to_vec())
}

/// Convert a packed RGBA8 buffer (`R,G,B,A,...`) into an RGB8 [`Frame`],
/// dropping the alpha channel.
///
/// # Errors
/// Returns an error if `data.len() != width * height * 4`.
pub fn rgba8_to_frame(width: u32, height: u32, data: &[u8]) -> Result<Frame> {
    let expected = width as usize * height as usize * 4;
    if data.len() != expected {
        bail!(
            "RGBA buffer size mismatch: expected {expected} bytes for {width}x{height} RGBA8, got {}",
            data.len()
        );
    }
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for px in data.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    Frame::new(width, height, rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb8_synthetic_roundtrips_bytes() {
        let data = vec![1, 2, 3, 4, 5, 6]; // 2 pixels
        let frame = rgb8_to_frame(2, 1, &data).unwrap();
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 1);
        assert_eq!(frame.data, data);
    }

    #[test]
    fn rgb8_wrong_size_errors() {
        assert!(rgb8_to_frame(2, 2, &[0u8; 10]).is_err());
    }

    #[test]
    fn rgba8_strips_alpha() {
        // 2 pixels: (10,20,30,255) and (40,50,60,0)
        let data = vec![10, 20, 30, 255, 40, 50, 60, 0];
        let frame = rgba8_to_frame(2, 1, &data).unwrap();
        assert_eq!(frame.data, vec![10, 20, 30, 40, 50, 60]);
        assert_eq!(Frame::expected_len(2, 1), frame.data.len());
    }

    #[test]
    fn rgba8_wrong_size_errors() {
        // 6 bytes is not a whole number of RGBA pixels for 2x1.
        let err = rgba8_to_frame(2, 1, &[0u8; 6]).unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
    }
}
