//! Small pixel helpers shared by the MediaPipe face stages.
//!
//! Ported from AvataCam `crates/face/src/backends/mediapipe/util.rs`, adapted
//! to operate on VRChatCameraOSC's packed-RGB8 `Frame` layout directly
//! (`width`, `height`, `&[u8]`) instead of `avatacam_core::ImageView`. The two
//! layouts are identical (packed RGB8, row-major, no padding), so the math is
//! unchanged.

/// Numerically stable enough for our range (logits from a trained net); no
/// need for the `libm`-style overflow guards a general-purpose sigmoid would want.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Bilinear sample of a packed-RGB8 frame at float pixel `(x, y)`; out-of-bounds
/// reads clamp to the edge. Returns RGB in `[0, 255]` (float).
pub fn sample_rgb255(width: u32, height: u32, rgb: &[u8], x: f32, y: f32) -> [f32; 3] {
    let w = width as i32;
    let h = height as i32;
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let at = |xi: i32, yi: i32, c: usize| -> f32 {
        let xc = xi.clamp(0, w - 1);
        let yc = yi.clamp(0, h - 1);
        rgb[((yc * w + xc) * 3) as usize + c] as f32
    };
    let mut out = [0.0f32; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let top = at(x0, y0, c) * (1.0 - fx) + at(x0 + 1, y0, c) * fx;
        let bot = at(x0, y0 + 1, c) * (1.0 - fx) + at(x0 + 1, y0 + 1, c) * fx;
        *o = top * (1.0 - fy) + bot * fy;
    }
    out
}

/// Intersection-over-union of two `[x1,y1,x2,y2]` boxes.
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_basic() {
        let a = [0.0, 0.0, 2.0, 2.0];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
        let b = [1.0, 1.0, 3.0, 3.0];
        // inter 1, union 7 -> 1/7.
        assert!((iou(&a, &b) - 1.0 / 7.0).abs() < 1e-6);
        let c = [5.0, 5.0, 6.0, 6.0];
        assert_eq!(iou(&a, &c), 0.0);
    }

    #[test]
    fn sigmoid_bounds_and_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(50.0) > 0.999);
        assert!(sigmoid(-50.0) < 0.001);
    }

    #[test]
    fn sample_rgb255_reads_exact_pixel_center() {
        // 2x2 image, distinct colors per pixel.
        let rgb = [
            255, 0, 0, // (0,0) red
            0, 255, 0, // (1,0) green
            0, 0, 255, // (0,1) blue
            255, 255, 0, // (1,1) yellow
        ];
        let px = sample_rgb255(2, 2, &rgb, 0.0, 0.0);
        assert_eq!(px, [255.0, 0.0, 0.0]);
    }

    #[test]
    fn sample_rgb255_clamps_out_of_bounds() {
        let rgb = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let inside = sample_rgb255(2, 2, &rgb, 0.0, 0.0);
        let outside = sample_rgb255(2, 2, &rgb, -5.0, -5.0);
        assert_eq!(inside, outside, "negative coords should clamp to (0,0)");
    }
}
