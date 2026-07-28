//! Rotated hand ROI geometry for the hand-landmark stage (issue #8).
//!
//! Mirrors `mediapipe/roi.rs` in spirit, but the math is ported from OpenCV
//! Zoo's MediaPipe-Hands pipeline
//! (`models/handpose_estimation_mediapipe/mp_handpose.py`, Apache-2.0):
//!
//! - **Rotation**: the crop is rotated so the hand is vertical with the
//!   fingers up — the "up" direction is palm base (wrist) → middle-finger
//!   MCP (`PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0`,
//!   `PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2`; their
//!   `radians = π/2 − atan2(−(p2.y−p1.y), p2.x−p1.x)` is exactly this
//!   direction expressed as an image-rotation angle).
//! - **Box**: the bounding box of the 7 palm keypoints *in rotated space*,
//!   shifted by `PALM_BOX_SHIFT_VECTOR = [0, -0.4]` (i.e. 0.4 box-heights
//!   toward the fingers — the palm sits at the bottom of the hand), then
//!   enlarged by `PALM_BOX_ENLARGE_FACTOR = 3` and padded to a square on the
//!   long side. Their pre-crop (`PALM_BOX_PRE_*`, enlarge 4, diagonal pad)
//!   only gives `warpAffine` working room and has no continuous-math
//!   equivalent to port — our sampler warps directly from the source frame.
//!
//! Between palm detections the next frame's ROI comes from the previous
//! frame's 21 landmarks: the same construction applied to their 7-point palm
//! subset (`PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2]` — wrist, the four
//! finger MCPs, thumb CMC + MCP), so a tracked crop is built identically to
//! a detector-seeded one. This is MediaPipe's own VIDEO-mode detect-then-track
//! design, and the same pattern as the face stack (issue #17).

/// The palm-keypoint subset of the 21 hand landmarks, in the palm detector's
/// 7-keypoint order (OpenCV Zoo `mp_handpose.py` `PALM_LANDMARK_IDS`):
/// wrist, index MCP, middle MCP, ring MCP, pinky MCP, thumb CMC, thumb MCP.
pub const PALM_LANDMARK_IDS: [usize; 7] = [0, 5, 9, 13, 17, 1, 2];

/// `PALM_BOX_SHIFT_VECTOR = [0, -0.4]` (OpenCV Zoo): the crop center moves
/// 0.4 palm-box-heights from the palm toward the fingers.
const SHIFT_TOWARD_FINGERS: f32 = 0.4;

/// `PALM_BOX_ENLARGE_FACTOR = 3` (OpenCV Zoo): the palm-keypoint box grows
/// 3x, then squares on the long side, to cover the whole hand.
const ENLARGE: f32 = 3.0;

/// A rotated square region of interest over a hand, in pixel coordinates —
/// the same shape as `mediapipe::FaceRoi` (`up` points from the wrist toward
/// the fingers, so the warped crop has the fingers at the top).
#[derive(Debug, Clone, Copy)]
pub struct HandRoi {
    pub cx: f32,
    pub cy: f32,
    /// Side length of the square crop.
    pub size: f32,
    pub right: [f32; 2],
    pub up: [f32; 2],
}

/// Build the hand ROI from the 7 palm keypoints (pixels), in the detector's
/// order (see [`PALM_LANDMARK_IDS`] for the semantic order). Ports OpenCV
/// Zoo's rotate → landmark-bbox → shift `[0,-0.4]` → enlarge ×3 → square
/// sequence (see the module docs).
pub fn hand_roi_from_palm_points(kp: &[[f32; 2]; 7]) -> HandRoi {
    // Rotation: wrist (kp 0) -> middle-finger MCP (kp 2) is "up".
    let (p0, p2) = (kp[0], kp[2]);
    let mut up = [p2[0] - p0[0], p2[1] - p0[1]];
    let len = (up[0] * up[0] + up[1] * up[1]).sqrt();
    if len > 1e-3 {
        up = [up[0] / len, up[1] / len];
    } else {
        up = [0.0, -1.0]; // degenerate: assume upright (image up)
    }
    let right = [-up[1], up[0]];

    // Bounding box of the palm keypoints in rotated coordinates
    // (s = along `right`, t = along `up`, i.e. toward the fingers).
    let (mut s_min, mut s_max) = (f32::MAX, f32::MIN);
    let (mut t_min, mut t_max) = (f32::MAX, f32::MIN);
    for p in kp {
        let s = p[0] * right[0] + p[1] * right[1];
        let t = p[0] * up[0] + p[1] * up[1];
        s_min = s_min.min(s);
        s_max = s_max.max(s);
        t_min = t_min.min(t);
        t_max = t_max.max(t);
    }
    let (w, h) = (s_max - s_min, t_max - t_min);
    let center_s = 0.5 * (s_min + s_max);
    // OpenCV Zoo shifts the box by [0, -0.4]·h in y-down rotated-image
    // coords — toward the fingers, which is +t here.
    let center_t = 0.5 * (t_min + t_max) + SHIFT_TOWARD_FINGERS * h;
    let size = (ENLARGE * w.max(h)).max(1.0);

    // Back to image coordinates: p = s·right + t·up (orthonormal basis).
    HandRoi {
        cx: center_s * right[0] + center_t * up[0],
        cy: center_s * right[1] + center_t * up[1],
        size,
        right,
        up,
    }
}

/// Next-frame ROI from the previous frame's 21 landmarks (normalized frame
/// coordinates) — the [`PALM_LANDMARK_IDS`] subset through the same palm-box
/// construction, so tracked and detector-seeded crops are built identically.
pub fn hand_roi_from_landmarks(points: &[[f32; 3]; 21], w: u32, h: u32) -> HandRoi {
    let (wf, hf) = (w as f32, h as f32);
    let mut kp = [[0.0f32; 2]; 7];
    for (slot, &idx) in PALM_LANDMARK_IDS.iter().enumerate() {
        kp[slot] = [points[idx][0] * wf, points[idx][1] * hf];
    }
    hand_roi_from_palm_points(&kp)
}

/// Whether a tracked hand ROI is still usable (mirrors the face stack's
/// `roi_in_frame`): center inside the frame, size plausible, nothing NaN.
pub fn roi_in_frame(r: &HandRoi, w: u32, h: u32) -> bool {
    let (wf, hf) = (w as f32, h as f32);
    r.cx.is_finite()
        && r.cy.is_finite()
        && r.size.is_finite()
        && r.cx >= 0.0
        && r.cx < wf
        && r.cy >= 0.0
        && r.cy < hf
        && r.size > 8.0
        && r.size < 4.0 * wf.max(hf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An upright palm (fingers toward image top): wrist at the bottom,
    /// MCP row above it, thumb points off to one side.
    fn upright_palm() -> [[f32; 2]; 7] {
        [
            [100.0, 200.0], // wrist
            [80.0, 140.0],  // index MCP
            [100.0, 140.0], // middle MCP
            [112.0, 142.0], // ring MCP
            [124.0, 146.0], // pinky MCP
            [76.0, 186.0],  // thumb CMC
            [70.0, 170.0],  // thumb MCP
        ]
    }

    #[test]
    fn upright_palm_gives_up_toward_fingers_and_square() {
        let roi = hand_roi_from_palm_points(&upright_palm());
        // wrist -> middle MCP is straight up in the image (y decreasing).
        assert!(
            roi.up[0].abs() < 1e-6 && (roi.up[1] + 1.0).abs() < 1e-6,
            "up {:?}",
            roi.up
        );
        assert!((roi.right[0] - 1.0).abs() < 1e-6 && roi.right[1].abs() < 1e-6);
        // Keypoint box: x 70..124 (w=54), y 140..200 (h=60);
        // size = 3 * max(54, 60) = 180.
        assert!((roi.size - 180.0).abs() < 1e-3, "size {}", roi.size);
        // Center: x = (70+124)/2 = 97; y = (140+200)/2 shifted 0.4*60 = 24
        // toward the fingers (up, i.e. smaller y): 170 - 24 = 146.
        assert!((roi.cx - 97.0).abs() < 1e-3, "cx {}", roi.cx);
        assert!((roi.cy - 146.0).abs() < 1e-3, "cy {}", roi.cy);
    }

    /// The construction must be rotation-equivariant: rotating the input
    /// points rotates the ROI (same size, rotated center/axes).
    #[test]
    fn rotated_palm_rotates_the_roi_with_it() {
        let base = upright_palm();
        let upright = hand_roi_from_palm_points(&base);

        // Rotate everything 90° clockwise (in y-down image coords) about the
        // origin: (x, y) -> (-y, x).
        let rotated: Vec<[f32; 2]> = base.iter().map(|p| [-p[1], p[0]]).collect();
        let rotated: [[f32; 2]; 7] = rotated.try_into().unwrap();
        let roi = hand_roi_from_palm_points(&rotated);

        assert!((roi.size - upright.size).abs() < 1e-3);
        // Center follows the same rotation.
        assert!((roi.cx - -upright.cy).abs() < 1e-3, "cx {}", roi.cx);
        assert!((roi.cy - upright.cx).abs() < 1e-3, "cy {}", roi.cy);
        // Up follows too: (0,-1) -> (1, 0).
        assert!((roi.up[0] - 1.0).abs() < 1e-6 && roi.up[1].abs() < 1e-6);
    }

    #[test]
    fn landmark_subset_matches_direct_palm_points() {
        // Normalized 21-landmark set over a 200x200 frame whose palm subset
        // reproduces `upright_palm` exactly; all other landmarks are placed
        // far away to prove they are ignored.
        let palm = upright_palm();
        let mut pts = [[5.0f32, 5.0, 0.0]; 21];
        for (slot, &idx) in PALM_LANDMARK_IDS.iter().enumerate() {
            pts[idx] = [palm[slot][0] / 200.0, palm[slot][1] / 200.0, 0.0];
        }
        let from_landmarks = hand_roi_from_landmarks(&pts, 200, 200);
        let direct = hand_roi_from_palm_points(&palm);
        assert!((from_landmarks.cx - direct.cx).abs() < 1e-3);
        assert!((from_landmarks.cy - direct.cy).abs() < 1e-3);
        assert!((from_landmarks.size - direct.size).abs() < 1e-3);
    }

    #[test]
    fn degenerate_rotation_defaults_to_upright() {
        let mut kp = upright_palm();
        kp[2] = kp[0]; // middle MCP collapsed onto the wrist
        let roi = hand_roi_from_palm_points(&kp);
        assert!((roi.up[1] + 1.0).abs() < 1e-6, "up {:?}", roi.up);
    }

    #[test]
    fn roi_in_frame_rejects_offframe_and_degenerate() {
        let ok = hand_roi_from_palm_points(&upright_palm());
        assert!(roi_in_frame(&ok, 320, 240));

        let mut off = ok;
        off.cx = 500.0;
        assert!(!roi_in_frame(&off, 320, 240));

        let mut tiny = ok;
        tiny.size = 2.0;
        assert!(!roi_in_frame(&tiny, 320, 240));

        let mut nan = ok;
        nan.cy = f32::NAN;
        assert!(!roi_in_frame(&nan, 320, 240));
    }
}
