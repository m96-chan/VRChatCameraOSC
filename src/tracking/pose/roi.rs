//! Rotated person ROI geometry for the pose-landmark stage (issue #28
//! phase 2). Mirrors `hands/roi.rs`; the math is ported from OpenCV Zoo
//! `models/pose_estimation_mediapipe/mp_pose.py` (Apache-2.0):
//!
//! - **Construction**: the ROI is defined by a *center* point and a *scale
//!   point* — the crop is the square of side `2·dist(center, scale_point)`
//!   centered on `center` (their `full_bbox = [mid_hip ± full_dist]`,
//!   `PERSON_BOX_PRE_ENLARGE_FACTOR = 1`), rotated so `center → scale_point`
//!   is "up" (their `radians = π/2 − atan2(−(p1.y−p0.y), p1.x−p0.x)` — the
//!   identical formula the hand ROI ports).
//! - **Seeding**: the person detector's keypoint 0 (mid-hip center) and
//!   keypoint 1 (full-body point) are exactly this center/scale-point pair.
//! - **Tracking**: between detections the next frame's ROI comes from the
//!   previous frame's auxiliary landmarks 33 (body center) and 34 (scale
//!   point) through the same construction — MediaPipe's own VIDEO-mode
//!   detect-then-track design (its `pose_landmarks_to_roi` uses keypoints
//!   33/34), the same pattern as hands and the face stack.

/// A rotated square region of interest over a person, in pixel coordinates —
/// the same shape as `hands::HandRoi` (`up` points from the hips toward the
/// head, so the warped crop has the head at the top).
#[derive(Debug, Clone, Copy)]
pub struct PoseRoi {
    pub cx: f32,
    pub cy: f32,
    /// Side length of the square crop.
    pub size: f32,
    pub right: [f32; 2],
    pub up: [f32; 2],
}

/// Build the person ROI from its center point (mid-hip) and scale point
/// (full-body point): square of side `2·dist`, rotated so center → scale
/// point is "up" (see the module docs for the mp_pose.py provenance).
pub fn pose_roi_from_center_scale(center: [f32; 2], scale_point: [f32; 2]) -> PoseRoi {
    let mut up = [scale_point[0] - center[0], scale_point[1] - center[1]];
    let len = (up[0] * up[0] + up[1] * up[1]).sqrt();
    if len > 1e-3 {
        up = [up[0] / len, up[1] / len];
    } else {
        up = [0.0, -1.0]; // degenerate: assume upright (image up)
    }
    let right = [-up[1], up[0]];
    PoseRoi {
        cx: center[0],
        cy: center[1],
        size: (2.0 * len).max(1.0),
        right,
        up,
    }
}

/// Whether a tracked person ROI is still usable (mirrors
/// `hands::roi::roi_in_frame`): center inside the frame, size plausible,
/// nothing NaN.
pub fn roi_in_frame(r: &PoseRoi, w: u32, h: u32) -> bool {
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

    #[test]
    fn upright_person_gives_up_toward_head_and_square() {
        // Hips at (100, 200), full-body point 80 px straight above (head
        // direction in image coords = smaller y).
        let roi = pose_roi_from_center_scale([100.0, 200.0], [100.0, 120.0]);
        assert!(
            roi.up[0].abs() < 1e-6 && (roi.up[1] + 1.0).abs() < 1e-6,
            "up {:?}",
            roi.up
        );
        assert!((roi.right[0] - 1.0).abs() < 1e-6 && roi.right[1].abs() < 1e-6);
        assert!((roi.cx - 100.0).abs() < 1e-3);
        assert!((roi.cy - 200.0).abs() < 1e-3);
        // size = 2 * dist = 160 (mp_pose.py: full_bbox spans mid_hip ± dist).
        assert!((roi.size - 160.0).abs() < 1e-3, "size {}", roi.size);
    }

    /// Rotation-equivariance: rotating the two defining points rotates the
    /// ROI with them (same size, rotated center/axes).
    #[test]
    fn rotated_person_rotates_the_roi_with_it() {
        let upright = pose_roi_from_center_scale([100.0, 200.0], [100.0, 120.0]);
        // 90° clockwise in y-down image coords: (x, y) -> (-y, x).
        let roi = pose_roi_from_center_scale([-200.0, 100.0], [-120.0, 100.0]);
        assert!((roi.size - upright.size).abs() < 1e-3);
        assert!((roi.cx - -upright.cy).abs() < 1e-3);
        assert!((roi.cy - upright.cx).abs() < 1e-3);
        // Up follows too: (0,-1) -> (1, 0).
        assert!((roi.up[0] - 1.0).abs() < 1e-6 && roi.up[1].abs() < 1e-6);
    }

    #[test]
    fn degenerate_scale_point_defaults_to_upright() {
        let roi = pose_roi_from_center_scale([50.0, 50.0], [50.0, 50.0]);
        assert!((roi.up[1] + 1.0).abs() < 1e-6, "up {:?}", roi.up);
        assert!(roi.size >= 1.0);
    }

    #[test]
    fn roi_in_frame_rejects_offframe_and_degenerate() {
        let ok = pose_roi_from_center_scale([160.0, 120.0], [160.0, 40.0]);
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
