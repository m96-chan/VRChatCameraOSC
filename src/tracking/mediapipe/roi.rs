//! Face ROI geometry for the FaceMesh stage (issue #17, ported from AvataCam #5).
//!
//! Mirrors MediaPipe's `face_detection_front_detection_to_roi`: take the face
//! detection box, **square it to the longer side and scale 1.5x**, and rotate so
//! the two eyes are horizontal. The landmark model then runs on that upright
//! crop. We work in **pixel** space so the square stays square regardless of
//! frame aspect. `+up` = top of head, `right = [-up.y, up.x]`.

use super::detector::Detection;

/// MediaPipe `RectTransformationCalculator`: `scale_x/y = 1.5`, `square_long`.
const ROI_SCALE: f32 = 1.5;

/// A rotated square region of interest, in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct FaceRoi {
    pub cx: f32,
    pub cy: f32,
    pub size: f32,
    pub right: [f32; 2],
    pub up: [f32; 2],
    /// Corners (pixels), TL, TR, BR, BL of the upright crop.
    pub corners: [[f32; 2]; 4],
}

/// Compute the face ROI from a detection over a `w`x`h` frame (`det` in pixels).
pub fn face_roi(det: &Detection, _w: u32, _h: u32) -> FaceRoi {
    let [x1, y1, x2, y2] = det.bbox;
    // Eye keypoints (pixels): [0]=right-eye, [1]=left-eye (image-side, YuNet).
    // "Up" is disambiguated by the **mouth midpoint** ([3]/[4]), not the nose: at
    // an extreme look-up the nose tip rises to the eye line and the sign flips,
    // turning the crop upside-down. The mouth stays clearly below.
    let mouth = [
        0.5 * (det.keypoints[3][0] + det.keypoints[4][0]),
        0.5 * (det.keypoints[3][1] + det.keypoints[4][1]),
    ];
    roi_core([x1, y1, x2, y2], det.keypoints[0], det.keypoints[1], mouth)
}

/// Compute the next face ROI from the previous frame's FaceMesh landmarks
/// (detector-skip / tracking): the box from the landmark extent, eyes for the
/// roll, all in pixels of a `w`x`h` frame. `points` are normalized.
pub fn face_roi_from_landmarks(points: &[[f32; 3]], w: u32, h: u32) -> FaceRoi {
    let (wf, hf) = (w as f32, h as f32);
    let (mut x1, mut y1, mut x2, mut y2) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in points {
        x1 = x1.min(p[0] * wf);
        y1 = y1.min(p[1] * hf);
        x2 = x2.max(p[0] * wf);
        y2 = y2.max(p[1] * hf);
    }
    // Canonical FaceMesh indices: 33 = right-eye outer, 263 = left-eye outer,
    // 152 = chin. The chin (not the nose tip) disambiguates "up": at an extreme
    // look-up the nose rises to the eye line and its sign flips frame-to-frame,
    // locking the tracked ROI into a rotated attractor. The chin stays clearly
    // below the eyes across the whole meeting range.
    let px = |i: usize| [points[i][0] * wf, points[i][1] * hf];
    roi_core([x1, y1, x2, y2], px(33), px(263), px(152))
}

/// Shared ROI construction from a face box + eye points and a **below-the-eyes**
/// anchor (chin / mouth midpoint, pixels) that disambiguates which perpendicular
/// of the eye line is "up".
fn roi_core(bbox: [f32; 4], e0: [f32; 2], e1: [f32; 2], below: [f32; 2]) -> FaceRoi {
    let [x1, y1, x2, y2] = bbox;
    let cx = 0.5 * (x1 + x2);
    let cy = 0.5 * (y1 + y2);
    let long = (x2 - x1).abs().max((y2 - y1).abs()).max(1.0);
    let half = 0.5 * long * ROI_SCALE;

    let eye_mid = [0.5 * (e0[0] + e1[0]), 0.5 * (e0[1] + e1[1])];

    // Eye-line direction; "up" is the perpendicular pointing toward the forehead
    // (away from the below-the-eyes anchor), so the crop ends up upright.
    let mut d = [e0[0] - e1[0], e0[1] - e1[1]];
    let dl = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if dl > 1e-3 {
        d = [d[0] / dl, d[1] / dl];
    } else {
        d = [1.0, 0.0];
    }
    let perp = [-d[1], d[0]];
    let to_forehead = [eye_mid[0] - below[0], eye_mid[1] - below[1]];
    let up = if perp[0] * to_forehead[0] + perp[1] * to_forehead[1] >= 0.0 {
        perp
    } else {
        [-perp[0], -perp[1]]
    };
    let right = [-up[1], up[0]];

    let corner = |sx: f32, sy: f32| {
        [
            cx + sx * half * right[0] + sy * half * up[0],
            cy + sx * half * right[1] + sy * half * up[1],
        ]
    };
    let corners = [
        corner(-1.0, 1.0),  // TL
        corner(1.0, 1.0),   // TR
        corner(1.0, -1.0),  // BR
        corner(-1.0, -1.0), // BL
    ];

    FaceRoi {
        cx,
        cy,
        size: 2.0 * half,
        right,
        up,
        corners,
    }
}

/// Whether a tracked ROI is still usable: its center is inside the frame and its
/// size is plausible. A drift-locked / off-frame ROI fails this, so the caller
/// re-runs the detector to recover.
pub fn roi_in_frame(r: &FaceRoi, w: u32, h: u32) -> bool {
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

    fn det(bbox: [f32; 4], e0: [f32; 2], e1: [f32; 2], nose: [f32; 2]) -> Detection {
        let mut kp = [[0.0; 2]; 5];
        kp[0] = e0;
        kp[1] = e1;
        kp[2] = nose;
        // Mouth corners sit a bit below the nose (anatomically consistent) — the
        // detector ROI's "up" disambiguation reads their midpoint.
        kp[3] = [nose[0] - 5.0, nose[1] + 6.0];
        kp[4] = [nose[0] + 5.0, nose[1] + 6.0];
        Detection {
            bbox,
            keypoints: kp,
            score: 1.0,
        }
    }

    #[test]
    fn upright_face_gives_axis_aligned_square() {
        // 40-wide box centered at (50,50); eyes level, nose below mid.
        let d = det(
            [30.0, 30.0, 70.0, 70.0],
            [60.0, 45.0],
            [40.0, 45.0],
            [50.0, 58.0],
        );
        let roi = face_roi(&d, 100, 100);
        assert!((roi.cx - 50.0).abs() < 1e-3 && (roi.cy - 50.0).abs() < 1e-3);
        // long side 40 x 1.5 = 60.
        assert!((roi.size - 60.0).abs() < 1e-3, "size {}", roi.size);
        // Upright: up ~= image-up (0,-1), right ~= (1,0).
        assert!(
            roi.up[0].abs() < 1e-3 && (roi.up[1] + 1.0).abs() < 1e-3,
            "up {:?}",
            roi.up
        );
        assert!((roi.right[0] - 1.0).abs() < 1e-3 && roi.right[1].abs() < 1e-3);
    }

    #[test]
    fn up_points_away_from_nose() {
        // Rotated/odd eye positions: up must still point from nose toward eyes.
        let d = det(
            [30.0, 30.0, 70.0, 70.0],
            [60.0, 40.0],
            [42.0, 50.0],
            [55.0, 65.0],
        );
        let roi = face_roi(&d, 100, 100);
        let eye_mid = [51.0, 45.0];
        let to_forehead = [eye_mid[0] - 55.0, eye_mid[1] - 65.0];
        assert!(roi.up[0] * to_forehead[0] + roi.up[1] * to_forehead[1] > 0.0);
        // right perp up.
        assert!((roi.right[0] * roi.up[0] + roi.right[1] * roi.up[1]).abs() < 1e-5);
    }

    /// At an extreme look-up the **nose tip rises to / above the eye line** in
    /// the image, so an eye-mid-nose "which way is up" disambiguation degenerates
    /// and flips sign frame to frame — the crop turns upside-down. The chin
    /// stays clearly below the eyes even then, so "up" must survive
    /// nose-above-eyes landmarks.
    #[test]
    fn lookup_face_with_nose_at_eye_line_keeps_up_toward_forehead() {
        // Landmark-tracking path: eyes level at y=45, nose tip slightly ABOVE the
        // eye line (extreme look-up), chin well below.
        let mut pts = vec![[0.45f32, 0.45, 0.0]; 478];
        pts[33] = [0.40, 0.45, 0.0]; // right-eye outer
        pts[263] = [0.60, 0.45, 0.0]; // left-eye outer
        pts[1] = [0.50, 0.44, 0.0]; // nose tip — ABOVE the eye midpoint
        pts[10] = [0.50, 0.38, 0.0]; // forehead top
        pts[152] = [0.50, 0.58, 0.0]; // chin — still clearly below
        let roi = face_roi_from_landmarks(&pts, 100, 100);
        // Image y grows downward, so "toward the forehead" is up[1] < 0.
        assert!(
            roi.up[1] < -0.5,
            "ROI up flipped on a look-up face: up={:?}",
            roi.up
        );

        // Detector path: same situation via YuNet keypoints (nose above eye mid,
        // mouth corners below).
        let mut kp = [[0.0f32; 2]; 5];
        kp[0] = [40.0, 45.0]; // right eye
        kp[1] = [60.0, 45.0]; // left eye
        kp[2] = [50.0, 44.0]; // nose — above the eye line
        kp[3] = [45.0, 52.0]; // mouth corners — below
        kp[4] = [55.0, 52.0];
        let d = Detection {
            bbox: [30.0, 30.0, 70.0, 70.0],
            keypoints: kp,
            score: 1.0,
        };
        let roi = face_roi(&d, 100, 100);
        assert!(
            roi.up[1] < -0.5,
            "detector ROI up flipped on a look-up face: up={:?}",
            roi.up
        );
    }

    #[test]
    fn roi_in_frame_rejects_offframe_and_degenerate() {
        let ok = face_roi(
            &det(
                [30.0, 30.0, 70.0, 70.0],
                [60.0, 45.0],
                [40.0, 45.0],
                [50.0, 58.0],
            ),
            100,
            100,
        );
        assert!(roi_in_frame(&ok, 100, 100));

        let mut off = ok;
        off.cx = 150.0; // center past the right edge
        assert!(!roi_in_frame(&off, 100, 100));

        let mut tiny = ok;
        tiny.size = 2.0; // collapsed
        assert!(!roi_in_frame(&tiny, 100, 100));

        let mut nan = ok;
        nan.cx = f32::NAN;
        assert!(!roi_in_frame(&nan, 100, 100));
    }
}
