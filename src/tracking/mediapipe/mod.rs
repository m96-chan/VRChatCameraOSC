//! MediaPipe face backend (issue #17): YuNet -> FaceMesh V2 -> Blendshape V2.
//!
//! Ported from AvataCam `crates/face/src/backends/mediapipe/` (~1.7k lines).
//! Three ONNX models on the `m96-chan/candle` fork via `candle-onnx`. One
//! frame: detect the face -> rotated eyes-aligned 1.5x square ROI -> 478
//! landmarks -> fixed 146-landmark subset -> 52 ARKit blendshapes, plus a
//! robust per-axis head rotation derived from the 478 landmarks.
//!
//! Target quality bar (issue #17): this backend tracks blinks and pitch
//! robustly where the FAN 68-point + geometric-heuristics backend
//! ([`super::fan`]) cannot (blink never saturates, pitch is mistaken for
//! blink). [`MediapipeTracker`] implements [`super::arkit::ArkitFaceTracker`]
//! — the richer, backend-independent signal `mapping::arkit` consumes,
//! parallel to how [`super::fan::FanTracker`] implements [`super::FaceTracker`].
//!
//! Not ported: `burn_mesh.rs` (the optional GPU/wgpu FaceMesh path — out of
//! scope, candle/CPU only here) and `gaze.rs` (iris gaze — out of scope for
//! issue #17's OSC parameter set).

mod blendshape;
mod detector;
mod mesh;
mod roi;
mod subset;
mod util;

pub use detector::{Detection, FaceDetector};
pub use roi::{face_roi, face_roi_from_landmarks, roi_in_frame, FaceRoi};

use std::path::Path;

use anyhow::{bail, Result};

use crate::capture::Frame;
use crate::tracking::arkit::{ArkitFaceFrame, ArkitFaceTracker, HeadPose};

use blendshape::BlendshapeModel;
use mesh::FaceMesh;

/// The 478 MediaPipe FaceMesh V2 landmarks (468 face + 10 iris).
pub const NUM_FACE_LANDMARKS: usize = 478;

/// One frame of raw FaceMesh output: 478 3-D landmarks (normalized image
/// space, `x,y in [0,1]`, origin top-left, y down, `z` relative depth — the
/// MediaPipe convention) plus the model's face-presence confidence.
#[derive(Debug, Clone)]
pub(crate) struct FaceMeshLandmarks {
    pub points: Vec<[f32; 3]>,
    pub presence: f32,
}

/// Minimum face-presence confidence to accept a frame (below this, the face
/// is treated as lost and the detector re-seeds next frame).
const PRESENCE_THRESHOLD: f32 = 0.5;

// --- Head rotation (issue #17, ported from AvataCam `head_rotation`, #225) ---
//
// AvataCam's original returns a `glam::Quat`; we don't add a glam dependency
// here; instead the same per-axis math is ported directly to the three
// scalar radians `mapping::arkit` and `HeadPose` want, skipping the
// quaternion entirely (AvataCam already computes roll/yaw/pitch as plain
// f32s before composing them into a `Quat` — we stop one step earlier).
//
// Sign conventions differ between AvataCam's raw per-axis values and this
// project's `HeadPose` (documented in `tracking::arkit`): verified by hand
// against AvataCam's own `head_pose_tests` (roll/yaw physical derivations)
// and re-verified here with the `head_pose_tests` module below.
//
// - **roll** and **yaw** come out already in `HeadPose`'s convention (no
//   flip): AvataCam's raw roll is "positive = tilt toward the subject's
//   right, CCW as seen by the viewer" and raw yaw is "positive = turn toward
//   the subject's left" — both match `HeadPose` verbatim.
// - **pitch** is flipped: AvataCam's raw pitch is positive for looking
//   *down*, but `HeadPose::pitch` is positive for looking *up*.

/// FaceMesh `z` is small relative to `x,y`; amplify it so the depth-difference
/// yaw/pitch channels read with usable magnitude (ported constant, tuned by
/// AvataCam by eye against live head-follow).
const HEAD_Z_GAIN: f32 = 1.5;
/// Per-axis ceiling (deg): each axis eases into this with a `tanh`
/// soft-saturation (no hard wall to snag on).
const HEAD_MAX_DEG: f32 = 60.0;
/// Output gains on the recovered yaw / pitch angles, 1:1 (the proven
/// AvataCam default). AvataCam exposes these as env-var overrides
/// (`AVATACAM_HEAD_YAW_GAIN`/`_PITCH_GAIN`); not ported here as there's no
/// established env-tuning convention in this project yet.
const HEAD_YAW_GAIN: f32 = 1.0;
const HEAD_PITCH_GAIN: f32 = 1.0;

/// World-space landmark (x right/subject-left, y up, z toward viewer) from a
/// normalized-image-space FaceMesh point (x right, y down, z more-negative
/// = closer to camera).
fn to_world(p: [f32; 3]) -> [f32; 3] {
    [p[0], -p[1], -p[2] * HEAD_Z_GAIN]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Active right-handed rotation about world +Z by `theta` radians.
fn rotate_z(theta: f32, v: [f32; 3]) -> [f32; 3] {
    let (s, c) = theta.sin_cos();
    [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]]
}

/// Active right-handed rotation about world +Y by `theta` radians.
fn rotate_y(theta: f32, v: [f32; 3]) -> [f32; 3] {
    let (s, c) = theta.sin_cos();
    [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

/// Estimate head orientation from the 478 landmarks as a [`HeadPose`],
/// relative to facing the camera. Ported from AvataCam
/// `backends::mediapipe::head_rotation` (#225): **per-axis, decoupled** — each
/// axis is read from the one landmark cue directly sensitive to it and
/// nothing else (no 3D basis, no cross products, no per-face proportion
/// constants), so noise in one axis's cue can't leak into the others:
///
/// - **roll** — the eye line's in-image angle (pure 2D);
/// - **yaw** — the eyes' depth difference;
/// - **pitch** — the chin-to-forehead tilt out of the image plane, de-rotated
///   by the measured roll/yaw first so the reading stays a pure vertical tilt
///   even when the neck itself is pitched (a "leaned-back, then turn
///   left/right" pose reads as pure yaw, not a tilted turn).
///
/// Returns [`HeadPose::default`] (all zero) when the eye line or the
/// chin-forehead line is degenerate (near length zero) or `points` is too
/// short to hold the landmarks this reads (indices up to 263).
pub fn head_pose(points: &[[f32; 3]]) -> HeadPose {
    if points.len() <= 263 {
        return HeadPose::default();
    }
    let a = |i: usize| to_world(points[i]);
    // 33 = right-eye outer, 263 = left-eye outer; 10 = forehead top, 152 = chin.
    let right = sub(a(263), a(33)); // subject right eye -> left eye (head +X)
    let up0 = sub(a(10), a(152)); // chin -> forehead (head +Y-ish)
    let r_plane = (right[0] * right[0] + right[1] * right[1]).sqrt();
    let u_plane = (up0[0] * up0[0] + up0[1] * up0[1]).sqrt();
    if r_plane < 1e-4 || u_plane < 1e-4 {
        return HeadPose::default();
    }

    // Roll: the eye line's tilt in the image plane (2D only).
    let roll = right[1].atan2(right[0]);
    // Yaw: how far the eye line tips out of the image plane.
    let mut yaw = (-right[2]).atan2(r_plane);

    // Pitch must be the de-rotated (ZYX-consistent) third angle, not the raw
    // chin->forehead tilt (#234): de-rotating the vertical vector by the
    // measured roll+yaw first keeps a pitched-neck turn reading as pure yaw.
    let u = rotate_y(-yaw, rotate_z(-roll, up0));
    let mut pitch = u[2].atan2((u[0] * u[0] + u[1] * u[1]).sqrt());

    yaw *= HEAD_YAW_GAIN;
    pitch *= HEAD_PITCH_GAIN;

    // Soft-saturate toward the range limit with `tanh` instead of a hard
    // clamp, so an estimate riding the boundary at an extreme pose eases in
    // instead of jittering against a wall.
    let max = HEAD_MAX_DEG.to_radians();
    let soft = |a: f32| max * (a / max).tanh();

    HeadPose {
        roll: soft(roll),
        yaw: soft(yaw),
        // Sign flip: AvataCam's raw pitch is positive = looking down;
        // `HeadPose::pitch` is positive = looking up.
        pitch: -soft(pitch),
    }
}

/// A [`ArkitFaceTracker`] backed by the MediaPipe YuNet + FaceMesh V2 +
/// Blendshape V2 stack.
///
/// Mirrors [`super::fan::FanTracker`]'s detect-throttling knob
/// ([`MediapipeTracker::with_detect_interval`]): the (expensive) YuNet
/// detector only re-runs every `detect_interval` frames; between detections
/// the ROI is derived from the *previous frame's* FaceMesh landmarks instead
/// (mirroring AvataCam's detector-skip tracking, #50/#53). It always
/// re-detects on the very next frame after losing the face, or when the
/// tracked ROI has drifted off-frame, regardless of interval.
pub struct MediapipeTracker {
    detector: FaceDetector,
    mesh: FaceMesh,
    blendshape: BlendshapeModel,
    detect_interval: u32,
    frame_index: u32,
    last_roi: Option<FaceRoi>,
}

impl MediapipeTracker {
    /// Load the three ONNX models from explicit paths.
    ///
    /// Runs on GPU when built with the `cuda` feature and a device is found
    /// (`Device::cuda_if_available` falls back to CPU otherwise, matching
    /// `FanTracker`/`SfdDetector`).
    pub fn from_paths(
        detector: impl AsRef<Path>,
        landmark: impl AsRef<Path>,
        blendshape: impl AsRef<Path>,
    ) -> Result<Self> {
        Ok(Self {
            detector: FaceDetector::from_path(detector)?,
            mesh: FaceMesh::from_path(landmark)?,
            blendshape: BlendshapeModel::from_path(blendshape)?,
            detect_interval: 1,
            frame_index: 0,
            last_roi: None,
        })
    }

    /// Load from the paths the app auto-downloads to (issue #17):
    /// `models/face_detection.onnx`, `models/face_landmarks.onnx`,
    /// `models/face_blendshapes.onnx` (see [`crate::models`]). Checks each
    /// path exists first and returns a clear, named error (rather than an
    /// opaque ONNX-parse failure) when a model hasn't been downloaded yet.
    pub fn from_default_paths() -> Result<Self> {
        let d = crate::models::default_face_detection_path();
        let l = crate::models::default_face_landmarks_path();
        let b = crate::models::default_face_blendshapes_path();
        require_model(&d, "face detector (YuNet)")?;
        require_model(&l, "face landmark (FaceMesh V2)")?;
        require_model(&b, "blendshape (Blendshape V2)")?;
        Self::from_paths(d, l, b)
    }

    /// Only re-run the (expensive) YuNet detector every `n` frames (`n <= 1`
    /// re-runs every frame, the default). See the [`MediapipeTracker`] doc
    /// for the fallback behavior between detections.
    pub fn with_detect_interval(mut self, n: u32) -> Self {
        self.detect_interval = n.max(1);
        self
    }
}

/// Check `path` exists, else a clear "missing, will be downloadable" error
/// naming which model and where it's expected — the model-download wiring
/// itself lives in `main.rs` (auto-download-on-first-run, mirroring
/// FAN/S3FD), so this only needs to fail clearly, not fetch anything.
fn require_model(path: &Path, label: &str) -> Result<()> {
    if !path.exists() {
        bail!(
            "{label} model missing at {} — will be downloadable from this repo's models-v1 \
             GitHub Release on first run once uploaded; see README",
            path.display()
        );
    }
    Ok(())
}

impl ArkitFaceTracker for MediapipeTracker {
    fn track(&mut self, frame: &Frame) -> Result<Option<ArkitFaceFrame>> {
        let (w, h) = (frame.width, frame.height);
        let rgb = &frame.data;

        let tracked = self.last_roi;
        let due = self.frame_index.is_multiple_of(self.detect_interval);
        self.frame_index = self.frame_index.wrapping_add(1);
        let offframe = tracked.map(|r| !roi_in_frame(&r, w, h)).unwrap_or(false);
        let need_redetect = tracked.is_none() || offframe || due;

        let roi = if need_redetect {
            let Some(det) = self.detector.detect(w, h, rgb)? else {
                self.last_roi = None;
                return Ok(None);
            };
            face_roi(&det, w, h)
        } else {
            tracked.unwrap()
        };

        let lms = self.mesh.run(w, h, rgb, &roi)?;
        if lms.presence < PRESENCE_THRESHOLD {
            // Lost: drop the tracked ROI so the detector re-seeds next frame.
            self.last_roi = None;
            return Ok(None);
        }
        // Track: derive next frame's ROI from these landmarks.
        self.last_roi = Some(face_roi_from_landmarks(&lms.points, w, h));

        let blendshapes = self.blendshape.run(&lms, w, h)?;
        let head = head_pose(&lms.points);
        Ok(Some(ArkitFaceFrame { blendshapes, head }))
    }
}

#[cfg(test)]
mod head_pose_tests {
    use super::*;

    /// Canonical front-facing face reference points in world space (x
    /// right/subject-left, y up, z toward viewer), realistic proportions in
    /// eye-half-width units. Only the landmarks `head_pose` reads are
    /// populated (mirrors AvataCam's `head_pose_tests::canonical`).
    fn canonical() -> [(usize, [f32; 3]); 4] {
        [
            (33, [-0.06, 0.0, 0.0]),   // right-eye outer
            (263, [0.06, 0.0, 0.0]),   // left-eye outer
            (10, [0.0, 0.09, -0.01]),  // forehead top
            (152, [0.0, -0.13, 0.02]), // chin
        ]
    }

    fn rotate_x(theta: f32, v: [f32; 3]) -> [f32; 3] {
        let (s, c) = theta.sin_cos();
        [v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c]
    }

    /// `R = Rz(roll) * Ry(yaw) * Rx(pitch)`, applied to `v` (pitch innermost,
    /// roll outermost) — the same composition order AvataCam's Quat products
    /// use, so a test rotation built from one axis alone reproduces
    /// AvataCam's own single-axis test rotations exactly.
    fn apply_rotation(roll: f32, yaw: f32, pitch: f32, v: [f32; 3]) -> [f32; 3] {
        let v = rotate_x(pitch, v);
        let v = rotate_y(yaw, v);
        rotate_z(roll, v)
    }

    /// Build a 478-landmark **image-space** set from a true world rotation.
    /// Inverse of `to_world`: world `(wx,wy,wz)` -> image `(wx, -wy, -wz/GAIN)`.
    fn landmarks(roll: f32, yaw: f32, pitch: f32) -> Vec<[f32; 3]> {
        let mut pts = vec![[0.0f32; 3]; 478];
        for (i, w) in canonical() {
            let wr = apply_rotation(roll, yaw, pitch, w);
            pts[i] = [wr[0], -wr[1], -wr[2] / HEAD_Z_GAIN];
        }
        pts
    }

    /// `canonical()`'s forehead/chin carry a small real anatomical z-curvature
    /// (forehead recedes, chin comes forward, even face-on) — the *raw* signal
    /// deliberately does, matching AvataCam and this project's own convention
    /// of calibrating a per-user neutral baseline downstream (see the
    /// README's "Neutral-pose calibration"), rather than baking a synthetic
    /// zero-curvature face into the estimator. Roll and yaw have no such
    /// baseline (both come from the eye line alone, which the canonical
    /// points hold level and equidistant), so only pitch needs the neutral
    /// subtracted before comparing against an expected sign/magnitude.
    fn neutral_pitch_bias() -> f32 {
        head_pose(&landmarks(0.0, 0.0, 0.0)).pitch
    }

    #[test]
    fn neutral_pose_has_zero_roll_and_yaw() {
        let hp = head_pose(&landmarks(0.0, 0.0, 0.0));
        assert!(hp.roll.abs() < 1e-3, "roll {}", hp.roll);
        assert!(hp.yaw.abs() < 1e-3, "yaw {}", hp.yaw);
    }

    /// `HeadPose::pitch` is positive = looking up (per `tracking::arkit`
    /// doc). AvataCam's raw pitch is positive for looking *down*, so this
    /// also pins down the sign flip in `head_pose`.
    #[test]
    fn looking_up_reads_positive_pitch() {
        let bias = neutral_pitch_bias();
        let up = head_pose(&landmarks(0.0, 0.0, -25f32.to_radians()));
        assert!(
            up.pitch - bias > 0.3,
            "pitch {} (bias {bias}) should read positive (up)",
            up.pitch
        );
        assert!(
            up.roll.abs() < 0.05 && up.yaw.abs() < 0.05,
            "pitch leaked: {up:?}"
        );

        let down = head_pose(&landmarks(0.0, 0.0, 25f32.to_radians()));
        assert!(
            down.pitch - bias < -0.3,
            "pitch {} (bias {bias}) should read negative (down)",
            down.pitch
        );
    }

    /// `HeadPose::yaw` is positive = turning toward the subject's left.
    #[test]
    fn turning_to_subjects_left_reads_positive_yaw() {
        let bias = neutral_pitch_bias();
        let left = head_pose(&landmarks(0.0, 20f32.to_radians(), 0.0));
        assert!(left.yaw > 0.2, "yaw {} should read positive", left.yaw);
        assert!(
            left.roll.abs() < 0.05 && (left.pitch - bias).abs() < 0.05,
            "yaw leaked: {left:?}"
        );

        let right = head_pose(&landmarks(0.0, -20f32.to_radians(), 0.0));
        assert!(right.yaw < -0.2, "yaw {} should read negative", right.yaw);
    }

    /// `HeadPose::roll` is positive = tilting toward the subject's right
    /// (CCW as seen by the viewer).
    #[test]
    fn tilting_toward_subjects_right_reads_positive_roll() {
        let bias = neutral_pitch_bias();
        let r = head_pose(&landmarks(15f32.to_radians(), 0.0, 0.0));
        assert!(r.roll > 0.15, "roll {} should read positive", r.roll);
        assert!(
            r.yaw.abs() < 0.05 && (r.pitch - bias).abs() < 0.05,
            "roll leaked: {r:?}"
        );
    }

    /// Chin/forehead z-jitter (the pitch cue) must never move yaw or roll —
    /// each axis is read from its own dedicated cue, decoupled (#225).
    #[test]
    fn vertical_z_noise_never_moves_yaw_or_roll() {
        for pitch_deg in [0.0f32, 25.0, 45.0] {
            let clean_pts = landmarks(0.0, 0.0, -pitch_deg.to_radians());
            let mut noisy_pts = clean_pts.clone();
            noisy_pts[10][2] += 0.2; // big adversarial forehead z-jitter
            noisy_pts[152][2] -= 0.2; // and chin
            let clean = head_pose(&clean_pts);
            let noisy = head_pose(&noisy_pts);
            assert!(
                (clean.yaw - noisy.yaw).abs() < 1e-3,
                "z-noise moved yaw at pitch {pitch_deg}"
            );
            assert!(
                (clean.roll - noisy.roll).abs() < 1e-3,
                "z-noise moved roll at pitch {pitch_deg}"
            );
        }
    }

    #[test]
    fn boundary_saturates_smoothly_and_never_exceeds_ceiling() {
        let max = HEAD_MAX_DEG.to_radians();
        let mut prev = f32::NEG_INFINITY;
        for deg in (0..=70).step_by(5) {
            let hp = head_pose(&landmarks(0.0, (deg as f32).to_radians(), 0.0));
            assert!(hp.yaw <= max + 1e-3, "yaw {} exceeded ceiling", hp.yaw);
            assert!(hp.yaw >= prev - 1e-3, "yaw non-monotonic at {deg}");
            prev = hp.yaw;
        }
        assert!(
            prev > 0.7 * max && prev < max,
            "did not ease into ceiling: {prev}"
        );
    }

    #[test]
    fn degenerate_or_short_landmarks_return_default() {
        assert_eq!(head_pose(&[[0.0f32; 3]; 478]), HeadPose::default());
        assert_eq!(head_pose(&[[0.0f32; 3]; 10]), HeadPose::default());
    }
}
