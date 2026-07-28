//! MediaPipe face backend (issue #17): YuNet -> FaceMesh V2 -> Blendshape V2.
//!
//! Ported from AvataCam `crates/face/src/backends/mediapipe/` (~1.7k lines).
//! Three ONNX models on the `m96-chan/candle` fork via `candle-onnx`. One
//! frame: detect the face -> rotated eyes-aligned 1.5x square ROI -> 478
//! landmarks -> fixed 146-landmark subset -> 52 ARKit blendshapes, plus a
//! robust per-axis head rotation derived from the 478 landmarks.
//!
//! Target quality bar (issue #17): this backend tracks blinks and pitch
//! robustly where the retired FAN 68-point + geometric-heuristics backend
//! could not (blink never saturates, pitch is mistaken for blink).
//! [`MediapipeTracker`] implements [`super::arkit::ArkitFaceTracker`] — the
//! backend-independent signal the mappers consume.
//!
//! The FaceMesh stage runs on **burn-wgpu** (GPU, driver-only; `burn_mesh`,
//! ported from AvataCam #62) when the default `mesh-gpu` feature is on and a
//! GPU is usable, falling back to the candle-onnx CPU path otherwise — the
//! CPU interpreter alone can't reach the 30 FPS realtime bar. Not ported:
//! `gaze.rs` (iris gaze — out of scope for issue #17's OSC parameter set).

mod blendshape;
#[cfg(feature = "mesh-gpu")]
mod burn_mesh;
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

/// Nose-tip forward protrusion as a fraction of the inter-ocular distance
/// (canonical-face geometry) — the scale constant of the 2D yaw cue (issue
/// #27). Per-user deviation from it is a pure gain error: the direction and
/// zero point stay exact, and neutral calibration + the head range configs
/// absorb the rest.
const NOSE_DEPTH_RATIO: f32 = 0.58;

/// Nose-tip drop below the eye line as a fraction of the inter-ocular
/// distance (canonical-face geometry) — compensated in the 2D pitch cue so
/// its raw value rests near zero (see `head_pose`).
const NOSE_DROP_RATIO: f32 = 0.33;

/// Empirical FaceMesh large-yaw warp: beyond what the true-IOD geometry
/// explains, the mesh's nose/eye fit drifts upward as the face turns,
/// reading as pitch. Fitted 2026-07-28 from a 3000-frame live sweep (head
/// held level, slow full-range turns): `pitch_leak ≈ k·(1−cos yaw)` with
/// k = 0.51 rad; fit residual mean 2.4°, max 5.7° — inside ±25° of yaw the
/// residual is ~0, matching the live report. Per-user variation is expected
/// to be mild; refit from a sweep capture if a report says otherwise.
const MESH_WARP_PITCH_PER_YAW: f32 = 0.51;

/// World-space landmark (x right/subject-left, y up, z toward viewer) from a
/// normalized-image-space FaceMesh point (x right, y down, z more-negative
/// = closer to camera).
fn to_world(p: [f32; 3]) -> [f32; 3] {
    [p[0], -p[1], -p[2] * HEAD_Z_GAIN]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Estimate head orientation from the 478 landmarks as a [`HeadPose`],
/// relative to facing the camera. Ported from AvataCam
/// `backends::mediapipe::head_rotation` (#225): **per-axis, decoupled** — each
/// axis is read from the one landmark cue directly sensitive to it and
/// nothing else (no 3D basis, no cross products, no per-face proportion
/// constants), so noise in one axis's cue can't leak into the others:
///
/// - **roll** — the eye line's in-image angle (pure 2D);
/// - **yaw** — the nose tip's in-plane offset from the eye midpoint (pure
///   2D, issue #27 — replaced the original eyes'-z-difference cue, whose
///   FaceMesh depth drifted with subject distance);
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
    let r_plane = (right[0] * right[0] + right[1] * right[1]).sqrt();
    if r_plane < 1e-4 {
        return HeadPose::default();
    }

    // Roll: the eye line's tilt in the image plane (2D only).
    let roll = right[1].atan2(right[0]);

    // Yaw (issue #27): a pure-2D cue — the nose tip's in-plane offset from
    // the eye midpoint, projected onto the (roll-carrying) eye line and
    // normalized by the in-image eye distance. Geometry: with the tip
    // protruding NOSE_DEPTH_RATIO×IOD in front of the eye line, a yaw of θ
    // projects it sideways by d·sinθ while the eye distance foreshortens to
    // IOD·cosθ — so atan recovers θ exactly, independent of scale and roll.
    // The previous cue read the eye landmarks' FaceMesh **z** difference,
    // the model's least stable output: it drifts with subject distance /
    // crop, observed live as a session-dependent ~30° yaw bias surviving
    // neutral calibration (deliberate divergence from AvataCam #225 here).
    let nose = a(1);
    let mid = [(a(33)[0] + a(263)[0]) * 0.5, (a(33)[1] + a(263)[1]) * 0.5];
    let eye_dir = [right[0] / r_plane, right[1] / r_plane];
    let off = [nose[0] - mid[0], nose[1] - mid[1]];
    let x_off = off[0] * eye_dir[0] + off[1] * eye_dir[1];
    let mut yaw = (x_off / (r_plane * NOSE_DEPTH_RATIO)).atan();

    // Pitch (issue #27 follow-up): the same 2D nose cue, perpendicular
    // component — the previous chin→forehead z-tilt read FaceMesh depth,
    // and real-face z warp under yaw leaked into pitch symmetrically
    // ("looking straight left reads up-left"). The nose sits
    // NOSE_DROP_RATIO×IOD below the eye line on the canonical face; the
    // anatomical drop is compensated here so the raw value rests near 0
    // and the tanh ceiling stays symmetric (per-user residual is absorbed
    // by the downstream neutral calibration, like every other channel).
    // With this, the whole head estimator is z-free.
    // The drop compensation and the normalization must be in TRUE
    // inter-ocular units: the in-image eye distance foreshortens by
    // cos(yaw), so using `r_plane` directly under-compensates at large
    // turns and reads as pitch ("looking hard left pitches up" — follow-up
    // 2). The already-computed yaw estimate un-foreshortens it; clamp the
    // cosine away from 0 so a saturated profile view degrades gracefully.
    let iod_true = r_plane / yaw.cos().max(0.35);
    let up_dir = [-eye_dir[1], eye_dir[0]];
    let y_off = off[0] * up_dir[0] + off[1] * up_dir[1] + NOSE_DROP_RATIO * iod_true;
    let mut pitch = (y_off / (iod_true * NOSE_DEPTH_RATIO)).atan()
        - MESH_WARP_PITCH_PER_YAW * (1.0 - yaw.cos());

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
        // The 2D nose cue already reads positive = nose above its neutral
        // drop = looking up, matching `HeadPose::pitch` directly (the
        // AvataCam-era sign flip died with the z-based cue).
        pitch: soft(pitch),
    }
}

/// A [`ArkitFaceTracker`] backed by the MediaPipe YuNet + FaceMesh V2 +
/// Blendshape V2 stack.
///
/// Detect-throttling ([`MediapipeTracker::with_detect_interval`],
/// issue #13's detect-then-track pattern): the (expensive) YuNet
/// detector only re-runs every `detect_interval` frames; between detections
/// the ROI is derived from the *previous frame's* FaceMesh landmarks instead
/// (mirroring AvataCam's detector-skip tracking, #50/#53). It always
/// re-detects on the very next frame after losing the face, or when the
/// tracked ROI has drifted off-frame, regardless of interval.
pub struct MediapipeTracker {
    detector: std::sync::Arc<FaceDetector>,
    mesh: MeshStage,
    blendshape: BlendshapeModel,
    detect_interval: u32,
    frame_index: u32,
    last_roi: Option<FaceRoi>,
    /// In-flight async safety-net redetect (see [`ArkitFaceTracker::track`]):
    /// the periodic YuNet pass (~216 ms on CPU) runs on its own thread so the
    /// 30 FPS loop never blocks on it; only total loss detects synchronously.
    pending_detect: Option<std::sync::mpsc::Receiver<Option<Detection>>>,
}

/// The FaceMesh stage backend: burn-wgpu (GPU, driver-only) when the
/// `mesh-gpu` feature is on and a usable GPU is present, else the candle-onnx
/// CPU path. GPU is what makes 30 FPS possible (the CPU interpreter measured
/// ~109 ms/frame on this depthwise-heavy graph); CPU remains the fallback so
/// machines without a GPU still track, just slower.
// One long-lived instance per tracker; the variants' size gap is irrelevant.
#[allow(clippy::large_enum_variant)]
enum MeshStage {
    Cpu(FaceMesh),
    #[cfg(feature = "mesh-gpu")]
    Gpu(burn_mesh::BurnFaceMesh),
}

impl MeshStage {
    fn run(&self, width: u32, height: u32, rgb: &[u8], roi: &FaceRoi) -> Result<FaceMeshLandmarks> {
        match self {
            MeshStage::Cpu(m) => m.run(width, height, rgb, roi),
            #[cfg(feature = "mesh-gpu")]
            MeshStage::Gpu(m) => m.run(width, height, rgb, roi),
        }
    }

    /// Human-readable stage backend name (for startup logging).
    fn name(&self) -> &'static str {
        match self {
            MeshStage::Cpu(_) => "candle-cpu",
            #[cfg(feature = "mesh-gpu")]
            MeshStage::Gpu(_) => "burn-wgpu",
        }
    }
}

/// Default detector cadence: YuNet is only needed to (re)seed the ROI — the
/// frames in between derive it from the previous frame's landmarks — so while
/// tracking it re-runs at most about once a second (at ~30 FPS) as a safety
/// net, mirroring AvataCam. Loss and off-frame drift always force an
/// immediate redetect regardless of this.
pub const DEFAULT_REDETECT_INTERVAL: u32 = 30;

impl MediapipeTracker {
    /// Load the three ONNX models from explicit paths.
    ///
    /// FaceMesh runs on burn-wgpu when built with the (default) `mesh-gpu`
    /// feature and a GPU is usable, else on CPU. YuNet and the blendshape
    /// model stay on CPU (cheap or rarely run).
    pub fn from_paths(
        detector: impl AsRef<Path>,
        landmark: impl AsRef<Path>,
        blendshape: impl AsRef<Path>,
    ) -> Result<Self> {
        let mesh = Self::build_mesh_stage(landmark)?;
        eprintln!("mediapipe: FaceMesh stage = {}", mesh.name());
        Ok(Self {
            detector: std::sync::Arc::new(FaceDetector::from_path(detector)?),
            mesh,
            blendshape: BlendshapeModel::from_path(blendshape)?,
            detect_interval: DEFAULT_REDETECT_INTERVAL,
            frame_index: 0,
            last_roi: None,
            pending_detect: None,
        })
    }

    /// Try the burn-wgpu GPU stage first (feature-gated), fall back to the
    /// CPU stage on any failure — including a panic out of wgpu adapter
    /// initialization on truly headless machines.
    #[cfg(feature = "mesh-gpu")]
    fn build_mesh_stage(landmark: impl AsRef<Path>) -> Result<MeshStage> {
        let path = landmark.as_ref();
        let gpu = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            burn_mesh::BurnFaceMesh::from_path(path)
        }));
        match gpu {
            Ok(Ok(m)) => Ok(MeshStage::Gpu(m)),
            Ok(Err(e)) => {
                eprintln!("mediapipe: burn-wgpu FaceMesh unavailable ({e:#}) — using CPU");
                Ok(MeshStage::Cpu(FaceMesh::from_path(path)?))
            }
            Err(_) => {
                eprintln!("mediapipe: burn-wgpu init panicked (no GPU/driver?) — using CPU");
                Ok(MeshStage::Cpu(FaceMesh::from_path(path)?))
            }
        }
    }

    #[cfg(not(feature = "mesh-gpu"))]
    fn build_mesh_stage(landmark: impl AsRef<Path>) -> Result<MeshStage> {
        Ok(MeshStage::Cpu(FaceMesh::from_path(landmark)?))
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
    /// re-runs every frame; default [`DEFAULT_REDETECT_INTERVAL`]). See the
    /// [`MediapipeTracker`] doc for the fallback behavior between detections.
    pub fn with_detect_interval(mut self, n: u32) -> Self {
        self.detect_interval = n.max(1);
        self
    }
}

/// Check `path` exists, else a clear "missing, will be downloadable" error
/// naming which model and where it's expected — the model-download wiring
/// itself lives in `main.rs` (auto-download-on-first-run), so this only
/// needs to fail clearly, not fetch anything.
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

        // Fold in a finished async safety-net redetect, if any: a hit reseeds
        // the ROI (the face barely moves in the ~0.2s the detect took, and
        // the very next frame's landmarks re-center it anyway); a stale miss
        // changes nothing — the presence gate below handles genuine loss.
        if let Some(rx) = &self.pending_detect {
            match rx.try_recv() {
                Ok(Some(det)) => {
                    self.last_roi = Some(face_roi(&det, w, h));
                    self.pending_detect = None;
                }
                Ok(None) => self.pending_detect = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending_detect = None,
            }
        }

        let tracked = self.last_roi;
        let due = self.frame_index.is_multiple_of(self.detect_interval);
        self.frame_index = self.frame_index.wrapping_add(1);
        let offframe = tracked.map(|r| !roi_in_frame(&r, w, h)).unwrap_or(false);

        let roi = match tracked {
            // Tracking normally: never block on YuNet. When the periodic
            // safety-net is due, kick it off on a thread and keep using the
            // landmark-derived ROI meanwhile.
            Some(r) if !offframe => {
                if due && self.pending_detect.is_none() {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let det = std::sync::Arc::clone(&self.detector);
                    let buf = rgb.to_vec();
                    std::thread::spawn(move || {
                        let _ = tx.send(det.detect(w, h, &buf).ok().flatten());
                    });
                    self.pending_detect = Some(rx);
                }
                r
            }
            // First frame, or the face was lost / drifted off-frame: nothing
            // to track from, so a synchronous detect is unavoidable.
            _ => {
                let Some(det) = self.detector.detect(w, h, rgb)? else {
                    self.last_roi = None;
                    return Ok(None);
                };
                face_roi(&det, w, h)
            }
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
    fn canonical() -> [(usize, [f32; 3]); 5] {
        [
            (33, [-0.06, 0.0, 0.0]),   // right-eye outer
            (263, [0.06, 0.0, 0.0]),   // left-eye outer
            (10, [0.0, 0.09, -0.01]),  // forehead top
            (152, [0.0, -0.13, 0.02]), // chin
            (1, [0.0, -0.04, 0.07]), // nose tip (protrudes toward viewer; 0.07/0.12 ≈ NOSE_DEPTH_RATIO)
        ]
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

    /// Real-face z warp under yaw leaked into PITCH (issue #27 follow-up:
    /// looking straight left/right pushed the avatar's head up-left /
    /// up-right symmetrically) — the whole head estimator must be z-free.
    /// Scaling z arbitrarily while yawing must leave pitch unchanged.
    /// Large pure yaw on IDEAL geometry reads the deliberate empirical
    /// down-correction, not zero: real FaceMesh drifts the nose/eye fit
    /// upward by ≈ MESH_WARP_PITCH_PER_YAW·(1−cos yaw) at large turns
    /// (measured live, issue #27 follow-up 2), and the estimator subtracts
    /// that. The synthetic fixture has no such warp, so the corrected
    /// output must equal exactly −correction (through the tanh ceiling);
    /// the level-headed-at-45° acceptance lives on a real face.
    #[test]
    fn large_pure_yaw_reads_the_mesh_warp_correction() {
        let max = HEAD_MAX_DEG.to_radians();
        for yaw_deg in [-45.0f32, -30.0, 30.0, 45.0] {
            let hp = head_pose(&landmarks(0.0, yaw_deg.to_radians(), 0.0));
            let raw = -MESH_WARP_PITCH_PER_YAW * (1.0 - yaw_deg.to_radians().cos());
            let expected = max * (raw / max).tanh();
            assert!(
                (hp.pitch - expected).abs() < 2f32.to_radians(),
                "pitch {:.1}° vs expected correction {:.1}° at yaw {yaw_deg}°",
                hp.pitch.to_degrees(),
                expected.to_degrees()
            );
        }
    }

    #[test]
    fn pitch_immune_to_z_warp_under_yaw() {
        for yaw_deg in [-30.0f32, 30.0] {
            let clean = head_pose(&landmarks(0.0, yaw_deg.to_radians(), 0.0));
            let mut warped = landmarks(0.0, yaw_deg.to_radians(), 0.0);
            for p in warped.iter_mut() {
                p[2] *= 1.7;
            }
            let w = head_pose(&warped);
            assert!(
                (clean.pitch - w.pitch).abs() < 1e-3,
                "z warp moved pitch at yaw {yaw_deg}°: {} vs {}",
                clean.pitch,
                w.pitch
            );
        }
    }

    /// FaceMesh z is the model's least stable output (it drifts with crop
    /// size / subject distance) — the yaw cue must not read it (issue #27):
    /// a uniform z-scale drift, simulating the subject settling at a
    /// different distance than they calibrated at, must leave yaw unchanged.
    /// The retired z-depth yaw estimator failed exactly this, showing up
    /// live as a session-dependent ~30° yaw bias that survived neutral
    /// calibration ("can't turn left/right" while roll worked).
    #[test]
    fn yaw_immune_to_z_scale_drift() {
        for yaw_deg in [-25.0f32, 0.0, 25.0] {
            let clean = head_pose(&landmarks(0.0, yaw_deg.to_radians(), 0.0));
            let mut drifted = landmarks(0.0, yaw_deg.to_radians(), 0.0);
            for p in drifted.iter_mut() {
                p[2] *= 1.5;
            }
            let d = head_pose(&drifted);
            assert!(
                (clean.yaw - d.yaw).abs() < 1e-3,
                "z drift moved yaw at {yaw_deg}°: {} vs {}",
                clean.yaw,
                d.yaw
            );
        }
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

#[cfg(test)]
mod stage_bench {
    use super::*;

    /// Per-stage wall-clock bench on the real models + test photo. Ignored in
    /// normal runs; execute with:
    /// `cargo test --release stage_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_stages() {
        let (det_p, mesh_p, bs_p) = (
            "models/face_detection.onnx",
            "models/face_landmarks.onnx",
            "models/face_blendshapes.onnx",
        );
        if !std::path::Path::new(det_p).exists() {
            eprintln!("models missing — skipping bench");
            return;
        }
        let img = image::open("testdata/astronaut.png").unwrap().to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb = img.into_raw();

        let detector = FaceDetector::from_path(det_p).unwrap();
        let mesh = mesh::FaceMesh::from_path(mesh_p).unwrap();
        let bs = blendshape::BlendshapeModel::from_path(bs_p).unwrap();

        let det = detector.detect(w, h, &rgb).unwrap().unwrap();
        let roi = face_roi(&det, w, h);
        let lms = mesh.run(w, h, &rgb, &roi).unwrap();

        let time = |label: &str, mut f: Box<dyn FnMut()>| {
            // Warmup.
            f();
            let n = 10;
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                f();
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            eprintln!("{label:>12}: {ms:8.2} ms");
        };

        let (d2, m2, b2) = (&detector, &mesh, &bs);
        let (rgb2, lms2, roi2) = (rgb.clone(), lms.clone(), roi);
        time(
            "detector",
            Box::new(move || {
                d2.detect(w, h, &rgb2).unwrap();
            }),
        );
        let rgb3 = rgb.clone();
        time(
            "facemesh",
            Box::new(move || {
                m2.run(w, h, &rgb3, &roi2).unwrap();
            }),
        );
        time(
            "blendshape",
            Box::new(move || {
                b2.run(&lms2, w, h).unwrap();
            }),
        );
    }
}
