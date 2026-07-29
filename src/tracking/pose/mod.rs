//! MediaPipe-Pose (BlazePose) backend (issue #28 phase 2): person detection
//! → rotated person ROI → 33 3-D pose landmarks + visibility, for one
//! tracked person per frame — the input to full-arm retargeting
//! (`mapping::arm`).
//!
//! Two OpenCV Zoo ONNX conversions of Google's MediaPipe Pose (both
//! Apache-2.0, hosted on this repo's `models-v1` release like the face and
//! hand stacks):
//!
//! - [`person::PersonDetector`] — `person_detection_mediapipe` (224×224 SSD,
//!   anchors + NMS), candle-onnx CPU (~500 ms/eval, measured), so it runs
//!   **rarely** behind the async safety-net-redetect pattern (issue #17).
//! - [`landmark::PoseLandmarker`] — `pose_estimation_mediapipe` (256×256 →
//!   39×(x,y,z,visibility,presence) + confidence), the per-frame hot path,
//!   on burn-wgpu (~8 ms/eval) with a candle CPU fallback.
//!
//! Between detections the ROI is derived from the previous frame's
//! auxiliary landmarks 33/34 ([`roi::pose_roi_from_center_scale`]) —
//! MediaPipe's own VIDEO-mode design, the same pattern as hands and the
//! face stack. [`BlazePoseTracker`] orchestrates; the [`PoseTracker`] trait
//! keeps the pipeline stub-testable and the model swappable (CLAUDE.md
//! "models are pluggable").

pub mod landmark;
pub mod person;
pub mod roi;

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::capture::Frame;

pub use landmark::PoseLandmarker;
pub use person::{PersonDetection, PersonDetector};
pub use roi::PoseRoi;

use person::{KP_FULL_BODY, KP_HIP_CENTER};
use roi::{pose_roi_from_center_scale, roi_in_frame};

/// The 33 BlazePose body landmarks exposed per frame.
pub const NUM_POSE_LANDMARKS: usize = 33;
/// The raw landmark count out of the net: 33 body + 6 auxiliary ROI points.
pub const NUM_RAW_POSE_LANDMARKS: usize = 39;

// Landmark indices (BlazePose convention; "left"/"right" name the side **as
// labeled by the model** — how that maps to the user's physical side is the
// arm mapper's `mirror` concern, exactly like hand handedness).
pub const LEFT_SHOULDER: usize = 11;
pub const RIGHT_SHOULDER: usize = 12;
pub const LEFT_ELBOW: usize = 13;
pub const RIGHT_ELBOW: usize = 14;
pub const LEFT_WRIST: usize = 15;
pub const RIGHT_WRIST: usize = 16;

/// Auxiliary landmark 33: body-center ROI point (the tracked-ROI center —
/// the landmark twin of the detector's mid-hip keypoint).
const AUX_ROI_CENTER: usize = 33;
/// Auxiliary landmark 34: ROI scale/rotation point (twin of the detector's
/// full-body keypoint) — MediaPipe's `pose_landmarks_to_roi` keypoint pair.
const AUX_ROI_SCALE: usize = 34;

/// One tracked person for one frame: 33 3-D landmarks in normalized frame
/// coordinates (`x,y in [0,1]`, y down, z hip-relative in frame-width units
/// — the FaceMesh/hands convention), per-landmark sigmoid visibility, and
/// the model's pose-presence confidence.
#[derive(Debug, Clone)]
pub struct PoseFrame {
    pub confidence: f32,
    pub points: [[f32; 3]; NUM_POSE_LANDMARKS],
    /// Per-landmark visibility, sigmoid-applied (`0..1`): probability the
    /// keypoint is in frame *and* not occluded.
    pub visibility: [f32; NUM_POSE_LANDMARKS],
}

/// Pose tracking behind a pluggable boundary: a frame in, at most one
/// tracked person out. Implemented by [`BlazePoseTracker`]; pipeline tests
/// stub it.
pub trait PoseTracker {
    fn track(&mut self, frame: &Frame) -> Result<Option<PoseFrame>>;
}

/// Minimum pose-presence confidence to keep tracking (below it the ROI is
/// dropped and the detector re-seeds). `mp_pose.py`'s `confThreshold`
/// default.
const CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Default safety-net person-redetect cadence in frames (~1/s at 30 FPS),
/// mirroring the face and hand stacks.
pub const DEFAULT_REDETECT_INTERVAL: u32 = 30;

/// The detect-then-track pose tracker (see module docs for the two stages).
///
/// Person-detection scheduling: the first frame detects synchronously (the
/// only way to start — matches hands/face); afterwards the detector runs
/// **only asynchronously** (~1/s on its own thread while untracked). Unlike
/// hands there is no synchronous loss-transition redetect: this detector
/// costs ~500 ms, which would visibly freeze the whole loop — instead a
/// loss immediately schedules an async detect, so reacquisition starts at
/// once without blocking a single frame.
pub struct BlazePoseTracker {
    person: Arc<PersonDetector>,
    landmarker: PoseLandmarker,
    redetect_interval: u32,
    frame_index: u32,
    /// Tracked ROI, refreshed from each frame's aux landmarks 33/34.
    tracked: Option<PoseRoi>,
    /// In-flight async safety-net person detect (at most one).
    pending_detect: Option<std::sync::mpsc::Receiver<Vec<PersonDetection>>>,
}

impl BlazePoseTracker {
    /// Load the two ONNX models from explicit paths.
    pub fn from_paths(person: impl AsRef<Path>, landmarks: impl AsRef<Path>) -> Result<Self> {
        let landmarker = PoseLandmarker::from_path(landmarks)?;
        eprintln!("pose: landmark stage = {}", landmarker.name());
        Ok(Self {
            person: Arc::new(PersonDetector::from_path(person)?),
            landmarker,
            redetect_interval: DEFAULT_REDETECT_INTERVAL,
            frame_index: 0,
            tracked: None,
            pending_detect: None,
        })
    }

    /// Load from the paths the app auto-downloads to
    /// (`models/person_detection.onnx`, `models/pose_estimation.onnx` — see
    /// [`crate::models`]), with a clear per-model error when one is missing.
    pub fn from_default_paths() -> Result<Self> {
        let p = crate::models::default_person_detection_path();
        let l = crate::models::default_pose_estimation_path();
        require_model(&p, "person detection")?;
        require_model(&l, "pose landmark")?;
        Self::from_paths(p, l)
    }

    /// Safety-net person-redetect cadence in frames (`n >= 1`; default
    /// [`DEFAULT_REDETECT_INTERVAL`]).
    pub fn with_redetect_interval(mut self, n: u32) -> Self {
        self.redetect_interval = n.max(1);
        self
    }

    fn seed_from_detections(&mut self, dets: &[PersonDetection]) {
        if self.tracked.is_some() {
            // The tracked ROI is landmark-refreshed every frame and thus
            // fresher than a detection from ~0.5 s ago.
            return;
        }
        if let Some(best) = dets.first() {
            self.tracked = Some(pose_roi_from_center_scale(
                best.keypoints[KP_HIP_CENTER],
                best.keypoints[KP_FULL_BODY],
            ));
        }
    }

    /// Kick off an async person detect on a worker thread (at most one in
    /// flight).
    fn spawn_detect(&mut self, w: u32, h: u32, rgb: &[u8]) {
        if self.pending_detect.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let person = Arc::clone(&self.person);
        let buf = rgb.to_vec();
        std::thread::spawn(move || {
            let _ = tx.send(person.detect(w, h, &buf, 1).unwrap_or_default());
        });
        self.pending_detect = Some(rx);
    }
}

/// Check `path` exists, else a clear "missing, will be downloadable" error
/// (mirrors the face/hand stacks; the download wiring lives in `main.rs`).
fn require_model(path: &Path, label: &str) -> Result<()> {
    if !path.exists() {
        bail!(
            "{label} model missing at {} — auto-downloaded from this repo's models-v1 \
             GitHub Release on first run; see README",
            path.display()
        );
    }
    Ok(())
}

impl PoseTracker for BlazePoseTracker {
    fn track(&mut self, frame: &Frame) -> Result<Option<PoseFrame>> {
        let (w, h) = (frame.width, frame.height);
        let rgb = &frame.data;

        // Fold in a finished async safety-net detect, if any.
        if let Some(rx) = &self.pending_detect {
            match rx.try_recv() {
                Ok(dets) => {
                    self.pending_detect = None;
                    self.seed_from_detections(&dets);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending_detect = None,
            }
        }

        // Drop a tracked ROI that drifted off-frame or collapsed.
        if let Some(roi) = &self.tracked {
            if !roi_in_frame(roi, w, h) {
                self.tracked = None;
            }
        }

        let first_frame = self.frame_index == 0;
        let due = self.frame_index.is_multiple_of(self.redetect_interval);
        self.frame_index = self.frame_index.wrapping_add(1);

        // First frame: nothing to track from — a synchronous detect is the
        // only way to start (matches the face/hand stacks).
        if first_frame {
            let dets = self.person.detect(w, h, rgb, 1)?;
            self.seed_from_detections(&dets);
        } else if due && self.tracked.is_none() {
            // Periodic async safety net while no person is tracked.
            self.spawn_detect(w, h, rgb);
        }

        let Some(roi) = self.tracked else {
            return Ok(None);
        };

        // Landmark the tracked person; low confidence drops it and starts
        // an async reacquisition immediately (never a ~500 ms sync stall —
        // see the struct docs).
        let raw = self.landmarker.run(w, h, rgb, &roi)?;
        if raw.confidence < CONFIDENCE_THRESHOLD {
            self.tracked = None;
            self.spawn_detect(w, h, rgb);
            return Ok(None);
        }

        // Next frame's ROI from this frame's aux landmarks 33/34 (pixels).
        let (wf, hf) = (w as f32, h as f32);
        let center = [
            raw.points[AUX_ROI_CENTER][0] * wf,
            raw.points[AUX_ROI_CENTER][1] * hf,
        ];
        let scale_point = [
            raw.points[AUX_ROI_SCALE][0] * wf,
            raw.points[AUX_ROI_SCALE][1] * hf,
        ];
        self.tracked = Some(pose_roi_from_center_scale(center, scale_point));

        let mut points = [[0.0f32; 3]; NUM_POSE_LANDMARKS];
        let mut visibility = [0.0f32; NUM_POSE_LANDMARKS];
        points.copy_from_slice(&raw.points[..NUM_POSE_LANDMARKS]);
        visibility.copy_from_slice(&raw.visibility[..NUM_POSE_LANDMARKS]);
        Ok(Some(PoseFrame {
            confidence: raw.confidence,
            points,
            visibility,
        }))
    }
}
