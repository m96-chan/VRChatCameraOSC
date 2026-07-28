//! MediaPipe-Hands backend (issue #8): palm detection → rotated hand ROI →
//! 21 3-D hand landmarks + handedness, for up to two hands per frame.
//!
//! Two OpenCV Zoo ONNX conversions of Google's MediaPipe Hands (both
//! Apache-2.0, hosted on this repo's `models-v1` release like the face
//! stack):
//!
//! - [`palm::PalmDetector`] — `palm_detection_mediapipe` (192×192 SSD,
//!   anchors + NMS), candle-onnx CPU (~200 ms/eval, measured), so it runs
//!   **rarely**: synchronously on the first frame and on hand-loss
//!   transitions, otherwise as an async safety-net (~1/s on its own thread)
//!   — the issue-#17 detect-then-track pattern, again.
//! - [`landmark::HandLandmarker`] — `handpose_estimation_mediapipe`
//!   (224×224 → 21 landmarks + confidence + handedness), the per-frame hot
//!   path, on burn-wgpu (~5 ms/eval) with a candle CPU fallback.
//!
//! Between detections each hand's ROI is derived from its previous frame's
//! landmarks ([`roi::hand_roi_from_landmarks`]) — MediaPipe's own VIDEO-mode
//! design. [`HandsTracker`] orchestrates; the [`HandTracker`] trait keeps
//! the pipeline stub-testable and the model swappable (CLAUDE.md "models
//! are pluggable").

pub mod landmark;
pub mod palm;
pub mod roi;

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::capture::Frame;

pub use palm::{PalmDetection, PalmDetector};
pub use roi::HandRoi;

use landmark::HandLandmarker;
use roi::{hand_roi_from_landmarks, hand_roi_from_palm_points, roi_in_frame};

/// The 21 MediaPipe hand landmarks per hand.
pub const NUM_HAND_LANDMARKS: usize = 21;

// Landmark indices (MediaPipe Hands convention).
pub const WRIST: usize = 0;
pub const THUMB_CMC: usize = 1;
pub const THUMB_MCP: usize = 2;
pub const THUMB_IP: usize = 3;
pub const THUMB_TIP: usize = 4;
pub const INDEX_MCP: usize = 5;
pub const INDEX_PIP: usize = 6;
pub const INDEX_DIP: usize = 7;
pub const INDEX_TIP: usize = 8;
pub const MIDDLE_MCP: usize = 9;
pub const MIDDLE_PIP: usize = 10;
pub const MIDDLE_DIP: usize = 11;
pub const MIDDLE_TIP: usize = 12;
pub const RING_MCP: usize = 13;
pub const RING_PIP: usize = 14;
pub const RING_DIP: usize = 15;
pub const RING_TIP: usize = 16;
pub const PINKY_MCP: usize = 17;
pub const PINKY_PIP: usize = 18;
pub const PINKY_DIP: usize = 19;
pub const PINKY_TIP: usize = 20;

/// Which hand the landmark model saw, **as labeled by the model** (the
/// image-side label; how it maps to the user's physical hand — and to
/// `GestureLeft`/`GestureRight` — is the gesture mapper's `mirror` concern).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
}

/// One tracked hand for one frame: 21 3-D landmarks in normalized frame
/// coordinates (`x,y in [0,1]`, y down, z wrist-relative — the FaceMesh
/// convention), the model's presence confidence, and its handedness label.
#[derive(Debug, Clone)]
pub struct HandFrame {
    pub handedness: Handedness,
    pub confidence: f32,
    pub points: [[f32; 3]; NUM_HAND_LANDMARKS],
}

/// Hand tracking behind a pluggable boundary: a frame in, zero or more
/// tracked hands out. Implemented by [`HandsTracker`]; pipeline tests stub it.
pub trait HandTracker {
    fn track(&mut self, frame: &Frame) -> Result<Vec<HandFrame>>;
}

/// Minimum landmark-stage confidence to keep tracking a hand (below it the
/// hand is dropped and the detector re-seeds). MediaPipe's own
/// `min_tracking_confidence` default; Zoo's demo bar (0.8) proved too eager
/// to drop mid-gesture in MediaPipe's tuning notes, and a dropped hand costs
/// a full ~200 ms redetect.
const CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Default safety-net palm-redetect cadence in frames (~1/s at 30 FPS),
/// mirroring the face stack's `DEFAULT_REDETECT_INTERVAL`.
pub const DEFAULT_REDETECT_INTERVAL: u32 = 30;

/// Default maximum number of simultaneously tracked hands.
pub const DEFAULT_MAX_HANDS: usize = 2;

/// The detect-then-track hand tracker (see module docs for the two stages).
///
/// Palm-detection scheduling — the one place this deliberately deviates from
/// the face stack: a missing *face* is transient, so `MediapipeTracker`
/// detects synchronously while lost; missing *hands* are the steady state
/// whenever they're out of frame, and a ~200 ms synchronous detect per frame
/// would pin the whole loop at ~5 FPS. So only the first frame and a
/// loss *transition* (a tracked hand vanishing) detect synchronously;
/// while hands stay absent (or one slot is free) the detector runs as the
/// ~1/s async safety net.
pub struct HandsTracker {
    palm: Arc<PalmDetector>,
    landmarker: HandLandmarker,
    max_hands: usize,
    redetect_interval: u32,
    frame_index: u32,
    /// Tracked ROIs, one per live hand, refreshed from each frame's
    /// landmarks.
    tracked: Vec<HandRoi>,
    /// In-flight async safety-net palm detect (at most one).
    pending_detect: Option<std::sync::mpsc::Receiver<Vec<PalmDetection>>>,
}

impl HandsTracker {
    /// Load the two ONNX models from explicit paths.
    pub fn from_paths(palm: impl AsRef<Path>, landmarks: impl AsRef<Path>) -> Result<Self> {
        let landmarker = HandLandmarker::from_path(landmarks)?;
        eprintln!("hands: landmark stage = {}", landmarker.name());
        Ok(Self {
            palm: Arc::new(PalmDetector::from_path(palm)?),
            landmarker,
            max_hands: DEFAULT_MAX_HANDS,
            redetect_interval: DEFAULT_REDETECT_INTERVAL,
            frame_index: 0,
            tracked: Vec::new(),
            pending_detect: None,
        })
    }

    /// Load from the paths the app auto-downloads to
    /// (`models/palm_detection.onnx`, `models/hand_landmarks.onnx` — see
    /// [`crate::models`]), with a clear per-model error when one is missing.
    pub fn from_default_paths() -> Result<Self> {
        let p = crate::models::default_palm_detection_path();
        let l = crate::models::default_hand_landmarks_path();
        require_model(&p, "palm detection")?;
        require_model(&l, "hand landmark")?;
        Self::from_paths(p, l)
    }

    /// Track at most `n` hands (`n >= 1`; default [`DEFAULT_MAX_HANDS`]).
    pub fn with_max_hands(mut self, n: usize) -> Self {
        self.max_hands = n.max(1);
        self
    }

    /// Safety-net palm-redetect cadence in frames (`n >= 1`; default
    /// [`DEFAULT_REDETECT_INTERVAL`]).
    pub fn with_redetect_interval(mut self, n: u32) -> Self {
        self.redetect_interval = n.max(1);
        self
    }

    fn seed_from_detections(&mut self, dets: &[PalmDetection]) {
        let additions = merge_detections(&self.tracked, dets, self.max_hands);
        self.tracked.extend(additions);
    }
}

/// Check `path` exists, else a clear "missing, will be downloadable" error
/// (mirrors the face stack; the download wiring lives in `main.rs`).
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

/// ROIs to add for detections that aren't already covered by a tracked hand,
/// keeping the total at most `max_hands`. "Already covered" = the detection's
/// ROI center falls within half the size of an existing ROI (cheap proximity
/// test; exact overlap doesn't matter — a duplicate ROI would just track the
/// same hand twice).
fn merge_detections(tracked: &[HandRoi], dets: &[PalmDetection], max_hands: usize) -> Vec<HandRoi> {
    let mut additions: Vec<HandRoi> = Vec::new();
    for det in dets {
        if tracked.len() + additions.len() >= max_hands {
            break;
        }
        let roi = hand_roi_from_palm_points(&det.keypoints);
        let covered = tracked.iter().chain(additions.iter()).any(|t| {
            let (dx, dy) = (roi.cx - t.cx, roi.cy - t.cy);
            (dx * dx + dy * dy).sqrt() < 0.5 * t.size.max(roi.size)
        });
        if !covered {
            additions.push(roi);
        }
    }
    additions
}

impl HandTracker for HandsTracker {
    fn track(&mut self, frame: &Frame) -> Result<Vec<HandFrame>> {
        let (w, h) = (frame.width, frame.height);
        let rgb = &frame.data;

        // Fold in a finished async safety-net detect, if any: new hands are
        // added next to the ones already tracked; detections of hands we
        // already track are ignored (their ROI is landmark-refreshed every
        // frame and thus fresher than the ~0.2 s-old detection).
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

        // Drop tracked ROIs that drifted off-frame or collapsed.
        self.tracked.retain(|r| roi_in_frame(r, w, h));

        let first_frame = self.frame_index == 0;
        let due = self.frame_index.is_multiple_of(self.redetect_interval);
        self.frame_index = self.frame_index.wrapping_add(1);

        // First frame: nothing to track from — a synchronous detect is the
        // only way to start (matches the face stack).
        if first_frame {
            let dets = self.palm.detect(w, h, rgb, self.max_hands)?;
            self.seed_from_detections(&dets);
        } else if due && self.pending_detect.is_none() && self.tracked.len() < self.max_hands {
            // Periodic async safety net, only while a hand slot is free (a
            // full house has nothing to gain from a detect).
            let (tx, rx) = std::sync::mpsc::channel();
            let palm = Arc::clone(&self.palm);
            let buf = rgb.to_vec();
            let max = self.max_hands;
            std::thread::spawn(move || {
                let _ = tx.send(palm.detect(w, h, &buf, max).unwrap_or_default());
            });
            self.pending_detect = Some(rx);
        }

        // Landmark every tracked hand; low confidence drops the hand.
        let had_hands = !self.tracked.is_empty();
        let mut hands = Vec::with_capacity(self.tracked.len());
        let mut next_rois = Vec::with_capacity(self.tracked.len());
        for roi in &self.tracked {
            let raw = self.landmarker.run(w, h, rgb, roi)?;
            if raw.confidence < CONFIDENCE_THRESHOLD {
                continue; // lost this hand
            }
            next_rois.push(hand_roi_from_landmarks(&raw.points, w, h));
            hands.push(HandFrame {
                handedness: if raw.handedness >= 0.5 {
                    Handedness::Right
                } else {
                    Handedness::Left
                },
                confidence: raw.confidence,
                points: raw.points,
            });
        }
        self.tracked = next_rois;

        // Loss transition (every hand vanished this frame): one synchronous
        // redetect to catch a quick reappearance — steady-state absence is
        // handled by the async safety net above, never blocking the loop.
        if had_hands && self.tracked.is_empty() {
            let dets = self.palm.detect(w, h, rgb, self.max_hands)?;
            self.seed_from_detections(&dets);
            for roi in std::mem::take(&mut self.tracked) {
                let raw = self.landmarker.run(w, h, rgb, &roi)?;
                if raw.confidence < CONFIDENCE_THRESHOLD {
                    continue;
                }
                self.tracked
                    .push(hand_roi_from_landmarks(&raw.points, w, h));
                hands.push(HandFrame {
                    handedness: if raw.handedness >= 0.5 {
                        Handedness::Right
                    } else {
                        Handedness::Left
                    },
                    confidence: raw.confidence,
                    points: raw.points,
                });
            }
        }

        Ok(hands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det_at(cx: f32, cy: f32) -> PalmDetection {
        // A palm whose keypoints produce an ROI centered near (cx, cy):
        // wrist below the center, middle MCP above (fingers up).
        let kp = [
            [cx, cy + 40.0],        // wrist
            [cx - 15.0, cy - 10.0], // index MCP
            [cx, cy - 12.0],        // middle MCP
            [cx + 12.0, cy - 10.0], // ring MCP
            [cx + 24.0, cy - 6.0],  // pinky MCP
            [cx - 20.0, cy + 30.0], // thumb CMC
            [cx - 24.0, cy + 16.0], // thumb MCP
        ];
        PalmDetection {
            bbox: [cx - 30.0, cy - 15.0, cx + 30.0, cy + 45.0],
            keypoints: kp,
            score: 0.9,
        }
    }

    #[test]
    fn merge_adds_new_hands_up_to_max() {
        let dets = [det_at(100.0, 100.0), det_at(400.0, 100.0)];
        let added = merge_detections(&[], &dets, 2);
        assert_eq!(added.len(), 2);

        let capped = merge_detections(&[], &dets, 1);
        assert_eq!(capped.len(), 1);
    }

    #[test]
    fn merge_skips_detections_of_already_tracked_hands() {
        let dets = [det_at(100.0, 100.0)];
        let tracked = merge_detections(&[], &dets, 2);
        // The same hand detected again (a few pixels off): no addition.
        let redet = [det_at(104.0, 97.0)];
        let added = merge_detections(&tracked, &redet, 2);
        assert!(added.is_empty(), "duplicate hand must not be re-added");
        // A genuinely different hand still gets in.
        let other = [det_at(400.0, 100.0)];
        assert_eq!(merge_detections(&tracked, &other, 2).len(), 1);
    }

    #[test]
    fn merge_dedups_within_one_detection_batch() {
        let dets = [det_at(100.0, 100.0), det_at(103.0, 101.0)];
        let added = merge_detections(&[], &dets, 2);
        assert_eq!(added.len(), 1, "near-identical detections collapse");
    }
}
