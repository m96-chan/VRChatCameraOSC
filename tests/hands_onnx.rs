//! De-risk / smoke tests for the MediaPipe hand backend (issue #8): the two
//! OpenCV Zoo ONNX models (palm detection, hand landmarks) load and run
//! through the real [`HandsTracker`] plumbing. Model-gated like
//! `tests/mediapipe_onnx.rs`: each test skips with a clear message when its
//! model isn't present (they're gitignored / auto-downloaded), so CI without
//! the weights stays green.
//!
//! Cross-backend numeric parity for the landmark net lives in
//! `tests/mediapipe_onnx.rs::hand_landmark_burn_matches_candle`; gesture
//! classification is covered by `src/mapping/gesture.rs` unit tests. No
//! hand-photo fixture exists in `testdata/`, so these pin plumbing (shapes,
//! thresholds, no-hand behavior), not recognition accuracy — that is the
//! live `--monitor` demo's job (CLAUDE.md rule 2).

use std::path::PathBuf;

use vrchat_camera_osc::capture::Frame;
use vrchat_camera_osc::models::{default_hand_landmarks_path, default_palm_detection_path};
use vrchat_camera_osc::tracking::hands::{HandTracker, HandsTracker, PalmDetector};

macro_rules! require_model {
    ($path:expr, $label:literal) => {{
        let p: PathBuf = $path;
        if !p.exists() {
            eprintln!(
                "skip: {} model not found at {} (models-v1 release; run the app once online)",
                $label,
                p.display()
            );
            return;
        }
        p
    }};
}

/// The palm detector must evaluate a real frame without erroring, and a
/// blank frame must produce no detections (no anchors above threshold).
#[test]
fn palm_detector_runs_and_blank_frame_has_no_palms() {
    let path = require_model!(default_palm_detection_path(), "palm detection");
    let det = PalmDetector::from_path(&path).expect("load palm model");
    let dets = det
        .detect(320, 240, &vec![0u8; 320 * 240 * 3], 2)
        .expect("detect on a blank frame");
    assert!(dets.is_empty(), "blank frame should contain no palms");
}

/// End-to-end tracker on a blank frame: no hands, no error — and repeated
/// frames (crossing the redetect interval) stay stable.
#[test]
fn hands_tracker_returns_empty_on_blank_frames() {
    let palm = require_model!(default_palm_detection_path(), "palm detection");
    let lms = require_model!(default_hand_landmarks_path(), "hand landmark");
    let mut tracker = HandsTracker::from_paths(&palm, &lms)
        .expect("load hand models")
        .with_redetect_interval(3);
    let frame = Frame::new(320, 240, vec![0u8; 320 * 240 * 3]).unwrap();
    for i in 0..7 {
        let hands = tracker.track(&frame).expect("track should not error");
        assert!(hands.is_empty(), "frame {i}: no hands in a blank frame");
    }
}

/// The tracker must also survive a busy natural image with no hands in it
/// (the astronaut face photo): whatever the detector proposes, the landmark
/// stage's confidence gate keeps output sane — every reported hand (if any)
/// carries finite landmarks and an above-threshold confidence.
#[test]
fn hands_tracker_output_is_sane_on_a_natural_image() {
    let palm = require_model!(default_palm_detection_path(), "palm detection");
    let lms = require_model!(default_hand_landmarks_path(), "hand landmark");
    let img = image::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/astronaut.png"
    ))
    .expect("open testdata/astronaut.png")
    .to_rgb8();
    let (w, h) = img.dimensions();
    let frame = Frame::new(w, h, img.into_raw()).unwrap();

    let mut tracker = HandsTracker::from_paths(&palm, &lms).expect("load hand models");
    for _ in 0..3 {
        let hands = tracker.track(&frame).expect("track should not error");
        for hand in &hands {
            assert!(hand.confidence >= 0.5, "gate: {}", hand.confidence);
            assert!(hand.points.iter().all(|p| p.iter().all(|v| v.is_finite())));
        }
    }
}
