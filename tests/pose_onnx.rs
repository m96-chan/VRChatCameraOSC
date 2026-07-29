//! De-risk / smoke tests for the MediaPipe pose backend (issue #28 phase
//! 2): the two OpenCV Zoo ONNX models (person detection, BlazePose
//! landmarks) load and run through the real [`BlazePoseTracker`] plumbing.
//! Model-gated like `tests/mediapipe_onnx.rs`: each test skips with a clear
//! message when its model isn't present (they're gitignored /
//! auto-downloaded), so CI without the weights stays green.
//!
//! Cross-backend numeric parity for the landmark net lives in
//! `tests/mediapipe_onnx.rs::pose_estimation_burn_matches_candle`; arm
//! retargeting is covered by `src/mapping/arm.rs` unit tests. No
//! full-person photo fixture exists in `testdata/`, so these pin plumbing
//! (shapes, thresholds, no-person behavior), not recognition accuracy —
//! that is the live `--monitor` demo's job (CLAUDE.md rule 2).

use std::path::PathBuf;

use vrchat_camera_osc::capture::Frame;
use vrchat_camera_osc::models::{default_person_detection_path, default_pose_estimation_path};
use vrchat_camera_osc::tracking::pose::{
    BlazePoseTracker, PersonDetector, PoseLandmarker, PoseRoi, PoseTracker, NUM_RAW_POSE_LANDMARKS,
};

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

/// The person detector must evaluate a real frame without erroring, and a
/// blank frame must produce no detections (no anchors above threshold).
#[test]
fn person_detector_runs_and_blank_frame_has_no_persons() {
    let path = require_model!(default_person_detection_path(), "person detection");
    let det = PersonDetector::from_path(&path).expect("load person model");
    let dets = det
        .detect(320, 240, &vec![0u8; 320 * 240 * 3], 1)
        .expect("detect on a blank frame");
    assert!(dets.is_empty(), "blank frame should contain no persons");
}

/// The pose landmark net decodes to sane values on a deterministic
/// pseudo-random frame (the LCG pattern of `tests/mediapipe_onnx.rs`),
/// driven through the real landmark stage with an identity ROI: 39 finite
/// frame-normalized landmarks, every visibility a valid sigmoid
/// probability, confidence in `[0,1]` — never an error or NaN.
#[test]
fn pose_landmarks_decode_sanely_on_synthetic_input() {
    let landmarks = require_model!(default_pose_estimation_path(), "pose landmark");

    // Deterministic pseudo-random RGB frame (LCG; no rand dep).
    let (w, h) = (320u32, 240u32);
    let n = (w * h * 3) as usize;
    let mut seed: u32 = 0x8765_4321;
    let mut rgb = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        rgb.push((seed >> 24) as u8);
    }

    let landmarker = PoseLandmarker::from_path(&landmarks).expect("load pose landmark model");
    let roi = PoseRoi {
        cx: w as f32 / 2.0,
        cy: h as f32 / 2.0,
        size: h as f32,
        right: [1.0, 0.0],
        up: [0.0, -1.0],
    };
    let raw = landmarker
        .run(w, h, &rgb, &roi)
        .expect("landmark stage should not error");
    assert_eq!(raw.points.len(), NUM_RAW_POSE_LANDMARKS);
    assert!(
        (0.0..=1.0).contains(&raw.confidence),
        "confidence {} (Identity_1 is already sigmoid)",
        raw.confidence
    );
    for (i, (p, v)) in raw.points.iter().zip(raw.visibility).enumerate() {
        assert!(
            p.iter().all(|c| c.is_finite()),
            "landmark {i} not finite: {p:?}"
        );
        assert!((0.0..=1.0).contains(&v), "visibility {i} = {v}");
    }
}

/// End-to-end tracker on blank frames: no person, no error — and repeated
/// frames (crossing the redetect interval, exercising the async safety-net
/// scheduling) stay stable.
#[test]
fn pose_tracker_returns_none_on_blank_frames() {
    let person = require_model!(default_person_detection_path(), "person detection");
    let landmarks = require_model!(default_pose_estimation_path(), "pose landmark");
    let mut tracker = BlazePoseTracker::from_paths(&person, &landmarks)
        .expect("load pose models")
        .with_redetect_interval(3);
    let frame = Frame::new(320, 240, vec![0u8; 320 * 240 * 3]).unwrap();
    for i in 0..7 {
        let out = tracker.track(&frame).expect("track should not error");
        assert!(out.is_none(), "frame {i}: no person in a blank frame");
    }
}
