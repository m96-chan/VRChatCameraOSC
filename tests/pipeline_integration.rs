//! End-to-end wiring test for the realtime pipeline (issue #8):
//! synthetic camera → stub tracker → real mapping → recording OSC sink.
//! Needs no webcam and no model weights, so it runs in CI.
//!
//! Since issue #21 the pipeline has a single stack: ARKit frames (MediaPipe
//! backend) feeding the VRCFT-compatible `UnifiedMapper`.

use std::sync::{Arc, Mutex};

use vrchat_camera_osc::capture::{FakeCamera, Frame};
use vrchat_camera_osc::mapping::arkit::ArkitMappingConfig;
use vrchat_camera_osc::mapping::unified::UnifiedMapper;
use vrchat_camera_osc::osc::{OscParam, OscSink};
use vrchat_camera_osc::pipeline::Pipeline;
use vrchat_camera_osc::tracking::arkit::{
    ArkitBlendshapes, ArkitFaceFrame, ArkitFaceTracker, HeadPose, NUM_BLENDSHAPES,
};

/// Records every parameter it is asked to send, via a shared handle.
#[derive(Clone)]
struct RecordingSink(Arc<Mutex<Vec<OscParam>>>);
impl OscSink for RecordingSink {
    fn send(&mut self, params: &[OscParam]) -> anyhow::Result<()> {
        self.0.lock().unwrap().extend_from_slice(params);
        Ok(())
    }
}

/// An ARKit tracker that always returns the same frame.
struct StubArkitTracker(ArkitFaceFrame);
impl ArkitFaceTracker for StubArkitTracker {
    fn track(&mut self, _frame: &Frame) -> anyhow::Result<Option<ArkitFaceFrame>> {
        Ok(Some(self.0))
    }
}

/// A tracker that reports no face on its first `misses` calls, then the
/// wrapped frame forever after — exercises calibration skipping unusable
/// frames.
struct FlakyArkitTracker {
    misses_left: u32,
    frame: ArkitFaceFrame,
}
impl ArkitFaceTracker for FlakyArkitTracker {
    fn track(&mut self, _frame: &Frame) -> anyhow::Result<Option<ArkitFaceFrame>> {
        if self.misses_left > 0 {
            self.misses_left -= 1;
            Ok(None)
        } else {
            Ok(Some(self.frame))
        }
    }
}

fn resting_arkit_frame() -> ArkitFaceFrame {
    // Non-zero resting baselines, like the real Blendshape V2 output.
    let mut bs = [0.0f32; NUM_BLENDSHAPES];
    bs[9] = 0.08; // eyeBlinkLeft
    bs[10] = 0.12; // eyeBlinkRight
    bs[25] = 0.22; // jawOpen
    bs[38] = 0.30; // mouthPucker
    ArkitFaceFrame {
        blendshapes: ArkitBlendshapes(bs),
        head: HeadPose {
            roll: 0.02,
            yaw: -0.01,
            pitch: -0.05,
        },
    }
}

fn mapper() -> UnifiedMapper {
    UnifiedMapper::new(ArkitMappingConfig::default(), Vec::new())
}

#[test]
fn pipeline_wires_capture_to_osc() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());

    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(320, 240)),
        Some(Box::new(StubArkitTracker(resting_arkit_frame()))),
        mapper(),
        Box::new(sink),
    );

    assert_eq!(pipeline.resolution(), (320, 240));

    let outcome = pipeline.step().unwrap();
    assert_eq!(outcome.frame_size, (320, 240));
    let names: Vec<&str> = outcome.params.iter().map(|p| p.name.as_str()).collect();
    // Default prefixes: bare v2/ and FT/v2/ copies, plus status bools.
    assert!(names.contains(&"v2/JawOpen"));
    assert!(names.contains(&"FT/v2/JawOpen"));
    assert!(names.contains(&"v2/EyeLidLeft"));
    assert!(names.contains(&"EyeTrackingActive"));
    assert!(names.contains(&"FT/LipTrackingActive"));
    for p in &outcome.params {
        assert!(p.address().starts_with("/avatar/parameters/"));
    }

    // A few more steps; the sink must accumulate the same count each frame.
    let per_frame = outcome.params.len();
    for _ in 0..3 {
        pipeline.step().unwrap();
    }
    assert_eq!(recorded.lock().unwrap().len(), per_frame * 4);
}

#[test]
fn pipeline_without_tracker_sends_nothing() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(64, 48)),
        None,
        mapper(),
        Box::new(sink),
    );
    let outcome = pipeline.step().unwrap();
    assert!(outcome.params.is_empty());
    assert_eq!(recorded.lock().unwrap().len(), 0);
}

#[test]
fn calibrate_neutral_zeroes_resting_baselines_and_sends_no_osc() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(320, 240)),
        Some(Box::new(StubArkitTracker(resting_arkit_frame()))),
        mapper(),
        Box::new(sink),
    );

    let n = pipeline.calibrate_neutral(5).unwrap();
    assert_eq!(n, 5, "every frame found a face");
    assert_eq!(
        recorded.lock().unwrap().len(),
        0,
        "calibration must not send OSC"
    );

    // After calibration, the same resting frame must map to ~0 on the
    // expression channels — that is the point of the neutral baseline.
    // (EyeLid*/EyeOpen* deliberately rest at ~0.75/1.0 — VRCFT semantics.)
    let outcome = pipeline.step().unwrap();
    for name in ["v2/JawOpen", "v2/SmileFrown", "v2/BrowUp"] {
        let p = outcome
            .params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} present"));
        if let vrchat_camera_osc::osc::OscValue::Float(v) = p.value {
            assert!(v.abs() < 0.05, "{name} should read ~0 at rest, got {v}");
        }
    }
}

#[test]
fn calibrate_neutral_skips_frames_with_no_face() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(320, 240)),
        Some(Box::new(FlakyArkitTracker {
            misses_left: 3,
            frame: resting_arkit_frame(),
        })),
        mapper(),
        Box::new(sink),
    );

    // 3 misses + 4 hits out of 7 frames.
    let n = pipeline.calibrate_neutral(7).unwrap();
    assert_eq!(n, 4);
}

#[test]
fn calibrate_neutral_without_tracker_is_a_harmless_noop() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(64, 48)),
        None,
        mapper(),
        Box::new(sink),
    );
    let n = pipeline.calibrate_neutral(5).unwrap();
    assert_eq!(n, 0);
    assert_eq!(recorded.lock().unwrap().len(), 0);
}

// Native /tracking/eye/* output (issue #19) is appended to the parameter set
// and participates in neutral calibration.
#[test]
fn pipeline_with_native_eye_appends_tracking_eye_messages() {
    use vrchat_camera_osc::mapping::eye::{EyeRange, NativeEyeMapper};

    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(320, 240)),
        Some(Box::new(StubArkitTracker(resting_arkit_frame()))),
        mapper(),
        Box::new(sink),
    )
    .with_native_eye(NativeEyeMapper::new(
        ArkitMappingConfig::default(),
        EyeRange::default(),
    ));

    let n = pipeline.calibrate_neutral(5).unwrap();
    assert_eq!(n, 5);
    assert_eq!(
        recorded.lock().unwrap().len(),
        0,
        "calibration sends no OSC"
    );

    let outcome = pipeline.step().unwrap();
    let gaze = outcome
        .params
        .iter()
        .find(|p| p.name == "/tracking/eye/LeftRightPitchYaw")
        .expect("gaze message present");
    assert!(matches!(
        gaze.value,
        vrchat_camera_osc::osc::OscValue::Floats4(_)
    ));
    assert_eq!(gaze.address(), "/tracking/eye/LeftRightPitchYaw");
    let closed = outcome
        .params
        .iter()
        .find(|p| p.name == "/tracking/eye/EyesClosedAmount")
        .expect("eyes-closed message present");
    if let vrchat_camera_osc::osc::OscValue::Float(v) = closed.value {
        assert!(v.abs() < 0.05, "calibrated rest should read ~0: {v}");
    } else {
        panic!("EyesClosedAmount must be a float");
    }
}
