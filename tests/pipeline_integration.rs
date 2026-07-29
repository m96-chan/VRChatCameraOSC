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

// Hand tracking → gesture ints (issue #8): a stub HandTracker drives the
// real GestureMapper through Pipeline::with_hands.
mod hands {
    use super::*;
    use vrchat_camera_osc::mapping::gesture::{
        GestureMapper, GestureMapperConfig, DEBOUNCE_FRAMES,
    };
    use vrchat_camera_osc::osc::OscValue;
    use vrchat_camera_osc::tracking::hands::{HandFrame, HandTracker, Handedness};

    /// A hand tracker that always reports the same hands.
    struct StubHandTracker(Vec<HandFrame>);
    impl HandTracker for StubHandTracker {
        fn track(&mut self, _frame: &Frame) -> anyhow::Result<Vec<HandFrame>> {
            Ok(self.0.clone())
        }
    }

    /// An upright open hand (all fingers extended, thumb splayed) — the same
    /// synthetic geometry the gesture unit tests use.
    fn open_hand(handedness: Handedness) -> HandFrame {
        let mut p = [[0.0f32; 3]; 21];
        p[0] = [0.50, 0.90, 0.0]; // wrist
        p[1] = [0.42, 0.82, 0.0]; // thumb CMC
        p[2] = [0.37, 0.76, 0.0]; // thumb MCP
        p[3] = [0.32, 0.70, 0.0]; // thumb IP
        p[4] = [0.27, 0.65, 0.0]; // thumb TIP
        for (f, mcp) in [[0.40f32, 0.60], [0.47, 0.58], [0.54, 0.60], [0.60, 0.63]]
            .iter()
            .enumerate()
        {
            let base = 5 + f * 4;
            p[base] = [mcp[0], mcp[1], 0.0];
            p[base + 1] = [mcp[0], mcp[1] - 0.07, 0.0];
            p[base + 2] = [mcp[0], mcp[1] - 0.12, 0.0];
            p[base + 3] = [mcp[0], mcp[1] - 0.16, 0.0];
        }
        HandFrame {
            handedness,
            confidence: 0.9,
            points: p,
        }
    }

    fn int_param(params: &[OscParam], name: &str) -> Option<i32> {
        params
            .iter()
            .find(|p| p.name == name)
            .map(|p| match p.value {
                OscValue::Int(v) => v,
                ref other => panic!("{name} is not an int: {other:?}"),
            })
    }

    #[test]
    fn pipeline_with_hands_appends_gesture_ints_even_without_a_face() {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink(recorded.clone());
        // No face tracker at all: gestures must still flow.
        let mut pipeline = Pipeline::new(
            Box::new(FakeCamera::new(320, 240)),
            None,
            mapper(),
            Box::new(sink),
        )
        .with_hands(
            Box::new(StubHandTracker(vec![open_hand(Handedness::Left)])),
            GestureMapper::new(GestureMapperConfig::default()),
            1,
        );

        let mut outcome = pipeline.step().unwrap();
        for _ in 0..DEBOUNCE_FRAMES {
            outcome = pipeline.step().unwrap();
        }
        // mirror (default): the model-Right hand is the user's left.
        assert_eq!(int_param(&outcome.params, "VCO_GestureLeft"), Some(2));
        assert_eq!(int_param(&outcome.params, "VCO_GestureRight"), Some(0));
        assert_eq!(int_param(&outcome.params, "GestureLeft"), Some(2));
        // Tier-2 curls (issue #8 phase 3, default on) flow with the ints.
        let names: Vec<&str> = outcome.params.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"VCO_L_IndexCurl"));
        assert!(names.contains(&"VCO_R_LittleCurl"));
        assert!(!recorded.lock().unwrap().is_empty(), "params were sent");
    }

    #[test]
    fn hands_interval_throttles_the_hand_stage() {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink(recorded.clone());
        let mut pipeline = Pipeline::new(
            Box::new(FakeCamera::new(64, 48)),
            None,
            mapper(),
            Box::new(sink),
        )
        .with_hands(
            Box::new(StubHandTracker(Vec::new())),
            GestureMapper::new(GestureMapperConfig {
                emit_native: false,
                ..Default::default()
            }),
            2,
        );

        // Interval 2: frames 0, 2 run the hand stage (2 gesture ints + 10
        // curl floats each); frames 1, 3 skip it entirely.
        let per_hand_frame = 12;
        for (i, expect) in [(0u32, per_hand_frame), (1, 0), (2, per_hand_frame), (3, 0)] {
            let outcome = pipeline.step().unwrap();
            assert_eq!(outcome.params.len(), expect, "frame {i}");
        }
    }

    #[test]
    fn gesture_params_ride_alongside_face_params() {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink(recorded.clone());
        let mut pipeline = Pipeline::new(
            Box::new(FakeCamera::new(320, 240)),
            Some(Box::new(StubArkitTracker(resting_arkit_frame()))),
            mapper(),
            Box::new(sink),
        )
        .with_hands(
            Box::new(StubHandTracker(Vec::new())),
            GestureMapper::new(GestureMapperConfig::default()),
            1,
        );

        let outcome = pipeline.step().unwrap();
        let names: Vec<&str> = outcome.params.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"v2/JawOpen"), "face params still present");
        assert!(names.contains(&"VCO_GestureLeft"));
        assert!(names.contains(&"VCO_GestureRight"));
    }
}

// Pose tracking → arm parameters (issue #28 phase 2): a stub PoseTracker
// drives the real ArmMapper through Pipeline::with_pose.
mod pose {
    use super::*;
    use vrchat_camera_osc::mapping::arm::ArmMapper;
    use vrchat_camera_osc::osc::OscValue;
    use vrchat_camera_osc::tracking::pose::{
        PoseFrame, PoseTracker, LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, NUM_POSE_LANDMARKS,
        RIGHT_SHOULDER,
    };

    /// A pose tracker that always reports the same pose (or none).
    struct StubPoseTracker(Option<PoseFrame>);
    impl PoseTracker for StubPoseTracker {
        fn track(&mut self, _frame: &Frame) -> anyhow::Result<Option<PoseFrame>> {
            Ok(self.0.clone())
        }
    }

    /// A pose with the model-left arm raised straight overhead, all its
    /// landmarks fully visible.
    fn overhead_left_arm() -> PoseFrame {
        let mut points = [[0.5f32, 0.5, 0.0]; NUM_POSE_LANDMARKS];
        let mut visibility = [0.0f32; NUM_POSE_LANDMARKS];
        points[LEFT_SHOULDER] = [0.6, 0.3, 0.0];
        points[LEFT_ELBOW] = [0.6, 0.15, 0.0];
        points[LEFT_WRIST] = [0.6, 0.0, 0.0];
        points[RIGHT_SHOULDER] = [0.4, 0.3, 0.0];
        for i in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER] {
            visibility[i] = 0.99;
        }
        PoseFrame {
            confidence: 0.9,
            points,
            visibility,
        }
    }

    fn float(params: &[OscParam], name: &str) -> f32 {
        params
            .iter()
            .find(|p| p.name == name)
            .map(|p| match p.value {
                OscValue::Float(v) => v,
                ref other => panic!("{name} not a float: {other:?}"),
            })
            .unwrap_or_else(|| panic!("{name} missing"))
    }

    fn boolean(params: &[OscParam], name: &str) -> bool {
        params
            .iter()
            .find(|p| p.name == name)
            .map(|p| match p.value {
                OscValue::Bool(v) => v,
                ref other => panic!("{name} not a bool: {other:?}"),
            })
            .unwrap_or_else(|| panic!("{name} missing"))
    }

    /// Arm parameters (issue #28 phase 2) flow through the pipeline even
    /// without a face: a raised model-left arm reads +1 UpDown on the left
    /// params (mirror default), the other side reports untracked.
    #[test]
    fn arm_params_flow_through_the_pipeline() {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink(recorded.clone());
        let mut pipeline = Pipeline::new(
            Box::new(FakeCamera::new(320, 240)),
            None,
            mapper(),
            Box::new(sink),
        )
        .with_pose(
            Box::new(StubPoseTracker(Some(overhead_left_arm()))),
            ArmMapper::new(true),
        );

        let outcome = pipeline.step().unwrap();
        assert!(boolean(&outcome.params, "VCO_L_ArmTracked"));
        assert!(
            float(&outcome.params, "VCO_L_ArmUpDown") > 0.9,
            "left arm overhead"
        );
        assert_eq!(float(&outcome.params, "VCO_L_Elbow"), 0.0, "arm straight");
        assert!(!boolean(&outcome.params, "VCO_R_ArmTracked"));
        assert_eq!(float(&outcome.params, "VCO_R_ArmUpDown"), 0.0);
        assert!(!recorded.lock().unwrap().is_empty(), "params were sent");
    }

    /// No pose at all still emits the full 8-parameter set (untracked +
    /// exact zeros) every frame — the wire stays truthful.
    #[test]
    fn absent_pose_still_emits_untracked_arm_params() {
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink(recorded.clone());
        let mut pipeline = Pipeline::new(
            Box::new(FakeCamera::new(64, 48)),
            None,
            mapper(),
            Box::new(sink),
        )
        .with_pose(Box::new(StubPoseTracker(None)), ArmMapper::new(true));

        let outcome = pipeline.step().unwrap();
        assert_eq!(outcome.params.len(), 8, "2 bools + 6 floats");
        for side in ["L", "R"] {
            assert!(!boolean(&outcome.params, &format!("VCO_{side}_ArmTracked")));
            assert_eq!(
                float(&outcome.params, &format!("VCO_{side}_ArmUpDown")),
                0.0
            );
        }
    }
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
