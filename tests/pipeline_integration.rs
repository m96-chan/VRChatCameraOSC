//! End-to-end wiring test for the realtime pipeline (issue #8):
//! synthetic camera → stub tracker → real mapping → recording OSC sink.
//! Needs no webcam and no model weights, so it runs in CI.

use std::sync::{Arc, Mutex};

use vrchat_camera_osc::capture::{FakeCamera, Frame};
use vrchat_camera_osc::mapping::Mapper;
use vrchat_camera_osc::osc::{OscParam, OscSink};
use vrchat_camera_osc::pipeline::Pipeline;
use vrchat_camera_osc::tracking::{FaceLandmarks, FaceTracker, Landmark, NUM_LANDMARKS};

/// A tracker that always returns the same plausible neutral face.
struct StubTracker(FaceLandmarks);
impl FaceTracker for StubTracker {
    fn track(&mut self, _frame: &Frame) -> anyhow::Result<Option<FaceLandmarks>> {
        Ok(Some(self.0.clone()))
    }
}

/// Records every parameter it is asked to send, via a shared handle.
#[derive(Clone)]
struct RecordingSink(Arc<Mutex<Vec<OscParam>>>);
impl OscSink for RecordingSink {
    fn send(&mut self, params: &[OscParam]) -> anyhow::Result<()> {
        self.0.lock().unwrap().extend_from_slice(params);
        Ok(())
    }
}

fn neutral_face() -> FaceLandmarks {
    let mut pts = vec![
        Landmark {
            x: 0.0,
            y: 0.0,
            score: 1.0
        };
        NUM_LANDMARKS
    ];
    let set = |pts: &mut Vec<Landmark>, i: usize, x: f32, y: f32| {
        pts[i] = Landmark { x, y, score: 1.0 };
    };
    // Right eye (36..41) and left eye (42..47) as small open hexagons.
    let right = [
        (36.0, 52.0),
        (38.0, 49.0),
        (42.0, 49.0),
        (46.0, 52.0),
        (42.0, 55.0),
        (38.0, 55.0),
    ];
    let left = [
        (56.0, 52.0),
        (58.0, 49.0),
        (62.0, 49.0),
        (66.0, 52.0),
        (62.0, 55.0),
        (58.0, 55.0),
    ];
    for (k, (x, y)) in right.into_iter().enumerate() {
        set(&mut pts, 36 + k, x, y);
    }
    for (k, (x, y)) in left.into_iter().enumerate() {
        set(&mut pts, 42 + k, x, y);
    }
    // Brows (17..26) above the eyes.
    for i in 17..=21 {
        set(&mut pts, i, 34.0 + (i - 17) as f32 * 3.0, 44.0);
    }
    for i in 22..=26 {
        set(&mut pts, i, 56.0 + (i - 22) as f32 * 3.0, 44.0);
    }
    // Nose bridge 27..30 and tip 30.
    set(&mut pts, 27, 51.0, 50.0);
    set(&mut pts, 30, 51.0, 62.0);
    // Mouth: outer corners/top/bottom + inner.
    set(&mut pts, 48, 42.0, 74.0);
    set(&mut pts, 54, 60.0, 74.0);
    set(&mut pts, 51, 51.0, 71.0);
    set(&mut pts, 57, 51.0, 79.0);
    set(&mut pts, 60, 44.0, 74.0);
    set(&mut pts, 64, 58.0, 74.0);
    set(&mut pts, 62, 51.0, 73.0);
    set(&mut pts, 66, 51.0, 76.0);
    FaceLandmarks::new(pts).unwrap()
}

#[test]
fn pipeline_wires_capture_to_osc() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());

    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(320, 240)),
        Some(Box::new(StubTracker(neutral_face()))),
        Mapper::new(),
        Box::new(sink),
    );

    assert_eq!(pipeline.resolution(), (320, 240));

    let outcome = pipeline.step().unwrap();
    assert_eq!(outcome.frame_size, (320, 240));
    assert!(outcome.landmarks.is_some());
    assert_eq!(outcome.params.len(), 10, "ten avatar params expected");
    for p in &outcome.params {
        match p.value {
            vrchat_camera_osc::osc::OscValue::Float(v) => assert!(v.is_finite()),
            _ => panic!("mapping should emit floats"),
        }
        assert!(p.address().starts_with("/avatar/parameters/"));
    }

    // A few more steps; the sink must accumulate params each frame.
    for _ in 0..3 {
        pipeline.step().unwrap();
    }
    let total = recorded.lock().unwrap().len();
    assert_eq!(total, 10 * 4, "4 frames x 10 params recorded by the sink");
}

#[test]
fn pipeline_without_tracker_sends_nothing() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(64, 48)),
        None,
        Mapper::new(),
        Box::new(sink),
    );
    let outcome = pipeline.step().unwrap();
    assert!(outcome.params.is_empty());
    assert!(outcome.landmarks.is_none());
    assert_eq!(recorded.lock().unwrap().len(), 0);
}

/// A tracker that reports no face on its first `misses` calls, then the
/// wrapped face forever after — exercises calibration skipping unusable frames.
struct FlakyTracker {
    misses_left: u32,
    face: FaceLandmarks,
}
impl FaceTracker for FlakyTracker {
    fn track(&mut self, _frame: &Frame) -> anyhow::Result<Option<FaceLandmarks>> {
        if self.misses_left > 0 {
            self.misses_left -= 1;
            Ok(None)
        } else {
            Ok(Some(self.face.clone()))
        }
    }
}

#[test]
fn calibrate_neutral_collects_samples_and_sends_no_osc() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(320, 240)),
        Some(Box::new(StubTracker(neutral_face()))),
        Mapper::new(),
        Box::new(sink),
    );

    let n = pipeline.calibrate_neutral(5).unwrap();
    assert_eq!(n, 5, "every frame found a face");
    assert_eq!(
        recorded.lock().unwrap().len(),
        0,
        "calibration must not send OSC"
    );
}

#[test]
fn calibrate_neutral_skips_frames_with_no_face() {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let mut pipeline = Pipeline::new(
        Box::new(FakeCamera::new(320, 240)),
        Some(Box::new(FlakyTracker {
            misses_left: 3,
            face: neutral_face(),
        })),
        Mapper::new(),
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
        Mapper::new(),
        Box::new(sink),
    );
    let n = pipeline.calibrate_neutral(5).unwrap();
    assert_eq!(n, 0);
    assert_eq!(recorded.lock().unwrap().len(), 0);
}
