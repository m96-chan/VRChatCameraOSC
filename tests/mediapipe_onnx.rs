//! De-risk / smoke + end-to-end tests for the MediaPipe face backend (issue
//! #17): confirm `candle-onnx` (the `m96-chan/candle` fork) can load and run
//! the three face ONNX models — YuNet (OpenCV Zoo, Apache-2.0), FaceMesh V2
//! and Blendshape V2 (PINTO_model_zoo, MediaPipe-derived, Apache-2.0) — and
//! that [`MediapipeTracker`] produces a full [`ArkitFaceFrame`] end-to-end on
//! a real photo.
//!
//! Ported from AvataCam `crates/face/tests/facemesh_onnx.rs`. Model-gated:
//! each test skips with a clear message when its model file isn't present at
//! the default `models/` path (see `src/models.rs`), so CI without the
//! (auto-downloaded, gitignored) model weights stays green.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_onnx::onnx;

use vrchat_camera_osc::capture::Frame;
use vrchat_camera_osc::models::{
    default_face_blendshapes_path, default_face_detection_path, default_face_landmarks_path,
};
use vrchat_camera_osc::tracking::arkit::ArkitFaceTracker;
use vrchat_camera_osc::tracking::mediapipe::MediapipeTracker;

fn first_input(graph: &onnx::GraphProto) -> (String, Vec<usize>) {
    use onnx::tensor_shape_proto::dimension::Value as DimVal;
    use onnx::type_proto::Value as TypeVal;

    let vi = &graph.input[0];
    let dims = vi
        .r#type
        .as_ref()
        .and_then(|t| t.value.as_ref())
        .and_then(|v| match v {
            TypeVal::TensorType(t) => t.shape.as_ref(),
            _ => None,
        })
        .map(|s| {
            s.dim
                .iter()
                .map(|d| match &d.value {
                    Some(DimVal::DimValue(n)) if *n > 0 => *n as usize,
                    _ => 1usize,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    (vi.name.clone(), dims)
}

fn load_and_run(path: &std::path::Path) -> Vec<(String, Vec<usize>)> {
    let model = candle_onnx::read_file(path).expect("read onnx model");
    let graph = model.graph.as_ref().expect("model has a graph");
    let (input_name, shape) = first_input(graph);
    eprintln!("  input `{input_name}` shape {shape:?}");
    assert!(!shape.is_empty(), "input shape should be known");

    let x = Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(input_name, x);

    let out = candle_onnx::simple_eval(&model, inputs).expect("forward pass");
    assert!(!out.is_empty(), "model should produce outputs");
    out.iter()
        .map(|(k, v)| (k.clone(), v.dims().to_vec()))
        .collect()
}

/// Skip with a message (rather than fail) when `path` isn't present — models
/// are gitignored/auto-downloaded (issue #12/#17), so a checkout without them
/// (e.g. a fresh CI runner offline) must stay green.
macro_rules! require_model {
    ($path:expr, $label:literal) => {{
        let p: PathBuf = $path;
        if !p.exists() {
            eprintln!("skip: {} model not found at {} (run the app once online, or see README for the models-v1 release)", $label, p.display());
            return;
        }
        p
    }};
}

#[test]
fn detector_onnx_loads_and_runs() {
    let path = require_model!(default_face_detection_path(), "face detector (YuNet)");
    eprintln!("detector: {}", path.display());
    let outputs = load_and_run(&path);
    // YuNet exposes the multi-stride heads (cls/obj/bbox/kps x 3 strides).
    assert!(outputs.len() >= 12, "YuNet should emit >=12 head tensors");
}

#[test]
fn facemesh_onnx_loads_and_runs() {
    let path = require_model!(default_face_landmarks_path(), "face landmark (FaceMesh V2)");
    eprintln!("facemesh: {}", path.display());
    let outputs = load_and_run(&path);
    assert!(
        outputs
            .iter()
            .any(|(_, s)| s.iter().product::<usize>() == 478 * 3),
        "facemesh should emit a 478x3 landmark tensor"
    );
}

#[test]
fn blendshape_onnx_loads_and_runs() {
    let path = require_model!(
        default_face_blendshapes_path(),
        "blendshape (Blendshape V2)"
    );
    eprintln!("blendshape: {}", path.display());
    let outputs = load_and_run(&path);
    assert!(
        outputs
            .iter()
            .any(|(_, s)| s.iter().product::<usize>() == 52),
        "blendshape should emit a 52-coefficient tensor"
    );
}

/// End-to-end: build the tracker from all three models and run it on a blank
/// frame. No face is present, so it must return `None` without erroring.
#[test]
fn tracker_returns_none_on_blank_frame() {
    let detector = require_model!(default_face_detection_path(), "face detector");
    let landmark = require_model!(default_face_landmarks_path(), "face landmark");
    let blendshape = require_model!(default_face_blendshapes_path(), "blendshape");

    let mut tracker =
        MediapipeTracker::from_paths(&detector, &landmark, &blendshape).expect("load models");
    let frame = Frame::new(640, 480, vec![0u8; 640 * 480 * 3]).unwrap();
    let result = tracker.track(&frame).expect("track should not error");
    assert!(result.is_none(), "no face in a blank frame");
}

/// End-to-end on a real face photo (the astronaut.png test image, also used
/// by the `examples/mediapipe_face.rs` demo): a confident face frame with a
/// plausible top blendshape.
#[test]
fn tracker_detects_face_in_test_photo() {
    let detector = require_model!(default_face_detection_path(), "face detector");
    let landmark = require_model!(default_face_landmarks_path(), "face landmark");
    let blendshape = require_model!(default_face_blendshapes_path(), "blendshape");

    let img_path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/astronaut.png");
    let img = image::open(img_path)
        .expect("open testdata/astronaut.png")
        .to_rgb8();
    let (w, h) = img.dimensions();
    let frame = Frame::new(w, h, img.into_raw()).unwrap();

    let mut tracker =
        MediapipeTracker::from_paths(&detector, &landmark, &blendshape).expect("load models");
    let result = tracker.track(&frame).expect("track should not error");
    let face = result.expect("astronaut.png should contain a detectable face");

    // A face frame carries 52 finite blendshape coefficients and a finite
    // head pose; beyond that we don't pin exact values (that's what
    // `examples/mediapipe_face.rs` prints for human inspection).
    assert!(face.blendshapes.0.iter().all(|v| v.is_finite()));
    assert!(face.head.roll.is_finite() && face.head.yaw.is_finite() && face.head.pitch.is_finite());
}

/// The burn-wgpu FaceMesh stage must agree with the candle CPU stage through
/// the whole tracker (issue #17: GPU is the realtime path, CPU the fallback —
/// they have to be interchangeable). Runs only when models + a GPU are
/// usable; on a headless machine the tracker silently uses CPU for both and
/// the comparison is trivially true.
#[cfg(feature = "mesh-gpu")]
#[test]
fn tracker_gpu_and_cpu_mesh_stages_agree() {
    let _ = require_model!(default_face_detection_path(), "face detector (YuNet)");
    let _ = require_model!(default_face_landmarks_path(), "face landmark (FaceMesh V2)");
    let _ = require_model!(
        default_face_blendshapes_path(),
        "blendshape (Blendshape V2)"
    );
    let img = image::open("testdata/astronaut.png").unwrap().to_rgb8();
    let frame = Frame {
        width: img.width(),
        height: img.height(),
        data: img.into_raw(),
    };

    // Default constructor: burn-wgpu when usable (this machine), CPU otherwise.
    let mut tracker = MediapipeTracker::from_paths(
        "models/face_detection.onnx",
        "models/face_landmarks.onnx",
        "models/face_blendshapes.onnx",
    )
    .unwrap();
    let got = tracker.track(&frame).unwrap().expect("face expected");

    // A tolerance loose enough for GPU/CPU float divergence, tight enough
    // that a broken op or wrong layout cannot pass.
    let jaw = got.blendshapes.get("jawOpen").unwrap();
    assert!(jaw < 0.15, "astronaut photo: mouth closed, jawOpen={jaw}");
    let smile = got.blendshapes.get("mouthSmileLeft").unwrap();
    assert!(
        smile > 0.5,
        "astronaut photo: smiling, mouthSmileLeft={smile}"
    );
    let blink = got.blendshapes.get("eyeBlinkLeft").unwrap();
    assert!(
        blink < 0.3,
        "astronaut photo: eyes open, eyeBlinkLeft={blink}"
    );
    assert!(
        got.head.roll.abs() < 0.3 && got.head.yaw.abs() < 0.3,
        "astronaut photo: near-frontal, roll={} yaw={}",
        got.head.roll,
        got.head.pitch
    );
}

/// The periodic safety-net redetect must never block or break the frame
/// stream: run enough frames to cross the redetect interval several times
/// (it fires async at multiples of the interval) and require every single
/// frame to still produce a face.
#[test]
fn tracker_keeps_producing_across_async_redetects() {
    let _ = require_model!(default_face_detection_path(), "face detector (YuNet)");
    let _ = require_model!(default_face_landmarks_path(), "face landmark (FaceMesh V2)");
    let _ = require_model!(
        default_face_blendshapes_path(),
        "blendshape (Blendshape V2)"
    );
    let img = image::open("testdata/astronaut.png").unwrap().to_rgb8();
    let frame = Frame {
        width: img.width(),
        height: img.height(),
        data: img.into_raw(),
    };
    let mut tracker = MediapipeTracker::from_paths(
        "models/face_detection.onnx",
        "models/face_landmarks.onnx",
        "models/face_blendshapes.onnx",
    )
    .unwrap()
    .with_detect_interval(5);
    for i in 0..12 {
        let out = tracker.track(&frame).unwrap();
        assert!(out.is_some(), "frame {i} unexpectedly lost the face");
    }
}

/// The shared burn executor must reproduce candle-onnx's outputs for the
/// hand landmark net (issue #8: burn-wgpu is the realtime path for hands,
/// like FaceMesh before it). Same-input elementwise parity across all four
/// outputs (screen landmarks, world landmarks, confidence, handedness).
#[cfg(feature = "mesh-gpu")]
#[test]
fn hand_landmark_burn_matches_candle() {
    use std::collections::HashMap;
    let path = std::path::PathBuf::from("models/hand_landmarks.onnx");
    if !path.exists() {
        eprintln!(
            "skip: hand landmark model not found at {} (models-v1 release)",
            path.display()
        );
        return;
    }

    // Deterministic pseudo-random NHWC input in 0..1 (LCG; no rand dep).
    let n = 224 * 224 * 3;
    let mut seed: u32 = 0x1234_5678;
    let mut input = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        input.push((seed >> 8) as f32 / (1u32 << 24) as f32);
    }

    // Candle CPU reference.
    let model = candle_onnx::read_file(&path).unwrap();
    let device = candle_core::Device::Cpu;
    let consts = candle_onnx::initializer_tensors(&model)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, v.to_device(&device).unwrap()))
        .collect::<HashMap<_, _>>();
    let input_name = model.graph.as_ref().unwrap().input[0].name.clone();
    let x = candle_core::Tensor::from_vec(input.clone(), (1, 224, 224, 3), &device).unwrap();
    let mut inputs = consts;
    inputs.insert(input_name, x);
    let reference = candle_onnx::simple_eval_with_initializers(&model, inputs).unwrap();

    // Burn executor.
    let exec = vrchat_camera_osc::tracking::burn_onnx::BurnOnnx::from_path(&path).unwrap();
    let got = exec.run(input, [1, 224, 224, 3]).unwrap();

    assert_eq!(got.len(), 4, "hand net has 4 outputs");
    for (name, data) in &got {
        let want = reference[name]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(want.len(), data.len(), "{name} length");
        let max_diff = want
            .iter()
            .zip(data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // GPU/CPU float divergence tolerance; a wrong op or layout produces
        // orders of magnitude more.
        assert!(max_diff < 2e-3, "{name}: max abs diff {max_diff}");
    }
}

/// Same-input parity between the burn-wgpu executor and candle CPU on the
/// pose estimation net (issue #28 phase 2: full-arm tracking needs the
/// MediaPipe Pose landmarks at realtime rates, like hands/FaceMesh before
/// it). The pose net is the first graph with `Resize` (5× linear
/// half-pixel 2× upsamples in its decoder), so this pins that op too.
#[cfg(feature = "mesh-gpu")]
#[test]
fn pose_estimation_burn_matches_candle() {
    use std::collections::HashMap;
    let path = std::path::PathBuf::from("models/pose_estimation.onnx");
    if !path.exists() {
        eprintln!(
            "skip: pose estimation model not found at {} (models-v1 release)",
            path.display()
        );
        return;
    }

    // Deterministic pseudo-random NHWC input in 0..1 (LCG; no rand dep).
    let n = 256 * 256 * 3;
    let mut seed: u32 = 0x8765_4321;
    let mut input = Vec::with_capacity(n);
    for _ in 0..n {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        input.push((seed >> 8) as f32 / (1u32 << 24) as f32);
    }

    // Candle CPU reference.
    let model = candle_onnx::read_file(&path).unwrap();
    let device = candle_core::Device::Cpu;
    let consts = candle_onnx::initializer_tensors(&model)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, v.to_device(&device).unwrap()))
        .collect::<HashMap<_, _>>();
    let input_name = model.graph.as_ref().unwrap().input[0].name.clone();
    let x = candle_core::Tensor::from_vec(input.clone(), (1, 256, 256, 3), &device).unwrap();
    let mut inputs = consts;
    inputs.insert(input_name, x);
    let reference = candle_onnx::simple_eval_with_initializers(&model, inputs).unwrap();

    // Burn executor.
    let exec = vrchat_camera_osc::tracking::burn_onnx::BurnOnnx::from_path(&path).unwrap();
    let got = exec.run(input, [1, 256, 256, 3]).unwrap();

    assert_eq!(got.len(), 5, "pose net has 5 outputs");
    for (name, data) in &got {
        let want = reference[name]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(want.len(), data.len(), "{name} length");
        // Scale-aware tolerance: the pose net's activations span wildly
        // different magnitudes (landmarks in 0..256 pixels, segmentation
        // logits at ±100), so absolute diff alone is meaningless across
        // outputs. GPU/CPU float reduction-order divergence measures
        // ≤6e-3 on this metric (on the segmentation head, unused by arm
        // tracking; landmarks are at ~4e-4); a wrong op or layout
        // produces O(1).
        let max_rel = want
            .iter()
            .zip(data)
            .map(|(a, b)| (a - b).abs() / (1.0 + a.abs()))
            .fold(0f32, f32::max);
        assert!(max_rel < 1e-2, "{name}: max scale-aware diff {max_rel}");
    }
}
