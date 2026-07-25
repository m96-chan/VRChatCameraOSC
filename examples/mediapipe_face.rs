//! Run the MediaPipe face pipeline on a still image — a headless demo for
//! issue #17 (mirrors AvataCam `crates/face/examples/face_image.rs`).
//!
//! Decodes an image, runs YuNet -> FaceMesh V2 -> Blendshape V2 through
//! [`MediapipeTracker`], and prints the top-8 blendshapes plus the head pose.
//! Lets the port be verified against a real photo without a live webcam or
//! VRChat (the CLAUDE.md "demo before push" for this component).
//!
//! ```bash
//! cargo run --example mediapipe_face -- testdata/astronaut.png \
//!   models/face_detection.onnx models/face_landmarks.onnx models/face_blendshapes.onnx
//! ```

use anyhow::{bail, Result};

use vrchat_camera_osc::capture::Frame;
use vrchat_camera_osc::tracking::arkit::{ArkitFaceTracker, ARKIT_BLENDSHAPE_NAMES};
use vrchat_camera_osc::tracking::mediapipe::MediapipeTracker;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        bail!("usage: mediapipe_face <in.png> <detector.onnx> <landmark.onnx> <blendshape.onnx>");
    }

    let img = image::open(&a[0])?.to_rgb8();
    let (w, h) = img.dimensions();
    let frame = Frame::new(w, h, img.into_raw())?;

    let mut tracker = MediapipeTracker::from_paths(&a[1], &a[2], &a[3])?;
    let Some(face) = tracker.track(&frame)? else {
        bail!("no face detected in {}", a[0]);
    };

    println!("image {w}x{h}");
    println!("blendshapes: {}", face.blendshapes.0.len());

    // Top-8 blendshapes (excluding `_neutral`, index 0), descending.
    let mut ranked: Vec<(&'static str, f32)> = ARKIT_BLENDSHAPE_NAMES
        .iter()
        .copied()
        .zip(face.blendshapes.0)
        .skip(1)
        .collect();
    ranked.sort_by(|x, y| y.1.total_cmp(&x.1));
    println!("top blendshapes:");
    for (name, value) in ranked.into_iter().take(8) {
        println!("  {name:<20} {value:.3}");
    }

    println!("head pose (radians):");
    println!("  roll  {:+.3}", face.head.roll);
    println!("  yaw   {:+.3}", face.head.yaw);
    println!("  pitch {:+.3}", face.head.pitch);

    // A few named ones for an at-a-glance sanity check.
    for name in [
        "jawOpen",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "mouthSmileLeft",
        "browInnerUp",
    ] {
        let v = face.blendshapes.get(name).unwrap();
        println!("  [{name}] = {v:.3}");
    }

    Ok(())
}
