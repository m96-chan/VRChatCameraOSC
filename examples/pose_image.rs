//! Run the MediaPipe pose pipeline on a still image — a headless demo for
//! issue #28 phase 2 (mirrors `examples/hands_image.rs`).
//!
//! Decodes an image, runs person detection → rotated ROI → BlazePose
//! landmarks through [`BlazePoseTracker`], and prints the arm-relevant
//! landmarks plus the exact `VCO_*` arm parameters the tracker would put on
//! the wire ([`ArmMapper`]). Lets the port be verified against a real photo
//! without a live webcam or VRChat (the CLAUDE.md "demo before push" for
//! this component).
//!
//! ```bash
//! cargo run --release --example pose_image -- photo.jpg \
//!   models/person_detection.onnx models/pose_estimation.onnx
//! ```

use anyhow::{bail, Result};

use vrchat_camera_osc::capture::Frame;
use vrchat_camera_osc::mapping::arm::ArmMapper;
use vrchat_camera_osc::osc::OscValue;
use vrchat_camera_osc::tracking::pose::{
    BlazePoseTracker, PoseTracker, LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW,
    RIGHT_SHOULDER, RIGHT_WRIST,
};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        bail!("usage: pose_image <in.jpg> <person_detection.onnx> <pose_estimation.onnx>");
    }

    let img = image::open(&a[0])?.to_rgb8();
    let (w, h) = img.dimensions();
    let frame = Frame::new(w, h, img.into_raw())?;

    let mut tracker = BlazePoseTracker::from_paths(&a[1], &a[2])?;
    let pose = tracker.track(&frame)?;
    let Some(pose) = pose else {
        println!("image {w}x{h}: no person detected");
        // The mapper still emits the full untracked set — show it.
        print_params(&ArmMapper::new(true).map(None));
        return Ok(());
    };

    println!(
        "image {w}x{h}: person tracked, confidence {:.2}",
        pose.confidence
    );
    for (name, i) in [
        ("L shoulder", LEFT_SHOULDER),
        ("L elbow", LEFT_ELBOW),
        ("L wrist", LEFT_WRIST),
        ("R shoulder", RIGHT_SHOULDER),
        ("R elbow", RIGHT_ELBOW),
        ("R wrist", RIGHT_WRIST),
    ] {
        let p = pose.points[i];
        println!(
            "  {name:<10} ({:.3}, {:.3})  visibility {:.2}",
            p[0], p[1], pose.visibility[i]
        );
    }

    // The exact wire output ("L"/"R" are model labels here — the live app
    // assigns physical sides via the shared mirror flag, default true).
    print_params(&ArmMapper::new(true).map(Some(&pose)));
    Ok(())
}

fn print_params(params: &[vrchat_camera_osc::osc::OscParam]) {
    println!("arm parameters:");
    for p in params {
        match p.value {
            OscValue::Float(v) => println!("  {} f {v:.4}", p.address()),
            OscValue::Bool(v) => println!("  {} b {v}", p.address()),
            ref other => println!("  {} {:?}", p.address(), other),
        }
    }
}
