//! Run the MediaPipe hand pipeline on a still image — a headless demo for
//! issue #8 (mirrors `examples/mediapipe_face.rs`).
//!
//! Decodes an image, runs palm detection → rotated ROI → hand landmarks
//! through [`HandsTracker`], and prints each hand's handedness, confidence,
//! per-finger curls, and the classified gesture. Lets the port be verified
//! against a real photo without a live webcam or VRChat (the CLAUDE.md
//! "demo before push" for this component).
//!
//! ```bash
//! cargo run --release --example hands_image -- photo.jpg \
//!   models/palm_detection.onnx models/hand_landmarks.onnx
//! ```

use anyhow::{bail, Result};

use vrchat_camera_osc::capture::Frame;
use vrchat_camera_osc::mapping::gesture::{classify, finger_curls, FingerHysteresis};
use vrchat_camera_osc::tracking::hands::{HandTracker, HandsTracker};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        bail!("usage: hands_image <in.jpg> <palm.onnx> <landmarks.onnx>");
    }

    let img = image::open(&a[0])?.to_rgb8();
    let (w, h) = img.dimensions();
    let frame = Frame::new(w, h, img.into_raw())?;

    let mut tracker = HandsTracker::from_paths(&a[1], &a[2])?;
    let hands = tracker.track(&frame)?;
    println!("image {w}x{h}: {} hand(s)", hands.len());

    for (i, hand) in hands.iter().enumerate() {
        let curls = finger_curls(&hand.points);
        let mut hysteresis = FingerHysteresis::default();
        let gesture = classify(&hysteresis.update(curls));
        println!(
            "hand {i}: {:?} (model label), confidence {:.2}",
            hand.handedness, hand.confidence
        );
        for (name, c) in ["thumb", "index", "middle", "ring", "pinky"]
            .iter()
            .zip(curls)
        {
            println!("  {name:<6} curl {c:.2}");
        }
        println!("  gesture: {} ({})", gesture.index(), gesture.name());
        let wrist = hand.points[0];
        println!("  wrist at ({:.2}, {:.2}) normalized", wrist[0], wrist[1]);
    }
    Ok(())
}
