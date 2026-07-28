//! Issue #20 R&D spike: can the Project Babble lower-face model detect the
//! tongue from a FRONT-FACING desk webcam? (The model is trained on HMD
//! mouth-cam views; this is the go/no-go question for webcam tongue
//! tracking.)
//!
//! Run:
//! ```bash
//! cargo run --release --example babble_spike -- <babble_model.onnx> [frames]
//! ```
//! Requires a webcam and `models/face_detection.onnx` (auto-downloaded by
//! the main app). Prints the tongue channels + a jawOpen sanity reference
//! per frame, then a summary. NOT part of the product: the Babble model is
//! under a non-commercial license and is deliberately not bundled or
//! auto-downloaded here — pass a locally obtained copy.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};

use vrchat_camera_osc::capture::CameraSource;
use vrchat_camera_osc::tracking::mediapipe::FaceDetector;

const BABBLE_INPUT: usize = 224;
/// Babble output order (BabbleApp/osc.py). Indices of interest.
const IDX_JAW_OPEN: usize = 4;
const IDX_MOUTH_PUCKER: usize = 11;
const IDX_TONGUE_OUT: usize = 33;
const IDX_TONGUE_UP: usize = 34;
const IDX_TONGUE_DOWN: usize = 35;
const IDX_TONGUE_LEFT: usize = 36;
const IDX_TONGUE_RIGHT: usize = 37;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().context("usage: babble_spike <model.onnx> [frames]")?;
    let frames: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(300);

    let model = candle_onnx::read_file(&model_path)?;
    let device = Device::Cpu;
    let consts = candle_onnx::initializer_tensors(&model)?
        .into_iter()
        .map(|(k, v)| Ok((k, v.to_device(&device)?)))
        .collect::<candle_core::Result<HashMap<_, _>>>()?;
    let input_name = model
        .graph
        .as_ref()
        .and_then(|g| g.input.first())
        .map(|i| i.name.clone())
        .context("babble model has no input")?;
    eprintln!("babble model loaded: input '{input_name}'");

    let detector = FaceDetector::from_path("models/face_detection.onnx")?;

    let mut camera = open_camera()?;
    let mut peaks = [0f32; 45];
    let mut sums = [0f32; 45];
    let mut n_ok = 0u32;

    for frame_no in 0..frames {
        let frame = camera.next_frame()?;
        let Some(det) = detector.detect(frame.width, frame.height, &frame.data)? else {
            println!("[{frame_no}] no face");
            continue;
        };
        // Mouth-centric square crop: centered on the mouth-corner midpoint,
        // sized from the face box (Babble sees a mouth-dominant view on HMD
        // cams; this approximates it from a front camera).
        let mouth_mid = [
            (det.keypoints[3][0] + det.keypoints[4][0]) / 2.0,
            (det.keypoints[3][1] + det.keypoints[4][1]) / 2.0,
        ];
        let face_w = det.bbox[2] - det.bbox[0];
        let half = (face_w * 0.45).max(16.0);
        let crop = gray_crop_224(&frame.data, frame.width, frame.height, mouth_mid, half)?;

        let input = Tensor::from_vec(crop, (1, 1, BABBLE_INPUT, BABBLE_INPUT), &device)?;
        let mut inputs = consts.clone();
        inputs.insert(input_name.clone(), input);
        let outputs = candle_onnx::simple_eval_with_initializers(&model, inputs)?;
        let out = outputs
            .values()
            .next()
            .context("no output")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if out.len() < 45 {
            bail!("unexpected output length {}", out.len());
        }
        for (i, v) in out.iter().enumerate().take(45) {
            let v = v.clamp(0.0, 1.0);
            peaks[i] = peaks[i].max(v);
            sums[i] += v;
        }
        n_ok += 1;
        println!(
            "[{frame_no}] jawOpen={:.2} pucker={:.2} | tongueOut={:.2} up={:.2} down={:.2} left={:.2} right={:.2}",
            out[IDX_JAW_OPEN].clamp(0.0, 1.0),
            out[IDX_MOUTH_PUCKER].clamp(0.0, 1.0),
            out[IDX_TONGUE_OUT].clamp(0.0, 1.0),
            out[IDX_TONGUE_UP].clamp(0.0, 1.0),
            out[IDX_TONGUE_DOWN].clamp(0.0, 1.0),
            out[IDX_TONGUE_LEFT].clamp(0.0, 1.0),
            out[IDX_TONGUE_RIGHT].clamp(0.0, 1.0),
        );
    }

    println!("\n== summary over {n_ok} frames (peak / mean) ==");
    let names = [
        (IDX_JAW_OPEN, "jawOpen"),
        (IDX_MOUTH_PUCKER, "mouthPucker"),
        (IDX_TONGUE_OUT, "tongueOut"),
        (IDX_TONGUE_UP, "tongueUp"),
        (IDX_TONGUE_DOWN, "tongueDown"),
        (IDX_TONGUE_LEFT, "tongueLeft"),
        (IDX_TONGUE_RIGHT, "tongueRight"),
    ];
    for (i, name) in names {
        println!(
            "{name:12} peak={:.3} mean={:.3}",
            peaks[i],
            sums[i] / n_ok.max(1) as f32
        );
    }
    Ok(())
}

/// Square crop around `center` with half-size `half`, converted to grayscale
/// and bilinearly resized to 224×224, values in `0..1`.
fn gray_crop_224(
    rgb: &[u8],
    w: u32,
    h: u32,
    center: [f32; 2],
    half: f32,
) -> Result<Vec<f32>> {
    let (w, h) = (w as f32, h as f32);
    let side = half * 2.0;
    let x0 = center[0] - half;
    let y0 = center[1] - half;
    let mut out = Vec::with_capacity(BABBLE_INPUT * BABBLE_INPUT);
    let sample = |x: f32, y: f32| -> f32 {
        let xi = x.clamp(0.0, w - 1.0) as usize;
        let yi = y.clamp(0.0, h - 1.0) as usize;
        let p = (yi * w as usize + xi) * 3;
        // Rec.601 luma.
        0.299 * rgb[p] as f32 + 0.587 * rgb[p + 1] as f32 + 0.114 * rgb[p + 2] as f32
    };
    for oy in 0..BABBLE_INPUT {
        for ox in 0..BABBLE_INPUT {
            let sx = x0 + side * ox as f32 / BABBLE_INPUT as f32;
            let sy = y0 + side * oy as f32 / BABBLE_INPUT as f32;
            out.push(sample(sx, sy) / 255.0);
        }
    }
    Ok(out)
}

#[cfg(feature = "camera")]
fn open_camera() -> Result<Box<dyn CameraSource>> {
    use vrchat_camera_osc::capture::native::NativeCamera;
    Ok(Box::new(NativeCamera::open(0, 640, 480)?))
}

#[cfg(not(feature = "camera"))]
fn open_camera() -> Result<Box<dyn CameraSource>> {
    bail!("this spike needs the `camera` feature (a real webcam)")
}
