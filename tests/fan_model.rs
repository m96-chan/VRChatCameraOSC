//! CI-runnable tests for the FAN model plumbing and the `FanTracker` end-to-end
//! path, using randomly-initialised weights (no model download needed). These
//! exercise the full forward graph, preprocessing, and decode for coverage;
//! numeric correctness against PyTorch is covered by `fan_parity`/`fan_units`.

mod common;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use image::{Rgb, RgbImage};
use vrchat_camera_osc::capture::Frame;
use vrchat_camera_osc::tracking::fan::model::Fan;
use vrchat_camera_osc::tracking::fan::{preprocess, FanTracker};
use vrchat_camera_osc::tracking::{FaceTracker, NUM_LANDMARKS};

fn random_fan(device: &Device) -> (VarMap, Fan) {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Fan::load(vb).expect("build random FAN");
    (varmap, model)
}

#[test]
fn fan_forward_shape_and_finite() {
    let device = Device::Cpu;
    let (_vm, model) = random_fan(&device);
    let x = Tensor::zeros((1, 3, 256, 256), DType::F32, &device).unwrap();

    // One forward pass (debug-mode candle conv is slow); `forward()` is a thin
    // wrapper over `forward_all()` and is additionally covered by `fan_parity`.
    let all = model.forward_all(&x).unwrap();
    assert_eq!(all.len(), 4, "2DFAN4 has four output stages");
    for stage in &all {
        assert_eq!(stage.dims(), &[1, 68, 64, 64]);
    }
    let v = all
        .last()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "heatmaps must be finite");
}

#[test]
fn fantracker_track_returns_68_landmarks() {
    let device = Device::Cpu;
    let (_vm, model) = random_fan(&device);
    let mut tracker = FanTracker::new(model, device);

    // A non-trivial synthetic frame (a gradient), sized differently from 256 to
    // exercise the resize + coordinate mapping back to frame pixels.
    let (w, h) = (320u32, 240u32);
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            data.push((x % 256) as u8);
            data.push((y % 256) as u8);
            data.push(128);
        }
    }
    let frame = Frame::new(w, h, data).unwrap();

    let lms = tracker
        .track(&frame)
        .unwrap()
        .expect("a face region is assumed");
    assert_eq!(lms.points.len(), NUM_LANDMARKS);
    for p in &lms.points {
        assert!(p.x >= 0.0 && p.x <= w as f32, "x {} out of frame", p.x);
        assert!(p.y >= 0.0 && p.y <= h as f32, "y {} out of frame", p.y);
        assert!(p.score.is_finite());
    }
}

#[test]
fn landmarks_in_image_maps_to_image_space() {
    // Random FAN + a face box → landmarks_in_image runs the crop→FAN→decode→
    // transform(invert) path and returns 68 points in image coordinates. Covers
    // the detector-fed path in CI without weights.
    use vrchat_camera_osc::tracking::detect::BoundingBox;
    let device = Device::Cpu;
    let (_vm, model) = random_fan(&device);
    let tracker = FanTracker::new(model, device);

    let img = RgbImage::from_pixel(400, 400, Rgb([90, 110, 130]));
    let bbox = BoundingBox {
        x1: 120.0,
        y1: 130.0,
        x2: 280.0,
        y2: 320.0,
        score: 0.99,
    };
    let lms = tracker.landmarks_in_image(&img, &bbox).unwrap();
    assert_eq!(lms.points.len(), NUM_LANDMARKS);
    for p in &lms.points {
        assert!(p.x.is_finite() && p.y.is_finite());
    }
}

#[test]
fn crop_and_input_tensor() {
    // crop() to the network input size, then pack to a normalised tensor.
    let mut img = RgbImage::new(200, 200);
    for p in img.pixels_mut() {
        *p = Rgb([10, 20, 30]);
    }
    let cropped = preprocess::crop(&img, (100.0, 100.0), 1.0, preprocess::INPUT_RES);
    assert_eq!(cropped.width(), preprocess::INPUT_RES);
    assert_eq!(cropped.height(), preprocess::INPUT_RES);

    let t = preprocess::to_input_tensor(&cropped, &Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[1, 3, 256, 256]);
    let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        v.iter().all(|&x| (0.0..=1.0).contains(&x)),
        "values must be in [0,1]"
    );
}
