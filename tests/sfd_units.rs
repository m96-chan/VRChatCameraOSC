//! CI-runnable unit tests for the S3FD post-processing and net plumbing
//! (no model download needed): anchor decode, NMS, box→center/scale, and a
//! random-weight forward shape check (exercises the net + L2Norm).

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use image::RgbImage;
use vrchat_camera_osc::tracking::detect::sfd::decode::{decode_boxes, filter_boxes, nms};
use vrchat_camera_osc::tracking::detect::sfd::model::S3fd;
use vrchat_camera_osc::tracking::detect::sfd::SfdDetector;
use vrchat_camera_osc::tracking::detect::{BoundingBox, FaceDetector};
use vrchat_camera_osc::tracking::fan::box_to_center_scale;

fn bb(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> BoundingBox {
    BoundingBox {
        x1,
        y1,
        x2,
        y2,
        score,
    }
}

#[test]
fn nms_suppresses_overlap_keeps_best() {
    // Two nearly-identical boxes (high overlap) + one far away.
    let boxes = vec![
        bb(10.0, 10.0, 50.0, 50.0, 0.9),
        bb(12.0, 12.0, 52.0, 52.0, 0.8),     // overlaps the first
        bb(200.0, 200.0, 240.0, 240.0, 0.7), // disjoint
    ];
    let keep = nms(&boxes, 0.3);
    assert!(keep.contains(&0), "highest-score box kept");
    assert!(!keep.contains(&1), "overlapping weaker box suppressed");
    assert!(keep.contains(&2), "disjoint box kept");
}

#[test]
fn nms_empty_is_empty() {
    assert!(nms(&[], 0.3).is_empty());
}

#[test]
fn filter_applies_threshold() {
    let boxes = vec![
        bb(0.0, 0.0, 10.0, 10.0, 0.9),
        bb(100.0, 100.0, 110.0, 110.0, 0.2),
    ];
    let kept = filter_boxes(boxes, 0.5);
    assert_eq!(kept.len(), 1);
    assert!((kept[0].score - 0.9).abs() < 1e-6);
}

#[test]
fn box_to_center_scale_matches_formula() {
    // center = box centre, nudged up by 0.12*height; scale = (w+h)/195.
    let b = bb(100.0, 120.0, 200.0, 260.0, 1.0);
    let ((cx, cy), scale) = box_to_center_scale(&b);
    let (w, h) = (100.0f32, 140.0f32);
    assert!((cx - 150.0).abs() < 1e-4);
    assert!((cy - (190.0 - h * 0.12)).abs() < 1e-4);
    assert!((scale - (w + h) / 195.0).abs() < 1e-5);
}

/// Build the 12 head tensors with all anchors "background" except one planted
/// face anchor at scale 0, location (0,0), with zero regression.
fn synthetic_heads(device: &Device) -> Vec<Tensor> {
    let mut heads = Vec::with_capacity(12);
    // scale sizes (arbitrary small); scale 0 is 2x2, the rest 1x1.
    let sizes = [2usize, 1, 1, 1, 1, 1];
    for (i, &n) in sizes.iter().enumerate() {
        // cls: [1,2,n,n] logits. channel0 = 0 (background), channel1 = -20 except
        // the planted anchor (+20).
        let mut cls = vec![0f32; 2 * n * n]; // channel0 all 0
        for v in cls.iter_mut().take(2 * n * n).skip(n * n) {
            *v = -20.0; // channel1 background
        }
        if i == 0 {
            cls[n * n] = 20.0; // channel1 at (h=0,w=0) => face
        }
        heads.push(Tensor::from_vec(cls, (1, 2, n, n), device).unwrap());
        // reg: [1,4,n,n] zeros
        heads.push(Tensor::zeros((1, 4, n, n), DType::F32, device).unwrap());
    }
    heads
}

#[test]
fn decode_boxes_from_single_anchor() {
    let device = Device::Cpu;
    let heads = synthetic_heads(&device);
    let boxes = decode_boxes(&heads).unwrap();
    // Exactly one candidate above the 0.05 collect threshold.
    assert_eq!(boxes.len(), 1, "one planted face anchor");
    // scale 0 => stride 4, anchor centre (2,2), size stride*4 = 16, reg=0.
    let b = boxes[0];
    assert!((b.x1 - (2.0 - 8.0)).abs() < 1e-3, "x1 {}", b.x1);
    assert!((b.y1 - (2.0 - 8.0)).abs() < 1e-3, "y1 {}", b.y1);
    assert!((b.x2 - (2.0 + 8.0)).abs() < 1e-3, "x2 {}", b.x2);
    assert!((b.y2 - (2.0 + 8.0)).abs() < 1e-3, "y2 {}", b.y2);
    assert!(b.score > 0.99, "score {}", b.score);
}

#[test]
fn sfd_detector_detect_runs_with_random_weights() {
    // Exercises the full app path — to_input_tensor (BGR/mean) → forward →
    // decode → NMS/filter — in CI without any download. Random weights, so we
    // only assert it runs and returns well-formed boxes.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let detector = SfdDetector::new(S3fd::load(vb).unwrap(), device);

    let img = RgbImage::from_pixel(96, 96, image::Rgb([120, 100, 80]));
    let boxes = detector.detect(&img).unwrap();
    for b in &boxes {
        assert!(b.score.is_finite());
        assert!(b.x2 >= b.x1 && b.y2 >= b.y1);
    }
}

#[test]
fn s3fd_forward_shape_random_weights() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let net = S3fd::load(vb).unwrap();

    let x = Tensor::zeros((1, 3, 128, 128), DType::F32, &device).unwrap();
    let heads = net.forward(&x).unwrap();
    assert_eq!(heads.len(), 12);
    // cls heads (even) have 2 channels, reg heads (odd) have 4.
    for (i, t) in heads.iter().enumerate() {
        let c = t.dims4().unwrap().1;
        let expect = if i % 2 == 0 { 2 } else { 4 };
        assert_eq!(c, expect, "head {i} channel count");
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
