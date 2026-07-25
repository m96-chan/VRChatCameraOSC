//! Full detector + end-to-end parity vs the PyTorch `face_alignment` reference,
//! on the bundled face photo (issue #9).
//!
//! Requires `models/s3fd.safetensors` and the reference outputs under
//! `reference/assets/` (both gitignored — a face photo + ~85 MB weights),
//! produced by `reference/gen_fixtures.py`. Skips when absent; the committed
//! `sfd_units` test covers the port in CI.

use std::path::{Path, PathBuf};

use image::RgbImage;
use serde::Deserialize;
use vrchat_camera_osc::tracking::detect::sfd::SfdDetector;
use vrchat_camera_osc::tracking::detect::{BoundingBox, FaceDetector};
use vrchat_camera_osc::tracking::fan::FanTracker;

fn root(p: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(p)
}

#[derive(Deserialize)]
struct RefBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
}

#[derive(Deserialize)]
struct RefPt {
    x: f32,
    y: f32,
}

#[derive(Deserialize)]
struct ImgMeta {
    h: u32,
    w: u32,
}

fn load_image() -> Option<RgbImage> {
    let meta_p = root("reference/assets/aflw-test.json");
    let rgb_p = root("reference/assets/aflw-test.rgb");
    if !meta_p.exists() || !rgb_p.exists() {
        return None;
    }
    let meta: ImgMeta = serde_json::from_slice(&std::fs::read(meta_p).unwrap()).unwrap();
    let data = std::fs::read(rgb_p).unwrap();
    RgbImage::from_raw(meta.w, meta.h, data)
}

fn best(boxes: &[BoundingBox]) -> Option<BoundingBox> {
    boxes
        .iter()
        .cloned()
        .max_by(|a, b| a.score.total_cmp(&b.score))
}

#[test]
fn sfd_boxes_match_reference() {
    let weights = root("models/s3fd.safetensors");
    let boxes_json = root("reference/assets/sfd_boxes.json");
    let (Some(image), true, true) = (load_image(), weights.exists(), boxes_json.exists()) else {
        eprintln!("SKIP sfd_boxes_match_reference: weights/assets missing (run gen_fixtures.py).");
        return;
    };

    let det = SfdDetector::from_safetensors(&weights).unwrap();
    let got = det.detect(&image).unwrap();
    let refs: Vec<RefBox> = serde_json::from_slice(&std::fs::read(boxes_json).unwrap()).unwrap();

    assert!(!refs.is_empty(), "reference detected at least one face");
    let rb = &refs[0];
    let gb = best(&got).expect("Rust detector found a face");

    eprintln!(
        "SFD box parity: ref=({:.2},{:.2},{:.2},{:.2}) got=({:.2},{:.2},{:.2},{:.2})",
        rb.x1, rb.y1, rb.x2, rb.y2, gb.x1, gb.y1, gb.x2, gb.y2
    );
    for (g, r, name) in [
        (gb.x1, rb.x1, "x1"),
        (gb.y1, rb.y1, "y1"),
        (gb.x2, rb.x2, "x2"),
        (gb.y2, rb.y2, "y2"),
    ] {
        assert!((g - r).abs() < 2.0, "box {name} off by {}", (g - r).abs());
    }
    assert!((gb.score - rb.score).abs() < 5e-2, "score off");
}

#[test]
fn end_to_end_landmarks_match_reference() {
    let fan_w = root("models/2dfan4.safetensors");
    let sfd_w = root("models/s3fd.safetensors");
    let boxes_json = root("reference/assets/sfd_boxes.json");
    let lms_json = root("reference/assets/fan_image_landmarks.json");
    let (Some(image), true, true, true) = (
        load_image(),
        fan_w.exists(),
        boxes_json.exists() && lms_json.exists(),
        sfd_w.exists(),
    ) else {
        eprintln!("SKIP end_to_end_landmarks_match_reference: weights/assets missing.");
        return;
    };

    // Feed the SAME (reference) box so this isolates crop→FAN→decode→transform
    // parity from any detector difference.
    let refs: Vec<RefBox> = serde_json::from_slice(&std::fs::read(boxes_json).unwrap()).unwrap();
    let rb = &refs[0];
    let bbox = BoundingBox {
        x1: rb.x1,
        y1: rb.y1,
        x2: rb.x2,
        y2: rb.y2,
        score: rb.score,
    };

    let tracker = FanTracker::from_safetensors(&fan_w).unwrap();
    let got = tracker.landmarks_in_image(&image, &bbox).unwrap();

    let refs_lms: Vec<RefPt> = serde_json::from_slice(&std::fs::read(lms_json).unwrap()).unwrap();
    assert_eq!(got.points.len(), refs_lms.len());

    let mut max_d = 0f32;
    let mut sum_d = 0f32;
    for (g, r) in got.points.iter().zip(&refs_lms) {
        let d = ((g.x - r.x).powi(2) + (g.y - r.y).powi(2)).sqrt();
        max_d = max_d.max(d);
        sum_d += d;
    }
    let mean_d = sum_d / got.points.len() as f32;
    eprintln!("end-to-end landmark parity: mean={mean_d:.3}px max={max_d:.3}px");
    // Bilinear crop-resize differs slightly from OpenCV, so allow a few px.
    assert!(mean_d < 3.0, "mean landmark error {mean_d}px too large");
    assert!(max_d < 12.0, "max landmark error {max_d}px too large");
}
