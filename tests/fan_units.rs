//! CI-runnable parity tests that need no model download:
//! - `transform()` bit-parity against the reference (`transform.json`).
//! - heatmap→landmark decode parity against the reference (`fan_preds.json`,
//!   decoded from the committed `fan_output.f32`).
//! - a synthetic decode unit test with a hand-placed peak.

mod common;

use candle_core::{Device, Tensor};
use serde::Deserialize;
use vrchat_camera_osc::tracking::fan::{decode, preprocess};

#[derive(Deserialize)]
struct TransformCase {
    point: [f32; 2],
    center: [f32; 2],
    scale: f32,
    res: f32,
    invert: bool,
    expected: [i64; 2],
}

#[test]
fn transform_matches_reference() {
    let cases: Vec<TransformCase> =
        serde_json::from_slice(&std::fs::read(common::fixture("transform.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    for c in cases {
        let got = preprocess::transform(
            (c.point[0], c.point[1]),
            (c.center[0], c.center[1]),
            c.scale,
            c.res,
            c.invert,
        );
        assert_eq!(
            [got.0, got.1],
            c.expected,
            "transform mismatch for point {:?} invert={}",
            c.point,
            c.invert
        );
    }
}

#[derive(Deserialize)]
struct RefPred {
    x: f32,
    y: f32,
    score: f32,
}

#[test]
fn decode_matches_reference_preds() {
    // Decode the committed final-stage heatmaps and compare to the reference
    // `get_preds_fromhm` output. No model weights needed.
    let out = common::read_f32(&common::fixture("fan_output.f32"));
    assert_eq!(out.len(), 68 * 64 * 64, "unexpected fan_output size");
    let hm = Tensor::from_vec(out, (1, 68, 64, 64), &Device::Cpu).unwrap();

    let preds = decode::get_preds_from_hm(&hm).unwrap();
    let refs: Vec<RefPred> =
        serde_json::from_slice(&std::fs::read(common::fixture("fan_preds.json")).unwrap()).unwrap();
    assert_eq!(preds[0].len(), refs.len());

    let mut max_xy = 0f32;
    let mut max_score = 0f32;
    for (got, want) in preds[0].iter().zip(&refs) {
        max_xy = max_xy
            .max((got.x - want.x).abs())
            .max((got.y - want.y).abs());
        max_score = max_score.max((got.score - want.score).abs());
    }
    eprintln!("decode parity: max_xy_diff={max_xy:.3e} max_score_diff={max_score:.3e}");
    assert!(max_xy < 1e-3, "decode coord diff {max_xy} too large");
    assert!(max_score < 1e-4, "decode score diff {max_score} too large");
}

#[test]
fn decode_synthetic_peak() {
    // A single heatmap (1x1x64x64) with a clear peak at (px=10, py=20) and a
    // higher right/down neighbour, so the sub-pixel nudge is +0.25 on each axis.
    let mut plane = vec![0f32; 64 * 64];
    let at = |v: &mut Vec<f32>, r: usize, c: usize, val: f32| v[r * 64 + c] = val;
    at(&mut plane, 20, 10, 1.0);
    at(&mut plane, 20, 11, 0.6); // right neighbour > left(0) => +x
    at(&mut plane, 21, 10, 0.6); // down neighbour  > up(0)   => +y
    let hm = Tensor::from_vec(plane, (1, 1, 64, 64), &Device::Cpu).unwrap();

    let p = decode::get_preds_from_hm(&hm).unwrap()[0][0];
    // x = (px+1) + 0.25 - 0.5 = 10 + 0.75 ; y = 20 + 0.75
    assert!((p.x - 10.75).abs() < 1e-6, "x was {}", p.x);
    assert!((p.y - 20.75).abs() < 1e-6, "y was {}", p.y);
    assert!((p.score - 1.0).abs() < 1e-6);
}
