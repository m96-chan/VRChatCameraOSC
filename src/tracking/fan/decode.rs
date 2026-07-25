//! Heatmap → landmark decoding.
//!
//! Faithful port of `get_preds_fromhm` / `_get_preds_fromhm` from
//! `face_alignment/utils.py`: argmax over each `HxW` heatmap, a ±0.25
//! sub-pixel nudge toward the higher neighbour, then a −0.5 offset. Output
//! coordinates are in **heatmap space** (1-indexed centre convention, matching
//! the reference `preds` return value); map to image space with
//! [`super::preprocess::transform`] `invert=true`.

use candle_core::{Result, Tensor};

/// A decoded point in heatmap coordinates plus its heatmap peak score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatmapPoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

fn sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Decode a batch of heatmaps of shape `[N, C, H, W]` into `N` sets of `C`
/// points (in heatmap space). Mirrors `get_preds_fromhm` exactly.
pub fn get_preds_from_hm(hm: &Tensor) -> Result<Vec<Vec<HeatmapPoint>>> {
    let (n, c, h, w) = hm.dims4()?;
    // Pull the whole tensor to host once; heatmaps are small (68*64*64).
    let data = hm.to_dtype(candle_core::DType::F32)?.flatten_all()?;
    let data = data.to_vec1::<f32>()?;
    let hw = h * w;

    let mut out = Vec::with_capacity(n);
    for ni in 0..n {
        let mut pts = Vec::with_capacity(c);
        for ci in 0..c {
            let base = (ni * c + ci) * hw;
            let plane = &data[base..base + hw];

            // argmax over the HxW plane.
            let mut best = 0usize;
            let mut best_v = plane[0];
            for (i, &v) in plane.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            let px = best % w; // 0-indexed column
            let py = best / w; // 0-indexed row

            // 1-indexed centre coords (see reference).
            let mut x = (px + 1) as f32;
            let mut y = (py + 1) as f32;

            // Sub-pixel refinement toward the higher neighbour.
            if px > 0 && px < w - 1 && py > 0 && py < h - 1 {
                let at = |r: usize, col: usize| plane[r * w + col];
                let diff_x = at(py, px + 1) - at(py, px - 1);
                let diff_y = at(py + 1, px) - at(py - 1, px);
                x += sign(diff_x) * 0.25;
                y += sign(diff_y) * 0.25;
            }
            x -= 0.5;
            y -= 0.5;

            pts.push(HeatmapPoint {
                x,
                y,
                score: best_v,
            });
        }
        out.push(pts);
    }
    Ok(out)
}
