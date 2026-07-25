//! S3FD post-processing: softmax → per-scale anchor decode → NMS → score
//! filter. Faithful port of `sfd/detect.py::get_predictions` and
//! `sfd/bbox.py::{decode, nms}`.

use super::super::BoundingBox;
use candle_core::{Result, Tensor};
use candle_nn::ops::softmax;

const VARIANCES: [f32; 2] = [0.1, 0.2];
/// Per-anchor score threshold used while collecting candidates (reference 0.05).
const COLLECT_THRESHOLD: f32 = 0.05;

/// Decode the 12 raw head tensors into candidate boxes (pre-NMS).
///
/// `olist` is `[cls1, reg1, ..., cls6, reg6]` exactly as [`super::model::S3fd::forward`]
/// returns them; softmax is applied here to each `cls`.
pub fn decode_boxes(olist: &[Tensor]) -> Result<Vec<BoundingBox>> {
    assert_eq!(olist.len(), 12, "expected 6 (cls,reg) head pairs");
    let mut boxes = Vec::new();

    for i in 0..6 {
        let cls = softmax(&olist[i * 2], 1)?;
        let reg = &olist[i * 2 + 1];
        let (_, _, hh, ww) = cls.dims4()?;
        let stride = (1u32 << (i + 2)) as f32; // 4, 8, 16, 32, 64, 128

        // Face score is channel 1 of cls; reg has 4 channels.
        let cls_v = cls.flatten_all()?.to_vec1::<f32>()?; // [1,2,H,W]
        let reg_v = reg.flatten_all()?.to_vec1::<f32>()?; // [1,4,H,W]
        let plane = hh * ww;
        let score_ch = &cls_v[plane..2 * plane]; // channel 1

        for h in 0..hh {
            for w in 0..ww {
                let idx = h * ww + w;
                let score = score_ch[idx];
                if score <= COLLECT_THRESHOLD {
                    continue;
                }
                let axc = stride / 2.0 + w as f32 * stride;
                let ayc = stride / 2.0 + h as f32 * stride;
                let (pw, ph) = (stride * 4.0, stride * 4.0);

                let loc0 = reg_v[idx];
                let loc1 = reg_v[plane + idx];
                let loc2 = reg_v[2 * plane + idx];
                let loc3 = reg_v[3 * plane + idx];

                let cx = axc + loc0 * VARIANCES[0] * pw;
                let cy = ayc + loc1 * VARIANCES[0] * ph;
                let bw = pw * (loc2 * VARIANCES[1]).exp();
                let bh = ph * (loc3 * VARIANCES[1]).exp();

                let x1 = cx - bw / 2.0;
                let y1 = cy - bh / 2.0;
                boxes.push(BoundingBox {
                    x1,
                    y1,
                    x2: x1 + bw,
                    y2: y1 + bh,
                    score,
                });
            }
        }
    }
    Ok(boxes)
}

/// Greedy non-max suppression, returning kept indices. Matches
/// `sfd/bbox.py::nms` including its `+1` box-size convention.
pub fn nms(boxes: &[BoundingBox], thresh: f32) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let area = |b: &BoundingBox| (b.x2 - b.x1 + 1.0) * (b.y2 - b.y1 + 1.0);
    let areas: Vec<f32> = boxes.iter().map(area).collect();

    // Indices sorted by descending score.
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| boxes[b].score.total_cmp(&boxes[a].score));

    let mut keep = Vec::new();
    while !order.is_empty() {
        let i = order[0];
        keep.push(i);
        let bi = &boxes[i];
        let mut rest = Vec::new();
        for &j in &order[1..] {
            let bj = &boxes[j];
            let xx1 = bi.x1.max(bj.x1);
            let yy1 = bi.y1.max(bj.y1);
            let xx2 = bi.x2.min(bj.x2);
            let yy2 = bi.y2.min(bj.y2);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let ovr = inter / (areas[i] + areas[j] - inter);
            if ovr <= thresh {
                rest.push(j);
            }
        }
        order = rest;
    }
    keep
}

/// Full filter: NMS (0.3) then keep boxes scoring above `filter_threshold`.
pub fn filter_boxes(boxes: Vec<BoundingBox>, filter_threshold: f32) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return boxes;
    }
    let keep = nms(&boxes, 0.3);
    keep.into_iter()
        .map(|i| boxes[i])
        .filter(|b| b.score > filter_threshold)
        .collect()
}
