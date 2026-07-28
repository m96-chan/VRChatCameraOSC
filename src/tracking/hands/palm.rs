//! Palm detection stage (issue #8): the OpenCV Zoo
//! `palm_detection_mediapipe` ONNX (192×192 NHWC, RGB `[0,1]`) via
//! candle-onnx on CPU — like YuNet it only (re)seeds ROIs, running rarely
//! behind the async safety-net-redetect pattern (issue #17), so its ~200 ms
//! CPU cost never sits on the frame loop.
//!
//! Pre/post-processing is ported from OpenCV Zoo
//! `models/palm_detection_mediapipe/mp_palmdet.py` (Apache-2.0):
//!
//! - **Preprocess**: keep-aspect resize into 192×192, RGB, `/255`, NHWC. Zoo
//!   centers the letterbox and carries a `pad_bias`; we anchor it top-left
//!   (pad right/bottom only) so `pad_bias = 0` — the decode math is
//!   otherwise identical.
//! - **Anchors**: the SSD anchor grid for strides `[8, 16, 16, 16]` on
//!   192×192 — a 24×24 grid with 2 anchors/cell then a 12×12 grid with 6
//!   anchors/cell, centers `((c+0.5)/grid, (r+0.5)/grid)`, 2016 total.
//!   [`palm_anchors`] regenerates the exact list `mp_palmdet.py` hardcodes
//!   (verified elementwise against it).
//! - **Decode**: `score = sigmoid(raw)`; box center/size deltas are in input
//!   pixels (`/192` normalizes), added to the anchor center;
//!   `scale = max(frame_w, frame_h)` maps normalized coords back to frame
//!   pixels (the exact inverse of the keep-aspect resize). The 7 palm
//!   keypoints decode the same way. Greedy NMS (IoU 0.3) picks the winners.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

use crate::tracking::mediapipe::util::{iou, sample_rgb255, sigmoid};

/// Model input side (192×192).
pub const INPUT: usize = 192;
/// Total anchors across the strides (24·24·2 + 12·12·6).
pub const NUM_ANCHORS: usize = 2016;
/// Values per anchor in the box output: cx, cy, w, h + 7 keypoints × (x, y).
const BOX_VALUES: usize = 18;
/// Minimum sigmoid score to keep an anchor. Zoo's class default is 0.5 and
/// its demo uses 0.8; 0.6 matches the confidence bar we use for YuNet.
const SCORE_THRESHOLD: f32 = 0.6;
/// Greedy-NMS IoU threshold (`nmsThreshold` in `mp_palmdet.py`).
const NMS_IOU: f32 = 0.3;

/// One palm detection, in **original-frame pixels**.
#[derive(Debug, Clone, Copy)]
pub struct PalmDetection {
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// 7 palm keypoints: wrist, index MCP, middle MCP, ring MCP, pinky MCP,
    /// thumb CMC, thumb MCP (the order `hands::roi` expects).
    pub keypoints: [[f32; 2]; 7],
    pub score: f32,
}

/// The SSD anchor centers (normalized `[x, y]`), regenerating the list
/// `mp_palmdet.py` hardcodes: strides `[8, 16, 16, 16]` on a 192×192 input
/// collapse to a 24×24 grid with 2 anchors per cell followed by a 12×12
/// grid with 6 (= 3 stride-16 layers × 2), centers `(c+0.5)/grid`.
pub fn palm_anchors() -> Vec<[f32; 2]> {
    let mut anchors = Vec::with_capacity(NUM_ANCHORS);
    for (grid, per_cell) in [(24usize, 2usize), (12, 6)] {
        for r in 0..grid {
            for c in 0..grid {
                let a = [
                    (c as f32 + 0.5) / grid as f32,
                    (r as f32 + 0.5) / grid as f32,
                ];
                for _ in 0..per_cell {
                    anchors.push(a);
                }
            }
        }
    }
    debug_assert_eq!(anchors.len(), NUM_ANCHORS);
    anchors
}

pub struct PalmDetector {
    model: ModelProto,
    /// Graph initializers (weights), extracted once at load — re-parsing
    /// them per eval is the dominant per-frame cost (fork commit 7b0427d).
    consts: HashMap<String, Tensor>,
    input_name: String,
    device: Device,
    anchors: Vec<[f32; 2]>,
}

impl PalmDetector {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        // CPU on purpose: this stage runs rarely (async redetect) and the
        // candle-onnx interpreter is kernel-launch-bound on GPU anyway (same
        // rationale as YuNet — see `mediapipe::detector`).
        let device = Device::Cpu;
        let model = candle_onnx::read_file(path.as_ref())?;
        let input_name = model
            .graph
            .as_ref()
            .and_then(|g| g.input.first())
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow::anyhow!("palm detector model has no input"))?;
        let consts = candle_onnx::initializer_tensors(&model)?
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_device(&device)?)))
            .collect::<candle_core::Result<HashMap<_, _>>>()?;
        Ok(Self {
            model,
            consts,
            input_name,
            device,
            anchors: palm_anchors(),
        })
    }

    /// Detect up to `max` palms in a packed-RGB8 frame, best score first.
    pub fn detect(
        &self,
        width: u32,
        height: u32,
        rgb: &[u8],
        max: usize,
    ) -> Result<Vec<PalmDetection>> {
        let input = self.preprocess(width, height, rgb)?;
        let mut inputs = self.consts.clone();
        inputs.insert(self.input_name.clone(), input);
        let outputs = candle_onnx::simple_eval_with_initializers(&self.model, inputs)?;

        // Identify outputs by element count (names are `Identity`/
        // `Identity_1` in this export, but counts are unambiguous):
        // boxes+keypoints = 2016×18, scores = 2016×1.
        let mut boxes: Option<Vec<f32>> = None;
        let mut scores: Option<Vec<f32>> = None;
        for v in outputs.values() {
            let n: usize = v.dims().iter().product();
            if n == NUM_ANCHORS * BOX_VALUES {
                boxes = Some(v.flatten_all()?.to_vec1::<f32>()?);
            } else if n == NUM_ANCHORS {
                scores = Some(v.flatten_all()?.to_vec1::<f32>()?);
            }
        }
        let boxes = boxes.ok_or_else(|| anyhow::anyhow!("palm output [1,2016,18] missing"))?;
        let scores = scores.ok_or_else(|| anyhow::anyhow!("palm output [1,2016,1] missing"))?;

        Ok(decode_palms(
            &boxes,
            &scores,
            &self.anchors,
            width,
            height,
            max,
        ))
    }

    /// Keep-aspect resize into a top-left-anchored 192×192 letterbox
    /// (pad right/bottom with 0), RGB `[0,1]`, **NHWC**.
    fn preprocess(&self, width: u32, height: u32, rgb: &[u8]) -> Result<Tensor> {
        let (w, h) = (width as i32, height as i32);
        if w == 0 || h == 0 || rgb.len() < (w * h * 3) as usize {
            bail!("invalid frame for palm detection");
        }
        let scale = (INPUT as f32 / w as f32).min(INPUT as f32 / h as f32);
        let new_w = ((w as f32 * scale).round() as i32).min(INPUT as i32);
        let new_h = ((h as f32 * scale).round() as i32).min(INPUT as i32);

        let mut buf = vec![0f32; INPUT * INPUT * 3]; // NHWC
        for oy in 0..new_h {
            for ox in 0..new_w {
                let sx = (ox as f32 + 0.5) / scale - 0.5;
                let sy = (oy as f32 + 0.5) / scale - 0.5;
                let [r, g, b] = sample_rgb255(width, height, rgb, sx, sy);
                let o = ((oy as usize) * INPUT + ox as usize) * 3;
                buf[o] = r / 255.0;
                buf[o + 1] = g / 255.0;
                buf[o + 2] = b / 255.0;
            }
        }
        Ok(Tensor::from_vec(buf, (1, INPUT, INPUT, 3), &self.device)?)
    }
}

/// Decode raw anchor deltas + scores into NMS'd frame-pixel detections
/// (`mp_palmdet.py::_postprocess`, with `pad_bias = 0` — see module docs).
fn decode_palms(
    boxes: &[f32],
    scores: &[f32],
    anchors: &[[f32; 2]],
    width: u32,
    height: u32,
    max: usize,
) -> Vec<PalmDetection> {
    // The inverse of the keep-aspect resize: normalized [0,1] coords of the
    // 192 input span `max(w, h)` frame pixels (top-left anchored).
    let scale = (width.max(height)) as f32;
    let mut dets = Vec::new();
    for (i, anchor) in anchors.iter().enumerate() {
        let score = sigmoid(scores[i]);
        if score < SCORE_THRESHOLD {
            continue;
        }
        let b = &boxes[i * BOX_VALUES..(i + 1) * BOX_VALUES];
        let inv = 1.0 / INPUT as f32;
        let (cx, cy) = (b[0] * inv, b[1] * inv);
        let (bw, bh) = (b[2] * inv, b[3] * inv);
        let bbox = [
            (cx - bw * 0.5 + anchor[0]) * scale,
            (cy - bh * 0.5 + anchor[1]) * scale,
            (cx + bw * 0.5 + anchor[0]) * scale,
            (cy + bh * 0.5 + anchor[1]) * scale,
        ];
        let mut keypoints = [[0.0f32; 2]; 7];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            *kp = [
                (b[4 + 2 * k] * inv + anchor[0]) * scale,
                (b[4 + 2 * k + 1] * inv + anchor[1]) * scale,
            ];
        }
        dets.push(PalmDetection {
            bbox,
            keypoints,
            score,
        });
    }

    // Greedy NMS, best score first; keep at most `max`.
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<PalmDetection> = Vec::new();
    for d in dets {
        if kept.len() >= max {
            break;
        }
        if kept.iter().all(|k| iou(&k.bbox, &d.bbox) <= NMS_IOU) {
            kept.push(d);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated anchors must reproduce `mp_palmdet.py`'s hardcoded
    /// list — spot-checked at the documented layout boundaries (first
    /// stride-8 anchor, first stride-16 anchor at index 1152, last).
    #[test]
    fn anchors_match_reference_layout() {
        let a = palm_anchors();
        assert_eq!(a.len(), NUM_ANCHORS);
        let close =
            |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < 1e-6 && (p[1] - q[1]).abs() < 1e-6;
        assert!(close(a[0], [0.5 / 24.0, 0.5 / 24.0]), "{:?}", a[0]);
        assert!(close(a[1], a[0]), "2 anchors per stride-8 cell");
        assert!(close(a[2], [1.5 / 24.0, 0.5 / 24.0]), "{:?}", a[2]);
        assert!(close(a[1151], [23.5 / 24.0, 23.5 / 24.0]), "{:?}", a[1151]);
        assert!(close(a[1152], [0.5 / 12.0, 0.5 / 12.0]), "{:?}", a[1152]);
        assert!(close(a[1157], a[1152]), "6 anchors per stride-16 cell");
        assert!(close(a[2015], [11.5 / 12.0, 11.5 / 12.0]), "{:?}", a[2015]);
    }

    /// One confident anchor decodes to the expected frame-pixel box and
    /// keypoints (square frame, so scale == frame side and the letterbox is
    /// the identity).
    #[test]
    fn decode_single_anchor_maps_to_frame_pixels() {
        let anchors = palm_anchors();
        let mut boxes = vec![0.0f32; NUM_ANCHORS * BOX_VALUES];
        let mut scores = vec![-50.0f32; NUM_ANCHORS]; // sigmoid ~ 0
        let i = 100;
        scores[i] = 50.0; // sigmoid ~ 1
                          // Deltas in input pixels: center +9.6 px right, box 48x48 px,
                          // keypoint 0 at the anchor center exactly.
        boxes[i * BOX_VALUES] = 9.6;
        boxes[i * BOX_VALUES + 2] = 48.0;
        boxes[i * BOX_VALUES + 3] = 48.0;

        let dets = decode_palms(&boxes, &scores, &anchors, 200, 200, 4);
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert!(d.score > 0.99);
        let (ax, ay) = (anchors[i][0], anchors[i][1]);
        // 9.6/192 = 0.05 normalized; 48/192 = 0.25; frame scale = 200.
        let cx = (ax + 0.05) * 200.0;
        let cy = ay * 200.0;
        assert!((d.bbox[0] - (cx - 25.0)).abs() < 1e-3, "{:?}", d.bbox);
        assert!((d.bbox[1] - (cy - 25.0)).abs() < 1e-3);
        assert!((d.bbox[2] - (cx + 25.0)).abs() < 1e-3);
        assert!((d.bbox[3] - (cy + 25.0)).abs() < 1e-3);
        // Zero keypoint delta decodes to the anchor center.
        assert!((d.keypoints[0][0] - ax * 200.0).abs() < 1e-3);
        assert!((d.keypoints[0][1] - ay * 200.0).abs() < 1e-3);
    }

    /// Two overlapping confident anchors: NMS keeps only the better one; a
    /// third far away survives. `max` caps the list.
    #[test]
    fn decode_nms_suppresses_overlaps_and_caps_max() {
        let anchors = palm_anchors();
        let mut boxes = vec![0.0f32; NUM_ANCHORS * BOX_VALUES];
        let mut scores = vec![-50.0f32; NUM_ANCHORS];
        // Anchors 0 and 1 share a center (2 per cell) -> identical boxes.
        for i in [0usize, 1] {
            boxes[i * BOX_VALUES + 2] = 40.0;
            boxes[i * BOX_VALUES + 3] = 40.0;
        }
        scores[0] = 5.0;
        scores[1] = 6.0; // better of the overlapping pair
                         // A far-away cell (last stride-16 anchor).
        let j = NUM_ANCHORS - 1;
        boxes[j * BOX_VALUES + 2] = 40.0;
        boxes[j * BOX_VALUES + 3] = 40.0;
        scores[j] = 4.0;

        let dets = decode_palms(&boxes, &scores, &anchors, 192, 192, 4);
        assert_eq!(dets.len(), 2, "one of the pair suppressed");
        assert!(dets[0].score > dets[1].score);
        assert!((dets[0].score - sigmoid(6.0)).abs() < 1e-6);

        let capped = decode_palms(&boxes, &scores, &anchors, 192, 192, 1);
        assert_eq!(capped.len(), 1);
    }

    #[test]
    fn decode_all_low_scores_is_empty() {
        let anchors = palm_anchors();
        let boxes = vec![0.0f32; NUM_ANCHORS * BOX_VALUES];
        let scores = vec![-2.0f32; NUM_ANCHORS]; // sigmoid ~ 0.12 < 0.6
        assert!(decode_palms(&boxes, &scores, &anchors, 100, 100, 4).is_empty());
    }
}
