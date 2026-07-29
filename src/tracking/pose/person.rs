//! Person detection stage (issue #28 phase 2): the OpenCV Zoo
//! `person_detection_mediapipe` ONNX (224×224 NCHW, RGB `[-1,1]`) via
//! candle-onnx on CPU — like YuNet/palm it only (re)seeds the pose ROI,
//! running rarely behind the async safety-net-redetect pattern (issue #17),
//! so its ~500 ms CPU cost never sits on the frame loop.
//!
//! Pre/post-processing is ported from OpenCV Zoo
//! `models/person_detection_mediapipe/mp_persondet.py` (Apache-2.0):
//!
//! - **Preprocess**: keep-aspect resize into 224×224, RGB,
//!   `(v/255 − 0.5)·2` → `[-1, 1]`, **NCHW** (their `np.transpose` to chw —
//!   unlike the NHWC palm net). Zoo centers the letterbox and carries a
//!   `pad_bias`; we anchor it top-left (pad right/bottom only) so
//!   `pad_bias = 0` — the decode math is otherwise identical.
//! - **Anchors**: the SSD anchor grid on 224×224 — a 28×28 grid with 2
//!   anchors/cell, a 14×14 grid with 2, then a 7×7 grid with 6, centers
//!   `((c+0.5)/grid, (r+0.5)/grid)`, 2254 total. [`person_anchors`]
//!   regenerates the exact list `mp_persondet.py` hardcodes (verified
//!   elementwise against it, 2026-07-29).
//! - **Decode**: `score = sigmoid(raw)`; box center/size deltas are in input
//!   pixels (`/224` normalizes), added to the anchor center;
//!   `scale = max(frame_w, frame_h)` maps normalized coords back to frame
//!   pixels. The 4 person keypoints — mid-hip center, full-body point,
//!   shoulder center, upper-body point — decode the same way. Greedy NMS
//!   (IoU 0.3) picks the winners.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

use crate::tracking::mediapipe::util::{iou, sample_rgb255, sigmoid};

/// Model input side (224×224).
pub const INPUT: usize = 224;
/// Total anchors across the grids (28·28·2 + 14·14·2 + 7·7·6).
pub const NUM_ANCHORS: usize = 2254;
/// Values per anchor in the box output: cx, cy, w, h + 4 keypoints × (x, y).
const BOX_VALUES: usize = 12;
/// Minimum sigmoid score to keep an anchor (`scoreThreshold` in the Zoo
/// class and demo).
const SCORE_THRESHOLD: f32 = 0.5;
/// Greedy-NMS IoU threshold (`nmsThreshold` in `mp_persondet.py`).
const NMS_IOU: f32 = 0.3;

// Keypoint indices in [`PersonDetection::keypoints`] (mp_persondet.py's
// landmark comment: "hip center point; full body point; shoulder center
// point; upper body point").
/// Mid-hip center — the pose ROI center.
pub const KP_HIP_CENTER: usize = 0;
/// Full-body point — the pose ROI scale/rotation point.
pub const KP_FULL_BODY: usize = 1;

/// One person detection, in **original-frame pixels**.
#[derive(Debug, Clone, Copy)]
pub struct PersonDetection {
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// 4 person keypoints: mid-hip center, full-body point, shoulder
    /// center, upper-body point (see [`KP_HIP_CENTER`] / [`KP_FULL_BODY`]).
    pub keypoints: [[f32; 2]; 4],
    pub score: f32,
}

/// The SSD anchor centers (normalized `[x, y]`), regenerating the list
/// `mp_persondet.py` hardcodes: a 28×28 grid with 2 anchors per cell, a
/// 14×14 grid with 2, then a 7×7 grid with 6, centers `(c+0.5)/grid`
/// (verified elementwise against the reference's 2254-entry array).
pub fn person_anchors() -> Vec<[f32; 2]> {
    let mut anchors = Vec::with_capacity(NUM_ANCHORS);
    for (grid, per_cell) in [(28usize, 2usize), (14, 2), (7, 6)] {
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

pub struct PersonDetector {
    model: ModelProto,
    /// Graph initializers (weights), extracted once at load — re-parsing
    /// them per eval is the dominant per-frame cost (fork commit 7b0427d).
    consts: HashMap<String, Tensor>,
    input_name: String,
    device: Device,
    anchors: Vec<[f32; 2]>,
}

impl PersonDetector {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        // CPU on purpose: this stage runs rarely (async redetect) and the
        // candle-onnx interpreter is kernel-launch-bound on GPU anyway (same
        // rationale as YuNet/palm).
        let device = Device::Cpu;
        let model = candle_onnx::read_file(path.as_ref())?;
        let input_name = model
            .graph
            .as_ref()
            .and_then(|g| g.input.first())
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow::anyhow!("person detector model has no input"))?;
        let consts = candle_onnx::initializer_tensors(&model)?
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_device(&device)?)))
            .collect::<candle_core::Result<HashMap<_, _>>>()?;
        Ok(Self {
            model,
            consts,
            input_name,
            device,
            anchors: person_anchors(),
        })
    }

    /// Detect up to `max` persons in a packed-RGB8 frame, best score first.
    pub fn detect(
        &self,
        width: u32,
        height: u32,
        rgb: &[u8],
        max: usize,
    ) -> Result<Vec<PersonDetection>> {
        let input = self.preprocess(width, height, rgb)?;
        let mut inputs = self.consts.clone();
        inputs.insert(self.input_name.clone(), input);
        let outputs = candle_onnx::simple_eval_with_initializers(&self.model, inputs)?;

        // Identify outputs by element count (names are `Identity`/
        // `Identity_1` in this export, but counts are unambiguous):
        // boxes+keypoints = 2254×12, scores = 2254×1.
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
        let boxes = boxes.ok_or_else(|| anyhow::anyhow!("person output [1,2254,12] missing"))?;
        let scores = scores.ok_or_else(|| anyhow::anyhow!("person output [1,2254,1] missing"))?;

        Ok(decode_persons(
            &boxes,
            &scores,
            &self.anchors,
            width,
            height,
            max,
        ))
    }

    /// Keep-aspect resize into a top-left-anchored 224×224 letterbox
    /// (pad right/bottom with 0), RGB `(v/255 − 0.5)·2` → `[-1,1]`, **NCHW**.
    fn preprocess(&self, width: u32, height: u32, rgb: &[u8]) -> Result<Tensor> {
        let (w, h) = (width as i32, height as i32);
        if w == 0 || h == 0 || rgb.len() < (w * h * 3) as usize {
            bail!("invalid frame for person detection");
        }
        let scale = (INPUT as f32 / w as f32).min(INPUT as f32 / h as f32);
        let new_w = ((w as f32 * scale).round() as i32).min(INPUT as i32);
        let new_h = ((h as f32 * scale).round() as i32).min(INPUT as i32);

        // Letterbox pad value: Zoo normalizes to [-1,1] *first* and then
        // pads with literal 0.0 (copyMakeBorder constant 0 on the
        // normalized image), so the zero-initialized buffer is already the
        // correct padding.
        let mut buf = vec![0f32; 3 * INPUT * INPUT]; // NCHW
        for oy in 0..new_h {
            for ox in 0..new_w {
                let sx = (ox as f32 + 0.5) / scale - 0.5;
                let sy = (oy as f32 + 0.5) / scale - 0.5;
                let [r, g, b] = sample_rgb255(width, height, rgb, sx, sy);
                let o = (oy as usize) * INPUT + ox as usize;
                buf[o] = (r / 255.0 - 0.5) * 2.0;
                buf[INPUT * INPUT + o] = (g / 255.0 - 0.5) * 2.0;
                buf[2 * INPUT * INPUT + o] = (b / 255.0 - 0.5) * 2.0;
            }
        }
        Ok(Tensor::from_vec(buf, (1, 3, INPUT, INPUT), &self.device)?)
    }
}

/// Decode raw anchor deltas + scores into NMS'd frame-pixel detections
/// (`mp_persondet.py::_postprocess`, with `pad_bias = 0` — see module docs).
fn decode_persons(
    boxes: &[f32],
    scores: &[f32],
    anchors: &[[f32; 2]],
    width: u32,
    height: u32,
    max: usize,
) -> Vec<PersonDetection> {
    // The inverse of the keep-aspect resize: normalized [0,1] coords of the
    // 224 input span `max(w, h)` frame pixels (top-left anchored).
    let scale = (width.max(height)) as f32;
    let mut dets = Vec::new();
    for (i, anchor) in anchors.iter().enumerate() {
        let score = sigmoid(scores[i].clamp(-100.0, 100.0));
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
        let mut keypoints = [[0.0f32; 2]; 4];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            *kp = [
                (b[4 + 2 * k] * inv + anchor[0]) * scale,
                (b[4 + 2 * k + 1] * inv + anchor[1]) * scale,
            ];
        }
        dets.push(PersonDetection {
            bbox,
            keypoints,
            score,
        });
    }

    // Greedy NMS, best score first; keep at most `max`.
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<PersonDetection> = Vec::new();
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

    /// The generated anchors must reproduce `mp_persondet.py`'s hardcoded
    /// 2254-entry list — spot-checked at the documented layout boundaries
    /// (first 28-grid anchor, first 14-grid anchor at 1568, first 7-grid
    /// anchor at 1960, last). The full list was verified elementwise against
    /// the reference offline (2026-07-29).
    #[test]
    fn anchors_match_reference_layout() {
        let a = person_anchors();
        assert_eq!(a.len(), NUM_ANCHORS);
        let close =
            |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < 1e-6 && (p[1] - q[1]).abs() < 1e-6;
        // 28×28 grid, 2 anchors/cell: first cell center 0.5/28 (the
        // reference's literal 0.017857142857142856).
        assert!(close(a[0], [0.5 / 28.0, 0.5 / 28.0]), "{:?}", a[0]);
        assert!(close(a[1], a[0]), "2 anchors per 28-grid cell");
        assert!(close(a[2], [1.5 / 28.0, 0.5 / 28.0]), "{:?}", a[2]);
        assert!(close(a[1567], [27.5 / 28.0, 27.5 / 28.0]), "{:?}", a[1567]);
        // 14×14 grid, 2/cell, starts at 28·28·2 = 1568.
        assert!(close(a[1568], [0.5 / 14.0, 0.5 / 14.0]), "{:?}", a[1568]);
        assert!(close(a[1569], a[1568]), "2 anchors per 14-grid cell");
        assert!(close(a[1959], [13.5 / 14.0, 13.5 / 14.0]), "{:?}", a[1959]);
        // 7×7 grid, 6/cell, starts at 1568 + 14·14·2 = 1960.
        assert!(close(a[1960], [0.5 / 7.0, 0.5 / 7.0]), "{:?}", a[1960]);
        assert!(close(a[1965], a[1960]), "6 anchors per 7-grid cell");
        assert!(close(a[2253], [6.5 / 7.0, 6.5 / 7.0]), "{:?}", a[2253]);
    }

    /// One confident anchor decodes to the expected frame-pixel box and
    /// keypoints (square frame, so scale == frame side and the letterbox is
    /// the identity).
    #[test]
    fn decode_single_anchor_maps_to_frame_pixels() {
        let anchors = person_anchors();
        let mut boxes = vec![0.0f32; NUM_ANCHORS * BOX_VALUES];
        let mut scores = vec![-50.0f32; NUM_ANCHORS]; // sigmoid ~ 0
        let i = 200;
        scores[i] = 50.0; // sigmoid ~ 1
                          // Deltas in input pixels: center +11.2 px right, box 56x56 px,
                          // keypoint 0 at the anchor center, keypoint 1 straight above by
                          // 22.4 px.
        boxes[i * BOX_VALUES] = 11.2;
        boxes[i * BOX_VALUES + 2] = 56.0;
        boxes[i * BOX_VALUES + 3] = 56.0;
        boxes[i * BOX_VALUES + 7] = -22.4; // kp1.y delta

        let dets = decode_persons(&boxes, &scores, &anchors, 200, 200, 4);
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert!(d.score > 0.99);
        let (ax, ay) = (anchors[i][0], anchors[i][1]);
        // 11.2/224 = 0.05 normalized; 56/224 = 0.25; frame scale = 200.
        let cx = (ax + 0.05) * 200.0;
        let cy = ay * 200.0;
        assert!((d.bbox[0] - (cx - 25.0)).abs() < 1e-3, "{:?}", d.bbox);
        assert!((d.bbox[1] - (cy - 25.0)).abs() < 1e-3);
        assert!((d.bbox[2] - (cx + 25.0)).abs() < 1e-3);
        assert!((d.bbox[3] - (cy + 25.0)).abs() < 1e-3);
        // Zero keypoint delta decodes to the anchor center; kp1 is 22.4/224
        // = 0.1 normalized above it (20 frame px).
        assert!((d.keypoints[KP_HIP_CENTER][0] - ax * 200.0).abs() < 1e-3);
        assert!((d.keypoints[KP_HIP_CENTER][1] - ay * 200.0).abs() < 1e-3);
        assert!((d.keypoints[KP_FULL_BODY][0] - ax * 200.0).abs() < 1e-3);
        assert!((d.keypoints[KP_FULL_BODY][1] - (ay * 200.0 - 20.0)).abs() < 1e-3);
    }

    /// Two overlapping confident anchors: NMS keeps only the better one; a
    /// third far away survives. `max` caps the list.
    #[test]
    fn decode_nms_suppresses_overlaps_and_caps_max() {
        let anchors = person_anchors();
        let mut boxes = vec![0.0f32; NUM_ANCHORS * BOX_VALUES];
        let mut scores = vec![-50.0f32; NUM_ANCHORS];
        // Anchors 0 and 1 share a center (2 per cell) -> identical boxes.
        for i in [0usize, 1] {
            boxes[i * BOX_VALUES + 2] = 40.0;
            boxes[i * BOX_VALUES + 3] = 40.0;
        }
        scores[0] = 5.0;
        scores[1] = 6.0; // better of the overlapping pair
                         // A far-away cell (last 7-grid anchor).
        let j = NUM_ANCHORS - 1;
        boxes[j * BOX_VALUES + 2] = 40.0;
        boxes[j * BOX_VALUES + 3] = 40.0;
        scores[j] = 4.0;

        let dets = decode_persons(&boxes, &scores, &anchors, 224, 224, 4);
        assert_eq!(dets.len(), 2, "one of the pair suppressed");
        assert!(dets[0].score > dets[1].score);
        assert!((dets[0].score - sigmoid(6.0)).abs() < 1e-6);

        let capped = decode_persons(&boxes, &scores, &anchors, 224, 224, 1);
        assert_eq!(capped.len(), 1);
    }

    #[test]
    fn decode_all_low_scores_is_empty() {
        let anchors = person_anchors();
        let boxes = vec![0.0f32; NUM_ANCHORS * BOX_VALUES];
        let scores = vec![-2.0f32; NUM_ANCHORS]; // sigmoid ~ 0.12 < 0.5
        assert!(decode_persons(&boxes, &scores, &anchors, 100, 100, 4).is_empty());
    }
}
