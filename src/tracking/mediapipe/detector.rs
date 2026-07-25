//! YuNet face detector stage (issue #17, ported from AvataCam #5).
//!
//! Wraps the OpenCV Zoo `face_detection_yunet_2023mar` ONNX. The fixed graph
//! takes a `1x3x640x640` **BGR, raw 0-255** input; we letterbox the frame into
//! the top-left (scale to fit, pad bottom/right with 0) — matching OpenCV's
//! pad convention — and decode the multi-stride SSD heads into one best face
//! (bbox + 5 keypoints), in **original-frame pixels**.
//!
//! Decode is verbatim from OpenCV `modules/objdetect/src/face_detect.cpp`:
//! row-major grid, base cell `(c, r)`, `cx=(c+dx)*s`, `w=exp(dw)*s`,
//! `score=sqrt(cls*obj)` (cls/obj already sigmoid).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

use super::util::{iou, sample_rgb255};

const INPUT: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
const SCORE_THRESHOLD: f32 = 0.6;
const NMS_IOU: f32 = 0.3;

/// One face detection, in **original-frame pixels**.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// 5 keypoints: right-eye, left-eye, nose, right-mouth, left-mouth.
    pub keypoints: [[f32; 2]; 5],
    pub score: f32,
}

/// Inverse-letterbox scale: `orig = px640 / scale` (pad is top-left anchored).
#[derive(Clone, Copy)]
struct Letterbox {
    scale: f32,
}

pub struct FaceDetector {
    model: ModelProto,
    input_name: String,
    device: Device,
}

impl FaceDetector {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        // CPU on purpose, even under `--features cuda`: candle-onnx's
        // `simple_eval` materializes the graph's initializers (weights) on
        // CPU, so a CUDA input tensor hits "device mismatch in conv2d".
        // GPU execution for the ONNX path is future work (issue #17).
        let device = Device::Cpu;
        let model = candle_onnx::read_file(path.as_ref())?;
        let input_name = model
            .graph
            .as_ref()
            .and_then(|g| g.input.first())
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow::anyhow!("YuNet detector model has no input"))?;
        Ok(Self {
            model,
            input_name,
            device,
        })
    }

    /// Detect the most confident face, or `None`, in a packed-RGB8 frame.
    pub fn detect(&self, width: u32, height: u32, rgb: &[u8]) -> Result<Option<Detection>> {
        let (input, lb) = self.preprocess(width, height, rgb)?;
        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.clone(), input);
        let outputs = candle_onnx::simple_eval(&self.model, inputs)?;

        let mut dets = Vec::new();
        for &s in &STRIDES {
            let cls = fetch(&outputs, &format!("cls_{s}"))?;
            let obj = fetch(&outputs, &format!("obj_{s}"))?;
            let bbox = fetch(&outputs, &format!("bbox_{s}"))?;
            let kps = fetch(&outputs, &format!("kps_{s}"))?;
            decode_stride(s, &cls, &obj, &bbox, &kps, lb, &mut dets);
        }
        Ok(nms_best(dets))
    }

    /// Letterbox the frame into a `1x3x640x640` BGR, raw-0-255 NCHW tensor.
    fn preprocess(&self, width: u32, height: u32, rgb: &[u8]) -> Result<(Tensor, Letterbox)> {
        let (w, h) = (width as i32, height as i32);
        if w == 0 || h == 0 || rgb.len() < (w * h * 3) as usize {
            bail!("invalid frame for face detection");
        }
        let scale = (INPUT as f32 / w as f32).min(INPUT as f32 / h as f32);
        let new_w = (w as f32 * scale).round() as i32;
        let new_h = (h as f32 * scale).round() as i32;

        // NCHW, channel order B,G,R.
        let plane = INPUT * INPUT;
        let mut buf = vec![0f32; 3 * plane];
        for oy in 0..new_h.min(INPUT as i32) {
            for ox in 0..new_w.min(INPUT as i32) {
                // Source pixel (bilinear) at the un-scaled location.
                let sx = (ox as f32 + 0.5) / scale - 0.5;
                let sy = (oy as f32 + 0.5) / scale - 0.5;
                let [r, g, b] = sample_rgb255(width, height, rgb, sx, sy);
                let o = (oy as usize) * INPUT + ox as usize;
                buf[o] = b; // B plane
                buf[plane + o] = g; // G plane
                buf[2 * plane + o] = r; // R plane
            }
        }
        let t = Tensor::from_vec(buf, (1, 3, INPUT, INPUT), &self.device)?;
        Ok((t, Letterbox { scale }))
    }
}

fn fetch(outputs: &HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    let t = outputs
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("YuNet output {name} missing"))?;
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

#[allow(clippy::too_many_arguments)]
fn decode_stride(
    s: usize,
    cls: &[f32],
    obj: &[f32],
    bbox: &[f32],
    kps: &[f32],
    lb: Letterbox,
    out: &mut Vec<Detection>,
) {
    let cols = INPUT / s;
    let rows = INPUT / s;
    let sf = s as f32;
    let inv = 1.0 / lb.scale;
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if idx >= cls.len() || idx >= obj.len() {
                continue;
            }
            let score = (cls[idx].clamp(0.0, 1.0) * obj[idx].clamp(0.0, 1.0)).sqrt();
            if score < SCORE_THRESHOLD {
                continue;
            }
            let b = &bbox[idx * 4..idx * 4 + 4];
            let cx = (c as f32 + b[0]) * sf;
            let cy = (r as f32 + b[1]) * sf;
            let bw = b[2].exp() * sf;
            let bh = b[3].exp() * sf;
            let bbox_px = [
                (cx - bw * 0.5) * inv,
                (cy - bh * 0.5) * inv,
                (cx + bw * 0.5) * inv,
                (cy + bh * 0.5) * inv,
            ];
            let mut keypoints = [[0.0f32; 2]; 5];
            for (n, kp) in keypoints.iter_mut().enumerate() {
                let kx = (kps[idx * 10 + 2 * n] + c as f32) * sf;
                let ky = (kps[idx * 10 + 2 * n + 1] + r as f32) * sf;
                *kp = [kx * inv, ky * inv];
            }
            out.push(Detection {
                bbox: bbox_px,
                keypoints,
                score,
            });
        }
    }
}

/// Greedy NMS (IoU); returns the single best-scoring surviving detection.
fn nms_best(mut dets: Vec<Detection>) -> Option<Detection> {
    if dets.is_empty() {
        return None;
    }
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for d in dets {
        if kept.iter().all(|k| iou(&k.bbox, &d.bbox) <= NMS_IOU) {
            kept.push(d);
        }
    }
    kept.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_keeps_best_drops_overlap() {
        let mk = |bbox, score| Detection {
            bbox,
            keypoints: [[0.0; 2]; 5],
            score,
        };
        let dets = vec![
            mk([0.0, 0.0, 10.0, 10.0], 0.7),
            mk([1.0, 1.0, 11.0, 11.0], 0.9), // overlaps the first; higher score
            mk([100.0, 100.0, 110.0, 110.0], 0.8), // far away
        ];
        let best = nms_best(dets).unwrap();
        assert!((best.score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn nms_best_empty_is_none() {
        assert!(nms_best(Vec::new()).is_none());
    }

    #[test]
    fn decode_stride_skips_below_threshold() {
        // A single grid cell (stride 32 -> 20x20 grid) with a low score.
        let n = (INPUT / 32) * (INPUT / 32);
        let cls = vec![0.1f32; n];
        let obj = vec![0.1f32; n];
        let bbox = vec![0.0f32; n * 4];
        let kps = vec![0.0f32; n * 10];
        let mut out = Vec::new();
        decode_stride(
            32,
            &cls,
            &obj,
            &bbox,
            &kps,
            Letterbox { scale: 1.0 },
            &mut out,
        );
        assert!(out.is_empty(), "low-confidence cells must be filtered");
    }
}
