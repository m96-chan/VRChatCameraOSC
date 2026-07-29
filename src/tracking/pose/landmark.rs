//! Pose landmark stage (issue #28 phase 2): the OpenCV Zoo
//! `pose_estimation_mediapipe` ONNX (256×256 NHWC, RGB `[0,1]`) — 39
//! BlazePose landmarks (33 body + 6 auxiliary ROI points) with visibility,
//! from an upright rotated person crop.
//!
//! This is the per-frame hot path (once per tracked person), so it runs on
//! **burn-wgpu** (the shared `tracking::burn_onnx` executor, ~8 ms/eval)
//! when the default `mesh-gpu` feature is on and a GPU is usable, falling
//! back to the candle-onnx CPU interpreter exactly like the FaceMesh and
//! hand-landmark stages. Cross-backend parity is pinned by
//! `pose_estimation_burn_matches_candle` in `tests/mediapipe_onnx.rs`.
//!
//! Pre/post-processing ports OpenCV Zoo
//! `models/pose_estimation_mediapipe/mp_pose.py` (Apache-2.0): the rotated
//! crop is resized to 256×256 and normalized `/255`; the screen landmarks
//! come back in crop pixels as 39×(x, y, z, visibility, presence) — z is
//! hip-relative at the same scale, visibility/presence are **raw logits**
//! (their line `landmarks[:, 3:] = sigmoid(...)`), and the crop-pixel
//! coordinates map back through the ROI (`(lm − 128)·size/256`, de-rotated,
//! plus the crop center — our warp/decode expresses the identical affine
//! map, like the hand stage before it). One deviation: Zoo pads its crop
//! with black at the frame edge, we clamp to the edge pixel (the face
//! stack's convention; `sample_rgb255`).
//!
//! Output names are fixed by the TFLite→ONNX export (order confirmed
//! against Zoo's unpack `landmarks, conf, mask, heatmap, landmarks_word`):
//! `Identity` = screen landmarks `[1,195]`, `Identity_1` = pose presence
//! confidence `[1,1]` (**already sigmoid** — Zoo compares it to 0.5 raw),
//! `Identity_2` = segmentation mask `[1,256,256,1]` (unused),
//! `Identity_3` = heatmap `[1,64,64,39]` (unused), `Identity_4` = world
//! landmarks `[1,117]` (unused).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

use crate::tracking::mediapipe::util::{sample_rgb255, sigmoid};

use super::roi::PoseRoi;
use super::NUM_RAW_POSE_LANDMARKS;

/// Model input side (256×256).
pub const INPUT: usize = 256;

/// Output-tensor names in the OpenCV Zoo export (see module docs).
const OUT_LANDMARKS: &str = "Identity";
const OUT_CONFIDENCE: &str = "Identity_1";

/// One raw landmark-stage result: 39 3-D points (33 body + 6 aux) in
/// normalized frame coordinates (`x,y in [0,1]`, y down; z hip-relative,
/// frame-width units — the FaceMesh/hands convention), each point's
/// sigmoid visibility, and the model's pose-presence confidence.
///
/// Public (unlike the hand stage's raw type) so the model-gated decode
/// test and the `pose_image` example can drive the stage directly.
#[derive(Debug, Clone)]
pub struct RawPoseLandmarks {
    pub points: [[f32; 3]; NUM_RAW_POSE_LANDMARKS],
    pub visibility: [f32; NUM_RAW_POSE_LANDMARKS],
    pub confidence: f32,
}

/// The landmark stage: burn-wgpu when available, candle CPU otherwise
/// (mirrors the FaceMesh/hand-landmark stage selection, issue #17).
// One long-lived instance per tracker; the variants' size gap is irrelevant.
#[allow(clippy::large_enum_variant)]
enum Stage {
    Cpu(CpuPoseLandmark),
    #[cfg(feature = "mesh-gpu")]
    Gpu(crate::tracking::burn_onnx::BurnOnnx),
}

pub struct PoseLandmarker {
    stage: Stage,
}

impl PoseLandmarker {
    /// Load the model, preferring the burn-wgpu executor (feature-gated) and
    /// falling back to the candle CPU path on any failure — including a
    /// panic out of wgpu adapter init on truly headless machines.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            stage: Self::build_stage(path)?,
        })
    }

    #[cfg(feature = "mesh-gpu")]
    fn build_stage(path: impl AsRef<Path>) -> Result<Stage> {
        let path = path.as_ref();
        let gpu = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::tracking::burn_onnx::BurnOnnx::from_path(path)
        }));
        match gpu {
            Ok(Ok(exec)) => Ok(Stage::Gpu(exec)),
            Ok(Err(e)) => {
                eprintln!("pose: burn-wgpu landmark stage unavailable ({e:#}) — using CPU");
                Ok(Stage::Cpu(CpuPoseLandmark::from_path(path)?))
            }
            Err(_) => {
                eprintln!("pose: burn-wgpu init panicked (no GPU/driver?) — using CPU");
                Ok(Stage::Cpu(CpuPoseLandmark::from_path(path)?))
            }
        }
    }

    #[cfg(not(feature = "mesh-gpu"))]
    fn build_stage(path: impl AsRef<Path>) -> Result<Stage> {
        Ok(Stage::Cpu(CpuPoseLandmark::from_path(path)?))
    }

    /// Human-readable stage backend name (for startup logging).
    pub fn name(&self) -> &'static str {
        match self.stage {
            Stage::Cpu(_) => "candle-cpu",
            #[cfg(feature = "mesh-gpu")]
            Stage::Gpu(_) => "burn-wgpu",
        }
    }

    /// Run the pose landmark net on the person ROI of a packed-RGB8 frame.
    pub fn run(
        &self,
        width: u32,
        height: u32,
        rgb: &[u8],
        roi: &PoseRoi,
    ) -> Result<RawPoseLandmarks> {
        let buf = warp_crop_nhwc(width, height, rgb, roi)?;
        let (raw, confidence) = match &self.stage {
            Stage::Cpu(m) => m.eval(buf)?,
            #[cfg(feature = "mesh-gpu")]
            Stage::Gpu(exec) => {
                let outputs = exec.run(buf, [1, INPUT, INPUT, 3])?;
                let get = |name: &str| -> Result<&Vec<f32>> {
                    outputs
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, d)| d)
                        .ok_or_else(|| anyhow!("pose landmark output {name} missing"))
                };
                let lms = get(OUT_LANDMARKS)?.clone();
                let conf = get(OUT_CONFIDENCE)?[0];
                (lms, conf)
            }
        };
        if raw.len() != NUM_RAW_POSE_LANDMARKS * 5 {
            bail!(
                "pose landmark output has {} values, expected {}",
                raw.len(),
                NUM_RAW_POSE_LANDMARKS * 5
            );
        }
        Ok(decode_pose_landmarks(&raw, confidence, width, height, roi))
    }
}

/// Candle-onnx CPU fallback path.
struct CpuPoseLandmark {
    model: ModelProto,
    /// Graph initializers (weights), extracted once at load — re-parsing
    /// them per eval is the dominant per-frame cost (fork commit 7b0427d).
    consts: HashMap<String, Tensor>,
    input_name: String,
    device: Device,
}

impl CpuPoseLandmark {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let device = Device::Cpu;
        let model = candle_onnx::read_file(path.as_ref())?;
        let input_name = model
            .graph
            .as_ref()
            .and_then(|g| g.input.first())
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow!("pose landmark model has no input"))?;
        let consts = candle_onnx::initializer_tensors(&model)?
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_device(&device)?)))
            .collect::<candle_core::Result<HashMap<_, _>>>()?;
        Ok(Self {
            model,
            consts,
            input_name,
            device,
        })
    }

    /// Evaluate one NHWC crop; returns (raw 39×5 landmarks, confidence).
    fn eval(&self, buf: Vec<f32>) -> Result<(Vec<f32>, f32)> {
        let input = Tensor::from_vec(buf, (1, INPUT, INPUT, 3), &self.device)?;
        let mut inputs = self.consts.clone();
        inputs.insert(self.input_name.clone(), input);
        let outputs = candle_onnx::simple_eval_with_initializers(&self.model, inputs)?;
        let get = |name: &str| -> Result<Vec<f32>> {
            outputs
                .get(name)
                .ok_or_else(|| anyhow!("pose landmark output {name} missing"))?
                .flatten_all()?
                .to_vec1::<f32>()
                .map_err(Into::into)
        };
        let lms = get(OUT_LANDMARKS)?;
        let conf = get(OUT_CONFIDENCE)?[0];
        Ok((lms, conf))
    }
}

/// Bilinear-warp the rotated person ROI into a 256×256 RGB `[0,1]` **NHWC**
/// buffer (same construction as the hand stage's crop, 256 instead of 224).
pub(super) fn warp_crop_nhwc(
    width: u32,
    height: u32,
    rgb: &[u8],
    roi: &PoseRoi,
) -> Result<Vec<f32>> {
    let (w, h) = (width as i32, height as i32);
    if w == 0 || h == 0 || rgb.len() < (w * h * 3) as usize {
        bail!("invalid frame for pose landmark warp");
    }
    let mut buf = vec![0f32; INPUT * INPUT * 3];
    for oy in 0..INPUT {
        let ly = 0.5 - (oy as f32 + 0.5) / INPUT as f32;
        for ox in 0..INPUT {
            let lx = (ox as f32 + 0.5) / INPUT as f32 - 0.5;
            let sx = roi.cx + lx * roi.size * roi.right[0] + ly * roi.size * roi.up[0];
            let sy = roi.cy + lx * roi.size * roi.right[1] + ly * roi.size * roi.up[1];
            let [r, g, b] = sample_rgb255(width, height, rgb, sx, sy);
            let o = (oy * INPUT + ox) * 3;
            buf[o] = r / 255.0;
            buf[o + 1] = g / 255.0;
            buf[o + 2] = b / 255.0;
        }
    }
    Ok(buf)
}

/// Decode raw `39×5` crop-pixel landmarks back into normalized frame
/// coordinates through the ROI — the same affine map as the hand/FaceMesh
/// decode (the exact inverse of Zoo's `(lm − 128)·size/256` + de-rotate +
/// center). Visibility logits get their sigmoid here (`mp_pose.py` line
/// `landmarks[:, 3:] = 1 / (1 + exp(-...))`); presence is dropped
/// (visibility is the stricter of the two — "in frame *and* not occluded").
pub(super) fn decode_pose_landmarks(
    raw: &[f32],
    confidence: f32,
    width: u32,
    height: u32,
    roi: &PoseRoi,
) -> RawPoseLandmarks {
    let (wf, hf) = (width as f32, height as f32);
    let mut points = [[0.0f32; 3]; NUM_RAW_POSE_LANDMARKS];
    let mut visibility = [0.0f32; NUM_RAW_POSE_LANDMARKS];
    for (i, p) in raw.chunks_exact(5).take(NUM_RAW_POSE_LANDMARKS).enumerate() {
        let div = INPUT as f32;
        let lx = p[0] / div - 0.5;
        let ly = 0.5 - p[1] / div;
        let sx = roi.cx + lx * roi.size * roi.right[0] + ly * roi.size * roi.up[0];
        let sy = roi.cy + lx * roi.size * roi.right[1] + ly * roi.size * roi.up[1];
        let z = (p[2] / div) * roi.size / wf;
        points[i] = [sx / wf, sy / hf, z];
        visibility[i] = sigmoid(p[3]);
    }
    RawPoseLandmarks {
        points,
        visibility,
        confidence: confidence.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_roi(w: u32, h: u32) -> PoseRoi {
        PoseRoi {
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
            size: w.min(h) as f32,
            right: [1.0, 0.0],
            up: [0.0, -1.0],
        }
    }

    #[test]
    fn decode_maps_crop_center_to_roi_center_and_sigmoids_visibility() {
        // Landmark 0 at the crop center (128, 128) with a +2 visibility
        // logit; the rest at the origin with a large negative logit.
        let mut raw = vec![0.0f32; NUM_RAW_POSE_LANDMARKS * 5];
        for p in raw.chunks_exact_mut(5) {
            p[3] = -10.0;
        }
        raw[0] = 128.0;
        raw[1] = 128.0;
        raw[3] = 2.0;
        let roi = identity_roi(100, 100);
        let out = decode_pose_landmarks(&raw, 0.9, 100, 100, &roi);
        assert!((out.points[0][0] - 0.5).abs() < 1e-3);
        assert!((out.points[0][1] - 0.5).abs() < 1e-3);
        assert!((out.visibility[0] - sigmoid(2.0)).abs() < 1e-6);
        assert!(out.visibility[1] < 1e-3, "raw logit must be sigmoided");
        assert!((out.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn decode_respects_roi_rotation() {
        // ROI rotated 90°: crop-right is image-down. A landmark right of the
        // crop center must land below the ROI center in the frame.
        let roi = PoseRoi {
            cx: 50.0,
            cy: 50.0,
            size: 40.0,
            right: [0.0, 1.0],
            up: [1.0, 0.0],
        };
        let mut raw = vec![0.0f32; NUM_RAW_POSE_LANDMARKS * 5];
        raw[0] = 256.0; // right edge of the crop
        raw[1] = 128.0; // vertical center
        let out = decode_pose_landmarks(&raw, 1.0, 100, 100, &roi);
        assert!((out.points[0][0] - 0.5).abs() < 1e-3, "{:?}", out.points[0]);
        assert!(
            (out.points[0][1] - 0.7).abs() < 1e-3, // 50 + 0.5*40 = 70 px
            "{:?}",
            out.points[0]
        );
    }

    #[test]
    fn decode_clamps_confidence() {
        let raw = vec![0.0f32; NUM_RAW_POSE_LANDMARKS * 5];
        let roi = identity_roi(10, 10);
        assert_eq!(
            decode_pose_landmarks(&raw, 1.7, 10, 10, &roi).confidence,
            1.0
        );
        assert_eq!(
            decode_pose_landmarks(&raw, -0.3, 10, 10, &roi).confidence,
            0.0
        );
    }

    #[test]
    fn warp_crop_nhwc_is_pixel_interleaved() {
        // A uniform red frame: every NHWC triple must read (1, 0, 0).
        let (w, h) = (8u32, 8u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for px in rgb.chunks_exact_mut(3) {
            px[0] = 255;
        }
        let buf = warp_crop_nhwc(w, h, &rgb, &identity_roi(w, h)).unwrap();
        assert_eq!(buf.len(), INPUT * INPUT * 3);
        for triple in buf.chunks_exact(3) {
            assert_eq!(triple, [1.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn warp_crop_nhwc_rejects_undersized_buffer() {
        let roi = identity_roi(10, 10);
        let err = warp_crop_nhwc(10, 10, &[0u8; 4], &roi).unwrap_err();
        assert!(err.to_string().contains("invalid frame"));
    }
}
