//! FaceMesh V2 landmark stage (issue #17, ported from AvataCam #5).
//!
//! Warps the rotated face ROI into an upright 256x256 crop and runs MediaPipe's
//! FaceMesh V2 ONNX (`1x3x256x256`, **RGB, [0,1], NCHW**), decoding 478 3D
//! landmarks back into normalized coordinates of the original frame (`x,y in
//! [0,1]`, y-down convention).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

use super::roi::FaceRoi;
use super::util::{sample_rgb255, sigmoid};
use super::{FaceMeshLandmarks, NUM_FACE_LANDMARKS};

const INPUT: usize = 256;

pub struct FaceMesh {
    model: ModelProto,
    input_name: String,
    device: Device,
}

impl FaceMesh {
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
            .ok_or_else(|| anyhow::anyhow!("facemesh model has no input"))?;
        Ok(Self {
            model,
            input_name,
            device,
        })
    }

    pub fn run(
        &self,
        width: u32,
        height: u32,
        rgb: &[u8],
        roi: &FaceRoi,
    ) -> Result<FaceMeshLandmarks> {
        let buf = warp_crop(width, height, rgb, roi)?;
        let input = Tensor::from_vec(buf, (1, 3, INPUT, INPUT), &self.device)?;
        let mut inputs = HashMap::new();
        inputs.insert(self.input_name.clone(), input);
        let outputs = candle_onnx::simple_eval(&self.model, inputs)?;

        // Landmarks = the 478x3 = 1434-element output; presence = a 1-element
        // output (sigmoid). Identify by element count (names vary per export).
        let mut landmarks: Option<Vec<f32>> = None;
        let mut presence_raw: Option<f32> = None;
        for v in outputs.values() {
            let n: usize = v.dims().iter().product();
            if n == NUM_FACE_LANDMARKS * 3 {
                landmarks = Some(v.flatten_all()?.to_vec1::<f32>()?);
            } else if n == 1 {
                let val = v.flatten_all()?.to_vec1::<f32>()?[0];
                // Keep the most "present-looking" of the 1-element outputs.
                presence_raw = Some(presence_raw.map_or(val, |p| p.max(val)));
            }
        }
        let raw = landmarks
            .ok_or_else(|| anyhow::anyhow!("facemesh landmark output [.,1434] missing"))?;
        let presence = presence_raw.map(sigmoid).unwrap_or(1.0);

        Ok(decode_landmarks(&raw, presence, width, height, roi))
    }
}

/// Bilinear-warp the rotated ROI into a 256x256 RGB, `[0,1]`, **NCHW** buffer.
fn warp_crop(width: u32, height: u32, rgb: &[u8], roi: &FaceRoi) -> Result<Vec<f32>> {
    let (w, h) = (width as i32, height as i32);
    if w == 0 || h == 0 || rgb.len() < (w * h * 3) as usize {
        bail!("invalid frame for facemesh warp");
    }
    let plane = INPUT * INPUT;
    let mut buf = vec![0f32; 3 * plane]; // NCHW
    for oy in 0..INPUT {
        let ly = 0.5 - (oy as f32 + 0.5) / INPUT as f32;
        for ox in 0..INPUT {
            let lx = (ox as f32 + 0.5) / INPUT as f32 - 0.5;
            let sx = roi.cx + lx * roi.size * roi.right[0] + ly * roi.size * roi.up[0];
            let sy = roi.cy + lx * roi.size * roi.right[1] + ly * roi.size * roi.up[1];
            let [r, g, b] = sample_rgb255(width, height, rgb, sx, sy);
            let o = oy * INPUT + ox;
            buf[o] = r / 255.0; // R plane
            buf[plane + o] = g / 255.0; // G plane
            buf[2 * plane + o] = b / 255.0; // B plane
        }
    }
    Ok(buf)
}

/// Decode raw `478x3` landmarks (+ presence) into frame-normalized
/// [`FaceMeshLandmarks`].
fn decode_landmarks(
    raw: &[f32],
    presence: f32,
    width: u32,
    height: u32,
    roi: &FaceRoi,
) -> FaceMeshLandmarks {
    let (wf, hf) = (width as f32, height as f32);
    // Some exports emit crop pixels (0..256), others already-normalized (0..1).
    let max_xy = raw
        .chunks_exact(3)
        .map(|p| p[0].abs().max(p[1].abs()))
        .fold(0.0f32, f32::max);
    let div = if max_xy > 2.0 { INPUT as f32 } else { 1.0 };

    let mut points = Vec::with_capacity(NUM_FACE_LANDMARKS);
    for p in raw.chunks_exact(3) {
        let lx = p[0] / div - 0.5;
        let ly = 0.5 - p[1] / div;
        let sx = roi.cx + lx * roi.size * roi.right[0] + ly * roi.size * roi.up[0];
        let sy = roi.cy + lx * roi.size * roi.right[1] + ly * roi.size * roi.up[1];
        let z = (p[2] / div) * roi.size / wf;
        points.push([sx / wf, sy / hf, z]);
    }
    FaceMeshLandmarks { points, presence }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_roi(w: u32, h: u32) -> FaceRoi {
        FaceRoi {
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
            size: w.min(h) as f32,
            right: [1.0, 0.0],
            up: [0.0, -1.0],
            corners: [[0.0, 0.0]; 4],
        }
    }

    #[test]
    fn decode_landmarks_normalizes_pixel_space_output() {
        // Two "landmarks" (only the first 6 floats matter), model output in
        // crop-pixel space (>2.0 => pixel-space branch).
        let raw = vec![128.0, 128.0, 0.0, 0.0, 0.0, 0.0];
        let roi = identity_roi(100, 100);
        let lms = decode_landmarks(&raw, 0.9, 100, 100, &roi);
        assert_eq!(lms.presence, 0.9);
        // First point is the crop center -> maps back to the ROI center (50,50)
        // normalized -> (0.5, 0.5).
        assert!((lms.points[0][0] - 0.5).abs() < 1e-3);
        assert!((lms.points[0][1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn decode_landmarks_accepts_prenormalized_output() {
        // Values <= 2.0 => already-normalized branch (div = 1.0).
        let raw = vec![0.5, 0.5, 0.0];
        let roi = identity_roi(100, 100);
        let lms = decode_landmarks(&raw, 1.0, 100, 100, &roi);
        assert!((lms.points[0][0] - 0.5).abs() < 1e-3);
        assert!((lms.points[0][1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn warp_crop_rejects_undersized_buffer() {
        let roi = identity_roi(10, 10);
        let err = warp_crop(10, 10, &[0u8; 4], &roi).unwrap_err();
        assert!(err.to_string().contains("invalid frame"));
    }
}
