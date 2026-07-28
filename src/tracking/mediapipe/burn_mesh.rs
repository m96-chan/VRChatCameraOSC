//! GPU FaceMesh stage on **burn-wgpu** — ported from AvataCam
//! `backends/mediapipe/burn_mesh.rs` (#62) for issue #17; the graph
//! execution itself now lives in the shared [`crate::tracking::burn_onnx`]
//! mini executor (generalized for the hand landmark net, issue #8). This
//! wrapper owns the FaceMesh-specific parts: the rotated 256×256 crop and
//! the landmark/presence decode.
//!
//! Why the GPU path exists: candle-onnx's per-op interpreter measured ~109
//! ms/frame on CPU for this depthwise-separable-heavy graph — nowhere near
//! the 30 FPS realtime requirement. Output parity with the candle path is
//! pinned by `tests/mediapipe_onnx.rs`.

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::tracking::burn_onnx::BurnOnnx;

use super::mesh::{decode_landmarks, warp_crop, INPUT};
use super::roi::FaceRoi;
use super::{FaceMeshLandmarks, NUM_FACE_LANDMARKS};

pub struct BurnFaceMesh {
    exec: BurnOnnx,
}

impl BurnFaceMesh {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            exec: BurnOnnx::from_path(path)?,
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
        let outputs = self.exec.run(buf, [1, 3, INPUT, INPUT])?;

        // Landmarks = 1434-elem output; presence = a 1-elem output. Mirror
        // the candle CPU stage exactly: keep the most "present-looking"
        // (max) of the raw 1-element logits, then sigmoid; 1.0 if none.
        let mut raw: Option<Vec<f32>> = None;
        let mut presence_raw: Option<f32> = None;
        for (_, data) in outputs {
            if data.len() == NUM_FACE_LANDMARKS * 3 {
                raw = Some(data);
            } else if data.len() == 1 {
                presence_raw = Some(presence_raw.map_or(data[0], |p: f32| p.max(data[0])));
            }
        }
        let raw = raw.ok_or_else(|| anyhow!("burn facemesh: 1434 output missing"))?;
        let presence = presence_raw
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .unwrap_or(1.0);
        Ok(decode_landmarks(&raw, presence, width, height, roi))
    }
}
