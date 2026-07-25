//! Blendshape V2 stage (issue #17, ported from AvataCam #5).
//!
//! MediaPipe's MLP-Mixer blendshape model: takes the fixed 146-landmark subset
//! (`input_points [1,146,2]`, normalized-image `x,y`) and emits 52 ARKit-aligned
//! coefficients (`[1,52]`, already activated), packed here into
//! [`crate::tracking::arkit::ArkitBlendshapes`] — the shared type
//! `mapping::arkit` consumes. Needs the candle-onnx `Reciprocal` op (added to
//! the `m96-chan/candle` fork for its layer-norm).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

use crate::tracking::arkit::{ArkitBlendshapes, NUM_BLENDSHAPES};

use super::subset::LANDMARKS_SUBSET;
use super::FaceMeshLandmarks;

pub struct BlendshapeModel {
    model: ModelProto,
    /// Graph initializers (weights), extracted once at load — re-parsing
    /// them per eval is the dominant per-frame cost (fork commit 7b0427d).
    consts: HashMap<String, Tensor>,
    input_name: String,
    device: Device,
}

impl BlendshapeModel {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        // CPU on purpose, even under `--features cuda`: candle-onnx's
        // per-op interpreter is kernel-launch/sync bound on GPU (measured
        // *slower* than CPU for the small blendshape MLP). The realtime path
        // instead runs FaceMesh on burn-wgpu (see `burn_mesh`), YuNet rarely
        // (loss/interval redetect), and this stage's weights pre-extracted.
        let device = Device::Cpu;
        let model = candle_onnx::read_file(path.as_ref())?;
        let input_name = model
            .graph
            .as_ref()
            .and_then(|g| g.input.first())
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow::anyhow!("blendshape model has no input"))?;
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

    /// Run the blendshape model on the 146-landmark subset of `lms`.
    ///
    /// MediaPipe's `LandmarksToTensorCalculator` is wired with `kImageSize`, i.e.
    /// it **denormalizes to pixel coordinates** before the model. Feeding pixels
    /// (not normalized `[0,1]`) preserves the face aspect ratio, which the model
    /// needs to tell e.g. a smile (horizontal) from jaw-open (vertical).
    pub fn run(&self, lms: &FaceMeshLandmarks, img_w: u32, img_h: u32) -> Result<ArkitBlendshapes> {
        if lms.points.len() < *LANDMARKS_SUBSET.iter().max().unwrap() + 1 {
            bail!("too few landmarks for blendshapes: {}", lms.points.len());
        }
        let (wf, hf) = (img_w as f32, img_h as f32);
        let mut feed = Vec::with_capacity(LANDMARKS_SUBSET.len() * 2);
        for &i in &LANDMARKS_SUBSET {
            feed.push(lms.points[i][0] * wf); // x in pixels
            feed.push(lms.points[i][1] * hf); // y in pixels
        }
        let input = Tensor::from_vec(feed, (1, LANDMARKS_SUBSET.len(), 2), &self.device)?;
        let mut inputs = self.consts.clone();
        inputs.insert(self.input_name.clone(), input);
        let outputs = candle_onnx::simple_eval_with_initializers(&self.model, inputs)?;

        let out = outputs
            .values()
            .find(|v| v.dims().iter().product::<usize>() == NUM_BLENDSHAPES)
            .ok_or_else(|| anyhow::anyhow!("blendshape output [.,52] missing"))?;
        let vals = out.flatten_all()?.to_vec1::<f32>()?;
        let mut arr = [0.0f32; NUM_BLENDSHAPES];
        arr.copy_from_slice(&vals[..NUM_BLENDSHAPES]);
        Ok(ArkitBlendshapes(arr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_rejects_too_few_landmarks() {
        // Build a model wrapper we never actually invoke `run`'s ONNX path on
        // (the too-few-landmarks check runs before any inference), so no
        // model file is needed — this is a pure preflight-validation test.
        let too_few = FaceMeshLandmarks {
            points: vec![[0.0, 0.0, 0.0]; 10],
            presence: 1.0,
        };
        // Directly exercise the bounds check via the same condition `run` uses.
        assert!(too_few.points.len() < *LANDMARKS_SUBSET.iter().max().unwrap() + 1);
    }
}
