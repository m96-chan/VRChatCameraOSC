//! FAN (Face Alignment Network) — pure-Rust inference via candle.
//!
//! Ported from `1adrianb/face-alignment` (`models/fan.py`). The default 2D
//! model is `2DFAN4` (`num_modules = 4`): input `1x3x256x256`, output four
//! stages of `1x68x64x64` heatmaps; the last stage is decoded to landmarks.
//!
//! Layer names mirror the PyTorch module names exactly so weights exported from
//! the reference load 1:1 (see `reference/gen_fixtures.py`). Numeric parity
//! against the PyTorch reference is enforced by `tests/fan_parity.rs`.

pub mod decode;
pub mod model;
pub mod preprocess;

use crate::capture::Frame;
use crate::tracking::{FaceLandmarks, FaceTracker, Landmark, NUM_LANDMARKS};
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;

pub use model::Fan;

/// FAN heatmap resolution (64x64).
pub const HEATMAP_RES: u32 = 64;

/// A [`FaceTracker`] backed by the FAN network.
///
/// Detection is not yet wired (a follow-up will add an SFD/BlazeFace detector);
/// [`FanTracker::track`] currently treats the **whole frame** as the face
/// region — correct for a face-fills-the-frame webcam setup — by resizing the
/// frame to the network's 256x256 input. Landmarks are returned in frame pixels.
pub struct FanTracker {
    model: Fan,
    device: Device,
}

impl FanTracker {
    /// Load a tracker from a safetensors weights file (exported by the reference
    /// harness). Runs on CPU.
    pub fn from_safetensors(path: impl AsRef<std::path::Path>) -> CandleResult<Self> {
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.as_ref()], candle_core::DType::F32, &device)?
        };
        let model = Fan::load(vb)?;
        Ok(Self { model, device })
    }

    /// Construct from an already-built model (used in tests).
    pub fn new(model: Fan, device: Device) -> Self {
        Self { model, device }
    }

    /// Run the network on a preprocessed `1x3x256x256` tensor and return the
    /// final heatmap stage (`1x68x64x64`).
    pub fn forward_heatmaps(&self, input: &Tensor) -> CandleResult<Tensor> {
        self.model.forward(input)
    }

    /// Run FAN on a 256x256 RGB crop and decode landmarks in **crop pixel**
    /// coordinates (0..256).
    pub fn landmarks_from_crop(&self, crop: &RgbImage) -> CandleResult<FaceLandmarks> {
        let input = preprocess::to_input_tensor(crop, &self.device)?;
        let hm = self.model.forward(&input)?;
        let preds = decode::get_preds_from_hm(&hm)?;
        let scale = preprocess::INPUT_RES as f32 / HEATMAP_RES as f32; // 256/64 = 4
        let points: Vec<Landmark> = preds[0]
            .iter()
            .map(|p| Landmark {
                x: p.x * scale,
                y: p.y * scale,
                score: p.score,
            })
            .collect();
        FaceLandmarks::new(points)
            .map_err(|e| candle_core::Error::Msg(format!("landmark decode: {e}")))
    }
}

impl FaceTracker for FanTracker {
    fn track(&mut self, frame: &Frame) -> anyhow::Result<Option<FaceLandmarks>> {
        // Whole-frame-as-face: resize to the 256x256 network input.
        let img = RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| anyhow::anyhow!("frame buffer did not match dimensions"))?;
        let resized = image::imageops::resize(
            &img,
            preprocess::INPUT_RES,
            preprocess::INPUT_RES,
            image::imageops::FilterType::Triangle,
        );
        let lms = self.landmarks_from_crop(&resized)?;

        // Map crop-space (0..256) landmarks back to frame pixels.
        let sx = frame.width as f32 / preprocess::INPUT_RES as f32;
        let sy = frame.height as f32 / preprocess::INPUT_RES as f32;
        let points: Vec<Landmark> = lms
            .points
            .into_iter()
            .map(|p| Landmark {
                x: p.x * sx,
                y: p.y * sy,
                score: p.score,
            })
            .collect();
        debug_assert_eq!(points.len(), NUM_LANDMARKS);
        Ok(Some(FaceLandmarks::new(points)?))
    }
}
