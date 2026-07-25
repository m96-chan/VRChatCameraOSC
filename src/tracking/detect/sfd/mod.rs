//! S3FD detector — pure-Rust (candle) port of the `face_alignment` SFD backend.
//!
//! Pipeline (matching `sfd/detect.py`): the full-resolution RGB image is
//! converted to BGR, the per-channel means `[104, 117, 123]` are subtracted, the
//! network runs at native resolution, and the 12 heads are decoded to boxes,
//! NMS-filtered (0.3) and score-thresholded (0.5).

pub mod decode;
pub mod model;

use super::{BoundingBox, FaceDetector};
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;
pub use model::S3fd;

/// BGR channel means subtracted before the network (reference constant).
const BGR_MEAN: [f32; 3] = [104.0, 117.0, 123.0];
/// Default score threshold applied after NMS.
pub const FILTER_THRESHOLD: f32 = 0.5;

/// The S3FD face detector.
pub struct SfdDetector {
    model: S3fd,
    device: Device,
    filter_threshold: f32,
}

impl SfdDetector {
    /// Load detector weights from a safetensors file (exported by the reference
    /// harness). Runs on CPU.
    pub fn from_safetensors(path: impl AsRef<std::path::Path>) -> CandleResult<Self> {
        let device = Device::Cpu;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], DType::F32, &device)? };
        let model = S3fd::load(vb)?;
        Ok(Self {
            model,
            device,
            filter_threshold: FILTER_THRESHOLD,
        })
    }

    pub fn new(model: S3fd, device: Device) -> Self {
        Self {
            model,
            device,
            filter_threshold: FILTER_THRESHOLD,
        }
    }

    /// Pack an RGB image into the S3FD input tensor: `1x3xHxW`, BGR order, with
    /// the per-channel mean subtracted (no scaling).
    pub fn to_input_tensor(&self, image: &RgbImage) -> CandleResult<Tensor> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let mut chw = vec![0f32; 3 * h * w];
        for (x, y, px) in image.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            let base = y * w + x;
            // BGR channel order.
            chw[base] = px[2] as f32 - BGR_MEAN[0]; // B
            chw[h * w + base] = px[1] as f32 - BGR_MEAN[1]; // G
            chw[2 * h * w + base] = px[0] as f32 - BGR_MEAN[2]; // R
        }
        Tensor::from_vec(chw, (1, 3, h, w), &self.device)
    }

    /// Raw forward pass, returning the 12 head tensors (for parity tests).
    pub fn forward_heads(&self, input: &Tensor) -> CandleResult<Vec<Tensor>> {
        self.model.forward(input)
    }
}

impl FaceDetector for SfdDetector {
    fn detect(&self, image: &RgbImage) -> anyhow::Result<Vec<BoundingBox>> {
        let input = self.to_input_tensor(image)?;
        let heads = self.model.forward(&input)?;
        let boxes = decode::decode_boxes(&heads)?;
        Ok(decode::filter_boxes(boxes, self.filter_threshold))
    }
}
