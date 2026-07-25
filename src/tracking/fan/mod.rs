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
use crate::tracking::detect::{BoundingBox, FaceDetector};
use crate::tracking::{FaceLandmarks, FaceTracker, Landmark, NUM_LANDMARKS};
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;

pub use model::Fan;

/// FAN heatmap resolution (64x64).
pub const HEATMAP_RES: u32 = 64;
/// Face-box centre Y offset, as a fraction of box height (reference constant).
pub const CENTER_Y_OFFSET: f32 = 0.12;
/// Detector reference scale used to turn a box into a FAN crop scale.
pub const REFERENCE_SCALE: f32 = 195.0;

/// Convert a detected face box to the FAN `(center, scale)`, exactly as
/// `face_alignment` does: centre is the box centre nudged up by
/// `CENTER_Y_OFFSET * height`, and `scale = (w + h) / REFERENCE_SCALE`.
pub fn box_to_center_scale(b: &BoundingBox) -> ((f32, f32), f32) {
    let cx = (b.x1 + b.x2) / 2.0;
    let cy = (b.y1 + b.y2) / 2.0 - (b.y2 - b.y1) * CENTER_Y_OFFSET;
    let scale = (b.x2 - b.x1 + (b.y2 - b.y1)) / REFERENCE_SCALE;
    ((cx, cy), scale)
}

/// A [`FaceTracker`] backed by the FAN network.
///
/// With a detector attached, [`FanTracker::track`] detects the face, crops
/// around it (reference `crop`), runs FAN, and maps landmarks back to image
/// pixels. Without one, it falls back to treating the whole frame as the face
/// (resize to 256) — fine when the face fills the frame.
pub struct FanTracker {
    model: Fan,
    device: Device,
    detector: Option<Box<dyn FaceDetector>>,
}

impl FanTracker {
    /// Load a tracker from a safetensors weights file. No detector (whole-frame).
    pub fn from_safetensors(path: impl AsRef<std::path::Path>) -> CandleResult<Self> {
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.as_ref()], candle_core::DType::F32, &device)?
        };
        let model = Fan::load(vb)?;
        Ok(Self {
            model,
            device,
            detector: None,
        })
    }

    /// Construct from an already-built model (used in tests).
    pub fn new(model: Fan, device: Device) -> Self {
        Self {
            model,
            device,
            detector: None,
        }
    }

    /// Attach a face detector so `track` auto-crops around the detected face.
    pub fn with_detector(mut self, detector: Box<dyn FaceDetector>) -> Self {
        self.detector = Some(detector);
        self
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

    /// Detect-free path: crop around a known face box, run FAN, and return
    /// landmarks in **image pixel** coordinates (mirrors the reference:
    /// `crop` → FAN → `get_preds_fromhm(center, scale)`).
    pub fn landmarks_in_image(
        &self,
        image: &RgbImage,
        bbox: &BoundingBox,
    ) -> CandleResult<FaceLandmarks> {
        let (center, scale) = box_to_center_scale(bbox);
        let crop = preprocess::crop(image, center, scale, preprocess::INPUT_RES);
        let input = preprocess::to_input_tensor(&crop, &self.device)?;
        let hm = self.model.forward(&input)?;
        let preds = decode::get_preds_from_hm(&hm)?;
        let points: Vec<Landmark> = preds[0]
            .iter()
            .map(|p| {
                let (ix, iy) =
                    preprocess::transform((p.x, p.y), center, scale, HEATMAP_RES as f32, true);
                Landmark {
                    x: ix as f32,
                    y: iy as f32,
                    score: p.score,
                }
            })
            .collect();
        FaceLandmarks::new(points)
            .map_err(|e| candle_core::Error::Msg(format!("landmark decode: {e}")))
    }
}

impl FaceTracker for FanTracker {
    fn track(&mut self, frame: &Frame) -> anyhow::Result<Option<FaceLandmarks>> {
        let img = RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| anyhow::anyhow!("frame buffer did not match dimensions"))?;

        // With a detector: detect, take the highest-scoring face, crop around it.
        if let Some(detector) = self.detector.as_ref() {
            let boxes = detector.detect(&img)?;
            return match boxes.iter().max_by(|a, b| a.score.total_cmp(&b.score)) {
                Some(best) => Ok(Some(self.landmarks_in_image(&img, best)?)),
                None => Ok(None),
            };
        }

        // Whole-frame fallback: resize to the 256x256 network input.
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
