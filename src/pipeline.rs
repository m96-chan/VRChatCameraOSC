//! The realtime pipeline: capture → track → map → OSC (issue #8).
//!
//! [`Pipeline`] wires the four stages behind their traits so it is agnostic to
//! the concrete camera, tracker, and OSC sink — the app builds it from config,
//! and tests build it from a synthetic camera + stub tracker + monitor sink.
//!
//! Two tracker/mapper stacks exist behind one pipeline type (issue #17):
//!
//! - **Landmarks** ([`Pipeline::new`]): a 68-point [`FaceTracker`] (FAN) feeding
//!   the geometric [`Mapper`].
//! - **ARKit** ([`Pipeline::new_arkit`]): an [`ArkitFaceTracker`] (MediaPipe)
//!   producing 52 blendshape coefficients + head pose, feeding any
//!   [`ArkitOscMapper`] — the 10-parameter `ArkitMapper` or the
//!   VRCFT-compatible `UnifiedMapper` (issue #18).
//!
//! Every mapper emits plain [`OscParam`]s, so everything downstream (sink,
//! TUI, monitor mode) is stack- and profile-agnostic.

use crate::capture::CameraSource;
use crate::mapping::ArkitOscMapper;
use crate::mapping::Mapper;
use crate::osc::{OscParam, OscSink};
use crate::tracking::arkit::ArkitFaceTracker;
use crate::tracking::{FaceLandmarks, FaceTracker};
use anyhow::Result;

/// What one [`Pipeline::step`] produced — for the TUI/monitor to display.
#[derive(Debug, Default)]
pub struct StepOutcome {
    /// `(width, height)` of the frame that was captured.
    pub frame_size: (u32, u32),
    /// Landmarks detected this frame, if the landmarks stack is attached and
    /// found a face (`None` on the ARKit stack — it has no 68-point output).
    pub landmarks: Option<FaceLandmarks>,
    /// OSC parameters produced and sent this frame (empty if no face/tracker).
    pub params: Vec<OscParam>,
}

/// One of the two tracker→mapper stacks (see the module docs).
enum Stack {
    Landmarks {
        tracker: Option<Box<dyn FaceTracker>>,
        mapper: Mapper,
    },
    Arkit {
        tracker: Option<Box<dyn ArkitFaceTracker>>,
        /// Trait object: either mapping profile (10-param `ArkitMapper` or
        /// VRCFT `UnifiedMapper`), chosen by config/CLI (issue #18).
        mapper: Box<dyn ArkitOscMapper>,
    },
}

/// The capture→track→map→OSC pipeline. Each stage is a trait object so the
/// concrete backends stay swappable.
pub struct Pipeline {
    camera: Box<dyn CameraSource>,
    stack: Stack,
    sink: Box<dyn OscSink>,
}

impl Pipeline {
    /// Build the 68-landmark (FAN) stack.
    pub fn new(
        camera: Box<dyn CameraSource>,
        tracker: Option<Box<dyn FaceTracker>>,
        mapper: Mapper,
        sink: Box<dyn OscSink>,
    ) -> Self {
        Self {
            camera,
            stack: Stack::Landmarks { tracker, mapper },
            sink,
        }
    }

    /// Build the ARKit-blendshape (MediaPipe) stack — issue #17. The mapper
    /// is any [`ArkitOscMapper`] (mapping profile — issue #18).
    pub fn new_arkit(
        camera: Box<dyn CameraSource>,
        tracker: Option<Box<dyn ArkitFaceTracker>>,
        mapper: Box<dyn ArkitOscMapper>,
        sink: Box<dyn OscSink>,
    ) -> Self {
        Self {
            camera,
            stack: Stack::Arkit { tracker, mapper },
            sink,
        }
    }

    /// Pull one frame through the whole pipeline. Captures a frame, tracks a
    /// face (if a tracker is attached), maps to OSC parameters, and sends
    /// them to the sink.
    pub fn step(&mut self) -> Result<StepOutcome> {
        let frame = self.camera.next_frame()?;
        let frame_size = (frame.width, frame.height);

        let mut outcome = StepOutcome {
            frame_size,
            ..Default::default()
        };

        match &mut self.stack {
            Stack::Landmarks { tracker, mapper } => {
                if let Some(tracker) = tracker.as_mut() {
                    if let Some(landmarks) = tracker.track(&frame)? {
                        let params = mapper.map(&landmarks);
                        self.sink.send(&params)?;
                        outcome.params = params;
                        outcome.landmarks = Some(landmarks);
                    }
                }
            }
            Stack::Arkit { tracker, mapper } => {
                if let Some(tracker) = tracker.as_mut() {
                    if let Some(face) = tracker.track(&frame)? {
                        let params = mapper.map(&face);
                        self.sink.send(&params)?;
                        outcome.params = params;
                    }
                }
            }
        }
        Ok(outcome)
    }

    /// Access the camera's current resolution.
    pub fn resolution(&self) -> (u32, u32) {
        self.camera.resolution()
    }

    /// Capture and track `frames` iterations to calibrate the mapper's
    /// neutral-pose baselines (issue #15), assuming the subject holds a
    /// relaxed, forward-facing expression throughout. Frames where no face is
    /// tracked are skipped. Sends no OSC — this runs before normal operation.
    ///
    /// Returns the number of usable samples fed to the mapper's calibration
    /// (0 if no tracker is attached, or no frame ever found a face — the
    /// caller decides how to report that; the mapper's prior config, whatever
    /// it was, is left untouched in that case).
    pub fn calibrate_neutral(&mut self, frames: u32) -> Result<usize> {
        match &mut self.stack {
            Stack::Landmarks { tracker, mapper } => {
                let Some(tracker) = tracker.as_mut() else {
                    return Ok(0);
                };
                let mut samples = Vec::new();
                for _ in 0..frames {
                    let frame = self.camera.next_frame()?;
                    if let Some(landmarks) = tracker.track(&frame)? {
                        samples.push(landmarks);
                    }
                }
                mapper.calibrate(&samples);
                Ok(samples.len())
            }
            Stack::Arkit { tracker, mapper } => {
                let Some(tracker) = tracker.as_mut() else {
                    return Ok(0);
                };
                let mut samples = Vec::new();
                for _ in 0..frames {
                    let frame = self.camera.next_frame()?;
                    if let Some(face) = tracker.track(&frame)? {
                        samples.push(face);
                    }
                }
                mapper.calibrate(&samples);
                Ok(samples.len())
            }
        }
    }
}
