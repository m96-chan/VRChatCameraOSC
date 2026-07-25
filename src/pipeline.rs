//! The realtime pipeline: capture → track → map → OSC (issue #8).
//!
//! [`Pipeline`] wires the four stages behind their traits so it is agnostic to
//! the concrete camera, tracker, and OSC sink — the app builds it from config,
//! and tests build it from a synthetic camera + stub tracker + monitor sink.

use crate::capture::CameraSource;
use crate::mapping::Mapper;
use crate::osc::{OscParam, OscSink};
use crate::tracking::{FaceLandmarks, FaceTracker};
use anyhow::Result;

/// What one [`Pipeline::step`] produced — for the TUI/monitor to display.
#[derive(Debug, Default)]
pub struct StepOutcome {
    /// `(width, height)` of the frame that was captured.
    pub frame_size: (u32, u32),
    /// Landmarks detected this frame, if a tracker is attached and found a face.
    pub landmarks: Option<FaceLandmarks>,
    /// OSC parameters produced and sent this frame (empty if no face/tracker).
    pub params: Vec<OscParam>,
}

/// The capture→track→map→OSC pipeline. Each stage is a trait object so the
/// concrete backends stay swappable.
pub struct Pipeline {
    camera: Box<dyn CameraSource>,
    tracker: Option<Box<dyn FaceTracker>>,
    mapper: Mapper,
    sink: Box<dyn OscSink>,
}

impl Pipeline {
    pub fn new(
        camera: Box<dyn CameraSource>,
        tracker: Option<Box<dyn FaceTracker>>,
        mapper: Mapper,
        sink: Box<dyn OscSink>,
    ) -> Self {
        Self {
            camera,
            tracker,
            mapper,
            sink,
        }
    }

    /// Pull one frame through the whole pipeline. Captures a frame, tracks a
    /// face (if a tracker is attached), maps landmarks to OSC parameters, and
    /// sends them to the sink.
    pub fn step(&mut self) -> Result<StepOutcome> {
        let frame = self.camera.next_frame()?;
        let frame_size = (frame.width, frame.height);

        let mut outcome = StepOutcome {
            frame_size,
            ..Default::default()
        };

        if let Some(tracker) = self.tracker.as_mut() {
            if let Some(landmarks) = tracker.track(&frame)? {
                let params = self.mapper.map(&landmarks);
                self.sink.send(&params)?;
                outcome.params = params;
                outcome.landmarks = Some(landmarks);
            }
        }
        Ok(outcome)
    }

    /// Access the camera's current resolution.
    pub fn resolution(&self) -> (u32, u32) {
        self.camera.resolution()
    }
}
