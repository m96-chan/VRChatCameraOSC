//! The realtime pipeline: capture → track → map → OSC (issue #8).
//!
//! [`Pipeline`] wires the stages behind their traits so it is agnostic to the
//! concrete camera, tracker, and OSC sink — the app builds it from config,
//! and tests build it from a synthetic camera + stub tracker + monitor sink.
//!
//! Since issue #21 there is a single tracker/mapper stack: an
//! [`ArkitFaceTracker`] (MediaPipe) producing 52 blendshape coefficients +
//! head pose, feeding the VRCFT-compatible [`UnifiedMapper`] (issue #18) —
//! plus the optional native `/tracking/eye/*` mapper (issue #19). Every
//! mapper emits plain [`OscParam`]s, so everything downstream (sink, TUI,
//! monitor mode) is mapper-agnostic.

use crate::capture::CameraSource;
use crate::mapping::arm::ArmMapper;
use crate::mapping::avatar::AvatarWatcher;
use crate::mapping::eye::NativeEyeMapper;
use crate::mapping::gesture::GestureMapper;
use crate::mapping::unified::UnifiedMapper;
use crate::osc::{OscParam, OscSink};
use crate::tracking::arkit::ArkitFaceTracker;
use crate::tracking::hands::HandTracker;
use crate::tracking::pose::PoseTracker;
use anyhow::Result;

/// What one [`Pipeline::step`] produced — for the TUI/monitor to display.
#[derive(Debug, Default)]
pub struct StepOutcome {
    /// `(width, height)` of the frame that was captured.
    pub frame_size: (u32, u32),
    /// OSC parameters produced and sent this frame (empty if no face/tracker).
    pub params: Vec<OscParam>,
}

/// The capture→track→map→OSC pipeline. Camera, tracker, and sink stay trait
/// objects so the concrete backends are swappable; the mapper is the (only)
/// [`UnifiedMapper`].
pub struct Pipeline {
    camera: Box<dyn CameraSource>,
    tracker: Option<Box<dyn ArkitFaceTracker>>,
    mapper: UnifiedMapper,
    /// Optional native `/tracking/eye/*` output appended to every frame's
    /// params (issue #19) — orthogonal to the parameter mapping.
    eye: Option<Box<NativeEyeMapper>>,
    /// Optional hand tracking → gesture ints appended to the frame's params
    /// (issue #8) — runs on the same captured frame, independent of the
    /// face path (gestures still go out when no face is tracked).
    hands: Option<HandsStage>,
    /// Optional pose tracking → arm parameters appended to the frame's
    /// params (issue #28 phase 2) — runs every frame on the same captured
    /// frame, independent of the face and hand paths.
    pose: Option<PoseStage>,
    sink: Box<dyn OscSink>,
    /// Avatar-aware gating (issue #18 phase 3): polled once per step for
    /// `/avatar/change`; feeds the mapper's `set_avatar_config`.
    watcher: Option<AvatarWatcher>,
}

/// The attached hand stage (issue #8): tracker + gesture mapper + cadence.
struct HandsStage {
    tracker: Box<dyn HandTracker>,
    gesture: GestureMapper,
    /// Run the hand stage every `interval`-th frame (1 = every frame).
    interval: u32,
    frame_index: u32,
}

/// The attached pose stage (issue #28 phase 2): tracker + arm mapper. Runs
/// every frame — the landmark net is the burn-wgpu hot path; the slow
/// person detector schedules itself asynchronously inside the tracker.
struct PoseStage {
    tracker: Box<dyn PoseTracker>,
    arms: ArmMapper,
}

impl Pipeline {
    /// Build the pipeline. `tracker: None` (models missing/failed to load)
    /// still runs the loop — it just never emits parameters.
    pub fn new(
        camera: Box<dyn CameraSource>,
        tracker: Option<Box<dyn ArkitFaceTracker>>,
        mapper: UnifiedMapper,
        sink: Box<dyn OscSink>,
    ) -> Self {
        Self {
            camera,
            tracker,
            mapper,
            eye: None,
            hands: None,
            pose: None,
            sink,
            watcher: None,
        }
    }

    /// Attach the hand tracking stage (issue #8): `tracker` runs on the same
    /// captured frame every `interval`-th step (clamped to >= 1), `gesture`
    /// turns its hands into the `VCO_Gesture*` (+ optional native) ints
    /// appended to the frame's parameters.
    pub fn with_hands(
        mut self,
        tracker: Box<dyn HandTracker>,
        gesture: GestureMapper,
        interval: u32,
    ) -> Self {
        self.hands = Some(HandsStage {
            tracker,
            gesture,
            interval: interval.max(1),
            frame_index: 0,
        });
        self
    }

    /// Attach the pose tracking stage (issue #28 phase 2): `tracker` runs on
    /// the same captured frame **every** step; `arms` turns its landmarks
    /// into the `VCO_{L,R}_ArmTracked/ArmUpDown/ArmAcross/Elbow` parameters
    /// appended to the frame's output (all eight are emitted every frame —
    /// an untracked arm reports `false` + decayed-to-0 floats).
    pub fn with_pose(mut self, tracker: Box<dyn PoseTracker>, arms: ArmMapper) -> Self {
        self.pose = Some(PoseStage { tracker, arms });
        self
    }

    /// Attach an [`AvatarWatcher`] (issue #18 phase 3): applies its startup
    /// best-effort avatar config immediately, then polls it once per
    /// [`step`](Self::step) — a single nonblocking socket read; the
    /// filesystem is only touched when the avatar actually changes (see
    /// `mapping::avatar` for the polling-not-thread rationale).
    pub fn with_avatar_watcher(mut self, mut watcher: AvatarWatcher) -> Self {
        if let Some(cfg) = watcher.initial_config() {
            self.mapper.set_avatar_config(Some(&cfg));
        }
        self.watcher = Some(watcher);
        self
    }

    /// Attach the native `/tracking/eye/*` mapper (issue #19).
    pub fn with_native_eye(mut self, eye_mapper: NativeEyeMapper) -> Self {
        self.eye = Some(Box::new(eye_mapper));
        self
    }

    /// Pull one frame through the whole pipeline. Captures a frame, tracks a
    /// face (if a tracker is attached), maps to OSC parameters, and sends
    /// them to the sink.
    pub fn step(&mut self) -> Result<StepOutcome> {
        // Avatar switches first, so this frame already goes out gated to the
        // new avatar's param set.
        if let Some(watcher) = &mut self.watcher {
            if let Some(update) = watcher.poll() {
                self.mapper.set_avatar_config(update.as_ref());
            }
        }

        let frame = self.camera.next_frame()?;
        let mut outcome = StepOutcome {
            frame_size: (frame.width, frame.height),
            ..Default::default()
        };

        let mut params = Vec::new();
        if let Some(tracker) = self.tracker.as_mut() {
            if let Some(face) = tracker.track(&frame)? {
                params = self.mapper.map(&face);
                if let Some(eye) = self.eye.as_mut() {
                    params.extend(eye.map(&face));
                }
            }
        }

        // Hands run on the same frame, independent of the face outcome
        // (issue #8) — gesture ints go out even when no face is tracked.
        if let Some(hands) = self.hands.as_mut() {
            if hands.frame_index.is_multiple_of(hands.interval) {
                let tracked = hands.tracker.track(&frame)?;
                params.extend(hands.gesture.map(&tracked));
            }
            hands.frame_index = hands.frame_index.wrapping_add(1);
        }

        // Pose runs every frame (issue #28), independent of face/hands —
        // the arm parameters go out even when no face is tracked.
        if let Some(pose) = self.pose.as_mut() {
            let tracked = pose.tracker.track(&frame)?;
            params.extend(pose.arms.map(tracked.as_ref()));
        }

        if !params.is_empty() {
            self.sink.send(&params)?;
        }
        outcome.params = params;
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
        let Some(tracker) = self.tracker.as_mut() else {
            return Ok(0);
        };
        let mut samples = Vec::new();
        for _ in 0..frames {
            let frame = self.camera.next_frame()?;
            if let Some(face) = tracker.track(&frame)? {
                samples.push(face);
            }
        }
        self.mapper.calibrate(&samples);
        if let Some(eye) = self.eye.as_mut() {
            eye.calibrate(&samples);
        }
        Ok(samples.len())
    }
}
