//! Wrist height → arm raise/lower floats (issue #28 phase 1).
//!
//! The owner request behind this: "指だけは動いてる。腕があがったりしない" —
//! raising a hand into the webcam frame should also *raise the avatar's
//! arm*, not just pose the fingers. The #8 hand stack already reports the
//! wrist (landmark 0) in normalized frame coordinates every frame, so phase
//! 1 maps its **height** to one float per arm:
//!
//! ```text
//! /avatar/parameters/VCO_L_ArmUpDown   float -1..1
//! /avatar/parameters/VCO_R_ArmUpDown   float -1..1
//! ```
//!
//! `+1` = arm straight up (wrist above the head), `+0.5` ≈ wrist at face
//! height, `0` ≈ horizontal-ish forward-neutral, `-1` = hanging down. When
//! the hand is **not tracked** the value decays to **exactly 0.0** within a
//! few frames — the `unity/` wizard gates an empty passthrough state near 0
//! so VRChat's idle animation resumes (parameter contract, issues #8/#28).
//!
//! # Reference frame (documented assumption)
//!
//! The face pipeline does not currently expose the face box/center through
//! the tracker trait (only blendshapes + head pose reach the mapper), and
//! plumbing it across `pipeline::Pipeline` into the hand stage would touch
//! every stack for a value that phase 1 only needs as a *rough* anchor. So
//! the mapping normalizes by **frame height** and assumes typical seated
//! webcam framing: face center around [`FACE_CENTER_Y`] of the frame, top
//! of the head around [`HEAD_TOP_Y`]. Both are plain constants documented
//! below precisely so the expected live sign/gain tuning round (issue #28:
//! "expect another live sign-flip round") has obvious knobs; swap them for
//! real face-pipeline anchors if the assumption proves too coarse.
//!
//! # Mapping
//!
//! Monotone linear in wrist `y` (normalized, y down), clamped to `[-1, 1]`:
//!
//! ```text
//! value(y) = 0.5 + (FACE_CENTER_Y - y) * ARM_GAIN
//! ```
//!
//! anchored so `y = FACE_CENTER_Y` → `+0.5` and `y = HEAD_TOP_Y` → `+1.0`;
//! with the defaults the value reaches `-1` around `y ≈ 0.84` — a wrist
//! held low in frame reads "hanging". A wrist inside the bottom margin
//! ([`BOTTOM_MARGIN_Y`]) is about to leave the frame (and half-cropped
//! landmarks are unreliable), so it is treated as untracked → decay to 0.
//!
//! Output is One-Euro smoothed ([`ARM_MINCUTOFF`]/[`ARM_BETA`], the head-pose
//! filter class — an arm is a big slow limb like the head, not a fast
//! expression channel). On hand loss the filter is dropped and the last
//! value decays geometrically ([`DECAY_FACTOR`] per hand-stage frame),
//! snapping to exactly `0.0` below [`SNAP_EPSILON`] (≈6 frames from ±1).

use crate::mapping::arkit::OneEuro;
use crate::osc::OscParam;
use crate::tracking::hands::{HandFrame, WRIST};

/// Assumed face-center height in the frame (normalized y, y down, typical
/// seated webcam framing). Wrist here reads `+0.5`. Tuning knob.
pub const FACE_CENTER_Y: f32 = 0.30;
/// Assumed top-of-head height. Wrist here (or above) reads `+1.0`. Tuning
/// knob.
pub const HEAD_TOP_Y: f32 = 0.12;
/// Slope of the linear map, derived from the two anchors above:
/// `0.5 / (FACE_CENTER_Y - HEAD_TOP_Y)` ≈ 2.78 per frame-height.
pub const ARM_GAIN: f32 = 0.5 / (FACE_CENTER_Y - HEAD_TOP_Y);
/// Wrist below this is about to exit the frame bottom (half-cropped hand,
/// unreliable landmarks) → treated as untracked, value decays to 0.
pub const BOTTOM_MARGIN_Y: f32 = 0.97;
/// Geometric per-frame decay of the emitted value while the hand is
/// untracked.
pub const DECAY_FACTOR: f32 = 0.5;
/// Decayed values below this snap to exactly `0.0` (the contract requires
/// exact zero so the Unity passthrough gate opens). From ±1 this takes 6
/// frames (~200 ms at 30 Hz).
pub const SNAP_EPSILON: f32 = 0.02;
/// One-Euro min-cutoff — the head-pose class (AvataCam
/// `AVATACAM_HEAD_MINCUTOFF`, same value as
/// `ArkitMappingConfig::default().head_mincutoff`).
pub const ARM_MINCUTOFF: f32 = 1.2;
/// One-Euro beta — head-pose class (`head_beta`).
pub const ARM_BETA: f32 = 0.6;
/// Assumed seconds-per-frame (no per-frame timestamps — same simplification
/// as `mapping::arkit`).
const ASSUMED_DT: f32 = 1.0 / 30.0;

/// The raw (unsmoothed) monotone wrist-height → ArmUpDown map. See the
/// module docs for the anchors.
pub fn arm_up_down_raw(wrist_y: f32) -> f32 {
    (0.5 + (FACE_CENTER_Y - wrist_y) * ARM_GAIN).clamp(-1.0, 1.0)
}

/// One arm's state: the smoothing filter while tracked, the decaying value
/// while not.
#[derive(Debug, Clone, Copy)]
struct SideArm {
    filter: OneEuro,
    value: f32,
}

impl Default for SideArm {
    fn default() -> Self {
        Self {
            filter: OneEuro::new(ARM_MINCUTOFF, ARM_BETA),
            value: 0.0,
        }
    }
}

impl SideArm {
    fn step(&mut self, wrist_y: Option<f32>) -> f32 {
        match wrist_y {
            Some(y) if y < BOTTOM_MARGIN_Y => {
                self.value = self.filter.filter(arm_up_down_raw(y), ASSUMED_DT);
            }
            _ => {
                // Untracked (or leaving through the bottom): decay to exact
                // 0 and drop the filter history, so a reacquired hand
                // starts fresh (One-Euro's first sample passes through —
                // a deliberate snap-to-position, not a lerp from stale
                // several-seconds-old state).
                self.filter = OneEuro::new(ARM_MINCUTOFF, ARM_BETA);
                self.value *= DECAY_FACTOR;
                if self.value.abs() < SNAP_EPSILON {
                    self.value = 0.0;
                }
            }
        }
        self.value
    }
}

/// Side-assigned hands → `VCO_{L,R}_ArmUpDown`. Left/right assignment
/// (including `mirror`) is the caller's job — in practice
/// [`GestureMapper`](super::gesture::GestureMapper), so arms and gesture
/// ints always agree on which hand is "left".
#[derive(Debug, Clone, Default)]
pub struct ArmMapper {
    left: SideArm,
    right: SideArm,
}

impl ArmMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map one frame's side-assigned hands. Always emits both parameters —
    /// a decayed-out arm keeps reporting exactly `0.0` (harmless, and the
    /// monitor/TUI stay truthful about what is on the wire).
    pub fn map(&mut self, left: Option<&HandFrame>, right: Option<&HandFrame>) -> Vec<OscParam> {
        let ly = left.map(|h| h.points[WRIST][1]);
        let ry = right.map(|h| h.points[WRIST][1]);
        vec![
            OscParam::float("VCO_L_ArmUpDown", self.left.step(ly)),
            OscParam::float("VCO_R_ArmUpDown", self.right.step(ry)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::OscValue;
    use crate::tracking::hands::{Handedness, NUM_HAND_LANDMARKS};

    fn hand_with_wrist_y(y: f32) -> HandFrame {
        let mut points = [[0.5, 0.5, 0.0]; NUM_HAND_LANDMARKS];
        points[WRIST] = [0.5, y, 0.0];
        HandFrame {
            handedness: Handedness::Right,
            confidence: 0.9,
            points,
        }
    }

    fn float_value(params: &[OscParam], name: &str) -> f32 {
        let p = params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} missing"));
        match p.value {
            OscValue::Float(v) => v,
            ref other => panic!("{name} is not a float: {other:?}"),
        }
    }

    // --- raw map ---------------------------------------------------------

    #[test]
    fn raw_map_hits_the_documented_anchors() {
        assert!((arm_up_down_raw(FACE_CENTER_Y) - 0.5).abs() < 1e-6, "face");
        assert!((arm_up_down_raw(HEAD_TOP_Y) - 1.0).abs() < 1e-6, "head top");
        assert_eq!(arm_up_down_raw(0.0), 1.0, "above head clamps to +1");
        assert_eq!(arm_up_down_raw(1.0), -1.0, "frame bottom clamps to -1");
    }

    #[test]
    fn raw_map_is_monotone_decreasing_in_y() {
        let mut prev = arm_up_down_raw(0.0);
        for i in 1..=20 {
            let v = arm_up_down_raw(i as f32 / 20.0);
            assert!(v <= prev, "not monotone at y={}", i as f32 / 20.0);
            prev = v;
        }
    }

    // --- decay-to-zero contract ------------------------------------------

    #[test]
    fn untracked_arm_decays_to_exactly_zero_within_a_few_frames() {
        let mut m = ArmMapper::new();
        // Raise the (physical-side pre-assigned) left arm above the head.
        let high = hand_with_wrist_y(HEAD_TOP_Y);
        for _ in 0..10 {
            m.map(Some(&high), None);
        }
        let v = float_value(&m.map(Some(&high), None), "VCO_L_ArmUpDown");
        assert!(v > 0.9, "raised arm ~+1, got {v}");

        // Hand gone: exact 0.0 within a few frames (contract: the Unity
        // passthrough gate needs exact zero).
        let mut frames = 0;
        loop {
            let v = float_value(&m.map(None, None), "VCO_L_ArmUpDown");
            frames += 1;
            if v == 0.0 {
                break;
            }
            assert!(frames < 10, "still {v} after {frames} frames");
        }
        assert!(frames <= 7, "took {frames} frames");
        // And stays pinned at exact zero.
        let v = float_value(&m.map(None, None), "VCO_L_ArmUpDown");
        assert_eq!(v, 0.0);
    }

    #[test]
    fn wrist_in_the_bottom_margin_counts_as_untracked() {
        let mut m = ArmMapper::new();
        let low = hand_with_wrist_y(0.99);
        for _ in 0..10 {
            m.map(Some(&low), None);
        }
        let v = float_value(&m.map(Some(&low), None), "VCO_L_ArmUpDown");
        assert_eq!(v, 0.0, "bottom-margin wrist must decay to 0, got {v}");
    }

    // --- smoothing & emission --------------------------------------------

    #[test]
    fn both_params_are_always_emitted_and_sides_are_independent() {
        let mut m = ArmMapper::new();
        let params = m.map(None, None);
        assert_eq!(params.len(), 2);
        assert_eq!(float_value(&params, "VCO_L_ArmUpDown"), 0.0);
        assert_eq!(float_value(&params, "VCO_R_ArmUpDown"), 0.0);

        let high = hand_with_wrist_y(HEAD_TOP_Y);
        let params = m.map(None, Some(&high));
        assert_eq!(float_value(&params, "VCO_L_ArmUpDown"), 0.0, "left idle");
        assert!(
            float_value(&params, "VCO_R_ArmUpDown") > 0.9,
            "right raised"
        );
    }

    #[test]
    fn tracked_motion_is_smoothed_not_snapped() {
        let mut m = ArmMapper::new();
        let face_high = hand_with_wrist_y(FACE_CENTER_Y);
        // Settle at +0.5 (first sample passes through the One-Euro).
        for _ in 0..30 {
            m.map(Some(&face_high), None);
        }
        // Jump the wrist to the head top (+1.0 raw): the smoothed output
        // moves toward it but does not cover the step in one frame.
        let top = hand_with_wrist_y(HEAD_TOP_Y);
        let v = float_value(&m.map(Some(&top), None), "VCO_L_ArmUpDown");
        assert!(v > 0.5, "moves toward the target, got {v}");
        assert!(v < 0.95, "one frame must not complete the step, got {v}");
    }
}
