//! Native VRChat eye-tracking OSC — `/tracking/eye/*` (issue #19).
//!
//! VRChat accepts eye-tracking data at raw OSC addresses that drive the
//! avatar's **existing eye bones directly** — no avatar parameters, no
//! wizard, no VRCFT support required (verified against
//! <https://docs.vrchat.com/docs/osc-eye-tracking>):
//!
//! - `/tracking/eye/LeftRightPitchYaw` — **one message, four floats**: left
//!   pitch, left yaw, right pitch, right yaw, in **degrees**; positive pitch
//!   looks down, positive yaw looks right (Unity convention).
//! - `/tracking/eye/EyesClosedAmount` — one float `0..1`; VRChat currently
//!   supports only a single value blinking **both** eyes together.
//! - Data times out after 10 s, reverting to VRChat's own eye behaviour — so
//!   losing the face falls back gracefully without any cleanup message.
//!
//! This mapper is orthogonal to the mapping profile (`custom10`/`vrcft`):
//! the pipeline appends its output to whichever profile mapper is active
//! (the 竹 tier of the 松竹梅 design in issue #19). It consumes the same
//! [`ArkitFaceFrame`] as the profile mappers — mediapipe backend only.
//!
//! # Derivation
//!
//! Twelve raw channels (blink L/R, squint L/R, the eight `eyeLook*`
//! directions) get the same neutral-baseline calibration (per-eye blink
//! self-calibration included) and One-Euro smoothing as
//! [`super::arkit`]/[`super::unified`]. Then, matching the VRCFT/LiveLink
//! semantics used by [`super::unified`]:
//!
//! - gaze `x = lookIn − lookOut` (left eye) / `lookOut − lookIn` (right eye),
//!   positive = **subject's right**; `y = lookUp − lookDown`, positive = up
//! - `yaw_deg = x · yaw_range_deg` (positive OSC yaw = avatar's right, which
//!   mirrors the subject's right), `pitch_deg = −y · pitch_range_deg`
//!   (positive OSC pitch = down) — **signs to be confirmed in the live
//!   VRChat demo**, like every other sign convention in this crate
//! - `openness = 1 − clamp(blink + blink·squint, 0, 1)` per eye;
//!   `EyesClosedAmount = 1 − (openness_L + openness_R)/2`

use super::arkit::{deadzone, ArkitMappingConfig, OneEuro};
use crate::osc::OscParam;
use crate::tracking::arkit::ArkitFaceFrame;

/// The raw OSC addresses this mapper emits (also used by tests/docs).
pub const ADDR_GAZE: &str = "/tracking/eye/LeftRightPitchYaw";
pub const ADDR_EYES_CLOSED: &str = "/tracking/eye/EyesClosedAmount";

/// The 12 ARKit channels consumed, as indices into `ArkitBlendshapes.0`
/// (MediaPipe order — pinned to names by a test in [`super::unified`]).
const CH: [usize; NUM_CH] = [
    9,  // eyeBlinkLeft
    10, // eyeBlinkRight
    19, // eyeSquintLeft
    20, // eyeSquintRight
    11, // eyeLookDownLeft
    12, // eyeLookDownRight
    13, // eyeLookInLeft
    14, // eyeLookInRight
    15, // eyeLookOutLeft
    16, // eyeLookOutRight
    17, // eyeLookUpLeft
    18, // eyeLookUpRight
];
const NUM_CH: usize = 12;
// Indices into the *local* 12-channel array above.
const L_BLINK_LEFT: usize = 0;
const L_BLINK_RIGHT: usize = 1;
const L_SQUINT_LEFT: usize = 2;
const L_SQUINT_RIGHT: usize = 3;
const L_LOOK_DOWN_LEFT: usize = 4;
const L_LOOK_DOWN_RIGHT: usize = 5;
const L_LOOK_IN_LEFT: usize = 6;
const L_LOOK_IN_RIGHT: usize = 7;
const L_LOOK_OUT_LEFT: usize = 8;
const L_LOOK_OUT_RIGHT: usize = 9;
const L_LOOK_UP_LEFT: usize = 10;
const L_LOOK_UP_RIGHT: usize = 11;

/// Gaze range tuning: degrees of eye rotation at a saturated (1.0) look
/// coefficient. VRChat wants degrees; ARKit look coefficients are 0..1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeRange {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

impl Default for EyeRange {
    fn default() -> Self {
        // A human eye's comfortable rotation range is roughly ±30° laterally
        // and a bit less vertically; these are display gains, not biometrics,
        // and are config-tunable ([eye] yaw_range_deg / pitch_range_deg).
        Self {
            yaw_deg: 30.0,
            pitch_deg: 25.0,
        }
    }
}

/// Maps [`ArkitFaceFrame`]s to the native `/tracking/eye/*` messages. Holds
/// calibration + smoothing state between frames, mirroring the profile
/// mappers' design (see the module docs).
#[derive(Debug, Clone)]
pub struct NativeEyeMapper {
    config: ArkitMappingConfig,
    range: EyeRange,
    base: [f32; NUM_CH],
    n: u32,
    filters: [OneEuro; NUM_CH],
}

impl Default for NativeEyeMapper {
    fn default() -> Self {
        Self::new(ArkitMappingConfig::default(), EyeRange::default())
    }
}

impl NativeEyeMapper {
    /// A mapper with the given tuning config (calibration window, blink
    /// gain, One-Euro cutoffs and deadzone are shared with the profile
    /// mappers' [`ArkitMappingConfig`]) and gaze range.
    pub fn new(config: ArkitMappingConfig, range: EyeRange) -> Self {
        Self {
            config,
            range,
            base: [0.0; NUM_CH],
            n: 0,
            filters: build_filters(&config),
        }
    }

    /// Produce the two `/tracking/eye/*` messages for one frame.
    pub fn map(&mut self, frame: &ArkitFaceFrame) -> Vec<OscParam> {
        let dt = self.config.assumed_dt.max(1e-4);
        let dz = self.config.deadzone.clamp(0.0, 0.9);
        let gain = self.config.blink_gain.max(1e-4);

        let mut c = [0.0f32; NUM_CH];
        for (i, &src) in CH.iter().enumerate() {
            c[i] = frame.blendshapes.0[src];
        }
        // Accumulate the neutral baseline over the calibration window, then
        // normalise: per-eye blink self-calibration for the blink channels,
        // plain baseline subtraction for the rest (same rule as the profile
        // mappers).
        if self.n < self.config.calib_frames {
            let k = self.n as f32;
            for (b, &ci) in self.base.iter_mut().zip(c.iter()) {
                *b = (*b * k + ci) / (k + 1.0);
            }
            self.n += 1;
        }
        for (i, v) in c.iter_mut().enumerate() {
            if i == L_BLINK_LEFT || i == L_BLINK_RIGHT {
                *v = ((*v - self.base[i]) / gain).clamp(0.0, 1.0);
            } else {
                *v = (*v - self.base[i]).max(0.0);
            }
            *v = deadzone(self.filters[i].filter(*v, dt), dz);
        }

        // Gaze, positive x = subject's right / y = up (module docs).
        let left_x = c[L_LOOK_IN_LEFT] - c[L_LOOK_OUT_LEFT];
        let right_x = c[L_LOOK_OUT_RIGHT] - c[L_LOOK_IN_RIGHT];
        let left_y = c[L_LOOK_UP_LEFT] - c[L_LOOK_DOWN_LEFT];
        let right_y = c[L_LOOK_UP_RIGHT] - c[L_LOOK_DOWN_RIGHT];

        let left_pitch = -left_y.clamp(-1.0, 1.0) * self.range.pitch_deg;
        let left_yaw = left_x.clamp(-1.0, 1.0) * self.range.yaw_deg;
        let right_pitch = -right_y.clamp(-1.0, 1.0) * self.range.pitch_deg;
        let right_yaw = right_x.clamp(-1.0, 1.0) * self.range.yaw_deg;

        // VRCFT/LiveLink openness, then VRChat's single both-eyes blink.
        let openness_l =
            1.0 - (c[L_BLINK_LEFT] + c[L_BLINK_LEFT] * c[L_SQUINT_LEFT]).clamp(0.0, 1.0);
        let openness_r =
            1.0 - (c[L_BLINK_RIGHT] + c[L_BLINK_RIGHT] * c[L_SQUINT_RIGHT]).clamp(0.0, 1.0);
        let eyes_closed = 1.0 - (openness_l + openness_r) / 2.0;

        vec![
            OscParam::raw_floats4(ADDR_GAZE, [left_pitch, left_yaw, right_pitch, right_yaw]),
            OscParam::raw_float(ADDR_EYES_CLOSED, eyes_closed.clamp(0.0, 1.0)),
        ]
    }

    /// Re-derive the neutral baseline from a batch of relaxed forward-facing
    /// frames — same contract as the profile mappers. Resets smoothing.
    pub fn calibrate(&mut self, frames: &[ArkitFaceFrame]) {
        if frames.is_empty() {
            return;
        }
        let n = frames.len() as f32;
        let mut sum = [0.0f32; NUM_CH];
        for f in frames {
            for (s, &src) in sum.iter_mut().zip(CH.iter()) {
                *s += f.blendshapes.0[src];
            }
        }
        for (b, s) in self.base.iter_mut().zip(sum.iter()) {
            *b = s / n;
        }
        self.n = self.config.calib_frames;
        self.filters = build_filters(&self.config);
    }
}

fn build_filters(cfg: &ArkitMappingConfig) -> [OneEuro; NUM_CH] {
    // All eye channels are fast (blinks, saccades) — no slow brow-style lane.
    std::array::from_fn(|_| OneEuro::new(cfg.fast_mincutoff, cfg.expr_beta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::OscValue;
    use crate::tracking::arkit::{
        ArkitBlendshapes, HeadPose, ARKIT_BLENDSHAPE_NAMES, NUM_BLENDSHAPES,
    };

    /// The CH index table must match the intended channel names.
    #[test]
    fn channel_table_matches_names() {
        let names = [
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "eyeSquintLeft",
            "eyeSquintRight",
            "eyeLookDownLeft",
            "eyeLookDownRight",
            "eyeLookInLeft",
            "eyeLookInRight",
            "eyeLookOutLeft",
            "eyeLookOutRight",
            "eyeLookUpLeft",
            "eyeLookUpRight",
        ];
        for (i, name) in names.iter().enumerate() {
            assert_eq!(ARKIT_BLENDSHAPE_NAMES[CH[i]], *name);
        }
    }

    fn frame(values: &[(&str, f32)]) -> ArkitFaceFrame {
        let mut a = [0.0f32; NUM_BLENDSHAPES];
        for &(name, v) in values {
            let idx = ARKIT_BLENDSHAPE_NAMES
                .iter()
                .position(|&n| n == name)
                .unwrap_or_else(|| panic!("unknown blendshape {name}"));
            a[idx] = v;
        }
        ArkitFaceFrame {
            blendshapes: ArkitBlendshapes(a),
            head: HeadPose::default(),
        }
    }

    fn gaze_of(out: &[OscParam]) -> [f32; 4] {
        let p = out.iter().find(|p| p.name == ADDR_GAZE).expect("gaze");
        match p.value {
            OscValue::Floats4(v) => v,
            _ => panic!("gaze must be Floats4"),
        }
    }

    fn closed_of(out: &[OscParam]) -> f32 {
        let p = out
            .iter()
            .find(|p| p.name == ADDR_EYES_CLOSED)
            .expect("closed");
        match p.value {
            OscValue::Float(v) => v,
            _ => panic!("EyesClosedAmount must be Float"),
        }
    }

    fn mapper(calib_frames: u32) -> NativeEyeMapper {
        NativeEyeMapper::new(
            ArkitMappingConfig {
                calib_frames,
                ..Default::default()
            },
            EyeRange::default(),
        )
    }

    // Neutral (with resting baselines) → gaze ~0°, EyesClosedAmount ~0.
    #[test]
    fn neutral_rests_at_zero() {
        let mut m = mapper(10);
        let rest = frame(&[("eyeBlinkLeft", 0.08), ("eyeBlinkRight", 0.12)]);
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&rest);
        }
        for v in gaze_of(&last) {
            assert!(v.abs() < 1.0, "gaze not at rest: {v}");
        }
        assert!(closed_of(&last) < 0.05);
        // Exactly two messages, at the raw addresses (no /avatar/parameters/).
        assert_eq!(last.len(), 2);
        for p in &last {
            assert!(p.address().starts_with("/tracking/eye/"));
        }
    }

    // A blink (both eyes) saturates EyesClosedAmount to ~1.
    #[test]
    fn blink_saturates_eyes_closed() {
        let mut m = mapper(10);
        let rest = frame(&[("eyeBlinkLeft", 0.08), ("eyeBlinkRight", 0.12)]);
        for _ in 0..10 {
            m.map(&rest);
        }
        let shut = frame(&[
            ("eyeBlinkLeft", 0.08 + 0.35),
            ("eyeBlinkRight", 0.12 + 0.35),
        ]);
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&shut);
        }
        assert!((closed_of(&last) - 1.0).abs() < 2e-2);
    }

    // Looking toward the subject's right → positive yaw on both eyes,
    // saturating at yaw_range_deg; pitch stays ~0.
    #[test]
    fn look_right_gives_positive_yaw_in_degrees() {
        let mut m = mapper(5);
        let rest = frame(&[]);
        for _ in 0..5 {
            m.map(&rest);
        }
        let right = frame(&[("eyeLookInLeft", 1.0), ("eyeLookOutRight", 1.0)]);
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&right);
        }
        let [lp, ly, rp, ry] = gaze_of(&last);
        // Deadzone rescale keeps saturated input at full range.
        assert!((ly - 30.0).abs() < 1.0, "left yaw should be ~+30°: {ly}");
        assert!((ry - 30.0).abs() < 1.0, "right yaw should be ~+30°: {ry}");
        assert!(lp.abs() < 1.0 && rp.abs() < 1.0, "pitch should stay ~0");
    }

    // Looking up → negative pitch (VRChat: positive pitch = down).
    #[test]
    fn look_up_gives_negative_pitch() {
        let mut m = mapper(5);
        let rest = frame(&[]);
        for _ in 0..5 {
            m.map(&rest);
        }
        let up = frame(&[("eyeLookUpLeft", 1.0), ("eyeLookUpRight", 1.0)]);
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&up);
        }
        let [lp, _ly, rp, _ry] = gaze_of(&last);
        assert!((lp + 25.0).abs() < 1.0, "left pitch should be ~−25°: {lp}");
        assert!((rp + 25.0).abs() < 1.0, "right pitch should be ~−25°: {rp}");
    }

    // Explicit calibrate() seeds the baseline immediately.
    #[test]
    fn explicit_calibrate_seeds_baseline() {
        let mut m = mapper(30);
        let rest = frame(&[
            ("eyeBlinkLeft", 0.3),
            ("eyeLookUpLeft", 0.4),
            ("eyeLookUpRight", 0.4),
        ]);
        m.calibrate(&vec![rest; 5]);
        let mut last = Vec::new();
        for _ in 0..20 {
            last = m.map(&rest);
        }
        let [lp, _, rp, _] = gaze_of(&last);
        assert!(lp.abs() < 1.0 && rp.abs() < 1.0, "calibrated rest not 0");
        assert!(closed_of(&last) < 0.05);
    }
}
