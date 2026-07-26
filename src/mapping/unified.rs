//! VRCFaceTracking-compatible OSC output — Unified Expressions `v2/` float
//! parameters (issue #18).
//!
//! Emits the same `/avatar/parameters/<prefix>v2/<Name>` float parameters that
//! **VRCFaceTracking** (VRCFT) sends, so avatars already set up for VRCFT /
//! Unified Expressions work with **zero avatar-side setup**. This mapper is
//! the "option B" design recorded in issue #18: the app acts as a VRCFT
//! replacement rather than a VRCFT (C#/.NET) tracking module.
//!
//! # Sources (CLAUDE.md rule 3 — ported from upstream GitHub, not memory)
//!
//! - **ARKit-52 → UE shape correlation:** the official
//!   [`LiveLinkTrackingModule`](https://github.com/VRCFaceTracking/LiveLinkTrackingModule)
//!   (`LiveLinkExtTrackingInterface.cs`) — iPhone ARKit input is exactly our
//!   MediaPipe Blendshape V2 coefficient set. Includes the eye-openness
//!   formula `1 − clamp(blink + blink·squint, 0, 1)` and the
//!   `LipSuckUpper* = min(1 − upperUp^(1/6), mouthRollUpper)` special case.
//! - **Float parameter names & combined formulas:**
//!   [`UnifiedExpressionsParameters.cs`](https://github.com/benaclejames/VRCFaceTracking/blob/master/VRCFaceTracking.Core/Params/Expressions/UnifiedExpressionsParameters.cs),
//!   `UnifiedSimpleExpressions.cs` (BrowUp/BrowDown/MouthSmile/MouthSad
//!   weighted blends) and `UnifiedHeadParameters.cs` (`v2/Head/*`).
//! - **Status parameters:** `EyeTrackingActive` / `ExpressionTrackingActive` /
//!   `LipTrackingActive` bools (no `v2/` prefix), which FT avatars use to gate
//!   their face-tracking animator layers.
//!
//! # Pipeline
//!
//! ```text
//! ArkitFaceFrame (52 coeffs + head pose)
//!   → per-channel neutral-baseline calibration      (UeCalib — all 51 real
//!     channels; per-eye blink self-calibration)      channels, not 11)
//!   → One-Euro smoothing + output deadzone per channel
//!   → UE shape weights (LiveLink correlation)
//!   → float parameter table (VRCFT combined formulas we can drive)
//!   → OscParam floats under each configured prefix + 3 status bools
//! ```
//!
//! Calibration, smoothing, and deadzone reuse the [`super::arkit`] design
//! (AvataCam-ported `BsCalib`/`OneEuro`) — the difference is scope: this
//! mapper calibrates **every** real channel because nearly all 52 reach an
//! output here, where [`super::arkit::ArkitMapper`] only needs 11.
//!
//! # Avatar-aware gating (issue #18 phases 2+3) vs. blind prefixes
//!
//! VRCFT discovers the avatar's parameter addresses (arbitrary user prefix,
//! `FT/` by common template convention) from the avatar's OSC config JSON and
//! sends only those. When an avatar config is loaded
//! ([`UnifiedMapper::set_avatar_config`] — fed by
//! [`super::avatar::AvatarWatcher`] via the pipeline), this mapper does the
//! same: each conceptual value is resolved through
//! [`super::avatar::AvatarParamSet`] and emitted **only** in the forms the
//! avatar declares — float, plain bool, and/or **binary bit params**
//! (`<Name>1/2/4/8...` + `<Name>Negative`) — at the exact declared
//! addresses.
//!
//! Without a config (none discovered, or an avatar we have no JSON for),
//! the phase-1 fallback applies: the same value is sent as floats under
//! **each configured prefix** — default `""` and `"FT/"`, i.e. `v2/JawOpen`
//! *and* `FT/v2/JawOpen`. VRChat ignores addresses the avatar doesn't
//! declare, so the extra copies are harmless.
//!
//! # Not sent (no source data in ARKit-52)
//!
//! Pupil dilation/constriction, `NasalDilation/Constrict`, all tongue shapes
//! (MediaPipe Blendshape V2 has no `tongueOut`), and eye-gaze *shape* params
//! (UE keeps gaze in the eye struct, not shapes). Shapes whose UE definition
//! exists but has no ARKit source (`CheekSuck*`, `JawBackward`,
//! `MouthTightener*`) are sent as constant `0`, matching VRCFT running a
//! LiveLink module.
//!
//! # Conventions to verify in the live demo (like `super::arkit`'s head signs)
//!
//! - **Gaze:** `v2/Eye*X` positive = looking toward the **subject's right**
//!   (left eye `lookIn − lookOut`, right eye `lookOut − lookIn`), `v2/Eye*Y`
//!   positive = up.
//! - **Head:** `v2/Head/{Yaw,Pitch,Roll}` reuse the exact normalization and
//!   sign conventions of [`super::arkit::ArkitMapper`]'s
//!   `HeadYaw`/`HeadPitch`/`HeadRoll` (roll negated, yaw/pitch as-is).

use super::arkit::{deadzone, ArkitMappingConfig, HeadCalib, OneEuro};
use super::avatar::{AvatarOscConfig, AvatarParamSet};
use crate::osc::OscParam;
use crate::tracking::arkit::{ArkitFaceFrame, NUM_BLENDSHAPES};

// Indices into `ArkitBlendshapes.0` / `ARKIT_BLENDSHAPE_NAMES` (fixed
// MediaPipe Blendshape V2 order; a test below pins each one to its name).
const CH_BROW_DOWN_LEFT: usize = 1;
const CH_BROW_DOWN_RIGHT: usize = 2;
const CH_BROW_INNER_UP: usize = 3;
const CH_BROW_OUTER_UP_LEFT: usize = 4;
const CH_BROW_OUTER_UP_RIGHT: usize = 5;
const CH_CHEEK_PUFF: usize = 6;
const CH_CHEEK_SQUINT_LEFT: usize = 7;
const CH_CHEEK_SQUINT_RIGHT: usize = 8;
const CH_EYE_BLINK_LEFT: usize = 9;
const CH_EYE_BLINK_RIGHT: usize = 10;
const CH_EYE_LOOK_DOWN_LEFT: usize = 11;
const CH_EYE_LOOK_DOWN_RIGHT: usize = 12;
const CH_EYE_LOOK_IN_LEFT: usize = 13;
const CH_EYE_LOOK_IN_RIGHT: usize = 14;
const CH_EYE_LOOK_OUT_LEFT: usize = 15;
const CH_EYE_LOOK_OUT_RIGHT: usize = 16;
const CH_EYE_LOOK_UP_LEFT: usize = 17;
const CH_EYE_LOOK_UP_RIGHT: usize = 18;
const CH_EYE_SQUINT_LEFT: usize = 19;
const CH_EYE_SQUINT_RIGHT: usize = 20;
const CH_EYE_WIDE_LEFT: usize = 21;
const CH_EYE_WIDE_RIGHT: usize = 22;
const CH_JAW_FORWARD: usize = 23;
const CH_JAW_LEFT: usize = 24;
const CH_JAW_OPEN: usize = 25;
const CH_JAW_RIGHT: usize = 26;
const CH_MOUTH_CLOSE: usize = 27;
const CH_MOUTH_DIMPLE_LEFT: usize = 28;
const CH_MOUTH_DIMPLE_RIGHT: usize = 29;
const CH_MOUTH_FROWN_LEFT: usize = 30;
const CH_MOUTH_FROWN_RIGHT: usize = 31;
const CH_MOUTH_FUNNEL: usize = 32;
const CH_MOUTH_LEFT: usize = 33;
const CH_MOUTH_LOWER_DOWN_LEFT: usize = 34;
const CH_MOUTH_LOWER_DOWN_RIGHT: usize = 35;
const CH_MOUTH_PRESS_LEFT: usize = 36;
const CH_MOUTH_PRESS_RIGHT: usize = 37;
const CH_MOUTH_PUCKER: usize = 38;
const CH_MOUTH_RIGHT: usize = 39;
const CH_MOUTH_ROLL_LOWER: usize = 40;
const CH_MOUTH_ROLL_UPPER: usize = 41;
const CH_MOUTH_SHRUG_LOWER: usize = 42;
const CH_MOUTH_SHRUG_UPPER: usize = 43;
const CH_MOUTH_SMILE_LEFT: usize = 44;
const CH_MOUTH_SMILE_RIGHT: usize = 45;
const CH_MOUTH_STRETCH_LEFT: usize = 46;
const CH_MOUTH_STRETCH_RIGHT: usize = 47;
const CH_MOUTH_UPPER_UP_LEFT: usize = 48;
const CH_MOUTH_UPPER_UP_RIGHT: usize = 49;
const CH_NOSE_SNEER_LEFT: usize = 50;
const CH_NOSE_SNEER_RIGHT: usize = 51;

/// Default address prefixes — send both the bare `v2/...` form and the `FT/`
/// template convention (see the module docs).
pub const DEFAULT_PREFIXES: [&str; 2] = ["", "FT/"];

/// The status bools FT avatars gate their animator layers on (VRCFT
/// `ConditionalBoolParameter` names — no `v2/` prefix).
pub const STATUS_PARAM_NAMES: [&str; 3] = [
    "EyeTrackingActive",
    "ExpressionTrackingActive",
    "LipTrackingActive",
];

/// Every canonical float name this mapper can emit (`v2/...` incl. head) —
/// the name universe [`AvatarParamSet`] resolution gates against.
pub fn float_param_names() -> Vec<&'static str> {
    param_values(&[0.0; NUM_BLENDSHAPES], 0.0, 0.0, 0.0)
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// Whether channel `idx` gets the snappier `fast_mincutoff` — mirrors
/// [`super::arkit`]: only the (slow-moving) brow channels use the general
/// expression cutoff; blink/gaze/jaw/mouth/cheek stay snappy.
fn is_fast_channel(idx: usize) -> bool {
    !matches!(
        idx,
        CH_BROW_DOWN_LEFT
            | CH_BROW_DOWN_RIGHT
            | CH_BROW_INNER_UP
            | CH_BROW_OUTER_UP_LEFT
            | CH_BROW_OUTER_UP_RIGHT
    )
}

/// Per-channel neutral-baseline calibration over the full 52-channel array —
/// the full-width counterpart of [`super::arkit`]'s 11-channel `BsCalib`,
/// same accumulate-and-normalise rule, same per-eye blink self-calibration
/// (`(raw − open baseline) / blink_gain`, clamped `0..1`).
#[derive(Debug, Clone, Copy)]
struct UeCalib {
    base: [f32; NUM_BLENDSHAPES],
    n: u32,
    frames: u32,
    blink_gain: f32,
}

impl UeCalib {
    fn new(frames: u32, blink_gain: f32) -> Self {
        Self {
            base: [0.0; NUM_BLENDSHAPES],
            n: 0,
            frames,
            blink_gain: blink_gain.max(1e-4),
        }
    }

    fn seed(&mut self, avg: [f32; NUM_BLENDSHAPES]) {
        self.base = avg;
        self.n = self.frames;
    }

    fn apply(&mut self, c: &mut [f32; NUM_BLENDSHAPES]) {
        if self.n < self.frames {
            let k = self.n as f32;
            for (b, &ci) in self.base.iter_mut().zip(c.iter()) {
                *b = (*b * k + ci) / (k + 1.0);
            }
            self.n += 1;
        }
        for (i, v) in c.iter_mut().enumerate() {
            if i == CH_EYE_BLINK_LEFT || i == CH_EYE_BLINK_RIGHT {
                *v = ((*v - self.base[i]) / self.blink_gain).clamp(0.0, 1.0);
            } else {
                *v = (*v - self.base[i]).max(0.0);
            }
        }
    }
}

fn build_channel_filters(cfg: &ArkitMappingConfig) -> [OneEuro; NUM_BLENDSHAPES] {
    std::array::from_fn(|i| {
        let mc = if is_fast_channel(i) {
            cfg.fast_mincutoff
        } else {
            cfg.expr_mincutoff
        };
        OneEuro::new(mc, cfg.expr_beta)
    })
}

fn build_head_filters(cfg: &ArkitMappingConfig) -> [OneEuro; 3] {
    std::array::from_fn(|_| OneEuro::new(cfg.head_mincutoff, cfg.head_beta))
}

/// Maps [`ArkitFaceFrame`]s to VRCFT-compatible Unified Expressions OSC
/// parameters. Holds calibration + smoothing state between frames; see the
/// module docs for the full derivation and conventions.
#[derive(Debug, Clone)]
pub struct UnifiedMapper {
    config: ArkitMappingConfig,
    prefixes: Vec<String>,
    calib: UeCalib,
    head_calib: HeadCalib,
    filters: [OneEuro; NUM_BLENDSHAPES],
    head_filters: [OneEuro; 3],
    /// The worn avatar's declared params (issue #18 phase 3). `Some` gates
    /// output to exactly the declared forms/addresses; `None` falls back to
    /// the blind-prefix float behavior.
    avatar: Option<AvatarParamSet>,
}

impl Default for UnifiedMapper {
    fn default() -> Self {
        Self::new(ArkitMappingConfig::default(), Vec::new())
    }
}

impl UnifiedMapper {
    /// A mapper with the given tuning config and address prefixes. An empty
    /// `prefixes` falls back to [`DEFAULT_PREFIXES`]. Neutral baselines
    /// auto-accumulate over the first `config.calib_frames` `map` calls
    /// unless [`calibrate`](Self::calibrate) is called first.
    pub fn new(config: ArkitMappingConfig, prefixes: Vec<String>) -> Self {
        let prefixes = if prefixes.is_empty() {
            DEFAULT_PREFIXES.iter().map(|s| s.to_string()).collect()
        } else {
            prefixes
        };
        Self {
            config,
            prefixes,
            calib: UeCalib::new(config.calib_frames, config.blink_gain),
            head_calib: HeadCalib::new(config.calib_frames),
            filters: build_channel_filters(&config),
            head_filters: build_head_filters(&config),
            avatar: None,
        }
    }

    /// Adopt (`Some`) or clear (`None`) an avatar's OSC config: resolves the
    /// declared parameters against this mapper's name universe
    /// ([`float_param_names`] + [`STATUS_PARAM_NAMES`]) once, so `map` only
    /// does hash lookups. See the module docs (avatar-aware gating).
    pub fn set_avatar_config(&mut self, config: Option<&AvatarOscConfig>) {
        self.avatar =
            config.map(|c| AvatarParamSet::resolve(c, &float_param_names(), &STATUS_PARAM_NAMES));
    }

    /// The currently active avatar param set, if any (for status display).
    pub fn avatar_params(&self) -> Option<&AvatarParamSet> {
        self.avatar.as_ref()
    }

    /// Produce the VRCFT-compatible parameter updates for one frame.
    pub fn map(&mut self, frame: &ArkitFaceFrame) -> Vec<OscParam> {
        let dt = self.config.assumed_dt.max(1e-4);
        let dz = self.config.deadzone.clamp(0.0, 0.9);

        let mut c = frame.blendshapes.0;
        self.calib.apply(&mut c);
        for (i, v) in c.iter_mut().enumerate() {
            *v = deadzone(self.filters[i].filter(*v, dt), dz);
        }

        let neutral = self.head_calib.accumulate(&frame.head);
        let rel_roll = self.head_filters[0].filter(frame.head.roll - neutral[0], dt);
        let rel_yaw = self.head_filters[1].filter(frame.head.yaw - neutral[1], dt);
        let rel_pitch = self.head_filters[2].filter(frame.head.pitch - neutral[2], dt);
        // Same normalization + sign conventions as `super::arkit` (module docs).
        let head_yaw = deadzone(rel_yaw / self.config.yaw_max_rad, dz);
        let head_pitch = deadzone(rel_pitch / self.config.pitch_max_rad, dz);
        let head_roll = deadzone(-(rel_roll / self.config.roll_max_rad), dz);

        let vals = param_values(&c, head_yaw, head_pitch, head_roll);

        // Avatar-aware mode: emit only what the worn avatar declares, in the
        // declared forms (float / bool / binary bits), at the exact declared
        // addresses. The three status bools read true whenever a face is
        // tracked, same as the fallback path below.
        if let Some(set) = &self.avatar {
            let mut out = Vec::new();
            for (name, v) in &vals {
                set.emit_value(name, v.clamp(-1.0, 1.0), &mut out);
            }
            for name in STATUS_PARAM_NAMES {
                set.emit_status(name, true, &mut out);
            }
            return out;
        }

        let mut out = Vec::with_capacity((vals.len() + 3) * self.prefixes.len());
        for prefix in &self.prefixes {
            for (name, v) in &vals {
                out.push(OscParam::float(
                    format!("{prefix}{name}"),
                    v.clamp(-1.0, 1.0),
                ));
            }
            // Status bools FT avatars gate their animator layers on. We have
            // no separate eye/lip hardware: one tracker drives everything, so
            // all three read true whenever a face is being tracked (map() is
            // only called on tracked frames).
            out.push(OscParam::bool(format!("{prefix}EyeTrackingActive"), true));
            out.push(OscParam::bool(
                format!("{prefix}ExpressionTrackingActive"),
                true,
            ));
            out.push(OscParam::bool(format!("{prefix}LipTrackingActive"), true));
        }
        out
    }

    /// Re-derive the neutral baselines (52 channels + 3 head axes) from a
    /// batch of relaxed forward-facing frames — same contract as
    /// [`super::arkit::ArkitMapper::calibrate`]. Resets smoothing state.
    pub fn calibrate(&mut self, frames: &[ArkitFaceFrame]) {
        if frames.is_empty() {
            return;
        }
        let n = frames.len() as f32;
        let mut ch_sum = [0.0f32; NUM_BLENDSHAPES];
        let mut head_sum = [0.0f32; 3];
        for f in frames {
            for (s, v) in ch_sum.iter_mut().zip(f.blendshapes.0.iter()) {
                *s += v;
            }
            head_sum[0] += f.head.roll;
            head_sum[1] += f.head.yaw;
            head_sum[2] += f.head.pitch;
        }
        let mut ch_avg = [0.0f32; NUM_BLENDSHAPES];
        for (a, s) in ch_avg.iter_mut().zip(ch_sum.iter()) {
            *a = s / n;
        }
        self.calib.seed(ch_avg);
        self.head_calib
            .seed([head_sum[0] / n, head_sum[1] / n, head_sum[2] / n]);
        self.filters = build_channel_filters(&self.config);
        self.head_filters = build_head_filters(&self.config);
    }
}

impl super::ArkitOscMapper for UnifiedMapper {
    fn map(&mut self, frame: &ArkitFaceFrame) -> Vec<OscParam> {
        UnifiedMapper::map(self, frame)
    }
    fn calibrate(&mut self, frames: &[ArkitFaceFrame]) {
        UnifiedMapper::calibrate(self, frames)
    }
    fn set_avatar_config(&mut self, config: Option<&AvatarOscConfig>) {
        UnifiedMapper::set_avatar_config(self, config)
    }
}

/// Compute every emitted `(name, value)` pair from the calibrated + smoothed
/// channel array `c` and the normalized head axes.
///
/// Ported formula-for-formula from VRCFT (see the module docs for sources):
/// first the **UE shapes** (LiveLink correlation), then the **simple shapes**
/// (`UnifiedSimpleExpressions` weighted blends), then the **combined
/// parameters** (`UnifiedExpressionsParameters`). Names are the wire names
/// minus the `/avatar/parameters/` + user-prefix parts.
fn param_values(
    c: &[f32; NUM_BLENDSHAPES],
    head_yaw: f32,
    head_pitch: f32,
    head_roll: f32,
) -> Vec<(&'static str, f32)> {
    // --- UE shapes (LiveLink ARKit correlation) ---
    let eye_squint_l = c[CH_EYE_SQUINT_LEFT];
    let eye_squint_r = c[CH_EYE_SQUINT_RIGHT];
    let eye_wide_l = c[CH_EYE_WIDE_LEFT];
    let eye_wide_r = c[CH_EYE_WIDE_RIGHT];
    let brow_pinch_l = c[CH_BROW_DOWN_LEFT];
    let brow_pinch_r = c[CH_BROW_DOWN_RIGHT];
    let brow_lowerer_l = c[CH_BROW_DOWN_LEFT];
    let brow_lowerer_r = c[CH_BROW_DOWN_RIGHT];
    let brow_inner_up_l = c[CH_BROW_INNER_UP];
    let brow_inner_up_r = c[CH_BROW_INNER_UP];
    let brow_outer_up_l = c[CH_BROW_OUTER_UP_LEFT];
    let brow_outer_up_r = c[CH_BROW_OUTER_UP_RIGHT];
    let cheek_squint_l = c[CH_CHEEK_SQUINT_LEFT];
    let cheek_squint_r = c[CH_CHEEK_SQUINT_RIGHT];
    let cheek_puff = c[CH_CHEEK_PUFF]; // both sides
    let cheek_suck = 0.0; // no ARKit source
    let jaw_open = c[CH_JAW_OPEN];
    let jaw_left = c[CH_JAW_LEFT];
    let jaw_right = c[CH_JAW_RIGHT];
    let jaw_forward = c[CH_JAW_FORWARD];
    let jaw_backward = 0.0; // no ARKit source
    let mouth_closed = c[CH_MOUTH_CLOSE];
    let mouth_upper_l = c[CH_MOUTH_LEFT];
    let mouth_lower_l = c[CH_MOUTH_LEFT];
    let mouth_upper_r = c[CH_MOUTH_RIGHT];
    let mouth_lower_r = c[CH_MOUTH_RIGHT];
    let corner_pull_l = c[CH_MOUTH_SMILE_LEFT];
    let corner_pull_r = c[CH_MOUTH_SMILE_RIGHT];
    let corner_slant_l = c[CH_MOUTH_SMILE_LEFT];
    let corner_slant_r = c[CH_MOUTH_SMILE_RIGHT];
    let mouth_frown_l = c[CH_MOUTH_FROWN_LEFT];
    let mouth_frown_r = c[CH_MOUTH_FROWN_RIGHT];
    let mouth_lower_down_l = c[CH_MOUTH_LOWER_DOWN_LEFT];
    let mouth_lower_down_r = c[CH_MOUTH_LOWER_DOWN_RIGHT];
    let mouth_upper_up_l = c[CH_MOUTH_UPPER_UP_LEFT];
    let mouth_upper_up_r = c[CH_MOUTH_UPPER_UP_RIGHT];
    let mouth_raiser_upper = c[CH_MOUTH_SHRUG_UPPER];
    let mouth_raiser_lower = c[CH_MOUTH_SHRUG_LOWER];
    let mouth_dimple_l = c[CH_MOUTH_DIMPLE_LEFT];
    let mouth_dimple_r = c[CH_MOUTH_DIMPLE_RIGHT];
    let mouth_press_l = c[CH_MOUTH_PRESS_LEFT];
    let mouth_press_r = c[CH_MOUTH_PRESS_RIGHT];
    let mouth_stretch_l = c[CH_MOUTH_STRETCH_LEFT];
    let mouth_stretch_r = c[CH_MOUTH_STRETCH_RIGHT];
    let mouth_tightener = 0.0; // no ARKit source (both sides)
    let lip_pucker = c[CH_MOUTH_PUCKER]; // all four corners
    let lip_funnel = c[CH_MOUTH_FUNNEL]; // all four corners
                                         // LiveLink: LipSuckUpper* = min(1 − upperUp^(1/6), mouthRollUpper).
    let lip_suck_upper_l = (1.0 - mouth_upper_up_l.powf(1.0 / 6.0)).min(c[CH_MOUTH_ROLL_UPPER]);
    let lip_suck_upper_r = (1.0 - mouth_upper_up_r.powf(1.0 / 6.0)).min(c[CH_MOUTH_ROLL_UPPER]);
    let lip_suck_lower = c[CH_MOUTH_ROLL_LOWER]; // both sides
    let nose_sneer_l = c[CH_NOSE_SNEER_LEFT];
    let nose_sneer_r = c[CH_NOSE_SNEER_RIGHT];

    // --- UnifiedEye state (LiveLink) ---
    let blink_l = c[CH_EYE_BLINK_LEFT];
    let blink_r = c[CH_EYE_BLINK_RIGHT];
    let openness_l = 1.0 - (blink_l + blink_l * eye_squint_l).clamp(0.0, 1.0);
    let openness_r = 1.0 - (blink_r + blink_r * eye_squint_r).clamp(0.0, 1.0);
    // Gaze: positive X = subject's right, positive Y = up (see module docs).
    let eye_left_x = c[CH_EYE_LOOK_IN_LEFT] - c[CH_EYE_LOOK_OUT_LEFT];
    let eye_right_x = c[CH_EYE_LOOK_OUT_RIGHT] - c[CH_EYE_LOOK_IN_RIGHT];
    let eye_left_y = c[CH_EYE_LOOK_UP_LEFT] - c[CH_EYE_LOOK_DOWN_LEFT];
    let eye_right_y = c[CH_EYE_LOOK_UP_RIGHT] - c[CH_EYE_LOOK_DOWN_RIGHT];

    // --- Simple shapes (UnifiedSimpleExpressions.ExpressionMap) ---
    let brow_up_l = brow_outer_up_l * 0.6 + brow_inner_up_l * 0.4;
    let brow_up_r = brow_outer_up_r * 0.6 + brow_inner_up_r * 0.4;
    let brow_down_l = brow_lowerer_l * 0.75 + brow_pinch_l * 0.25;
    let brow_down_r = brow_lowerer_r * 0.75 + brow_pinch_r * 0.25;
    let mouth_smile_l = corner_pull_l * 0.8 + corner_slant_l * 0.2;
    let mouth_smile_r = corner_pull_r * 0.8 + corner_slant_r * 0.2;
    let mouth_sad_l = mouth_frown_l.max(mouth_stretch_l);
    let mouth_sad_r = mouth_frown_r.max(mouth_stretch_r);

    vec![
        // ---- Raw UE shapes, `v2/<Shape>` ----
        ("v2/EyeSquintRight", eye_squint_r),
        ("v2/EyeSquintLeft", eye_squint_l),
        ("v2/EyeWideRight", eye_wide_r),
        ("v2/EyeWideLeft", eye_wide_l),
        ("v2/BrowPinchRight", brow_pinch_r),
        ("v2/BrowPinchLeft", brow_pinch_l),
        ("v2/BrowLowererRight", brow_lowerer_r),
        ("v2/BrowLowererLeft", brow_lowerer_l),
        ("v2/BrowInnerUpRight", brow_inner_up_r),
        ("v2/BrowInnerUpLeft", brow_inner_up_l),
        ("v2/BrowOuterUpRight", brow_outer_up_r),
        ("v2/BrowOuterUpLeft", brow_outer_up_l),
        ("v2/CheekSquintRight", cheek_squint_r),
        ("v2/CheekSquintLeft", cheek_squint_l),
        ("v2/CheekPuffRight", cheek_puff),
        ("v2/CheekPuffLeft", cheek_puff),
        ("v2/CheekSuckRight", cheek_suck),
        ("v2/CheekSuckLeft", cheek_suck),
        ("v2/JawOpen", jaw_open),
        ("v2/JawRight", jaw_right),
        ("v2/JawLeft", jaw_left),
        ("v2/JawForward", jaw_forward),
        ("v2/JawBackward", jaw_backward),
        ("v2/MouthClosed", mouth_closed),
        ("v2/MouthUpperUpRight", mouth_upper_up_r),
        ("v2/MouthUpperUpLeft", mouth_upper_up_l),
        ("v2/MouthUpperDeepenRight", mouth_upper_up_r),
        ("v2/MouthUpperDeepenLeft", mouth_upper_up_l),
        ("v2/MouthLowerDownRight", mouth_lower_down_r),
        ("v2/MouthLowerDownLeft", mouth_lower_down_l),
        ("v2/MouthUpperRight", mouth_upper_r),
        ("v2/MouthUpperLeft", mouth_upper_l),
        ("v2/MouthLowerRight", mouth_lower_r),
        ("v2/MouthLowerLeft", mouth_lower_l),
        ("v2/MouthCornerPullRight", corner_pull_r),
        ("v2/MouthCornerPullLeft", corner_pull_l),
        ("v2/MouthCornerSlantRight", corner_slant_r),
        ("v2/MouthCornerSlantLeft", corner_slant_l),
        ("v2/MouthFrownRight", mouth_frown_r),
        ("v2/MouthFrownLeft", mouth_frown_l),
        ("v2/MouthStretchRight", mouth_stretch_r),
        ("v2/MouthStretchLeft", mouth_stretch_l),
        ("v2/MouthDimpleRight", mouth_dimple_r),
        ("v2/MouthDimpleLeft", mouth_dimple_l),
        ("v2/MouthRaiserUpper", mouth_raiser_upper),
        ("v2/MouthRaiserLower", mouth_raiser_lower),
        ("v2/MouthPressRight", mouth_press_r),
        ("v2/MouthPressLeft", mouth_press_l),
        ("v2/MouthTightenerRight", mouth_tightener),
        ("v2/MouthTightenerLeft", mouth_tightener),
        ("v2/LipSuckUpperRight", lip_suck_upper_r),
        ("v2/LipSuckUpperLeft", lip_suck_upper_l),
        ("v2/LipSuckLowerRight", lip_suck_lower),
        ("v2/LipSuckLowerLeft", lip_suck_lower),
        ("v2/LipFunnelUpperRight", lip_funnel),
        ("v2/LipFunnelUpperLeft", lip_funnel),
        ("v2/LipFunnelLowerRight", lip_funnel),
        ("v2/LipFunnelLowerLeft", lip_funnel),
        ("v2/LipPuckerUpperRight", lip_pucker),
        ("v2/LipPuckerUpperLeft", lip_pucker),
        ("v2/LipPuckerLowerRight", lip_pucker),
        ("v2/LipPuckerLowerLeft", lip_pucker),
        ("v2/NoseSneerRight", nose_sneer_r),
        ("v2/NoseSneerLeft", nose_sneer_l),
        // ---- Simple shapes, `v2/<Simple>` ----
        ("v2/BrowUpRight", brow_up_r),
        ("v2/BrowUpLeft", brow_up_l),
        ("v2/BrowDownRight", brow_down_r),
        ("v2/BrowDownLeft", brow_down_l),
        ("v2/MouthSmileRight", mouth_smile_r),
        ("v2/MouthSmileLeft", mouth_smile_l),
        ("v2/MouthSadRight", mouth_sad_r),
        ("v2/MouthSadLeft", mouth_sad_l),
        // ---- Combined parameters (UnifiedExpressionsParameters) ----
        // Eye gaze
        ("v2/EyeLeftX", eye_left_x),
        ("v2/EyeLeftY", eye_left_y),
        ("v2/EyeRightX", eye_right_x),
        ("v2/EyeRightY", eye_right_y),
        ("v2/EyeX", (eye_left_x + eye_right_x) / 2.0),
        ("v2/EyeY", (eye_left_y + eye_right_y) / 2.0),
        // Eye openness
        ("v2/EyeOpenLeft", openness_l),
        ("v2/EyeOpenRight", openness_r),
        ("v2/EyeOpen", (openness_l + openness_r) / 2.0),
        ("v2/EyeClosedLeft", 1.0 - openness_l),
        ("v2/EyeClosedRight", 1.0 - openness_r),
        ("v2/EyeClosed", 1.0 - (openness_l + openness_r) / 2.0),
        // Eyes widen / lids / squint
        ("v2/EyeWide", eye_wide_l.max(eye_wide_r)),
        ("v2/EyeLidLeft", openness_l * 0.75 + eye_wide_l * 0.25),
        ("v2/EyeLidRight", openness_r * 0.75 + eye_wide_r * 0.25),
        (
            "v2/EyeLid",
            ((openness_l + openness_r) / 2.0) * 0.75 + ((eye_wide_r + eye_wide_l) / 2.0) * 0.25,
        ),
        ("v2/EyeSquint", eye_squint_l.max(eye_squint_r)),
        ("v2/EyesSquint", eye_squint_l.max(eye_squint_r)),
        // Eyebrows compacted
        ("v2/BrowUp", (brow_up_r + brow_up_l) * 0.5),
        ("v2/BrowDown", (brow_down_r + brow_down_l) * 0.5),
        ("v2/BrowInnerUp", (brow_inner_up_l + brow_inner_up_r) / 2.0),
        ("v2/BrowOuterUp", (brow_outer_up_l + brow_outer_up_r) / 2.0),
        (
            "v2/BrowExpressionRight",
            (brow_inner_up_r * 0.5 + brow_outer_up_r * 0.5).min(1.0) - brow_down_r,
        ),
        (
            "v2/BrowExpressionLeft",
            (brow_inner_up_l * 0.5 + brow_outer_up_l * 0.5).min(1.0) - brow_down_l,
        ),
        (
            "v2/BrowExpression",
            (((brow_inner_up_r + brow_outer_up_r) * 0.5).min(1.0) - brow_down_r
                + ((brow_inner_up_l + brow_outer_up_l) * 0.5).min(1.0)
                - brow_down_l)
                * 0.5,
        ),
        // Jaw combined
        ("v2/JawX", jaw_right - jaw_left),
        ("v2/JawZ", jaw_forward - jaw_backward),
        // Cheeks combined
        ("v2/CheekSquint", (cheek_squint_l + cheek_squint_r) / 2.0),
        ("v2/CheekPuffSuckLeft", cheek_puff - cheek_suck),
        ("v2/CheekPuffSuckRight", cheek_puff - cheek_suck),
        ("v2/CheekPuffSuck", cheek_puff - cheek_suck),
        ("v2/CheekSuck", cheek_suck),
        // Mouth direction
        ("v2/MouthUpperX", mouth_upper_r - mouth_upper_l),
        ("v2/MouthLowerX", mouth_lower_r - mouth_lower_l),
        (
            "v2/MouthX",
            (mouth_upper_r + mouth_lower_r) / 2.0 - (mouth_upper_l + mouth_lower_l) / 2.0,
        ),
        // Lip combined
        (
            "v2/LipSuckUpper",
            (lip_suck_upper_r + lip_suck_upper_l) / 2.0,
        ),
        ("v2/LipSuckLower", lip_suck_lower),
        (
            "v2/LipSuck",
            (lip_suck_upper_r + lip_suck_upper_l + lip_suck_lower + lip_suck_lower) / 4.0,
        ),
        ("v2/LipFunnelUpper", lip_funnel),
        ("v2/LipFunnelLower", lip_funnel),
        ("v2/LipFunnel", lip_funnel),
        ("v2/LipPuckerUpper", lip_pucker),
        ("v2/LipPuckerLower", lip_pucker),
        ("v2/LipPuckerRight", lip_pucker),
        ("v2/LipPuckerLeft", lip_pucker),
        ("v2/LipPucker", lip_pucker),
        (
            "v2/LipSuckFunnelUpper",
            (lip_suck_upper_r + lip_suck_upper_l) / 2.0 - lip_funnel,
        ),
        ("v2/LipSuckFunnelLower", lip_suck_lower - lip_funnel),
        ("v2/LipSuckFunnelLowerLeft", lip_suck_lower - lip_funnel),
        ("v2/LipSuckFunnelLowerRight", lip_suck_lower - lip_funnel),
        ("v2/LipSuckFunnelUpperLeft", lip_suck_upper_l - lip_funnel),
        ("v2/LipSuckFunnelUpperRight", lip_suck_upper_r - lip_funnel),
        // Mouth combined
        (
            "v2/MouthUpperUp",
            mouth_upper_up_r * 0.5 + mouth_upper_up_l * 0.5,
        ),
        (
            "v2/MouthLowerDown",
            mouth_lower_down_r * 0.5 + mouth_lower_down_l * 0.5,
        ),
        (
            "v2/MouthOpen",
            mouth_upper_up_r * 0.25
                + mouth_upper_up_l * 0.25
                + mouth_lower_down_r * 0.25
                + mouth_lower_down_l * 0.25,
        ),
        ("v2/MouthStretch", (mouth_stretch_r + mouth_stretch_l) / 2.0),
        ("v2/MouthTightener", mouth_tightener),
        ("v2/MouthPress", (mouth_press_r + mouth_press_l) / 2.0),
        ("v2/MouthDimple", (mouth_dimple_r + mouth_dimple_l) / 2.0),
        ("v2/NoseSneer", (nose_sneer_r + nose_sneer_l) / 2.0),
        (
            "v2/MouthTightenerStretch",
            mouth_tightener - (mouth_stretch_r + mouth_stretch_l) / 2.0,
        ),
        (
            "v2/MouthTightenerStretchLeft",
            mouth_tightener - mouth_stretch_l,
        ),
        (
            "v2/MouthTightenerStretchRight",
            mouth_tightener - mouth_stretch_r,
        ),
        // Lip corners combined
        ("v2/MouthCornerYLeft", corner_slant_l - mouth_frown_l),
        ("v2/MouthCornerYRight", corner_slant_r - mouth_frown_r),
        (
            "v2/MouthCornerY",
            (corner_slant_l - mouth_frown_l + corner_slant_r - mouth_frown_r) * 0.5,
        ),
        ("v2/SmileFrownRight", mouth_smile_r - mouth_frown_r),
        ("v2/SmileFrownLeft", mouth_smile_l - mouth_frown_l),
        (
            "v2/SmileFrown",
            mouth_smile_r * 0.5 + mouth_smile_l * 0.5 - mouth_frown_r * 0.5 - mouth_frown_l * 0.5,
        ),
        ("v2/SmileSadRight", mouth_smile_r - mouth_sad_r),
        ("v2/SmileSadLeft", mouth_smile_l - mouth_sad_l),
        (
            "v2/SmileSad",
            (mouth_smile_l + mouth_smile_r) / 2.0 - (mouth_sad_l + mouth_sad_r) / 2.0,
        ),
        // Head rotation (UnifiedHeadParameters; position axes have no source)
        ("v2/Head/Yaw", head_yaw),
        ("v2/Head/Pitch", head_pitch),
        ("v2/Head/Roll", head_roll),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::OscValue;
    use crate::tracking::arkit::{ArkitBlendshapes, HeadPose, ARKIT_BLENDSHAPE_NAMES};

    /// Every `CH_*` index must match its name in the MediaPipe order.
    #[test]
    fn channel_index_consts_match_names() {
        let pairs = [
            (CH_BROW_DOWN_LEFT, "browDownLeft"),
            (CH_BROW_DOWN_RIGHT, "browDownRight"),
            (CH_BROW_INNER_UP, "browInnerUp"),
            (CH_BROW_OUTER_UP_LEFT, "browOuterUpLeft"),
            (CH_BROW_OUTER_UP_RIGHT, "browOuterUpRight"),
            (CH_CHEEK_PUFF, "cheekPuff"),
            (CH_CHEEK_SQUINT_LEFT, "cheekSquintLeft"),
            (CH_CHEEK_SQUINT_RIGHT, "cheekSquintRight"),
            (CH_EYE_BLINK_LEFT, "eyeBlinkLeft"),
            (CH_EYE_BLINK_RIGHT, "eyeBlinkRight"),
            (CH_EYE_LOOK_DOWN_LEFT, "eyeLookDownLeft"),
            (CH_EYE_LOOK_DOWN_RIGHT, "eyeLookDownRight"),
            (CH_EYE_LOOK_IN_LEFT, "eyeLookInLeft"),
            (CH_EYE_LOOK_IN_RIGHT, "eyeLookInRight"),
            (CH_EYE_LOOK_OUT_LEFT, "eyeLookOutLeft"),
            (CH_EYE_LOOK_OUT_RIGHT, "eyeLookOutRight"),
            (CH_EYE_LOOK_UP_LEFT, "eyeLookUpLeft"),
            (CH_EYE_LOOK_UP_RIGHT, "eyeLookUpRight"),
            (CH_EYE_SQUINT_LEFT, "eyeSquintLeft"),
            (CH_EYE_SQUINT_RIGHT, "eyeSquintRight"),
            (CH_EYE_WIDE_LEFT, "eyeWideLeft"),
            (CH_EYE_WIDE_RIGHT, "eyeWideRight"),
            (CH_JAW_FORWARD, "jawForward"),
            (CH_JAW_LEFT, "jawLeft"),
            (CH_JAW_OPEN, "jawOpen"),
            (CH_JAW_RIGHT, "jawRight"),
            (CH_MOUTH_CLOSE, "mouthClose"),
            (CH_MOUTH_DIMPLE_LEFT, "mouthDimpleLeft"),
            (CH_MOUTH_DIMPLE_RIGHT, "mouthDimpleRight"),
            (CH_MOUTH_FROWN_LEFT, "mouthFrownLeft"),
            (CH_MOUTH_FROWN_RIGHT, "mouthFrownRight"),
            (CH_MOUTH_FUNNEL, "mouthFunnel"),
            (CH_MOUTH_LEFT, "mouthLeft"),
            (CH_MOUTH_LOWER_DOWN_LEFT, "mouthLowerDownLeft"),
            (CH_MOUTH_LOWER_DOWN_RIGHT, "mouthLowerDownRight"),
            (CH_MOUTH_PRESS_LEFT, "mouthPressLeft"),
            (CH_MOUTH_PRESS_RIGHT, "mouthPressRight"),
            (CH_MOUTH_PUCKER, "mouthPucker"),
            (CH_MOUTH_RIGHT, "mouthRight"),
            (CH_MOUTH_ROLL_LOWER, "mouthRollLower"),
            (CH_MOUTH_ROLL_UPPER, "mouthRollUpper"),
            (CH_MOUTH_SHRUG_LOWER, "mouthShrugLower"),
            (CH_MOUTH_SHRUG_UPPER, "mouthShrugUpper"),
            (CH_MOUTH_SMILE_LEFT, "mouthSmileLeft"),
            (CH_MOUTH_SMILE_RIGHT, "mouthSmileRight"),
            (CH_MOUTH_STRETCH_LEFT, "mouthStretchLeft"),
            (CH_MOUTH_STRETCH_RIGHT, "mouthStretchRight"),
            (CH_MOUTH_UPPER_UP_LEFT, "mouthUpperUpLeft"),
            (CH_MOUTH_UPPER_UP_RIGHT, "mouthUpperUpRight"),
            (CH_NOSE_SNEER_LEFT, "noseSneerLeft"),
            (CH_NOSE_SNEER_RIGHT, "noseSneerRight"),
        ];
        for (idx, name) in pairs {
            assert_eq!(
                ARKIT_BLENDSHAPE_NAMES[idx], name,
                "index {idx} should be {name}"
            );
        }
    }

    fn bs(values: &[(&str, f32)]) -> ArkitBlendshapes {
        let mut a = [0.0f32; NUM_BLENDSHAPES];
        for &(name, v) in values {
            let idx = ARKIT_BLENDSHAPE_NAMES
                .iter()
                .position(|&n| n == name)
                .unwrap_or_else(|| panic!("unknown blendshape {name}"));
            a[idx] = v;
        }
        ArkitBlendshapes(a)
    }

    fn frame(values: &[(&str, f32)], head: HeadPose) -> ArkitFaceFrame {
        ArkitFaceFrame {
            blendshapes: bs(values),
            head,
        }
    }

    fn neutral_frame() -> ArkitFaceFrame {
        frame(
            &[
                ("jawOpen", 0.22),
                ("mouthPucker", 0.3),
                ("eyeBlinkLeft", 0.08),
                ("eyeBlinkRight", 0.12),
                ("browInnerUp", 0.15),
                ("mouthSmileLeft", 0.05),
                ("mouthSmileRight", 0.05),
            ],
            HeadPose::default(),
        )
    }

    fn float_of(out: &[OscParam], name: &str) -> f32 {
        let p = out.iter().find(|p| p.name == name).expect("param present");
        match p.value {
            OscValue::Float(v) => v,
            _ => panic!("expected float param for {name}"),
        }
    }

    fn bool_of(out: &[OscParam], name: &str) -> bool {
        let p = out.iter().find(|p| p.name == name).expect("param present");
        match p.value {
            OscValue::Bool(v) => v,
            _ => panic!("expected bool param for {name}"),
        }
    }

    fn mapper(calib_frames: u32) -> UnifiedMapper {
        UnifiedMapper::new(
            ArkitMappingConfig {
                calib_frames,
                ..Default::default()
            },
            vec![String::new()],
        )
    }

    // Neutral face after the calibration window: expressions rest at 0 and the
    // eyes read fully open (VRCFT semantics: EyeOpen 1, EyeLid 0.75).
    #[test]
    fn neutral_rests_at_vrcft_neutral_values() {
        let mut m = mapper(10);
        let mut last = Vec::new();
        for _ in 0..80 {
            last = m.map(&neutral_frame());
        }
        for name in [
            "v2/JawOpen",
            "v2/LipPucker",
            "v2/SmileFrown",
            "v2/BrowInnerUpLeft",
            "v2/MouthOpen",
            "v2/EyeClosedLeft",
            "v2/EyeClosedRight",
            "v2/Head/Yaw",
        ] {
            let v = float_of(&last, name);
            assert!(v.abs() < 0.05, "{name} did not rest near 0: {v}");
        }
        assert!((float_of(&last, "v2/EyeOpenLeft") - 1.0).abs() < 0.05);
        assert!((float_of(&last, "v2/EyeOpenRight") - 1.0).abs() < 0.05);
        assert!((float_of(&last, "v2/EyeLidLeft") - 0.75).abs() < 0.05);
        assert!((float_of(&last, "v2/EyeLidRight") - 0.75).abs() < 0.05);
    }

    // A left blink (raw jumps blink_gain above the calibrated open baseline)
    // closes the left eye only.
    #[test]
    fn blink_left_closes_left_eye_only() {
        let mut m = mapper(10);
        let rest = frame(
            &[("eyeBlinkLeft", 0.08), ("eyeBlinkRight", 0.12)],
            HeadPose::default(),
        );
        for _ in 0..10 {
            m.map(&rest);
        }
        let blink = frame(
            &[("eyeBlinkLeft", 0.08 + 0.35), ("eyeBlinkRight", 0.12)],
            HeadPose::default(),
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&blink);
        }
        assert!((float_of(&last, "v2/EyeClosedLeft") - 1.0).abs() < 2e-2);
        assert!(float_of(&last, "v2/EyeOpenLeft") < 0.05);
        assert!(float_of(&last, "v2/EyeLidLeft") < 0.05);
        assert!(float_of(&last, "v2/EyeClosedRight") < 0.05);
        assert!((float_of(&last, "v2/EyeLidRight") - 0.75).abs() < 0.05);
    }

    // Opening the jaw drives v2/JawOpen; closing lips (mouthClose) drives
    // v2/MouthClosed.
    #[test]
    fn jaw_open_and_mouth_closed_drive_their_shapes() {
        let mut m = mapper(5);
        let rest = frame(&[("jawOpen", 0.2)], HeadPose::default());
        for _ in 0..5 {
            m.map(&rest);
        }
        let open = frame(
            &[("jawOpen", 0.9), ("mouthClose", 0.4)],
            HeadPose::default(),
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&open);
        }
        let jaw = float_of(&last, "v2/JawOpen");
        assert!(jaw > 0.5, "v2/JawOpen should be large: {jaw}");
        let closed = float_of(&last, "v2/MouthClosed");
        assert!(closed > 0.2, "v2/MouthClosed should engage: {closed}");
    }

    // Smiling drives SmileFrown positive; frowning drives it negative.
    #[test]
    fn smile_frown_axis_signs() {
        let mut m = mapper(5);
        let rest = frame(&[], HeadPose::default());
        for _ in 0..5 {
            m.map(&rest);
        }
        let smile = frame(
            &[("mouthSmileLeft", 0.8), ("mouthSmileRight", 0.8)],
            HeadPose::default(),
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&smile);
        }
        assert!(float_of(&last, "v2/SmileFrown") > 0.5);
        assert!(float_of(&last, "v2/SmileFrownLeft") > 0.5);
        assert!(float_of(&last, "v2/MouthSmileLeft") > 0.5);

        let mut m = mapper(5);
        for _ in 0..5 {
            m.map(&rest);
        }
        let frown = frame(
            &[("mouthFrownLeft", 0.8), ("mouthFrownRight", 0.8)],
            HeadPose::default(),
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&frown);
        }
        assert!(float_of(&last, "v2/SmileFrown") < -0.5);
        assert!(float_of(&last, "v2/MouthSadLeft") > 0.5);
    }

    // Both eyes looking toward the subject's right → positive EyeX family.
    #[test]
    fn gaze_right_is_positive_x() {
        let mut m = mapper(5);
        let rest = frame(&[], HeadPose::default());
        for _ in 0..5 {
            m.map(&rest);
        }
        let right = frame(
            &[("eyeLookInLeft", 0.8), ("eyeLookOutRight", 0.8)],
            HeadPose::default(),
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&right);
        }
        assert!(float_of(&last, "v2/EyeLeftX") > 0.5);
        assert!(float_of(&last, "v2/EyeRightX") > 0.5);
        assert!(float_of(&last, "v2/EyeX") > 0.5);
        assert!(float_of(&last, "v2/EyeY").abs() < 0.05);
    }

    // Head axes: same conventions as mapping::arkit — +yaw_max → Head/Yaw 1,
    // +roll (subject's right) → Head/Roll −1 (flipped).
    #[test]
    fn head_axes_follow_arkit_mapper_conventions() {
        let cfg = ArkitMappingConfig {
            calib_frames: 4,
            ..Default::default()
        };
        let mut m = UnifiedMapper::new(cfg, vec![String::new()]);
        let neutral = frame(&[], HeadPose::default());
        for _ in 0..4 {
            m.map(&neutral);
        }
        let turned = frame(
            &[],
            HeadPose {
                roll: cfg.roll_max_rad,
                yaw: cfg.yaw_max_rad,
                pitch: 0.0,
            },
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&turned);
        }
        assert!((float_of(&last, "v2/Head/Yaw") - 1.0).abs() < 2e-2);
        assert!((float_of(&last, "v2/Head/Roll") + 1.0).abs() < 2e-2);
        assert!(float_of(&last, "v2/Head/Pitch").abs() < 0.05);
    }

    // Default prefixes: every param goes out both bare and under FT/, and the
    // three status bools are true under each prefix.
    #[test]
    fn default_prefixes_emit_bare_and_ft_copies() {
        let mut m = UnifiedMapper::new(ArkitMappingConfig::default(), Vec::new());
        let out = m.map(&neutral_frame());
        for name in ["v2/JawOpen", "v2/EyeLidLeft", "v2/Head/Yaw"] {
            let bare = float_of(&out, name);
            let ft = float_of(&out, &format!("FT/{name}"));
            assert_eq!(bare, ft, "{name}: FT/ copy must carry the same value");
        }
        for name in [
            "EyeTrackingActive",
            "ExpressionTrackingActive",
            "LipTrackingActive",
        ] {
            assert!(bool_of(&out, name));
            assert!(bool_of(&out, &format!("FT/{name}")));
        }
    }

    // Custom prefix list is respected verbatim (no bare copy).
    #[test]
    fn custom_prefix_list_is_respected() {
        let mut m = UnifiedMapper::new(ArkitMappingConfig::default(), vec!["Custom/".to_string()]);
        let out = m.map(&neutral_frame());
        assert!(out.iter().any(|p| p.name == "Custom/v2/JawOpen"));
        assert!(!out.iter().any(|p| p.name == "v2/JawOpen"));
        assert!(bool_of(&out, "Custom/EyeTrackingActive"));
    }

    // All emitted floats stay within [-1, 1] for fuzzed inputs.
    #[test]
    fn outputs_stay_in_range_for_fuzzed_input() {
        let mut m = UnifiedMapper::new(ArkitMappingConfig::default(), vec![String::new()]);
        let mut lcg = Lcg(1234);
        for _ in 0..300 {
            let mut a = [0.0f32; NUM_BLENDSHAPES];
            for v in a.iter_mut() {
                *v = lcg.next_f32() * 2.0; // allow out-of-[0,1] noise
            }
            let f = ArkitFaceFrame {
                blendshapes: ArkitBlendshapes(a),
                head: HeadPose {
                    roll: (lcg.next_f32() - 0.5) * 4.0,
                    yaw: (lcg.next_f32() - 0.5) * 4.0,
                    pitch: (lcg.next_f32() - 0.5) * 4.0,
                },
            };
            for p in m.map(&f) {
                if let OscValue::Float(v) = p.value {
                    assert!(
                        (-1.0 - 1e-4..=1.0 + 1e-4).contains(&v),
                        "{} out of range: {v}",
                        p.name
                    );
                }
            }
        }
    }

    // Explicit calibrate() seeds the baseline immediately.
    #[test]
    fn explicit_calibrate_seeds_baseline() {
        let mut m = mapper(30); // large auto window; calibrate must bypass it
        let samples = vec![neutral_frame(); 5];
        m.calibrate(&samples);
        let mut last = Vec::new();
        for _ in 0..20 {
            last = m.map(&neutral_frame());
        }
        assert!(float_of(&last, "v2/JawOpen").abs() < 0.05);
        assert!((float_of(&last, "v2/EyeOpenLeft") - 1.0).abs() < 0.05);
    }

    // ---- Avatar-aware gating (issue #18 phases 2+3) ----

    fn avatar_cfg(json: &str) -> AvatarOscConfig {
        AvatarOscConfig::from_json(json).unwrap()
    }

    // Every canonical name is unique (the AvatarParamSet resolution keys on
    // it) and the universe includes shapes, combined params, and head axes.
    #[test]
    fn float_param_names_are_unique_and_complete() {
        let names = float_param_names();
        let mut dedup = names.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), names.len(), "canonical names must be unique");
        for expected in [
            "v2/JawOpen",
            "v2/SmileFrown",
            "v2/EyeLidLeft",
            "v2/Head/Yaw",
        ] {
            assert!(names.contains(&expected), "{expected} missing");
        }
    }

    // With an avatar config set, only declared params go out — at their
    // exact declared addresses (arbitrary prefix), and the blind-prefix
    // copies disappear.
    #[test]
    fn avatar_config_gates_output_to_declared_addresses() {
        let mut m = UnifiedMapper::new(ArkitMappingConfig::default(), Vec::new());
        m.set_avatar_config(Some(&avatar_cfg(
            r#"{
              "id": "avtr_x", "name": "x",
              "parameters": [
                {"name":"Odd/v2/JawOpen",
                 "input":{"address":"/avatar/parameters/Odd/v2/JawOpen","type":"Float"}},
                {"name":"LipTrackingActive",
                 "input":{"address":"/avatar/parameters/LipTrackingActive","type":"Bool"}}
              ]
            }"#,
        )));
        let out = m.map(&neutral_frame());
        assert_eq!(out.len(), 2, "exactly the declared params: {out:?}");
        assert!(out
            .iter()
            .any(|p| p.name == "/avatar/parameters/Odd/v2/JawOpen"
                && matches!(p.value, OscValue::Float(_))));
        assert!(bool_of(&out, "/avatar/parameters/LipTrackingActive"));

        // Clearing the config restores the blind-prefix fallback exactly.
        m.set_avatar_config(None);
        let out = m.map(&neutral_frame());
        assert!(out.iter().any(|p| p.name == "v2/JawOpen"));
        assert!(out.iter().any(|p| p.name == "FT/v2/JawOpen"));
        assert!(bool_of(&out, "EyeTrackingActive"));
    }

    // Binary-only declaration: bits + Negative carry the value; the 1.0
    // edge case saturates to all-ones through the whole mapper path.
    #[test]
    fn avatar_config_binary_bits_flow_through_map() {
        let mut m = mapper(0); // calib_frames 0 → raw values flow through
        m.set_avatar_config(Some(&avatar_cfg(
            r#"{
              "id": "avtr_b", "name": "b",
              "parameters": [
                {"name":"FT/v2/JawOpen1",
                 "input":{"address":"/avatar/parameters/FT/v2/JawOpen1","type":"Bool"}},
                {"name":"FT/v2/JawOpen2",
                 "input":{"address":"/avatar/parameters/FT/v2/JawOpen2","type":"Bool"}}
              ]
            }"#,
        )));
        let open = frame(&[("jawOpen", 1.0)], HeadPose::default());
        let mut last = Vec::new();
        for _ in 0..30 {
            last = m.map(&open);
        }
        assert_eq!(last.len(), 2);
        assert!(bool_of(&last, "/avatar/parameters/FT/v2/JawOpen1"));
        assert!(bool_of(&last, "/avatar/parameters/FT/v2/JawOpen2"));

        // Closed jaw → both bits clear.
        let closed = frame(&[], HeadPose::default());
        let mut last = Vec::new();
        for _ in 0..60 {
            last = m.map(&closed);
        }
        assert!(!bool_of(&last, "/avatar/parameters/FT/v2/JawOpen1"));
        assert!(!bool_of(&last, "/avatar/parameters/FT/v2/JawOpen2"));
    }

    // An avatar that declares no FT params at all: nothing is sent — same
    // as VRCFT (its params all become irrelevant), NOT the blind fallback.
    #[test]
    fn avatar_without_ft_params_sends_nothing() {
        let mut m = UnifiedMapper::new(ArkitMappingConfig::default(), Vec::new());
        m.set_avatar_config(Some(&avatar_cfg(
            r#"{"id":"avtr_plain","name":"plain","parameters":[
                {"name":"GestureLeft",
                 "input":{"address":"/avatar/parameters/GestureLeft","type":"Int"}}
            ]}"#,
        )));
        assert!(m.avatar_params().unwrap().is_empty());
        assert!(m.map(&neutral_frame()).is_empty());
    }

    /// Minimal seeded LCG for deterministic fuzz (mirrors mapping::arkit).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as u32 as f32) / (u32::MAX as f32)
        }
    }
}
