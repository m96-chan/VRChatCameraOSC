//! ARKit-blendshape → VRChat avatar OSC parameters (issue #17).
//!
//! Companion to [`super`]'s FAN/iBUG-68 [`super::Mapper`]: same 10 [`OscParam`]
//! outputs, same [`crate::osc::OscParam`] shape, so the pipeline can swap
//! tracking backends (`fan` vs `mediapipe`) without touching downstream code.
//! The *input* signal is fundamentally different — 52 NN-predicted ARKit
//! coefficients plus a 3-axis head rotation (see
//! [`crate::tracking::arkit::ArkitFaceFrame`]) instead of 68 geometric 2D
//! points — so the derivation is a different module, per the "models are
//! pluggable" boundary in CLAUDE.md.
//!
//! # Design ported from AvataCam
//!
//! MediaPipe's Blendshape V2 coefficients have a non-zero, per-person,
//! per-channel resting baseline (a relaxed face reads as a permanent pucker
//! or scowl), and its blink coefficient does not reliably saturate to a
//! shared `0/1` window across eyes or people. AvataCam's `crates/app/src/
//! track.rs` (`BsCalib`) and `crates/app/src/expression.rs` (tuning
//! constants) fix this with per-channel neutral-baseline subtraction plus
//! per-eye blink self-calibration, and a One-Euro low-pass per channel class
//! to kill idle jitter without dulling fast transients (blinks, speech).
//! This module ports that design, scoped to the 11 raw channels the mapping
//! table below actually needs (not all 52 — the rest never reach an OSC
//! param, so calibrating/smoothing them would be dead weight).
//!
//! # Mapping table
//!
//! | Parameter       | Range   | Source ARKit channel(s)                              | Formula |
//! |------------------|---------|-------------------------------------------------------|---------|
//! | `MouthOpen`      | `0..1`  | `jawOpen`                                              | calibrated `jawOpen` |
//! | `EyeBlinkLeft`   | `0..1`  | `eyeBlinkLeft`                                         | `(raw − left-eye open baseline) / blink_gain`, clamped `0..1` |
//! | `EyeBlinkRight`  | `0..1`  | `eyeBlinkRight`                                        | `(raw − right-eye open baseline) / blink_gain`, clamped `0..1` |
//! | `BrowUpLeft`     | `0..1`  | `browInnerUp`, `browOuterUpLeft`                       | `max(calibrated browInnerUp, calibrated browOuterUpLeft)` |
//! | `BrowUpRight`    | `0..1`  | `browInnerUp`, `browOuterUpRight`                      | `max(calibrated browInnerUp, calibrated browOuterUpRight)` |
//! | `MouthSmile`     | `0..1`  | `mouthSmileLeft`, `mouthSmileRight`                    | `avg(calibrated mouthSmileLeft, calibrated mouthSmileRight)` |
//! | `MouthWide`      | `-1..1` | `mouthStretchLeft`, `mouthStretchRight`, `mouthPucker` | `avg(calibrated mouthStretch*) − calibrated mouthPucker` |
//! | `HeadRoll`       | `-1..1` | [`HeadPose::roll`]                                     | see sign note below |
//! | `HeadYaw`        | `-1..1` | [`HeadPose::yaw`]                                      | see sign note below |
//! | `HeadPitch`      | `-1..1` | [`HeadPose::pitch`]                                    | see sign note below |
//!
//! Every "calibrated `X`" above is the raw coefficient, neutral-baseline
//! subtracted and clamped `≥ 0` (see [`BsCalib`]), then One-Euro smoothed
//! (see [`OneEuro`]). Every final output is clamped to [`PARAM_RANGES`]
//! regardless of the formula above, so out-of-range intermediate arithmetic
//! (e.g. `MouthWide`'s subtraction) can never leak out of range.
//!
//! `BrowUpLeft`/`Right` deliberately reuse `browInnerUp` on both sides (an
//! inner-brow raise reads on both `BrowUp*` channels) — `browInnerUp` has no
//! left/right split in the ARKit set, and `max` means either the shared inner
//! raise or that side's *outer* raise can drive the parameter, matching how
//! a real face's brow-up expression usually moves both together.
//!
//! # Head-pose sign conventions — **to be verified in the live demo**
//!
//! [`HeadPose`] (in [`crate::tracking::arkit`], not modified by this module)
//! documents its own signs precisely: `roll` positive = tilting toward the
//! subject's **right** (counter-clockwise as seen by the viewer); `yaw`
//! positive = turning toward the subject's **left**; `pitch` positive =
//! looking **up**. [`super`]'s FAN-based `Mapper` predates this module and
//! defines `HeadRoll`/`HeadYaw`/`HeadPitch` from 2D landmark geometry instead
//! — to keep the Unity SDK wiring unchanged (CLAUDE.md), this module's output
//! is chosen to match *that* existing semantics, not to introduce a new one:
//!
//! - **`HeadRoll`: sign is flipped (`osc = −normalized(roll)`).** Working
//!   through `super::Mapper`'s formula (`atan2(dy, dx)` of the right→left
//!   eye-centre line, iBUG-68 "right"/"left" meaning the *subject's* own
//!   eyes): a positive FAN `HeadRoll` comes from the subject's left eye
//!   sitting lower in the image than the right, i.e. the subject tilting
//!   their head toward their own **left**, which appears **clockwise** as
//!   seen by the viewer. [`HeadPose::roll`] is positive for the *opposite*
//!   tilt (subject's right / counter-clockwise-as-viewed), so producing the
//!   same OSC sign as the FAN mapper requires negating it.
//! - **`HeadYaw`: sign is kept as-is (`osc = +normalized(yaw)`).** FAN's
//!   `HeadYaw` is positive when the nose tip offsets toward the image side
//!   holding the subject's left eye — i.e. turning toward the subject's own
//!   **left** — which is exactly [`HeadPose::yaw`]'s own stated positive
//!   direction. No flip needed.
//! - **`HeadPitch`: sign is kept as-is (`osc = +normalized(pitch)`), but
//!   this is a judgement call, not a derivation.** FAN's `HeadPitch` is a
//!   coarse proxy (nose-bridge foreshortening) whose up/down sign isn't
//!   pinned down precisely enough in [`super`]'s own docs to derive a
//!   provably-matching sign here. This module keeps [`HeadPose::pitch`]'s
//!   own stated convention (`+` = looking up) as the least-surprising
//!   choice. **Flip `pitch_max_rad`'s sign (or negate in `map`) if the live
//!   demo shows the avatar's head pitching backwards from user intent.**
//!
//! # Calibration (neutral baseline)
//!
//! Two ways to seed the neutral baseline, mirroring [`super::Mapper::calibrate`]:
//!
//! - **Explicit** — [`ArkitMapper::calibrate`], called with a batch of frames
//!   captured while the subject holds a relaxed, forward-facing expression
//!   (same contract as the pipeline's `--calibrate-frames` flow). Immediately
//!   overwrites the baseline and resets all smoothing state.
//! - **Automatic** — if `calibrate` is never called, [`ArkitMapper::map`]
//!   accumulates a running average baseline over the first
//!   [`ArkitMappingConfig::calib_frames`] calls (AvataCam's `BsCalib`
//!   behaviour), producing usable (if initially noisier) output even without
//!   an explicit calibration step.
//!
//! Blink channels (`eyeBlinkLeft`/`eyeBlinkRight`) never use plain baseline
//! subtraction — see [`BsCalib`] — because a fixed open→shut window doesn't
//! transfer across eyes or people (AvataCam issue #77): each eye is
//! calibrated to its own open-baseline, so open reads `0` and a blink
//! `blink_gain` above that baseline saturates to `1`, independently per eye.
//!
//! # Smoothing
//!
//! Each of the 11 raw channels (and each head axis, after neutral
//! subtraction) is smoothed by its own [`OneEuro`] adaptive low-pass filter —
//! a fixed exponential-smoothing `alpha` would either leave idle jitter
//! visible or dull fast transients (blinks, speech) to unacceptable lag; the
//! One-Euro filter widens its cutoff automatically when the signal is moving
//! fast. Per AvataCam, blink/jaw/mouth channels get the snappier
//! [`ArkitMappingConfig::fast_mincutoff`]; brow channels and head use the
//! slower [`ArkitMappingConfig::expr_mincutoff`] / [`ArkitMappingConfig::head_mincutoff`].
//!
//! [`ArkitFaceFrame`] carries no per-frame timestamp (unlike AvataCam, which
//! smooths against the real capture timestamp), so this module assumes a
//! fixed nominal frame period, [`ArkitMappingConfig::assumed_dt`] (default
//! `1/30`s), between successive [`ArkitMapper::map`] calls. This is a
//! deliberate scope simplification, not a design gap: revisit if the pipeline
//! starts calling `map` at a visibly variable rate (open question for
//! integration — see the final report).

use crate::osc::OscParam;
use crate::tracking::arkit::{ArkitBlendshapes, ArkitFaceFrame, HeadPose};

/// Output parameter names, in a stable emission order — mirrors
/// [`super::PARAM_NAMES`] exactly (duplicated here since that const isn't
/// `pub`; keep the two in sync by hand, same as CLAUDE.md's existing
/// Rust/Unity manual-sync boundary).
const PARAM_NAMES: [&str; 10] = [
    "MouthOpen",
    "EyeBlinkLeft",
    "EyeBlinkRight",
    "BrowUpLeft",
    "BrowUpRight",
    "MouthSmile",
    "MouthWide",
    "HeadRoll",
    "HeadYaw",
    "HeadPitch",
];

/// Per-parameter output range `(min, max)`, aligned with [`PARAM_NAMES`] —
/// mirrors [`super::PARAM_RANGES`] (see the sync note there).
const PARAM_RANGES: [(f32, f32); 10] = [
    (0.0, 1.0),  // MouthOpen
    (0.0, 1.0),  // EyeBlinkLeft
    (0.0, 1.0),  // EyeBlinkRight
    (0.0, 1.0),  // BrowUpLeft
    (0.0, 1.0),  // BrowUpRight
    (0.0, 1.0),  // MouthSmile
    (-1.0, 1.0), // MouthWide
    (-1.0, 1.0), // HeadRoll
    (-1.0, 1.0), // HeadYaw
    (-1.0, 1.0), // HeadPitch
];

/// The 11 raw ARKit blendshape channels this mapper reads, in a fixed local
/// order used by [`BsCalib`] / [`OneEuro`] arrays. Not all 52 — only the ones
/// feeding the mapping table above.
const CHANNEL_NAMES: [&str; NUM_CH] = [
    "jawOpen",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthPucker",
];

const NUM_CH: usize = 11;
const CH_JAW_OPEN: usize = 0;
const CH_BLINK_LEFT: usize = 1;
const CH_BLINK_RIGHT: usize = 2;
const CH_BROW_INNER_UP: usize = 3;
const CH_BROW_OUTER_UP_LEFT: usize = 4;
const CH_BROW_OUTER_UP_RIGHT: usize = 5;
const CH_MOUTH_SMILE_LEFT: usize = 6;
const CH_MOUTH_SMILE_RIGHT: usize = 7;
const CH_MOUTH_STRETCH_LEFT: usize = 8;
const CH_MOUTH_STRETCH_RIGHT: usize = 9;
const CH_MOUTH_PUCKER: usize = 10;

/// Whether channel `idx` gets the snappier `fast_mincutoff` (mouth/jaw/blink
/// transients) instead of the general `expr_mincutoff` (brow). Mirrors
/// AvataCam's `expression::is_fast_channel`: only the brow channels are slow.
fn is_fast_channel(idx: usize) -> bool {
    !matches!(
        idx,
        CH_BROW_INNER_UP | CH_BROW_OUTER_UP_LEFT | CH_BROW_OUTER_UP_RIGHT
    )
}

fn extract_channels(bs: &ArkitBlendshapes) -> [f32; NUM_CH] {
    let mut out = [0.0f32; NUM_CH];
    for (i, name) in CHANNEL_NAMES.iter().enumerate() {
        out[i] = bs.get(name).unwrap_or(0.0);
    }
    out
}

/// Tunable thresholds/gains for the ARKit → OSC mapping. See the module docs
/// for how each is used; defaults are ported from AvataCam
/// (`crates/app/src/expression.rs` / `track.rs`), sources noted per field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArkitMappingConfig {
    /// Frames over which the neutral baseline auto-accumulates (both the 11
    /// blendshape channels and the head-pose neutral) when [`ArkitMapper::calibrate`]
    /// is never called. Source: AvataCam `AVATACAM_FACE_CALIB_FRAMES` default (30).
    pub calib_frames: u32,
    /// Per-eye blink self-calibration gain: a blink this far above that eye's
    /// open baseline reads as fully closed. Source: AvataCam `BLINK_GAIN` (0.35).
    pub blink_gain: f32,
    /// General (brow) One-Euro min-cutoff. Source: AvataCam `EXPR_MINCUTOFF` (1.5).
    pub expr_mincutoff: f32,
    /// One-Euro beta (shared by fast and general expression channels).
    /// Source: AvataCam `EXPR_BETA` (0.5).
    pub expr_beta: f32,
    /// Fast (mouth/jaw/blink) One-Euro min-cutoff, so speech/blink peaks
    /// aren't damped. Source: AvataCam `FAST_MINCUTOFF` (4.0).
    pub fast_mincutoff: f32,
    /// Head-pose One-Euro min-cutoff. Source: AvataCam `AVATACAM_HEAD_MINCUTOFF`
    /// default (1.2).
    pub head_mincutoff: f32,
    /// Head-pose One-Euro beta. Source: AvataCam `AVATACAM_HEAD_BETA` default (0.6).
    pub head_beta: f32,
    /// Roll angle (radians, neutral-relative) mapped to `±1`. No direct
    /// AvataCam equivalent (its head rotation drives a VRM bone directly, not
    /// a normalised `-1..1` OSC axis) — falls back to the range specified for
    /// this module (30°).
    pub roll_max_rad: f32,
    /// Yaw angle (radians, neutral-relative) mapped to `±1`. Fallback default
    /// (45°) — see `roll_max_rad`.
    pub yaw_max_rad: f32,
    /// Pitch angle (radians, neutral-relative) mapped to `±1`. Fallback
    /// default (30°) — see `roll_max_rad`.
    pub pitch_max_rad: f32,
    /// Assumed seconds-per-frame fed to every One-Euro filter, since
    /// [`ArkitFaceFrame`] carries no timestamp (see module docs). Default:
    /// `1/30` (typical webcam rate).
    pub assumed_dt: f32,
}

impl Default for ArkitMappingConfig {
    fn default() -> Self {
        Self {
            calib_frames: 30,
            blink_gain: 0.35,
            expr_mincutoff: 1.5,
            expr_beta: 0.5,
            fast_mincutoff: 4.0,
            head_mincutoff: 1.2,
            head_beta: 0.6,
            roll_max_rad: 30.0_f32.to_radians(),
            yaw_max_rad: 45.0_f32.to_radians(),
            pitch_max_rad: 30.0_f32.to_radians(),
            assumed_dt: 1.0 / 30.0,
        }
    }
}

/// One-Euro low-pass filter (Casiez, Roussel & Vogel 2012) for one scalar
/// signal. Ported from AvataCam `crates/app/src/track.rs` (self-contained
/// math, no new dependency). Adaptive cutoff: smooths hard when the signal is
/// slow (kills jitter), widens to let fast motion through with little lag.
#[derive(Debug, Clone, Copy)]
struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    dcutoff: f32,
    x_prev: Option<f32>,
    dx_prev: f32,
}

impl OneEuro {
    fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            dcutoff: 1.0,
            x_prev: None,
            dx_prev: 0.0,
        }
    }

    fn alpha(cutoff: f32, dt: f32) -> f32 {
        let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
        1.0 / (1.0 + tau / dt)
    }

    fn filter(&mut self, x: f32, dt: f32) -> f32 {
        let Some(xp) = self.x_prev else {
            self.x_prev = Some(x);
            return x;
        };
        let dx = (x - xp) / dt;
        let a_d = Self::alpha(self.dcutoff, dt);
        let dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev;
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let a = Self::alpha(cutoff, dt);
        let x_hat = a * x + (1.0 - a) * xp;
        self.x_prev = Some(x_hat);
        self.dx_prev = dx_hat;
        x_hat
    }
}

/// Per-channel neutral-baseline calibration for the 11 raw ARKit channels.
/// Ported from AvataCam `crates/app/src/track.rs::BsCalib`.
///
/// Emotion/mouth/jaw/brow channels: subtract the accumulated baseline,
/// clamped `≥ 0` (a resting face reads `0`, not a negative "anti-expression").
/// Blink channels ([`CH_BLINK_LEFT`] / [`CH_BLINK_RIGHT`]): **per-eye**
/// self-calibration instead — `(raw − that eye's own open baseline) /
/// blink_gain`, clamped to `0..1` — so each eye independently reads open as
/// `0` and a blink `blink_gain` above its own baseline as a saturated `1`,
/// regardless of the other eye's baseline or gain (AvataCam issue #77: a
/// shared open→shut window left the higher-baseline eye stuck).
#[derive(Debug, Clone, Copy)]
struct BsCalib {
    base: [f32; NUM_CH],
    n: u32,
    frames: u32,
    blink_gain: f32,
}

impl BsCalib {
    fn new(frames: u32, blink_gain: f32) -> Self {
        Self {
            base: [0.0; NUM_CH],
            n: 0,
            frames,
            blink_gain: blink_gain.max(1e-4),
        }
    }

    /// Seed the baseline directly (explicit calibration) and mark it as
    /// already-warmed-up, so `apply` no longer auto-accumulates.
    fn seed(&mut self, avg: [f32; NUM_CH]) {
        self.base = avg;
        self.n = self.frames;
    }

    /// Accumulate the running neutral baseline over the first `frames` calls,
    /// then normalise every channel in place (same rule as AvataCam's
    /// `BsCalib::apply`: accumulate *and* normalise using the
    /// currently-accumulated baseline on every call, so output is usable —
    /// if initially noisier — even mid-calibration).
    fn apply(&mut self, c: &mut [f32; NUM_CH]) {
        if self.n < self.frames {
            let k = self.n as f32;
            for (b, &ci) in self.base.iter_mut().zip(c.iter()) {
                *b = (*b * k + ci) / (k + 1.0);
            }
            self.n += 1;
        }
        for (i, v) in c.iter_mut().enumerate() {
            if i == CH_BLINK_LEFT || i == CH_BLINK_RIGHT {
                *v = ((*v - self.base[i]) / self.blink_gain).clamp(0.0, 1.0);
            } else {
                *v = (*v - self.base[i]).max(0.0);
            }
        }
    }
}

/// Per-axis neutral-baseline calibration for [`HeadPose`] (`roll`, `yaw`,
/// `pitch`, in that fixed order), mirroring [`BsCalib`]'s accumulate-then-seed
/// shape (AvataCam's `track.rs` "Neutral calibration (#28)" pattern, applied
/// here per-axis instead of as a quaternion since [`HeadPose`] is already
/// decomposed).
#[derive(Debug, Clone, Copy)]
struct HeadCalib {
    base: [f32; 3],
    n: u32,
    frames: u32,
}

impl HeadCalib {
    fn new(frames: u32) -> Self {
        Self {
            base: [0.0; 3],
            n: 0,
            frames,
        }
    }

    fn seed(&mut self, avg: [f32; 3]) {
        self.base = avg;
        self.n = self.frames;
    }

    /// Accumulate `h` into the running baseline if still within the
    /// calibration window; always returns the current baseline.
    fn accumulate(&mut self, h: &HeadPose) -> [f32; 3] {
        if self.n < self.frames {
            let k = self.n as f32;
            self.base[0] = (self.base[0] * k + h.roll) / (k + 1.0);
            self.base[1] = (self.base[1] * k + h.yaw) / (k + 1.0);
            self.base[2] = (self.base[2] * k + h.pitch) / (k + 1.0);
            self.n += 1;
        }
        self.base
    }
}

fn build_channel_filters(cfg: &ArkitMappingConfig) -> [OneEuro; NUM_CH] {
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

/// Maps [`ArkitFaceFrame`]s to OSC parameters — the ARKit-blendshape
/// counterpart of [`super::Mapper`]. Holds calibration + smoothing state
/// between frames. See the module docs for the full derivation.
#[derive(Debug, Clone)]
pub struct ArkitMapper {
    config: ArkitMappingConfig,
    calib: BsCalib,
    head_calib: HeadCalib,
    filters: [OneEuro; NUM_CH],
    head_filters: [OneEuro; 3],
}

impl Default for ArkitMapper {
    fn default() -> Self {
        Self::new(ArkitMappingConfig::default())
    }
}

impl ArkitMapper {
    /// A mapper with the given config; neutral baselines start at zero and
    /// auto-accumulate over the first `config.calib_frames` [`map`](Self::map)
    /// calls unless [`calibrate`](Self::calibrate) is called first.
    pub fn new(config: ArkitMappingConfig) -> Self {
        let calib = BsCalib::new(config.calib_frames, config.blink_gain);
        let head_calib = HeadCalib::new(config.calib_frames);
        let filters = build_channel_filters(&config);
        let head_filters = build_head_filters(&config);
        Self {
            config,
            calib,
            head_calib,
            filters,
            head_filters,
        }
    }

    /// Produce the OSC parameter updates for one frame. See the module docs
    /// for the full derivation, sign conventions, and smoothing model.
    pub fn map(&mut self, frame: &ArkitFaceFrame) -> Vec<OscParam> {
        let dt = self.config.assumed_dt.max(1e-4);

        let mut ch = extract_channels(&frame.blendshapes);
        self.calib.apply(&mut ch);
        for (i, v) in ch.iter_mut().enumerate() {
            *v = self.filters[i].filter(*v, dt);
        }

        let mouth_open = ch[CH_JAW_OPEN];
        let blink_left = ch[CH_BLINK_LEFT];
        let blink_right = ch[CH_BLINK_RIGHT];
        let brow_up_left = ch[CH_BROW_INNER_UP].max(ch[CH_BROW_OUTER_UP_LEFT]);
        let brow_up_right = ch[CH_BROW_INNER_UP].max(ch[CH_BROW_OUTER_UP_RIGHT]);
        let mouth_smile = (ch[CH_MOUTH_SMILE_LEFT] + ch[CH_MOUTH_SMILE_RIGHT]) / 2.0;
        let mouth_wide =
            (ch[CH_MOUTH_STRETCH_LEFT] + ch[CH_MOUTH_STRETCH_RIGHT]) / 2.0 - ch[CH_MOUTH_PUCKER];

        let neutral = self.head_calib.accumulate(&frame.head);
        let rel_roll = self.head_filters[0].filter(frame.head.roll - neutral[0], dt);
        let rel_yaw = self.head_filters[1].filter(frame.head.yaw - neutral[1], dt);
        let rel_pitch = self.head_filters[2].filter(frame.head.pitch - neutral[2], dt);

        // Sign conventions — see the module doc section above.
        let head_roll = -(rel_roll / self.config.roll_max_rad);
        let head_yaw = rel_yaw / self.config.yaw_max_rad;
        let head_pitch = rel_pitch / self.config.pitch_max_rad;

        let raw = [
            mouth_open,
            blink_left,
            blink_right,
            brow_up_left,
            brow_up_right,
            mouth_smile,
            mouth_wide,
            head_roll,
            head_yaw,
            head_pitch,
        ];

        if std::env::var_os("DEBUG_MAPPING").is_some() {
            eprintln!("[DEBUG_MAPPING][arkit] raw = {raw:?}");
        }

        let mut out = Vec::with_capacity(PARAM_NAMES.len());
        for i in 0..PARAM_NAMES.len() {
            let (lo, hi) = PARAM_RANGES[i];
            out.push(OscParam::float(PARAM_NAMES[i], raw[i].clamp(lo, hi)));
        }
        out
    }

    /// Re-derive the neutral baseline (11 blendshape channels + 3 head axes)
    /// from a batch of frames captured while the subject holds a relaxed,
    /// forward-facing expression — the ARKit counterpart of
    /// [`super::Mapper::calibrate`]. Overwrites the baseline immediately
    /// (bypassing further auto-accumulation) and resets all smoothing state,
    /// since the neutral point just moved. Does nothing if `frames` is empty.
    pub fn calibrate(&mut self, frames: &[ArkitFaceFrame]) {
        if frames.is_empty() {
            return;
        }
        let n = frames.len() as f32;

        let mut ch_sum = [0.0f32; NUM_CH];
        let mut head_sum = [0.0f32; 3];
        for f in frames {
            let ch = extract_channels(&f.blendshapes);
            for i in 0..NUM_CH {
                ch_sum[i] += ch[i];
            }
            head_sum[0] += f.head.roll;
            head_sum[1] += f.head.yaw;
            head_sum[2] += f.head.pitch;
        }

        let mut ch_avg = [0.0f32; NUM_CH];
        for i in 0..NUM_CH {
            ch_avg[i] = ch_sum[i] / n;
        }
        let head_avg = [head_sum[0] / n, head_sum[1] / n, head_sum[2] / n];

        self.calib.seed(ch_avg);
        self.head_calib.seed(head_avg);
        self.filters = build_channel_filters(&self.config);
        self.head_filters = build_head_filters(&self.config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bs(values: &[(&str, f32)]) -> ArkitBlendshapes {
        let mut a = [0.0f32; crate::tracking::arkit::NUM_BLENDSHAPES];
        for &(name, v) in values {
            let idx = crate::tracking::arkit::ARKIT_BLENDSHAPE_NAMES
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
                ("browOuterUpLeft", 0.15),
                ("browOuterUpRight", 0.15),
                ("mouthSmileLeft", 0.05),
                ("mouthSmileRight", 0.05),
                ("mouthStretchLeft", 0.05),
                ("mouthStretchRight", 0.05),
            ],
            HeadPose::default(),
        )
    }

    fn value_of(out: &[OscParam], name: &str) -> f32 {
        let p = out.iter().find(|p| p.name == name).expect("param present");
        match p.value {
            crate::osc::OscValue::Float(v) => v,
            _ => panic!("expected float param"),
        }
    }

    // 1. Neutral sequence -> after calibration window, all 10 outputs ~ 0.
    #[test]
    fn neutral_sequence_calibrates_all_outputs_to_zero() {
        let cfg = ArkitMappingConfig {
            calib_frames: 10,
            ..Default::default()
        };
        let mut mapper = ArkitMapper::new(cfg);
        let mut last = Vec::new();
        // Run well past the calibration window and let the (fast) filters settle.
        for _ in 0..80 {
            last = mapper.map(&neutral_frame());
        }
        for name in PARAM_NAMES {
            let v = value_of(&last, name);
            assert!(v.abs() < 0.05, "{name} did not settle near 0: {v}");
        }
    }

    // 2. Blink saturates to (approximately) 1.0 once raw jumps blink_gain
    //    above the calibrated open baseline.
    #[test]
    fn blink_saturates_to_one_after_gain_jump() {
        let cfg = ArkitMappingConfig {
            calib_frames: 10,
            blink_gain: 0.35,
            ..Default::default()
        };
        let mut mapper = ArkitMapper::new(cfg);
        let rest = frame(
            &[("eyeBlinkLeft", 0.08), ("eyeBlinkRight", 0.08)],
            HeadPose::default(),
        );
        for _ in 0..10 {
            mapper.map(&rest);
        }
        let blink = frame(
            &[("eyeBlinkLeft", 0.08 + 0.35), ("eyeBlinkRight", 0.08)],
            HeadPose::default(),
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = mapper.map(&blink);
        }
        let l = value_of(&last, "EyeBlinkLeft");
        let r = value_of(&last, "EyeBlinkRight");
        assert!((l - 1.0).abs() < 1e-2, "left blink not saturated: {l}");
        assert!(r < 0.05, "right eye should stay open (rest): {r}");
    }

    // 2b. Per-eye: asymmetric baselines both reach 1.0 on their own blink.
    #[test]
    fn blink_per_eye_asymmetric_baselines_both_saturate() {
        let cfg = ArkitMappingConfig {
            calib_frames: 8,
            blink_gain: 0.3,
            ..Default::default()
        };
        let mut mapper = ArkitMapper::new(cfg);
        // Left eye has a much higher open baseline than right.
        let rest = frame(
            &[("eyeBlinkLeft", 0.55), ("eyeBlinkRight", 0.15)],
            HeadPose::default(),
        );
        for _ in 0..8 {
            mapper.map(&rest);
        }
        let at_rest = mapper.map(&rest);
        let l0 = value_of(&at_rest, "EyeBlinkLeft");
        let r0 = value_of(&at_rest, "EyeBlinkRight");
        assert!(l0 < 0.05 && r0 < 0.05, "rest not 0: l={l0} r={r0}");

        let both_shut = frame(
            &[("eyeBlinkLeft", 0.55 + 0.3), ("eyeBlinkRight", 0.15 + 0.3)],
            HeadPose::default(),
        );
        let mut last = Vec::new();
        for _ in 0..60 {
            last = mapper.map(&both_shut);
        }
        let l = value_of(&last, "EyeBlinkLeft");
        let r = value_of(&last, "EyeBlinkRight");
        assert!((l - 1.0).abs() < 1e-2, "left not saturated: {l}");
        assert!((r - 1.0).abs() < 1e-2, "right not saturated: {r}");
    }

    // 3. MouthOpen: baseline at rest -> 0; large raw -> large value; never negative.
    #[test]
    fn mouth_open_calibrated_baseline_and_never_negative() {
        let cfg = ArkitMappingConfig {
            calib_frames: 10,
            ..Default::default()
        };
        let mut mapper = ArkitMapper::new(cfg);
        let rest = frame(&[("jawOpen", 0.22)], HeadPose::default());
        let mut last = Vec::new();
        for _ in 0..40 {
            last = mapper.map(&rest);
        }
        let at_rest = value_of(&last, "MouthOpen");
        assert!(at_rest.abs() < 0.05, "rest MouthOpen not ~0: {at_rest}");

        let open = frame(&[("jawOpen", 0.9)], HeadPose::default());
        let mut last = Vec::new();
        for _ in 0..40 {
            last = mapper.map(&open);
            assert!(
                value_of(&last, "MouthOpen") >= 0.0,
                "MouthOpen went negative"
            );
        }
        let big = value_of(&last, "MouthOpen");
        assert!(
            big > 0.3,
            "MouthOpen should be large when jaw is open: {big}"
        );
    }

    // 4. Head: neutral pitch held during calibration -> HeadPitch settles to 0;
    //    +max_angle above neutral -> +1 saturation; -max_angle -> -1.
    #[test]
    fn head_pitch_neutral_calibration_and_saturation() {
        let cfg = ArkitMappingConfig {
            calib_frames: 10,
            pitch_max_rad: 30.0_f32.to_radians(),
            ..Default::default()
        };
        let mut mapper = ArkitMapper::new(cfg);
        let neutral_pitch = -0.05;
        let rest = ArkitFaceFrame {
            blendshapes: bs(&[]),
            head: HeadPose {
                roll: 0.0,
                yaw: 0.0,
                pitch: neutral_pitch,
            },
        };
        let mut last = Vec::new();
        for _ in 0..40 {
            last = mapper.map(&rest);
        }
        let at_rest = value_of(&last, "HeadPitch");
        assert!(at_rest.abs() < 0.05, "HeadPitch not ~0 at rest: {at_rest}");

        let up = ArkitFaceFrame {
            blendshapes: bs(&[]),
            head: HeadPose {
                roll: 0.0,
                yaw: 0.0,
                pitch: neutral_pitch + 30.0_f32.to_radians(),
            },
        };
        let mut last = Vec::new();
        for _ in 0..60 {
            last = mapper.map(&up);
        }
        let hi = value_of(&last, "HeadPitch");
        assert!((hi - 1.0).abs() < 1e-2, "HeadPitch not +1 saturated: {hi}");

        let down = ArkitFaceFrame {
            blendshapes: bs(&[]),
            head: HeadPose {
                roll: 0.0,
                yaw: 0.0,
                pitch: neutral_pitch - 30.0_f32.to_radians(),
            },
        };
        let mut last = Vec::new();
        for _ in 0..60 {
            last = mapper.map(&down);
        }
        let lo = value_of(&last, "HeadPitch");
        assert!((lo + 1.0).abs() < 1e-2, "HeadPitch not -1 saturated: {lo}");
    }

    // 4b. HeadRoll sign is flipped relative to HeadPose::roll (see module docs):
    // a positive (subject's-right) HeadPose roll saturates HeadRoll to -1.
    #[test]
    fn head_roll_sign_is_flipped_from_head_pose() {
        let cfg = ArkitMappingConfig {
            calib_frames: 4,
            roll_max_rad: 30.0_f32.to_radians(),
            ..Default::default()
        };
        let mut mapper = ArkitMapper::new(cfg);
        let neutral = ArkitFaceFrame {
            blendshapes: bs(&[]),
            head: HeadPose::default(),
        };
        for _ in 0..4 {
            mapper.map(&neutral);
        }
        let tilted = ArkitFaceFrame {
            blendshapes: bs(&[]),
            head: HeadPose {
                roll: 30.0_f32.to_radians(),
                yaw: 0.0,
                pitch: 0.0,
            },
        };
        let mut last = Vec::new();
        for _ in 0..60 {
            last = mapper.map(&tilted);
        }
        let roll = value_of(&last, "HeadRoll");
        assert!(
            (roll + 1.0).abs() < 1e-2,
            "HeadRoll sign not flipped: {roll}"
        );
    }

    // 4c. HeadYaw sign is kept as-is relative to HeadPose::yaw.
    #[test]
    fn head_yaw_sign_matches_head_pose() {
        let cfg = ArkitMappingConfig {
            calib_frames: 4,
            yaw_max_rad: 45.0_f32.to_radians(),
            ..Default::default()
        };
        let mut mapper = ArkitMapper::new(cfg);
        let neutral = ArkitFaceFrame {
            blendshapes: bs(&[]),
            head: HeadPose::default(),
        };
        for _ in 0..4 {
            mapper.map(&neutral);
        }
        let turned = ArkitFaceFrame {
            blendshapes: bs(&[]),
            head: HeadPose {
                roll: 0.0,
                yaw: 45.0_f32.to_radians(),
                pitch: 0.0,
            },
        };
        let mut last = Vec::new();
        for _ in 0..60 {
            last = mapper.map(&turned);
        }
        let yaw = value_of(&last, "HeadYaw");
        assert!(
            (yaw - 1.0).abs() < 1e-2,
            "HeadYaw sign flipped unexpectedly: {yaw}"
        );
    }

    // 5a. Smoothing: noisy input has strictly lower output variance.
    #[test]
    fn smoothing_reduces_noise_variance() {
        let mut mapper = ArkitMapper::new(ArkitMappingConfig {
            calib_frames: 0,
            ..Default::default()
        });
        let mut lcg = Lcg::new(42);
        let mut outs = Vec::new();
        for _ in 0..120 {
            let noise = lcg.next_f32() * 0.2 - 0.1; // +/- 0.1 around 0.5
            let f = frame(&[("browInnerUp", 0.5 + noise)], HeadPose::default());
            let out = mapper.map(&f);
            outs.push(value_of(&out, "BrowUpLeft"));
        }
        let input_var = {
            let mut lcg2 = Lcg::new(42);
            let vals: Vec<f32> = (0..120).map(|_| lcg2.next_f32() * 0.2 - 0.1).collect();
            variance(&vals)
        };
        let output_var = variance(&outs[20..]); // after warm-up
        assert!(
            output_var < input_var,
            "output variance ({output_var}) not lower than input ({input_var})"
        );
    }

    // 5b. A step input on a fast channel (blink) reaches >0.9 of target within
    // a few frames (not over-smoothed).
    #[test]
    fn fast_channel_step_response_is_snappy() {
        let mut mapper = ArkitMapper::new(ArkitMappingConfig {
            calib_frames: 1,
            blink_gain: 0.35,
            ..Default::default()
        });
        let open = frame(&[("eyeBlinkLeft", 0.0)], HeadPose::default());
        mapper.map(&open); // seed baseline at 0
        for _ in 0..5 {
            mapper.map(&open);
        }
        let shut = frame(&[("eyeBlinkLeft", 0.35)], HeadPose::default());
        let mut reached = None;
        for i in 0..10 {
            let out = mapper.map(&shut);
            let v = value_of(&out, "EyeBlinkLeft");
            if v > 0.9 {
                reached = Some(i);
                break;
            }
        }
        assert!(
            reached.is_some(),
            "fast blink channel did not reach 0.9 of target within 10 frames"
        );
        assert!(
            reached.unwrap() < 8,
            "step response too slow: frame {:?}",
            reached
        );
    }

    // 6. All outputs stay within PARAM_RANGES for random inputs (seeded LCG fuzz).
    #[test]
    fn outputs_stay_within_param_ranges_for_fuzzed_input() {
        let mut mapper = ArkitMapper::new(ArkitMappingConfig::default());
        let mut lcg = Lcg::new(1234);
        for _ in 0..500 {
            let mut vals: Vec<(&str, f32)> = Vec::new();
            for name in CHANNEL_NAMES {
                vals.push((name, lcg.next_f32() * 2.0)); // allow out-of-[0,1] noise too
            }
            let head = HeadPose {
                roll: (lcg.next_f32() - 0.5) * 4.0,
                yaw: (lcg.next_f32() - 0.5) * 4.0,
                pitch: (lcg.next_f32() - 0.5) * 4.0,
            };
            let f = frame(&vals, head);
            let out = mapper.map(&f);
            for (i, name) in PARAM_NAMES.iter().enumerate() {
                let (lo, hi) = PARAM_RANGES[i];
                let v = value_of(&out, name);
                assert!(
                    v >= lo - 1e-4 && v <= hi + 1e-4,
                    "{name} out of range: {v} (expected {lo}..{hi})"
                );
            }
        }
    }

    // Explicit calibrate() entry point seeds the baseline immediately from a
    // batch of samples, without needing to wait through map()'s auto-window.
    #[test]
    fn explicit_calibrate_seeds_baseline_from_samples() {
        let mut mapper = ArkitMapper::new(ArkitMappingConfig {
            calib_frames: 30, // large window; explicit calibrate should bypass it
            ..Default::default()
        });
        let samples: Vec<ArkitFaceFrame> = (0..5)
            .map(|_| {
                frame(
                    &[("jawOpen", 0.22), ("mouthPucker", 0.3)],
                    HeadPose {
                        roll: 0.0,
                        yaw: 0.0,
                        pitch: -0.05,
                    },
                )
            })
            .collect();
        mapper.calibrate(&samples);

        // Immediately (no further warm-up needed) a repeat of the neutral
        // sample should read close to 0, in just a few frames of smoothing.
        let neutral = frame(
            &[("jawOpen", 0.22), ("mouthPucker", 0.3)],
            HeadPose {
                roll: 0.0,
                yaw: 0.0,
                pitch: -0.05,
            },
        );
        let mut last = Vec::new();
        for _ in 0..20 {
            last = mapper.map(&neutral);
        }
        let mo = value_of(&last, "MouthOpen");
        let pitch = value_of(&last, "HeadPitch");
        assert!(
            mo.abs() < 0.05,
            "MouthOpen not ~0 after explicit calibrate: {mo}"
        );
        assert!(
            pitch.abs() < 0.05,
            "HeadPitch not ~0 after explicit calibrate: {pitch}"
        );
    }

    fn variance(v: &[f32]) -> f32 {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32
    }

    /// Minimal seeded linear-congruential generator for deterministic fuzz
    /// tests (no new dependency).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes LCG constants.
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 33) as u32
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32) / (u32::MAX as f32)
        }
    }
}
