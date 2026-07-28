//! Shared machinery for the ARKit-blendshape → OSC mappers (issues #17/#18).
//!
//! Until issue #21 this module also held `ArkitMapper`, the 10-parameter
//! `custom10` mapper. That profile is retired — [`super::unified`] (the
//! VRCFT Unified Expressions mapper) is the only avatar-parameter mapper —
//! and what remains here is the calibration/smoothing machinery both it and
//! [`super::eye`] share, ported from AvataCam:
//!
//! - [`ArkitMappingConfig`] — tuning constants (calibration window, One-Euro
//!   cutoffs, head ranges, output deadzone), sources noted per field.
//! - [`OneEuro`] — adaptive low-pass filter (Casiez, Roussel & Vogel 2012):
//!   smooths hard when the signal is slow (kills idle jitter), widens its
//!   cutoff when it moves fast (blinks and speech come through with little
//!   lag). A fixed exponential-smoothing `alpha` can't do both.
//! - [`HeadCalib`] — per-axis neutral-baseline calibration for
//!   [`HeadPose`] (accumulate-then-seed, AvataCam's "Neutral calibration
//!   (#28)" pattern applied per-axis).
//! - [`deadzone`] — continuous-rescale output deadzone, so calibration
//!   residue doesn't keep a partial expression permanently on the avatar.
//!
//! MediaPipe's Blendshape V2 coefficients have a non-zero, per-person,
//! per-channel resting baseline (a relaxed face reads as a permanent pucker
//! or scowl), and its blink coefficient does not reliably saturate to a
//! shared `0/1` window across eyes or people — which is why every consumer
//! of this machinery subtracts a calibrated neutral first. See
//! [`super::unified`]'s module docs for the full pipeline.
//!
//! [`ArkitFaceFrame`] carries no per-frame timestamp (unlike AvataCam, which
//! smooths against the real capture timestamp), so consumers assume a fixed
//! nominal frame period, [`ArkitMappingConfig::assumed_dt`] (default
//! `1/30`s), between successive `map` calls. This is a deliberate scope
//! simplification, not a design gap: revisit if the pipeline starts calling
//! `map` at a visibly variable rate.
//!
//! [`ArkitFaceFrame`]: crate::tracking::arkit::ArkitFaceFrame
//! [`HeadPose`]: crate::tracking::arkit::HeadPose

use crate::tracking::arkit::HeadPose;

/// Tunable thresholds/gains for the ARKit → OSC mapping. Defaults are ported
/// from AvataCam (`crates/app/src/expression.rs` / `track.rs`), sources noted
/// per field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArkitMappingConfig {
    /// Frames over which the neutral baseline auto-accumulates (blendshape
    /// channels and the head-pose neutral) when explicit calibration is
    /// never run. Source: AvataCam `AVATACAM_FACE_CALIB_FRAMES` default (30).
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
    /// [`crate::tracking::arkit::ArkitFaceFrame`] carries no timestamp (see
    /// module docs). Default: `1/30` (typical webcam rate).
    pub assumed_dt: f32,
    /// Output deadzone applied to every final parameter (issue #16 live
    /// test): baseline calibration is only as good as the subject's
    /// stillness during the window, and a few hundredths of residual on
    /// e.g. brow/smile channels keeps a partial expression permanently on
    /// the avatar ("worried eyebrows"), while residual head values keep
    /// nudging muscle layers and exciting PhysBones. Values inside the
    /// deadzone read exactly 0; above it the output rescales continuously
    /// to preserve full range.
    pub deadzone: f32,
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
            deadzone: 0.06,
        }
    }
}

/// One-Euro low-pass filter (Casiez, Roussel & Vogel 2012) for one scalar
/// signal. Ported from AvataCam `crates/app/src/track.rs` (self-contained
/// math, no new dependency). Adaptive cutoff: smooths hard when the signal is
/// slow (kills jitter), widens to let fast motion through with little lag.
/// `pub(crate)`: shared by [`super::unified`] and [`super::eye`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    dcutoff: f32,
    x_prev: Option<f32>,
    dx_prev: f32,
}

impl OneEuro {
    pub(crate) fn new(min_cutoff: f32, beta: f32) -> Self {
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

    pub(crate) fn filter(&mut self, x: f32, dt: f32) -> f32 {
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

/// Per-axis neutral-baseline calibration for [`HeadPose`] (`roll`, `yaw`,
/// `pitch`, in that fixed order): accumulate a running average over the first
/// `frames` calls, or [`seed`](Self::seed) it directly from an explicit
/// calibration batch. `pub(crate)`: shared with [`super::unified`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeadCalib {
    base: [f32; 3],
    n: u32,
    frames: u32,
}

impl HeadCalib {
    pub(crate) fn new(frames: u32) -> Self {
        Self {
            base: [0.0; 3],
            n: 0,
            frames,
        }
    }

    pub(crate) fn seed(&mut self, avg: [f32; 3]) {
        self.base = avg;
        self.n = self.frames;
    }

    /// Accumulate `h` into the running baseline if still within the
    /// calibration window; always returns the current baseline.
    pub(crate) fn accumulate(&mut self, h: &HeadPose) -> [f32; 3] {
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

/// Deadzone with continuous rescale: `|v| <= dz` reads exactly 0, and the
/// remaining span rescales so the output still reaches ±1 at input ±1 (no
/// dead range at the top, no step at the threshold).
/// `pub(crate)`: shared by [`super::unified`] and [`super::eye`].
pub(crate) fn deadzone(v: f32, dz: f32) -> f32 {
    if v.abs() <= dz {
        0.0
    } else {
        (v.abs() - dz) / (1.0 - dz) * v.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_zeroes_inside_and_rescales_outside() {
        assert_eq!(deadzone(0.03, 0.06), 0.0);
        assert_eq!(deadzone(-0.06, 0.06), 0.0);
        // Continuous at the threshold, full range at the extremes.
        assert!(deadzone(0.0601, 0.06).abs() < 1e-3);
        assert!((deadzone(1.0, 0.06) - 1.0).abs() < 1e-6);
        assert!((deadzone(-1.0, 0.06) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn one_euro_first_sample_passes_through_then_smooths() {
        let mut f = OneEuro::new(1.5, 0.5);
        assert_eq!(f.filter(0.4, 1.0 / 30.0), 0.4);
        // A step lands strictly between the previous output and the target.
        let next = f.filter(1.0, 1.0 / 30.0);
        assert!(next > 0.4 && next < 1.0, "not smoothed: {next}");
    }

    #[test]
    fn head_calib_accumulates_then_seeds() {
        let mut c = HeadCalib::new(2);
        let pose = |r: f32| HeadPose {
            roll: r,
            yaw: 0.0,
            pitch: 0.0,
        };
        assert_eq!(c.accumulate(&pose(0.2))[0], 0.2);
        assert!((c.accumulate(&pose(0.4))[0] - 0.3).abs() < 1e-6);
        // Window exhausted: baseline frozen.
        assert!((c.accumulate(&pose(10.0))[0] - 0.3).abs() < 1e-6);
        c.seed([1.0, 2.0, 3.0]);
        assert_eq!(c.accumulate(&pose(0.0)), [1.0, 2.0, 3.0]);
    }
}
