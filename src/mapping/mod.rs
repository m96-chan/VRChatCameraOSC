//! Landmarks → VRChat avatar OSC parameters (issue #5).
//!
//! Turns iBUG-68 facial landmarks (see [`crate::tracking`]) into a small set of
//! normalised avatar parameters, then clamps and exponentially smooths them
//! before emitting them as [`OscParam`] floats. VRChat renders the avatar; this
//! module only decides *what* to send.
//!
//! # Parameter design (rule 8 — designed before wiring)
//!
//! Every parameter is derived from **ratios of landmark distances**, never raw
//! pixels, so the mapping is invariant to how large the face appears in frame.
//! The face-scale reference is the **inter-ocular distance (IOD)** — the
//! distance between the two eye centres, where an eye centre is the mean of that
//! eye's six landmarks. Angles are likewise scale-free.
//!
//! | Parameter      | Range   | Source landmarks                     | Formula (summary) |
//! |----------------|---------|--------------------------------------|-------------------|
//! | `MouthOpen`    | `0..1`  | inner lips 62↔66, mouth width 60↔64  | `gap/width`, rescaled by [`MappingConfig::mouth_open_max`] |
//! | `EyeBlinkRight`| `0..1`  | right eye 36–41                      | `1 − EAR/ear_open` (1 = closed) |
//! | `EyeBlinkLeft` | `0..1`  | left eye 42–47                       | `1 − EAR/ear_open` (1 = closed) |
//! | `BrowUpRight`  | `0..1`  | right brow 17–21 vs right eye centre | `(d/IOD − brow_neutral)/brow_span` |
//! | `BrowUpLeft`   | `0..1`  | left brow 22–26 vs left eye centre   | `(d/IOD − brow_neutral)/brow_span` |
//! | `MouthSmile`   | `0..1`  | corners 48,54 vs mouth centre 51,57  | corner vertical lift / IOD, rescaled by [`MappingConfig::mouth_smile_span`] |
//! | `MouthWide`    | `-1..1` | corners 48↔54                        | `(width/IOD − mouth_width_neutral)/mouth_width_span`; + stretched, − puckered |
//! | `HeadRoll`     | `-1..1` | eye centres (36–41 vs 42–47)         | `atan2(dy,dx)/roll_max_rad` |
//! | `HeadYaw`      | `-1..1` | nose tip 30 vs eye midpoint          | `(tip.x − midx)/(IOD·yaw_half_span)` |
//! | `HeadPitch`    | `-1..1` | nose bridge 27→30 vs IOD             | `(len/IOD − pitch_neutral)/pitch_span` |
//!
//! `MouthSmile` and `MouthWide` are deliberately distinct signals from the same
//! two corner landmarks: smile is the *vertical* lift of the corners relative to
//! the mouth centre (a curved smile), width is the *horizontal* corner distance
//! (stretch vs. pucker) — geometrically orthogonal, so a wide-but-flat mouth and
//! a small closed-lip smile read differently.
//!
//! **Eye-aspect-ratio (EAR).** Using the standard six-point EAR on each eye,
//! `EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)`. For the right eye the points are
//! `p1..p6 = 36,37,38,39,40,41`; for the left eye `42,43,44,45,46,47`. EAR is
//! large when the eye is open and → 0 when the lids meet, so blink is defined as
//! `1 − clamp(EAR/ear_open, 0, 1)` (1 = fully closed).
//!
//! **Head pose** values are deliberately simple, clearly-documented
//! approximations of true 3-D pose (this is a single ordinary webcam, no depth):
//! - *Roll* is the tilt angle of the right→left eye-centre line. `dy>0` (left
//!   eye lower in the image than the right) yields positive roll.
//! - *Yaw* is the horizontal offset of the nose tip from the midpoint of the two
//!   eye centres, normalised so the tip reaching an eye centre saturates ±1.
//!   Positive = tip toward the left-eye side of the image.
//! - *Pitch* is a proxy from the projected nose-bridge length (27→30) relative
//!   to IOD: the bridge foreshortens as the head pitches away from neutral.
//!   Positive = longer-than-neutral bridge.
//!
//! # Clamping & smoothing
//!
//! Each raw value is clamped to its range, then exponentially smoothed against
//! the previous emitted value:
//!
//! ```text
//! smoothed = prev + alpha · (raw − prev)
//! ```
//!
//! `alpha` is [`MappingConfig::smoothing`] (`0 < alpha ≤ 1`); `alpha = 1` (the
//! default from [`Mapper::new`]) disables smoothing. Previous values start at
//! `0`, so the first frame of a step input lands between `0` and the target and
//! successive identical frames converge toward it. Because `prev` and every raw
//! value lie inside the parameter range, the smoothed output does too.
//!
//! # Neutral-pose calibration (issue #15)
//!
//! The "neutral" reference ratios above (`brow_neutral_*`, `mouth_smile_neutral`,
//! `mouth_width_neutral`, `pitch_neutral`, `ear_open`) default to values
//! calibrated against a synthetic test face; real faces/cameras sit at a
//! different baseline, which left some parameters permanently clamped to `0`
//! despite genuinely moving. [`Mapper::calibrate`] re-derives these from a
//! batch of real landmark samples captured while the subject holds a relaxed,
//! forward-facing expression — see its doc for details. `mouth_open_max` is a
//! *span* (how open counts as "fully open"), not a neutral baseline, and is
//! deliberately left untouched by calibration.

use crate::osc::OscParam;
use crate::tracking::{FaceLandmarks, Landmark};

pub mod arkit;
pub mod eye;
pub mod unified;

/// Common interface over the mappers that consume ARKit-style frames
/// ([`crate::tracking::arkit::ArkitFaceFrame`]), so the pipeline can hold
/// either the 10-parameter [`arkit::ArkitMapper`] or the VRCFT-compatible
/// [`unified::UnifiedMapper`] (issue #18) behind one trait object.
pub trait ArkitOscMapper {
    /// Produce the OSC parameter updates for one frame.
    fn map(&mut self, frame: &crate::tracking::arkit::ArkitFaceFrame) -> Vec<OscParam>;
    /// Re-derive the neutral baselines from relaxed forward-facing samples.
    fn calibrate(&mut self, frames: &[crate::tracking::arkit::ArkitFaceFrame]);
}

/// Output parameter names, in a stable emission order.
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

/// Per-parameter output range `(min, max)`, aligned with [`PARAM_NAMES`].
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

/// Tunable thresholds for the landmark → parameter formulas.
///
/// Defaults are chosen so a relaxed, forward-facing neutral face maps to the
/// resting value of every parameter (0 for the `0..1` expressions, 0 for the
/// signed head-pose axes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MappingConfig {
    /// Exponential-smoothing factor `alpha` in `(0, 1]`. `1.0` = no smoothing.
    pub smoothing: f32,
    /// `gap/width` ratio treated as a fully-open mouth (`MouthOpen = 1`).
    pub mouth_open_max: f32,
    /// Eye-aspect-ratio of a fully-open eye (`EyeBlink = 0`).
    pub ear_open: f32,
    /// Neutral right-brow-to-eye distance as a fraction of IOD (`BrowUpRight = 0`).
    pub brow_neutral_right: f32,
    /// Neutral left-brow-to-eye distance as a fraction of IOD (`BrowUpLeft = 0`).
    pub brow_neutral_left: f32,
    /// Extra brow-to-eye/IOD fraction that spans neutral → fully raised.
    pub brow_span: f32,
    /// Neutral corner-lift/IOD ratio (`MouthSmile = 0`).
    pub mouth_smile_neutral: f32,
    /// Corner-lift/IOD fraction that spans neutral → a full smile (`MouthSmile = 1`).
    pub mouth_smile_span: f32,
    /// Neutral mouth-corner-width/IOD ratio (`MouthWide = 0`).
    pub mouth_width_neutral: f32,
    /// Corner-width/IOD fraction spanning neutral → `±1` (stretched/puckered).
    pub mouth_width_span: f32,
    /// Roll angle (radians) mapped to `±1`.
    pub roll_max_rad: f32,
    /// Nose-tip horizontal offset that saturates yaw, as a fraction of IOD.
    pub yaw_half_span: f32,
    /// Neutral nose-bridge length as a fraction of IOD (`HeadPitch = 0`).
    pub pitch_neutral: f32,
    /// Bridge-length/IOD fraction spanning neutral → `±1` pitch.
    pub pitch_span: f32,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self {
            smoothing: 1.0,
            mouth_open_max: 0.5,
            ear_open: 0.3,
            brow_neutral_right: 0.5,
            brow_neutral_left: 0.5,
            brow_span: 0.2,
            mouth_smile_neutral: 0.0,
            mouth_smile_span: 0.15,
            mouth_width_neutral: 0.6,
            mouth_width_span: 0.3,
            roll_max_rad: 0.6,
            yaw_half_span: 0.5,
            pitch_neutral: 0.45,
            pitch_span: 0.2,
        }
    }
}

/// Maps landmarks to OSC parameters. Holds smoothing state between frames.
#[derive(Debug, Clone)]
pub struct Mapper {
    config: MappingConfig,
    /// Previously emitted (smoothed) value per parameter; starts at all-zero.
    prev: [f32; 10],
}

impl Default for Mapper {
    fn default() -> Self {
        Self::new()
    }
}

impl Mapper {
    /// A mapper with default thresholds and **no** smoothing (`alpha = 1`).
    pub fn new() -> Self {
        Self::with_config(MappingConfig::default())
    }

    /// A mapper with the default thresholds but a custom smoothing factor.
    ///
    /// `alpha` is clamped to `(0, 1]`; smaller values smooth (lag) more.
    pub fn with_smoothing(alpha: f32) -> Self {
        Self::with_config(MappingConfig {
            smoothing: alpha,
            ..MappingConfig::default()
        })
    }

    /// A mapper with a fully custom [`MappingConfig`].
    pub fn with_config(config: MappingConfig) -> Self {
        Self {
            config,
            prev: [0.0; 10],
        }
    }

    /// Produce the OSC parameter updates for one frame of landmarks.
    pub fn map(&mut self, landmarks: &FaceLandmarks) -> Vec<OscParam> {
        let raw = self.raw_params(&landmarks.points);
        if std::env::var_os("DEBUG_MAPPING").is_some() {
            eprintln!("[DEBUG_MAPPING] raw = {raw:?}");
        }
        let alpha = self.config.smoothing.clamp(f32::MIN_POSITIVE, 1.0);

        let mut out = Vec::with_capacity(PARAM_NAMES.len());
        for i in 0..PARAM_NAMES.len() {
            let (lo, hi) = PARAM_RANGES[i];
            let clamped = raw[i].clamp(lo, hi);
            let smoothed = self.prev[i] + alpha * (clamped - self.prev[i]);
            self.prev[i] = smoothed;
            out.push(OscParam::float(PARAM_NAMES[i], smoothed));
        }
        out
    }

    /// Compute the raw (unsmoothed, unclamped) parameter values from landmarks.
    fn raw_params(&self, p: &[Landmark]) -> [f32; 10] {
        let cfg = &self.config;

        let right_eye = centroid(p, &[36, 37, 38, 39, 40, 41]);
        let left_eye = centroid(p, &[42, 43, 44, 45, 46, 47]);
        let iod = dist(right_eye, left_eye);

        // Degenerate geometry (no detectable face scale): emit resting values.
        if iod < EPS {
            return [0.0; 10];
        }

        // MouthOpen: inner-lip gap over mouth width. Self-normalising (a
        // closed mouth has ~zero gap regardless of who's wearing it), so this
        // is the one ratio calibration deliberately leaves alone.
        let mouth_width = dist(pt(p, 60), pt(p, 64));
        let mouth_open = if mouth_width < EPS {
            0.0
        } else {
            let gap = dist(pt(p, 62), pt(p, 66));
            (gap / mouth_width) / cfg.mouth_open_max
        };

        let g = GeometryRatios::measure(p, right_eye, left_eye, iod);

        let blink_right = blink(g.ear_right, cfg.ear_open);
        let blink_left = blink(g.ear_left, cfg.ear_open);
        let brow_up_right = (g.brow_ratio_right - cfg.brow_neutral_right) / cfg.brow_span;
        let brow_up_left = (g.brow_ratio_left - cfg.brow_neutral_left) / cfg.brow_span;
        let mouth_smile =
            (g.mouth_smile_lift_ratio - cfg.mouth_smile_neutral) / cfg.mouth_smile_span;
        let mouth_wide = (g.mouth_width_ratio - cfg.mouth_width_neutral) / cfg.mouth_width_span;

        // HeadRoll: tilt angle of the right→left eye-centre line.
        let dx = left_eye.0 - right_eye.0;
        let dy = left_eye.1 - right_eye.1;
        let head_roll = dy.atan2(dx) / cfg.roll_max_rad;

        // HeadYaw: nose-tip horizontal offset from the eye midpoint.
        let mid_x = (right_eye.0 + left_eye.0) / 2.0;
        let head_yaw = (pt(p, 30).0 - mid_x) / (iod * cfg.yaw_half_span);

        let head_pitch = (g.pitch_ratio - cfg.pitch_neutral) / cfg.pitch_span;

        [
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
        ]
    }

    /// Re-derive the neutral-pose baselines from a batch of real landmark
    /// samples, captured while the subject holds a relaxed, forward-facing
    /// expression (issue #15).
    ///
    /// The defaults in [`MappingConfig`] are calibrated against a synthetic
    /// test face; real faces and camera angles sit at a different baseline
    /// ratio, which can leave a parameter permanently clamped at one end of
    /// its range even while it's genuinely responding to expression changes.
    /// This averages the *raw geometry ratios* (the same ones [`Mapper::map`]
    /// subtracts a neutral from) across every sample with a detectable face
    /// and overwrites `ear_open`, `brow_neutral_left`/`brow_neutral_right`,
    /// `mouth_smile_neutral`, `mouth_width_neutral`, and `pitch_neutral`.
    /// `mouth_open_max` is a span, not a baseline, and is left untouched.
    ///
    /// Samples with no detectable face scale are skipped; if none are usable
    /// this does nothing (existing config, whatever it was, is kept). Resets
    /// the smoothing state, since the neutral point just moved.
    pub fn calibrate(&mut self, samples: &[FaceLandmarks]) {
        let (mut sum, mut n) = (GeometryRatios::zero(), 0u32);
        for lm in samples {
            let p = &lm.points;
            let right_eye = centroid(p, &[36, 37, 38, 39, 40, 41]);
            let left_eye = centroid(p, &[42, 43, 44, 45, 46, 47]);
            let iod = dist(right_eye, left_eye);
            if iod < EPS {
                continue;
            }
            sum = sum + GeometryRatios::measure(p, right_eye, left_eye, iod);
            n += 1;
        }
        if n == 0 {
            return;
        }
        let avg = sum / n as f32;
        self.config.ear_open = (avg.ear_right + avg.ear_left) / 2.0;
        self.config.brow_neutral_right = avg.brow_ratio_right;
        self.config.brow_neutral_left = avg.brow_ratio_left;
        self.config.mouth_smile_neutral = avg.mouth_smile_lift_ratio;
        self.config.mouth_width_neutral = avg.mouth_width_ratio;
        self.config.pitch_neutral = avg.pitch_ratio;
        self.prev = [0.0; 10];
    }
}

/// Raw (pre-neutral-subtraction) geometry ratios, shared by `raw_params`
/// (which subtracts the configured neutral) and `calibrate` (which averages
/// these directly to *become* the new neutral) so the two stay in sync.
#[derive(Debug, Clone, Copy)]
struct GeometryRatios {
    ear_right: f32,
    ear_left: f32,
    brow_ratio_right: f32,
    brow_ratio_left: f32,
    mouth_smile_lift_ratio: f32,
    mouth_width_ratio: f32,
    pitch_ratio: f32,
}

impl GeometryRatios {
    fn zero() -> Self {
        Self {
            ear_right: 0.0,
            ear_left: 0.0,
            brow_ratio_right: 0.0,
            brow_ratio_left: 0.0,
            mouth_smile_lift_ratio: 0.0,
            mouth_width_ratio: 0.0,
            pitch_ratio: 0.0,
        }
    }

    /// Measure every neutral-relative ratio from one frame's landmarks.
    /// `iod` must already be checked non-degenerate (`>= EPS`) by the caller.
    fn measure(p: &[Landmark], right_eye: (f32, f32), left_eye: (f32, f32), iod: f32) -> Self {
        let ear_right = ear(p, [36, 37, 38, 39, 40, 41]);
        let ear_left = ear(p, [42, 43, 44, 45, 46, 47]);

        let brow_y_right = mean_y(p, 17..=21);
        let brow_y_left = mean_y(p, 22..=26);
        let brow_ratio_right = (right_eye.1 - brow_y_right) / iod;
        let brow_ratio_left = (left_eye.1 - brow_y_left) / iod;

        // Vertical lift of corners (48,54) above the mouth centre (51,57), and
        // their horizontal distance — see the module doc for why these two
        // are deliberately separate signals.
        let mouth_center_y = (pt(p, 51).1 + pt(p, 57).1) / 2.0;
        let corner_avg_y = (pt(p, 48).1 + pt(p, 54).1) / 2.0;
        let mouth_smile_lift_ratio = (mouth_center_y - corner_avg_y) / iod;
        let mouth_width_ratio = dist(pt(p, 48), pt(p, 54)) / iod;

        let pitch_ratio = dist(pt(p, 27), pt(p, 30)) / iod;

        Self {
            ear_right,
            ear_left,
            brow_ratio_right,
            brow_ratio_left,
            mouth_smile_lift_ratio,
            mouth_width_ratio,
            pitch_ratio,
        }
    }
}

impl std::ops::Add for GeometryRatios {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            ear_right: self.ear_right + rhs.ear_right,
            ear_left: self.ear_left + rhs.ear_left,
            brow_ratio_right: self.brow_ratio_right + rhs.brow_ratio_right,
            brow_ratio_left: self.brow_ratio_left + rhs.brow_ratio_left,
            mouth_smile_lift_ratio: self.mouth_smile_lift_ratio + rhs.mouth_smile_lift_ratio,
            mouth_width_ratio: self.mouth_width_ratio + rhs.mouth_width_ratio,
            pitch_ratio: self.pitch_ratio + rhs.pitch_ratio,
        }
    }
}

impl std::ops::Div<f32> for GeometryRatios {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self {
            ear_right: self.ear_right / rhs,
            ear_left: self.ear_left / rhs,
            brow_ratio_right: self.brow_ratio_right / rhs,
            brow_ratio_left: self.brow_ratio_left / rhs,
            mouth_smile_lift_ratio: self.mouth_smile_lift_ratio / rhs,
            mouth_width_ratio: self.mouth_width_ratio / rhs,
            pitch_ratio: self.pitch_ratio / rhs,
        }
    }
}

/// Numerical guard for near-zero denominators.
const EPS: f32 = 1e-6;

fn pt(p: &[Landmark], i: usize) -> (f32, f32) {
    (p[i].x, p[i].y)
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Mean `(x, y)` of the given landmark indices.
fn centroid(p: &[Landmark], idxs: &[usize]) -> (f32, f32) {
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for &i in idxs {
        sx += p[i].x;
        sy += p[i].y;
    }
    let n = idxs.len() as f32;
    (sx / n, sy / n)
}

/// Mean `y` over an inclusive index range.
fn mean_y(p: &[Landmark], range: std::ops::RangeInclusive<usize>) -> f32 {
    let (start, end) = (*range.start(), *range.end());
    let n = (end - start + 1) as f32;
    (start..=end).map(|i| p[i].y).sum::<f32>() / n
}

/// Standard six-point eye-aspect-ratio for `[p1..p6]`.
fn ear(p: &[Landmark], ids: [usize; 6]) -> f32 {
    let horiz = dist(pt(p, ids[0]), pt(p, ids[3]));
    if horiz < EPS {
        return 0.0;
    }
    let v1 = dist(pt(p, ids[1]), pt(p, ids[5]));
    let v2 = dist(pt(p, ids[2]), pt(p, ids[4]));
    (v1 + v2) / (2.0 * horiz)
}

/// Convert an EAR to a blink amount in `0..1` (1 = closed), clamped.
fn blink(ear_value: f32, ear_open: f32) -> f32 {
    if ear_open < EPS {
        return 0.0;
    }
    1.0 - (ear_value / ear_open).clamp(0.0, 1.0)
}
