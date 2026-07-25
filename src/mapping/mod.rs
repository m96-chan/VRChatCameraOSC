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
//! | `BrowUp`       | `0..1`  | brows 17–26 vs eye centres           | `(d/IOD − brow_neutral)/brow_span` |
//! | `HeadRoll`     | `-1..1` | eye centres (36–41 vs 42–47)         | `atan2(dy,dx)/roll_max_rad` |
//! | `HeadYaw`      | `-1..1` | nose tip 30 vs eye midpoint          | `(tip.x − midx)/(IOD·yaw_half_span)` |
//! | `HeadPitch`    | `-1..1` | nose bridge 27→30 vs IOD             | `(len/IOD − pitch_neutral)/pitch_span` |
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

use crate::osc::OscParam;
use crate::tracking::{FaceLandmarks, Landmark};

/// Output parameter names, in a stable emission order.
const PARAM_NAMES: [&str; 7] = [
    "MouthOpen",
    "EyeBlinkLeft",
    "EyeBlinkRight",
    "BrowUp",
    "HeadRoll",
    "HeadYaw",
    "HeadPitch",
];

/// Per-parameter output range `(min, max)`, aligned with [`PARAM_NAMES`].
const PARAM_RANGES: [(f32, f32); 7] = [
    (0.0, 1.0),  // MouthOpen
    (0.0, 1.0),  // EyeBlinkLeft
    (0.0, 1.0),  // EyeBlinkRight
    (0.0, 1.0),  // BrowUp
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
    /// Neutral brow-to-eye distance as a fraction of IOD (`BrowUp = 0`).
    pub brow_neutral: f32,
    /// Extra brow-to-eye/IOD fraction that spans neutral → fully raised.
    pub brow_span: f32,
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
            brow_neutral: 0.5,
            brow_span: 0.2,
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
    prev: [f32; 7],
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
            prev: [0.0; 7],
        }
    }

    /// Produce the OSC parameter updates for one frame of landmarks.
    pub fn map(&mut self, landmarks: &FaceLandmarks) -> Vec<OscParam> {
        let raw = self.raw_params(&landmarks.points);
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
    fn raw_params(&self, p: &[Landmark]) -> [f32; 7] {
        let cfg = &self.config;

        let right_eye = centroid(p, &[36, 37, 38, 39, 40, 41]);
        let left_eye = centroid(p, &[42, 43, 44, 45, 46, 47]);
        let iod = dist(right_eye, left_eye);

        // Degenerate geometry (no detectable face scale): emit resting values.
        if iod < EPS {
            return [0.0; 7];
        }

        // MouthOpen: inner-lip gap over mouth width.
        let mouth_width = dist(pt(p, 60), pt(p, 64));
        let mouth_open = if mouth_width < EPS {
            0.0
        } else {
            let gap = dist(pt(p, 62), pt(p, 66));
            (gap / mouth_width) / cfg.mouth_open_max
        };

        // Eye blink from EAR, inverted so 1 = closed.
        let blink_right = blink(ear(p, [36, 37, 38, 39, 40, 41]), cfg.ear_open);
        let blink_left = blink(ear(p, [42, 43, 44, 45, 46, 47]), cfg.ear_open);

        // BrowUp: brow-to-eye vertical distance (fraction of IOD) vs neutral.
        let brow_y = mean_y(p, 17..=26);
        let eye_y = (right_eye.1 + left_eye.1) / 2.0;
        let brow_ratio = (eye_y - brow_y) / iod;
        let brow_up = (brow_ratio - cfg.brow_neutral) / cfg.brow_span;

        // HeadRoll: tilt angle of the right→left eye-centre line.
        let dx = left_eye.0 - right_eye.0;
        let dy = left_eye.1 - right_eye.1;
        let head_roll = dy.atan2(dx) / cfg.roll_max_rad;

        // HeadYaw: nose-tip horizontal offset from the eye midpoint.
        let mid_x = (right_eye.0 + left_eye.0) / 2.0;
        let head_yaw = (pt(p, 30).0 - mid_x) / (iod * cfg.yaw_half_span);

        // HeadPitch: nose-bridge length (27→30) vs IOD, relative to neutral.
        let bridge = dist(pt(p, 27), pt(p, 30));
        let head_pitch = (bridge / iod - cfg.pitch_neutral) / cfg.pitch_span;

        [
            mouth_open,
            blink_left,
            blink_right,
            brow_up,
            head_roll,
            head_yaw,
            head_pitch,
        ]
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
