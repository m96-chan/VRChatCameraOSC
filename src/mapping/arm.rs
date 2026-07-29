//! Pose landmarks → full-arm parameters (issue #28 phase 2).
//!
//! Phase 1 mapped the *wrist height* of a hand to a single `ArmUpDown`
//! float; phase 2 replaces that input entirely with the BlazePose landmarks
//! (`tracking::pose::PoseFrame` — shoulder/elbow/wrist per side, with
//! per-landmark visibility) and emits the full fixed contract the `unity/`
//! wizard binds (parameter names/semantics/ranges are FIXED — the Unity
//! side is implemented against exactly this):
//!
//! ```text
//! /avatar/parameters/VCO_{L,R}_ArmTracked  bool
//! /avatar/parameters/VCO_{L,R}_ArmUpDown   float -1..1
//! /avatar/parameters/VCO_{L,R}_ArmAcross   float -1..1
//! /avatar/parameters/VCO_{L,R}_Elbow       float  0..1
//! ```
//!
//! - **ArmTracked** — true while that arm's shoulder **and** elbow
//!   visibility (sigmoid, from the pose net) exceed
//!   [`VISIBILITY_THRESHOLD`]; false the moment they don't (emitted
//!   immediately, no debounce — the Unity gate needs to open fast).
//! - **ArmUpDown** — direction cosine of the upper-arm direction (shoulder →
//!   elbow unit vector in the image plane) against image-up: `-1` hanging
//!   straight down, `0` horizontal (or pointing at the camera), `+1`
//!   straight overhead. I.e. `UpDown = −dy` (y is image-down).
//! - **ArmAcross** — lateral direction cosine toward the body midline,
//!   measured against the shoulder→shoulder axis so a tilted torso reads
//!   correctly: `+1` = upper arm pointing across the chest toward the
//!   *opposite shoulder*, `-1` = pointing outward away from the body, `0` =
//!   in the vertical/forward plane. The sign is per-side (positive is
//!   toward the other shoulder for *each* arm).
//! - **Elbow** — bend from the interior elbow angle (between elbow→shoulder
//!   and elbow→wrist): `0` = straight (180°), `1` = fully bent (interior
//!   angle [`ELBOW_BENT_DEG`] ≈ 35° or less), linear in between and
//!   clamped. Needs the wrist: if the wrist's visibility is below the bar
//!   while shoulder+elbow stay visible, the arm stays tracked and only the
//!   elbow channel decays toward 0 (straight).
//!
//! When an arm becomes untracked, `ArmTracked` goes false immediately and
//! the three floats decay to **exactly 0.0** within a few frames
//! ([`DECAY_FACTOR`] per frame, snapping below [`SNAP_EPSILON`] — ~6 frames
//! from ±1, ≈200 ms at 30 Hz), so the `unity/` gate opens and VRChat's idle
//! animation resumes (same machinery as phase 1).
//!
//! **Left/right**: the pose model's landmark labels name the user's
//! **physical** side directly on our unmirrored webcam frames — the same
//! live-verified convention as hand handedness (see
//! `mapping::gesture::GestureMapperConfig::mirror`, 2026-07-29): with
//! `mirror = true` (default) the model's "left shoulder" (index 11) drives
//! `VCO_L_*`; `mirror = false` swaps, for capture sources that pre-mirror.
//!
//! Tracked output is One-Euro smoothed per channel ([`ARM_MINCUTOFF`]/
//! [`ARM_BETA`], the head-pose filter class — an arm is a big slow limb).
//! On loss the filters are dropped so a reacquired arm snaps to position
//! (One-Euro's first sample passes through) instead of lerping from stale
//! state.

use crate::mapping::arkit::OneEuro;
use crate::osc::OscParam;
use crate::tracking::pose::{
    PoseFrame, LEFT_ELBOW, LEFT_SHOULDER, LEFT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_WRIST,
};

/// Minimum sigmoid visibility for a landmark to count as reliably seen
/// (MediaPipe's conventional bar).
pub const VISIBILITY_THRESHOLD: f32 = 0.5;
/// Interior elbow angle (degrees) that reads as fully straight → `Elbow 0`.
pub const ELBOW_STRAIGHT_DEG: f32 = 180.0;
/// Interior elbow angle (degrees) at (or below) which `Elbow` saturates
/// to 1 — a fully bent arm.
pub const ELBOW_BENT_DEG: f32 = 35.0;
/// Geometric per-frame decay of an emitted value while its input is
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

/// The raw interior-elbow-angle → bend map: linear from
/// [`ELBOW_STRAIGHT_DEG`]→0 to [`ELBOW_BENT_DEG`]→1, clamped.
pub fn elbow_bend_raw(interior_deg: f32) -> f32 {
    ((ELBOW_STRAIGHT_DEG - interior_deg) / (ELBOW_STRAIGHT_DEG - ELBOW_BENT_DEG)).clamp(0.0, 1.0)
}

/// One arm's raw geometry for one frame, pre-smoothing.
struct ArmSample {
    up_down: f32,
    across: f32,
    /// `None` when the wrist isn't reliably visible (elbow channel decays
    /// while the arm stays tracked).
    elbow: Option<f32>,
}

/// Compute one arm's raw sample from its landmarks, or `None` when the arm
/// doesn't qualify as tracked (shoulder/elbow visibility below the bar, or
/// degenerate geometry).
fn arm_sample(
    shoulder: [f32; 3],
    elbow: [f32; 3],
    wrist: [f32; 3],
    other_shoulder: [f32; 3],
    vis_shoulder: f32,
    vis_elbow: f32,
    vis_wrist: f32,
) -> Option<ArmSample> {
    if vis_shoulder <= VISIBILITY_THRESHOLD || vis_elbow <= VISIBILITY_THRESHOLD {
        return None;
    }
    // Upper-arm direction in the image plane (y down).
    let u = [elbow[0] - shoulder[0], elbow[1] - shoulder[1]];
    let ulen = (u[0] * u[0] + u[1] * u[1]).sqrt();
    if ulen < 1e-6 {
        return None; // degenerate (arm pointing straight at the camera)
    }
    let u = [u[0] / ulen, u[1] / ulen];
    let up_down = (-u[1]).clamp(-1.0, 1.0);

    // Toward-midline axis: this shoulder → the other shoulder (so torso
    // roll doesn't corrupt the reading). Degenerate → 0 (no lateral cue).
    let m = [
        other_shoulder[0] - shoulder[0],
        other_shoulder[1] - shoulder[1],
    ];
    let mlen = (m[0] * m[0] + m[1] * m[1]).sqrt();
    let across = if mlen < 1e-6 {
        0.0
    } else {
        ((u[0] * m[0] + u[1] * m[1]) / mlen).clamp(-1.0, 1.0)
    };

    // Interior elbow angle between elbow→shoulder and elbow→wrist (3-D,
    // like the finger-curl cues) — only when the wrist is reliably seen.
    let bend = (vis_wrist > VISIBILITY_THRESHOLD)
        .then(|| {
            let a = [
                shoulder[0] - elbow[0],
                shoulder[1] - elbow[1],
                shoulder[2] - elbow[2],
            ];
            let b = [
                wrist[0] - elbow[0],
                wrist[1] - elbow[1],
                wrist[2] - elbow[2],
            ];
            let (na, nb) = (
                (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt(),
                (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt(),
            );
            if na < 1e-6 || nb < 1e-6 {
                return None;
            }
            let cos = ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)).clamp(-1.0, 1.0);
            Some(elbow_bend_raw(cos.acos().to_degrees()))
        })
        .flatten();

    Some(ArmSample {
        up_down,
        across,
        elbow: bend,
    })
}

/// One smoothed/decaying output channel: One-Euro while fed, geometric
/// decay to an exact 0.0 while not (see the module docs).
#[derive(Debug, Clone, Copy)]
struct Channel {
    filter: OneEuro,
    value: f32,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            filter: OneEuro::new(ARM_MINCUTOFF, ARM_BETA),
            value: 0.0,
        }
    }
}

impl Channel {
    fn step(&mut self, target: Option<f32>) -> f32 {
        match target {
            Some(t) => {
                self.value = self.filter.filter(t, ASSUMED_DT);
            }
            None => {
                // Drop the filter history so a reacquired arm snaps to
                // position (One-Euro's first sample passes through).
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

/// One arm's channels.
#[derive(Debug, Clone, Copy, Default)]
struct SideArm {
    up_down: Channel,
    across: Channel,
    elbow: Channel,
}

impl SideArm {
    /// Returns `(tracked, up_down, across, elbow)`.
    fn step(&mut self, sample: Option<ArmSample>) -> (bool, f32, f32, f32) {
        match sample {
            Some(s) => (
                true,
                self.up_down.step(Some(s.up_down)),
                self.across.step(Some(s.across)),
                self.elbow.step(s.elbow),
            ),
            None => (
                false,
                self.up_down.step(None),
                self.across.step(None),
                self.elbow.step(None),
            ),
        }
    }
}

/// Pose frame → the 8 fixed arm parameters (see module docs). Always emits
/// all 8 — a decayed-out arm keeps reporting `false`/exact `0.0` (harmless,
/// and the monitor/TUI stay truthful about what is on the wire).
#[derive(Debug, Clone)]
pub struct ArmMapper {
    /// See the module docs: `true` (default, raw webcam) takes the model's
    /// physical-side landmark labels as-is; `false` swaps them.
    mirror: bool,
    left: SideArm,
    right: SideArm,
}

impl Default for ArmMapper {
    fn default() -> Self {
        Self::new(true)
    }
}

impl ArmMapper {
    pub fn new(mirror: bool) -> Self {
        Self {
            mirror,
            left: SideArm::default(),
            right: SideArm::default(),
        }
    }

    /// Map one frame's pose (or its absence) to the arm parameter set.
    pub fn map(&mut self, pose: Option<&PoseFrame>) -> Vec<OscParam> {
        let sample_for = |model_left: bool| -> Option<ArmSample> {
            let p = pose?;
            let (sh, el, wr, other) = if model_left {
                (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER)
            } else {
                (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, LEFT_SHOULDER)
            };
            arm_sample(
                p.points[sh],
                p.points[el],
                p.points[wr],
                p.points[other],
                p.visibility[sh],
                p.visibility[el],
                p.visibility[wr],
            )
        };
        // mirror = true: the model's landmark labels name the physical side
        // directly (live-verified for hands 2026-07-29 — same unmirrored
        // frames, same convention).
        let (l_sample, r_sample) = if self.mirror {
            (sample_for(true), sample_for(false))
        } else {
            (sample_for(false), sample_for(true))
        };

        let (lt, lu, la, le) = self.left.step(l_sample);
        let (rt, ru, ra, re) = self.right.step(r_sample);
        vec![
            OscParam::bool("VCO_L_ArmTracked", lt),
            OscParam::float("VCO_L_ArmUpDown", lu),
            OscParam::float("VCO_L_ArmAcross", la),
            OscParam::float("VCO_L_Elbow", le),
            OscParam::bool("VCO_R_ArmTracked", rt),
            OscParam::float("VCO_R_ArmUpDown", ru),
            OscParam::float("VCO_R_ArmAcross", ra),
            OscParam::float("VCO_R_Elbow", re),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::OscValue;
    use crate::tracking::pose::NUM_POSE_LANDMARKS;

    /// A pose whose model-LEFT arm is set from the given 2-D points, with
    /// the model-right shoulder placed for the midline axis. All listed
    /// landmarks fully visible unless overridden.
    fn pose(shoulder: [f32; 2], elbow: [f32; 2], wrist: [f32; 2]) -> PoseFrame {
        let mut points = [[0.5f32, 0.5, 0.0]; NUM_POSE_LANDMARKS];
        let mut visibility = [0.0f32; NUM_POSE_LANDMARKS];
        points[LEFT_SHOULDER] = [shoulder[0], shoulder[1], 0.0];
        points[LEFT_ELBOW] = [elbow[0], elbow[1], 0.0];
        points[LEFT_WRIST] = [wrist[0], wrist[1], 0.0];
        points[RIGHT_SHOULDER] = [0.4, 0.3, 0.0]; // opposite shoulder
        for i in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER] {
            visibility[i] = 0.99;
        }
        PoseFrame {
            confidence: 0.9,
            points,
            visibility,
        }
    }

    /// Model-left shoulder anchor used by all fixtures (opposite shoulder
    /// is at x = 0.4, so the toward-midline axis points −x).
    const SHOULDER: [f32; 2] = [0.6, 0.3];

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

    fn bool_value(params: &[OscParam], name: &str) -> bool {
        let p = params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} missing"));
        match p.value {
            OscValue::Bool(v) => v,
            ref other => panic!("{name} is not a bool: {other:?}"),
        }
    }

    // --- raw elbow map ----------------------------------------------------

    #[test]
    fn elbow_map_hits_the_documented_anchors() {
        assert_eq!(elbow_bend_raw(180.0), 0.0, "straight");
        assert_eq!(elbow_bend_raw(35.0), 1.0, "fully bent");
        assert_eq!(elbow_bend_raw(10.0), 1.0, "clamps past fully bent");
        assert_eq!(elbow_bend_raw(200.0), 0.0, "clamps past straight");
        // 90° interior: (180 − 90) / (180 − 35) = 90/145.
        assert!((elbow_bend_raw(90.0) - 90.0 / 145.0).abs() < 1e-6);
    }

    // --- geometry fixtures (first sample passes through the One-Euro, so
    // --- frame 1 reads the raw value) --------------------------------------

    #[test]
    fn arm_straight_down_reads_minus_one_up_down() {
        let mut m = ArmMapper::new(true);
        let p = pose(SHOULDER, [0.6, 0.45], [0.6, 0.6]);
        let out = m.map(Some(&p));
        assert!(bool_value(&out, "VCO_L_ArmTracked"));
        assert!((float_value(&out, "VCO_L_ArmUpDown") + 1.0).abs() < 1e-5);
        assert!(float_value(&out, "VCO_L_ArmAcross").abs() < 1e-5);
        assert!(float_value(&out, "VCO_L_Elbow").abs() < 1e-5);
        // The other arm never had visible landmarks: untracked, all zero.
        assert!(!bool_value(&out, "VCO_R_ArmTracked"));
        assert_eq!(float_value(&out, "VCO_R_ArmUpDown"), 0.0);
    }

    #[test]
    fn arm_overhead_reads_plus_one_up_down() {
        let mut m = ArmMapper::new(true);
        let p = pose(SHOULDER, [0.6, 0.15], [0.6, 0.0]);
        let out = m.map(Some(&p));
        assert!(bool_value(&out, "VCO_L_ArmTracked"));
        assert!((float_value(&out, "VCO_L_ArmUpDown") - 1.0).abs() < 1e-5);
        assert!(float_value(&out, "VCO_L_ArmAcross").abs() < 1e-5);
    }

    #[test]
    fn arm_horizontal_outward_reads_minus_one_across() {
        // The opposite shoulder is at smaller x, so +x is away from the
        // midline: elbow out at (0.75, 0.3) → Across −1, UpDown 0.
        let mut m = ArmMapper::new(true);
        let p = pose(SHOULDER, [0.75, 0.3], [0.9, 0.3]);
        let out = m.map(Some(&p));
        assert!(float_value(&out, "VCO_L_ArmUpDown").abs() < 1e-5);
        assert!((float_value(&out, "VCO_L_ArmAcross") + 1.0).abs() < 1e-5);
        assert!(float_value(&out, "VCO_L_Elbow").abs() < 1e-5);
    }

    #[test]
    fn arm_across_chest_reads_plus_one_across() {
        let mut m = ArmMapper::new(true);
        let p = pose(SHOULDER, [0.45, 0.3], [0.3, 0.3]);
        let out = m.map(Some(&p));
        assert!(float_value(&out, "VCO_L_ArmUpDown").abs() < 1e-5);
        assert!((float_value(&out, "VCO_L_ArmAcross") - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ninety_degree_elbow_reads_the_linear_map_value() {
        // Upper arm straight down, forearm horizontal: interior angle 90°
        // → (180 − 90)/145 ≈ 0.6207.
        let mut m = ArmMapper::new(true);
        let p = pose(SHOULDER, [0.6, 0.5], [0.75, 0.5]);
        let out = m.map(Some(&p));
        assert!(bool_value(&out, "VCO_L_ArmTracked"));
        let e = float_value(&out, "VCO_L_Elbow");
        assert!((e - 90.0 / 145.0).abs() < 1e-4, "elbow {e}");
    }

    #[test]
    fn fully_folded_elbow_saturates_to_one() {
        // Wrist folded back next to the shoulder: interior angle well under
        // 35° → 1.0.
        let mut m = ArmMapper::new(true);
        let p = pose(SHOULDER, [0.6, 0.5], [0.61, 0.32]);
        let out = m.map(Some(&p));
        assert_eq!(float_value(&out, "VCO_L_Elbow"), 1.0);
    }

    // --- visibility gating -------------------------------------------------

    #[test]
    fn low_shoulder_or_elbow_visibility_untracks_and_decays_to_exact_zero() {
        let mut m = ArmMapper::new(true);
        let raised = pose(SHOULDER, [0.6, 0.15], [0.6, 0.0]);
        for _ in 0..10 {
            m.map(Some(&raised));
        }
        let out = m.map(Some(&raised));
        assert!(float_value(&out, "VCO_L_ArmUpDown") > 0.9);

        // Elbow visibility collapses: untracked immediately, floats decay
        // to exactly 0.0 within a few frames (the contract's ~200 ms).
        let mut occluded = raised.clone();
        occluded.visibility[LEFT_ELBOW] = 0.2;
        let out = m.map(Some(&occluded));
        assert!(!bool_value(&out, "VCO_L_ArmTracked"), "immediate false");
        let mut frames = 1;
        loop {
            let out = m.map(Some(&occluded));
            frames += 1;
            if float_value(&out, "VCO_L_ArmUpDown") == 0.0 {
                break;
            }
            assert!(frames < 10, "still nonzero after {frames} frames");
        }
        assert!(frames <= 7, "took {frames} frames");
        // And stays pinned at exact zero.
        let out = m.map(Some(&occluded));
        assert_eq!(float_value(&out, "VCO_L_ArmUpDown"), 0.0);
        assert_eq!(float_value(&out, "VCO_L_ArmAcross"), 0.0);
        assert_eq!(float_value(&out, "VCO_L_Elbow"), 0.0);
    }

    #[test]
    fn invisible_wrist_keeps_arm_tracked_but_decays_the_elbow() {
        let mut m = ArmMapper::new(true);
        let bent = pose(SHOULDER, [0.6, 0.5], [0.75, 0.5]);
        for _ in 0..10 {
            m.map(Some(&bent));
        }
        let out = m.map(Some(&bent));
        assert!(float_value(&out, "VCO_L_Elbow") > 0.5);

        let mut no_wrist = bent.clone();
        no_wrist.visibility[LEFT_WRIST] = 0.1;
        let mut last_elbow = f32::MAX;
        for _ in 0..8 {
            let out = m.map(Some(&no_wrist));
            assert!(
                bool_value(&out, "VCO_L_ArmTracked"),
                "shoulder+elbow visible: arm stays tracked"
            );
            // UpDown keeps reporting live geometry.
            assert!((float_value(&out, "VCO_L_ArmUpDown") + 1.0).abs() < 1e-4);
            last_elbow = float_value(&out, "VCO_L_Elbow");
        }
        assert_eq!(last_elbow, 0.0, "elbow decays to exact 0 without a wrist");
    }

    #[test]
    fn no_pose_at_all_emits_all_eight_params_untracked() {
        let mut m = ArmMapper::new(true);
        let out = m.map(None);
        assert_eq!(out.len(), 8);
        for side in ["L", "R"] {
            assert!(!bool_value(&out, &format!("VCO_{side}_ArmTracked")));
            assert_eq!(float_value(&out, &format!("VCO_{side}_ArmUpDown")), 0.0);
            assert_eq!(float_value(&out, &format!("VCO_{side}_ArmAcross")), 0.0);
            assert_eq!(float_value(&out, &format!("VCO_{side}_Elbow")), 0.0);
        }
    }

    // --- side assignment / mirror ------------------------------------------

    #[test]
    fn mirror_true_routes_model_left_landmarks_to_the_left_params() {
        // Default (unmirrored webcam): the model's physical-side labels are
        // taken as-is — the fixture arms live on indices 11/13/15
        // ("left") and must drive VCO_L_*.
        let mut m = ArmMapper::new(true);
        let p = pose(SHOULDER, [0.6, 0.15], [0.6, 0.0]);
        let out = m.map(Some(&p));
        assert!(bool_value(&out, "VCO_L_ArmTracked"));
        assert!(!bool_value(&out, "VCO_R_ArmTracked"));
        assert!(float_value(&out, "VCO_L_ArmUpDown") > 0.9);
        assert_eq!(float_value(&out, "VCO_R_ArmUpDown"), 0.0);
    }

    #[test]
    fn mirror_false_swaps_the_sides() {
        let mut m = ArmMapper::new(false);
        let p = pose(SHOULDER, [0.6, 0.15], [0.6, 0.0]);
        let out = m.map(Some(&p));
        assert!(!bool_value(&out, "VCO_L_ArmTracked"));
        assert!(bool_value(&out, "VCO_R_ArmTracked"));
        assert!(float_value(&out, "VCO_R_ArmUpDown") > 0.9);
        assert_eq!(float_value(&out, "VCO_L_ArmUpDown"), 0.0);
    }

    #[test]
    fn both_arms_track_independently() {
        // Mirror the fixture's model-left arm onto the model-right indices
        // too, hanging instead of raised.
        let mut p = pose(SHOULDER, [0.6, 0.15], [0.6, 0.0]); // L overhead
        p.points[RIGHT_SHOULDER] = [0.4, 0.3, 0.0];
        p.points[RIGHT_ELBOW] = [0.4, 0.45, 0.0];
        p.points[RIGHT_WRIST] = [0.4, 0.6, 0.0];
        for i in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST] {
            p.visibility[i] = 0.99;
        }
        let mut m = ArmMapper::new(true);
        let out = m.map(Some(&p));
        assert!(float_value(&out, "VCO_L_ArmUpDown") > 0.9, "left overhead");
        assert!(float_value(&out, "VCO_R_ArmUpDown") < -0.9, "right hanging");
        // Right arm hanging: across ~0 relative to its own midline axis.
        assert!(float_value(&out, "VCO_R_ArmAcross").abs() < 1e-4);
    }

    // --- smoothing ---------------------------------------------------------

    #[test]
    fn tracked_motion_is_smoothed_not_snapped() {
        let mut m = ArmMapper::new(true);
        let down = pose(SHOULDER, [0.6, 0.45], [0.6, 0.6]);
        for _ in 0..30 {
            m.map(Some(&down));
        }
        // Jump straight to overhead: the smoothed output moves toward +1
        // but does not cover the −1 → +1 step in one frame.
        let up = pose(SHOULDER, [0.6, 0.15], [0.6, 0.0]);
        let v = float_value(&m.map(Some(&up)), "VCO_L_ArmUpDown");
        assert!(v > -1.0, "moves toward the target, got {v}");
        assert!(v < 0.95, "one frame must not complete the step, got {v}");
    }
}
