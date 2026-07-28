//! 21 hand landmarks → VRChat gesture ints (issue #8 phase 1).
//!
//! Pipeline per hand and frame: [`finger_curls`] reduces the landmarks to
//! five per-finger curl scalars (0 = straight .. 1 = fully curled), a
//! per-finger **hysteresis** turns those into stable extended/curled states
//! (extend < [`EXTEND_THRESHOLD`], curl > [`CURL_THRESHOLD`], in between
//! keeps the previous state), [`classify`] matches the state pattern against
//! the standard VRChat gesture table, and a **debounce** only lets a new
//! gesture through after [`DEBOUNCE_FRAMES`] consecutive frames — ints snap
//! avatars hard, so stability beats latency (issue #8 design §4).
//!
//! Curl heuristics follow the approaches proven in
//! [andypotato/fingerpose](https://github.com/andypotato/fingerpose) (joint
//! bend angles) and the
//! [TheJLifeX gist](https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a)
//! (thumb via distance to the pinky base) — see the issue for the survey.
//!
//! [`GestureMapper`] assigns hands to the left/right slots (see the `mirror`
//! note below) and emits the OSC ints: `VCO_GestureLeft`/`VCO_GestureRight`
//! always (the custom params the `unity/` wizard binds — VRChat's native
//! gesture endpoints are read-only over OSC, see issue #8 §1), plus the
//! native `GestureLeft`/`GestureRight` addresses when configured (default
//! on: harmless today, free forward-compat if vrchat-community/osc#42 ever
//! ships).

use crate::osc::OscParam;
use crate::tracking::hands::{
    HandFrame, Handedness, INDEX_DIP, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_DIP, MIDDLE_MCP,
    MIDDLE_PIP, MIDDLE_TIP, NUM_HAND_LANDMARKS, PINKY_DIP, PINKY_MCP, PINKY_PIP, PINKY_TIP,
    RING_DIP, RING_MCP, RING_PIP, RING_TIP, THUMB_IP, THUMB_TIP, WRIST,
};

/// Below this curl a finger reads **extended** (hysteresis low bar).
pub const EXTEND_THRESHOLD: f32 = 0.35;
/// Above this curl a finger reads **curled** (hysteresis high bar).
pub const CURL_THRESHOLD: f32 = 0.65;
/// A new gesture must persist this many consecutive frames (~130 ms at
/// 30 FPS) before it is emitted.
pub const DEBOUNCE_FRAMES: u32 = 4;

/// The standard VRChat gesture indices (0–7), exactly as documented at
/// [creators.vrchat.com/avatars/animator-parameters](https://creators.vrchat.com/avatars/animator-parameters/).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gesture {
    #[default]
    Neutral = 0,
    Fist = 1,
    HandOpen = 2,
    FingerPoint = 3,
    Victory = 4,
    RockNRoll = 5,
    HandGun = 6,
    ThumbsUp = 7,
}

impl Gesture {
    pub fn index(self) -> i32 {
        self as i32
    }

    pub fn name(self) -> &'static str {
        match self {
            Gesture::Neutral => "Neutral",
            Gesture::Fist => "Fist",
            Gesture::HandOpen => "HandOpen",
            Gesture::FingerPoint => "FingerPoint",
            Gesture::Victory => "Victory",
            Gesture::RockNRoll => "RockNRoll",
            Gesture::HandGun => "HandGun",
            Gesture::ThumbsUp => "ThumbsUp",
        }
    }
}

/// Human-readable name for a gesture int (TUI/monitor display; out-of-range
/// values read `"?"`).
pub fn gesture_name(index: i32) -> &'static str {
    match index {
        0 => Gesture::Neutral.name(),
        1 => Gesture::Fist.name(),
        2 => Gesture::HandOpen.name(),
        3 => Gesture::FingerPoint.name(),
        4 => Gesture::Victory.name(),
        5 => Gesture::RockNRoll.name(),
        6 => Gesture::HandGun.name(),
        7 => Gesture::ThumbsUp.name(),
        _ => "?",
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    norm(sub(a, b))
}

/// Angle (radians, `0..π`) between two 3-D vectors; 0 for degenerate input.
fn angle(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (na, nb) = (norm(a), norm(b));
    if na < 1e-6 || nb < 1e-6 {
        return 0.0;
    }
    let cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
    cos.clamp(-1.0, 1.0).acos()
}

/// Bend of a 4-joint finger chain `mcp → pip → dip → tip`: the sum of the
/// joint angles at PIP and DIP, normalized by π (straight = 0, folded back
/// on itself ≈ 1) — the fingerpose NoCurl/HalfCurl/FullCurl signal made
/// continuous.
fn chain_curl(
    p: &[[f32; 3]; NUM_HAND_LANDMARKS],
    mcp: usize,
    pip: usize,
    dip: usize,
    tip: usize,
) -> f32 {
    let at_pip = angle(sub(p[pip], p[mcp]), sub(p[dip], p[pip]));
    let at_dip = angle(sub(p[dip], p[pip]), sub(p[tip], p[dip]));
    ((at_pip + at_dip) / std::f32::consts::PI).clamp(0.0, 1.0)
}

/// Thumb gain on the distance cue: how fast the tip-vs-IP margin (in palm
/// units) saturates the 0..1 curl. Chosen so a clearly splayed thumb
/// (margin ≈ +0.2 palms) reads ~0 and a tucked one (margin ≈ −0.2) reads ~1.
const THUMB_GAIN: f32 = 2.5;

/// Per-finger curl scalars `[thumb, index, middle, ring, pinky]`, each
/// 0 = straight/extended .. 1 = fully curled, from the 21 3-D landmarks.
///
/// Fingers use joint bend angles ([`chain_curl`]). The thumb has no PIP/DIP
/// pair and folds *across* the palm rather than onto it, so it is
/// special-cased with the TheJLifeX distance cue: an extended thumb's tip is
/// farther from the pinky base than its IP joint is, a tucked thumb's tip is
/// nearer. The margin is normalized by the palm size (wrist → middle MCP),
/// making it scale- and rotation-free.
pub fn finger_curls(p: &[[f32; 3]; NUM_HAND_LANDMARKS]) -> [f32; 5] {
    let palm = dist(p[WRIST], p[MIDDLE_MCP]).max(1e-6);
    let margin = (dist(p[THUMB_TIP], p[PINKY_MCP]) - dist(p[THUMB_IP], p[PINKY_MCP])) / palm;
    let thumb = (0.5 - THUMB_GAIN * margin).clamp(0.0, 1.0);
    [
        thumb,
        chain_curl(p, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        chain_curl(p, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        chain_curl(p, RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        chain_curl(p, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]
}

/// A finger's hysteresis state. `Unknown` (start, or after losing the hand)
/// matches only the `*` slots of the gesture table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FingerState {
    #[default]
    Unknown,
    Extended,
    Curled,
}

/// Per-finger hysteresis: crossing [`EXTEND_THRESHOLD`] downward latches
/// Extended, crossing [`CURL_THRESHOLD`] upward latches Curled, the band in
/// between keeps the previous state.
#[derive(Debug, Default, Clone)]
pub struct FingerHysteresis {
    states: [FingerState; 5],
}

impl FingerHysteresis {
    pub fn update(&mut self, curls: [f32; 5]) -> [FingerState; 5] {
        for (state, curl) in self.states.iter_mut().zip(curls) {
            if curl < EXTEND_THRESHOLD {
                *state = FingerState::Extended;
            } else if curl > CURL_THRESHOLD {
                *state = FingerState::Curled;
            } // else: keep the previous state
        }
        self.states
    }

    /// Forget everything (the hand was lost).
    pub fn reset(&mut self) {
        self.states = [FingerState::Unknown; 5];
    }
}

/// Finger-state pattern `[thumb, index, middle, ring, pinky]` → gesture, per
/// the table in issue #8 §4 (E = Extended, C = Curled, `*` = don't care):
///
/// | T I M R P | Gesture |
/// |---|---|
/// | C C C C C | 1 Fist |
/// | * E E E E | 2 HandOpen (thumb E or C) |
/// | * E C C E | 5 RockNRoll — before FingerPoint |
/// | * E E C C | 4 Victory |
/// | E E C C C | 6 HandGun — before FingerPoint (differs only in thumb) |
/// | * E C C C (thumb not E) | 3 FingerPoint |
/// | E C C C C | 7 ThumbsUp |
/// | anything else | 0 Neutral |
pub fn classify(states: &[FingerState; 5]) -> Gesture {
    use FingerState::{Curled as C, Extended as E};
    match *states {
        [C, C, C, C, C] => Gesture::Fist,
        [_, E, E, E, E] => Gesture::HandOpen,
        [_, E, C, C, E] => Gesture::RockNRoll,
        [_, E, E, C, C] => Gesture::Victory,
        [E, E, C, C, C] => Gesture::HandGun,
        [t, E, C, C, C] if t != E => Gesture::FingerPoint,
        [E, C, C, C, C] => Gesture::ThumbsUp,
        _ => Gesture::Neutral,
    }
}

/// Debounce on the final gesture int: a change is emitted only once the new
/// gesture has been observed [`DEBOUNCE_FRAMES`] consecutive frames.
#[derive(Debug, Default, Clone)]
pub struct GestureDebounce {
    emitted: Gesture,
    candidate: Gesture,
    count: u32,
}

impl GestureDebounce {
    /// Feed one frame's raw classification; returns the debounced gesture.
    pub fn feed(&mut self, raw: Gesture) -> Gesture {
        if raw == self.emitted {
            self.candidate = raw;
            self.count = 0;
        } else if raw == self.candidate {
            self.count += 1;
            if self.count >= DEBOUNCE_FRAMES {
                self.emitted = raw;
                self.count = 0;
            }
        } else {
            self.candidate = raw;
            self.count = 1;
        }
        self.emitted
    }
}

/// Configuration for [`GestureMapper`].
#[derive(Debug, Clone, Copy)]
pub struct GestureMapperConfig {
    /// How the model's handedness label maps to the user's physical hand.
    /// The MediaPipe hand model labels handedness **assuming a mirrored
    /// (selfie) image**; webcam frames are unmirrored, so with the default
    /// `mirror = true` the label is swapped — the user's left hand lands on
    /// `GestureLeft`. Set `false` for capture sources that already mirror.
    /// (Empirical validation with a live avatar is a phase-2 acceptance
    /// item; this flag is the calibration-free escape hatch either way.)
    pub mirror: bool,
    /// Also emit the native `GestureLeft`/`GestureRight` addresses —
    /// read-only in current VRChat (issue #8 §1) but harmless, and free
    /// forward-compat if vrchat-community/osc#42 ships. Default on.
    pub emit_native: bool,
}

impl Default for GestureMapperConfig {
    fn default() -> Self {
        Self {
            mirror: true,
            emit_native: true,
        }
    }
}

/// Per-hand-slot state: finger hysteresis + gesture debounce.
#[derive(Debug, Default, Clone)]
struct SideState {
    hysteresis: FingerHysteresis,
    debounce: GestureDebounce,
}

impl SideState {
    fn step(&mut self, hand: Option<&HandFrame>) -> Gesture {
        let raw = match hand {
            Some(h) => {
                let curls = finger_curls(&h.points);
                classify(&self.hysteresis.update(curls))
            }
            None => {
                // No hand: forget the finger states so a reacquired hand
                // starts clean, and settle (debounced) to Neutral.
                self.hysteresis.reset();
                Gesture::Neutral
            }
        };
        self.debounce.feed(raw)
    }
}

/// Tracked hands → the per-frame gesture parameter set (see module docs).
#[derive(Debug, Default, Clone)]
pub struct GestureMapper {
    config: GestureMapperConfig,
    left: SideState,
    right: SideState,
}

impl GestureMapper {
    pub fn new(config: GestureMapperConfig) -> Self {
        Self {
            config,
            left: SideState::default(),
            right: SideState::default(),
        }
    }

    /// The gestures currently emitted for (left, right) — for the TUI.
    pub fn current(&self) -> (Gesture, Gesture) {
        (self.left.debounce.emitted, self.right.debounce.emitted)
    }

    /// Map one frame's tracked hands to OSC parameters. Always emits both
    /// `VCO_Gesture*` ints (Neutral when a hand is absent); native
    /// `Gesture*` ints follow `emit_native`.
    pub fn map(&mut self, hands: &[HandFrame]) -> Vec<OscParam> {
        // Assign each hand to a physical side; on a double booking keep the
        // more confident one.
        let mut left: Option<&HandFrame> = None;
        let mut right: Option<&HandFrame> = None;
        for hand in hands {
            let physical_left = match hand.handedness {
                Handedness::Left => !self.config.mirror,
                Handedness::Right => self.config.mirror,
            };
            let slot = if physical_left { &mut left } else { &mut right };
            if slot.is_none_or(|h| hand.confidence > h.confidence) {
                *slot = Some(hand);
            }
        }

        let l = self.left.step(left);
        let r = self.right.step(right);

        let mut params = vec![
            OscParam::int("VCO_GestureLeft", l.index()),
            OscParam::int("VCO_GestureRight", r.index()),
        ];
        if self.config.emit_native {
            params.push(OscParam::int("GestureLeft", l.index()));
            params.push(OscParam::int("GestureRight", r.index()));
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osc::OscValue;

    // --- Synthetic hands -------------------------------------------------
    //
    // Normalized frame coordinates, y down, fingers pointing up (an upright
    // hand facing the camera), z = 0 throughout: the curl cues are all
    // scale-/rotation-free 3-D constructions, so a planar fixture exercises
    // them fully.

    const WRIST_P: [f32; 3] = [0.50, 0.90, 0.0];
    const MCPS: [([f32; 3], usize); 4] = [
        ([0.40, 0.60, 0.0], INDEX_MCP),
        ([0.47, 0.58, 0.0], MIDDLE_MCP),
        ([0.54, 0.60, 0.0], RING_MCP),
        ([0.60, 0.63, 0.0], PINKY_MCP),
    ];

    /// Build a hand with the given per-finger extension pattern
    /// (`[thumb, index, middle, ring, pinky]`, `true` = extended).
    fn hand(extended: [bool; 5]) -> [[f32; 3]; NUM_HAND_LANDMARKS] {
        let mut p = [[0.0f32; 3]; NUM_HAND_LANDMARKS];
        p[WRIST] = WRIST_P;
        // Thumb base (CMC, MCP) is fixed; IP/TIP depend on the pattern.
        p[crate::tracking::hands::THUMB_CMC] = [0.42, 0.82, 0.0];
        p[crate::tracking::hands::THUMB_MCP] = [0.37, 0.76, 0.0];
        if extended[0] {
            // Splayed away from the palm (up-left).
            p[THUMB_IP] = [0.32, 0.70, 0.0];
            p[THUMB_TIP] = [0.27, 0.65, 0.0];
        } else {
            // Tucked across the palm, tip toward the pinky base.
            p[THUMB_IP] = [0.45, 0.70, 0.0];
            p[THUMB_TIP] = [0.50, 0.68, 0.0];
        }
        for (finger, &(mcp, mcp_idx)) in MCPS.iter().enumerate() {
            let (pip_i, dip_i, tip_i) = (mcp_idx + 1, mcp_idx + 2, mcp_idx + 3);
            p[mcp_idx] = mcp;
            if extended[finger + 1] {
                // Straight up from the MCP: zero joint bend.
                p[pip_i] = [mcp[0], mcp[1] - 0.07, 0.0];
                p[dip_i] = [mcp[0], mcp[1] - 0.12, 0.0];
                p[tip_i] = [mcp[0], mcp[1] - 0.16, 0.0];
            } else {
                // Folded: PIP up, then DIP/TIP back down toward the palm.
                p[pip_i] = [mcp[0], mcp[1] - 0.06, 0.0];
                p[dip_i] = [mcp[0] + 0.01, mcp[1] - 0.01, 0.0];
                p[tip_i] = [mcp[0] + 0.02, mcp[1] + 0.03, 0.0];
            }
        }
        p
    }

    fn frame(extended: [bool; 5], handedness: Handedness) -> HandFrame {
        HandFrame {
            handedness,
            confidence: 0.95,
            points: hand(extended),
        }
    }

    // --- finger_curls ----------------------------------------------------

    #[test]
    fn open_hand_reads_all_extended() {
        let curls = finger_curls(&hand([true; 5]));
        for (i, c) in curls.iter().enumerate() {
            assert!(*c < EXTEND_THRESHOLD, "finger {i} curl {c} not extended");
        }
    }

    #[test]
    fn fist_reads_all_curled() {
        let curls = finger_curls(&hand([false; 5]));
        for (i, c) in curls.iter().enumerate() {
            assert!(*c > CURL_THRESHOLD, "finger {i} curl {c} not curled");
        }
    }

    #[test]
    fn mixed_pattern_reads_per_finger() {
        // Victory: index+middle out, ring+pinky+thumb in.
        let curls = finger_curls(&hand([false, true, true, false, false]));
        assert!(curls[0] > CURL_THRESHOLD, "thumb {}", curls[0]);
        assert!(curls[1] < EXTEND_THRESHOLD, "index {}", curls[1]);
        assert!(curls[2] < EXTEND_THRESHOLD, "middle {}", curls[2]);
        assert!(curls[3] > CURL_THRESHOLD, "ring {}", curls[3]);
        assert!(curls[4] > CURL_THRESHOLD, "pinky {}", curls[4]);
    }

    /// The cues must be scale- and translation-free: a shifted, half-size
    /// hand produces the same curls.
    #[test]
    fn curls_are_scale_invariant() {
        let base = hand([false, true, false, false, true]);
        let mut scaled = base;
        for p in scaled.iter_mut() {
            p[0] = 0.1 + (p[0] - 0.5) * 0.5;
            p[1] = 0.2 + (p[1] - 0.7) * 0.5;
        }
        let a = finger_curls(&base);
        let b = finger_curls(&scaled);
        for i in 0..5 {
            assert!(
                (a[i] - b[i]).abs() < 1e-4,
                "finger {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn degenerate_all_zero_landmarks_do_not_panic() {
        let curls = finger_curls(&[[0.0; 3]; NUM_HAND_LANDMARKS]);
        for c in curls {
            assert!((0.0..=1.0).contains(&c));
        }
    }

    // --- hysteresis ------------------------------------------------------

    #[test]
    fn hysteresis_holds_state_in_the_dead_band() {
        let mut h = FingerHysteresis::default();
        let s = h.update([0.2; 5]);
        assert_eq!(s[0], FingerState::Extended);
        // Mid-band: state must hold.
        let s = h.update([0.5; 5]);
        assert_eq!(s[0], FingerState::Extended);
        // Crossing the high bar flips it.
        let s = h.update([0.7; 5]);
        assert_eq!(s[0], FingerState::Curled);
        // Mid-band again: still curled.
        let s = h.update([0.5; 5]);
        assert_eq!(s[0], FingerState::Curled);
    }

    #[test]
    fn hysteresis_starts_and_resets_to_unknown() {
        let mut h = FingerHysteresis::default();
        let s = h.update([0.5; 5]);
        assert_eq!(s[0], FingerState::Unknown, "mid-band from cold is unknown");
        h.update([0.9; 5]);
        h.reset();
        let s = h.update([0.5; 5]);
        assert_eq!(s[0], FingerState::Unknown, "reset forgets the latch");
    }

    // --- classify: all 8 gestures ---------------------------------------

    fn classify_hand(extended: [bool; 5]) -> Gesture {
        let mut h = FingerHysteresis::default();
        classify(&h.update(finger_curls(&hand(extended))))
    }

    #[test]
    fn classifies_all_eight_gestures() {
        assert_eq!(classify_hand([false; 5]), Gesture::Fist);
        assert_eq!(classify_hand([true; 5]), Gesture::HandOpen);
        // Open palm with the thumb tucked still reads HandOpen (table: `*`).
        assert_eq!(
            classify_hand([false, true, true, true, true]),
            Gesture::HandOpen
        );
        assert_eq!(
            classify_hand([false, true, false, false, false]),
            Gesture::FingerPoint
        );
        assert_eq!(
            classify_hand([false, true, true, false, false]),
            Gesture::Victory
        );
        // Victory with a splayed thumb is still Victory (table: `*`).
        assert_eq!(
            classify_hand([true, true, true, false, false]),
            Gesture::Victory
        );
        assert_eq!(
            classify_hand([false, true, false, false, true]),
            Gesture::RockNRoll
        );
        // RockNRoll with the thumb out is still RockNRoll (table: `*`).
        assert_eq!(
            classify_hand([true, true, false, false, true]),
            Gesture::RockNRoll
        );
        assert_eq!(
            classify_hand([true, true, false, false, false]),
            Gesture::HandGun
        );
        assert_eq!(
            classify_hand([true, false, false, false, false]),
            Gesture::ThumbsUp
        );
    }

    #[test]
    fn unmatched_patterns_read_neutral() {
        // Middle+ring only: not in the table.
        assert_eq!(
            classify_hand([false, false, true, true, false]),
            Gesture::Neutral
        );
        // All-unknown (cold start, mid-band curls) is Neutral.
        assert_eq!(classify(&[FingerState::Unknown; 5]), Gesture::Neutral);
    }

    /// HandGun vs FingerPoint differ only in the thumb; an *unknown* thumb
    /// with an index point must fall to FingerPoint (the `*`-thumb row),
    /// never HandGun (which requires a definitively extended thumb).
    #[test]
    fn unknown_thumb_point_is_fingerpoint_not_handgun() {
        use FingerState::{Curled as C, Extended as E, Unknown as U};
        assert_eq!(classify(&[U, E, C, C, C]), Gesture::FingerPoint);
        assert_eq!(classify(&[E, E, C, C, C]), Gesture::HandGun);
        assert_eq!(classify(&[C, E, C, C, C]), Gesture::FingerPoint);
    }

    // --- debounce --------------------------------------------------------

    #[test]
    fn debounce_delays_changes_by_debounce_frames() {
        let mut d = GestureDebounce::default();
        assert_eq!(d.feed(Gesture::Neutral), Gesture::Neutral);
        // A new gesture appears on its DEBOUNCE_FRAMES-th consecutive frame.
        for _ in 0..DEBOUNCE_FRAMES - 1 {
            assert_eq!(d.feed(Gesture::Fist), Gesture::Neutral);
        }
        assert_eq!(d.feed(Gesture::Fist), Gesture::Fist);
        // And stays.
        assert_eq!(d.feed(Gesture::Fist), Gesture::Fist);
    }

    #[test]
    fn debounce_suppresses_flicker() {
        let mut d = GestureDebounce::default();
        for _ in 0..DEBOUNCE_FRAMES {
            d.feed(Gesture::Fist);
        }
        assert_eq!(d.feed(Gesture::Fist), Gesture::Fist);
        // 1–3 frame blips of anything else never surface.
        for blip in [Gesture::HandOpen, Gesture::Neutral] {
            for _ in 0..DEBOUNCE_FRAMES - 1 {
                assert_eq!(d.feed(blip), Gesture::Fist, "blip leaked through");
            }
            assert_eq!(d.feed(Gesture::Fist), Gesture::Fist);
        }
        // An interrupted candidate restarts its count.
        d.feed(Gesture::HandOpen);
        d.feed(Gesture::HandOpen);
        d.feed(Gesture::Victory); // resets the HandOpen streak
        for _ in 0..DEBOUNCE_FRAMES - 1 {
            assert_eq!(d.feed(Gesture::HandOpen), Gesture::Fist);
        }
        assert_eq!(d.feed(Gesture::HandOpen), Gesture::HandOpen);
    }

    // --- GestureMapper ---------------------------------------------------

    fn int_value(params: &[OscParam], name: &str) -> i32 {
        let p = params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} missing"));
        match p.value {
            OscValue::Int(v) => v,
            ref other => panic!("{name} is not an int: {other:?}"),
        }
    }

    /// Feed the same hands enough frames to clear the debounce.
    fn settle(mapper: &mut GestureMapper, hands: &[HandFrame]) -> Vec<OscParam> {
        let mut out = mapper.map(hands);
        for _ in 0..DEBOUNCE_FRAMES {
            out = mapper.map(hands);
        }
        out
    }

    #[test]
    fn no_hands_emit_neutral_on_both_sides() {
        let mut mapper = GestureMapper::new(GestureMapperConfig::default());
        let params = mapper.map(&[]);
        assert_eq!(int_value(&params, "VCO_GestureLeft"), 0);
        assert_eq!(int_value(&params, "VCO_GestureRight"), 0);
        assert_eq!(int_value(&params, "GestureLeft"), 0);
        assert_eq!(int_value(&params, "GestureRight"), 0);
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn emit_native_off_drops_the_native_addresses() {
        let mut mapper = GestureMapper::new(GestureMapperConfig {
            emit_native: false,
            ..Default::default()
        });
        let params = mapper.map(&[]);
        assert_eq!(params.len(), 2);
        assert!(params.iter().all(|p| p.name.starts_with("VCO_")));
    }

    /// `mirror = true` (default, raw webcam): the model labels an unmirrored
    /// image's hands with the selfie convention, so its "Right" is the
    /// user's left hand and must land on the Left params.
    #[test]
    fn mirror_maps_model_right_to_gesture_left() {
        let mut mapper = GestureMapper::new(GestureMapperConfig::default());
        let fist = frame([false; 5], Handedness::Right);
        let params = settle(&mut mapper, std::slice::from_ref(&fist));
        assert_eq!(int_value(&params, "VCO_GestureLeft"), 1, "fist on left");
        assert_eq!(int_value(&params, "VCO_GestureRight"), 0);
        assert_eq!(int_value(&params, "GestureLeft"), 1);
    }

    #[test]
    fn mirror_off_uses_the_model_label_directly() {
        let mut mapper = GestureMapper::new(GestureMapperConfig {
            mirror: false,
            ..Default::default()
        });
        let fist = frame([false; 5], Handedness::Right);
        let params = settle(&mut mapper, std::slice::from_ref(&fist));
        assert_eq!(int_value(&params, "VCO_GestureRight"), 1);
        assert_eq!(int_value(&params, "VCO_GestureLeft"), 0);
    }

    #[test]
    fn two_hands_drive_both_sides_independently() {
        let mut mapper = GestureMapper::new(GestureMapperConfig::default());
        let hands = [
            frame([false; 5], Handedness::Right), // user's left: fist
            frame([true, true, false, false, false], Handedness::Left), // user's right: gun
        ];
        let params = settle(&mut mapper, &hands);
        assert_eq!(int_value(&params, "VCO_GestureLeft"), 1);
        assert_eq!(int_value(&params, "VCO_GestureRight"), 6);
    }

    #[test]
    fn double_booked_side_keeps_the_more_confident_hand() {
        let mut mapper = GestureMapper::new(GestureMapperConfig {
            mirror: false,
            ..Default::default()
        });
        let mut weak = frame([false; 5], Handedness::Left); // fist
        weak.confidence = 0.6;
        let strong = frame([true; 5], Handedness::Left); // open, confidence 0.95
        let params = settle(&mut mapper, &[weak, strong]);
        assert_eq!(int_value(&params, "VCO_GestureLeft"), 2, "open hand wins");
    }

    #[test]
    fn losing_the_hand_returns_to_neutral_after_debounce() {
        let mut mapper = GestureMapper::new(GestureMapperConfig::default());
        let fist = frame([false; 5], Handedness::Right);
        let params = settle(&mut mapper, std::slice::from_ref(&fist));
        assert_eq!(int_value(&params, "VCO_GestureLeft"), 1);

        // Hand gone: holds Fist through the debounce window, then Neutral.
        for _ in 0..DEBOUNCE_FRAMES - 1 {
            let params = mapper.map(&[]);
            assert_eq!(int_value(&params, "VCO_GestureLeft"), 1);
        }
        let params = mapper.map(&[]);
        assert_eq!(int_value(&params, "VCO_GestureLeft"), 0);
    }

    /// End-to-end flicker: a steady fist with single-frame open-hand
    /// glitches never leaves Fist.
    #[test]
    fn single_frame_glitches_do_not_change_the_gesture() {
        let mut mapper = GestureMapper::new(GestureMapperConfig::default());
        let fist = frame([false; 5], Handedness::Right);
        let open = frame([true; 5], Handedness::Right);
        settle(&mut mapper, std::slice::from_ref(&fist));
        for _ in 0..3 {
            let params = mapper.map(std::slice::from_ref(&open));
            assert_eq!(int_value(&params, "VCO_GestureLeft"), 1);
            for _ in 0..3 {
                let params = mapper.map(std::slice::from_ref(&fist));
                assert_eq!(int_value(&params, "VCO_GestureLeft"), 1);
            }
        }
    }

    #[test]
    fn gesture_names_cover_the_table() {
        assert_eq!(gesture_name(0), "Neutral");
        assert_eq!(gesture_name(1), "Fist");
        assert_eq!(gesture_name(2), "HandOpen");
        assert_eq!(gesture_name(3), "FingerPoint");
        assert_eq!(gesture_name(4), "Victory");
        assert_eq!(gesture_name(5), "RockNRoll");
        assert_eq!(gesture_name(6), "HandGun");
        assert_eq!(gesture_name(7), "ThumbsUp");
        assert_eq!(gesture_name(42), "?");
    }
}
