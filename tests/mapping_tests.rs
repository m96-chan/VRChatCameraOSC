//! Integration tests for the landmark -> VRChat OSC parameter mapping (issue #5).
//!
//! Each test builds a synthetic [`FaceLandmarks`] from a known neutral geometry
//! (helpers below place the 68 iBUG points at fixed coordinates), optionally
//! deforms it (open the mouth, close an eye, tilt the head), and asserts the
//! resulting parameters behave as the mapping design promises.

use approx::assert_abs_diff_eq;
use vrchat_camera_osc::mapping::Mapper;
use vrchat_camera_osc::osc::{OscParam, OscValue};
use vrchat_camera_osc::tracking::{FaceLandmarks, Landmark, NUM_LANDMARKS};

/// Build a neutral, forward-facing synthetic face.
///
/// Geometry (image coords, +y down):
/// - eyes at y=50, right-eye centre (40,50), left-eye centre (60,50) => IOD 20
/// - brows at y=40 (10px above the eyes)
/// - nose bridge 27..30 straight down the midline, tip (50,54)
/// - mouth centred on the midline, inner lips almost touching (nearly closed)
fn neutral_points() -> Vec<Landmark> {
    let mut p = vec![
        Landmark {
            x: 0.0,
            y: 0.0,
            score: 1.0
        };
        NUM_LANDMARKS
    ];
    let mut set = |i: usize, x: f32, y: f32| {
        p[i] = Landmark { x, y, score: 1.0 };
    };

    // Jaw 0..16 — not used by the mapping; place along a lower arc.
    for i in 0..=16 {
        set(i, 30.0 + i as f32 * 2.5, 80.0);
    }

    // Right brow 17..21, left brow 22..26 — mean y = 40.
    set(17, 34.0, 40.0);
    set(18, 37.0, 40.0);
    set(19, 40.0, 40.0);
    set(20, 43.0, 40.0);
    set(21, 46.0, 40.0);
    set(22, 54.0, 40.0);
    set(23, 57.0, 40.0);
    set(24, 60.0, 40.0);
    set(25, 63.0, 40.0);
    set(26, 66.0, 40.0);

    // Nose bridge 27..30 (tip 30) and nose bottom 31..35.
    set(27, 50.0, 45.0);
    set(28, 50.0, 48.0);
    set(29, 50.0, 51.0);
    set(30, 50.0, 54.0); // tip
    set(31, 46.0, 56.0);
    set(32, 48.0, 57.0);
    set(33, 50.0, 58.0);
    set(34, 52.0, 57.0);
    set(35, 54.0, 56.0);

    // Right eye 36..41 (corners 36,39; upper 37,38; lower 41,40). Centre (40,50).
    set(36, 34.0, 50.0);
    set(37, 37.0, 47.0);
    set(38, 43.0, 47.0);
    set(39, 46.0, 50.0);
    set(40, 43.0, 53.0);
    set(41, 37.0, 53.0);

    // Left eye 42..47 (corners 42,45; upper 43,44; lower 47,46). Centre (60,50).
    set(42, 54.0, 50.0);
    set(43, 57.0, 47.0);
    set(44, 63.0, 47.0);
    set(45, 66.0, 50.0);
    set(46, 63.0, 53.0);
    set(47, 57.0, 53.0);

    // Outer mouth 48..59 (corners 48,54; top 51; bottom 57).
    set(48, 44.0, 65.0);
    set(49, 46.0, 62.0);
    set(50, 48.0, 61.0);
    set(51, 50.0, 60.0);
    set(52, 52.0, 61.0);
    set(53, 54.0, 62.0);
    set(54, 56.0, 65.0);
    set(55, 54.0, 68.0);
    set(56, 52.0, 69.0);
    set(57, 50.0, 70.0);
    set(58, 48.0, 69.0);
    set(59, 46.0, 68.0);

    // Inner mouth 60..67 (corners 60,64; top 62; bottom 66). Nearly closed.
    set(60, 45.0, 65.0);
    set(61, 47.0, 64.5);
    set(62, 50.0, 64.5); // inner top
    set(63, 53.0, 64.5);
    set(64, 55.0, 65.0);
    set(65, 53.0, 65.5);
    set(66, 50.0, 65.5); // inner bottom
    set(67, 47.0, 65.5);

    p
}

fn neutral_face() -> FaceLandmarks {
    FaceLandmarks::new(neutral_points()).unwrap()
}

fn face_from(points: Vec<Landmark>) -> FaceLandmarks {
    FaceLandmarks::new(points).unwrap()
}

/// Extract the float value of a named parameter (panics if missing / not float).
fn get(params: &[OscParam], name: &str) -> f32 {
    let p = params
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("missing param {name}"));
    match p.value {
        OscValue::Float(v) => v,
        other => panic!("param {name} is not a float: {other:?}"),
    }
}

/// Set the inner-lip vertical gap (points 62 top / 66 bottom) about the midline.
fn set_mouth_gap(points: &mut [Landmark], gap: f32) {
    let mid = 65.0;
    points[62].y = mid - gap / 2.0;
    points[66].y = mid + gap / 2.0;
}

/// Collapse one eye's lids together (upper onto lower) to simulate a blink.
fn close_eye(points: &mut [Landmark], upper: [usize; 2], lower: [usize; 2]) {
    for (u, l) in upper.iter().zip(lower.iter()) {
        let midy = (points[*u].y + points[*l].y) / 2.0;
        points[*u].y = midy;
        points[*l].y = midy;
    }
}

#[test]
fn emits_all_ten_named_params_with_finite_values() {
    let mut m = Mapper::new();
    let out = m.map(&neutral_face());
    let expected = [
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
    for name in expected {
        let v = get(&out, name);
        assert!(v.is_finite(), "{name} not finite: {v}");
    }
    for p in &out {
        assert!(!p.name.is_empty(), "empty param name");
    }
    assert_eq!(out.len(), expected.len());
}

#[test]
fn mouth_open_high_when_open_low_when_closed() {
    let mut m = Mapper::new();

    let mut open = neutral_points();
    set_mouth_gap(&mut open, 10.0);
    let open_v = get(&m.map(&face_from(open)), "MouthOpen");

    let mut closed = neutral_points();
    set_mouth_gap(&mut closed, 0.0);
    let mut m2 = Mapper::new();
    let closed_v = get(&m2.map(&face_from(closed)), "MouthOpen");

    assert!(open_v > 0.8, "open mouth should be high, got {open_v}");
    assert!(closed_v < 0.1, "closed mouth should be low, got {closed_v}");
    assert!(open_v > closed_v);
}

#[test]
fn blink_high_when_eye_closed_low_when_open_independent_sides() {
    // Neutral (both open) -> both blinks low.
    let mut m = Mapper::new();
    let open = m.map(&neutral_face());
    assert!(get(&open, "EyeBlinkLeft") < 0.1);
    assert!(get(&open, "EyeBlinkRight") < 0.1);

    // Close only the RIGHT eye (points 36..41: upper 37,38 / lower 41,40).
    let mut r = neutral_points();
    close_eye(&mut r, [37, 38], [41, 40]);
    let mut mr = Mapper::new();
    let out = mr.map(&face_from(r));
    assert!(
        get(&out, "EyeBlinkRight") > 0.9,
        "right eye should read closed, got {}",
        get(&out, "EyeBlinkRight")
    );
    assert!(
        get(&out, "EyeBlinkLeft") < 0.1,
        "left eye should stay open, got {}",
        get(&out, "EyeBlinkLeft")
    );

    // Close only the LEFT eye (points 42..47: upper 43,44 / lower 47,46).
    let mut l = neutral_points();
    close_eye(&mut l, [43, 44], [47, 46]);
    let mut ml = Mapper::new();
    let out = ml.map(&face_from(l));
    assert!(get(&out, "EyeBlinkLeft") > 0.9);
    assert!(get(&out, "EyeBlinkRight") < 0.1);
}

#[test]
fn brow_up_rises_when_brows_raised_both_sides() {
    let mut m = Mapper::new();
    let out = m.map(&neutral_face());
    let (neutral_l, neutral_r) = (get(&out, "BrowUpLeft"), get(&out, "BrowUpRight"));

    // Raise both brows (move them further above the eyes: smaller y).
    let mut raised = neutral_points();
    for lm in &mut raised[17..=26] {
        lm.y -= 6.0;
    }
    let mut m2 = Mapper::new();
    let out2 = m2.map(&face_from(raised));
    let (raised_l, raised_r) = (get(&out2, "BrowUpLeft"), get(&out2, "BrowUpRight"));

    assert!(
        neutral_l < 0.1,
        "neutral left brow should be ~0, got {neutral_l}"
    );
    assert!(
        neutral_r < 0.1,
        "neutral right brow should be ~0, got {neutral_r}"
    );
    assert!(raised_l > neutral_l && raised_r > neutral_r);
    assert!(
        raised_l > 0.5 && raised_r > 0.5,
        "raised brows should be substantial, got left={raised_l} right={raised_r}"
    );
}

#[test]
fn brow_up_left_and_right_are_independent() {
    // Raise only the LEFT brow (22..26).
    let mut left_only = neutral_points();
    for lm in &mut left_only[22..=26] {
        lm.y -= 6.0;
    }
    let mut m = Mapper::new();
    let out = m.map(&face_from(left_only));
    assert!(
        get(&out, "BrowUpLeft") > 0.5,
        "left brow should read raised, got {}",
        get(&out, "BrowUpLeft")
    );
    assert!(
        get(&out, "BrowUpRight") < 0.1,
        "right brow should stay neutral, got {}",
        get(&out, "BrowUpRight")
    );

    // Raise only the RIGHT brow (17..21).
    let mut right_only = neutral_points();
    for lm in &mut right_only[17..=21] {
        lm.y -= 6.0;
    }
    let mut m2 = Mapper::new();
    let out2 = m2.map(&face_from(right_only));
    assert!(get(&out2, "BrowUpRight") > 0.5);
    assert!(get(&out2, "BrowUpLeft") < 0.1);
}

#[test]
fn mouth_smile_rises_when_corners_lift() {
    let mut m = Mapper::new();
    let neutral = get(&m.map(&neutral_face()), "MouthSmile");
    assert!(neutral < 0.1, "neutral smile should be ~0, got {neutral}");

    // Lift both mouth corners (48, 54) up (smaller y) toward the mouth centre.
    let mut smiling = neutral_points();
    smiling[48].y -= 5.0;
    smiling[54].y -= 5.0;
    let mut m2 = Mapper::new();
    let smile_v = get(&m2.map(&face_from(smiling)), "MouthSmile");

    assert!(smile_v > neutral, "lifted corners should raise MouthSmile");
    assert!((0.0..=1.0).contains(&smile_v));
}

#[test]
fn mouth_wide_signed_for_stretch_vs_pucker() {
    let mut m = Mapper::new();
    let neutral = get(&m.map(&neutral_face()), "MouthWide");
    assert_abs_diff_eq!(neutral, 0.0, epsilon = 0.15);

    // Stretch the corners apart horizontally -> positive.
    let mut wide = neutral_points();
    wide[48].x -= 8.0;
    wide[54].x += 8.0;
    let mut mw = Mapper::new();
    let wide_v = get(&mw.map(&face_from(wide)), "MouthWide");
    assert!(
        wide_v > 0.1,
        "stretched mouth should be positive, got {wide_v}"
    );

    // Pull the corners together (pucker) -> negative.
    let mut pucker = neutral_points();
    pucker[48].x += 5.0;
    pucker[54].x -= 5.0;
    let mut mp = Mapper::new();
    let pucker_v = get(&mp.map(&face_from(pucker)), "MouthWide");
    assert!(
        pucker_v < -0.1,
        "puckered mouth should be negative, got {pucker_v}"
    );
    assert!((-1.0..=1.0).contains(&wide_v) && (-1.0..=1.0).contains(&pucker_v));
}

#[test]
fn head_roll_sign_and_magnitude_track_eye_line_tilt() {
    // Level face -> roll ~0.
    let mut m = Mapper::new();
    let level = get(&m.map(&neutral_face()), "HeadRoll");
    assert_abs_diff_eq!(level, 0.0, epsilon = 1e-3);

    // Tilt: push the right eye up and the left eye down so the right->left
    // eye vector gains positive dy (positive roll by our convention).
    let mut tilt = neutral_points();
    for lm in &mut tilt[36..=41] {
        lm.y -= 4.0;
    }
    for lm in &mut tilt[42..=47] {
        lm.y += 4.0;
    }
    let mut mt = Mapper::new();
    let pos = get(&mt.map(&face_from(tilt)), "HeadRoll");
    assert!(pos > 0.1, "tilt should give positive roll, got {pos}");
    assert!((-1.0..=1.0).contains(&pos));

    // Opposite tilt -> negative roll.
    let mut tilt2 = neutral_points();
    for lm in &mut tilt2[36..=41] {
        lm.y += 4.0;
    }
    for lm in &mut tilt2[42..=47] {
        lm.y -= 4.0;
    }
    let mut mt2 = Mapper::new();
    let neg = get(&mt2.map(&face_from(tilt2)), "HeadRoll");
    assert!(
        neg < -0.1,
        "opposite tilt should give negative roll, got {neg}"
    );
    assert_abs_diff_eq!(pos, -neg, epsilon = 1e-4);
}

#[test]
fn extreme_geometry_stays_within_range() {
    // Wildly deformed face: giant mouth, closed eyes, brows off the top,
    // strong tilt and off-centre nose.
    let mut p = neutral_points();
    set_mouth_gap(&mut p, 40.0);
    close_eye(&mut p, [37, 38], [41, 40]);
    close_eye(&mut p, [43, 44], [47, 46]);
    for lm in &mut p[17..=26] {
        lm.y -= 40.0;
    }
    for lm in &mut p[36..=41] {
        lm.y -= 30.0;
    }
    for lm in &mut p[42..=47] {
        lm.y += 30.0;
    }
    p[30].x += 40.0; // yank the nose tip sideways
    p[30].y += 40.0; // and down (pitch)

    let mut m = Mapper::new();
    let out = m.map(&face_from(p));

    for name in [
        "MouthOpen",
        "EyeBlinkLeft",
        "EyeBlinkRight",
        "BrowUpLeft",
        "BrowUpRight",
        "MouthSmile",
    ] {
        let v = get(&out, name);
        assert!((0.0..=1.0).contains(&v), "{name}={v} out of 0..1");
    }
    for name in ["MouthWide", "HeadRoll", "HeadYaw", "HeadPitch"] {
        let v = get(&out, name);
        assert!((-1.0..=1.0).contains(&v), "{name}={v} out of -1..1");
    }
}

#[test]
fn degenerate_landmarks_produce_bounded_finite_output() {
    // All points coincide -> zero interocular distance; mapping must not blow up.
    let pts = vec![
        Landmark {
            x: 5.0,
            y: 5.0,
            score: 1.0
        };
        NUM_LANDMARKS
    ];
    let mut m = Mapper::new();
    let out = m.map(&face_from(pts));
    assert_eq!(out.len(), 10);
    for p in &out {
        if let OscValue::Float(v) = p.value {
            assert!(v.is_finite());
            assert!((-1.0..=1.0).contains(&v));
        }
    }
}

#[test]
fn smoothing_converges_from_zero_toward_target() {
    let alpha = 0.5_f32;

    // Target value for a fully open mouth, measured with no smoothing.
    let mut open = neutral_points();
    set_mouth_gap(&mut open, 10.0);
    let target = get(&Mapper::new().map(&face_from(open.clone())), "MouthOpen");
    assert!(target > 0.8);

    let mut m = Mapper::with_smoothing(alpha);

    // First output must sit strictly between the initial prev (0) and target.
    let first = get(&m.map(&face_from(open.clone())), "MouthOpen");
    assert!(
        first > 0.0 && first < target,
        "first={first} target={target}"
    );
    assert_abs_diff_eq!(first, alpha * target, epsilon = 1e-5);

    // Successive identical inputs converge monotonically toward the target.
    let mut prev = first;
    for _ in 0..25 {
        let v = get(&m.map(&face_from(open.clone())), "MouthOpen");
        assert!(v >= prev - 1e-6, "should be non-decreasing: {v} < {prev}");
        assert!(v <= target + 1e-6, "should not overshoot: {v} > {target}");
        prev = v;
    }
    assert_abs_diff_eq!(prev, target, epsilon = 1e-2);
}

#[test]
fn default_mapper_applies_no_smoothing() {
    // Mapper::new() should pass geometry straight through (alpha = 1.0), so a
    // single call already yields the full target on the first frame.
    let mut open = neutral_points();
    set_mouth_gap(&mut open, 10.0);
    let a = get(&Mapper::new().map(&face_from(open.clone())), "MouthOpen");
    let b = get(&Mapper::new().map(&face_from(open)), "MouthOpen");
    assert_abs_diff_eq!(a, b, epsilon = 1e-6);
    assert!(a > 0.8);
}

/// A "neutral" pose whose brows sit lower and mouth corners sit lower than
/// `neutral_points()` assumes — simulating a real face/camera whose resting
/// geometry doesn't match the synthetic defaults (issue #15: this is exactly
/// what left `BrowUpLeft`/`BrowUpRight`/`MouthSmile` clamped to 0 on a real
/// webcam despite genuinely responding to expression changes).
fn biased_neutral_points() -> Vec<Landmark> {
    let mut p = neutral_points();
    for lm in &mut p[17..=26] {
        lm.y += 4.0; // brows sit closer to the eyes than the synthetic default
    }
    p[48].y += 3.0; // mouth corners rest lower than the synthetic default
    p[54].y += 3.0;
    p
}

#[test]
fn calibrate_zeroes_previously_biased_params_at_the_calibrated_pose() {
    let biased = biased_neutral_points();

    // Before calibration: the biased pose reads negative (clamped to 0) for
    // all three, exactly the reported "stuck at 0" symptom.
    let before = Mapper::new().map(&face_from(biased.clone()));
    assert_eq!(get(&before, "BrowUpLeft"), 0.0);
    assert_eq!(get(&before, "BrowUpRight"), 0.0);
    assert_eq!(get(&before, "MouthSmile"), 0.0);

    let mut m = Mapper::new();
    let samples: Vec<FaceLandmarks> = (0..5).map(|_| face_from(biased.clone())).collect();
    m.calibrate(&samples);

    // After calibrating on that exact pose, mapping it again should read ~0
    // (the new neutral), not clamped-away-from-negative.
    let after = m.map(&face_from(biased));
    assert_abs_diff_eq!(get(&after, "BrowUpLeft"), 0.0, epsilon = 1e-4);
    assert_abs_diff_eq!(get(&after, "BrowUpRight"), 0.0, epsilon = 1e-4);
    assert_abs_diff_eq!(get(&after, "MouthSmile"), 0.0, epsilon = 1e-4);
}

#[test]
fn calibrate_then_further_movement_still_reads_positive() {
    let biased = biased_neutral_points();
    let mut m = Mapper::new();
    let samples: Vec<FaceLandmarks> = (0..5).map(|_| face_from(biased.clone())).collect();
    m.calibrate(&samples);

    // Raise both brows further from the *new* (biased) baseline.
    let mut raised = biased.clone();
    for lm in &mut raised[17..=26] {
        lm.y -= 6.0;
    }
    let raised_out = m.map(&face_from(raised));
    assert!(
        get(&raised_out, "BrowUpLeft") > 0.3,
        "got {}",
        get(&raised_out, "BrowUpLeft")
    );
    assert!(
        get(&raised_out, "BrowUpRight") > 0.3,
        "got {}",
        get(&raised_out, "BrowUpRight")
    );
}

#[test]
fn calibrate_with_no_usable_samples_is_a_noop() {
    let degenerate = vec![
        Landmark {
            x: 5.0,
            y: 5.0,
            score: 1.0,
        };
        NUM_LANDMARKS
    ];

    let mut m1 = Mapper::new();
    let before = m1.map(&neutral_face());

    let mut m2 = Mapper::new();
    m2.calibrate(&[face_from(degenerate)]);
    let after = m2.map(&neutral_face());

    for name in ["BrowUpLeft", "BrowUpRight", "MouthSmile", "MouthOpen"] {
        assert_abs_diff_eq!(get(&before, name), get(&after, name), epsilon = 1e-6);
    }
}
