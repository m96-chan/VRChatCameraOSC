//! Avatar-aware VRCFT output — issue #18 phases 2+3.
//!
//! End-to-end wiring: fixture avatar OSC-config JSONs → [`AvatarWatcher`] →
//! ARKit pipeline with the `vrcft` [`UnifiedMapper`] → recording sink,
//! including a live `/avatar/change` OSC message over a real UDP socket.
//! Needs no webcam, no model weights, and no VRChat, so it runs in CI —
//! this is also the recorded rule-2 demo for the gating + binary path.

use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vrchat_camera_osc::capture::{FakeCamera, Frame};
use vrchat_camera_osc::mapping::arkit::ArkitMappingConfig;
use vrchat_camera_osc::mapping::avatar::AvatarWatcher;
use vrchat_camera_osc::mapping::unified::UnifiedMapper;
use vrchat_camera_osc::osc::{AvatarChangeListener, OscParam, OscSink, OscValue};
use vrchat_camera_osc::pipeline::Pipeline;
use vrchat_camera_osc::tracking::arkit::{
    ArkitBlendshapes, ArkitFaceFrame, ArkitFaceTracker, HeadPose, ARKIT_BLENDSHAPE_NAMES,
    NUM_BLENDSHAPES,
};

const FLOAT_AVATAR_ID: &str = "avtr_00000000-0000-4000-8000-00000000f10a";
const BINARY_AVATAR_ID: &str = "avtr_00000000-0000-4000-8000-00000000b14a";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/avatar_configs")
}

/// Copy the two fixture avatar configs into a fresh temp `OSC/usr_x/Avatars`
/// layout so file mtimes (and therefore the "most recent" startup pick) are
/// deterministic: `last` is written last and is the most recently modified.
fn temp_config_dir(first: &str, last: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vco-avatar-cfg-{}-{}",
        std::process::id(),
        first.len() * 1000 + last.len() // distinct per (first,last) pair
    ));
    let avatars = dir.join("usr_test").join("Avatars");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&avatars).unwrap();
    let src = fixture_dir().join("usr_test-user").join("Avatars");
    // Not fs::copy: on Windows it preserves the source's last-write time, so
    // copied fixtures would keep their checkout mtimes and the "written last
    // = newest" ordering below would not hold. read+write always stamps a
    // fresh mtime on every platform.
    let rewrite = |name: &str| {
        let bytes = std::fs::read(src.join(name)).unwrap();
        std::fs::write(avatars.join(name), bytes).unwrap();
    };
    rewrite(first);
    // Some filesystems have coarse mtime granularity; force an ordering gap.
    std::thread::sleep(Duration::from_millis(30));
    rewrite(last);
    dir
}

/// An ARKit tracker that always returns the same frame.
struct StubArkitTracker(ArkitFaceFrame);
impl ArkitFaceTracker for StubArkitTracker {
    fn track(&mut self, _frame: &Frame) -> anyhow::Result<Option<ArkitFaceFrame>> {
        Ok(Some(self.0))
    }
}

#[derive(Clone)]
struct RecordingSink(Arc<Mutex<Vec<OscParam>>>);
impl OscSink for RecordingSink {
    fn send(&mut self, params: &[OscParam]) -> anyhow::Result<()> {
        self.0.lock().unwrap().extend_from_slice(params);
        Ok(())
    }
}

/// A frame with the jaw fully open (1.0) and a strong symmetric frown (0.8):
/// drives `v2/JawOpen` to 1.0 (binary all-ones edge case) and `v2/SmileFrown`
/// to a negative value (binary Negative bit).
fn expressive_frame() -> ArkitFaceFrame {
    let mut bs = [0.0f32; NUM_BLENDSHAPES];
    let set = |bs: &mut [f32; NUM_BLENDSHAPES], name: &str, v: f32| {
        let idx = ARKIT_BLENDSHAPE_NAMES
            .iter()
            .position(|&n| n == name)
            .unwrap();
        bs[idx] = v;
    };
    set(&mut bs, "jawOpen", 1.0);
    set(&mut bs, "mouthFrownLeft", 0.8);
    set(&mut bs, "mouthFrownRight", 0.8);
    ArkitFaceFrame {
        blendshapes: ArkitBlendshapes(bs),
        head: HeadPose::default(),
    }
}

/// The `vrcft` mapper with baseline auto-calibration disabled so raw
/// blendshape values flow through (minus smoothing/deadzone) — keeps value
/// assertions deterministic.
fn mapper() -> UnifiedMapper {
    UnifiedMapper::new(
        ArkitMappingConfig {
            calib_frames: 0,
            ..Default::default()
        },
        Vec::new(),
    )
}

fn build_pipeline(watcher: AvatarWatcher) -> (Pipeline, Arc<Mutex<Vec<OscParam>>>) {
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let sink = RecordingSink(recorded.clone());
    let pipeline = Pipeline::new_arkit(
        Box::new(FakeCamera::new(64, 48)),
        Some(Box::new(StubArkitTracker(expressive_frame()))),
        Box::new(mapper()),
        Box::new(sink),
    )
    .with_avatar_watcher(watcher);
    (pipeline, recorded)
}

fn names(params: &[OscParam]) -> Vec<String> {
    params.iter().map(|p| p.name.clone()).collect()
}

fn bool_of(params: &[OscParam], addr: &str) -> bool {
    let p = params
        .iter()
        .find(|p| p.name == addr)
        .unwrap_or_else(|| panic!("{addr} should be present"));
    match p.value {
        OscValue::Bool(b) => b,
        _ => panic!("{addr} should be a bool"),
    }
}

// Startup with no /avatar/change yet: the most recently modified avatar json
// wins (best-effort), and only its declared params are sent, at their exact
// declared addresses.
#[test]
fn initial_load_picks_most_recent_config_and_gates_floats() {
    let dir = temp_config_dir("avtr_binary.json", "avtr_float.json");
    let watcher = AvatarWatcher::new(vec![dir], None);
    let (mut pipeline, _recorded) = build_pipeline(watcher);

    let out = pipeline.step().unwrap();
    let names = names(&out.params);

    // Exactly the declared float params at their declared (FT/-prefixed)
    // addresses, plus the two declared status bools — nothing else.
    let mut expected = vec![
        "/avatar/parameters/FT/v2/JawOpen",
        "/avatar/parameters/FT/v2/MouthClosed",
        "/avatar/parameters/FT/v2/SmileFrownRight",
        "/avatar/parameters/FT/v2/EyeLidLeft",
        "/avatar/parameters/EyeTrackingActive",
        "/avatar/parameters/ExpressionTrackingActive",
    ];
    expected.sort_unstable();
    let mut got = names.clone();
    got.sort_unstable();
    assert_eq!(got, expected, "only the avatar's declared params are sent");

    // The declared address is used verbatim (raw address, not re-prefixed).
    let jaw = out
        .params
        .iter()
        .find(|p| p.name == "/avatar/parameters/FT/v2/JawOpen")
        .unwrap();
    assert_eq!(jaw.address(), "/avatar/parameters/FT/v2/JawOpen");
    match jaw.value {
        OscValue::Float(v) => assert!(v > 0.9, "open jaw should read high: {v}"),
        _ => panic!("declared Float param must be sent as float"),
    }
    // LipTrackingActive is NOT declared by this avatar — it must be gated out.
    assert!(!names.iter().any(|n| n.contains("LipTrackingActive")));
}

// A live /avatar/change OSC message switches the active param set to the new
// avatar's config — here a binary-only avatar, exercising the bit encoding
// end-to-end (all-ones 1.0 edge case + negative value with Negative bit).
#[test]
fn avatar_change_switches_to_binary_param_set() {
    let dir = temp_config_dir("avtr_binary.json", "avtr_float.json");
    let listener = AvatarChangeListener::bind(0).unwrap(); // 0 = ephemeral for tests
    let port = listener.local_port();
    let watcher = AvatarWatcher::new(vec![dir], Some(listener));
    let (mut pipeline, _recorded) = build_pipeline(watcher);

    // Startup pick = float avatar.
    let out = pipeline.step().unwrap();
    assert!(names(&out.params).contains(&"/avatar/parameters/FT/v2/JawOpen".to_string()));

    // VRChat announces an avatar switch on its OSC output port.
    let client = UdpSocket::bind("127.0.0.1:0").unwrap();
    let msg = rosc::encoder::encode(&rosc::OscPacket::Message(rosc::OscMessage {
        addr: "/avatar/change".to_string(),
        args: vec![rosc::OscType::String(BINARY_AVATAR_ID.to_string())],
    }))
    .unwrap();
    client.send_to(&msg, ("127.0.0.1", port)).unwrap();

    // The pipeline polls the listener once per step; give the datagram a
    // moment to arrive.
    let deadline = Instant::now() + Duration::from_secs(2);
    let params = loop {
        let out = pipeline.step().unwrap();
        if names(&out.params).contains(&"/avatar/parameters/FT/v2/JawOpen1".to_string()) {
            break out.params;
        }
        assert!(Instant::now() < deadline, "avatar change never took effect");
        std::thread::sleep(Duration::from_millis(10));
    };

    // v2/JawOpen = 1.0 → BinaryBaseParameter's >0.99999 special case: every
    // bit reads 1 (4 declared bits: JawOpen1/2/4/8).
    for bit in ["JawOpen1", "JawOpen2", "JawOpen4", "JawOpen8"] {
        assert!(
            bool_of(&params, &format!("/avatar/parameters/FT/v2/{bit}")),
            "{bit} should be set for value 1.0"
        );
    }
    // The plain declared Bool v2/JawOpen carries VRCFT's EParam bool
    // semantics (value < 0.5) → false at 1.0.
    assert!(!bool_of(&params, "/avatar/parameters/FT/v2/JawOpen"));

    // v2/SmileFrown ≈ −0.79 (frown 0.8 through the 0.06 deadzone): with a
    // declared Negative bit the sign goes to SmileFrownNegative and the bits
    // encode |v|: floor(0.787 × 4) = 3 → both bits set.
    assert!(bool_of(
        &params,
        "/avatar/parameters/FT/v2/SmileFrownNegative"
    ));
    assert!(bool_of(&params, "/avatar/parameters/FT/v2/SmileFrown1"));
    assert!(bool_of(&params, "/avatar/parameters/FT/v2/SmileFrown2"));

    // Status bools follow the new avatar's declarations too.
    assert!(bool_of(&params, "/avatar/parameters/LipTrackingActive"));
    assert!(!names(&params)
        .iter()
        .any(|n| n.contains("EyeTrackingActive")));
    // No float params: this avatar declares none.
    assert!(params
        .iter()
        .all(|p| !matches!(p.value, OscValue::Float(_))));
}

// Switching to an avatar with no discoverable config falls back to the
// blind-prefix float behavior (exactly the phase-1 output).
#[test]
fn unknown_avatar_falls_back_to_blind_prefixes() {
    let dir = temp_config_dir("avtr_float.json", "avtr_binary.json");
    let listener = AvatarChangeListener::bind(0).unwrap();
    let port = listener.local_port();
    let watcher = AvatarWatcher::new(vec![dir], Some(listener));
    let (mut pipeline, _recorded) = build_pipeline(watcher);

    let out = pipeline.step().unwrap();
    assert!(
        !names(&out.params).contains(&"v2/JawOpen".to_string()),
        "gated mode must not emit blind-prefix names"
    );

    let client = UdpSocket::bind("127.0.0.1:0").unwrap();
    let msg = rosc::encoder::encode(&rosc::OscPacket::Message(rosc::OscMessage {
        addr: "/avatar/change".to_string(),
        args: vec![rosc::OscType::String(
            "avtr_ffffffff-dead-beef-0000-000000000000".into(),
        )],
    }))
    .unwrap();
    client.send_to(&msg, ("127.0.0.1", port)).unwrap();

    let deadline = Instant::now() + Duration::from_secs(2);
    let params = loop {
        let out = pipeline.step().unwrap();
        if names(&out.params).contains(&"v2/JawOpen".to_string()) {
            break out.params;
        }
        assert!(Instant::now() < deadline, "fallback never engaged");
        std::thread::sleep(Duration::from_millis(10));
    };
    // Blind mode: both default prefixes and the status bools, like phase 1.
    let names = names(&params);
    assert!(names.contains(&"FT/v2/JawOpen".to_string()));
    assert!(names.contains(&"EyeTrackingActive".to_string()));
}

// The fixture avatar id lookup itself (used on /avatar/change) reads the
// checked-in fixture tree, skipping the malformed json gracefully.
#[test]
fn find_by_id_reads_fixture_tree_and_skips_malformed_json() {
    use vrchat_camera_osc::mapping::avatar::find_config_by_id;
    let dirs = [fixture_dir()];
    let cfg = find_config_by_id(&dirs, FLOAT_AVATAR_ID).expect("float avatar found");
    assert_eq!(cfg.id, FLOAT_AVATAR_ID);
    assert_eq!(cfg.name, "Fixture Float FT Avatar");
    assert_eq!(cfg.parameters.len(), 8);
    let cfg = find_config_by_id(&dirs, BINARY_AVATAR_ID).expect("binary avatar found");
    assert_eq!(cfg.id, BINARY_AVATAR_ID);
    assert!(find_config_by_id(&dirs, "avtr_not-there").is_none());
}
