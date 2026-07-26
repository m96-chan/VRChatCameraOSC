//! Integration tests for the TOML config module (issue #7).
//!
//! `tempfile` is not a dependency, so each test that touches the filesystem
//! writes into a unique subdirectory of `std::env::temp_dir()` keyed by pid +
//! test name, and removes it on the way out.

use std::path::PathBuf;

use vrchat_camera_osc::config::{Config, ConfigOverrides};

/// A unique, self-cleaning temp directory for a single test.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "vrchat-camera-osc-test-{}-{}",
            std::process::id(),
            name
        ));
        // Start clean in case a previous run crashed before cleanup.
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn join(&self, rel: &str) -> PathBuf {
        self.path.join(rel)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn round_trip_default() {
    let cfg = Config::default();
    let toml = toml::to_string_pretty(&cfg).expect("serialize");
    let back: Config = toml::from_str(&toml).expect("deserialize");
    assert_eq!(cfg, back);
}

#[test]
fn partial_toml_fills_defaults() {
    let toml = "[osc]\nport = 9001\n";
    let cfg: Config = toml::from_str(toml).expect("deserialize partial");
    assert_eq!(cfg.osc.port, 9001);
    // Everything else falls back to defaults.
    assert_eq!(cfg.osc.host, "127.0.0.1");
    assert!(!cfg.osc.dry_run);
    assert_eq!(cfg.camera, Default::default());
    assert_eq!(cfg.tracking, Default::default());
}

#[test]
fn empty_toml_is_all_defaults() {
    let cfg: Config = toml::from_str("").expect("deserialize empty");
    assert_eq!(cfg, Config::default());
}

#[test]
fn load_missing_file_errors() {
    let dir = TempDir::new("load_missing");
    let path = dir.join("does-not-exist.toml");
    assert!(Config::load(&path).is_err());
}

#[test]
fn load_or_create_creates_then_loads() {
    let dir = TempDir::new("load_or_create");
    // Nested path exercises parent-dir creation.
    let path = dir.join("nested/dir/config.toml");
    assert!(!path.exists());

    let created = Config::load_or_create(&path).expect("create");
    assert!(path.exists(), "file should have been written");
    assert_eq!(created, Config::default());

    // Second call should load the existing file (not overwrite).
    let loaded = Config::load_or_create(&path).expect("load");
    assert_eq!(loaded, Config::default());
    assert_eq!(loaded, created);
}

#[test]
fn load_or_create_preserves_edited_file() {
    let dir = TempDir::new("preserve_edits");
    let path = dir.join("config.toml");

    let mut cfg = Config::default();
    cfg.osc.host = "192.168.0.5".to_string();
    cfg.osc.port = 9100;
    cfg.save(&path).expect("save");

    let loaded = Config::load_or_create(&path).expect("load existing");
    assert_eq!(loaded.osc.host, "192.168.0.5");
    assert_eq!(loaded.osc.port, 9100);
}

#[test]
fn save_then_load_round_trip() {
    let dir = TempDir::new("save_load");
    let path = dir.join("config.toml");

    let mut cfg = Config::default();
    cfg.camera.device_index = 2;
    cfg.tracking.smoothing = 0.25;
    cfg.save(&path).expect("save");

    let loaded = Config::load(&path).expect("load");
    assert_eq!(loaded, cfg);
}

#[test]
fn invalid_toml_errors() {
    let dir = TempDir::new("invalid_toml");
    let path = dir.join("config.toml");
    std::fs::write(&path, "this is = = not valid toml ][").expect("write");
    assert!(Config::load(&path).is_err());
}

#[test]
fn validate_accepts_defaults() {
    assert!(Config::default().validate().is_ok());
}

#[test]
fn validate_rejects_port_zero() {
    let mut cfg = Config::default();
    cfg.osc.port = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_rejects_out_of_range_smoothing() {
    let mut cfg = Config::default();
    cfg.tracking.smoothing = 1.5;
    assert!(cfg.validate().is_err());

    let mut cfg = Config::default();
    cfg.tracking.smoothing = -0.1;
    assert!(cfg.validate().is_err());
}

#[test]
fn validate_accepts_smoothing_bounds() {
    let mut cfg = Config::default();
    cfg.tracking.smoothing = 0.0;
    assert!(cfg.validate().is_ok());
    cfg.tracking.smoothing = 1.0;
    assert!(cfg.validate().is_ok());
}

#[test]
fn validate_rejects_zero_dimensions() {
    let mut cfg = Config::default();
    cfg.camera.width = 0;
    assert!(cfg.validate().is_err());

    let mut cfg = Config::default();
    cfg.camera.height = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn apply_overrides_precedence() {
    // Start from a "file" state that differs from defaults.
    let mut cfg = Config::default();
    cfg.osc.host = "10.0.0.1".to_string();
    cfg.osc.port = 8000;
    cfg.osc.dry_run = false;
    cfg.camera.device_index = 1;

    let overrides = ConfigOverrides {
        osc_host: Some("127.0.0.1".to_string()),
        osc_port: Some(9000),
        dry_run: Some(true),
        camera_index: Some(3),
    };
    cfg.apply_overrides(&overrides);

    // CLI overrides win.
    assert_eq!(cfg.osc.host, "127.0.0.1");
    assert_eq!(cfg.osc.port, 9000);
    assert!(cfg.osc.dry_run);
    assert_eq!(cfg.camera.device_index, 3);
}

#[test]
fn apply_overrides_none_keeps_file_values() {
    let mut cfg = Config::default();
    cfg.osc.host = "10.0.0.1".to_string();
    cfg.osc.port = 8000;
    cfg.camera.device_index = 1;

    let overrides = ConfigOverrides::default();
    cfg.apply_overrides(&overrides);

    // Nothing overridden -> file/default values survive.
    assert_eq!(cfg.osc.host, "10.0.0.1");
    assert_eq!(cfg.osc.port, 8000);
    assert!(!cfg.osc.dry_run);
    assert_eq!(cfg.camera.device_index, 1);
}

#[test]
fn apply_overrides_partial() {
    let mut cfg = Config::default();
    let overrides = ConfigOverrides {
        osc_port: Some(9500),
        ..Default::default()
    };
    cfg.apply_overrides(&overrides);
    assert_eq!(cfg.osc.port, 9500);
    // Untouched fields keep defaults.
    assert_eq!(cfg.osc.host, "127.0.0.1");
}

// The `default_path` env-var assertions all live in one test so they cannot
// race each other under the parallel test runner (env vars are process-global).
#[test]
fn default_path_respects_env() {
    let prev_xdg = std::env::var_os("XDG_CONFIG_HOME");
    let prev_home = std::env::var_os("HOME");

    // XDG_CONFIG_HOME takes precedence.
    std::env::set_var("XDG_CONFIG_HOME", "/tmp/xdg-config");
    let p = Config::default_path();
    assert_eq!(
        p,
        PathBuf::from("/tmp/xdg-config/vrchat-camera-osc/config.toml")
    );

    // Falls back to $HOME/.config when XDG unset.
    std::env::remove_var("XDG_CONFIG_HOME");
    std::env::set_var("HOME", "/tmp/home-dir");
    let p = Config::default_path();
    assert_eq!(
        p,
        PathBuf::from("/tmp/home-dir/.config/vrchat-camera-osc/config.toml")
    );

    // Restore environment for other tests.
    match prev_xdg {
        Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
        None => std::env::remove_var("XDG_CONFIG_HOME"),
    }
    match prev_home {
        Some(v) => std::env::set_var("HOME", v),
        None => std::env::remove_var("HOME"),
    }
}

#[test]
fn tracking_backend_defaults_to_mediapipe() {
    use vrchat_camera_osc::config::TrackingBackend;
    assert_eq!(
        Config::default().tracking.backend,
        TrackingBackend::Mediapipe
    );
    // A config file written before the backend field existed still loads.
    let cfg: Config = toml::from_str("[tracking]\nsmoothing = 0.5\n").unwrap();
    assert_eq!(cfg.tracking.backend, TrackingBackend::Mediapipe);
}

// Avatar-aware vrcft keys (issue #18 phases 2+3): [osc] listen_port and
// [mapping] avatar_config_dir.
#[test]
fn listen_port_defaults_to_9001_and_is_configurable() {
    assert_eq!(Config::default().osc.listen_port, 9001);
    // A config file written before the field existed still loads.
    let cfg: Config = toml::from_str("[osc]\nport = 9000\n").unwrap();
    assert_eq!(cfg.osc.listen_port, 9001);
    // 0 = disabled is a valid setting.
    let cfg: Config = toml::from_str("[osc]\nlisten_port = 0\n").unwrap();
    assert_eq!(cfg.osc.listen_port, 0);
    assert!(
        cfg.validate().is_ok(),
        "listen_port 0 means disabled, not invalid"
    );
    let cfg: Config = toml::from_str("[osc]\nlisten_port = 9051\n").unwrap();
    assert_eq!(cfg.osc.listen_port, 9051);
}

// OSCQuery (issue #18 phase 3b): [osc] oscquery, default ON.
#[test]
fn oscquery_defaults_on_and_is_configurable() {
    assert!(Config::default().osc.oscquery);
    // A config file written before the field existed still loads (default).
    let cfg: Config = toml::from_str("[osc]\nport = 9000\n").unwrap();
    assert!(cfg.osc.oscquery);
    let cfg: Config = toml::from_str("[osc]\noscquery = false\n").unwrap();
    assert!(!cfg.osc.oscquery);
}

#[test]
fn avatar_config_dir_defaults_to_none_and_round_trips() {
    assert_eq!(Config::default().mapping.avatar_config_dir, None);
    let cfg: Config =
        toml::from_str("[mapping]\navatar_config_dir = \"/tmp/osc-configs\"\n").unwrap();
    assert_eq!(
        cfg.mapping.avatar_config_dir.as_deref(),
        Some("/tmp/osc-configs")
    );
    // Round-trips through save/load (Option field serialization).
    let dir = TempDir::new("avatar-config-dir-roundtrip");
    let path = dir.join("config.toml");
    cfg.save(&path).unwrap();
    let loaded = Config::load(&path).unwrap();
    assert_eq!(
        loaded.mapping.avatar_config_dir.as_deref(),
        Some("/tmp/osc-configs")
    );
}

#[test]
fn tracking_backend_fan_is_selectable() {
    use vrchat_camera_osc::config::TrackingBackend;
    let cfg: Config = toml::from_str("[tracking]\nbackend = \"fan\"\n").unwrap();
    assert_eq!(cfg.tracking.backend, TrackingBackend::Fan);
    // And it round-trips through save/load.
    let dir = TempDir::new("backend-roundtrip");
    let path = dir.join("config.toml");
    cfg.save(&path).unwrap();
    let loaded = Config::load(&path).unwrap();
    assert_eq!(loaded.tracking.backend, TrackingBackend::Fan);
}
