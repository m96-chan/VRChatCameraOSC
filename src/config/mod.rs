//! Persisted configuration (TOML). Implemented by issue #7.
//!
//! # Design decisions (CLAUDE.md rule 8)
//!
//! - **Format:** TOML. It is human-editable, comment-friendly, already a
//!   dependency (`toml`), and maps cleanly onto the sectioned schema
//!   (`[osc]`, `[camera]`, `[tracking]`). Every field is `#[serde(default)]`
//!   so a missing or *partial* file still deserializes — absent keys fall back
//!   to the type's `Default`, satisfying the "partial config → defaults"
//!   acceptance criterion.
//! - **Location:** the platform config directory, resolved with `std` only (no
//!   `directories`/`dirs` crate):
//!   `$XDG_CONFIG_HOME/vrchat-camera-osc/config.toml` when `XDG_CONFIG_HOME` is
//!   set, otherwise `$HOME/.config/vrchat-camera-osc/config.toml`. If `HOME` is
//!   also unset, the path is resolved relative to the current directory so the
//!   app never hard-fails on a headless/degraded environment.
//! - **Precedence:** **CLI flags > config file > defaults.** Startup loads the
//!   file (or writes defaults if missing) via [`Config::load_or_create`], then
//!   applies any CLI overrides via [`Config::apply_overrides`]. Only
//!   explicitly-provided overrides (`Some(..)`) win; `None` leaves the
//!   file/default value untouched.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The application name used for the config subdirectory.
const APP_DIR: &str = "vrchat-camera-osc";
/// The config file name within the app directory.
const CONFIG_FILE: &str = "config.toml";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub osc: OscConfig,
    pub camera: CameraConfig,
    pub mapping: MappingConfig,
    pub eye: EyeConfig,
    pub hands: HandsConfig,
    pub pose: PoseConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OscConfig {
    pub host: String,
    pub port: u16,
    /// When true, don't send — only monitor (dry-run).
    pub dry_run: bool,
    /// UDP port to receive VRChat's OSC output on (`/avatar/change`, used by
    /// the avatar-aware `vrcft` profile — issue #18 phase 3). VRChat sends
    /// to 9001 by default; `0` disables the listener entirely.
    pub listen_port: u16,
    /// OSCQuery discovery + advertisement (issue #18 phase 3b, `vrcft`
    /// profile): browse mDNS for the VRChat client and fetch the avatar's
    /// declared parameters over HTTP (covers VRChat on another machine),
    /// and advertise our own `/avatar/change` listener so VRChat finds us
    /// with zero setup. Default on; turn off for manually-configured ports.
    /// CLI override: `--oscquery on|off`.
    pub oscquery: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub device_index: u32,
    pub width: u32,
    pub height: u32,
}

/// Native VRChat eye-tracking OSC (`/tracking/eye/*`, issue #19): drives the
/// avatar's existing eye bones directly — no avatar parameters needed. Works
/// alongside either mapping profile; mediapipe backend only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EyeConfig {
    /// Send `/tracking/eye/LeftRightPitchYaw` + `EyesClosedAmount`.
    pub native: bool,
    /// Degrees of eye yaw at a saturated (1.0) look coefficient.
    pub yaw_range_deg: f32,
    /// Degrees of eye pitch at a saturated (1.0) look coefficient.
    pub pitch_range_deg: f32,
}

impl Default for EyeConfig {
    fn default() -> Self {
        let range = crate::mapping::eye::EyeRange::default();
        Self {
            native: true,
            yaw_range_deg: range.yaw_deg,
            pitch_range_deg: range.pitch_deg,
        }
    }
}

/// Hand tracking → VRChat gestures (issue #8): the MediaPipe palm-detection
/// + hand-landmark stack feeding `mapping::gesture`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HandsConfig {
    /// Run the hand tracker (models auto-download like the face stack).
    /// CLI override: `--hands on|off`.
    pub enabled: bool,
    /// Handedness mirroring (see `mapping::gesture::GestureMapperConfig`):
    /// `true` (default, raw webcam) puts the user's left hand on
    /// `GestureLeft`.
    pub mirror: bool,
    /// Maximum simultaneously tracked hands (clamped to >= 1 at build).
    pub max_hands: u32,
    /// Also emit the native `GestureLeft`/`GestureRight` addresses
    /// (read-only in current VRChat, but forward-compatible — issue #8 §1).
    pub emit_native: bool,
    /// Run the hand stage every N pipeline frames (1 = every frame — the
    /// issue-#8 default decision; raise to shed load if combined FPS drops).
    pub interval: u32,
    /// Emit the Tier-2 per-finger curl floats (`VCO_L_ThumbCurl` ..
    /// `VCO_R_LittleCurl` — issue #8 phase 3). Emitting is harmless (VRChat
    /// ignores undeclared parameters); the `unity/` wizard controls whether
    /// the avatar declares them.
    pub finger_curls: bool,
    /// **Deprecated** (issue #28 phase 2): the arm floats no longer ride the
    /// hand stack — they come from the pose stack, controlled by `[pose]
    /// enabled`. This key is tolerated so pre-phase-2 config files still
    /// load, but it has no effect (precedence: `[pose] enabled` alone
    /// decides whether arm parameters are emitted).
    pub arms: bool,
}

impl Default for HandsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mirror: true,
            max_hands: 2,
            emit_native: true,
            interval: 1,
            finger_curls: true,
            arms: true,
        }
    }
}

/// Full-arm tracking → the `VCO_*_Arm*`/`VCO_*_Elbow` parameters (issue #28
/// phase 2): the MediaPipe person-detection + BlazePose stack feeding
/// `mapping::arm`. The pose landmark net runs every frame alongside face and
/// hands; person detection is a rare async safety net. Left/right honors the
/// shared `[hands] mirror` flag (one camera, one mirroring convention).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PoseConfig {
    /// Run the pose tracker (models auto-download like the face stack).
    /// CLI override: `--pose on|off`. This is the sole switch for the arm
    /// parameters — the legacy `[hands] arms` key is ignored.
    pub enabled: bool,
    /// Safety-net person-redetect cadence in pipeline frames (~1/s at 30
    /// FPS by default; clamped to >= 1 at build).
    pub redetect_interval: u32,
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            redetect_interval: crate::tracking::pose::DEFAULT_REDETECT_INTERVAL,
        }
    }
}

/// Output mapping settings. Since issue #21 there is a single output format:
/// the VRCFT-compatible Unified Expressions `v2/` parameter set (issue #18) —
/// the former `custom10` profile (and its `profile` key) is retired.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MappingConfig {
    /// Address prefixes for the fallback output: each parameter is sent once
    /// per prefix (`""` → `v2/JawOpen`, `"FT/"` → `FT/v2/JawOpen`, ...).
    /// Since issue #18 phase 3 this is the **no-avatar-config fallback
    /// only**: when the worn avatar's OSC config is discovered, output is
    /// gated to the declared parameters at their exact addresses instead.
    /// VRChat ignores undeclared addresses.
    pub vrcft_prefixes: Vec<String>,
    /// Explicit directory to read VRChat's per-avatar OSC config JSONs from
    /// (`.../VRChat/VRChat/OSC` or any level of that tree). Unset → the
    /// per-platform default locations (Windows `%USERPROFILE%`, Linux
    /// Steam/Proton prefix — see `mapping::avatar::default_osc_dirs`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avatar_config_dir: Option<String>,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self {
            vrcft_prefixes: crate::mapping::unified::DEFAULT_PREFIXES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            avatar_config_dir: None,
        }
    }
}

impl Default for OscConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9000,
            dry_run: false,
            listen_port: 9001,
            oscquery: true,
        }
    }
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            width: 640,
            height: 480,
        }
    }
}

/// Optional CLI-supplied overrides applied on top of the loaded config.
///
/// Precedence is **CLI (these) > config file > defaults**: a `Some(_)` field
/// replaces whatever the file/defaults provided; a `None` leaves it untouched.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConfigOverrides {
    pub osc_host: Option<String>,
    pub osc_port: Option<u16>,
    pub dry_run: Option<bool>,
    pub camera_index: Option<u32>,
}

/// Errors produced while validating a [`Config`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("OSC port must be non-zero")]
    ZeroPort,
    #[error("camera dimensions must be > 0 (got {width}x{height})")]
    ZeroDimensions { width: u32, height: u32 },
}

impl Config {
    /// Load a config from `path`, deserializing TOML.
    ///
    /// A missing file is an error here — use [`Config::load_or_create`] when a
    /// missing file should be materialized from defaults instead.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Config> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading config {}: {e}", path.display()))?;
        let cfg: Config = toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing config {}: {e}", path.display()))?;
        Ok(cfg)
    }

    /// Load `path` if it exists; otherwise write the default config there
    /// (creating parent directories) and return the defaults.
    pub fn load_or_create(path: impl AsRef<Path>) -> anyhow::Result<Config> {
        let path = path.as_ref();
        if path.exists() {
            Self::load(path)
        } else {
            let cfg = Config::default();
            cfg.save(path)?;
            Ok(cfg)
        }
    }

    /// Serialize to pretty TOML and write to `path`, creating parent dirs.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    anyhow::anyhow!("creating config dir {}: {e}", parent.display())
                })?;
            }
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)
            .map_err(|e| anyhow::anyhow!("writing config {}: {e}", path.display()))?;
        Ok(())
    }

    /// The default on-disk config path.
    ///
    /// `$XDG_CONFIG_HOME/vrchat-camera-osc/config.toml` when `XDG_CONFIG_HOME`
    /// is set, else `$HOME/.config/vrchat-camera-osc/config.toml`. Falls back to
    /// a current-directory-relative `.config/...` when neither is set.
    pub fn default_path() -> PathBuf {
        let base = if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty())
        {
            PathBuf::from(xdg)
        } else if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            PathBuf::from(home).join(".config")
        } else {
            PathBuf::from(".config")
        };
        base.join(APP_DIR).join(CONFIG_FILE)
    }

    /// Apply CLI overrides in place. Precedence: **CLI > file > defaults**.
    pub fn apply_overrides(&mut self, overrides: &ConfigOverrides) {
        if let Some(host) = &overrides.osc_host {
            self.osc.host = host.clone();
        }
        if let Some(port) = overrides.osc_port {
            self.osc.port = port;
        }
        if let Some(dry_run) = overrides.dry_run {
            self.osc.dry_run = dry_run;
        }
        if let Some(index) = overrides.camera_index {
            self.camera.device_index = index;
        }
    }

    /// Validate invariants, returning a clear error on the first violation.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.osc.port == 0 {
            return Err(ConfigError::ZeroPort.into());
        }
        if self.camera.width == 0 || self.camera.height == 0 {
            return Err(ConfigError::ZeroDimensions {
                width: self.camera.width,
                height: self.camera.height,
            }
            .into());
        }
        Ok(())
    }
}
