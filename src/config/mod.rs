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
    pub tracking: TrackingConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OscConfig {
    pub host: String,
    pub port: u16,
    /// When true, don't send — only monitor (dry-run).
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub device_index: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrackingConfig {
    /// Exponential smoothing factor in 0..1 (higher = snappier, less smooth).
    pub smoothing: f32,
}

impl Default for OscConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9000,
            dry_run: false,
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

impl Default for TrackingConfig {
    fn default() -> Self {
        Self { smoothing: 0.5 }
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
    #[error("tracking.smoothing must be within 0.0..=1.0 (got {0})")]
    SmoothingOutOfRange(String),
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
        if !(0.0..=1.0).contains(&self.tracking.smoothing) {
            return Err(
                ConfigError::SmoothingOutOfRange(self.tracking.smoothing.to_string()).into(),
            );
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
