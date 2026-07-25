//! Persisted configuration (TOML). Implemented by issue #7.

use serde::{Deserialize, Serialize};

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
