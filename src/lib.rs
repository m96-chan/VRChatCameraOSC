//! VRChatCameraOSC — capture a face from a webcam, track landmarks, and stream
//! avatar parameters to VRChat over OSC.
//!
//! The crate is split into clear module boundaries so platform- and
//! model-specific pieces stay swappable (see `CLAUDE.md`):
//!
//! - [`capture`] — frame source (webcam) behind the [`capture::CameraSource`] trait.
//! - [`tracking`] — pluggable face model behind the [`tracking::FaceTracker`] trait.
//! - [`mapping`] — landmarks → VRChat avatar parameters.
//! - [`osc`]     — OSC message building + UDP transport / dry-run monitor.
//! - [`config`]  — persisted settings.
//! - [`tui`]     — terminal UI state (the primary control surface).

pub mod capture;
pub mod config;
pub mod mapping;
pub mod osc;
pub mod tracking;
pub mod tui;
