//! Tracked face → VRChat avatar OSC parameters.
//!
//! Since issue #21 there is a single avatar-parameter format: the
//! VRCFT-compatible **Unified Expressions `v2/`** set (issue #18), emitted by
//! [`unified::UnifiedMapper`] — the same parameters VRCFaceTracking sends, so
//! any FT-ready avatar (including one set up by the `unity/` wizard) works
//! without app-specific wiring. The former `custom10` 10-parameter format and
//! its FAN/68-landmark geometric mapper are retired.
//!
//! - [`unified`] — ARKit-52 coefficients → Unified Expressions `v2/*` floats
//!   (+ avatar-aware gating and binary parameter encoding, issue #18).
//! - [`avatar`] — the worn avatar's declared OSC parameter set: config-file
//!   discovery, `/avatar/change` tracking, OSCQuery fetches.
//! - [`eye`] — VRChat's native `/tracking/eye/*` endpoints (issue #19),
//!   orthogonal to the parameter mapping.
//! - [`arkit`] — shared calibration/smoothing machinery (One-Euro filters,
//!   neutral baselines, deadzone), ported from AvataCam.
//! - [`gesture`] — 21 hand landmarks → VRChat gesture ints
//!   (`VCO_GestureLeft/Right` + optional native addresses, issue #8) and
//!   Tier-2 per-finger curl floats (`VCO_*Curl`, issue #8 phase 3).
//! - [`arm`] — wrist height → `VCO_{L,R}_ArmUpDown` arm-raise floats
//!   (issue #28 phase 1).

pub mod arkit;
pub mod arm;
pub mod avatar;
pub mod eye;
pub mod gesture;
pub mod unified;
