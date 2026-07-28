//! Face tracking behind a pluggable backend boundary.
//!
//! The (only) backend is the MediaPipe stack ported from AvataCam —
//! [`mediapipe::MediapipeTracker`], implementing
//! [`arkit::ArkitFaceTracker`]: YuNet face detection → FaceMesh V2 (478 3-D
//! landmarks) → Blendshape V2 (52 ARKit coefficients) + per-axis head
//! rotation. The former FAN/S3FD 68-landmark backend was retired in issue
//! #21; the trait boundary stays so a different expression model can slot in
//! without churn (CLAUDE.md "models are pluggable").

pub mod arkit;
#[cfg(feature = "mesh-gpu")]
pub mod burn_onnx;
pub mod mediapipe;
