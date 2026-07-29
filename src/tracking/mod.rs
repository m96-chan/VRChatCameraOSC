//! Face and hand tracking behind pluggable backend boundaries.
//!
//! The (only) face backend is the MediaPipe stack ported from AvataCam —
//! [`mediapipe::MediapipeTracker`], implementing
//! [`arkit::ArkitFaceTracker`]: YuNet face detection → FaceMesh V2 (478 3-D
//! landmarks) → Blendshape V2 (52 ARKit coefficients) + per-axis head
//! rotation. The former FAN/S3FD 68-landmark backend was retired in issue
//! #21; the trait boundary stays so a different expression model can slot in
//! without churn (CLAUDE.md "models are pluggable").
//!
//! Hands (issue #8): [`hands::HandsTracker`], implementing
//! [`hands::HandTracker`] — MediaPipe palm detection → rotated ROI → 21
//! hand landmarks + handedness, feeding `mapping::gesture`.
//!
//! Pose (issue #28 phase 2): [`pose::BlazePoseTracker`], implementing
//! [`pose::PoseTracker`] — MediaPipe person detection → rotated ROI → 33
//! BlazePose landmarks + visibility, feeding `mapping::arm`.

pub mod arkit;
#[cfg(feature = "mesh-gpu")]
pub mod burn_onnx;
pub mod hands;
pub mod mediapipe;
pub mod pose;
