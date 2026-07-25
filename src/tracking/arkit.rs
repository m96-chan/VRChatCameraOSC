//! Shared types for ARKit-style expression tracking backends (issue #17).
//!
//! The MediaPipe backend (`tracking::mediapipe`, ported from AvataCam) emits a
//! fundamentally richer signal than the FAN 68-landmark backend: 52
//! ARKit-aligned blendshape coefficients predicted by a dedicated network,
//! plus a head rotation derived from 478 3-D landmarks. These types are that
//! backend-independent signal; `mapping::arkit` consumes them to produce the
//! 10 OSC parameters.
//!
//! Kept separate from [`super::FaceLandmarks`] on purpose: geometric 68-point
//! landmarks and NN-predicted expression coefficients are different signals,
//! and squeezing both through one trait would force lossy conversions.

use crate::capture::Frame;
use anyhow::Result;

/// The 52 ARKit-aligned blendshape coefficients (MediaPipe Blendshape V2
/// output; index 0 is `_neutral`). See [`ARKIT_BLENDSHAPE_NAMES`].
pub const NUM_BLENDSHAPES: usize = 52;

/// Output order of the 52 blendshapes, matching MediaPipe Blendshape V2
/// (index 0 = `_neutral`). Source: MediaPipe `face_blendshapes_graph.cc`,
/// mirrored from AvataCam `crates/face`.
pub const ARKIT_BLENDSHAPE_NAMES: [&str; NUM_BLENDSHAPES] = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
];

/// 52 ARKit-aligned blendshape coefficients in [`ARKIT_BLENDSHAPE_NAMES`]
/// order. Each is roughly in `[0, 1]` (the raw model output has a non-zero
/// resting baseline per channel — `mapping::arkit` calibrates that away).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArkitBlendshapes(pub [f32; NUM_BLENDSHAPES]);

impl ArkitBlendshapes {
    /// Look up a coefficient by its ARKit name (case-sensitive), or `None`.
    pub fn get(&self, name: &str) -> Option<f32> {
        ARKIT_BLENDSHAPE_NAMES
            .iter()
            .position(|&n| n == name)
            .map(|i| self.0[i])
    }
}

/// Head rotation relative to facing the camera, in **radians**, decomposed
/// per axis in AvataCam's world convention (X right, Y up, Z toward viewer):
///
/// - `pitch`: about X — positive = looking **up**.
/// - `yaw`: about Y — positive = turning toward the subject's **left**.
/// - `roll`: about Z — positive = tilting toward the subject's **right**
///   (counter-clockwise as seen by the viewer).
///
/// `mapping::arkit` normalises these to the `-1..1` OSC ranges (and any sign
/// flips the avatar side needs happen there, in one documented place).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HeadPose {
    pub roll: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// One frame of ARKit-style face tracking: the 52 coefficients plus head pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArkitFaceFrame {
    pub blendshapes: ArkitBlendshapes,
    pub head: HeadPose,
}

/// An expression tracker producing ARKit-style frames (the MediaPipe backend
/// implements this; mirrors [`super::FaceTracker`] for the 68-point world).
pub trait ArkitFaceTracker {
    fn track(&mut self, frame: &Frame) -> Result<Option<ArkitFaceFrame>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blendshape_name_table_is_52_and_unique() {
        assert_eq!(ARKIT_BLENDSHAPE_NAMES.len(), NUM_BLENDSHAPES);
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[0], "_neutral");
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[9], "eyeBlinkLeft");
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[10], "eyeBlinkRight");
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[25], "jawOpen");
        let mut seen = std::collections::HashSet::new();
        for n in ARKIT_BLENDSHAPE_NAMES {
            assert!(seen.insert(n), "duplicate blendshape name {n}");
        }
    }

    #[test]
    fn get_looks_up_by_name() {
        let mut a = [0.0f32; NUM_BLENDSHAPES];
        a[25] = 0.8; // jawOpen
        a[9] = 0.5; // eyeBlinkLeft
        let bs = ArkitBlendshapes(a);
        assert_eq!(bs.get("jawOpen"), Some(0.8));
        assert_eq!(bs.get("eyeBlinkLeft"), Some(0.5));
        assert_eq!(bs.get("nope"), None);
    }
}
