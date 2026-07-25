//! The fixed 146-landmark subset the Blendshape V2 model consumes.
//!
//! Verbatim from MediaPipe `face_blendshapes_graph.cc` (`kLandmarksSubsetIdxs`),
//! ported from AvataCam `crates/face/src/backends/mediapipe/subset.rs`: the
//! blendshape model takes these 146 FaceMesh landmark indices (into the 478)
//! as `(x, y)` pairs, in this exact order. Indices >= 468 are iris points, so
//! the 478-landmark FaceMesh V2 output is required.

/// 146 landmark indices (into the 478 FaceMesh landmarks), in model input order.
pub const LANDMARKS_SUBSET: [usize; 146] = [
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65,
    66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136,
    144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172,
    173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283,
    284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
    332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385,
    386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472,
    473, 474, 475, 476, 477,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::mediapipe::NUM_FACE_LANDMARKS;

    #[test]
    fn subset_is_146_in_bounds_and_sane() {
        assert_eq!(LANDMARKS_SUBSET.len(), 146);
        for &i in &LANDMARKS_SUBSET {
            assert!(i < NUM_FACE_LANDMARKS, "index {i} out of range");
        }
        // Includes iris points (>=468), which only the 478-landmark model has.
        assert!(LANDMARKS_SUBSET.iter().any(|&i| i >= 468));
        assert_eq!(LANDMARKS_SUBSET[0], 0);
        assert_eq!(LANDMARKS_SUBSET[145], 477);
    }
}
