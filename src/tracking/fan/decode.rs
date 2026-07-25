//! Heatmap → landmark decoding.
//! Ported from `face_alignment/utils.py` (`get_preds_fromhm`): argmax over each
//! 64x64 heatmap plus a ±0.25 sub-pixel refinement toward the higher neighbour.

// Implemented by issue #4.
