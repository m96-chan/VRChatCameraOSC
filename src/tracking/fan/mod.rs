//! FAN (Face Alignment Network) — pure-Rust inference via candle.
//!
//! Ported from `1adrianb/face-alignment` (`models/fan.py`). The default 2D
//! model is `2DFAN4` (`num_modules = 4`): input `1x3x256x256`, output four
//! stages of `1x68x64x64` heatmaps; the last stage is decoded to landmarks.
//!
//! Layer names mirror the PyTorch module names exactly so weights exported from
//! the reference load 1:1 (see `reference/gen_fixtures.py`).

// Implemented by issue #4 (FAN tracker + PyTorch parity harness).
pub mod decode;
pub mod preprocess;
