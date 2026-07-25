//! Shared helpers for the FAN parity tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Repo-root-relative path to a fixture file.
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Path to a file under `models/`.
pub fn model(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// Read a raw little-endian f32 binary file into a `Vec<f32>`.
pub fn read_f32(path: &Path) -> Vec<f32> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "fixture {} not f32-aligned",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Max absolute difference between two equal-length slices.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Mean absolute difference between two equal-length slices.
pub fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}
