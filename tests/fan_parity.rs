//! Full-network numeric parity: candle FAN vs the PyTorch `face_alignment`
//! reference, on the same weights and the same deterministic input.
//!
//! Requires the pretrained weights `models/2dfan4.safetensors` (≈90 MB,
//! gitignored) and the input/output fixtures, all produced by
//! `reference/gen_fixtures.py`. When the weights are absent (e.g. a fresh CI
//! checkout) the test SKIPS with a notice rather than failing — the committed
//! `fan_convblock_parity` and `fan_units` tests still cover the port in CI.

mod common;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use vrchat_camera_osc::tracking::fan::model::Fan;

#[test]
fn fan_forward_matches_pytorch() {
    let weights = common::model("2dfan4.safetensors");
    if !weights.exists() {
        eprintln!(
            "SKIP fan_forward_matches_pytorch: {} not found. \
             Run `reference/.venv/bin/python reference/gen_fixtures.py` to generate it.",
            weights.display()
        );
        return;
    }

    let device = Device::Cpu;
    let input = common::read_f32(&common::fixture("fan_input.f32"));
    let expected = common::read_f32(&common::fixture("fan_output.f32"));
    assert_eq!(input.len(), 3 * 256 * 256);
    assert_eq!(expected.len(), 68 * 64 * 64);

    let x = Tensor::from_vec(input, (1, 3, 256, 256), &device).unwrap();
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device).unwrap() };
    let model = Fan::load(vb).unwrap();

    let out = model.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 68, 64, 64]);
    let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let max = common::max_abs_diff(&got, &expected);
    let mean = common::mean_abs_diff(&got, &expected);
    eprintln!("FAN full-network parity: max_abs_diff={max:.3e} mean_abs_diff={mean:.3e}");
    // Measured drift is ~4e-7 (f32 epsilon accumulated over the deep stacked-
    // hourglass net); these bounds catch any real regression while allowing it.
    assert!(max < 1e-4, "FAN max abs diff {max} exceeds tolerance");
    assert!(mean < 1e-5, "FAN mean abs diff {mean} exceeds tolerance");
}
