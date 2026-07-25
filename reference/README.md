# FAN PyTorch reference harness

This directory holds the **PyTorch reference** used to validate the pure-Rust
(candle) FAN inference for numeric parity (issue #4). The Rust port is only
"correct" if it reproduces this reference's outputs.

Reference model: [`1adrianb/face-alignment`](https://github.com/1adrianb/face-alignment)
— the **2DFAN4** network (68 iBUG landmarks, `1x3x256x256` input, four
`1x68x64x64` heatmap stages).

## Setup

```bash
cd reference
uv venv --python 3.11
uv pip install --python .venv "torch>=2.2" "numpy<2" safetensors face-alignment scikit-image
```

## Generate fixtures

```bash
reference/.venv/bin/python reference/gen_fixtures.py
```

This writes:

| Path | Committed? | Used by |
|------|-----------|---------|
| `models/2dfan4.safetensors` | no (≈90 MB, gitignored) | `tests/fan_parity.rs` (full-network parity) |
| `tests/fixtures/fan_input.f32`, `fan_output.f32` | yes | full-network parity input / expected |
| `tests/fixtures/fan_preds.json` | yes | `tests/fan_units.rs` decode parity |
| `tests/fixtures/transform.json` | yes | `tests/fan_units.rs` transform parity |
| `tests/fixtures/convblock_small.*` | yes | `tests/fan_convblock_parity.rs` (CI, no download) |

The pretrained weights are downloaded on first run and converted to
safetensors; they are **not** committed. The small ConvBlock fixture *is*
committed so architecture parity runs in CI without any download.

## What the Rust tests assert

- `fan_convblock_parity` — candle `ConvBlock` vs PyTorch, tiny committed
  weights. Runs everywhere. Observed max abs diff ≈ 5e-7.
- `fan_units` — `transform()` bit-parity and heatmap→landmark decode parity
  (exact). Runs everywhere.
- `fan_parity` — **full-network** candle FAN vs PyTorch on identical weights +
  input. Skips when `models/2dfan4.safetensors` is absent. Observed max abs
  diff ≈ 4e-7, mean ≈ 2e-8 (i.e. parity to f32 precision).
