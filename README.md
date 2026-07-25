# VRChatCameraOSC

Realtime face tracking for VRChat, driven from your webcam and delivered over OSC — written in Rust for a fast, low-latency loop, with a TUI front end.

## Overview

`VRChatCameraOSC` captures your face from a camera, extracts a face mesh, and streams the resulting expression / head-pose parameters to VRChat over OSC so your own avatar moves in realtime. No GUI window — everything runs in a terminal UI.

```
┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────┐
│  Camera  │──▶│  Face Mesh   │──▶│ OSC Parameters│──▶│ VRChat  │
│ capture  │   │  tracking    │   │  (UDP/OSC)    │   │ avatar  │
└──────────┘   └──────────────┘   └───────────────┘   └─────────┘
                         ▲
                   ┌───────────┐
                   │    TUI    │  status / config / live values
                   └───────────┘
```

## Features

- **Camera capture & tracking** — grab frames from a webcam and track the face in realtime.
- **Face mesh** — extract facial landmarks (face mesh) to derive expression and pose.
- **Hand tracking** — track hand/finger landmarks and drive avatar gestures. *(planned)*
- **OSC output** — connect to VRChat over OSC and move your own avatar.
- **Realtime face tracking → avatar** — map tracked landmarks to avatar parameters live.
- **TUI** — monitor and control everything from the terminal.

## Tech

- **Language:** Rust — chosen for a fast, predictable realtime loop.
- **UI:** TUI (terminal user interface).
- **Transport:** OSC over UDP to VRChat.

## Status

🚧 Early development. Interfaces and parameters are subject to change. The full
pipeline (capture → detect → face mesh → mapping → OSC) is wired end-to-end,
including face auto-crop (S3FD) before FAN. Model weights auto-download on
first run. Hand/finger tracking is the main remaining gap (see Roadmap).

## Architecture

Each stage sits behind a trait so platform- and model-specific pieces stay
swappable:

| Stage | Module | Notes |
|-------|--------|-------|
| Capture | `capture::CameraSource` | `capture::native::NativeCamera` — AVFoundation on macOS, V4L2 on Linux (`Windows` planned), via `nokhwa`; synthetic `FakeCamera` for tests/headless |
| Detection | `tracking::detect::FaceDetector` | `sfd::SfdDetector` — the **S3FD** face detector ported to candle; auto-crops before FAN |
| Tracking | `tracking::FaceTracker` | `fan::FanTracker` — the face-alignment **2DFAN4** net ported to pure-Rust **candle** |
| Mapping | `mapping::Mapper` | iBUG-68 landmarks → 10 normalised avatar params (mouth open/smile/wide, per-eye blink, per-brow raise, head pose), clamped + smoothed |
| OSC | `osc::OscSink` | `UdpOscSender` to VRChat, or `MonitorSink` dry-run |
| Loop | `pipeline::Pipeline` | `capture → track → map → OSC`, driven by the TUI or headless monitor |
| Models | `models::ensure_present` | auto-downloads `models/*.safetensors` from a GitHub Release on first run |

### Model & PyTorch parity

Face landmarks come from a candle port of [`1adrianb/face-alignment`](https://github.com/1adrianb/face-alignment)
(2DFAN4). The port is validated for **numeric parity** against the PyTorch
reference: pretrained weights are exported to safetensors and the Rust output is
compared to PyTorch on identical input. Observed agreement is to f32 precision
(full-network max abs diff ≈ 4e-7). See [`reference/`](reference/) and the
`fan_parity` / `fan_convblock_parity` / `fan_units` tests.

## Getting Started

> Requires a recent stable Rust toolchain ([rustup](https://rustup.rs/)).

```bash
# build
cargo build --release

# run (TUI)
cargo run --release

# headless OSC monitor / dry-run demo (no VRChat needed):
#   prints the /avatar/parameters/* it would send, using a synthetic camera
cargo run --release -- --monitor --fake --frames 20
```

The tracker loads FAN weights from `models/2dfan4.safetensors` and the S3FD
detector from `models/s3fd.safetensors`. **Both auto-download on first run**
(from this repo's [`models-v1` release](https://github.com/m96-chan/VRChatCameraOSC/releases/tag/models-v1))
if missing — no Python/PyTorch needed. Offline, or if the download fails, the
loop still runs with tracking disabled and prints a clear message.

Contributors validating the candle port against PyTorch (numeric-parity
tests) still use the reference harness to regenerate these from the original
pretrained weights — see [`reference/README.md`](reference/README.md):

```bash
cd reference && uv venv --python 3.11
uv pip install --python .venv "torch>=2.2" "numpy<2" safetensors face-alignment scikit-image
.venv/bin/python gen_fixtures.py   # downloads + converts the pretrained weights
```

CLI flags: `--monitor` (headless), `--fake` (synthetic camera), `--frames N`
(stop after N frames), `--weights PATH`, `--detector PATH` (a custom
`--weights`/`--detector` path is never auto-downloaded), `--detect-interval N`
(see Performance below), `--calibrate-frames N` (see Calibration below; `0`
skips calibration).

VRChat must have **OSC enabled** (Action Menu → Options → OSC → Enabled) for the
end-to-end avatar path.

### Neutral-pose calibration

The mapping's "neutral" reference points (what counts as a relaxed brow, a
flat mouth, etc.) are calibrated against a synthetic test face by default,
which doesn't match every real face/camera — this can leave a parameter
permanently reading `0` even while it's genuinely responding to expression
changes (issue #15). To fix this, the app captures a short window at startup
(`--calibrate-frames N`, default 10 frames) — **hold a relaxed, forward-facing
expression** while it says `calibrating neutral pose ...` — and re-derives the
neutral baselines from what it actually sees. In the TUI, press **`c`** at any
time to redo it (camera angle or lighting changed, or you weren't ready the
first time).

### Performance

`candle` builds with no acceleration backend by default (CPU fallback only),
which — combined with running the S3FD detector on every frame — measured
**~0.5 FPS** on a 16-core desktop CPU (issue #13). Two independent
optimizations address this:

- **Detect-then-track (always on, default `--detect-interval 8`):** the S3FD
  detector is by far the most expensive stage, and a face doesn't move far
  frame-to-frame, so it only re-runs every `detect_interval` frames; the
  frames in between derive their crop box from the previous frame's FAN
  landmarks instead. It always re-detects immediately after losing the face,
  regardless of interval. This alone roughly **doubles** throughput on CPU
  (S3FD's ~1.3s/frame vs. FAN's ~0.6s/frame, measured on the reference
  hardware above).
- **Opt-in `cuda` feature:** `cargo build --release --features cuda` runs
  inference on an NVIDIA GPU (`Device::cuda_if_available` — falls back to CPU
  automatically when the feature is off or no GPU is found). Requires the
  CUDA toolkit to *build*, not to run a non-cuda build. Measured **~29.5 FPS**
  (capture-rate-limited) on an RTX 5090, vs. ~0.5 FPS CPU-only — **not**
  part of `default`, since macOS has no CUDA and most end users won't have an
  NVIDIA GPU + toolkit to build against.

## Avatar setup (required for the avatar to actually move)

VRChat's OSC layer only **writes values into parameters your avatar already
defines** — it doesn't create facial expressions on its own, and none of
VRChat's built-in avatar parameters (`GestureLeft/Right`, `Viseme`, `AFK`,
etc.) cover facial expression or head pose. A stock/default avatar, or any
avatar that hasn't been specifically wired up, **will not react** to this
app's output. To make an avatar respond, add these as **Float** VRC
Expression Parameters (Unity + VRChat SDK3) and drive blend shapes / bones
from them in the FX Animator Controller:

| OSC address | Range | Meaning |
|---|---|---|
| `/avatar/parameters/MouthOpen` | `0..1` | mouth opening amount |
| `/avatar/parameters/EyeBlinkLeft` | `0..1` | left eye closed amount (`1` = closed) |
| `/avatar/parameters/EyeBlinkRight` | `0..1` | right eye closed amount |
| `/avatar/parameters/BrowUpLeft` | `0..1` | left eyebrow raise |
| `/avatar/parameters/BrowUpRight` | `0..1` | right eyebrow raise |
| `/avatar/parameters/MouthSmile` | `0..1` | mouth-corner vertical lift (smile) |
| `/avatar/parameters/MouthWide` | `-1..1` | mouth-corner horizontal stretch (`+` wide/grin, `−` pucker) |
| `/avatar/parameters/HeadRoll` | `-1..1` | head tilt |
| `/avatar/parameters/HeadYaw` | `-1..1` | head turn left/right |
| `/avatar/parameters/HeadPitch` | `-1..1` | head tilt up/down |

Full derivation (which landmarks, which formula) is documented in
[`src/mapping/mod.rs`](src/mapping/mod.rs). See VRChat's own docs for the
mechanics: [OSC Avatar Parameters](https://docs.vrchat.com/docs/osc-avatar-parameters),
[Avatar Animator Parameters](https://creators.vrchat.com/avatars/animator-parameters/).

## Configuration

OSC host/port, camera device, and tracking settings will be configurable from the TUI and/or a config file. (TBD — see Roadmap.)

## Roadmap

- [x] Camera capture pipeline
- [x] Face mesh landmark extraction (FAN / candle, PyTorch-parity verified)
- [x] Face detector (S3FD / candle) — auto-crop the face before FAN
- [ ] Hand / finger landmark extraction
- [x] Landmark → VRChat OSC parameter mapping
- [x] OSC sender (UDP) + dry-run monitor
- [x] TUI: live values, status, and configuration
- [x] Config file support

## License

TBD.
