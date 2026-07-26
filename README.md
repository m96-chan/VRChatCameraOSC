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
- **VRCFaceTracking-compatible output** (`--mapping vrcft`) — emit the Unified
  Expressions `v2/` float parameters that VRCFT sends, so VRCFT-ready avatars
  work with **zero avatar-side setup** (issue #18; float params — phase 1).
- **TUI** — monitor and control everything from the terminal.

## Tech

- **Language:** Rust — chosen for a fast, predictable realtime loop.
- **UI:** TUI (terminal user interface).
- **Transport:** OSC over UDP to VRChat.

## Status

🚧 Early development. Interfaces and parameters are subject to change. The full
pipeline (capture → detect → face mesh → mapping → OSC) is wired end-to-end,
with **two selectable tracking backends** (issue #17): the default
**MediaPipe** stack (YuNet → FaceMesh V2 → Blendshape V2, ported from
[AvataCam](https://github.com/m96-chan/AvataCam)) and the original **FAN**
stack (S3FD → 2DFAN4). Model weights auto-download on first run. Hand/finger
tracking is the main remaining gap (see Roadmap).

## Architecture

Each stage sits behind a trait so platform- and model-specific pieces stay
swappable:

| Stage | Module | Notes |
|-------|--------|-------|
| Capture | `capture::CameraSource` | `capture::native::NativeCamera` — AVFoundation on macOS, V4L2 on Linux (`Windows` planned), via `nokhwa`; synthetic `FakeCamera` for tests/headless |
| Tracking (default) | `tracking::arkit::ArkitFaceTracker` | `mediapipe::MediapipeTracker` — **YuNet** detector → rotated eyes-aligned ROI → **FaceMesh V2** (478 3-D landmarks, on **burn-wgpu** GPU with candle-onnx CPU fallback) → **Blendshape V2** (52 ARKit coefficients) + per-axis head rotation |
| Tracking (`fan`) | `tracking::FaceTracker` | `fan::FanTracker` — the face-alignment **2DFAN4** net ported to pure-Rust **candle**, with `sfd::SfdDetector` (**S3FD**) auto-cropping first |
| Mapping (default) | `mapping::arkit::ArkitMapper` | 52 ARKit coefficients + head pose → 10 avatar params, with per-channel neutral-baseline calibration (per-eye blink self-calibration so open→0, blink→1) and One-Euro smoothing |
| Mapping (`vrcft`) | `mapping::unified::UnifiedMapper` | 52 ARKit coefficients → Unified Expressions shapes (VRCFT LiveLink correlation) → the VRCFT `v2/` float parameter set + `*TrackingActive` bools, same calibration/smoothing design; profile selected via `[mapping]` / `--mapping` (issue #18) |
| Mapping (`fan`) | `mapping::Mapper` | iBUG-68 landmarks → the same 10 params via geometric ratios, clamped + smoothed |
| OSC | `osc::OscSink` | `UdpOscSender` to VRChat, or `MonitorSink` dry-run |
| Loop | `pipeline::Pipeline` | `capture → track → map → OSC`, driven by the TUI or headless monitor |
| Models | `models::ensure_present` | auto-downloads `models/*.safetensors` / `models/*.onnx` from a GitHub Release on first run |

### Why two backends?

The FAN pipeline derives expressions from 68 2-D landmark *geometry* (eye
aspect ratio etc.), which fundamentally can't do some things a dedicated
expression network can: FAN's landmarks never fully collapse on closed eyes
(so blink can't saturate to 1.0), and the eye aspect ratio shrinks when you
pitch your head down (so looking down falsely reads as a blink). The
MediaPipe Blendshape V2 model predicts `eyeBlinkLeft/Right`, `jawOpen`, etc.
directly from a rotation-normalized face crop, which is robust to head pose —
this is the same stack AvataCam uses, and it is the default here. FAN stays
selectable (`--backend fan` or `backend = "fan"` under `[tracking]` in the
config) per the "models are pluggable" principle.

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

The default MediaPipe backend loads `models/face_detection.onnx` (YuNet),
`models/face_landmarks.onnx` (FaceMesh V2), and `models/face_blendshapes.onnx`
(Blendshape V2); the `fan` backend loads `models/2dfan4.safetensors` and
`models/s3fd.safetensors`. **All auto-download on first run** (from this
repo's [`models-v1` release](https://github.com/m96-chan/VRChatCameraOSC/releases/tag/models-v1))
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
(stop after N frames), `--backend mediapipe|fan` (overrides the config's
`[tracking] backend`), `--mapping custom10|vrcft` (overrides `[mapping]
profile`; see Avatar setup below), `--weights PATH`, `--detector PATH` (FAN
backend only; a custom path is never auto-downloaded), `--detect-interval N`
(see Performance below), `--calibrate-frames N` (see Calibration below; `0`
skips calibration).

VRChat must have **OSC enabled** (Action Menu → Options → OSC → Enabled) for the
end-to-end avatar path.

### Neutral-pose calibration

Both backends capture a short window at startup (`--calibrate-frames N`,
default 10 frames) — **hold a relaxed, forward-facing expression** while it
says `calibrating neutral pose ...`. In the TUI, press **`c`** at any time to
redo it (camera angle or lighting changed, or you weren't ready the first
time).

- **MediaPipe backend:** the raw Blendshape V2 coefficients have a small
  non-zero resting baseline per channel (e.g. `jawOpen` ≈ 0.2 with the mouth
  closed, and each eye's `eyeBlink` baseline differs). Calibration subtracts
  the per-channel resting baseline so a resting face reads `0` everywhere, and
  blink is self-calibrated **per eye** — open reads `0`, a blink a fixed gain
  above your own baseline saturates to `1`. The head's "facing front" pose is
  baselined the same way. If startup calibration is skipped, the same
  baselines are accumulated automatically over the first frames of tracking.
- **FAN backend:** re-derives the geometric neutral ratios (issue #15) —
  without this some parameters stay permanently clamped at one end.

### Performance

The default MediaPipe backend runs at **camera rate — measured 30 FPS**
(640×480@30 webcam, `--release`, default features). Three things make that
possible (issue #17):

- **FaceMesh on burn-wgpu** (`mesh-gpu` feature, default ON): GPU via
  Vulkan/Metal/DX12 with only a graphics driver — no CUDA toolkit. The
  candle-onnx CPU interpreter measured ~105 ms/frame for this
  depthwise-conv-heavy graph (and is kernel-launch-bound on CUDA), nowhere
  near realtime; the burn executor uploads the weights to the GPU once at
  load. No usable GPU → automatic fallback to the CPU stage (a few FPS).
- **Async safety-net redetect**: YuNet (~216 ms on CPU) only reseeds the ROI —
  it runs synchronously on the first frame and after losing the face, and
  otherwise about once a second **on its own thread**, so the frame loop
  never blocks on it. Between detections the ROI comes from the previous
  frame's landmarks.
- **Cached ONNX initializers** (fork addition): graph weights are extracted
  once at load instead of being re-parsed from the proto every evaluation.

The blendshape stage (~2 ms) stays on CPU. The `cuda` feature does **not**
apply to this backend (see above — burn-wgpu is the GPU path here); `cuda`
still accelerates the FAN backend below.

The FAN backend: `candle` builds with no acceleration backend by default
(CPU fallback only), which — combined with running the S3FD detector on every
frame — measured **~0.5 FPS** on a 16-core desktop CPU (issue #13). Two
independent optimizations address this:

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
app's output.

### Option 1 — VRCFT-ready avatars: `--mapping vrcft`, no setup (issue #18)

If your avatar already supports **VRCFaceTracking / Unified Expressions**
(most commercial face-tracking-ready avatars do), run with `--mapping vrcft`
(or set `[mapping] profile = "vrcft"` in the config): the tracker emits the
same `/avatar/parameters/<prefix>v2/<Name>` float parameters VRCFT sends —
under both the bare and `FT/` prefixes by default (`[mapping] vrcft_prefixes`)
— plus the `EyeTrackingActive` / `ExpressionTrackingActive` /
`LipTrackingActive` bools that FT avatars gate their animator layers on. The
avatar needs **no wizard and no Unity work**. Formulas are ported from VRCFT's
own parameter definitions and its official LiveLink (ARKit) module; see
[`src/mapping/unified.rs`](src/mapping/unified.rs) for sources and details.

Current limits (phases 2–3 of issue #18): only **float** parameters are sent —
avatars that declare exclusively *binary* (`v2/JawOpen1/2/4...`) FT parameters
won't move yet — and there is no avatar-config/OSCQuery discovery, so the full
float set is sent regardless of what the avatar declares. Requires the
`mediapipe` backend (default).

### Option 2 — custom 10-parameter setup (the `unity/` wizard) Otherwise (default `custom10` profile), add these as **Float** VRC
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

**Doing this by hand is tedious and error-prone — [`unity/`](unity/) is an
editor wizard (VRChat SDK3, Humanoid avatars) that automates it**: point it at
your avatar and the blend shapes it already has, and it generates the
Expression Parameters and FX Animator Controller layers above for you. See
[`unity/README.md`](unity/README.md).

## Configuration

OSC host/port, camera device, and tracking settings will be configurable from the TUI and/or a config file. (TBD — see Roadmap.)

## Roadmap

- [x] Camera capture pipeline
- [x] Face mesh landmark extraction (FAN / candle, PyTorch-parity verified)
- [x] Face detector (S3FD / candle) — auto-crop the face before FAN
- [x] MediaPipe backend (YuNet → FaceMesh V2 → Blendshape V2, ARKit-52) — new default, AvataCam-parity expression quality
- [ ] Hand / finger landmark extraction
- [x] Landmark → VRChat OSC parameter mapping
- [x] ARKit blendshapes → OSC mapping (per-eye blink self-calibration, One-Euro smoothing)
- [x] VRCFaceTracking-compatible output — Unified Expressions `v2/` float params (issue #18 phase 1)
- [ ] VRCFT binary parameter encoding + avatar-config/OSCQuery param gating (issue #18 phases 2–3)
- [x] OSC sender (UDP) + dry-run monitor
- [x] TUI: live values, status, and configuration
- [x] Config file support
- [x] Unity SDK: avatar setup wizard (Humanoid) — see [`unity/`](unity/)

## License

TBD.
