#!/usr/bin/env python3
"""Generate golden fixtures for FAN numeric-parity tests (issue #4).

Reference implementation: 1adrianb/face-alignment (2DFAN4).

Produces, under the repo root:
  models/2dfan4.safetensors        full pretrained FAN weights (large; gitignored)
  tests/fixtures/fan_input.f32     deterministic 1x3x256x256 input (raw LE f32)
  tests/fixtures/fan_output.f32    final-stage heatmaps 1x68x64x64 (raw LE f32)
  tests/fixtures/fan_preds.json    decoded landmarks (heatmap space) + scores
  tests/fixtures/transform.json    transform() vectors (image<->crop)
  tests/fixtures/convblock_small.safetensors   tiny random ConvBlock weights (committed)
  tests/fixtures/convblock_small_io.f32        its input+output (committed)

The Rust tests load the same weights + input and assert their candle port
matches within tolerance. The full-network fixtures need the ~90MB pretrained
download and are gitignored; the small ConvBlock fixtures are tiny and committed
so architecture parity runs in CI without any download.

Run:  reference/.venv/bin/python reference/gen_fixtures.py
"""
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from face_alignment.models.fan import FAN, ConvBlock
from face_alignment.utils import get_preds_fromhm, transform

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FIX = ROOT / "tests" / "fixtures"
MODELS.mkdir(exist_ok=True)
FIX.mkdir(parents=True, exist_ok=True)

FAN_URL = "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar"


def write_f32(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    path.write_bytes(arr.tobytes())
    print(f"  wrote {path.relative_to(ROOT)}  shape={arr.shape}")


def clean_state_dict(sd: dict) -> dict:
    """safetensors needs plain contiguous tensors and no shared storage; drop the
    integer `num_batches_tracked` buffers (candle's BatchNorm ignores them)."""
    out = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        out[k] = v.detach().contiguous().float().clone()
    return out


def gen_full_fan() -> None:
    print("[full FAN] loading pretrained 2DFAN4 ...")
    net = FAN(4)
    try:
        from face_alignment.utils import load_file_from_url

        sd = torch.load(load_file_from_url(FAN_URL), map_location="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"  !! could not fetch pretrained weights: {e}")
        print("  !! skipping full-network fixtures (unit tests still cover the port).")
        return
    net.load_state_dict(sd)
    net.eval()

    save_file(clean_state_dict(net.state_dict()), str(MODELS / "2dfan4.safetensors"))
    print(f"  wrote {(MODELS / '2dfan4.safetensors').relative_to(ROOT)}")

    torch.manual_seed(0)
    inp = torch.rand(1, 3, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        outs = net(inp)  # list of 4 stages
    out = outs[-1]

    write_f32(FIX / "fan_input.f32", inp.numpy())
    write_f32(FIX / "fan_output.f32", out.numpy())

    preds, _, scores = get_preds_fromhm(out.numpy())
    preds = preds[0]  # (68, 2) heatmap space
    scores = scores[0]
    fan_preds = [
        {"x": float(preds[i, 0]), "y": float(preds[i, 1]), "score": float(scores[i])}
        for i in range(preds.shape[0])
    ]
    (FIX / "fan_preds.json").write_text(json.dumps(fan_preds, indent=2))
    print(f"  wrote {(FIX / 'fan_preds.json').relative_to(ROOT)}  ({len(fan_preds)} pts)")


def gen_transform_cases() -> None:
    cases = [
        {"point": [30.0, 40.0], "center": [128.0, 128.0], "scale": 1.2, "res": 256.0, "invert": False},
        {"point": [1.0, 1.0], "center": [100.0, 90.0], "scale": 0.8, "res": 256.0, "invert": True},
        {"point": [33.0, 50.0], "center": [120.0, 130.0], "scale": 1.0, "res": 64.0, "invert": True},
    ]
    for c in cases:
        out = transform(
            torch.tensor(c["point"]),
            torch.tensor(c["center"]),
            c["scale"],
            c["res"],
            c["invert"],
        )
        c["expected"] = [int(out[0]), int(out[1])]
    (FIX / "transform.json").write_text(json.dumps(cases, indent=2))
    print(f"  wrote {(FIX / 'transform.json').relative_to(ROOT)}  ({len(cases)} cases)")


def gen_small_convblock() -> None:
    """Tiny random ConvBlock (6->8) — committed, so CI validates the port with no download."""
    print("[small ConvBlock] generating committed fixture ...")
    torch.manual_seed(1234)
    blk = ConvBlock(6, 8)
    # Randomise BN running stats so eval-mode normalisation is non-trivial.
    for m in blk.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.randn_like(m.running_mean)
            m.running_var = torch.rand_like(m.running_var) + 0.5
            torch.nn.init.uniform_(m.weight, 0.5, 1.5)
            torch.nn.init.uniform_(m.bias, -0.5, 0.5)
    blk.eval()

    save_file(clean_state_dict(blk.state_dict()), str(FIX / "convblock_small.safetensors"))
    print(f"  wrote {(FIX / 'convblock_small.safetensors').relative_to(ROOT)}")

    torch.manual_seed(7)
    x = torch.rand(1, 6, 16, 16, dtype=torch.float32)
    with torch.no_grad():
        y = blk(x)  # (1, 8, 16, 16)

    # Pack input and output together with a tiny header so Rust knows the shapes.
    meta = {
        "in_shape": list(x.shape),
        "out_shape": list(y.shape),
    }
    (FIX / "convblock_small.json").write_text(json.dumps(meta, indent=2))
    write_f32(FIX / "convblock_small_input.f32", x.numpy())
    write_f32(FIX / "convblock_small_output.f32", y.numpy())


SFD_URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
ASSETS = ROOT / "reference" / "assets"


def gen_sfd_and_endtoend() -> None:
    """Export S3FD weights and generate detector-box + end-to-end landmark
    fixtures on the bundled face photo. These reference the face image, so they
    live under reference/assets/ (gitignored) and drive the LOCAL parity tests."""
    print("[SFD] generating detector + end-to-end fixtures ...")
    import face_alignment
    from face_alignment.detection.sfd.sfd_detector import SFDDetector
    from face_alignment.detection.sfd.net_s3fd import s3fd
    from skimage import io

    img_path = ASSETS / "aflw-test.jpg"
    if not img_path.exists():
        print(f"  !! {img_path} missing; skipping SFD fixtures.")
        return

    # Export S3FD weights to safetensors (mirrors the reference download).
    try:
        from face_alignment.utils import load_file_from_url

        sd = torch.load(load_file_from_url(SFD_URL), map_location="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"  !! could not fetch S3FD weights: {e}; skipping.")
        return
    net = s3fd()
    net.load_state_dict(sd)
    net.eval()
    save_file(clean_state_dict(net.state_dict()), str(MODELS / "s3fd.safetensors"))
    print(f"  wrote {(MODELS / 's3fd.safetensors').relative_to(ROOT)}")

    # Load the exact RGB pixels and dump them, so Rust reads identical input
    # (no JPEG-decoder mismatch).
    image = io.imread(str(img_path))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image[:, :, :3].astype(np.uint8)
    h, w = image.shape[:2]
    (ASSETS / "aflw-test.json").write_text(json.dumps({"h": int(h), "w": int(w)}))
    (ASSETS / "aflw-test.rgb").write_bytes(np.ascontiguousarray(image).tobytes())
    print(f"  wrote reference/assets/aflw-test.rgb  shape=({h},{w},3)")

    # Reference detector boxes on this image.
    det = SFDDetector(device="cpu")
    boxes = det.detect_from_image(image.copy())
    box_list = [
        {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]), "score": float(b[4])}
        for b in boxes
    ]
    (ASSETS / "sfd_boxes.json").write_text(json.dumps(box_list, indent=2))
    print(f"  wrote reference/assets/sfd_boxes.json  ({len(box_list)} boxes)")

    # End-to-end landmarks via the full reference pipeline (flip_input=False so
    # the Rust path, which doesn't flip-average, matches).
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu"
    )
    preds = fa.get_landmarks_from_image(image.copy())
    if preds:
        lms = preds[0]  # (68, 2) image space
        lm_list = [{"x": float(lms[i, 0]), "y": float(lms[i, 1])} for i in range(lms.shape[0])]
        (ASSETS / "fan_image_landmarks.json").write_text(json.dumps(lm_list, indent=2))
        print(f"  wrote reference/assets/fan_image_landmarks.json  ({len(lm_list)} pts)")
    else:
        print("  !! reference found no face; end-to-end fixture skipped.")


if __name__ == "__main__":
    gen_small_convblock()
    gen_transform_cases()
    gen_full_fan()
    gen_sfd_and_endtoend()
    print("done.")
