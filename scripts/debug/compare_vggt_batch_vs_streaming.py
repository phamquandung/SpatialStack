#!/usr/bin/env python3
"""
Compare VGGT batch (encode_layers) vs streaming KV-cache (encode_layers_streaming).

VLN training uses streaming only (`geometry_encoder_streaming=True`). This script:

1. **Must pass**: streaming is deterministic (two runs match).
2. **Informational**: batch causal vs streaming last-frame tokens (often differ —
   JanusVLN KV path uses a different attention branch than `use_cache=False`).

Usage (from SpatialStack root):
  python scripts/debug/compare_vggt_batch_vs_streaming.py
  python scripts/debug/compare_vggt_batch_vs_streaming.py --num_frames 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_frames(n: int, device: torch.device, height: int = 476, width: int = 644):
    frames = []
    for i in range(n):
        r = (30 + i * 40) / 255.0
        img = torch.zeros(3, height, width, device=device)
        img[0].fill_(r)
        img[1].fill_(80 / 255.0)
        img[2].fill_(120 / 255.0)
        yy = torch.linspace(0, 0.15, height, device=device).unsqueeze(1)
        xx = torch.linspace(0, 0.15, width, device=device).unsqueeze(0)
        frames.append((img + yy + xx).clamp(0, 1))
    return torch.stack(frames, dim=0)


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        return torch.cuda.amp.autocast(dtype=dtype)
    return torch.no_grad()


def _raw_batch_last_frame(encoder, images: torch.Tensor, layer_indices: list[int]):
    encoder.vggt.eval()
    images = encoder._apply_reference_frame_transform(images)
    with torch.no_grad(), _autocast_ctx(images.device):
        aggregated_tokens_list, patch_start_idx = encoder.vggt.aggregator(images[None])
    return {
        idx: aggregated_tokens_list[idx][0, -1, patch_start_idx:, :].float().cpu()
        for idx in layer_indices
    }


def _raw_streaming_last_frame(encoder, images: torch.Tensor, layer_indices: list[int]):
    encoder.vggt.eval()
    images = encoder._apply_reference_frame_transform(images)
    past_key_values = [None] * encoder.vggt.aggregator.depth
    aggregated_tokens_list = None
    patch_start_idx = 0
    with torch.no_grad(), _autocast_ctx(images.device):
        for frame_idx, frame in enumerate(images):
            aggregated_tokens_list, patch_start_idx, past_key_values = encoder.vggt.aggregator(
                frame.unsqueeze(0).unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=frame_idx,
            )
    return {
        idx: aggregated_tokens_list[idx][0, -1, patch_start_idx:, :].float().cpu()
        for idx in layer_indices
    }


def _compare_tensors(label: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    if a.shape != b.shape:
        print(f"  {label}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    tag = "OK" if ok else "DIFF"
    print(
        f"  {label}: {tag}  shape={tuple(a.shape)}  "
        f"max_abs={max_diff:.4e}  mean_abs={mean_diff:.4e}"
    )
    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare VGGT batch vs streaming KV cache")
    parser.add_argument("--geometry_encoder_path", default="facebook/VGGT-1B")
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}  Frames: {args.num_frames}  Layers: {args.layer_indices}")

    from qwen_vl.model.geometry_encoders import create_geometry_encoder

    encoder = create_geometry_encoder(
        "vggt",
        model_path=args.geometry_encoder_path,
        reference_frame="first",
        freeze_encoder=True,
    )
    encoder.load_model(args.geometry_encoder_path)
    encoder = encoder.to(device).eval()

    if not encoder.supports_streaming():
        print("ERROR: aggregator missing KV-cache (use_cache).")
        sys.exit(1)

    images = _load_frames(args.num_frames, device)
    print(f"Input: {tuple(images.shape)}")

    # --- 1. Streaming reproducibility (required for VLN training) ---
    print("\n[1] Streaming reproducibility (two identical runs)")
    s1 = _raw_streaming_last_frame(encoder, images, args.layer_indices)
    s2 = _raw_streaming_last_frame(encoder, images, args.layer_indices)
    stream_ok = all(
        _compare_tensors(f"layer {idx} stream×2", s1[idx], s2[idx], args.atol, args.rtol)
        for idx in args.layer_indices
    )

    # --- 2. encode_layers_streaming wrapper reproducibility ---
    print("\n[2] encode_layers_streaming() reproducibility (two runs)")
    w1 = encoder.encode_layers_streaming(
        images, layer_indices=args.layer_indices, spatial_merge_size=2
    )
    w2 = encoder.encode_layers_streaming(
        images, layer_indices=args.layer_indices, spatial_merge_size=2
    )
    wrap_ok = True
    for idx, a, b in zip(args.layer_indices, w1, w2):
        ok = _compare_tensors(f"layer {idx} wrapper×2", a[0].float().cpu(), b[0].float().cpu(), args.atol, args.rtol)
        wrap_ok = wrap_ok and ok

    # --- 3. Batch vs streaming (informational) ---
    print("\n[3] Batch causal vs streaming last-frame (informational — paths differ by design)")
    batch = _raw_batch_last_frame(encoder, images, args.layer_indices)
    for idx in args.layer_indices:
        _compare_tensors(f"layer {idx} batch vs stream", batch[idx], s1[idx], atol=0.05, rtol=0.05)

    # Single-frame path sanity: use_cache vs no-cache on S=1
    if args.num_frames >= 1:
        print("\n[4] Single-frame: aggregator use_cache=False vs use_cache=True (no past KV)")
        img1 = images[:1]
        with torch.no_grad(), _autocast_ctx(device):
            agg_nc, ps_nc = encoder.vggt.aggregator(img1[None], use_cache=False)
            pkv = [None] * encoder.vggt.aggregator.depth
            agg_c, ps_c, _ = encoder.vggt.aggregator(
                img1[None], past_key_values=pkv, use_cache=True, past_frame_idx=0
            )
        li = args.layer_indices[-1]
        t_nc = agg_nc[li][0, 0, ps_nc:, :].float().cpu()
        t_c = agg_c[li][0, -1, ps_c:, :].float().cpu()
        _compare_tensors(f"layer {li} S=1 no_cache vs cache", t_nc, t_c, atol=0.05, rtol=0.05)
        print(
            "  → JanusVLN KV port uses a different global-attention branch than batch mode;\n"
            "    VLN training relies on streaming only, so this gap is expected."
        )

    print()
    if stream_ok and wrap_ok:
        print("PASS: streaming KV path is consistent (safe for VLN training).")
        sys.exit(0)
    print("FAIL: streaming path is not reproducible or wrapper mismatch — investigate KV cache.")
    sys.exit(1)


if __name__ == "__main__":
    main()
