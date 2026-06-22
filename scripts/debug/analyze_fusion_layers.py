#!/usr/bin/env python3
"""
Per-layer "where should we fuse geometry?" profiler — architecture-grounded.

Runs ONE real VLN forward and, for every Qwen3.5 decoder layer, computes:

  (1) geometry<->hidden alignment (linear CKA) between the VGGT geometry features
      and the LLM hidden states at image-token positions  -> where geometry is
      representationally compatible (absorbable).
  (2) decision attribution: ||d(action_logit)/d(hidden)|| at image-token vs text
      positions  -> where the *visual* representation causally drives the action.

  Layers are color-coded linear vs full-attention (Qwen3.5 is hybrid: 3 linear
  then 1 full). A good fusion target is a FULL-attention layer with high vision
  attribution and decent geometry-CKA. The current [3,7,11] choice is marked.

Usage (spatialstack-qwen35 env, repo root):
  python scripts/debug/analyze_fusion_layers.py \
    --model_path model-checkpoint/smoke_test_spatial \
    --geometry_encoder_path model-checkpoint/VGGT-1B \
    --frames_dir debug_vln_output_100/frames \
    --out_dir debug_vln_output_100/fusion_layers
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
import evaluation as ev  # noqa: E402


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between [n,d1] and [n,d2]; handles different feature dims."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    xy = np.linalg.norm(Y.T @ X) ** 2
    xx = np.linalg.norm(X.T @ X)
    yy = np.linalg.norm(Y.T @ Y)
    denom = xx * yy
    return float(xy / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--geometry_encoder_path", default=None)
    ap.add_argument("--frames_dir", default="debug_vln_output_100/frames")
    ap.add_argument("--instruction", default="Walk forward, then stop at the doorway.")
    ap.add_argument("--out_dir", default="debug_vln_output_100/fusion_layers")
    ap.add_argument("--processor_path", default=None,
                    help="dir with a full processor (preprocessor_config.json); defaults to PROCESSOR_PATH env or model_path. "
                         "Intermediate checkpoints lack it — point at base Qwen3.5-4B.")
    args = ap.parse_args()
    import os as _os
    proc_src = args.processor_path or _os.environ.get("PROCESSOR_PATH") or args.model_path
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoProcessor
    from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    geo_path = args.geometry_encoder_path or getattr(cfg, "geometry_encoder_path", None)
    if geo_path:
        cfg.geometry_encoder_path = geo_path
    cfg.use_cache = False
    ev.require_qwen3_5_support(); ev.patch_qwen3_5_flash_attention()
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        args.model_path, config=cfg, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": "cuda:0"},
        geometry_encoder_path=geo_path, trust_remote_code=True,
    ).eval()
    model.model.geometry_encoder.set_eval_streaming(False)  # training geometry path

    dev = model.device
    layers = model.model.language_model.layers
    layer_types = [getattr(l, "layer_type", "?") for l in layers]
    n_layers = len(layers)

    # ---- build a real VLN input ----
    fps = sorted(Path(args.frames_dir).glob("frame_*_raw.png"))
    if not fps:
        raise SystemExit(f"no frames in {args.frames_dir}")
    frames = [Image.open(p).convert("RGB") for p in fps]
    try:
        proc = AutoProcessor.from_pretrained(proc_src, max_pixels=ev.MAX_PIXELS,
                                             min_pixels=ev.MIN_PIXELS, trust_remote_code=True)
    except Exception as e:
        raise SystemExit(
            f"Could not load a processor from '{proc_src}'. This dir lacks preprocessor_config.json "
            f"(intermediate checkpoints don't save it). Pass --processor_path <base Qwen3.5-4B dir> "
            f"or set PROCESSOR_PATH. Original error: {e}")
    msgs = ev.build_training_style_vln_messages(frames, args.instruction)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    mi = proc(text=text, images=frames, videos=None, padding=True, return_tensors="pt")
    geo = ev.build_qwen3_5_geometry_inputs(frames, mi["image_grid_thw"])
    geo_stack = torch.stack(geo).to(dev)
    input_ids = mi["input_ids"].to(dev)
    img_tok = int(getattr(model.config, "image_token_id"))
    vis_mask = (input_ids[0] == img_tok)
    n_vis = int(vis_mask.sum())
    print(f"[data] {len(frames)} frames, seq_len={input_ids.shape[1]}, image tokens={n_vis}")
    print(f"[arch] layer_types (0-7): {layer_types[:8]}  full_attn_interval={getattr(cfg.text_config,'full_attention_interval','?')}")

    # ---- geometry reference at vision-token granularity (for CKA) ----
    geo_rep = None
    try:
        feats = model.model.geometry_encoder.encode_layers_with_mode(
            geo_stack, layer_indices=getattr(cfg, "geometry_encoder_layers", [11, 17, 23]),
            spatial_merge_size=2, streaming=False)
        g = torch.stack([f.float() for f in feats], 0).mean(0)   # [n_image, n_tok, C] avg over geo layers
        n_image, n_tok, C = g.shape
        gm = g.reshape(n_image, n_tok // 4, 4 * C).reshape(-1, 4 * C)  # 2x2 merge -> [n_image*n_merged, 4C]
        if gm.shape[0] == n_vis:
            geo_rep = gm.cpu().numpy()
        else:
            print(f"[cka] geometry rows {gm.shape[0]} != vision tokens {n_vis} -> skipping CKA")
    except Exception as e:
        print(f"[cka] skipped ({e})")

    # ---- hooks to capture per-layer hidden states (with grad) ----
    captured = [None] * n_layers
    def mk(i):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            h.retain_grad(); captured[i] = h
        return hook
    handles = [layers[i].register_forward_hook(mk(i)) for i in range(n_layers)]

    # ---- forward + backward from the action decision ----
    torch.set_grad_enabled(True)
    out = model(input_ids=input_ids, attention_mask=mi["attention_mask"].to(dev),
                pixel_values=mi["pixel_values"].to(dev, torch.bfloat16),
                image_grid_thw=mi["image_grid_thw"].to(dev),
                geometry_encoder_inputs=[geo_stack])
    decision_logit = out.logits[0, -1].float().max()   # the model's chosen next-token logit
    model.zero_grad(set_to_none=True)
    decision_logit.backward()
    for h in handles:
        h.remove()

    # ---- per-layer metrics ----
    vis_attr, txt_attr, cka = [], [], []
    txt_mask = ~vis_mask
    for i in range(n_layers):
        H = captured[i][0]                       # [T, hid]
        g = captured[i].grad[0]                  # [T, hid]
        vis_attr.append(g[vis_mask].float().norm(dim=-1).mean().item())
        txt_attr.append(g[txt_mask].float().norm(dim=-1).mean().item())
        if geo_rep is not None:
            Hv = H[vis_mask].detach().float().cpu().numpy()
            cka.append(linear_cka(geo_rep, Hv))
        else:
            cka.append(float("nan"))

    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    cur_fuse = list(getattr(cfg, "geometry_fusion_layers", []) or [])
    print(f"[arch] full-attention layers: {full_idx}")
    print(f"[cfg]  current geometry_fusion_layers: {cur_fuse}")

    # ---- plot ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(n_layers)
    colors = ["#d95f0e" if t == "full_attention" else "#9ecae1" for t in layer_types]
    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    ax[0].bar(x, vis_attr, color=colors, label="_")
    ax[0].plot(x, txt_attr, "k.--", lw=1, ms=4, label="text-token attribution (baseline)")
    for i in full_idx:
        ax[0].axvline(i, color="#d95f0e", alpha=0.12)
    for i in cur_fuse:
        ax[0].axvline(i, color="green", ls="--", lw=1.5)
    ax[0].set_ylabel("||grad of action logit||  (per-token mean)")
    ax[0].set_title("Decision attribution per layer — orange bars = FULL-attention, blue = linear, "
                    "green dashed = current fuse layers\n(fuse where VISION attribution >> text, at a full-attn layer)", fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].bar(x, cka, color=colors)
    for i in cur_fuse:
        ax[1].axvline(i, color="green", ls="--", lw=1.5)
    ax[1].set_ylabel("geometry<->hidden CKA")
    ax[1].set_xlabel("decoder layer")
    ax[1].set_title("Geometry-language representational alignment per layer (higher = geometry more compatible)", fontsize=10)
    ax[1].set_xticks(x); ax[1].set_xticklabels(x, fontsize=7)

    fig.tight_layout()
    p = out_dir / "fusion_layer_profile.png"
    fig.savefig(p, dpi=130)
    print(f"[plot] wrote {p}")

    # ---- text ranking: best full-attention fusion targets ----
    print("\n=== full-attention layers ranked by vision attribution (x CKA) ===")
    scored = []
    for i in full_idx:
        score = vis_attr[i] * (cka[i] if cka[i] == cka[i] else 1.0)
        scored.append((i, vis_attr[i], txt_attr[i], cka[i], score))
    scored.sort(key=lambda r: -r[4])
    print(f"{'layer':>5} {'visAttr':>9} {'txtAttr':>9} {'CKA':>7} {'score':>9}")
    for i, va, ta, ck, sc in scored:
        print(f"{i:>5} {va:>9.3e} {ta:>9.3e} {ck:>7.3f} {sc:>9.3e}")
    top3 = sorted([r[0] for r in scored[:3]])
    print(f"\nSuggested fusion layers (top-3 full-attn by score): {top3}")
    print(f"Current: {cur_fuse}")


if __name__ == "__main__":
    main()