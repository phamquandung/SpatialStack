#!/usr/bin/env python3
"""
Diagnose whether the SpatialStack geometry fusion is actually USED and ALIGNED
when applied to the JanusVLN navigation task.

No Habitat needed. Runs a handful of single forwards on real saved VLN frames
and intervenes on the geometry stream to answer three questions:

  1. Is geometry used at all?        -> compare REAL vs ZERO geometry.
  2. Is geometry spatially aligned?  -> compare REAL vs SPATIAL-SHUFFLE geometry.
  3. Is it just a magnitude effect?  -> compare REAL vs NOISE (norm-matched).

It also measures the per-layer fusion contribution  ||geo_delta|| / ||vision||
(the implicit "lam") at fusion layers [0,1,2], and renders the geometry heatmap
overlaid on the current frame.

Interpretation:
  * REAL ~ ZERO  (action dist barely moves)  -> geometry is inert / ignored.
  * REAL ~ SHUFFLE (but != ZERO)             -> geometry used as a global bias,
                                                NOT spatially aligned to the image.
  * REAL differs from BOTH ZERO and SHUFFLE  -> geometry is used AND spatially
                                                aligned (wiring is correct;
                                                regression is elsewhere e.g.
                                                exposure bias).

Usage (from repo root, in the spatialstack-qwen35 env):

  python scripts/debug/diagnose_geometry_fusion.py \
      --model_path model-checkpoint/spatialstack_janus_vln_train \
      --geometry_encoder_path model-checkpoint/VGGT-1B \
      --frames_dir debug_vln_output_100/frames \
      --out_dir debug_vln_output_100/fusion_diag
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluation as ev  # noqa: E402  (SpatialStackVLN_Inference + input builders)

ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
INTERVENTIONS = ["real", "zero", "shuffle", "noise"]

# Module-level switches the hooks read.
_STATE = {"mode": "real", "rng": None, "fusion_log": []}


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #
def install_hooks(model):
    """Wrap (a) the geometry encoder to intervene on its features, and
    (b) the fusion module to record ||geo_delta|| / ||vision|| per layer."""
    enc = model.model.geometry_encoder
    fusion = model.model.language_feature_fusion

    orig_encode = enc.encode_layers_with_mode

    def transform(feat: torch.Tensor) -> torch.Tensor:
        mode = _STATE["mode"]
        if mode == "real":
            return feat
        if mode == "zero":
            return torch.zeros_like(feat)
        if mode == "shuffle":
            # permute the patch/token dimension -> break image<->geometry
            # spatial correspondence while keeping the value distribution.
            perm = torch.randperm(feat.shape[1], device=feat.device, generator=None)
            return feat[:, perm, :]
        if mode == "noise":
            # norm-matched Gaussian: same mean/std, no structure at all.
            return torch.randn_like(feat) * feat.float().std() + feat.float().mean()
        return feat

    def wrapped_encode(*args, **kwargs):
        out = orig_encode(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return [transform(t) for t in out]
        return transform(out)

    enc.encode_layers_with_mode = wrapped_encode

    orig_fusion_forward = fusion.forward

    def wrapped_fusion_forward(features_2d, features_3d, layer_num, *a, **kw):
        vin = features_2d.float().norm().item()
        out = orig_fusion_forward(features_2d, features_3d, layer_num, *a, **kw)
        delta = (out - features_2d).float().norm().item()
        _STATE["fusion_log"].append(
            {"layer": int(layer_num), "vis_norm": vin, "delta_norm": delta,
             "ratio": delta / (vin + 1e-8)}
        )
        return out

    fusion.forward = wrapped_fusion_forward
    return enc, orig_encode, fusion, orig_fusion_forward


# --------------------------------------------------------------------------- #
# Input construction (mirrors evaluation.call_model, TRAINING geometry path)
# --------------------------------------------------------------------------- #
def build_inputs(infer, frames, instruction):
    messages = ev.build_training_style_vln_messages(frames, instruction)
    text = infer.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = infer.processor(
        text=text, images=frames, videos=None, padding=True, return_tensors="pt"
    )
    # TRAINING-style geometry: stack ALL frames (encoder loops them internally).
    grid_thw = model_inputs["image_grid_thw"]
    geo = ev.build_qwen3_5_geometry_inputs(frames, grid_thw)
    model_inputs["geometry_encoder_inputs"] = [torch.stack(geo)]
    return model_inputs


@torch.inference_mode()
def score_actions(infer, base_inputs, tokenizer):
    """Teacher-forced score of each candidate action string; softmax -> dist."""
    device = infer.model_device
    dist = {}
    for action in ACTIONS:
        cand_ids = tokenizer(action, add_special_tokens=False, return_tensors="pt").input_ids
        cand_ids = cand_ids.to(device)
        inp = base_inputs.input_ids.to(device)
        full = torch.cat([inp, cand_ids], dim=1)
        attn = torch.ones_like(full)

        kwargs = dict(
            input_ids=full,
            attention_mask=attn,
            pixel_values=base_inputs["pixel_values"].to(device),
            image_grid_thw=base_inputs["image_grid_thw"].to(device),
            geometry_encoder_inputs=[g.to(device) for g in base_inputs["geometry_encoder_inputs"]],
        )
        out = infer.model(**kwargs)
        logits = out.logits[0].float()
        n_cand = cand_ids.shape[1]
        # logits at positions predicting the candidate tokens
        pred = logits[inp.shape[1] - 1 : inp.shape[1] - 1 + n_cand]
        logp = torch.log_softmax(pred, dim=-1)
        tot = logp[torch.arange(n_cand), cand_ids[0]].sum().item()
        dist[action] = tot / n_cand  # length-normalized avg logprob
    # softmax over actions
    vals = np.array([dist[a] for a in ACTIONS])
    probs = np.exp(vals - vals.max())
    probs = probs / probs.sum()
    return {a: float(p) for a, p in zip(ACTIONS, probs)}, dist


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def visualize(results, fusion_ratios, heatmap_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 5))

    # (1) action distribution across interventions
    ax1 = fig.add_subplot(1, 3, 1)
    x = np.arange(len(ACTIONS))
    w = 0.2
    colors = {"real": "#2c7fb8", "zero": "#bdbdbd", "shuffle": "#de2d26", "noise": "#fec44f"}
    for i, mode in enumerate(INTERVENTIONS):
        probs = [results[mode]["probs"][a] for a in ACTIONS]
        ax1.bar(x + (i - 1.5) * w, probs, w, label=mode, color=colors[mode])
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace("_", "\n") for a in ACTIONS], fontsize=8)
    ax1.set_ylabel("P(action)")
    ax1.set_title("Action distribution vs geometry intervention\n"
                  "(real≈zero → ignored; real≈shuffle → not aligned)", fontsize=9)
    ax1.legend(fontsize=8)

    # (2) fusion contribution per layer
    ax2 = fig.add_subplot(1, 3, 2)
    layers = sorted(fusion_ratios.keys())
    ratios = [fusion_ratios[l] for l in layers]
    ax2.bar([str(l) for l in layers], ratios, color="#2c7fb8")
    ax2.axhline(0.2, ls="--", c="k", lw=1, label="JanusVLN lam=0.2 ref")
    ax2.set_xlabel("fusion layer (decoder)")
    ax2.set_ylabel("||geo_delta|| / ||vision||")
    ax2.set_title("Geometry injection magnitude per layer", fontsize=9)
    ax2.legend(fontsize=8)

    # (3) geometry heatmap overlay
    ax3 = fig.add_subplot(1, 3, 3)
    if heatmap_path and Path(heatmap_path).exists():
        ax3.imshow(Image.open(heatmap_path))
    ax3.axis("off")
    ax3.set_title("VGGT geometry (layer 11) over current frame", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"[viz] saved {out_png}")


def save_heatmap(infer, frames, out_dir):
    """Render the layer-11 geometry heatmap over the current frame."""
    try:
        from qwen_vl.debug import geo_viz
    except Exception as e:  # pragma: no cover
        print(f"[heatmap] skipped ({e})")
        return None
    enc = infer.model.model.geometry_encoder
    grid_thw = None
    mi = infer.processor(text="x", images=[frames[-1]], return_tensors="pt")
    grid_thw = mi["image_grid_thw"]
    geo = ev.build_qwen3_5_geometry_inputs([frames[-1]], grid_thw)
    stack = torch.stack(geo).to(infer.model_device)
    _STATE["mode"] = "real"
    # spatial_merge_size=1 -> tokens stay in raster order (h_patch*w_patch)
    feats = enc.encode_layers_with_mode(stack, layer_indices=[11], spatial_merge_size=1, streaming=False)
    t = feats[0][0]  # [h_patch*w_patch, C]
    _, gh, gw = [int(v) for v in grid_thw[0].tolist()]
    try:
        hm = geo_viz.tokens_to_heatmap(t, gh, gw)
        base = frames[-1].resize((gw * 14, gh * 14))
        hm_img = Image.fromarray(hm).resize(base.size)
        blend = Image.blend(base.convert("RGB"), hm_img.convert("RGB"), 0.55)
        p = Path(out_dir) / "geometry_heatmap_overlay.png"
        blend.save(p)
        print(f"[heatmap] saved {p}")
        return str(p)
    except Exception as e:
        print(f"[heatmap] failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--geometry_encoder_path", default=None)
    ap.add_argument("--frames_dir", default="debug_vln_output_100/frames")
    ap.add_argument("--instruction", default="Walk forward, then stop at the doorway.")
    ap.add_argument("--out_dir", default="debug_vln_output_100/fusion_diag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(Path(args.frames_dir).glob("frame_*_raw.png"))
    if not frame_paths:
        raise SystemExit(f"No frame_*_raw.png in {args.frames_dir}")
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    print(f"[data] {len(frames)} frames from {args.frames_dir}")
    print(f"[data] instruction: {args.instruction!r}")

    infer = ev.SpatialStackVLN_Inference(
        args.model_path, device="cuda:0", geometry_encoder_path=args.geometry_encoder_path
    )
    # Use TRAINING geometry path (loop all frames), not eval current-frame-only.
    if hasattr(infer.model.model.geometry_encoder, "set_eval_streaming"):
        infer.model.model.geometry_encoder.set_eval_streaming(False)

    install_hooks(infer.model)
    base_inputs = build_inputs(infer, frames, args.instruction)

    results = {}
    fusion_ratios = {}
    for mode in INTERVENTIONS:
        _STATE["mode"] = mode
        _STATE["fusion_log"] = []
        torch.manual_seed(args.seed)  # same shuffle/noise draw across runs
        probs, raw = score_actions(infer, base_inputs, infer.tokenizer)
        # average fusion ratio per layer (collected during the scoring forwards)
        per_layer = {}
        for rec in _STATE["fusion_log"]:
            per_layer.setdefault(rec["layer"], []).append(rec["ratio"])
        results[mode] = {
            "probs": probs,
            "avg_logp": raw,
            "argmax": max(probs, key=probs.get),
        }
        if mode == "real":
            fusion_ratios = {l: float(np.mean(v)) for l, v in per_layer.items()}
        print(f"[{mode:8s}] argmax={results[mode]['argmax']:12s} "
              + "  ".join(f"{a}={probs[a]:.3f}" for a in ACTIONS))

    print("\n[fusion magnitude] ||geo_delta||/||vision|| per layer:")
    for l in sorted(fusion_ratios):
        print(f"   layer {l}: {fusion_ratios[l]:.4f}")

    heatmap_path = save_heatmap(infer, frames, out_dir)
    visualize(results, fusion_ratios, heatmap_path, out_dir / "fusion_diagnosis.png")

    summary = {
        "instruction": args.instruction,
        "n_frames": len(frames),
        "results": results,
        "fusion_ratios": fusion_ratios,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {out_dir/'summary.json'}")

    # quick automatic verdict
    def l1(m):
        return sum(abs(results["real"]["probs"][a] - results[m]["probs"][a]) for a in ACTIONS)
    d_zero, d_shuf, d_noise = l1("zero"), l1("shuffle"), l1("noise")
    print(f"\n[verdict] L1(real,zero)={d_zero:.3f}  L1(real,shuffle)={d_shuf:.3f}  L1(real,noise)={d_noise:.3f}")
    if d_zero < 0.05:
        print("  -> geometry is essentially IGNORED (real ~= zero).")
    elif d_shuf < 0.05:
        print("  -> geometry used as a GLOBAL BIAS, NOT spatially aligned (real ~= shuffle).")
    else:
        print("  -> geometry is used AND spatially aligned (wiring looks correct).")


if __name__ == "__main__":
    main()