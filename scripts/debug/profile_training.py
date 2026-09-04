#!/usr/bin/env python3
"""
Quick training profiler + config smoke test (single GPU, no dataset, no DeepSpeed).

Answers two questions in ~1 minute, so you never burn days on a 94h run:
  1. Does the chosen recipe run? (fusion layers, geometry_fusion_scale, STOP loss)
  2. Where does the time/memory go, and what's the full-run ETA?

It builds a representative VLN micro-batch from local frames (9 images, ~2.8k
tokens like real training), runs forward+backward, and times it — separating
the (frozen) VGGT geometry encode from the rest. ETA is extrapolated for the
real 8-GPU / grad-accum setup.

Usage (spatialstack-qwen35 env, repo root):

  GEOMETRY_FUSION_LAYERS="3 7 11" STOP_LOSS_WEIGHT=3.0 \
  python scripts/debug/profile_training.py \
    --model_path /path/to/Qwen3.5-4B-or-checkpoint \
    --geometry_encoder_path model-checkpoint/VGGT-1B \
    --frames_dir debug_vln_output_100/frames \
    --grad_accum 8 --total_steps 37963

Notes:
  * Single-GPU fwd+bwd time is a LOWER BOUND on real per-step time (it excludes
    ZeRO-2 all-reduce/gather), but the VGGT fraction and memory are accurate.
  * fusion layers / scale / stop-weight are read from env exactly like training.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluation as ev  # noqa: E402


def fmt_mem():
    if not torch.cuda.is_available():
        return "n/a"
    return f"{torch.cuda.max_memory_allocated()/1024**3:.1f} GB alloc / {torch.cuda.max_memory_reserved()/1024**3:.1f} GB reserved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--geometry_encoder_path", default=None)
    ap.add_argument("--frames_dir", default="debug_vln_output_100/frames")
    ap.add_argument("--instruction", default="Walk forward, then stop at the doorway.")
    ap.add_argument("--n_warmup", type=int, default=2)
    ap.add_argument("--n_iter", type=int, default=6)
    ap.add_argument("--grad_accum", type=int, default=8, help="micro-steps per optimizer step (TOTAL_BATCH/WORLD_SIZE)")
    ap.add_argument("--total_steps", type=int, default=37963, help="optimizer steps in the real run (for ETA)")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoProcessor
    from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

    # --- config (fusion layers/scale come from env, same as training) ---
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    fl = os.environ.get("GEOMETRY_FUSION_LAYERS")
    if fl:
        cfg.geometry_fusion_layers = [int(x) for x in fl.replace(",", " ").split()]
    geo_path = args.geometry_encoder_path or getattr(cfg, "geometry_encoder_path", None)
    if geo_path:
        cfg.geometry_encoder_path = geo_path
    cfg.use_cache = False
    print(f"[cfg] fusion_layers={getattr(cfg,'geometry_fusion_layers',None)} "
          f"encoder_layers={getattr(cfg,'geometry_encoder_layers',None)} "
          f"fusion_scale(env)={os.environ.get('GEOMETRY_FUSION_SCALE','1.0')} "
          f"stop_loss_weight(env)={os.environ.get('STOP_LOSS_WEIGHT','1.0')}")

    ev.require_qwen3_5_support()
    ev.patch_qwen3_5_flash_attention()
    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        args.model_path, config=cfg, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": "cuda:0"},
        geometry_encoder_path=geo_path, trust_remote_code=True,
    )
    model.model.geometry_encoder.set_eval_streaming(False)  # TRAINING geometry path
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()

    # trainable like training: LLM + lm_head + fusion + merger; VGGT frozen
    n_train = 0
    for name, p in model.named_parameters():
        train = (("language_model" in name or "lm_head" in name
                  or "language_feature_fusion" in name or "feature_fusion" in name
                  or "merger" in name) and "geometry_encoder" not in name)
        p.requires_grad_(train)
        if train:
            n_train += p.numel()
    print(f"[model] trainable params: {n_train/1e9:.2f} B")

    # STOP loss weight (same wiring as train_qwen.py)
    sw = float(os.environ.get("STOP_LOSS_WEIGHT", "1.0"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if sw != 1.0:
        model.stop_loss_weight = sw
        model.stop_token_ids = list(set(tokenizer("STOP", add_special_tokens=False).input_ids))
        print(f"[loss] STOP up-weight {sw} on ids {model.stop_token_ids}")

    # --- representative VLN micro-batch from local frames ---
    fps = sorted(Path(args.frames_dir).glob("frame_*_raw.png"))
    if not fps:
        raise SystemExit(f"no frames in {args.frames_dir}")
    frames = [Image.open(p).convert("RGB") for p in fps]
    proc = AutoProcessor.from_pretrained(args.model_path, max_pixels=ev.MAX_PIXELS,
                                         min_pixels=ev.MIN_PIXELS, trust_remote_code=True)
    msgs = ev.build_training_style_vln_messages(frames, args.instruction)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    mi = proc(text=text, images=frames, videos=None, padding=True, return_tensors="pt")
    geo = ev.build_qwen3_5_geometry_inputs(frames, mi["image_grid_thw"])
    dev = model.device
    # append a STOP action label so loss/backward exercises the STOP path
    act_ids = tokenizer("STOP", add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([mi["input_ids"], act_ids], dim=1).to(dev)
    labels = torch.full_like(input_ids, -100)
    labels[:, -act_ids.shape[1]:] = act_ids.to(dev)
    seq_len = input_ids.shape[1]
    batch = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        pixel_values=mi["pixel_values"].to(dev, torch.bfloat16),
        image_grid_thw=mi["image_grid_thw"].to(dev),
        geometry_encoder_inputs=[torch.stack(geo).to(dev)],
    )
    print(f"[data] {len(frames)} frames, seq_len={seq_len} tokens")

    # time the geometry encode separately
    enc = model.model.geometry_encoder
    orig = enc.encode_layers_with_mode
    geo_ms = {"v": 0.0}

    def timed(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig(*a, **k)
        torch.cuda.synchronize(); geo_ms["v"] = (time.perf_counter() - t0) * 1000
        return out
    enc.encode_layers_with_mode = timed

    # --- profile fwd+bwd ---
    step_ms, geo_per_step = [], []
    losses = []
    for it in range(args.n_warmup + args.n_iter):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(**batch)
        out.loss.backward()
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
        model.zero_grad(set_to_none=True)
        if it >= args.n_warmup:
            step_ms.append(dt); geo_per_step.append(geo_ms["v"]); losses.append(float(out.loss))
        tag = "warmup" if it < args.n_warmup else "measure"
        print(f"  [{tag}] iter {it}: {dt:7.0f} ms (vggt {geo_ms['v']:6.0f} ms)  loss={float(out.loss):.3f}")

    import statistics as st
    micro = st.median(step_ms)
    geo = st.median(geo_per_step)
    opt_step = micro * args.grad_accum
    eta_h = opt_step * args.total_steps / 1000 / 3600
    print("\n================ PROFILE SUMMARY ================")
    print(f"fwd+bwd micro-step : {micro:.0f} ms   (VGGT {geo:.0f} ms = {100*geo/micro:.0f}%)")
    print(f"peak memory        : {fmt_mem()}")
    print(f"loss (finite check): {losses}")
    print(f"optimizer step     : {opt_step/1000:.2f} s  (= micro x grad_accum {args.grad_accum})")
    print(f"full-run ETA       : ~{eta_h:.1f} h  ({args.total_steps} steps)  [LOWER BOUND: excludes ZeRO comm]")
    print("NOTE: VGGT is frozen -> its share is pure recompute; cache it offline to remove that fraction.")


if __name__ == "__main__":
    main()