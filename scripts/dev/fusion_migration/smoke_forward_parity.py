#!/usr/bin/env python
"""Forward-parity smoke for the fusion migration (Step 0 safety net).

Runs a SINGLE deterministic forward on a fixed image+prompt and fingerprints the
logits. Purpose: after any later migration step, run this with the new flag OFF
and confirm the fingerprint still matches the golden captured here — an instant
wiring-bug detector that does NOT need a full Habitat rollout.

Reuses scripts/inference/infer.py's tested input-building path, so it exercises
the same geometry pipeline as real inference.

Run on the GPU box:

    # capture the golden once (on the untouched baseline)
    python scripts/dev/fusion_migration/smoke_forward_parity.py \
        --model-path model_checkpoint/spatialstack_vln_fix_0.5_full --write-golden

    # later, after a step (flag OFF) — must match:
    python scripts/dev/fusion_migration/smoke_forward_parity.py \
        --model-path model_checkpoint/spatialstack_vln_fix_0.5_full
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))

import infer as I  # noqa: E402  (tested input-building helpers)

DEFAULT_IMAGE = REPO_ROOT / "assets" / "sofas.jpg"
DEFAULT_PROMPT = "Describe the spatial layout of this scene."
GOLDEN_PATH = Path(__file__).with_name("golden_fingerprint.json")
# bf16 forward: compare floats with a loose relative tolerance, token ids exactly.
FLOAT_RTOL = 2e-3


def fingerprint(logits: torch.Tensor) -> dict:
    l = logits.detach().float().cpu()
    last = l[0, -1]
    top5 = torch.topk(last, 5).indices.tolist()
    return {
        "shape": list(logits.shape),
        "sum": l.sum().item(),
        "mean": l.mean().item(),
        "std": l.std().item(),
        "last_argmax": int(last.argmax().item()),
        "last_top5": top5,
    }


def compare(cur: dict, golden: dict) -> bool:
    ok = True
    for k in ("last_argmax", "last_top5", "shape"):
        if cur[k] != golden[k]:
            print(f"  MISMATCH {k}: cur={cur[k]} golden={golden[k]}")
            ok = False
    for k in ("sum", "mean", "std"):
        c, g = cur[k], golden[k]
        denom = max(abs(g), 1e-6)
        if abs(c - g) / denom > FLOAT_RTOL:
            print(f"  MISMATCH {k}: cur={c:.6g} golden={g:.6g} (rtol {FLOAT_RTOL})")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(REPO_ROOT / "model_checkpoint" / "spatialstack_vln_fix_0.5_full"))
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--write-golden", action="store_true", help="save current fingerprint as the golden reference")
    args = ap.parse_args()

    torch.manual_seed(0)

    config = AutoConfig.from_pretrained(args.model_path)
    model_family = I.resolve_model_family(config)
    use_geo = getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False)
    model_class = I.resolve_model_class(model_family, use_geo)
    if model_family == "qwen3_5":
        I.patch_qwen3_5_flash_attention()

    model_kwargs = {
        "pretrained_model_name_or_path": args.model_path,
        "config": config,
        "torch_dtype": torch.bfloat16,
        "device_map": args.device,
    }
    if model_family == "qwen3_5" and use_geo:
        geo_path = getattr(config, "geometry_encoder_path", None)
        if geo_path:
            model_kwargs["geometry_encoder_path"] = geo_path
    model = model_class.from_pretrained(**model_kwargs, attn_implementation="flash_attention_2").eval()

    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")

    from PIL import Image
    visuals = [Image.open(args.image).convert("RGB")]
    messages = I.build_messages(args.prompt, visuals, add_frame_index=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    raw = I.prepare_raw_visual_inputs(messages)
    model_inputs = processor(text=text, images=raw, videos=None, padding=True, return_tensors="pt")
    if use_geo:
        geo = [torch.stack(I.build_qwen3_5_geometry_inputs(raw, model_inputs["image_grid_thw"]))]
        model_inputs["geometry_encoder_inputs"] = [f.to(args.device) for f in geo]
    model_inputs = model_inputs.to(args.device)

    with torch.no_grad():
        out = model(**model_inputs)

    fp = fingerprint(out.logits)
    print(json.dumps(fp, indent=2))

    if args.write_golden:
        GOLDEN_PATH.write_text(json.dumps(fp, indent=2))
        print(f"[golden written] {GOLDEN_PATH}")
        return

    if GOLDEN_PATH.exists():
        golden = json.loads(GOLDEN_PATH.read_text())
        print("PARITY OK" if compare(fp, golden) else "PARITY FAILED")
        sys.exit(0 if compare(fp, golden) else 1)
    print("[no golden yet] run with --write-golden first")


if __name__ == "__main__":
    main()
