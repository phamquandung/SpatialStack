#!/usr/bin/env python3
"""
Debug VLN data loading + streaming VGGT shapes without a full training run.

Examples (from SpatialStack repo root):

  # Inspect one dataset sample + save frame images
  python scripts/debug/debug_vln_pipeline.py --sample_idx 0

  # Run one collated batch + optional model forward
  python scripts/debug/debug_vln_pipeline.py --sample_idx 0 --run_forward \
      --model_path Qwen/Qwen3.5-4B

  # Local checkpoints on server:
  python scripts/debug/debug_vln_pipeline.py --sample_idx 0 --run_forward \
      --model_path /mnt/data/vmo-ai-task/dungpq6/model-checkpoint/Qwen3.5-4B \
      --geometry_encoder_path /mnt/data/vmo-ai-task/dungpq6/model-checkpoint/VGGT-1B

Env (same as training):
  export VLN_DATA_ROOT=.
  export VLN_ANNOTATION=data/train/train_r2r_rxr_extra.json
  export MODEL_PATH=/path/to/Qwen3.5-4B
  export GEOMETRY_ENCODER_PATH=/path/to/VGGT-1B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from qwen_vl.data import data_list
from qwen_vl.data.data_qwen import DataCollatorForSupervisedDataset, LazySupervisedDataset
from qwen_vl.train.argument import DataArguments, ModelArguments


def _save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """Save CHW float tensor in [0,1] as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dim() == 3:
        to_pil_image(tensor.clamp(0, 1)).save(path)
    else:
        raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")


def _decode_labels(tokenizer, labels: torch.Tensor) -> str:
    mask = labels != -100
    if mask.sum() == 0:
        return "(no trainable labels)"
    ids = labels[mask].tolist()
    return tokenizer.decode(ids, skip_special_tokens=False)


def inspect_raw_sample(annotation_path: str, sample_idx: int) -> dict:
    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)
    sample = data[sample_idx]
    print("\n=== Raw JSON sample ===")
    print(f"id: {sample.get('id')}")
    print(f"num images: {len(sample.get('images', sample.get('image', [])))}")
    print(f"human (first 200 chars): {sample['conversations'][0]['value'][:200]}...")
    print(f"label: {sample['conversations'][1]['value']}")
    return sample


def inspect_dataset_sample(dataset: LazySupervisedDataset, tokenizer, idx: int, out_dir: Path):
    print(f"\n=== Dataset __getitem__({idx}) ===")
    raw = dataset.list_data_dict[idx]
    item = dataset[idx]
    n_frames = len(item["pixel_values"])
    print(f"num frames: {n_frames}")
    print(f"input_ids shape: {tuple(item['input_ids'].shape)}")
    print(f"labels trainable tokens: {(item['labels'] != -100).sum().item()}")
    print(f"decoded label: {_decode_labels(tokenizer, item['labels'])}")
    print(f"image_grid_thw:\n{item['image_grid_thw']}")

    merge_size = getattr(dataset.data_args.image_processor, "merge_size", 2)
    tokens_per_image = (
        item["image_grid_thw"].prod(dim=-1) // (merge_size * merge_size)
    ).tolist()
    print(f"Qwen tokens per image (grid.prod/merge^2): {tokens_per_image}")
    print(f"total vision tokens: {sum(tokens_per_image)}")

    if "geometry_encoder_inputs" in item:
        geo = item["geometry_encoder_inputs"]
        print(f"geometry frames: {len(geo)}")
        for i, g in enumerate(geo):
            print(f"  geo[{i}] shape: {tuple(g.shape)}")
        stacked = torch.stack(geo)
        print(f"stacked geometry shape: {tuple(stacked.shape)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = raw.get("data_path", ".")
    image_paths = raw.get("images", raw.get("image", []))
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    for i, rel in enumerate(image_paths):
        src = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
        if os.path.isfile(src):
            Image.open(src).convert("RGB").save(out_dir / f"frame_{i:02d}_raw.png")
    for i, g in enumerate(item.get("geometry_encoder_inputs", [])):
        _save_tensor_image(g, out_dir / f"frame_{i:02d}_vggt_644.png")
    print(f"saved frame previews under: {out_dir}")
    return item


def inspect_batch(collator, items, tokenizer, out_dir: Path):
    print("\n=== Collated batch ===")
    batch = collator(items)
    for key in ("input_ids", "labels", "pixel_values", "image_grid_thw", "geometry_encoder_inputs"):
        if key not in batch:
            continue
        val = batch[key]
        if isinstance(val, torch.Tensor):
            print(f"{key}: {tuple(val.shape)} dtype={val.dtype}")
        elif isinstance(val, list):
            print(f"{key}: list len={len(val)}")
            if val and isinstance(val[0], torch.Tensor):
                print(f"  [0]: {tuple(val[0].shape)}")
        else:
            print(f"{key}: {type(val)}")

    if "geometry_encoder_inputs" in batch:
        geo = batch["geometry_encoder_inputs"][0]
        print(f"streaming input [S,C,H,W]: {tuple(geo.shape)}")
        n_geo = geo.shape[0]
        h, w = geo.shape[-2], geo.shape[-1]
        h_patch, w_patch = h // 14, w // 14
        m = 2
        n_geo_merged = (h_patch // m) * (w_patch // m)
        print(f"VGGT patches: {h_patch}x{w_patch} -> merged positions ~{n_geo_merged}")

    if "image_grid_thw" in batch:
        thw = batch["image_grid_thw"]
        n_vis = sum((thw[i].prod() // 4).item() for i in range(thw.shape[0]))
        print(f"approx total Qwen vision tokens: {n_vis}")
        if "geometry_encoder_inputs" in batch:
            geo = batch["geometry_encoder_inputs"][0]
            h, w = geo.shape[-2], geo.shape[-1]
            n_geo_merged = (h // 14 // 2) * (w // 14 // 2)
            if n_vis % n_geo_merged == 0:
                print(f"tiling factor (vision/geo): {n_vis // n_geo_merged}  (expect = num frames)")
            else:
                print(
                    f"WARNING: vision tokens ({n_vis}) not divisible by geo merged ({n_geo_merged}) — "
                    "tiling will fail in training!"
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        summary = {
            "num_input_ids": int(batch["input_ids"].shape[-1]),
            "trainable_labels": int((batch["labels"] != -100).sum()),
        }
        json.dump(summary, f, indent=2)
    return batch


def run_forward(batch, model_path: str, geometry_encoder_path: str):
    print("\n=== One forward pass (no backward) ===")
    from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    for k, v in {
        "use_geometry_encoder": True,
        "geometry_encoder_type": "vggt",
        "geometry_encoder_path": geometry_encoder_path,
        "geometry_encoder_streaming": True,
        "feature_fusion_method": "deepstack_language_add",
        "geometry_fusion_layers": [0, 1, 2],
        "geometry_encoder_layers": [11, 17, 23],
        "reference_frame": "first",
    }.items():
        setattr(config, k, v)

    model = Qwen3_5ForConditionalGenerationWithGeometry.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        geometry_encoder_path=geometry_encoder_path,
        trust_remote_code=True,
    ).cuda().eval()

    fwd_batch = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
        if k in {
            "input_ids", "attention_mask", "position_ids", "pixel_values",
            "image_grid_thw", "geometry_encoder_inputs", "labels",
        }
    }
    if "geometry_encoder_inputs" in fwd_batch:
        fwd_batch["geometry_encoder_inputs"] = [
            g.cuda().to(torch.bfloat16) for g in fwd_batch["geometry_encoder_inputs"]
        ]

    with torch.no_grad():
        out = model(**fwd_batch)
    loss = out.loss.item() if out.loss is not None else float("nan")
    print(f"forward OK, loss={loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Debug VLN training pipeline")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--dataset", default="train_r2r_rxr")
    parser.add_argument("--annotation", default=None, help="Override VLN_ANNOTATION path")
    parser.add_argument("--data_root", default=None, help="Override VLN_DATA_ROOT")
    parser.add_argument("--out_dir", default="./debug_vln_output")
    parser.add_argument("--run_forward", action="store_true")
    parser.add_argument("--model_path", default=None, help="Default: MODEL_PATH env or Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--geometry_encoder_path",
        default=None,
        help="Default: GEOMETRY_ENCODER_PATH env or facebook/VGGT-1B",
    )
    args = parser.parse_args()

    args.model_path = args.model_path or os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-4B")
    args.geometry_encoder_path = args.geometry_encoder_path or os.environ.get(
        "GEOMETRY_ENCODER_PATH", "facebook/VGGT-1B"
    )
    print(f"model_path: {args.model_path}")
    print(f"geometry_encoder_path: {args.geometry_encoder_path}")

    if args.data_root:
        os.environ["VLN_DATA_ROOT"] = args.data_root
    if args.annotation:
        os.environ["VLN_ANNOTATION"] = args.annotation

    out_dir = Path(args.out_dir)
    ann_path = os.environ.get("VLN_ANNOTATION", "data/train/train_r2r_rxr_extra.json")
    if not Path(ann_path).is_absolute():
        ann_path = str(REPO_ROOT / ann_path)

    inspect_raw_sample(ann_path, args.sample_idx)

    model_args = ModelArguments(
        model_name_or_path=args.model_path,
        use_geometry_encoder=True,
        geometry_encoder_streaming=True,
    )
    data_args = DataArguments(
        dataset_use=args.dataset,
        max_samples=10,
    )
    data_args.use_geometry_encoder = True
    data_args.geometry_encoder_streaming = True
    data_args.model_type = "qwen3.5"

    processor = transformers.AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    data_args.image_processor = processor.image_processor
    data_args.processor = processor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )

    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    item = inspect_dataset_sample(dataset, tokenizer, args.sample_idx, out_dir / "frames")

    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    batch = inspect_batch(collator, [item], tokenizer, out_dir)

    if args.run_forward:
        run_forward(batch, args.model_path, args.geometry_encoder_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
