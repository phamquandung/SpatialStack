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


def _geo_merged_positions(
    height: int,
    width: int,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
) -> int:
    """Match VGGTEncoder.encode_layers[_streaming] merge + tiling math."""
    h_patch = height // patch_size
    w_patch = width // patch_size
    trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
    trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
    return (trimmed_h * trimmed_w) // (spatial_merge_size * spatial_merge_size)


def _image_paths(sample: dict) -> list:
    if "image" in sample:
        paths = sample["image"]
    elif "images" in sample:
        paths = sample["images"]
    else:
        return []
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def _image_count(sample: dict) -> int:
    return len(_image_paths(sample))


def _stack_grid_thw(grid_thw) -> torch.Tensor:
    if isinstance(grid_thw, torch.Tensor):
        return grid_thw
    if isinstance(grid_thw, list):
        return torch.stack(grid_thw, dim=0)
    raise TypeError(f"image_grid_thw must be a tensor or list of tensors, got {type(grid_thw)}")


def _vision_tokens_per_image(grid_thw, merge_size: int) -> list[int]:
    thw = _stack_grid_thw(grid_thw)
    merge = merge_size * merge_size
    return (thw.prod(dim=-1) // merge).tolist()


def inspect_raw_sample(sample: dict) -> dict:
    print("\n=== Raw dataset entry (list_data_dict[sample_idx]) ===")
    print(f"id: {sample.get('id')}")
    print(f"num images: {_image_count(sample)}")
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
    tokens_per_image = _vision_tokens_per_image(item["image_grid_thw"], merge_size)
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
    image_paths = _image_paths(raw)
    for i, rel in enumerate(image_paths):
        src = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
        if os.path.isfile(src):
            Image.open(src).convert("RGB").save(out_dir / f"frame_{i:02d}_raw.png")
    for i, g in enumerate(item.get("geometry_encoder_inputs", [])):
        _save_tensor_image(g, out_dir / f"frame_{i:02d}_vggt_644.png")
    print(f"saved frame previews under: {out_dir}")
    return item


def _check_prepare_image_inputs_version() -> None:
    """Warn if the server copy is missing grid-aligned VGGT geometry resize."""
    import inspect
    from qwen_vl.data import utils as data_utils

    src = inspect.getsource(data_utils.prepare_image_inputs)
    if "Resize geometry to the Qwen patch grid" not in src:
        print(
            "WARNING: outdated src/qwen_vl/data/utils.py on this machine — "
            "geometry is still full 644px and training will fail at fusion. "
            "Sync the latest SpatialStack repo, then re-run."
        )
    else:
        print("prepare_image_inputs: grid-aligned VGGT geometry (OK)")


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
        h, w = geo.shape[-2], geo.shape[-1]
        merge_size = 2
        if collator.data_args is not None:
            merge_size = getattr(collator.data_args.image_processor, "merge_size", 2)
        n_geo_merged = _geo_merged_positions(h, w, spatial_merge_size=merge_size)
        print(f"VGGT merged positions (per frame): {n_geo_merged}")

    if "image_grid_thw" in batch:
        thw = batch["image_grid_thw"]
        merge_size = 2
        if collator.data_args is not None:
            merge_size = getattr(collator.data_args.image_processor, "merge_size", 2)
        n_vis = sum(_vision_tokens_per_image(thw, merge_size))
        print(f"approx total Qwen vision tokens: {n_vis}")
        if "geometry_encoder_inputs" in batch:
            geo = batch["geometry_encoder_inputs"][0]
            h, w = geo.shape[-2], geo.shape[-1]
            n_frames = int(geo.shape[0])
            n_geo_merged = _geo_merged_positions(h, w, spatial_merge_size=merge_size)
            per_frame_vis = n_vis // n_frames if n_frames else n_vis
            if per_frame_vis == n_geo_merged:
                print(
                    f"tiling OK: {n_frames} frame(s), "
                    f"{per_frame_vis} vision == {n_geo_merged} geo merged per frame"
                )
            elif n_vis % n_geo_merged == 0:
                factor = n_vis // n_geo_merged
                print(
                    f"tiling factor (vision/geo): {factor}  "
                    f"(expect = num frames {n_frames})"
                )
                if factor != n_frames:
                    print(
                        f"WARNING: tiling factor {factor} != num frames {n_frames}"
                    )
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
    parser.add_argument("--max_samples", type=int, default=128, help="Limit JSON rows loaded (faster debug)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed before dataset shuffle")
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
    _check_prepare_image_inputs_version()

    if args.data_root:
        os.environ["VLN_DATA_ROOT"] = args.data_root
    if args.annotation:
        os.environ["VLN_ANNOTATION"] = args.annotation

    out_dir = Path(args.out_dir)

    model_args = ModelArguments(
        model_name_or_path=args.model_path,
        use_geometry_encoder=True,
        geometry_encoder_streaming=True,
    )
    data_args = DataArguments(
        dataset_use=args.dataset,
        max_samples=args.max_samples,
        shuffle=False,
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

    import random

    random.seed(args.seed)

    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if args.sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx={args.sample_idx} out of range (loaded {len(dataset)} samples; "
            f"increase --max_samples or lower --sample_idx)"
        )
    inspect_raw_sample(dataset.list_data_dict[args.sample_idx])
    item = inspect_dataset_sample(dataset, tokenizer, args.sample_idx, out_dir / "frames")

    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    batch = inspect_batch(collator, [item], tokenizer, out_dir)

    if args.run_forward:
        run_forward(batch, args.model_path, args.geometry_encoder_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
