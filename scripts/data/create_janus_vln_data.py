"""Build JanusVLN-style VLN training JSON for SpatialStack (no JanusVLN repo required)."""

import argparse
import concurrent.futures
import glob
import gzip
import json
import os
from functools import partial

import numpy as np
from tqdm import tqdm

max_history_images = 8

VLN_PROMPT = (
    "You are a visual language navigation model, and your should go to the locations "
    "to complete the given task. Compare the observation and instruction to infer "
    "your current progress, and then select the correct direction from the candidates "
    "to go to the target location and finish the task.\n"
    " This is your historical observation:{his_img_tags}\n"
    " This is your current observation:<image>\n"
    " Your task is to {instruction}\n"
    " You should take one of the following actions:\n"
    " MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
)


def _rel(path, root):
    path = os.path.normpath(path)
    if os.path.isabs(path):
        return os.path.relpath(path, root)
    return path


def process_episode_scalevln(ep, img_root, act_map, data_root):
    episode_results = []
    episode_id = str(ep["id"])
    instruction = ep["instructions"][0]
    name = ep["video"].split("/")[1]
    img_dir = os.path.join(img_root, name, "rgb")

    try:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    except FileNotFoundError:
        print(f"Warning: Directory not found for scalevln episode {episode_id}: {img_dir}")
        return []

    for i in range(len(img_files)):
        if i <= max_history_images:
            idxs = list(range(i + 1))
        else:
            idxs = np.linspace(0, i, 9, dtype=int).tolist()
        sampled_imgs = [_rel(img_files[j], data_root) for j in idxs if os.path.exists(img_files[j])]
        if not sampled_imgs:
            continue

        his_img_tags = "<image>" * (len(sampled_imgs) - 1)
        action = "STOP" if i == len(img_files) - 1 else act_map[ep["actions"][i + 1]]
        episode_results.append(
            {
                "id": f"{episode_id}/{os.path.basename(img_files[i])}",
                "conversations": [
                    {"from": "human", "value": VLN_PROMPT.format(his_img_tags=his_img_tags, instruction=instruction)},
                    {"from": "gpt", "value": action},
                ],
                "images": sampled_imgs,
            }
        )
    return episode_results


def process_episode_vlnce(ep, img_root, data_root):
    episode_results = []
    episode_id = str(ep["episode_id"])
    instruction = ep["instruction"]["instruction_text"].strip()
    img_dir = os.path.join(img_root, episode_id)

    try:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    except FileNotFoundError:
        print(f"Warning: Directory not found for VLN-CE episode {episode_id}: {img_dir}")
        return []

    for i in range(len(img_files)):
        if i <= max_history_images:
            idxs = list(range(i + 1))
        else:
            idxs = np.linspace(0, i, 9, dtype=int).tolist()
        sampled_imgs = [_rel(img_files[j], data_root) for j in idxs]

        his_img_tags = "<image>" * (len(sampled_imgs) - 1)
        name_parts = os.path.basename(img_files[i]).replace(".png", "").split("_")
        action = (
            f"{name_parts[-2].upper()}_{name_parts[-1].upper()}"
            if len(name_parts) > 3
            else name_parts[-1].upper()
        )
        episode_results.append(
            {
                "id": f"{episode_id}/{os.path.basename(img_files[i])}",
                "conversations": [
                    {"from": "human", "value": VLN_PROMPT.format(his_img_tags=his_img_tags, instruction=instruction)},
                    {"from": "gpt", "value": action},
                ],
                "images": sampled_imgs,
            }
        )
    return episode_results


def main():
    parser = argparse.ArgumentParser(description="Build VLN training JSON for SpatialStack.")
    parser.add_argument(
        "--data_root",
        default=".",
        help="Root directory containing data/ (datasets, trajectory_data, dagger_data). Default: current dir.",
    )
    parser.add_argument(
        "--use_extra_data",
        action="store_true",
        help="Include ScaleVLN + DAgger R2R + DAgger RxR.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: data/train/train_r2r_rxr[_extra].json under data_root.",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    data = os.path.join(data_root, "data")

    paths = {
        "scalevln_img": os.path.join(data, "trajectory_data/ScaleVLN/images"),
        "scalevln_ann": os.path.join(data, "trajectory_data/ScaleVLN/annotations.json"),
        "dagger_r2r_img": os.path.join(data, "dagger_data/R2R/images"),
        "dagger_r2r_ann": os.path.join(data, "dagger_data/R2R/annotations.json"),
        "dagger_rxr_img": os.path.join(data, "dagger_data/RxR/images"),
        "dagger_rxr_ann": os.path.join(data, "dagger_data/RxR/annotations.json"),
        "r2r_img": os.path.join(data, "trajectory_data/R2R/train"),
        "r2r_ann": os.path.join(data, "datasets/r2r/train/train.json.gz"),
        "rxr_img": os.path.join(data, "trajectory_data/RxR/train"),
        "rxr_ann": os.path.join(data, "datasets/rxr/train/train_guide.json.gz"),
    }
    act_map = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    all_results = []
    print(f"Data root: {data_root}")

    if args.use_extra_data:
        for key in ("scalevln_ann", "dagger_r2r_ann", "dagger_rxr_ann"):
            if not os.path.isfile(paths[key]):
                raise FileNotFoundError(f"Missing required file for --use_extra_data: {paths[key]}")

    with gzip.open(paths["r2r_ann"], "rt", encoding="utf-8") as f:
        data_r2r = json.load(f)
    with gzip.open(paths["rxr_ann"], "rt", encoding="utf-8") as f:
        data_rxr = json.load(f)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        if args.use_extra_data:
            with open(paths["scalevln_ann"], encoding="utf-8") as f:
                data_scalevln = json.load(f)
            with open(paths["dagger_r2r_ann"], encoding="utf-8") as f:
                data_dagger_r2r = json.load(f)
            with open(paths["dagger_rxr_ann"], encoding="utf-8") as f:
                data_dagger_rxr = json.load(f)

            for label, dataset, img_key in [
                ("ScaleVLN", data_scalevln, "scalevln_img"),
                ("DAgger R2R", data_dagger_r2r, "dagger_r2r_img"),
                ("DAgger RxR", data_dagger_rxr, "dagger_rxr_img"),
            ]:
                print(f"\nProcessing {label}...")
                fn = partial(
                    process_episode_scalevln,
                    img_root=paths[img_key],
                    act_map=act_map,
                    data_root=data_root,
                )
                for episode_res in tqdm(executor.map(fn, dataset), total=len(dataset)):
                    all_results.extend(episode_res)
                print(f"Finished {label}. Total samples: {len(all_results)}")

        print("\nProcessing R2R...")
        fn_r2r = partial(process_episode_vlnce, img_root=paths["r2r_img"], data_root=data_root)
        for episode_res in tqdm(executor.map(fn_r2r, data_r2r["episodes"]), total=len(data_r2r["episodes"])):
            all_results.extend(episode_res)

        print("\nProcessing RxR...")
        fn_rxr = partial(process_episode_vlnce, img_root=paths["rxr_img"], data_root=data_root)
        for episode_res in tqdm(executor.map(fn_rxr, data_rxr["episodes"]), total=len(data_rxr["episodes"])):
            all_results.extend(episode_res)

    if args.output:
        output_path = args.output
    else:
        os.makedirs(os.path.join(data, "train"), exist_ok=True)
        name = "train_r2r_rxr_extra.json" if args.use_extra_data else "train_r2r_rxr.json"
        output_path = os.path.join(data, "train", name)

    print(f"\nSaving {len(all_results)} samples to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
