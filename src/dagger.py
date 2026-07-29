"""DAgger trajectory collection for SpatialStack (Qwen3.5 + streaming VGGT).

Ported from JanusVLN's ``src/dagger.py``. The collection loop, expert-mixing
(``beta = p ** data_it``), error-tolerance correction and quality filtering are
kept identical so the emitted data is drop-in compatible with JanusVLN's DAgger
format. What changes is the model driver and the geometry cache handling:

  * model wrapper  -> ``SpatialStackVLN_Inference`` (from ``evaluation.py``)
  * per-episode reset -> ``model.reset_geometry_cache()`` (was ``past_key_values_vggt=None``)
  * ``call_model``  -> takes ``frame_indices=`` (frame-strict streaming) and
                       returns parsed action NAMES ("MOVE_FORWARD", ...).

Streaming-geometry contract (see geometry_encoders/vggt_encoder.py): the VGGT KV
cache and the frame-strict buffer assume **buffer frame i == trajectory frame i**,
i.e. the model is invoked on every trajectory frame contiguously. JanusVLN skipped
the model on expert steps; here we ALWAYS call the model each step (to advance the
geometry) and only then arbitrate whether to execute the model's or the expert's
action. For the released recipe (dagger_p=0) this is what happened anyway.

Output layout (identical to JanusVLN, consumed directly by
``scripts/data/create_janus_vln_data.py`` -> ``process_episode_scalevln``):

    <output>/images/<scene>_<dataset>_<episode:06d>/rgb/<step:03d>.jpg
    <output>/annotations.json    # [{id, video, instructions, actions}]  actions[0] = -1
"""

import argparse
import copy  # noqa: F401  (kept for parity with JanusVLN import surface)
import gzip
import json
import os
import random
import sys
import time
from typing import Dict

import numpy as np
import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image

import habitat
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    append_text_underneath_image,
    images_to_video,
    observations_to_image,
)
from habitat_baselines.config.default import get_config as get_habitat_config

from habitat_extensions import measures  # noqa: F401  (registers PL / oracle measures)
from utils.dist import get_rank, get_world_size, init_distributed_mode
from evaluation import SpatialStackVLN_Inference, VLNEvaluator


DATASET = "r2r"
CONFIG_PATH = "./config/vln_dagger.yaml"

DEFAULT_EPISODE_LENGTH = 60
MIDGOAL_RADIUS = 0.5
GOAL_RADIUS = 0.25
RELATIVE_PATH_LENGTH_THRESHOLD = 0.93
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.85


def image_resize(img, size, channels_last=True):
    """Local replacement for JanusVLN's habitat_extensions.maps.image_resize.

    Resizes an HxWxC (channels_last) uint8 RGB frame to ``size=(H, W)`` with PIL.
    """
    target_h, target_w = int(size[0]), int(size[1])
    pil = Image.fromarray(np.asarray(img).astype(np.uint8))
    pil = pil.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil)


class DAggerCollector:
    def __init__(self, args, rank, world_size):
        self.device = torch.device("cuda")
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.dataset = self.args.dagger_dataset.lower()
        self.output_path = self.args.dagger_output_path
        self.data_path = self.args.dagger_data_path
        self.config = get_habitat_config(args.habitat_config_path)

        with gzip.open(self.args.dagger_gt_annotations_path, "rt", encoding="utf-8") as f:
            self.gt_annotations = json.load(f)

        with read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
            if torch.cuda.is_available():
                gpu_id = getattr(args, "local_rank", getattr(args, "gpu", 0))
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

        self.dagger_config = OmegaConf.create(
            {
                "p": self.args.dagger_p,
                "update_size": self.args.dagger_update_size,
                "commit_freq": self.args.dagger_commit_freq,
            }
        )
        if get_rank() == 0:
            print(self.dagger_config)

    def config_env(self, scene=None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.data_path = self.data_path
        return habitat.Env(config=self.config)

    def generate(
        self,
        env: habitat.Env,
        evaluator=None,
        save_video: bool = True,
        force_expert: bool = False,
    ) -> Dict:

        beta = 0 if self.dagger_config.p == 0 else self.dagger_config.p ** self.args.dagger_data_it

        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        episode = env.current_episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=1.8, return_one_hot=False)
        scene_id = episode.scene_id.split("/")[-2]
        episode_id = int(episode.episode_id)
        trajectory_id = getattr(episode, "trajectory_id", episode_id)
        instructions = episode.instruction.instruction_text
        ref_path = episode.reference_path

        observation = env.reset()
        annotation = []
        rgb_data_list = []
        step_id = 0
        actions = [-1]
        next_waypoint_id = 1

        if save_video:
            os.makedirs(os.path.join(self.output_path, "videos"), exist_ok=True)

        vis_frames = []
        left_expert_actions_num = 0
        from_expert = True if force_expert else False
        force_episode_end = False
        model_success = True
        action_seq = []
        rgb_list = []
        # Reset the streaming VGGT KV cache + frame-strict buffer for this episode.
        evaluator.model.reset_geometry_cache()
        metrics = None
        accumulated_error = 0

        ref_actions_len = DEFAULT_EPISODE_LENGTH
        ref_actions_len = len(self.gt_annotations[str(episode_id)]["actions"])

        while not env.episode_over:
            rgb = observation["rgb"]

            rgb_path = os.path.join(
                self.output_path,
                "images",
                f"{scene_id}_{self.dataset}_{episode_id:06d}",
                "rgb",
                f"{step_id:03d}.jpg",
            )
            rgb_data_list.append((rgb, rgb_path))

            image = Image.fromarray(rgb).convert("RGB")
            rgb_list.append(image)

            # --- Always advance the streaming geometry: encode this frame + get the
            # model's proposed action every step, keeping buffer frame i == traj frame i.
            if evaluator is not None:
                history_len = len(rgb_list) - 1
                if history_len <= evaluator.num_history:
                    frame_indices = list(range(len(rgb_list)))
                else:
                    frame_indices = np.linspace(
                        0, history_len, evaluator.num_history + 1, dtype=int
                    ).tolist()
                images = [rgb_list[i] for i in frame_indices]
                model_action_name = evaluator.model.call_model(
                    images, instructions, step_id, frame_indices=frame_indices
                )[0]
                model_action = evaluator.actions2idx.get(model_action_name, [0])[0]
            else:
                model_action = None

            # --- DAgger arbitration (expert mixing), identical to JanusVLN ---
            if evaluator is not None:
                if len(action_seq) == 0 and left_expert_actions_num == 0:
                    from_expert = True if force_expert else random.random() < beta

                if len(action_seq) == 0:
                    if left_expert_actions_num > 0:
                        action = agent.get_next_action(ref_path[next_waypoint_id])
                        action_seq = [action]
                        left_expert_actions_num -= 1
                    else:
                        if from_expert:
                            action = agent.get_next_action(ref_path[next_waypoint_id])
                            action_seq = [action]
                            left_expert_actions_num = self.args.num_future_steps - 1
                        else:
                            action_seq = [model_action]
            else:
                action = agent.get_next_action(ref_path[next_waypoint_id])
                action_seq = [action]

            action_source = "expert" if from_expert else "model"

            if len(action_seq) == 0:
                action_seq = [0]

            action = action_seq.pop(0)
            if action != agent.get_next_action(ref_path[next_waypoint_id]):
                accumulated_error += 1

            while agent.get_next_action(ref_path[next_waypoint_id]) == 0:
                next_waypoint_id += 1
                force_expert = False
                left_expert_actions_num = 0
                if next_waypoint_id == len(ref_path) - 1:
                    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
                if next_waypoint_id >= len(ref_path):
                    force_episode_end = True
                    action = 0
                    action_source = "expert"
                    break

            metrics = env.get_metrics()
            wp_id_available = next_waypoint_id < len(ref_path)

            error_not_toleranted = (
                (from_expert is False and action == 0 and metrics["distance_to_goal"] >= 3.0)
                or (accumulated_error / max(1, int(ref_actions_len / (len(ref_path) - 1))) > 0.8)
                or accumulated_error > 12
            )
            if wp_id_available and error_not_toleranted:
                model_success = False
                force_expert = True
                accumulated_error = 0
                action = agent.get_next_action(ref_path[next_waypoint_id])
                action_source = "expert"
                action_seq = []

            if action == 0 and not force_episode_end:
                action = agent.get_next_action(ref_path[next_waypoint_id])

            observation = env.step(action)
            metrics = env.get_metrics()

            if save_video:
                if metrics.get("top_down_map") is not None:
                    resized_rgb = np.array(
                        image_resize(
                            img=observation["rgb"],
                            size=(int(observation["rgb"].shape[0] * 1.6), int(observation["rgb"].shape[1] * 1.6)),
                            channels_last=True,
                        )
                    )
                    frame = observations_to_image({"rgb": resized_rgb}, metrics)
                    instr_text = (
                        episode.instruction.instruction_text
                        if isinstance(episode.instruction.instruction_text, str)
                        else episode.instruction.instruction_text[0]
                    )
                    frame = append_text_underneath_image(frame, instr_text)
                    frame = append_text_underneath_image(frame, action_source)
                    frame = append_text_underneath_image(frame, f"force_expert is {force_expert}")
                    frame = append_text_underneath_image(frame, f"step: {step_id}")
                    frame = append_text_underneath_image(frame, f"next wp id: {next_waypoint_id} / {len(ref_path) - 1}")
                    vis_frames.append(frame)

            if env.episode_over or force_episode_end:
                break
            actions.append(action)
            step_id += 1

        assert len(rgb_data_list) == len(actions), (
            f"Length of rgbs and actions mismatch, rgb_data_list: {len(rgb_data_list)}, actions: {len(actions)}"
        )

        annotation.append(
            {
                "id": episode_id,
                "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                "instructions": instructions if isinstance(instructions, list) else [instructions],
                "actions": actions,
            }
        )

        episode_save = metrics["distance_to_goal"] < MIDGOAL_RADIUS and (
            ((not model_success) and (metrics["pl"] < RELATIVE_PATH_LENGTH_THRESHOLD))
            or (metrics["pl"] < SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD)
        )
        if episode_save:
            os.makedirs(
                os.path.join(self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb"),
                exist_ok=True,
            )
            for rgb, rgb_path in rgb_data_list:
                Image.fromarray(rgb).convert("RGB").save(rgb_path)

        if save_video:
            tag = "save" if episode_save else "notsave"
            images_to_video(
                vis_frames,
                os.path.join(self.output_path, "videos"),
                f"{tag}_{scene_id}_{self.dataset}_{episode_id:06d}",
                fps=6,
                quality=10,
            )
            vis_frames.clear()

        metrics.update(
            {
                "step_id": step_id,
                "ref_actions_len": ref_actions_len,
                "accumulated_error": accumulated_error,
                "save": int(episode_save),
                "model_success": model_success,
                "force_episode_end": force_episode_end,
            }
        )

        return dict(anno=annotation, metrics=metrics)

    def _dump_rank_annotations(self, annotations):
        """Merge new annotations into this rank's file, de-duplicating by video path."""
        tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")
        merged_anno = json.load(open(tgt_anno_path)) if os.path.exists(tgt_anno_path) else []
        merged_anno.extend(annotations)
        seen = set()
        temp_anno = []
        for item in merged_anno:
            if item["video"] not in seen:
                seen.add(item["video"])
                temp_anno.append(item)
        with open(tgt_anno_path, "w") as json_file:
            json_file.write(json.dumps(temp_anno, indent=4))

    def update_dataset(self, evaluator, dataset=None):
        """Roll out episodes across ranks, collect the ones that pass the quality filter."""
        seed = self.rank
        random.seed(seed)
        np.random.seed(seed)

        if evaluator is None:
            self.args.force_expert = True

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        env = self.config_env()
        scene_episode_dict = {}
        episode_uuids = []
        start = time.time()
        for episode in env.episodes:
            episode_uuid = (episode.scene_id, episode.episode_id, getattr(episode, "trajectory_id", episode.episode_id))
            episode_uuids.append(episode_uuid)
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)
        sampled_episodes_uuids = episode_uuids
        sampled_episodes_by_scene = {}
        for scene_id in sorted(scene_episode_dict.keys()):
            sampled_episodes_traj_ids = [
                (u[1], u[2]) for u in sampled_episodes_uuids if u[0] == scene_id
            ]
            sampled_episodes_by_scene[scene_id] = [
                ep for ep in scene_episode_dict[scene_id]
                if (ep.episode_id, getattr(ep, "trajectory_id", ep.episode_id)) in sampled_episodes_traj_ids
            ]

        num_collect_episodes = 0
        annotations = []
        with tqdm.tqdm(
            total=min(self.dagger_config.update_size, len(sampled_episodes_uuids)) // self.world_size,
            dynamic_ncols=True,
        ) as pbar, torch.no_grad():
            for scene_id in sorted(scene_episode_dict.keys()):
                episodes = sampled_episodes_by_scene[scene_id]
                if len(episodes) == 0:
                    continue
                print(f"scene_id: {scene_id}, len of episodes: {len(episodes)}")
                for episode in episodes[self.rank::self.world_size]:
                    assert scene_id == episode.scene_id, f"scene mismatch: {scene_id} vs {episode.scene_id}"
                    scan = episode.scene_id.split("/")[-2]
                    env.current_episode = episode
                    env.current_episode.goals[0].radius = MIDGOAL_RADIUS

                    episode_dagger = self.generate(
                        env=env,
                        evaluator=evaluator,
                        save_video=self.args.dagger_save_video,
                        force_expert=self.args.force_expert,
                    )

                    with open(os.path.join(self.output_path, "result.json"), "a") as f:
                        result = {
                            "scene": scan,
                            "episode_id": episode.episode_id,
                            "trajectory_id": getattr(episode, "trajectory_id", episode.episode_id),
                            "save": episode_dagger["metrics"]["save"],
                            "model_success": episode_dagger["metrics"]["model_success"],
                            "success": episode_dagger["metrics"]["success"],
                            "relative_pl": episode_dagger["metrics"]["pl"],
                            "step_id": episode_dagger["metrics"]["step_id"],
                            "ref_actions": episode_dagger["metrics"]["ref_actions_len"],
                            "accumulated_error": episode_dagger["metrics"]["accumulated_error"],
                            "force_episode_end": episode_dagger["metrics"]["force_episode_end"],
                        }
                        f.write(json.dumps(result) + "\n")

                    if not episode_dagger["metrics"]["save"]:
                        pbar.update()
                        continue

                    print(
                        f"model_success = {episode_dagger['metrics']['model_success']}, "
                        f"scene {scan} id {episode.episode_id} trajectory {getattr(episode, 'trajectory_id', episode.episode_id)}"
                    )

                    annotations.extend(episode_dagger["anno"])
                    pbar.update()
                    num_collect_episodes += 1

                    if num_collect_episodes % self.dagger_config.commit_freq == 0:
                        self._dump_rank_annotations(annotations)
                        annotations = []

                    if num_collect_episodes >= self.dagger_config.update_size:
                        break
                if num_collect_episodes >= self.dagger_config.update_size:
                    break

            self._dump_rank_annotations(annotations)
            annotations = []
            print(f"save scene_id {scene_id} with total episodes {num_collect_episodes} time cost {time.time() - start}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if get_rank() == 0:
            tgt_anno_path = os.path.join(self.output_path, "annotations.json")
            merged_anno = []
            sub_tgt_anno_list = [
                os.path.join(self.output_path, f)
                for f in os.listdir(self.output_path)
                if f.startswith("annotations_") and f.endswith(".json")
            ]
            for sub_tgt_anno_path in sub_tgt_anno_list:
                if os.path.exists(sub_tgt_anno_path):
                    merged_anno.extend(json.load(open(sub_tgt_anno_path)))
            merged_anno = sorted(merged_anno, key=lambda x: x["id"])
            seen = set()
            temp_anno = []
            for item in merged_anno:
                if item["video"] not in seen:
                    seen.add(item["video"])
                    temp_anno.append(item)
            with open(tgt_anno_path, "w") as json_file:
                json_file.write(json.dumps(temp_anno, indent=4))
            print(f"[rank0] merged {len(temp_anno)} episodes -> {tgt_anno_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--geometry_encoder_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default=CONFIG_PATH)
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="./results/dagger")
    parser.add_argument("--num_future_steps", type=int, default=1)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--model_max_length", type=int, default=4096)

    parser.add_argument("--dagger_p", type=float, default=0.0)
    parser.add_argument("--dagger_update_size", type=int, default=160000)
    parser.add_argument("--dagger_commit_freq", type=int, default=50)
    parser.add_argument("--dagger_dataset", type=str, default=DATASET)
    parser.add_argument("--force_expert", action="store_true", default=False)
    parser.add_argument("--dagger_data_it", type=int, default=0)
    parser.add_argument("--dagger_output_path", type=str, default="data/dagger_data/R2R")
    parser.add_argument("--dagger_data_path", type=str, default=None)
    parser.add_argument("--dagger_gt_annotations_path", type=str, required=True)
    parser.add_argument("--dagger_save_video", action="store_true", default=False)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    init_distributed_mode(args)
    os.makedirs(args.dagger_output_path, exist_ok=True)

    geometry_encoder_path = args.geometry_encoder_path or os.environ.get("GEOMETRY_ENCODER_PATH")
    model = SpatialStackVLN_Inference(
        args.model_path,
        device=f"cuda:{args.local_rank}",
        geometry_encoder_path=geometry_encoder_path or None,
    )

    rank = get_rank()
    world_size = get_world_size()

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args,
    )

    collector = DAggerCollector(args=args, rank=rank, world_size=world_size)
    collector.update_dataset(evaluator=evaluator)


if __name__ == "__main__":
    main()
