# ros2_inference_model

ROS2 wrapper around SpatialStack's VLN model — reuses `SpatialStackVLN_Inference`
from `src/evaluation.py`, the same class the Habitat-Sim evaluator
(`VLNEvaluator` / `eval_janus_vln.sh`) uses, so the real-robot rollout loop
(prompt building, streaming-VGGT geometry, frame-strict windowing, history
subsampling) never drifts from what the model was trained/evaluated against.

Publishes `vm_vln_msgs/VlnHybridOutput` (`DISCRETE_ACTION` only — SpatialStack
has no trajectory head) on `/vln_hybrid_output`, the exact topic
`vln_robot_controller` already subscribes to
(`vm_vln_deployment/src/vln_robot_controller/config/controller.yaml`,
`dual_sys_output_topic`). That makes this node a drop-in alternative to
`vm_vln_deployment`'s `internnav_inference` behind the same robot controller —
same interface, different model.

`vm_vln_msgs` is vendored here (`ros2_ws/src/vm_vln_msgs`, copied from
`vm_vln_deployment/src/vm_vln_msgs`) so this `ros2_ws` builds standalone
without sourcing the deployment workspace first. If `VlnHybridOutput.msg` ever
changes in one place, update the other.

## Prerequisites

Install the SpatialStack conda env and download checkpoints as usual (see the
top-level `README.md` / `TRAINING_JANUS_VLN.md` — not repeated here). You need
a VLN-trained checkpoint (e.g. `train_janus_vln.sh` output) and the VGGT-1B
geometry encoder weights.

## Build

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

## Configuration

All topic names and runtime knobs live in
[`config/inference.yaml`](config/inference.yaml) (`ros__parameters` per node,
standard ROS2 param-file format). `set_instruction_topic` /
`task_success_topic` are shared by both nodes (the CLI publishes/subscribes
the same pair the inference node subscribes/publishes) and are duplicated
across the two `ros__parameters` blocks rather than using a YAML anchor —
`rcl`'s parameter YAML parser rejects aliases ("Will not support aliasing").
Keep the two copies identical if you change them.

`inference_node.py` auto-loads this file's `spatialstack_inference_node`
block as its parameter *defaults* via `get_package_share_directory` — so a
plain `ros2 run ros2_inference_model inference_node.py` already picks it up,
no `--ros-args` needed. `-p name:=value` / an explicit `--params-file` still
override it as usual (ROS2 parameter precedence). `set_instruction_node` (C++)
does not auto-load the file; pass it explicitly if you change the topic names
from their defaults (see below).

`model_path` / `geometry_encoder_path` are deliberately **not** in the YAML:
any key present there — even `""` — would override the node's env-var
fallback. Instead, `run_inference_node.sh` hardcodes both paths near its top
(same default checkpoint as `scripts/evaluation/eval_janus_vln_scene.sh`) —
**running on a different machine, edit those two lines** (or override for one
run via `MODEL_PATH=... GEOMETRY_ENCODER_PATH=...`, or
`-p model_path:=... -p geometry_encoder_path:=...`).

## Run

Inference node (loads the model once, then runs the rollout loop against the
live camera feed):

```bash
conda activate dagger
bash src/ros2_inference_model/scripts/run_inference_node.sh
```

`run_inference_node.sh` exports `PYTHONPATH` to SpatialStack's `src/` (same
pattern every other SpatialStack launcher script uses) and passes
`--params-file config/inference.yaml` explicitly before calling `ros2 run`.
If your shell already has that `PYTHONPATH` set (and `MODEL_PATH` /
`GEOMETRY_ENCODER_PATH`), `ros2 run` works directly (and still auto-loads the
same config, per above):

```bash
MODEL_PATH=<ckpt> GEOMETRY_ENCODER_PATH=<vggt-path> \
  ros2 run ros2_inference_model inference_node.py
```

Set the instruction — interactive CLI, waits for `task_success` before
prompting again (mirrors `internnav_inference`'s `set_instruction_node`):

```bash
ros2 run ros2_inference_model set_instruction_node \
  --ros-args --params-file install/ros2_inference_model/share/ros2_inference_model/config/inference.yaml
```

(Only needed if you changed `set_instruction_topic` / `task_success_topic`
from their defaults — otherwise the node's built-in defaults already match.)
Or publish the instruction directly:

```bash
ros2 topic pub --once /vln/input/set_instruction std_msgs/msg/String "{data: 'go to the kitchen'}"
```

## Topics / parameters

All defined in [`config/inference.yaml`](config/inference.yaml):

| Parameter | Default | Type | Direction | Purpose |
|---|---|---|---|---|
| `rgb_topic` | `/zed_x_mini/colored_image_compressed` | `sensor_msgs/CompressedImage` | sub | Monocular RGB stream — SpatialStack needs no depth input |
| `set_instruction_topic` | `/vln/input/set_instruction` | `std_msgs/String` | sub | New instruction; resets history + streaming-VGGT KV cache on the next frame |
| `output_topic` | `/vln_hybrid_output` | `vm_vln_msgs/VlnHybridOutput` | pub | `DISCRETE_ACTION`, one per inference step |
| `task_success_topic` | `/vln/output/task_success` | `std_msgs/Bool` | pub | `True` the instant STOP is predicted (no debounce — see below) |
| `device` | `cuda:0` | — | — | Torch device |
| `num_history` | `8` | — | — | Matches `eval_janus_vln.sh`'s `--num_history` |
| `max_new_tokens` | `24` | — | — | Generation length cap |

## Behavior notes

- History is never pruned mid-episode. Each step re-subsamples
  `num_history + 1` frames from the *entire* trajectory so far via
  `np.linspace(0, history_len, num_history + 1)`, exactly matching
  `VLNEvaluator.eval_action` in `src/evaluation.py` — the model was trained on
  that history distribution, not a fixed sliding window.
- A new instruction resets `rgb_list`, the step counter, and the streaming
  VGGT KV cache (`model.reset_geometry_cache()`) on the next frame the
  inference loop actually processes — matching `internnav_inference`'s
  `policy_init` handling, including under dropped frames (the image callback
  keeps only the latest frame; `policy_init` isn't cleared until a reset
  actually happens, so it survives being queued behind dropped frames).
- **STOP is authoritative and immediately ends the episode.** Unlike
  InternNav's dual-system STOP (which needed a multi-second debounce before
  it could be trusted), this model's STOP is treated as success on the very
  first prediction: the node publishes the STOP action once, publishes
  `task_success=True`, then goes idle — `current_instruction` is cleared and
  the inference loop stops calling the model (and stops publishing on
  `output_topic`) until a new instruction arrives. This also avoids running
  the policy far outside the episode lengths it was trained on, which is what
  caused it to eventually drift off STOP if the loop were left running.
