#!/usr/bin/env python3
"""SpatialStack VLN inference node: RGB stream + instruction -> VlnHybridOutput.

Reuses SpatialStackVLN_Inference from src/evaluation.py (the same wrapper used
for Habitat-Sim eval) so the real-robot loop - prompt building, streaming-VGGT
geometry, frame-strict windowing, history subsampling - never drifts from the
Habitat evaluator. The per-step logic below mirrors VLNEvaluator.eval_action
in that file; keep them in sync if that loop changes.
"""
import os
import sys
import threading
import queue

import cv2
import numpy as np
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool, UInt32

from vm_vln_msgs.msg import VlnHybridOutput


def _load_config_defaults(node_name: str) -> dict:
    """Read config/inference.yaml's ros__parameters block for `node_name`.

    Used as the `declare_parameter` default, so a bare `ros2 run` (no
    --params-file) still picks up the checked-in config, while -p /
    --params-file overrides still take priority as usual.
    """
    try:
        share_dir = get_package_share_directory("ros2_inference_model")
    except PackageNotFoundError:
        return {}
    config_path = os.path.join(share_dir, "config", "inference.yaml")
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return {}
    return (data.get(node_name) or {}).get("ros__parameters") or {}


def _ensure_spatialstack_on_path():
    try:
        import evaluation  # noqa: F401
        return
    except ModuleNotFoundError as e:
        # Only treat "evaluation.py itself isn't on sys.path" as fixable here.
        # Any other missing module (torch, transformers, habitat, ...) means
        # evaluation.py WAS found but the current Python isn't the SpatialStack
        # conda env -- that's a real error the caller needs to see as-is, not
        # a path problem we can silently work around.
        if e.name != "evaluation":
            raise
    root = os.environ.get("SPATIALSTACK_ROOT")
    if not root:
        raise ImportError(
            "Could not find SpatialStack's 'evaluation' module (src/evaluation.py). Run "
            "this node via scripts/run_inference_node.sh (sets PYTHONPATH automatically), "
            "or export PYTHONPATH=<SpatialStack>/src, or set SPATIALSTACK_ROOT=<SpatialStack> "
            "before running."
        )
    src_dir = os.path.join(root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import evaluation  # noqa: F401


_ensure_spatialstack_on_path()

from evaluation import SpatialStackVLN_Inference  # noqa: E402


ACTION_TO_CODE = {
    "STOP": VlnHybridOutput.STOP,
    "MOVE_FORWARD": VlnHybridOutput.MOVE_FORWARD,
    "TURN_LEFT": VlnHybridOutput.TURN_LEFT,
    "TURN_RIGHT": VlnHybridOutput.TURN_RIGHT,
}


class SpatialStackInferenceNode(Node):
    def __init__(self):
        super().__init__("spatialstack_inference_node")

        # model_path / geometry_encoder_path are intentionally not read from
        # this dict (see config/inference.yaml's comment): they fall back to
        # MODEL_PATH / GEOMETRY_ENCODER_PATH env vars instead.
        cfg = _load_config_defaults("spatialstack_inference_node")

        self.declare_parameter("model_path", os.environ.get("MODEL_PATH", ""))
        self.declare_parameter("geometry_encoder_path", os.environ.get("GEOMETRY_ENCODER_PATH", ""))
        self.declare_parameter("device", cfg.get("device", "cuda:0"))
        self.declare_parameter("num_history", cfg.get("num_history", 8))
        self.declare_parameter("max_new_tokens", cfg.get("max_new_tokens", 24))
        self.declare_parameter("rgb_topic", cfg.get("rgb_topic", "/zed_x_mini/colored_image_compressed"))
        self.declare_parameter("set_instruction_topic", cfg.get("set_instruction_topic", "/vln/input/set_instruction"))
        self.declare_parameter("task_success_topic", cfg.get("task_success_topic", "/vln/output/task_success"))
        self.declare_parameter("output_topic", cfg.get("output_topic", "/vln_hybrid_output"))
        self.declare_parameter("action_done_topic", cfg.get("action_done_topic", "/vln_controller/action_done"))
        self.declare_parameter("action_sync_enabled", cfg.get("action_sync_enabled", True))

        model_path = self.get_parameter("model_path").value
        if not model_path:
            raise ValueError(
                "model_path is required: pass --ros-args -p model_path:=<ckpt> or set MODEL_PATH."
            )
        geometry_encoder_path = self.get_parameter("geometry_encoder_path").value or None
        device = self.get_parameter("device").value
        self.num_history = int(self.get_parameter("num_history").value)
        self.max_new_tokens = int(self.get_parameter("max_new_tokens").value)

        self.get_logger().info(f"Loading SpatialStack VLN model from '{model_path}' on {device}...")
        self.model = SpatialStackVLN_Inference(
            model_path, device=device, geometry_encoder_path=geometry_encoder_path
        )
        self.get_logger().info("Model loaded.")

        rgb_topic = self.get_parameter("rgb_topic").value
        set_instruction_topic = self.get_parameter("set_instruction_topic").value
        task_success_topic = self.get_parameter("task_success_topic").value
        output_topic = self.get_parameter("output_topic").value
        action_done_topic = self.get_parameter("action_done_topic").value
        self.action_sync_enabled = bool(self.get_parameter("action_sync_enabled").value)

        self.create_subscription(CompressedImage, rgb_topic, self._on_rgb, 10)
        self.create_subscription(String, set_instruction_topic, self._on_set_instruction, 10)
        self.create_subscription(UInt32, action_done_topic, self._on_action_done, 10)
        self.task_success_pub = self.create_publisher(Bool, task_success_topic, 10)
        self.output_pub = self.create_publisher(VlnHybridOutput, output_topic, 10)

        self.lock = threading.Lock()
        self.infer_queue = queue.Queue(maxsize=1)  # only the latest frame matters
        self.current_instruction = None
        self.policy_init = False
        self.last_rgb_stamp = None

        # Set by _on_action_done once the robot confirms a discrete step
        # actually finished; _infer_loop waits on it before pulling the next
        # frame, so the next inference never runs on an image captured before
        # the previous action had any visible effect.
        self.action_done_event = threading.Event()
        self.last_done_step = None

        # Full trajectory history (PIL images). Never pruned: the model was
        # trained on linspace-subsampled history over the WHOLE episode so
        # far, not a fixed sliding window (see VLNEvaluator.eval_action).
        self.rgb_list = []
        self.step_id = 0

        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

        self.get_logger().info(
            f"Ready. rgb={rgb_topic} set_instruction={set_instruction_topic} "
            f"output={output_topic} task_success={task_success_topic}"
        )

    def _on_set_instruction(self, msg):
        instruction = msg.data.strip()
        if not instruction:
            return
        self.get_logger().info(f"New instruction: '{instruction}'")
        self.current_instruction = instruction
        self.policy_init = True

    def _on_action_done(self, msg):
        self.last_done_step = msg.data
        self.action_done_event.set()

    def _on_rgb(self, msg):
        if self.current_instruction is None:
            return
        if msg.header.stamp == self.last_rgb_stamp:
            return
        self.last_rgb_stamp = msg.header.stamp

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warn("Failed to decode RGB frame; skipping.")
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"Error decoding RGB frame: {e}")
            return

        # Only keep the latest frame; drop anything the inference loop hasn't
        # consumed yet rather than letting the queue back up.
        if not self.infer_queue.empty():
            try:
                self.infer_queue.get_nowait()
            except queue.Empty:
                pass
        # NOTE: don't clear self.policy_init here. It must stay True across
        # any dropped frames so the inference loop still resets the episode
        # on the next frame it actually processes.
        self.infer_queue.put((rgb, self.current_instruction, self.policy_init))

    def _infer_loop(self):
        while rclpy.ok():
            try:
                rgb, instruction, policy_init = self.infer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if instruction != self.current_instruction:
                # Stale frame queued before a STOP reset us to idle (or
                # before a newer instruction superseded this one); drop it.
                continue

            action = None
            published_step = None
            try:
                with self.lock:
                    if policy_init:
                        self.model.reset_geometry_cache()
                        self.rgb_list = []
                        self.step_id = 0
                        self.policy_init = False
                        self._publish_task_success(False)

                    self.rgb_list.append(Image.fromarray(rgb).convert("RGB"))
                    history_len = len(self.rgb_list) - 1
                    if history_len <= self.num_history:
                        frame_indices = list(range(len(self.rgb_list)))
                    else:
                        frame_indices = np.linspace(
                            0, history_len, self.num_history + 1, dtype=int
                        ).tolist()
                    images = [self.rgb_list[i] for i in frame_indices]

                    action = self.model.call_model(
                        images,
                        instruction,
                        self.step_id,
                        gen_kwargs={"max_new_tokens": self.max_new_tokens},
                        frame_indices=frame_indices,
                    )[0]
                    self.step_id += 1
                    published_step = self.step_id
                    self.action_done_event.clear()
                    self._publish_action(action)

                    if action == "STOP":
                        # Unlike InternNav's dual-system STOP (which needed
                        # debouncing), this model's STOP is authoritative:
                        # treat it as immediate success and go idle -- stop
                        # inferring until a new instruction is set.
                        self.get_logger().info("STOP predicted -> task success, back to idle.")
                        self._publish_task_success(True)
                        self.current_instruction = None
                        self.rgb_list = []
                        self.step_id = 0
            except Exception as e:
                self.get_logger().error(f"Inference step failed: {e}")
                continue

            if self.action_sync_enabled and action is not None and action != "STOP":
                while rclpy.ok():
                    if self.action_done_event.wait(timeout=1.0) and self.last_done_step == published_step:
                        break
                    self.action_done_event.clear()

    def _publish_task_success(self, success: bool):
        msg = Bool()
        msg.data = success
        self.task_success_pub.publish(msg)

    def _publish_action(self, action: str):
        msg = VlnHybridOutput()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.step_index = self.step_id
        msg.type_of_output = VlnHybridOutput.DISCRETE_ACTION
        msg.action_codes = [ACTION_TO_CODE[action]]
        self.output_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpatialStackInferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
