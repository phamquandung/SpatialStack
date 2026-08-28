"""VGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional, List

from .base import BaseGeometryEncoder, GeometryEncoderConfig


def _slice1d(x, start, end):
    return x[:, start:end, ...]


def _slice2d(x, start, end):
    return x[:, :, start:end, ...]


def _slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


_DIM_TO_SLICE = {1: _slice1d, 2: _slice2d, 3: _slice3d}


class StartRecentKVCache:
  """Trim VGGT KV cache to start+recent windows (JanusVLN eval)."""

  def __init__(self, start_size=8, recent_size=48, k_seq_dim=2, v_seq_dim=2):
    self.start_size = start_size
    self.recent_size = recent_size
    self.cache_size = start_size + recent_size
    self.k_slice = _DIM_TO_SLICE[k_seq_dim]
    self.v_slice = _DIM_TO_SLICE[v_seq_dim]

  def __call__(self, past_key_values):
    if past_key_values is None:
      return None
    seq_len = past_key_values[0][0].size(2)
    if seq_len <= self.cache_size:
      return past_key_values
    return [
      [
        torch.cat(
          [self.k_slice(k, 0, self.start_size), self.k_slice(k, seq_len - self.recent_size, seq_len)],
          dim=2,
        ),
        torch.cat(
          [self.v_slice(v, 0, self.start_size), self.v_slice(v, seq_len - self.recent_size, seq_len)],
          dim=2,
        ),
      ]
      for k, v in past_key_values
    ]


class VGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        # Initialize VGGT model
        self.vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame    
        self.patch_size = 14
        self._vggt_pretrained_path = config.model_path
        self._depth_head_ready = False
        self._eval_streaming = False
        self._streaming_past_key_values = None
        self._streaming_frame_idx = 0
        self._history_anchor_manager = None
        self.last_vggt_ms = 0.0
        # Legacy frame-strict fallback buffers raw geometry. The Qwen3.5 language-add
        # eval path instead enables _eval_projected_cache and stores post-MLP deltas.
        self._eval_frame_strict = False
        self._eval_projected_cache = False
        self._eval_window_indices = None
        self._frame_feature_buffer = None
        # Eval-time VGGT KV-cache window (in frames). Defaults match JanusVLN (8+48=56).
        # Override via env to test the long-horizon geometry-drift hypothesis, e.g.
        # VGGT_KV_START=1 VGGT_KV_RECENT=8 caps the cache at 9 frames (training horizon).
        import os as _os
        _kv_start = int(_os.environ.get("VGGT_KV_START", "8"))
        _kv_recent = int(_os.environ.get("VGGT_KV_RECENT", "48"))
        self.kv_cache_mode = _os.environ.get(
            "VGGT_KV_CACHE_MODE", "start_recent"
        ).strip().lower()
        if self.kv_cache_mode not in ("start_recent", "ovggt"):
            raise ValueError(
                "VGGT_KV_CACHE_MODE must be 'start_recent' or 'ovggt', got "
                f"{self.kv_cache_mode!r}"
            )
        self.vggt_total_budget = int(_os.environ.get("VGGT_TOTAL_BUDGET", "200000"))
        self.vggt_importance_weight = float(
            _os.environ.get("VGGT_IMPORTANCE_WEIGHT", "0.5")
        )
        self.vggt_intra_frame_keep_ratio = float(
            _os.environ.get("VGGT_INTRA_FRAME_KEEP_RATIO", "1.0")
        )
        self.vggt_global_anchor_frames = int(
            _os.environ.get("VGGT_GLOBAL_ANCHOR_FRAMES", "1")
        )
        self.vggt_recent_protect_frames = int(
            _os.environ.get("VGGT_RECENT_PROTECT_FRAMES", "0")
        )
        self.vggt_history_anchor_strategy = _os.environ.get(
            "VGGT_HISTORY_ANCHOR_STRATEGY", "none"
        ).strip().lower()
        self.vggt_history_anchor_interval = int(
            _os.environ.get("VGGT_HISTORY_ANCHOR_INTERVAL", "250")
        )
        _min_anchor_interval = _os.environ.get(
            "VGGT_HISTORY_ANCHOR_MIN_INTERVAL", "100"
        ).strip().lower()
        self.vggt_history_anchor_min_interval = (
            None
            if _min_anchor_interval in ("", "none")
            else int(_min_anchor_interval)
        )
        self.vggt_history_anchor_max = int(
            _os.environ.get("VGGT_HISTORY_ANCHOR_MAX", "3")
        )
        self.vggt_history_anchor_keep_ratio = float(
            _os.environ.get("VGGT_HISTORY_ANCHOR_KEEP_RATIO", "0.05")
        )
        self.vggt_history_anchor_coverage_threshold = float(
            _os.environ.get("VGGT_HISTORY_ANCHOR_COVERAGE_THRESHOLD", "0.2")
        )
        self.vggt_history_anchor_sample_ratio = float(
            _os.environ.get("VGGT_HISTORY_ANCHOR_SAMPLE_RATIO", "0.1")
        )
        self.vggt_kv_debug = _os.environ.get("VGGT_KV_DEBUG", "0").lower() in (
            "1", "true", "yes"
        )
        if self.vggt_total_budget <= 0:
            raise ValueError("VGGT_TOTAL_BUDGET must be positive")
        if not 0.0 <= self.vggt_importance_weight <= 1.0:
            raise ValueError("VGGT_IMPORTANCE_WEIGHT must be in [0, 1]")
        if not 0.0 < self.vggt_intra_frame_keep_ratio <= 1.0:
            raise ValueError("VGGT_INTRA_FRAME_KEEP_RATIO must be in (0, 1]")
        if self.vggt_global_anchor_frames != 1:
            raise ValueError("Only VGGT_GLOBAL_ANCHOR_FRAMES=1 is supported")
        if self.vggt_recent_protect_frames < 0:
            raise ValueError("VGGT_RECENT_PROTECT_FRAMES must be non-negative")
        from ..vggt.utils.history_anchor import HistoryAnchorConfig

        self._history_anchor_config = HistoryAnchorConfig(
            strategy=self.vggt_history_anchor_strategy,
            interval=self.vggt_history_anchor_interval,
            min_anchor_interval=self.vggt_history_anchor_min_interval,
            max_anchors=self.vggt_history_anchor_max,
            coverage_threshold=self.vggt_history_anchor_coverage_threshold,
            sample_ratio=self.vggt_history_anchor_sample_ratio,
            anchor_keep_ratio=self.vggt_history_anchor_keep_ratio,
        )
        if (
            self.kv_cache_mode == "ovggt"
            and self.vggt_history_anchor_strategy == "coverage"
        ):
            raise ValueError(
                "VGGT_HISTORY_ANCHOR_STRATEGY=coverage requires OVGGT camera/depth "
                "predictions, but SpatialStack constructs VGGT with those heads disabled. "
                "Use 'fixed_interval' for the faithful head-free OVGGT policy."
            )
        if (
            self.vggt_history_anchor_strategy == "fixed_interval"
            and self.vggt_history_anchor_max == 0
        ):
            raise ValueError(
                "VGGT_HISTORY_ANCHOR_MAX must be positive for fixed_interval"
            )
        print(
            f"[VGGTEncoder] KV cache mode={self.kv_cache_mode} "
            f"start={_kv_start} recent={_kv_recent} "
            f"total_budget={self.vggt_total_budget} "
            f"history_anchor={self.vggt_history_anchor_strategy}"
        )
        self._kv_cache_trim = StartRecentKVCache(start_size=_kv_start, recent_size=_kv_recent, k_seq_dim=2, v_seq_dim=2)

    def set_eval_streaming(self, enabled: bool) -> None:
        self._eval_streaming = bool(enabled)

    def set_eval_frame_strict(self, enabled: bool) -> None:
        """Incremental frame-strict eval: buffer each frame's geometry (encoded with the
        growing KV) and return the requested window per-frame instead of broadcasting."""
        self._eval_frame_strict = bool(enabled)
        if enabled and self._frame_feature_buffer is None:
            self._frame_feature_buffer = []

    def set_eval_projected_cache(self, enabled: bool) -> None:
        """Let the Qwen fusion path cache post-MLP deltas instead of raw VGGT features."""
        self._eval_projected_cache = bool(enabled)
        if enabled:
            self._frame_feature_buffer = None

    def set_eval_window_indices(self, indices) -> None:
        """Trajectory frame indices to gather from the per-frame buffer this step."""
        self._eval_window_indices = list(indices) if indices is not None else None

    def reset_streaming_cache(self) -> None:
        self._streaming_past_key_values = None
        self._streaming_frame_idx = 0
        self._history_anchor_manager = None
        self.last_vggt_ms = 0.0
        self._frame_feature_buffer = (
            [] if self._eval_frame_strict and not self._eval_projected_cache else None
        )
        self._eval_window_indices = None
        if hasattr(self.vggt.aggregator, "reset_ovggt_cache_state"):
            self.vggt.aggregator.reset_ovggt_cache_state()
        
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using VGGT and return the default (final) feature set."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]

        # Apply inverse reference frame transformation
        features = self._apply_inverse_reference_frame_transform(features)

        return features

    def encode_layers(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Encode images and return features from specific aggregator layers."""
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)

        # Determine dtype for mixed precision
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])

        n_image, _, height, width = images.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        spatial_merge_size = spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2

        tensor_features = []

        if layer_indices is None:
            layer_indices = [-2]

        for idx in layer_indices:
            tokens = aggregated_tokens_list[idx][0]
            tokens = self._apply_inverse_reference_frame_transform(tokens) # flip frames if ture
            patch_tokens = tokens[:, patch_start_idx:]
            camera_token = tokens[:, 0:1] # first token

            # reshape and trim
            patch_grid = patch_tokens.reshape(n_image, h_patch, w_patch, -1)
            trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
            trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
            patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
            patch_grid = patch_grid.reshape(n_image, trimmed_h // spatial_merge_size, spatial_merge_size, trimmed_w // spatial_merge_size, spatial_merge_size, -1)
            patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
            patch_tokens = patch_grid.reshape(n_image, trimmed_h * trimmed_w, -1)

            if not include_camera_token:
                geo_feature = patch_tokens
            else:
                geo_feature = torch.cat([camera_token, patch_tokens], dim=1)

            tensor_features.append(geo_feature.to(dtype).contiguous())

        self._maybe_debug_geometry_layers(
            layer_indices=layer_indices,
            tensor_features=tensor_features,
            images=images,
            trimmed_h=trimmed_h,
            trimmed_w=trimmed_w,
            streaming=False,
        )

        return tensor_features

    def _maybe_debug_geometry_layers(
        self,
        *,
        layer_indices: List[int],
        tensor_features: List[torch.Tensor],
        images: torch.Tensor,
        trimmed_h: int,
        trimmed_w: int,
        streaming: bool,
    ) -> None:
        from qwen_vl.debug import vln_debug

        if not vln_debug.is_enabled():
            return
        if vln_debug.should_save_geo_layers():
            vln_debug.save_geometry_encoder_layers(
                layer_indices=layer_indices,
                tensor_features=tensor_features,
                trimmed_h=trimmed_h,
                trimmed_w=trimmed_w,
                input_images=images,
                streaming=streaming,
            )
        if vln_debug.should_save_depth():
            vln_debug.save_vggt_depth_maps(self, images)

    def supports_streaming(self) -> bool:
        import inspect
        params = inspect.signature(self.vggt.aggregator.forward).parameters
        return "use_cache" in params

    def _format_streaming_layer_features(
        self,
        layer_output: torch.Tensor,
        *,
        h_patch: int,
        w_patch: int,
        spatial_merge_size: int,
        include_camera_token: bool,
        dtype: torch.dtype,
    ):
        frame_tokens = layer_output[0, -1:, :, :]
        patch_grid = frame_tokens.reshape(1, h_patch, w_patch, -1)
        trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
        trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
        patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
        patch_grid = patch_grid.reshape(
            1,
            trimmed_h // spatial_merge_size,
            spatial_merge_size,
            trimmed_w // spatial_merge_size,
            spatial_merge_size,
            -1,
        )
        patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
        patch_tokens = patch_grid.reshape(1, trimmed_h * trimmed_w, -1)

        if include_camera_token:
            camera_token = layer_output[0, -1:, 0:1, :]
            geo_feature = torch.cat([camera_token, patch_tokens], dim=1)
        else:
            geo_feature = patch_tokens

        return geo_feature.to(dtype).contiguous(), trimmed_h, trimmed_w

    def encode_layers_streaming(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
        frame_strict: bool = False,
    ):
        """Encode frames sequentially with VGGT KV cache (JanusVLN-style)."""
        if not self.supports_streaming():
            raise RuntimeError(
                "VGGT aggregator does not support streaming (missing KV cache). "
                "Re-install SpatialStack from the JanusVLN-VLN training branch."
            )

        if self._eval_streaming:
            return self._encode_layers_streaming_eval(
                images,
                layer_indices=layer_indices,
                spatial_merge_size=spatial_merge_size,
                include_camera_token=include_camera_token,
            )

        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        n_image, _, height, width = images.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        spatial_merge_size = spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2

        if layer_indices is None:
            layer_indices = [-2]

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        past_key_values = [None] * self.vggt.aggregator.depth
        aggregated_tokens_list = None
        patch_start_idx = 0
        # FUSION_FRAME_STRICT: keep every frame's geometry (each frame fused with its
        # own vision tokens) instead of only the last frame (broadcast to all frames).
        # per_frame_layers stays None for the non-strict path, which is unchanged.
        per_frame_layers = {idx: [] for idx in layer_indices} if frame_strict else None

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                for frame_idx, frame in enumerate(images):
                    frame_input = frame.unsqueeze(0).unsqueeze(0)
                    output = self.vggt.aggregator(
                        frame_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        past_frame_idx=frame_idx,
                    )
                    aggregated_tokens_list, patch_start_idx, past_key_values = output
                    if frame_strict:
                        for idx in layer_indices:
                            per_frame_layers[idx].append(aggregated_tokens_list[idx])

        tensor_features = []
        for idx in layer_indices:
            if frame_strict:
                # [n_image, n_patch, dim]: concatenate each frame's own current-frame
                # tokens (frame t already attended to <=t via the KV cache).
                frame_tokens = torch.cat(
                    [lo[0, -1:, patch_start_idx:, :] for lo in per_frame_layers[idx]], dim=0
                )
                camera_token = (
                    torch.cat([lo[0, -1:, 0:1, :] for lo in per_frame_layers[idx]], dim=0)
                    if include_camera_token
                    else None
                )
                batch = n_image
            else:
                layer_output = aggregated_tokens_list[idx]
                frame_tokens = layer_output[0, -1:, patch_start_idx:, :]  # [1, n_patch, dim]
                camera_token = layer_output[0, -1:, 0:1, :] if include_camera_token else None
                batch = 1
            # reference_frame flip is applied on the image sequence before the loop

            patch_grid = frame_tokens.reshape(batch, h_patch, w_patch, -1)
            trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
            trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
            patch_grid = patch_grid[:, :trimmed_h, :trimmed_w, :]
            patch_grid = patch_grid.reshape(
                batch,
                trimmed_h // spatial_merge_size,
                spatial_merge_size,
                trimmed_w // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            patch_grid = patch_grid.permute(0, 1, 3, 2, 4, 5)
            patch_tokens = patch_grid.reshape(batch, trimmed_h * trimmed_w, -1)

            if include_camera_token:
                geo_feature = torch.cat([camera_token, patch_tokens], dim=1)
            else:
                geo_feature = patch_tokens

            tensor_features.append(geo_feature.to(dtype).contiguous())

        from qwen_vl.debug import vln_debug

        if vln_debug.is_enabled() and tensor_features:
            vln_debug.log_geometry_streaming(
                n_image=n_image,
                h_patch=h_patch,
                w_patch=w_patch,
                spatial_merge_size=spatial_merge_size,
                patch_tokens_shape=tuple(tensor_features[0].shape),
            )

        trimmed_h = (h_patch // spatial_merge_size) * spatial_merge_size or h_patch
        trimmed_w = (w_patch // spatial_merge_size) * spatial_merge_size or w_patch
        self._maybe_debug_geometry_layers(
            layer_indices=layer_indices,
            tensor_features=tensor_features,
            images=images,
            trimmed_h=trimmed_h,
            trimmed_w=trimmed_w,
            streaming=True,
        )

        return tensor_features

    def _encode_layers_streaming_eval(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Habitat eval: encode only the current frame, keep VGGT KV across steps."""
        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        frame = images[-1]
        _, height, width = frame.shape
        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        spatial_merge_size = spatial_merge_size if spatial_merge_size and spatial_merge_size > 0 else 2

        if layer_indices is None:
            layer_indices = [-2]

        if self._streaming_past_key_values is None:
            self._streaming_past_key_values = [None] * self.vggt.aggregator.depth

        tokens_per_frame = self.vggt.aggregator.patch_start_idx + h_patch * w_patch
        anchor_token_count = None
        history_anchor_registered = False
        history_anchor_fifo = False
        history_anchor_reason = None
        if self.kv_cache_mode == "ovggt":
            from ..vggt.utils.history_anchor import HistoryAnchorManager

            if self._history_anchor_manager is None:
                self._history_anchor_manager = HistoryAnchorManager(
                    self._history_anchor_config, tokens_per_frame
                )
            elif self._history_anchor_manager.tokens_per_frame != tokens_per_frame:
                raise ValueError(
                    "VGGT frame token count changed inside one episode: "
                    f"{self._history_anchor_manager.tokens_per_frame} -> "
                    f"{tokens_per_frame}"
                )

            if self.vggt_history_anchor_strategy == "fixed_interval":
                (
                    history_anchor_registered,
                    history_anchor_fifo,
                    history_anchor_reason,
                ) = self._history_anchor_manager.should_become_anchor(
                    self._streaming_frame_idx
                )
                if history_anchor_registered:
                    self._history_anchor_manager.register_anchor(
                        self._streaming_frame_idx
                    )
            anchor_token_count = (
                self._history_anchor_manager.get_protected_token_count()
            )

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        frame_input = frame.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                if torch.cuda.is_available():
                    vggt_start = torch.cuda.Event(enable_timing=True)
                    vggt_end = torch.cuda.Event(enable_timing=True)
                    vggt_start.record()

                output = self.vggt.aggregator(
                    frame_input,
                    past_key_values=self._streaming_past_key_values,
                    use_cache=True,
                    past_frame_idx=self._streaming_frame_idx,
                    kv_cache_mode=self.kv_cache_mode,
                    total_budget=(
                        self.vggt_total_budget
                        if self.kv_cache_mode == "ovggt" else 0
                    ),
                    anchor_token_count=anchor_token_count,
                    importance_weight=self.vggt_importance_weight,
                    window_token_count=(
                        self.vggt_recent_protect_frames
                        * tokens_per_frame
                    ),
                    intra_frame_keep_ratio=self.vggt_intra_frame_keep_ratio,
                )
                aggregated_tokens_list, patch_start_idx, self._streaming_past_key_values = output
                if self.kv_cache_mode == "start_recent":
                    self._streaming_past_key_values = self._kv_cache_trim(
                        self._streaming_past_key_values
                    )
                elif history_anchor_registered:
                    self._streaming_past_key_values = (
                        self.vggt.aggregator.sync_anchor_change(
                            self._streaming_past_key_values,
                            anchor_token_count=anchor_token_count,
                            tokens_per_frame=tokens_per_frame,
                            anchor_keep_ratio=(
                                self.vggt_history_anchor_keep_ratio
                            ),
                            anchor_token_indices=None,
                            is_fifo=history_anchor_fifo,
                        )
                    )
                    if self.vggt_kv_debug:
                        fifo_text = " fifo" if history_anchor_fifo else ""
                        print(
                            f"[VGGT-HistoryAnchor] frame={self._streaming_frame_idx}"
                            f"{fifo_text} protected={anchor_token_count} "
                            f"reason={history_anchor_reason}",
                            flush=True,
                        )
                if self.vggt_kv_debug:
                    retained = [
                        int(kv[0].shape[2]) if kv is not None else 0
                        for kv in self._streaming_past_key_values
                    ]
                    budgets = getattr(self.vggt.aggregator, "last_budgets", None)
                    budget_text = (
                        budgets.detach().cpu().tolist() if budgets is not None else None
                    )
                    eviction_counts = getattr(
                        self.vggt.aggregator, "eviction_counts", None
                    )
                    eviction_text = (
                        eviction_counts.detach().cpu().tolist()
                        if eviction_counts is not None else None
                    )
                    print(
                        f"[VGGT-KV] frame={self._streaming_frame_idx} "
                        f"mode={self.kv_cache_mode} retained={retained} "
                        f"protected={anchor_token_count} budgets={budget_text} "
                        f"evictions={eviction_text}",
                        flush=True,
                    )
                self._streaming_frame_idx += 1

                if torch.cuda.is_available():
                    vggt_end.record()
                    torch.cuda.synchronize()
                    self.last_vggt_ms = vggt_start.elapsed_time(vggt_end)

        tensor_features = []
        trimmed_h = trimmed_w = 0
        for idx in layer_indices:
            layer_output = aggregated_tokens_list[idx][:, :, patch_start_idx:, :]
            geo_feature, trimmed_h, trimmed_w = self._format_streaming_layer_features(
                layer_output,
                h_patch=h_patch,
                w_patch=w_patch,
                spatial_merge_size=spatial_merge_size,
                include_camera_token=include_camera_token,
                dtype=dtype,
            )
            tensor_features.append(geo_feature)

        if self._eval_frame_strict and not self._eval_projected_cache:
            # Buffer this frame's per-layer features on CPU (it was encoded with the
            # growing KV), then return the requested window gathered PER-FRAME. Each
            # buffered frame i == trajectory frame i (one frame encoded per step).
            if self._frame_feature_buffer is None:
                self._frame_feature_buffer = []
            self._frame_feature_buffer.append([t.detach().to("cpu") for t in tensor_features])
            n_buf = len(self._frame_feature_buffer)
            window = self._eval_window_indices
            window = [i for i in window if 0 <= i < n_buf] if window else [n_buf - 1]
            if not window:
                window = [n_buf - 1]
            gathered = []
            for layer_pos in range(len(tensor_features)):
                frames = [self._frame_feature_buffer[i][layer_pos] for i in window]
                gathered.append(torch.cat(frames, dim=0).to(tensor_features[layer_pos].device))
            tensor_features = gathered

        from qwen_vl.debug import vln_debug

        if vln_debug.is_enabled() and tensor_features:
            vln_debug.log_geometry_streaming(
                n_image=1,
                h_patch=h_patch,
                w_patch=w_patch,
                spatial_merge_size=spatial_merge_size,
                patch_tokens_shape=tuple(tensor_features[0].shape),
            )

        self._maybe_debug_geometry_layers(
            layer_indices=layer_indices,
            tensor_features=tensor_features,
            images=frame.unsqueeze(0),
            trimmed_h=trimmed_h,
            trimmed_w=trimmed_w,
            streaming=True,
        )

        return tensor_features

    def encode_layers_with_mode(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
        streaming: bool = False,
        frame_strict: bool = False,
    ):
        if streaming:
            return self.encode_layers_streaming(
                images,
                layer_indices=layer_indices,
                spatial_merge_size=spatial_merge_size,
                include_camera_token=include_camera_token,
                frame_strict=frame_strict,
            )
        return self.encode_layers(
            images,
            layer_indices=layer_indices,
            spatial_merge_size=spatial_merge_size,
            include_camera_token=include_camera_token,
        )
    
    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048  # VGGT feature dimension
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images
    
    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    
    def load_model(self, model_path: str) -> None:
        """Load pretrained VGGT model."""
        from ..vggt.models.vggt import VGGT
        self._vggt_pretrained_path = model_path
        self.vggt = VGGT.from_pretrained(model_path, enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
                
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

    def _ensure_depth_head(self) -> None:
        if self._depth_head_ready:
            return
        if self.vggt.depth_head is not None:
            self._depth_head_ready = True
            return
        from ..vggt.models.vggt import VGGT

        path = self._vggt_pretrained_path or "facebook/VGGT-1B"
        tmp = VGGT.from_pretrained(
            path,
            enable_camera=False,
            enable_point=False,
            enable_depth=True,
            enable_track=False,
        )
        device = next(self.vggt.parameters()).device
        self.vggt.depth_head = tmp.depth_head.to(device)
        self.vggt.depth_head.eval()
        for param in self.vggt.depth_head.parameters():
            param.requires_grad = False
        self._depth_head_ready = True

    def predict_depth_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Debug helper: VGGT DPT depth for [S,3,H,W] in [0,1].
        Returns [S, H, W] (full-sequence aggregator, not streaming KV).
        """
        self._ensure_depth_head()
        self.vggt.eval()
        images = self._apply_reference_frame_transform(images)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                depth, _conf = self.vggt.depth_head(
                    aggregated_tokens_list,
                    images=images[None],
                    patch_start_idx=patch_start_idx,
                )
        return depth[0, :, 0]
