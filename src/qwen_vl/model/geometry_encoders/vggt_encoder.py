"""VGGT geometry encoder implementation."""

import json
import os
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
        self._ghost_heads_ready = False
        self._eval_streaming = False
        self._streaming_past_key_values = None
        self._streaming_past_key_values_camera = None
        self._streaming_importance_cache = None
        self._streaming_frame_metadata = None
        self._streaming_frame_idx = 0
        self.last_vggt_ms = 0.0
        # Incremental frame-strict eval: buffer each frame's geometry (computed with the
        # growing KV) and return the requested window per-frame, instead of broadcasting.
        self._eval_frame_strict = False
        self._eval_window_indices = None
        self._frame_feature_buffer = None
        # Eval-time VGGT KV-cache window (in frames). Defaults match JanusVLN (8+48=56).
        # Override via env to test the long-horizon geometry-drift hypothesis, e.g.
        # VGGT_KV_START=1 VGGT_KV_RECENT=8 caps the cache at 9 frames (training horizon).
        _ghost_env = os.environ.get("USE_GHOST_KV_CACHE")
        self.use_ghost_kv_cache = (
            _ghost_env.lower() in ("1", "true", "yes")
            if _ghost_env is not None
            else bool(config.use_ghost_kv_cache)
        )
        self.vggt_total_budget = int(config.vggt_total_budget)
        self.vggt_importance_weights_path = config.vggt_importance_weights_path
        self.vggt_budget_proportions_path = config.vggt_budget_proportions_path
        self.vggt_importance_weights = None
        self._vggt_configs_loaded = False
        print(f"[VGGTEncoder] use_ghost_kv_cache={self.use_ghost_kv_cache}")
        _kv_start = int(os.environ.get("VGGT_KV_START", "8"))
        _kv_recent = int(os.environ.get("VGGT_KV_RECENT", "48"))
        print(f"[VGGTEncoder] eval KV-cache window: start={_kv_start} recent={_kv_recent} (total={_kv_start + _kv_recent} frames)")
        if self.use_ghost_kv_cache:
            print(f"[VGGTEncoder] GHOST KV-cache enabled: total_budget={self.vggt_total_budget}")
        self._kv_cache_trim = StartRecentKVCache(start_size=_kv_start, recent_size=_kv_recent, k_seq_dim=2, v_seq_dim=2)

    def set_eval_streaming(self, enabled: bool) -> None:
        self._eval_streaming = bool(enabled)

    def set_eval_frame_strict(self, enabled: bool) -> None:
        """Incremental frame-strict eval: buffer each frame's geometry (encoded with the
        growing KV) and return the requested window per-frame instead of broadcasting."""
        self._eval_frame_strict = bool(enabled)
        if enabled and self._frame_feature_buffer is None:
            self._frame_feature_buffer = []

    def set_eval_window_indices(self, indices) -> None:
        """Trajectory frame indices to gather from the per-frame buffer this step."""
        self._eval_window_indices = list(indices) if indices is not None else None

    def reset_streaming_cache(self) -> None:
        self._streaming_past_key_values = None
        self._streaming_past_key_values_camera = None
        self._streaming_importance_cache = None
        self._streaming_frame_metadata = None
        self._streaming_frame_idx = 0
        self.last_vggt_ms = 0.0
        self._frame_feature_buffer = [] if self._eval_frame_strict else None
        self._eval_window_indices = None
        self._reset_vggt_attention_cache_state()

    def _reset_vggt_attention_cache_state(self) -> None:
        for block in self.vggt.aggregator.global_blocks:
            if hasattr(block.attn, "_reset_cache_state"):
                block.attn._reset_cache_state()

    @staticmethod
    def _resolve_config_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path) and os.path.isfile(path):
            return path
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        )
        for candidate in (path, os.path.join(os.getcwd(), path), os.path.join(repo_root, path)):
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _load_vggt_configs(self) -> None:
        if self._vggt_configs_loaded:
            return
        weights_path = self._resolve_config_path(self.vggt_importance_weights_path)
        if weights_path is not None:
            with open(weights_path) as f:
                self.vggt_importance_weights = json.load(f)
        proportions_path = self._resolve_config_path(self.vggt_budget_proportions_path)
        if proportions_path is not None:
            with open(proportions_path) as f:
                cfg = json.load(f)
            self.vggt.aggregator.budget_proportions = torch.tensor(
                cfg["proportions"], dtype=torch.float32
            )
        self._vggt_configs_loaded = True
        
    
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
            self._streaming_past_key_values_camera = None
            self._streaming_importance_cache = {}
            self._streaming_frame_metadata = []
            self._reset_vggt_attention_cache_state()
        elif self._streaming_importance_cache is None:
            self._streaming_importance_cache = {}
        if self._streaming_frame_metadata is None:
            self._streaming_frame_metadata = []

        if self.use_ghost_kv_cache:
            self._load_vggt_configs()
            self._ensure_ghost_heads()
            if self._streaming_past_key_values_camera is None:
                self._streaming_past_key_values_camera = [None] * self.vggt.camera_head.trunk_depth

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
                    total_budget=self.vggt_total_budget if self.use_ghost_kv_cache else 0,
                    eviction_mode="importance" if self.use_ghost_kv_cache else "",
                    frame_metadata=self._streaming_frame_metadata if self.use_ghost_kv_cache else None,
                    importance_cache=self._streaming_importance_cache if self.use_ghost_kv_cache else None,
                    importance_weights=self.vggt_importance_weights if self.use_ghost_kv_cache else None,
                )
                aggregated_tokens_list, patch_start_idx, self._streaming_past_key_values = output
                if self.use_ghost_kv_cache:
                    self._append_ghost_frame_metadata(
                        aggregated_tokens_list,
                        frame_input,
                        patch_start_idx,
                    )
                else:
                    self._streaming_past_key_values = self._kv_cache_trim(self._streaming_past_key_values)
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

        if self._eval_frame_strict:
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
        self._depth_head_ready = False
        self._ghost_heads_ready = False
                
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

    def _ensure_ghost_heads(self) -> None:
        if self._ghost_heads_ready:
            return
        if (
            self.vggt.camera_head is not None
            and self.vggt.depth_head is not None
            and self.vggt.point_head is not None
        ):
            self._ghost_heads_ready = True
            self._depth_head_ready = True
            return

        from ..vggt.models.vggt import VGGT

        path = self._vggt_pretrained_path or "facebook/VGGT-1B"
        tmp = VGGT.from_pretrained(
            path,
            enable_camera=True,
            enable_point=True,
            enable_depth=True,
            enable_track=False,
        )
        device = next(self.vggt.parameters()).device
        self.vggt.camera_head = tmp.camera_head.to(device)
        self.vggt.depth_head = tmp.depth_head.to(device)
        self.vggt.point_head = tmp.point_head.to(device)
        for head in (self.vggt.camera_head, self.vggt.depth_head, self.vggt.point_head):
            head.eval()
            for param in head.parameters():
                param.requires_grad = False
        del tmp
        self._ghost_heads_ready = True
        self._depth_head_ready = True

    def _append_ghost_frame_metadata(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> None:
        pose_enc, self._streaming_past_key_values_camera = self.vggt.camera_head(
            aggregated_tokens_list,
            past_key_values_camera=self._streaming_past_key_values_camera,
            use_cache=True,
        )
        camera_pose = pose_enc[-1][:, 0, :]

        depth, depth_conf = self.vggt.depth_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        _pts3d, pts3d_conf = self.vggt.point_head(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
        self._streaming_frame_metadata.append(
            {
                "camera_pose": camera_pose.detach().cpu(),
                "depth": depth[:, 0].detach().cpu(),
                "depth_conf": depth_conf[:, 0].detach().cpu(),
                "conf": pts3d_conf[:, 0].detach().cpu(),
            }
        )

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
