"""VGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional, List

from .base import BaseGeometryEncoder, GeometryEncoderConfig


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

    def encode_layers_streaming(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
    ):
        """Encode frames sequentially with VGGT KV cache (JanusVLN-style)."""
        if not self.supports_streaming():
            raise RuntimeError(
                "VGGT aggregator does not support streaming (missing KV cache). "
                "Re-install SpatialStack from the JanusVLN-VLN training branch."
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

        tensor_features = []
        for idx in layer_indices:
            layer_output = aggregated_tokens_list[idx]
            frame_tokens = layer_output[0, -1:, patch_start_idx:, :]
            # reference_frame flip is applied on the image sequence before the loop

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

    def encode_layers_with_mode(
        self,
        images: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        spatial_merge_size: int = 1,
        include_camera_token: bool = False,
        streaming: bool = False,
    ):
        if streaming:
            return self.encode_layers_streaming(
                images,
                layer_indices=layer_indices,
                spatial_merge_size=spatial_merge_size,
                include_camera_token=include_camera_token,
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
