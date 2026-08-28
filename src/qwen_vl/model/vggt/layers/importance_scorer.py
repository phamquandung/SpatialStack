"""OVGGT-compatible token importance scoring for VGGT KV eviction."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class SpatialImportanceScorer:
    def __init__(self, kernel_size: int = 3, alpha: float = 0.3):
        self.kernel_size = kernel_size
        self.alpha = alpha
        self._kernel: Optional[Tensor] = None

    def _get_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if (
            self._kernel is None
            or self._kernel.device != device
            or self._kernel.dtype != dtype
        ):
            kernel = torch.tensor(
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                device=device,
                dtype=dtype,
            ) / 16.0
            self._kernel = kernel.view(1, 1, 3, 3)
        return self._kernel

    def compute(
        self,
        importance: Tensor,
        patch_start_idx: int = 5,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        batch, _ = importance.shape
        special = importance[:, :patch_start_idx]
        patches = importance[:, patch_start_idx:]

        if grid_size is None:
            height = width = int(patches.shape[1] ** 0.5)
            if height * width != patches.shape[1]:
                return importance
        else:
            height, width = grid_size
        if height * width != patches.shape[1]:
            return importance

        patch_grid = patches.reshape(batch, 1, height, width)
        smoothed = F.conv2d(
            patch_grid,
            self._get_kernel(importance.device, importance.dtype),
            padding=1,
        )
        combined = self.alpha * smoothed + (1.0 - self.alpha) * patch_grid
        return torch.cat([special, combined.squeeze(1).reshape(batch, -1)], dim=-1)


class TokenImportanceScorer:
    VALID_STRATEGIES = ("baseline", "repr_shift", "repr_shift_spatial")

    def __init__(self, strategy: str = "baseline", spatial_alpha: float = 0.3):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. Must be one of {self.VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.spatial_scorer = (
            SpatialImportanceScorer(alpha=spatial_alpha)
            if strategy == "repr_shift_spatial"
            else None
        )

    def compute(
        self,
        k: Optional[Tensor] = None,
        x_before_mlp: Optional[Tensor] = None,
        mlp_output: Optional[Tensor] = None,
        patch_start_idx: int = 5,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if self.strategy == "baseline":
            if k is None:
                raise ValueError("Key tensor 'k' is required for baseline strategy")
            k_norm = F.normalize(k, p=2, dim=-1)
            mean_k = k_norm.mean(dim=2, keepdim=True)
            similarity = (k_norm * mean_k).sum(dim=-1)
            return 1.0 - similarity.mean(dim=1)

        if x_before_mlp is None or mlp_output is None:
            raise ValueError(
                "Both x_before_mlp and mlp_output are required for repr_shift strategy"
            )
        importance = mlp_output.norm(dim=-1)
        if self.strategy == "repr_shift_spatial":
            return self.spatial_scorer.compute(
                importance, patch_start_idx=patch_start_idx, grid_size=grid_size
            )
        return importance
