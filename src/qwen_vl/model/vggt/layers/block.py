import logging
import os
from typing import Callable, List, Any, Tuple, Dict, Union, Optional
import warnings

import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp
from .importance_scorer import TokenImportanceScorer


XFORMERS_AVAILABLE = False


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        eviction_strategy: str = "repr_shift_spatial",
        spatial_alpha: float = 0.5,
        patch_start_idx: int = 5,
        patch_grid_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.eviction_strategy = eviction_strategy
        self.patch_start_idx = patch_start_idx
        self.patch_grid_size = patch_grid_size
        self.importance_scorer = TokenImportanceScorer(
            strategy=eviction_strategy, spatial_alpha=spatial_alpha
        )

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def _forward_ovggt_cache(
        self,
        x,
        pos,
        past_key_values,
        cache_budget,
        prev_importance,
        intra_frame_keep_ratio,
        anchor_token_count,
        importance_weight,
        window_token_count,
    ):
        use_two_stage = intra_frame_keep_ratio < 1.0
        normed = self.norm1(x)

        if use_two_stage:
            attn_output, kv_info, scores = self.attn(
                normed,
                pos=pos,
                past_key_values=past_key_values,
                use_cache=True,
                kv_cache_mode="ovggt",
                cache_budget=cache_budget,
                importance_scores=prev_importance,
                defer_eviction=True,
                anchor_token_count=anchor_token_count,
                importance_weight=importance_weight,
                window_token_count=window_token_count,
            )
            _, _, k_current, v_current, past_kv = kv_info
        else:
            attn_output, new_kv_full, scores = self.attn(
                normed,
                pos=pos,
                past_key_values=past_key_values,
                use_cache=True,
                kv_cache_mode="ovggt",
                cache_budget=cache_budget,
                importance_scores=prev_importance,
                defer_eviction=False,
                anchor_token_count=anchor_token_count,
                importance_weight=importance_weight,
                window_token_count=window_token_count,
            )

        x_after_attn = x + self.ls1(attn_output)
        x_before_mlp = x_after_attn
        mlp_residual = self.ls2(self.mlp(self.norm2(x_before_mlp)))
        x_after_mlp = x_before_mlp + mlp_residual
        if self.eviction_strategy in ("repr_shift", "repr_shift_spatial"):
            new_importance = self.importance_scorer.compute(
                x_before_mlp=x_before_mlp,
                mlp_output=mlp_residual,
                patch_start_idx=self.patch_start_idx,
                grid_size=self.patch_grid_size,
            )
        else:
            current_k = (
                k_current
                if use_two_stage
                else new_kv_full[0][:, :, -x.shape[1] :, :]
            )
            new_importance = self.importance_scorer.compute(k=current_k)

        kept_indices = None
        if use_two_stage:
            num_new = x.shape[1]
            intra_keep = max(int(num_new * intra_frame_keep_ratio), 1)
            if intra_keep < num_new:
                k_current, v_current, intra_indices = self.attn.intra_frame_prune(
                    k_current, v_current, new_importance, intra_keep
                )
                importance_for_eviction = torch.gather(
                    new_importance, 1, intra_indices
                )
            else:
                importance_for_eviction = new_importance
            if past_kv is not None:
                k = torch.cat([past_kv[0], k_current], dim=2)
                v = torch.cat([past_kv[1], v_current], dim=2)
            else:
                k, v = k_current, v_current
            if cache_budget is not None and k.shape[2] > cache_budget:
                effective_anchor = (
                    anchor_token_count
                    if anchor_token_count is not None
                    else self.attn.num_anchor_tokens
                )
                k, v, scores, kept_indices = self.attn.eviction(
                    k,
                    v,
                    cache_budget,
                    effective_anchor,
                    importance_scores=importance_for_eviction,
                    num_new_tokens=k_current.shape[2],
                    importance_weight=importance_weight,
                    window_token_count=window_token_count,
                )
            new_kv = (k, v)
        else:
            k, v, kept_indices = new_kv_full
            new_kv = (k, v)

        return x_after_mlp, new_kv, scores, new_importance, kept_indices

    def forward(
        self,
        x: Tensor,
        pos=None,
        attn_mask=None,
        past_key_values=None,
        use_cache=False,
        kv_cache_mode="start_recent",
        cache_budget=None,
        prev_importance=None,
        intra_frame_keep_ratio=1.0,
        anchor_token_count=None,
        importance_weight=0.5,
        window_token_count=0,
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        if use_cache and kv_cache_mode == "ovggt":
            return self._forward_ovggt_cache(
                x,
                pos,
                past_key_values,
                cache_budget,
                prev_importance,
                intra_frame_keep_ratio,
                anchor_token_count,
                importance_weight,
                window_token_count,
            )
        if kv_cache_mode not in ("start_recent", "ovggt"):
            raise ValueError(f"Unsupported VGGT KV cache mode: {kv_cache_mode!r}")
            
        def attn_residual_func(x: Tensor, pos=None, attn_mask=None, past_key_values=None, use_cache=False) -> Union[Tensor, Tuple[Tensor, Dict]]:
            if use_cache:
                output, new_kv = self.attn(self.norm1(x), pos=pos, past_key_values=past_key_values, use_cache=True)
                return self.ls1(output), new_kv
            else:
                if attn_mask is not None:
                    return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
                else:
                    return self.ls1(self.attn(self.norm1(x), pos=pos))
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        if use_cache:
            attn_output, new_kv = attn_residual_func(x, pos=pos, past_key_values=past_key_values, use_cache=True)
            x = x + attn_output
            x = x + ffn_residual_func(x)
            return x, new_kv

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
