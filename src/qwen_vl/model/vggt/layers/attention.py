import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict, Optional

from einops import rearrange

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.num_anchor_tokens = 0

    def _reset_cache_state(self):
        self.num_anchor_tokens = 0

    def intra_frame_prune(self, k, v, importance, keep_count):
        """OVGGT stage-1 pruning: reduce only the current frame before storage."""
        batch, heads, num_tokens, head_dim = k.shape
        if keep_count >= num_tokens:
            return k, v, None
        keep_count = max(int(keep_count), 1)
        indices = torch.topk(importance, k=keep_count, dim=-1).indices
        indices = indices.sort(dim=-1).values
        expanded = indices[:, None, :, None].expand(
            batch, heads, keep_count, head_dim
        )
        return (
            torch.gather(k, 2, expanded),
            torch.gather(v, 2, expanded),
            indices,
        )

    def eviction(
        self,
        k,
        v,
        cache_budget,
        num_anchor_tokens,
        importance_scores=None,
        num_new_tokens=0,
        importance_weight=0.5,
        window_token_count=0,
    ):
        """Faithful OVGGT hybrid eviction on flattened per-layer token KV."""
        batch, heads, num_tokens, head_dim = k.shape
        cache_budget = int(cache_budget)
        num_anchor_tokens = int(num_anchor_tokens)
        if num_tokens <= cache_budget:
            return k, v, None, None
        if cache_budget <= num_anchor_tokens:
            anchor_indices = torch.arange(
                num_anchor_tokens, device=k.device
            )[None].expand(batch, -1)
            return (
                k[:, :, :num_anchor_tokens],
                v[:, :, :num_anchor_tokens],
                None,
                anchor_indices,
            )

        window_token_count = max(int(window_token_count or 0), 0)
        max_tail = max(num_tokens - num_anchor_tokens, 0)
        tail_count = min(
            window_token_count,
            max(cache_budget - num_anchor_tokens, 0),
            max_tail,
        )
        tail_start = num_tokens - tail_count if tail_count > 0 else num_tokens

        anchor_k = k[:, :, :num_anchor_tokens]
        anchor_v = v[:, :, :num_anchor_tokens]
        tail_k = k[:, :, tail_start:] if tail_count > 0 else None
        tail_v = v[:, :, tail_start:] if tail_count > 0 else None
        candidate_k = k[:, :, num_anchor_tokens:tail_start]
        candidate_v = v[:, :, num_anchor_tokens:tail_start]

        num_candidates = candidate_k.shape[2]
        num_to_keep = min(
            max(cache_budget - num_anchor_tokens - tail_count, 0), num_candidates
        )
        num_new_candidates = max(int(num_new_tokens) - tail_count, 0)
        num_old_candidates = max(num_candidates - num_new_candidates, 0)
        if num_to_keep >= num_candidates:
            return k, v, None, None

        anchor_indices = torch.arange(num_anchor_tokens, device=k.device)[None].expand(
            batch, -1
        )
        if num_to_keep <= 0:
            if tail_count > 0:
                tail_indices = torch.arange(tail_start, num_tokens, device=k.device)[
                    None
                ].expand(batch, -1)
                return (
                    torch.cat([anchor_k, tail_k], dim=2),
                    torch.cat([anchor_v, tail_v], dim=2),
                    None,
                    torch.cat([anchor_indices, tail_indices], dim=-1),
                )
            return anchor_k, anchor_v, None, anchor_indices

        candidate_norm = F.normalize(candidate_k, p=2, dim=-1)
        mean_vector = candidate_norm.mean(dim=2, keepdim=True)
        baseline_similarity = (candidate_norm * mean_vector).sum(dim=-1)
        baseline_diversity = 1.0 - baseline_similarity
        baseline_diversity_avg = baseline_diversity.mean(dim=1)

        if importance_scores is not None and num_new_candidates > 0:
            importance_scores = importance_scores[:, :num_new_candidates]
        use_hybrid = (
            importance_scores is not None
            and importance_scores.shape[1] == num_new_candidates
            and num_new_candidates > 0
            and num_old_candidates > 0
        )

        if use_hybrid:
            old_scores = baseline_diversity_avg[:, :num_old_candidates]
            old_min = old_scores.min(dim=-1, keepdim=True).values
            old_max = old_scores.max(dim=-1, keepdim=True).values
            old_normalized = (old_scores - old_min) / (old_max - old_min + 1e-8)
            new_min = importance_scores.min(dim=-1, keepdim=True).values
            new_max = importance_scores.max(dim=-1, keepdim=True).values
            new_normalized = (importance_scores - new_min) / (
                new_max - new_min + 1e-8
            )
            combined = torch.cat(
                [
                    (1.0 - importance_weight) * old_normalized,
                    importance_weight * new_normalized,
                ],
                dim=-1,
            )
            avg_scores = combined.mean().item()
            top_indices = torch.topk(combined, k=num_to_keep, dim=-1).indices
            top_indices = top_indices.sort(dim=-1).values
            expanded = top_indices[:, None, :, None].expand(
                batch, heads, num_to_keep, head_dim
            )
            kept_candidate_indices = top_indices + num_anchor_tokens
        else:
            avg_scores = baseline_similarity.mean().item()
            top_indices = torch.topk(
                -baseline_similarity, k=num_to_keep, dim=-1
            ).indices
            top_indices = top_indices.sort(dim=-1).values
            expanded = top_indices[..., None].expand(
                batch, heads, num_to_keep, head_dim
            )
            kept_candidate_indices = top_indices[:, 0] + num_anchor_tokens

        kept_k = torch.gather(candidate_k, 2, expanded)
        kept_v = torch.gather(candidate_v, 2, expanded)
        if tail_count > 0:
            tail_indices = torch.arange(tail_start, num_tokens, device=k.device)[
                None
            ].expand(batch, -1)
            return (
                torch.cat([anchor_k, kept_k, tail_k], dim=2),
                torch.cat([anchor_v, kept_v, tail_v], dim=2),
                avg_scores,
                torch.cat(
                    [anchor_indices, kept_candidate_indices, tail_indices], dim=-1
                ),
            )
        return (
            torch.cat([anchor_k, kept_k], dim=2),
            torch.cat([anchor_v, kept_v], dim=2),
            avg_scores,
            torch.cat([anchor_indices, kept_candidate_indices], dim=-1),
        )

    def _forward_ovggt(
        self,
        x,
        pos,
        attn_mask,
        past_key_values,
        cache_budget,
        importance_scores,
        defer_eviction,
        anchor_token_count,
        importance_weight,
        window_token_count,
    ):
        """OVGGT cache path; cached keys are normalized and RoPE-rotated."""
        batch, num_queries, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, num_queries, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.num_anchor_tokens == 0:
            self.num_anchor_tokens = k.shape[2]
        k_current, v_current = k.clone(), v.clone()
        past_kv_for_block = None
        if past_key_values is not None:
            past_k, past_v = past_key_values
            past_kv_for_block = (past_k, past_v)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = None
        kept_indices = None
        if not defer_eviction:
            if cache_budget is not None and k.shape[2] > cache_budget:
                effective_anchor = (
                    anchor_token_count
                    if anchor_token_count is not None
                    else self.num_anchor_tokens
                )
                k, v, scores, kept_indices = self.eviction(
                    k,
                    v,
                    cache_budget,
                    effective_anchor,
                    importance_scores=importance_scores,
                    num_new_tokens=num_queries,
                    importance_weight=importance_weight,
                    window_token_count=window_token_count,
                )
            new_kv = (k, v, kept_indices)
        else:
            new_kv = (k, v, k_current, v_current, past_kv_for_block)

        if self.fused_attn:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attention = (q * self.scale) @ k.transpose(-2, -1)
            if attn_mask is not None:
                attention = attention + attn_mask
            attention = self.attn_drop(attention.softmax(dim=-1))
            output = attention @ v
        output = output.transpose(1, 2).reshape(batch, num_queries, channels)
        output = self.proj_drop(self.proj(output))
        return output, new_kv, scores

    def forward(self, 
        x: torch.Tensor, 
        pos=None, 
        attn_mask=None, 
        past_key_values=None, 
        use_cache=False,
        kv_cache_mode="start_recent",
        cache_budget=None,
        importance_scores=None,
        defer_eviction=False,
        anchor_token_count=None,
        importance_weight=0.5,
        window_token_count=0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        if use_cache and kv_cache_mode == "ovggt":
            return self._forward_ovggt(
                x,
                pos,
                attn_mask,
                past_key_values,
                cache_budget,
                importance_scores,
                defer_eviction,
                anchor_token_count,
                importance_weight,
                window_token_count,
            )
        if kv_cache_mode not in ("start_recent", "ovggt"):
            raise ValueError(f"Unsupported VGGT KV cache mode: {kv_cache_mode!r}")
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        pos_k = pos
        if use_cache:
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)
            if past_key_values is not None:
                past_k, past_v = past_key_values
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
                
            new_kv = (k, v)
            a, b, c, d, e = k.shape
            k = k.reshape(a, b, c*d, e)
            v = v.reshape(a, b, c*d, e)
            if pos_k is not None:
                #print(pos_k.shape)
                pos_k = pos_k.repeat(1, c, 1)
                #print(pos_k.shape)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos_k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )

        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            # Mask
            if attn_mask is not None:
                assert attn_mask.shape[-2:] == (N, N), f"Expected mask shape [..., {N}, {N}], got {attn_mask.shape}"
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if use_cache:
            return x, new_kv
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
