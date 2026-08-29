"""Training-free instruction-segment and local-transition GHOST scoring."""

from collections import deque
from dataclasses import dataclass, fields
import re
from typing import Deque, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


NAVIGATION_VERBS = {
    "walk", "go", "move", "turn", "enter", "exit", "leave", "pass",
    "continue", "stop", "head", "proceed", "cross", "follow", "approach",
    "face", "take", "veer", "step",
}
# "next" is a sequencing marker only when it is not the spatial preposition "next to":
# "stop next to the plant" is one action, not two.
_MARKER_RE = re.compile(r"\b(?:and\s+then|after\s+that|then|next\b(?!\s+to\b))\b", re.I)
_ACTION_AND_RE = re.compile(
    r"\band\b(?=\s+(?:" + "|".join(sorted(NAVIGATION_VERBS)) + r")\b)", re.I
)


@dataclass
class InstructionSegmentState:
    raw_instruction: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_segment_ids: torch.Tensor
    segment_char_spans: torch.Tensor
    segment_embeddings: torch.Tensor


@dataclass
class RecentTransitionState:
    max_frames: int = 4

    def __post_init__(self):
        self.descriptors: Deque[torch.Tensor] = deque(maxlen=self.max_frames)

    def reset(self) -> None:
        self.descriptors.clear()


@dataclass
class VLNGhostTokenMetadata:
    frame_id: torch.Tensor
    geometry_score: torch.Tensor
    confidence_score: torch.Tensor
    instruction_score: torch.Tensor
    transition_score: torch.Tensor
    final_score: torch.Tensor
    is_special: torch.Tensor
    best_segment_id: Optional[torch.Tensor] = None


@torch.no_grad()
def zscore_normalize(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize scores without amplifying a constant or numerically dead term."""
    values = scores.float()
    if values.numel() == 0:
        raise ValueError("cannot z-score an empty score tensor")
    centered = values - values.mean()
    scale = centered.square().mean().sqrt()
    normalized = centered / scale.clamp_min(eps)
    return normalized * (scale > eps).to(normalized.dtype)


@torch.no_grad()
def sigmoid_normalize(scores: torch.Tensor) -> torch.Tensor:
    """GHOST's normalization: per-component sigmoid, no centering or scaling."""
    return torch.sigmoid(scores.float())


@torch.no_grad()
def compute_candidate_final_score(
    metadata: VLNGhostTokenMetadata,
    score_weights,
    special_token_boost: float,
    normalization: str = "zscore",
) -> Tuple[torch.Tensor, dict]:
    """Combine cache-wide standardized components for the eviction decision.

    Normalization happens over exactly the patch-token population considered by
    ``topk``. This removes the realised-scale mismatch between confidence, language
    relevance, geometry, and transition while preserving every component's ordering
    across both frames and tokens. Special tokens are scored above the best patch and
    therefore retain their explicit GHOST privilege.
    """
    patch_mask = ~metadata.is_special.bool()
    components = {
        "geometry": metadata.geometry_score,
        "confidence": metadata.confidence_score,
        "instruction": metadata.instruction_score,
        "transition": metadata.transition_score,
    }
    if "recency" in score_weights:
        # Absolute recency. `transition` freezes a frame's novelty at insertion time and
        # never revisits it, so nothing else in the score knows how old a token is. Any
        # affine reparameterisation ((t+1)/N, age, ...) is identical after z-scoring, so
        # the raw frame index is used and the term carries no hyperparameter. Added only
        # when declared, so configs without it neither pay for the extra standardization
        # nor gain a component in the diagnostics.
        components["recency"] = metadata.frame_id
    patch_final = torch.zeros_like(metadata.final_score[patch_mask], dtype=torch.float32)
    norm_fn = zscore_normalize if normalization == "zscore" else sigmoid_normalize
    normalized_components = {
        name: norm_fn(values[patch_mask]) for name, values in components.items()
    }
    for name, values in normalized_components.items():
        patch_final.add_(values, alpha=float(score_weights[name]))
    if normalization == "sigmoid":
        patch_final = patch_final / (patch_final.max() + 1e-8)

    final = torch.empty_like(metadata.final_score, dtype=torch.float32)
    final[patch_mask] = patch_final
    final[~patch_mask] = patch_final.max() + float(special_token_boost)
    return final, normalized_components


def split_instruction_with_spans(instruction: str) -> List[Tuple[int, int]]:
    """Return ordered action/clause spans while preserving every non-space character."""
    if not instruction or not instruction.strip():
        raise ValueError("VLN segment-transition scoring requires a non-empty instruction")
    starts = {0}
    # Punctuation remains on the preceding segment; the next lexical character starts one.
    for match in re.finditer(r"[.,;:]", instruction):
        nxt = match.end()
        while nxt < len(instruction) and instruction[nxt].isspace():
            nxt += 1
        if nxt < len(instruction):
            starts.add(nxt)
    for pattern in (_MARKER_RE, _ACTION_AND_RE):
        starts.update(match.start() for match in pattern.finditer(instruction))
    ordered = sorted(starts)
    spans: List[Tuple[int, int]] = []
    for idx, start in enumerate(ordered):
        end = ordered[idx + 1] if idx + 1 < len(ordered) else len(instruction)
        # Inter-segment whitespace is harmless, but include it so spans cover the source.
        if start < end and instruction[start:end].strip():
            spans.append((start, end))
    return spans


def token_segment_overlap(
    offset_mapping: torch.Tensor, segment_char_spans: Sequence[Tuple[int, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``([L, S]`` character overlap, ``[L]`` "is a real text token" mask).

    Tokenizer specials carry the offset ``(0, 0)``; they overlap nothing and are excluded
    from every segment.
    """
    offsets = torch.as_tensor(offset_mapping)
    if offsets.ndim == 3 and offsets.shape[0] == 1:
        offsets = offsets[0]
    if offsets.ndim != 2 or offsets.shape[-1] != 2:
        raise ValueError(f"offset_mapping must have shape [L,2], got {tuple(offsets.shape)}")
    offsets = offsets.long()
    spans = torch.tensor(segment_char_spans, dtype=torch.long, device=offsets.device)
    starts, ends = offsets[:, :1], offsets[:, 1:]
    is_text = (ends > starts).squeeze(-1)
    overlap = (
        torch.minimum(ends, spans[:, 1]) - torch.maximum(starts, spans[:, 0])
    ).clamp_min(0) * is_text[:, None]
    return overlap, is_text


def assign_tokens_to_segments(
    offset_mapping: torch.Tensor, segment_char_spans: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    overlap, is_text = token_segment_overlap(offset_mapping, segment_char_spans)
    best_overlap, best = overlap.max(dim=-1)
    result = torch.full(overlap.shape[:1], -1, dtype=torch.int32, device=overlap.device)
    assigned = best_overlap > 0
    result[assigned] = best[assigned].to(torch.int32)
    # Whitespace-only spans are dropped by split_instruction_with_spans, so a tokenizer
    # that emits e.g. a leading "  " as its own token overlaps nothing. Fall back to the
    # nearest segment instead of aborting the episode.
    orphan = is_text & ~assigned
    if orphan.any():
        spans = torch.tensor(
            segment_char_spans, dtype=torch.long, device=overlap.device
        )
        ends = torch.as_tensor(offset_mapping).reshape(-1, 2).long()[:, 1:]
        nearest = (spans[:, 0][None, :] - ends).abs().argmin(dim=-1)
        result[orphan] = nearest[orphan].to(torch.int32)
    return result


@torch.no_grad()
def build_instruction_segment_state(instruction, tokenizer, input_embedding_layer, device):
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("vln_segment_transition requires a fast tokenizer with offset mappings")
    encoded = tokenizer(
        instruction, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True
    )
    if "offset_mapping" not in encoded:
        raise RuntimeError("tokenizer did not return offset_mapping")
    input_ids = encoded["input_ids"][0].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0].to(device)
    spans = split_instruction_with_spans(instruction)
    overlap, is_text = token_segment_overlap(encoded["offset_mapping"][0], spans)
    segment_ids = assign_tokens_to_segments(encoded["offset_mapping"][0], spans).to(device)
    token_embeddings = input_embedding_layer(input_ids)
    overlap = overlap.to(device)
    pooled = []
    for segment_id in range(len(spans)):
        mask = segment_ids == segment_id
        if not mask.any():
            # A token straddling a boundary is assigned to whichever segment it overlaps
            # most, which can starve a short neighbour. Pool that segment from every
            # token touching it rather than aborting the episode.
            mask = overlap[:, segment_id] > 0
        if not mask.any():
            raise ValueError(f"instruction segment {segment_id} contains no tokenizer tokens")
        pooled.append(F.normalize(token_embeddings[mask].float().mean(0), dim=0))
    segment_embeddings = torch.stack(pooled)
    if (segment_ids[is_text.to(device)] < 0).any():
        raise AssertionError("every real instruction token must have a segment id")
    return InstructionSegmentState(
        raw_instruction=instruction,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_segment_ids=segment_ids,
        segment_char_spans=torch.tensor(spans, dtype=torch.int32, device=device),
        segment_embeddings=segment_embeddings,
    )


@torch.no_grad()
def compute_instruction_segment_relevance(aligned_visual_tokens, segment_embeddings):
    if aligned_visual_tokens.ndim != 2 or segment_embeddings.ndim != 2:
        raise ValueError("visual tokens and segment embeddings must both be rank-2")
    if aligned_visual_tokens.shape[-1] != segment_embeddings.shape[-1]:
        raise ValueError(
            f"language-space dimension mismatch: visual={aligned_visual_tokens.shape[-1]}, "
            f"text={segment_embeddings.shape[-1]}"
        )
    # segment_embeddings are already L2-normalised by build_instruction_segment_state and
    # are frozen for the episode, so only the visual side needs normalising each frame.
    similarity = F.normalize(aligned_visual_tokens.float(), dim=-1) @ segment_embeddings.T
    best, best_id = similarity.max(dim=-1)
    return ((best.clamp(-1, 1) + 1) * 0.5), best_id.to(torch.int16)


@torch.no_grad()
def build_frame_descriptor(aligned_visual_tokens, confidence, pooling="confidence"):
    """Frame-level summary vector; `transition` is the cosine distance between these.

    `pooling="uniform"` drops the confidence weighting. That weighting mechanically ties
    the descriptor to confidence — a low-confidence frame reweights the pool, shifting the
    descriptor and inflating its distance from the previous one. Measured across frames,
    corr(transition, confidence) = -0.84, and this is the suspected cause: it makes the
    transition term fight the confidence term inside the score instead of complementing it.
    """
    if pooling == "uniform":
        return F.normalize(aligned_visual_tokens.float().mean(0), dim=-1)
    weights = confidence.float().flatten().clamp(0, 1)
    if aligned_visual_tokens.shape[0] != weights.numel():
        raise ValueError("confidence count must equal aligned visual token count")
    descriptor = (aligned_visual_tokens.float() * weights[:, None]).sum(0)
    descriptor = descriptor / weights.sum().clamp_min(1e-6)
    return F.normalize(descriptor, dim=-1)


@torch.no_grad()
def compute_local_transition_score(current_descriptor, recent_descriptors, reduce="mean"):
    """Novelty of the current frame against previously seen ones.

    `reduce="max"` scores against the single most similar frame in the window rather than
    the average of it. Averaging over a short window answers "did I just turn?" — four
    consecutive navigation frames are near-identical, which is why the realised range of
    this term is only 0-0.057. Taking the closest match answers "have I been somewhere
    like this before?", so revisiting a room correctly scores low novelty, which the mean
    over a 4-frame window cannot detect.
    """
    if not recent_descriptors:
        return current_descriptor.new_zeros((), dtype=torch.float32)
    # Both sides come out of build_frame_descriptor already L2-normalised in fp32, so the
    # matvec is the cosine directly.
    recent = torch.stack([x.to(current_descriptor.device) for x in recent_descriptors]).float()
    cosine = (recent @ current_descriptor.float()).clamp(-1, 1)
    closeness = cosine.max() if reduce == "max" else cosine.mean()
    return ((1 - closeness) * 0.5).clamp(0, 1)


@torch.no_grad()
def compute_transition_anchor(
    transition_score,
    confidence,
    instruction_relevance,
    confidence_weight: float = 0.5,
    instruction_weight: float = 0.5,
):
    gate = (
        float(confidence_weight) * confidence.float()
        + float(instruction_weight) * instruction_relevance.float()
    )
    return (transition_score.float() * gate).clamp(0, 1)


def concat_metadata(old, new):
    if old is None:
        return new
    values = {}
    for field in fields(VLNGhostTokenMetadata):
        lhs, rhs = getattr(old, field.name), getattr(new, field.name)
        if (lhs is None) != (rhs is None):
            # Silently returning None here would drop the field from the cache forever
            # and desynchronise it from the K/V it describes.
            raise ValueError(
                f"metadata field {field.name!r} is present on one side only "
                f"(old={lhs is None}, new={rhs is None})"
            )
        values[field.name] = None if lhs is None else torch.cat([lhs, rhs], dim=0)
    return VLNGhostTokenMetadata(**values)


def gather_metadata(metadata, indices):
    values = {
        field.name: (None if getattr(metadata, field.name) is None else getattr(metadata, field.name).index_select(0, indices))
        for field in fields(VLNGhostTokenMetadata)
    }
    return VLNGhostTokenMetadata(**values)
