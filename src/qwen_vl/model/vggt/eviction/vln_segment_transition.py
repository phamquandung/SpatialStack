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
_MARKER_RE = re.compile(r"\b(?:and\s+then|after\s+that|then|next)\b", re.I)
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


def assign_tokens_to_segments(
    offset_mapping: torch.Tensor, segment_char_spans: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    offsets = torch.as_tensor(offset_mapping)
    if offsets.ndim == 3 and offsets.shape[0] == 1:
        offsets = offsets[0]
    if offsets.ndim != 2 or offsets.shape[-1] != 2:
        raise ValueError(f"offset_mapping must have shape [L,2], got {tuple(offsets.shape)}")
    result = torch.full((offsets.shape[0],), -1, dtype=torch.int32, device=offsets.device)
    spans = torch.tensor(segment_char_spans, dtype=torch.long, device=offsets.device)
    for token_idx, (start, end) in enumerate(offsets.long()):
        if end <= start:
            continue
        overlap = (torch.minimum(end, spans[:, 1]) - torch.maximum(start, spans[:, 0])).clamp_min(0)
        best = int(overlap.argmax())
        if overlap[best] > 0:
            result[token_idx] = best
        else:
            raise ValueError(f"normal instruction token at offset [{start}, {end}) has no segment")
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
    segment_ids = assign_tokens_to_segments(encoded["offset_mapping"][0], spans).to(device)
    token_embeddings = input_embedding_layer(input_ids)
    pooled = []
    for segment_id in range(len(spans)):
        mask = segment_ids == segment_id
        if not mask.any():
            raise ValueError(f"instruction segment {segment_id} contains no tokenizer tokens")
        pooled.append(F.normalize(token_embeddings[mask].float().mean(0), dim=0))
    segment_embeddings = torch.stack(pooled)
    normal = encoded["offset_mapping"][0].ne(0).any(dim=-1).to(device)
    if (segment_ids[normal] < 0).any():
        raise AssertionError("every normal instruction token must have a segment id")
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
    similarity = F.normalize(aligned_visual_tokens.float(), dim=-1) @ F.normalize(
        segment_embeddings.float(), dim=-1
    ).T
    best, best_id = similarity.max(dim=-1)
    return ((best.clamp(-1, 1) + 1) * 0.5), best_id.to(torch.int16)


@torch.no_grad()
def build_frame_descriptor(aligned_visual_tokens, confidence):
    weights = confidence.float().flatten().clamp(0, 1)
    if aligned_visual_tokens.shape[0] != weights.numel():
        raise ValueError("confidence count must equal aligned visual token count")
    descriptor = (aligned_visual_tokens.float() * weights[:, None]).sum(0)
    descriptor = descriptor / weights.sum().clamp_min(1e-6)
    return F.normalize(descriptor, dim=-1)


@torch.no_grad()
def compute_local_transition_score(current_descriptor, recent_descriptors):
    if not recent_descriptors:
        return current_descriptor.new_zeros((), dtype=torch.float32)
    recent = torch.stack([x.to(current_descriptor.device).float() for x in recent_descriptors])
    cosine = F.normalize(recent, dim=-1) @ F.normalize(current_descriptor.float(), dim=-1)
    return ((1 - cosine.clamp(-1, 1)) * 0.5).mean().clamp(0, 1)


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
        values[field.name] = None if lhs is None or rhs is None else torch.cat([lhs, rhs], dim=0)
    return VLNGhostTokenMetadata(**values)


def gather_metadata(metadata, indices):
    values = {
        field.name: (None if getattr(metadata, field.name) is None else getattr(metadata, field.name).index_select(0, indices))
        for field in fields(VLNGhostTokenMetadata)
    }
    return VLNGhostTokenMetadata(**values)
