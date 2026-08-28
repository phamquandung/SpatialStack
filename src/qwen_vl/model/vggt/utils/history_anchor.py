"""OVGGT-compatible history-anchor selection for streaming VGGT KV caches."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HistoryAnchorConfig:
    """Configuration names and defaults used by OVGGT's anchor manager."""

    strategy: str = "none"
    interval: int = 50
    min_anchor_interval: Optional[int] = None
    max_anchors: int = 3
    coverage_threshold: float = 0.4
    sample_ratio: float = 0.1
    anchor_keep_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy not in ("none", "fixed_interval", "coverage"):
            raise ValueError(
                "History-anchor strategy must be 'none', 'fixed_interval', or "
                f"'coverage', got {self.strategy!r}"
            )
        if self.interval <= 0:
            raise ValueError("History-anchor interval must be positive")
        if self.min_anchor_interval is not None and self.min_anchor_interval < 0:
            raise ValueError("History-anchor minimum interval must be non-negative")
        if self.max_anchors < 0:
            raise ValueError("History-anchor maximum must be non-negative")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("History-anchor coverage threshold must be in [0, 1]")
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError("History-anchor sample ratio must be in (0, 1]")
        if not 0.0 < self.anchor_keep_ratio <= 1.0:
            raise ValueError("History-anchor keep ratio must be in (0, 1]")


class HistoryAnchorManager:
    """Count-based anchor protection with OVGGT's fixed-interval FIFO policy.

    The coverage configuration is represented here for configuration parity, but
    selecting coverage anchors still requires OVGGT depth and camera-pose outputs.
    SpatialStack's VGGT encoder deliberately disables those prediction heads.
    """

    def __init__(self, config: HistoryAnchorConfig, tokens_per_frame: int):
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")
        self.config = config
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_history_anchors = 0
        self.history_anchor_frames: List[int] = []
        self.next_anchor_frame = config.interval
        self.last_anchor_frame: Optional[int] = None

    def is_eviction_paused(self) -> bool:
        return False

    def should_become_anchor(self, frame_idx: int):
        if self.config.strategy != "fixed_interval":
            return False, False, "disabled"
        if frame_idx != self.next_anchor_frame:
            return False, False, f"not_target_frame_{frame_idx}"

        self.next_anchor_frame += self.config.interval
        is_fifo = self.num_history_anchors >= self.config.max_anchors
        return True, is_fifo, f"interval_anchor_at_frame_{frame_idx}"

    def register_anchor(self, frame_idx: int) -> None:
        if self.config.max_anchors == 0:
            return
        if self.num_history_anchors < self.config.max_anchors:
            self.num_history_anchors += 1
        if len(self.history_anchor_frames) >= self.config.max_anchors:
            self.history_anchor_frames.pop(0)
        self.history_anchor_frames.append(int(frame_idx))
        self.last_anchor_frame = int(frame_idx)

    def get_protected_token_count(self) -> int:
        history_tokens = int(
            self.num_history_anchors
            * self.tokens_per_frame
            * self.config.anchor_keep_ratio
        )
        return self.tokens_per_frame + history_tokens

    def get_num_anchors(self) -> int:
        return 1 + self.num_history_anchors

    def __repr__(self) -> str:
        base = (
            "HistoryAnchorManager("
            f"strategy={self.config.strategy}, "
            f"num_anchors={self.get_num_anchors()}, "
            f"history_frames={self.history_anchor_frames}"
        )
        if self.config.strategy == "fixed_interval":
            return base + f", next_target={self.next_anchor_frame})"
        if self.config.strategy == "coverage":
            return base + f", threshold={self.config.coverage_threshold})"
        return base + ")"
