import copy
from types import SimpleNamespace

import torch

from qwen_vl.model.geometry_encoders.vggt_encoder import (
    StartRecentKVCache,
    VGGTEncoder,
)
from qwen_vl.model.vggt.layers.attention import Attention
from qwen_vl.model.vggt.layers.rope import RotaryPositionEmbedding2D
from qwen_vl.model.vggt.models.aggregator import Aggregator
from qwen_vl.model.vggt.utils.history_anchor import (
    HistoryAnchorConfig,
    HistoryAnchorManager,
)


def test_initial_dynamic_budgets_match_ovggt_baseline():
    state = SimpleNamespace(last_scores=torch.zeros(24))
    budgets = Aggregator._calculate_dynamic_budgets(state, 200_000)
    assert budgets.tolist() == [8_333] * 24
    assert int(budgets.sum()) == 199_992


def test_dynamic_budgets_change_with_previous_scores():
    state = SimpleNamespace(last_scores=torch.linspace(0.0, 1.0, 24))
    budgets = Aggregator._calculate_dynamic_budgets(state, 200_000)
    assert budgets.shape == (24,)
    assert budgets[0] > budgets[-1]
    assert int(budgets.sum()) <= 200_000


def test_eviction_preserves_anchor_and_does_not_restore_removed_tokens():
    attention = Attention(dim=8, num_heads=2, rope=None)
    token_ids = torch.arange(10, dtype=torch.float32)
    k = token_ids.view(1, 1, 10, 1).expand(1, 2, 10, 4).clone()
    v = k.clone()

    pruned_k, pruned_v, _, _ = attention.eviction(
        k, v, cache_budget=6, num_anchor_tokens=2
    )
    assert pruned_k.shape[2] == 6
    assert torch.equal(pruned_k[0, 0, :2, 0], torch.tensor([0.0, 1.0]))
    retained_after_low = set(pruned_k[0, 0, :, 0].tolist())

    new_k = torch.full((1, 2, 1, 4), 10.0)
    grown_k = torch.cat([pruned_k, new_k], dim=2)
    grown_v = torch.cat([pruned_v, new_k], dim=2)
    grown_k, _, _, _ = attention.eviction(
        grown_k, grown_v, cache_budget=8, num_anchor_tokens=2
    )
    assert grown_k.shape[2] == 7
    assert set(grown_k[0, 0, :, 0].tolist()) == retained_after_low | {10.0}


def test_start_recent_backend_keeps_original_5d_layout():
    cache = []
    for _ in range(2):
        k = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1)
        cache.append([k, k.clone()])
    trimmed = StartRecentKVCache(start_size=2, recent_size=3)(cache)
    assert trimmed[0][0].shape == (1, 1, 5, 1, 1)
    assert trimmed[0][0].flatten().tolist() == [0.0, 1.0, 5.0, 6.0, 7.0]


def test_ovggt_rotated_cache_matches_legacy_when_no_eviction():
    torch.manual_seed(7)
    rope = RotaryPositionEmbedding2D(frequency=100)
    legacy = Attention(dim=16, num_heads=4, qk_norm=True, rope=rope).eval()
    ovggt = copy.deepcopy(legacy).eval()
    x0 = torch.randn(1, 6, 16)
    x1 = torch.randn(1, 6, 16)
    pos = torch.tensor(
        [[[0, 0], [0, 0], [1, 1], [1, 2], [2, 1], [2, 2]]],
        dtype=torch.long,
    )

    out_legacy_0, legacy_kv = legacy(
        x0, pos=pos, use_cache=True, kv_cache_mode="start_recent"
    )
    out_ovggt_0, ovggt_kv_full, _ = ovggt(
        x0,
        pos=pos,
        use_cache=True,
        kv_cache_mode="ovggt",
        cache_budget=100,
        anchor_token_count=6,
    )
    ovggt_kv = ovggt_kv_full[:2]
    assert torch.allclose(out_legacy_0, out_ovggt_0, atol=1e-6, rtol=1e-5)

    out_legacy_1, _ = legacy(
        x1,
        pos=pos,
        past_key_values=legacy_kv,
        use_cache=True,
        kv_cache_mode="start_recent",
    )
    out_ovggt_1, _, _ = ovggt(
        x1,
        pos=pos,
        past_key_values=ovggt_kv,
        use_cache=True,
        kv_cache_mode="ovggt",
        cache_budget=100,
        anchor_token_count=6,
    )
    assert torch.allclose(out_legacy_1, out_ovggt_1, atol=1e-6, rtol=1e-5)


def test_small_aggregator_has_bounded_independent_layer_caches():
    model = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        patch_embed="conv",
    ).eval()
    cache = [None, None]
    for frame_idx in range(4):
        _, _, cache = model(
            torch.rand(1, 1, 3, 28, 28),
            past_key_values=cache,
            use_cache=True,
            past_frame_idx=frame_idx,
            kv_cache_mode="ovggt",
            total_budget=24,
            anchor_token_count=9,
        )
        for layer_idx, (k, v) in enumerate(cache):
            assert k.shape == v.shape
            assert k.ndim == 4
            assert k.shape[2] <= int(model.last_budgets[layer_idx])
            assert k.shape[2] >= 9

    model.reset_ovggt_cache_state()
    assert torch.equal(model.last_scores, torch.zeros(2))
    assert torch.equal(model.eviction_counts, torch.zeros(2, dtype=torch.long))
    assert model.last_budgets is None
    assert all(block.attn.num_anchor_tokens == 0 for block in model.global_blocks)


def test_fixed_interval_history_anchor_schedule_and_fifo():
    manager = HistoryAnchorManager(
        HistoryAnchorConfig(
            strategy="fixed_interval",
            interval=2,
            max_anchors=2,
            anchor_keep_ratio=0.5,
        ),
        tokens_per_frame=10,
    )
    assert manager.get_protected_token_count() == 10

    registrations = []
    for frame_idx in range(7):
        should_register, is_fifo, _ = manager.should_become_anchor(frame_idx)
        if should_register:
            manager.register_anchor(frame_idx)
            registrations.append((frame_idx, is_fifo))

    assert registrations == [(2, False), (4, False), (6, True)]
    assert manager.history_anchor_frames == [4, 6]
    assert manager.num_history_anchors == 2
    assert manager.get_num_anchors() == 3
    assert manager.get_protected_token_count() == 20


def test_history_anchor_sync_promotes_and_fifo_demotes_ovggt_prefix():
    state = SimpleNamespace(depth=1)

    def make_cache(start, end):
        ids = torch.arange(start, end, dtype=torch.float32)
        k = ids.view(1, 1, -1, 1)
        return [(k, k.clone())]

    cache = make_cache(0, 40)
    cache = Aggregator.sync_anchor_change(
        state,
        cache,
        anchor_token_count=15,
        tokens_per_frame=10,
        anchor_keep_ratio=0.5,
    )
    assert cache[0][0][0, 0, :15, 0].tolist() == (
        list(range(10)) + list(range(30, 35))
    )

    next_frame = make_cache(40, 50)[0]
    cache[0] = (
        torch.cat([cache[0][0], next_frame[0]], dim=2),
        torch.cat([cache[0][1], next_frame[1]], dim=2),
    )
    cache = Aggregator.sync_anchor_change(
        state,
        cache,
        anchor_token_count=20,
        tokens_per_frame=10,
        anchor_keep_ratio=0.5,
    )
    assert cache[0][0][0, 0, :20, 0].tolist() == (
        list(range(10)) + list(range(30, 35)) + list(range(40, 45))
    )

    fifo_frame = make_cache(50, 60)[0]
    cache[0] = (
        torch.cat([cache[0][0], fifo_frame[0]], dim=2),
        torch.cat([cache[0][1], fifo_frame[1]], dim=2),
    )
    cache = Aggregator.sync_anchor_change(
        state,
        cache,
        anchor_token_count=20,
        tokens_per_frame=10,
        anchor_keep_ratio=0.5,
        is_fifo=True,
    )
    protected = cache[0][0][0, 0, :20, 0].tolist()
    assert protected == list(range(10)) + list(range(40, 45)) + list(range(50, 55))
    assert cache[0][0][0, 0, -5:, 0].tolist() == list(range(30, 35))
    assert torch.equal(cache[0][0], cache[0][1])


def test_episode_reset_discards_history_anchor_manager():
    class AggregatorStub:
        def __init__(self):
            self.was_reset = False

        def reset_ovggt_cache_state(self):
            self.was_reset = True

    encoder = VGGTEncoder.__new__(VGGTEncoder)
    torch.nn.Module.__init__(encoder)
    aggregator = AggregatorStub()
    encoder.vggt = SimpleNamespace(aggregator=aggregator)
    encoder._streaming_past_key_values = [(torch.ones(1), torch.ones(1))]
    encoder._streaming_frame_idx = 17
    encoder._history_anchor_manager = object()
    encoder.last_vggt_ms = 1.0
    encoder._eval_frame_strict = False
    encoder._eval_projected_cache = False
    encoder._frame_feature_buffer = [object()]
    encoder._eval_window_indices = [1, 2]

    encoder.reset_streaming_cache()

    assert encoder._streaming_past_key_values is None
    assert encoder._streaming_frame_idx == 0
    assert encoder._history_anchor_manager is None
    assert encoder._eval_window_indices is None
    assert aggregator.was_reset
