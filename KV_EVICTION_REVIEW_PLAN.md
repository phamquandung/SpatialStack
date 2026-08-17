# VGGT KV Eviction: Memory–Navigation Trade-off Review & Plan

## Context

The repo runs streaming VGGT (24 cached global-attention layers) as a geometry memory for a
Qwen3.5-4B VLN policy on R2R `val_unseen`. The current eviction method
(`GHOST_SCORE_MODE=vln_segment_transition`) scores every cached token by
geometry + confidence + instruction-segment relevance + transition anchor, z-scores each
component over the candidate cache, and top-K's each layer to an offline-profiled budget.

The goal is minimum VGGT/CUDA memory at maximum retained SR/SPL — and, where possible,
an accuracy gain, since SR degrades badly on long episodes.

This plan is written from measurements already on disk plus code reading. **No code has been
changed.** Everything below is separated into Tier-1 (engineering, method untouched),
Tier-2 (method-level, changes the research contribution), and Tier-3 (research extensions).

---

## Part A — What the measurements actually say

### A1. Where CUDA memory goes

Measured per-episode from `evaluation_gate_scale_fix_eval/*/*/result.json` (fields written by
`src/evaluation.py:701-719`). Not estimated.

| Component | @ budget 900k | @ budget 360k |
|---|---:|---:|
| Peak CUDA allocated | 21 065 MB | 17 949 MB |
| VGGT KV cache | 5 273 MB (25%) | 2 109 MB (12%) |
| Camera-head KV | 16 MB avg / 50 MB @400 steps | 12 / 25 MB |
| Eviction metadata | 23.2 MB | 9.3 MB |
| **Everything else (fixed floor)** | **~15 806 MB** | **~15 819 MB** |

The ~15.8 GB floor is identical at both budgets: Qwen3.5-4B weights + VGGT-1B + heads +
per-step activations + `generate`. **A perfect eviction method addresses at most 25% of peak
CUDA today.** This bounds the whole exercise and should be stated in the paper.

### A2. The KV cost model, and a 33% accounting error

`Aggregator` (`src/qwen_vl/model/vggt/models/aggregator.py:107-126`): `depth=24`,
`embed_dim=1024`, `num_heads=16` → `head_dim=64`, `num_register_tokens=4` →
`patch_start_idx=5`. Only the 24 `global_blocks` cache; `frame_blocks` do not.

Expected bytes/token/layer for bf16 K+V: `2 × 16 × 64 × 2 = 4096`.
Measured: `2 109.299 MB / 359 987 tokens = 6144.00` and `5 273.350 MB / 899 985 = 6144.00`.

Cause: `qk_norm=True` makes `q_norm`/`k_norm` `nn.LayerNorm`
(`src/qwen_vl/model/vggt/layers/attention.py:55-56`), and autocast promotes LayerNorm to fp32.
So after `attention.py:1292` `k` is **fp32** while `v` stays **bf16**, and
`attention.py:1314` caches the fp32 K. `6144 = 1024×4 + 1024×2`. Confirmed by running
LayerNorm under `autocast(bfloat16)` on this machine.

`F.scaled_dot_product_attention` is on autocast's lower-precision list, so it downcasts K to
bf16 at use time regardless. Storing bf16 therefore produces **bit-identical attention output**
while removing exactly one third of the cache.

Derived `tokens_per_frame = 1205` (5 special + 30×40 patches, VGGT input 420×560 after Qwen
smart-resize of the 640×480 Habitat RGB) — fitted against the measured saturation curve and
consistent with it (saturation completes between step 50 and 75; `64607/1205 = 53.6`).
Retained history depth is therefore **25.7–53.6 frames at 900k** and **10.3–21.4 frames at 360k**.

### A3. The Pareto curve that already exists

Paired comparison on the **589 episodes both runs share**
(360k/`steps_200` vs 900k/`steps_400`, same checkpoint, same weights profile):

| | SR | SPL | NE | KV GB | Peak CUDA GB | step ms | vggt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| budget 360 000 | 48.90 | 44.79 | 5.84 | 2.06 | 17.5 | 643 | 65.0 |
| budget 900 000 | 50.42 | 45.46 | 5.99 | 5.11 | 20.6 | 671 | 81.0 |
| Δ | +1.53 | +0.66 | +0.16 (worse) | +2.5× | +3.1 GB | +28 | +16 |

Bootstrap 95% CI: ΔSR `[-1.53, +4.75]`, ΔSPL `[-2.06, +3.36]`. McNemar z = 0.96.
**Not significant.** NE moved the wrong way. The comparison is also biased *toward* 900k,
which ran with a 2× larger step cap; restricting to the 416 episodes where neither run hit its
cap gives the same answer (ΔSR +1.68, CI `[-1.44, +4.81]`, z = 1.02).

**Conclusion: 2.5× the KV memory currently buys no measurable navigation accuracy.**
The sweep must go *down*, not up. This is the single most important finding.

### A4. Accuracy decays with episode length

Full 1839-episode run, binned by episode length:

| steps | n | SR% | KV MB | camera KV MB |
|---:|---:|---:|---:|---:|
| 25–50 | 533 | 70.4 | 5 104 | 5.0 |
| 75–100 | 216 | 52.3 | 5 273 | 10.6 |
| 150–175 | 49 | 36.7 | 5 273 | 20.3 |
| 225–250 | 13 | 15.4 | 5 273 | 29.5 |
| 400 (cap) | 261 | 10.7 | 5 273 | 50.0 |

The 261 capped episodes are 14% of the set at 10.7% SR. Pulling them to even 30% is
**+2.7 SR overall** — a larger prize than anything else available.

The scorer has **no recency or decay term**; all four components are frozen at insertion
(`vggt_encoder.py:370-381`) and never recomputed. The only time-dependence is through the
per-component σ in `zscore_normalize`, and that σ is computed over the **survivor**
population — tokens already selected to maximise `Σ w_c z_c`. Selection truncates variance
along the dominant direction, shrinking its σ, which *raises* its effective weight `w_c/σ_c`
next step. That is positive feedback toward single-component ossification, and it predicts
exactly the decay pattern above. This is a hypothesis, not yet measured — see B0.

### A5. Other confirmed issues

- **Camera KV grows unbounded.** `_streaming_past_key_values_camera`
  (`vggt_encoder.py:1089`) is never pruned: 2.5 MB at <25 steps → 50 MB at 400.
  Only the current and previous pose are ever consumed (`vggt_encoder.py:461-463` trims
  `_streaming_frame_metadata` to `[-2:]`). It is the only component still growing with
  episode length.
- **Two DPT heads run every frame.** `_append_ghost_frame_metadata` (`vggt_encoder.py:1083`)
  runs `camera_head` + `depth_head` + `point_head`. `_pts3d` is discarded — only
  `pts3d_conf` is used. The full `depth` map is used only to compute **one scalar**
  (gradient-magnitude variance, `vggt_encoder.py:313-324`).
- **Special-token privilege is uncapped.** `vln_segment_transition.py:100` sets every special
  token above `patch_final.max()`, so all 5/frame survive forever. At 400 steps that is 2 000
  unevictable tokens/layer: 6.5% of the smallest layer at 900k, **16% at 360k, 32% at 180k**.
  This becomes the binding constraint below ~360k.
- **`geometry_score` has zero within-frame discriminative power.** It is a per-frame scalar
  broadcast to all 1205 tokens (`vggt_encoder.py:372`), yet carries the joint-largest weight
  (0.30). After z-scoring it acts as a whole-frame include/exclude bias — which is also the
  most likely ossification driver.
- **No valid reference baseline exists.** `StartRecentKVCache`
  (`vggt_encoder.py:29-56`) slices `dim=2`, the **token** dimension, with `start=8`/`recent=56`,
  while its log line and comments call these *frames*. That path retains 56 tokens, not 56
  frames. It only affects the non-GHOST baseline, but it means required experiments #1 and #2
  (large-cache reference, JanusVLN baseline) currently cannot be run correctly.
- **The score probe has never been run.** No `vln_score_probe/` output exists anywhere.
  Section 3 of the review ("does the score select the right tokens") **cannot be answered from
  existing artifacts**. No scoring change should be made before it is.
- `_frame_feature_buffer` (`vggt_encoder.py:957`) grows unbounded on **host** RAM
  (~one `[1205, 2048]` bf16 tensor per layer per step). Not CUDA, but a plausible cause of the
  SIGKILL on the `budget_900000_steps_300` run.

### A6. Two latent bugs found while validating the fixes

- **Frame-0 metadata aliasing.** `concat_metadata(None, new)` returns `new` *itself*
  (`vln_segment_transition.py:266-267`). On the first frame nobody is over budget, so no
  `gather_metadata` copy happens and **all 24 entries of `_vln_metadata_per_layer` are the same
  object**. Benign today because the next frame's `cat` allocates fresh tensors, but it is one
  in-place write away from corrupting every layer at once. Fix: `return replace(new)`.
- **~120 device syncs per frame.** `compute_candidate_final_score` does five boolean-mask
  selects (`values[patch_mask]` ×4 plus `final_score[patch_mask]`) per layer per frame. Boolean
  masking on CUDA calls `nonzero()`, which synchronises — 5 × 24 = 120 syncs/frame, despite the
  docstring at `vggt_encoder.py:198` claiming the scorer stays "GPU-only and
  synchronization-free". Replacing them with a single `nonzero` + `index_select` cuts this 5×.
  This is likely a measurable slice of the 16 ms/step gap between the 360k and 900k runs.
- **Landmine at `importance_eviction.py:208`**: `pc = meta.get("pts3d_conf") or meta.get("conf")`
  — `or` on a multi-element tensor raises. It only survives because the key is always present
  today; T1.3 must omit the key rather than store `None` (both readers already guard on `in`).

---

## Part B — Tier 1: engineering fixes (method unchanged, no ablation needed)

These change *bytes per token*, not *which tokens are kept*. They are orthogonal to the
research contribution and can be reported as implementation efficiency.

### T1.1 — Cache K in bf16 *(highest value/risk ratio in this document)*

**File:** `src/qwen_vl/model/vggt/layers/attention.py`, `Attention.forward`.
Insert `if k.dtype != v.dtype: k = k.to(v.dtype)` **after the RoPE block (after line 1296) and
before the cache concat at 1314**. Both placements are load-bearing:

- *After* RoPE, not after `k_norm`: `RotaryPositionEmbedding2D` caches cos/sin in
  `tokens.dtype` (`rope.py:100-117,178`), so casting earlier would run the rotation in bf16 and
  add a second rounding. Casting after RoPE applies exactly the one rounding SDPA applies
  internally — bit-identical on the fused path.
- *Before* the `torch.cat`, not at `new_kv = (k, v)` (line 1357): `torch.cat` **type-promotes**,
  so downcasting later means next frame's cat of bf16-past with fp32-new silently promotes the
  entire cache back to fp32 and the saving evaporates.

Do not gate on `use_cache` (no-op outside autocast, and `_run_flex_attention` already does the
same cast at 137-141). Do **not** also downcast `q` — it is never cached, so there is no
steady-state saving, and fp32 `q * self.scale` is marginally better on the non-fused path.

- Memory: **−33.3% of all VGGT KV.** 5 273 → 3 516 MB @900k; 2 109 → 1 406 MB @360k.
  Peak CUDA 21.1 → ~19.3 GB.
- Accuracy risk: **none.** SDPA under autocast already downcasts K to bf16 at use time, so the
  retained-cache path is bit-identical.
- Latency: neutral-to-better (33% less bandwidth on every concat, gather and matmul).
- **Mandatory companion edit, same commit.** `eviction()` casts importance scores to `k.dtype`
  and then runs `topk` on them (`attention.py:1180, 1228`, plus `1156, 1169, 1216, 1218, 1254`).
  With K in bf16 the *scores* would be quantized to ~8 mantissa bits → mass ties → the baseline
  GHOST ranking silently collapses toward `topk`'s arbitrary tiebreak. Pin all of these to
  `.float()`; only the multiply site (line 1397) should follow `k.dtype`. This path is dead in
  `vln_segment_transition` mode but live in the `importance` baseline.
- Non-fused paths (`fused_attn=False`, `return_attn_weights=True`) are safe: `torch.matmul` is
  on autocast's lower-precision policy and casts both operands to bf16 today as well.
- `_apply_delta_share` (`attention.py:964`) and `_append_motivation_kv_probe` (`:108-115`) do
  precision-sensitive arithmetic on cached K; add `.float()`. Both are currently unreachable
  (`kv_share_cfg` is `None`) / diagnostic-only.
- The camera-head trunk builds blocks with default `qk_norm=False` (`camera_head.py:56`), so its
  K is **already bf16**. The 6144 B/token figure applies only to the 24 aggregator global layers.

### T1.2 — Bound the camera-head KV

**File:** `vggt_encoder.py:1084-1092` / `_append_ghost_frame_metadata`.
`trunk_fn` appends one token per refinement *iteration* per trunk layer
(`camera_head.py:138,163-178`), i.e. 4 iters × 4 layers = 16 tokens/frame at 8192 B/token =
0.125 MB/frame — which closes exactly against the measured 50 MB at 400 steps. Token counts are
frame-aligned, so a start+recent slice is clean. Trim *after* the head call, keeping frame 0
(it defines VGGT's reference camera) plus a short recent window.

- Memory: 50 MB → ~1 MB **constant**; removes the last unbounded CUDA component.
- Risk: low. Only `camera_pose` is consumed, and only through
  `sigmoid(_pose_change_score(prev, cur).mean())` into the per-frame `geometry` scalar. Verify by
  streaming 40 synthetic frames trimmed vs untrimmed and comparing that scalar series.
- While there: `_reset_vggt_attention_cache_state` (`vggt_encoder.py:485-488`) only walks
  `aggregator.global_blocks`, so camera-trunk block state leaks across episodes. Harmless today.

### T1.3 — Make `point_head` optional in the per-step path

**Files:** `vggt_encoder.py:1096-1105` (skip `point_head`),
`vggt_encoder.py:272-283` (`pool_conf`/`torch.minimum` already handles a `None` branch).
Gate behind an env flag so it is A/B-testable.

- Saves a full DPT forward per step: latency (part of the 80 ms VGGT time) and its transient
  activation peak.
- **Changes `confidence` from `min(depth, point)` to `depth` only → needs an ablation.**
  Belongs in Tier-1 only as an *option*; promoting it to default requires E4 below.

### T1.4 — Slim the eviction metadata

**Files:** `vln_segment_transition.py:46-55` (`VLNGhostTokenMetadata`), plus
`concat_metadata`/`gather_metadata` and the probe.
`geometry_score` is a per-frame constant (store per frame, index by `frame_id`);
`final_score` is fully recomputed each step by `compute_candidate_final_score`;
`best_segment_id` is read only by the probe. 27 → ~13 bytes/token.

27 → ~17 bytes/token. Keep `geometry_score` as a *declared but unpopulated* field with an
override parameter on `compute_candidate_final_score`, so the existing test keeps passing
verbatim; store the per-frame values as a frame-indexed table on the encoder (4 bytes per frame
total, shared by all 24 layers) and gather them with `frame_id`.

- Memory: ~−11 MB @900k. **Low priority as a memory play** — 0.05% of peak CUDA. The real
  payoff is per-step work: 8 `cat`s + 8 `index_select`s × 24 layers × 400 steps, plus the 120
  syncs/frame from A6. Bundle it with the single-`nonzero` rewrite and the `replace(new)`
  aliasing fix, since they touch the same two functions.
- `tests/test_vln_segment_transition.py` must stay green **unchanged** — that is the acceptance
  criterion for this item.

### T1.5 — Do *not* prioritise `Attention.eviction` micro-optimisation

`attention.py:1128-1212` has real waste (Python index loops per layer per step, a dead
`patch_k` gather at :1177, top-k over `[B,16,N]` when all 16 heads carry identical scores).
But in `vln_segment_transition` mode `total_budget=0` and this path **never executes**
(`vggt_encoder.py:900-919`). Fix only if the `importance` baseline is re-run.

---

## Part C — Tier 2: method-level changes (change the contribution, require ablation)

Ordered by expected value. **Test one at a time** — never bundle.

### T2.1 — Aggressive downward budget sweep *(no code change, highest expected value)*

A3 shows 360k ≈ 900k. Sweep 180k / 120k / 90k / 45k. Combined with T1.1:
180k → 703 MB KV, i.e. **7.5× below today's 5 273 MB**. This is the headline result if it holds.
Prerequisite: T2.2, which becomes binding below ~360k.

### T2.2 — Cap the special-token privilege

**File:** `vln_segment_transition.py:100`. Currently every special token from every frame
outranks every patch forever. Options: keep specials only for the last N frames; or reserve a
fixed fraction of each layer budget; or let older specials compete normally.
Required before any budget below ~360k is meaningful — at 180k they would otherwise occupy
32% of the smallest layer.

### T2.3 — Replace z-score with rank/percentile normalization

**File:** `vln_segment_transition.py:58-67` + `compute_candidate_final_score:70-101`.
Rank normalization is scale-free, outlier-immune, and — critically — **breaks the survivor-
variance feedback loop in A4**, because a component's influence no longer depends on the
spread of the surviving population. It also preserves the property
`tests/test_vln_segment_transition.py:52-61` asserts (invariance to positive affine rescaling).
Note it *changes* the property asserted at `:75-142` (weighted variance share = `w_i²/Σw_j²`),
so that test encodes the current design and must be revised deliberately, not "fixed".

### T2.4 — Add a recency / age term or decay

`frame_id` is already stored, so age is free to compute (`vln_score_probe.py:182` already does).
The `importance` mode has `w_temporal`; the VLN mode dropped it entirely. Adding a small decay
directly targets the A4 long-episode collapse. This is the main **accuracy-upside** candidate.

### T2.5 — Redundancy-aware retention, reusing what already exists

`build_frame_descriptor` and `RecentTransitionState` already produce L2-normalised per-frame
descriptors. A frame-level redundancy check over stored descriptors is nearly free and answers
"are several retained tokens storing the same information?" without going back to
query-key similarity. Only pursue if B0 shows cache concentration.

### T2.6 — Give `geometry` within-frame resolution, or reduce its weight

A per-frame scalar carrying weight 0.30 is a whole-frame gate. Either make it per-token
(reuse the existing per-patch depth-gradient magnitude rather than its scalar variance) or
rebalance the weights. Decide from B0, not a priori.

---

## Part D — Tier 3: research extensions (higher risk)

- **Adaptive per-step budgets** `B_min ≤ B_t ≤ B_max` driven by score entropy or transition
  score. Note the objective is *peak* CUDA — an adaptive scheme that lowers average but keeps
  `B_max` unchanged does not meet the practical goal. Must report both.
- **Revisit the layer-budget table.** `configs/kv_budget_proportions_cosine.json` was derived
  from `1 − cos_sim` softmaxed at T=0.5 (verified: the arithmetic in the file is exact). The
  generating script is absent from the repo, so it is not reproducible. Layer 14 gets 2.1× the
  flat layers. Test uniform budgets as a control — if uniform matches, the table is not
  earning its complexity.
- **INT8/FP8 KV quantization.** Only *after* T1.1, which already takes 33% for free. Further
  quantization means dequant on a multi-GB cache every layer every step and real accuracy risk.

---

## Part E — Experiment matrix

Protocol: same checkpoint (`spatialstack_janus_vln_train-gate-scale-4B-loss-3`), same config,
**same fixed episode subset**, paired per-episode stats (McNemar + bootstrap CI), and an
identical `MAX_STEPS` across every arm — the existing 360k-vs-900k comparison is confounded by
a 200-vs-400 cap and that must not recur. The 589-episode 4-scene subset already covered by the
360k run is the natural anchor (~3 h/arm vs ~16.7 h for full `val_unseen`).

Report for every arm: KV GB, peak CUDA GB, SR, SPL, NE, mean step ms, mean VGGT ms, plus ΔSR/ΔSPL
vs reference with CIs.

| # | Arm | Purpose |
|---|---|---|
| E0 | 900k, probe on (`VLN_SCORE_PROBE=1`) | **Run first.** Score distributions, variance share over time, retention age, unique frames, per-segment counts. Answers the "is the score selecting the right tokens" question that currently has no data. |
| E1 | 900k + T1.1 | Confirm bit-identical SR/SPL at −33% KV. Sanity gate for all later arms. |
| E2 | Budget sweep with T1.1: 360k / 180k / 120k / 90k / 45k | The Pareto frontier. Identify the failure budget. |
| E3 | Failure budget + T2.2 (special-token cap) | Does capping specials move the frontier down? |
| E4 | T1.3 (depth-conf only) at the chosen budget | Is `min(depth, point)` worth a DPT forward per step? |
| E5 | Chosen budget + T2.3 (rank norm), then + T2.4 (recency), separately | Accuracy upside; measure specifically on the 400-step-cap subset. |
| E6 | Uniform layer budget, same total | Control for the GHOST table. |
| E7 | Full `val_unseen` on the single best config | Final headline number. |

**Predicted answer to "smallest realistic budget":** ~180k tokens with T1.1 + T2.2, i.e.
**~703 MB VGGT KV (7.5× below today) and peak CUDA ~16.5 GB**, at SR/SPL within noise of the
current 900k result. Below ~90k the smallest layer holds under 3 frames of history and I expect
a genuine break. E2 tests this directly.

---

## Part F — Explicitly not worth pursuing

- **Metadata as a memory play** (T1.4 beyond cleanup): 23 MB of a 21 GB peak.
- **`Attention.eviction` micro-optimisation** (T1.5): dead code in the active mode.
- **Quantization below bf16** before T1.1 and E2 land: the free 33% and the 2.5× budget cut
  dominate it, and it carries dequant latency plus real accuracy risk.
- **Raising the budget above 900k**: A3 shows the curve is already flat there.
- **Chasing reconstruction metrics**: nothing here should be optimised for depth/point quality
  unless E2/E5 show that degradation is what actually moves SR/SPL.

---

## Verification

- `python -m unittest discover tests` must stay green after any Tier-1 change
  (`tests/test_vln_segment_transition.py` pins the KV/metadata index-sync invariant and the
  special-token ordering).
- For T1.1, verify numerically rather than by argument: capture K before/after the cast for one
  frame and assert the SDPA output is bit-identical.
- Memory claims verify from the existing per-episode instrumentation
  (`peak_vggt_kv_mb`, `peak_alloc_mb`, `peak_vggt_camera_kv_mb`, `peak_vggt_metadata_mb`, all
  already written by `src/evaluation.py:701-719`) — no new profiling harness is needed.
  `scripts/evaluation/summarize_vln_results.py` should be extended to surface
  `peak_vggt_metadata_mb` and `peak_vggt_camera_kv_mb`, which it currently drops.
- Before any Tier-2 comparison against a "full cache" or JanusVLN baseline, resolve the
  frames-vs-tokens unit question in `StartRecentKVCache` (`vggt_encoder.py:29-56`) — otherwise
  that baseline is retaining 56 tokens and every comparison against it is meaningless.
