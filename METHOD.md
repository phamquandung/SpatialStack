# Token Eviction for Streaming VGGT in a VLN Policy

**What this document is.** A precise description of the token-eviction method currently
implemented in this repository, and of the GHOST method it is built on. Every claim below is
anchored to a file and function, or to a cited paper. Where the code deviates from the design it
appears to intend, that is stated explicitly rather than smoothed over.

---

## 1. TL;DR

The repository contains **two selectable eviction modes**, chosen by the environment variable
`GHOST_SCORE_MODE` and read in `VGGTEncoder.__init__` (`src/qwen_vl/model/geometry_encoders/vggt_encoder.py:102-111`):

| | `importance` | `vln_segment_transition` |
|---|---|---|
| What it is | A faithful reimplementation of **GHOST** (arXiv:2605.15852) | **This project's method** |
| Optimises for | Reconstruction quality | Navigation success (SR / SPL / NE) |
| Importance signal | Camera pose change, depth-gradient variance, temporal recency, patch saliency, depth/point confidence | Geometry, confidence, **instruction-segment relevance**, **transition anchor** |
| Normalisation | Per-component sigmoid, then divide by max | **Cache-wide z-score per component** |
| Where eviction runs | Inside `Attention.forward`, per layer, during the forward pass | In **one post-projector pass** over all 24 layers, after the language projector |
| Layer budgets | Cosine-profiled, non-uniform (GHOST) | Same table, unchanged |
| Special-token privilege | GHOST Eq. 7 (soft boost) | GHOST Eq. 7 at insertion, plus a **hard** floor above every patch at eviction |

The short version of the contribution: **GHOST's skeleton is retained, GHOST's importance signal
is replaced.** The privilege mechanism, the cosine-derived per-layer budgets, the per-layer
top-K, and the geometry-grounded framing all come from GHOST. What is new is that the score is
made *task-conditional* — it asks "is this token relevant to the instruction the agent is
currently executing, and does it record a scene transition?" instead of "is this token
geometrically informative?".

---

## 2. Background: why the KV cache needs evicting

VGGT processes a multi-view sequence with full joint attention across all frames. StreamVGGT
makes this causal and online: each new frame attends to a KV cache populated by all prior
frames. That cache grows linearly with sequence length, which is the memory bottleneck this
work addresses.

**The concrete cost model in this repo.** From `Aggregator.__init__`
(`src/qwen_vl/model/vggt/models/aggregator.py:107-126`):

| Constant | Value |
|---|---|
| `depth` | 24 |
| `embed_dim` | 1024 |
| `num_heads` | 16 → `head_dim` = 64 |
| `num_register_tokens` | 4 → `patch_start_idx` = 5 (1 camera + 4 register) |
| `patch_size` | 14 |
| `aa_order` | `["frame", "global"]`, `aa_block_size=1` |

There are **two** stacks of 24 blocks — `frame_blocks` (`aggregator.py:135-150`) and
`global_blocks` (`aggregator.py:152-167`) — but **only the 24 global blocks cache K/V**.
`_process_global_attention` is the only branch that receives `past_key_values_block`
(`aggregator.py:465-487`); frame attention is within-frame and needs no history.

At eval the Habitat 640×480 RGB is smart-resized to 420×560, giving a 30×40 patch grid, so
`tokens_per_frame = 5 + 1200 = 1205`.

Per token, per layer, the cache holds K and V of `num_heads × head_dim = 1024` values each.
In bf16 that is `2 × 1024 × 2 = 4096` bytes — the current measured value, confirmed at
`peak_vggt_kv_mb = 3515.6` at `VGGT_TOTAL_BUDGET=900000` (`3515.6 MiB × 1024² / 899985 tokens
= 4096.0`). This was **6144** bytes/token/layer until a recent fix; §9 records why and how it
was corrected.

---

## 3. The GHOST base

> Yuan et al., *GHOST: Geometry-Hierarchical Online Streaming Token Eviction for Efficient 3D
> Reconstruction*. [arXiv:2605.15852](https://arxiv.org/abs/2605.15852)

GHOST is training-free and derives its importance signal from the model's **own 3D outputs**
(depth maps, point confidence, camera poses) rather than from attention scores — the explicit
contrast being with attention-heuristic methods such as InfiniteVGGT, which score tokens by
current-query/key similarity and are therefore agnostic to 3D scene structure.

It has three components.

### 3.1 Hierarchical dual-level importance

**Frame level.** Camera pose change (Eq. 1):

```
s_cam(t) = ‖T_t − T_{t−1}‖₂ + 1 − |q̂_t · q̂_{t−1}|
```

translation distance plus quaternion angular distance between consecutive poses. Depth-gradient
variance (Eq. 2):

```
s_geo(t) = Var(‖∇d_t‖)
```

as a proxy for geometric richness. Plus linear temporal recency `s_temp(t) = t / T_cur`. These
combine (Eq. 3) by sigmoid-normalising each to [0,1], taking a weighted sum, and dividing by the
maximum over cached frames:

```
s_frame(t) = [w_cam·σ(s_cam) + w_geo·σ(s_geo) + w_temp·σ(s_temp)] / max(·)
```

**Token level.** Visual saliency (Eq. 4) as the spatial gradient magnitude of the patch feature
map:

```
s_sal(t,p) = √(‖δ_x F_t(p)‖² + ‖δ_y F_t(p)‖²)
```

combined (Eq. 5) with the model's predicted depth confidence `c_t^d(p)` and point confidence
`c_t^p(p)` under the same sigmoid-then-max-normalise scheme.

**Combination (Eq. 6):**

```
φ(t,p) = w_f · s_frame(t) + w_k · s_token(t,p),    w_f + w_k = 1
```

Published weights, obtained by **grid search**: `(w_cam, w_geo, w_temp) = (0.55, 0.55, 0.25)`,
`(w_sal, w_dc, w_pc) = (0.28, 0.45, 0.35)`, `w_f = w_k = 0.5`.

### 3.2 Special-token privilege (Eq. 7)

Camera tokens encode global scene geometry and register tokens encode structural priors, so
neither should be evicted on the strength of a patch-oriented score. GHOST gives them a
deterministic boost:

```
φ(t, p_sp) = s_frame(t) + Δ_boost + ε_tb · r_{p_sp}
```

with `Δ_boost = 0.3`, `ε_tb = 10⁻⁶`, and `r ∈ {0, …, R}` the token's **intra-frame** rank
(camera, reg₀, …, reg₃). The `ε_tb` term breaks ties deterministically without randomness.

### 3.3 Cosine-similarity-guided layer-wise budgets

Offline, per layer ℓ, measure the mean cosine similarity between that layer's input and output
representations:

```
ρ̄_ℓ = (1/|S|) Σ_s ⟨x_ℓ⁽ˢ⁾, y_ℓ⁽ˢ⁾⟩ / (‖x_ℓ⁽ˢ⁾‖ · ‖y_ℓ⁽ˢ⁾‖)
```

A layer whose output closely resembles its input is near-identity and therefore transforms
little, so it can afford a smaller cache. Layer importance is `a_ℓ = 1 − ρ̄_ℓ`, allocated by a
tempered softmax:

```
π_ℓ = softmax(a_ℓ / τ),    B_ℓ = ⌊π_ℓ · B_total⌋,    τ = 0.5
```

**Reported results.** On 7-Scenes at 300 frames versus InfiniteVGGT: accuracy 0.040 → 0.023,
KV cache 6.94 GB → 3.51 GB at half budget (≈49% reduction), 1.75× faster inference; on Long3D
(9,545 frames) a 24.6% accuracy improvement.

**Stated limitations.** Not applicable to fully offline joint-attention models, and the
frame-level score assumes a static scene.

Note the target: GHOST optimises **reconstruction** metrics (Accuracy, Completeness, Normal
Consistency, Abs Rel, δ<1.25). Pivoting that objective to navigation is the starting point of
§5.

---

## 4. How this repo implements GHOST (`GHOST_SCORE_MODE=importance`)

`src/qwen_vl/model/vggt/eviction/importance_eviction.py` is a direct transcription of §3.1–3.2.

| GHOST | Implementation |
|---|---|
| Eq. 1 `s_cam` | `_pose_change_score` (`:21-35`) — decodes the `absT_quaR_FoV` pose encoding, `trans_diff + rot_diff` |
| Eq. 2 `s_geo` | `_compute_geometry_score_for_frame` (`:305-327`) — replicate-padded ∇d, `mag.var()` |
| Eq. 3 `s_frame` | `compute_frame_importance` (`:38-130`) — `_sigmoid_norm` per term, weighted sum, `/ (max + 1e-8)` |
| Eq. 4 `s_sal` | `compute_patch_saliency` (`:133-153`) |
| Eq. 5 `s_token` | `compute_token_importance` (`:156-225`) — confidence maps pooled to the patch grid by `pool_to_patch` |
| Eq. 6 `φ(t,p)` | `compute_combined_importance` (`:228-280`) |
| Eq. 7 privilege | `compute_combined_importance` (`:255-266`) — `frame_importance[t] + boost + eps * rank_id` |
| §3.3 budgets | `Aggregator._calculate_dynamic_budgets` (`aggregator.py:958-972`) |

**The weights match the paper exactly.** `configs/importance_weights_default.json` contains
`w_camera=0.55, w_geometry=0.55, w_temporal=0.25, w_saliency=0.28, w_depth_conf=0.45,
w_pts_conf=0.35, w_frame=0.5, w_token=0.5, special_token_boost=0.3,
special_token_tiebreak_eps=1e-6` — i.e. GHOST's published grid-searched values verbatim.

**The budget table reproduces §3.3 exactly.** I verified the arithmetic in
`configs/kv_budget_proportions_cosine.json` numerically:

- `importance_per_layer == 1 − cos_sim_per_layer` — exact, all 24 entries
- `proportions == softmax(importance / 0.5)` — agrees to 5.1 × 10⁻⁹
- `budgets_per_layer == round(proportions × 1_200_000)`, summing to exactly 1,200,000

Layer 14 is the outlier: `cos_sim = 0.6144`, the lowest in the stack, so it receives proportion
0.0718 — about 2.1× the flat early layers. Resolved per-layer budgets (runtime uses `.int()`
truncation at `aggregator.py:972`, not rounding, hence the small shortfall):

| `VGGT_TOTAL_BUDGET` | min layer | max layer (L14) | actual sum |
|---:|---:|---:|---:|
| 1,200,000 | 41,299 | 86,143 | 1,199,987 |
| 900,000 | 30,974 | 64,607 | 899,985 |
| 360,000 | 12,389 | 25,843 | 359,987 |

**Two implementation details worth noting.**

*Incremental scoring.* `compute_importance_incremental` (`:437-585`) caches per-frame token
scores and raw camera/geometry scores so each step appends one frame rather than rescoring the
whole history — avoiding O(T²) work over an episode. Because a frame's own metadata only becomes
available on the following step, camera and geometry scores use a delay-one-frame update
(`:521-533`).

*Where the top-K happens.* Eviction is executed inside the attention layer:
`Attention.eviction` (`attention.py:1070-1255`), called from `Attention.forward:1341` whenever
`cache_budget` is exceeded. The anchor/special-token policy comes from `KV_MODE_CONFIG`
(`aggregator.py:57-65`); the importance mode forces `anchor1_only` (`aggregator.py:31, 461-464`),
meaning only frame 0 is a hard anchor and later frames' camera/register tokens compete on score
alone — consistent with GHOST, where the privilege is a boost rather than an exemption.

**Caveat on `importance_weights_from_hyperparams.py`.** This module presents "closed-form
structural priors" claiming to derive the importance weights from architecture hyperparameters —
e.g. `w_cam_geo = (P₀ + R)/H − (H·P₀ − 1)/(H·L·D)`. These formulas reproduce a *legacy* constant
set (0.5623, 0.2458, 0.2834, 0.4489, 0.3491) to within 5 × 10⁻⁵, but **GHOST reports these
weights as grid-searched**, and the file does not reproduce the current default config
(0.55/0.25/0.28/0.45/0.35). The expressions are post-hoc algebraic fits, not derivations, and
should not be presented as a principled result.

---

## 5. The current method: `vln_segment_transition`

**Source:** `src/qwen_vl/model/vggt/eviction/vln_segment_transition.py`, orchestrated by
`VGGTEncoder.finalize_vln_segment_transition` (`vggt_encoder.py:208-465`).

### 5.1 Motivation

GHOST's score asks which tokens best support 3D reconstruction. In a VLN agent, the geometry
memory exists to support *navigation decisions*: recognising the doorway the instruction refers
to, remembering the turn just taken, localising the goal. A token can be geometrically
uninformative (a flat, low-texture wall) yet decision-critical because it is the landmark the
current instruction clause names — and vice versa. This method therefore conditions the
importance score on the instruction and on scene-transition structure.

### 5.2 What is retained from GHOST

- Per-layer top-K against the cosine-profiled budget table (unchanged, §3.3 / §4)
- Special-token privilege with the same `Δ_boost = 0.3`, `ε_tb = 10⁻⁶`, intra-frame rank
- Geometry grounded in the model's own outputs: camera pose change and depth structure
- Training-free operation throughout

### 5.3 The four score components

All four are computed once per frame, at insertion, and stored per token.

**(a) Geometry** — `vggt_encoder.py:300-329`. GHOST's Eq. 1 and Eq. 2, combined by
`geometry_weights` (0.5 / 0.5 from `configs/vln_segment_transition_weights.json`):

```
geometry(t) = 0.5 · σ(s_cam(t)) + 0.5 · σ(Var(‖∇d_t‖))
```

This is a **per-frame scalar**, broadcast to all 1205 tokens of the frame
(`vggt_encoder.py:372`).

**(b) Confidence** — `vggt_encoder.py:255-283`. Per token, `min(depth_conf, point_conf)`
(`confidence.merge: "min"`). The remap matters: VGGT emits confidence through `expp1`, i.e.
`conf = 1 + exp(x) ∈ (1, ∞)`. Clamping that to [0,1] would saturate every token to exactly 1.0
and silently kill the term along with the descriptor weighting and the transition gate. The code
instead maps monotonically onto [0,1):

```
c = 1 − 1 / max(conf, 1 + 1e-6)
```

**(c) Instruction-segment relevance** — the first genuinely new signal.

The instruction is split into action/clause spans by `split_instruction_with_spans`
(`vln_segment_transition.py:104-125`): on punctuation, on sequencing markers
(`then`, `and then`, `after that`, `next`), and on `and` when followed by a navigation verb
(`walk`, `turn`, `enter`, `exit`, …). The regex deliberately excludes the spatial preposition
`next to`, because "stop next to the plant" is one action, not two.

`build_instruction_segment_state` (`:174-210`) requires a **fast tokenizer with offset
mappings**, assigns each instruction token to the span it overlaps most
(`assign_tokens_to_segments`, `:128-172`), and mean-pools then L2-normalises the token
embeddings per segment. These segment embeddings are frozen for the episode.

Per frame, `compute_instruction_segment_relevance` (`:213-226`) scores each visual token by its
best-matching segment:

```
relevance(p) = (max_s cos(v_p, e_s) + 1) / 2
```

The critical detail is *which* visual tokens: `v_p` are the geometry features **after the
trained language-fusion projector** (`modeling_qwen3_5.py:765-778`), so the cosine is taken in
the language embedding space where comparing against text embeddings is meaningful. This is what
forces the structural change in §5.5.

Empirically the instruction yields a median of 4 segments (range 1–13; 48 of 1,839 episodes
produce a single segment).

**(d) Transition anchor** — the second new signal, targeting doorways, turns and room changes.

`build_frame_descriptor` (`:229-236`) reduces each frame to one confidence-weighted, L2-normalised
descriptor. `RecentTransitionState` (`:35-43`) holds the last `recent_window_size = 4`.
`compute_local_transition_score` (`:239-247`) measures how *unlike* the recent past the current
frame is:

```
transition(t) = mean_{j ∈ recent} (1 − cos(d_t, d_j)) / 2
```

High values mean the view just changed sharply — a transition. Raw novelty is noisy, so
`compute_transition_anchor` (`:250-262`) gates it by confidence and instruction relevance:

```
anchor(p) = transition(t) · (0.5 · confidence(p) + 0.5 · relevance(p))
```

so a transition is only "anchored" when it is both geometrically trustworthy and relevant to the
instruction.

### 5.4 Cache-relative z-score normalisation

This replaces GHOST's per-component sigmoid-and-divide-by-max.

The four components have wildly different realised scales — instruction relevance spreads over
~10⁻², transition over ~10⁻⁵ — so a raw weighted sum would let whichever component happens to be
widest dominate, regardless of the configured weights. `zscore_normalize`
(`vln_segment_transition.py:58-67`) standardises each component over the candidate population:

```python
centered = values - values.mean()
scale = centered.square().mean().sqrt()        # population std, unbiased=False
normalized = centered / scale.clamp_min(eps)
return normalized * (scale > eps)              # a dead component contributes exactly 0
```

The trailing mask is deliberate: a constant or numerically dead component returns exactly zero
rather than amplified noise.

`compute_candidate_final_score` (`:70-101`) then combines them over **exactly the patch
population that `topk` will consider**:

```
final(p) = Σ_c w_c · z_c(p),   w = (geometry 0.30, confidence 0.20, instruction 0.30, transition 0.20)
final(p_sp) = max_p final(p) + Δ_boost
```

Two consequences the design commits to, both pinned by `tests/test_vln_segment_transition.py`:

- The final score is **invariant to any positive affine rescaling** of a component
  (`:52-61`) — a component measured in different units cannot buy influence.
- Each component's share of the final variance is `w_i² / Σ_j w_j²` — i.e. the configured
  weights translate into actual influence (`:75-142`).

Note that special tokens get a **hard** floor here — strictly above every patch — which is
stronger than GHOST's soft boost. `test_special_tokens_score_above_every_patch` (`:63-73`)
asserts this.

Scores are kept in **fp32** throughout, with an explicit justification at
`vggt_encoder.py:349-353`: the realised spread is ~10⁻² while the fp16 ULP near 0.6 is
~4.9 × 10⁻⁴, so fp16 storage would collapse most of a frame into exact ties and hand the
eviction decision to `topk`'s arbitrary tiebreak.

### 5.5 Where eviction runs — the structural change from GHOST

GHOST evicts inside the attention layer as the forward pass proceeds. This method **cannot**,
because the instruction-relevance score requires the trained language projector's output, which
does not exist yet while VGGT's attention is still running.

So the aggregator call explicitly disables in-attention eviction — `total_budget=0`,
`eviction_mode=""` (`vggt_encoder.py:900-919`, with the comment *"The segment-transition mode
prunes immediately after its existing language-space projector runs; baseline GHOST still prunes
here"*) — and all 24 layers are pruned in a single pass afterwards, triggered from
`modeling_qwen3_5.py::_finalize_vln_segment_cache:753`.

The prune loop (`vggt_encoder.py:403-456`):

```python
for layer_idx, kv in enumerate(self._streaming_past_key_values):
    candidate_meta = concat_metadata(self._vln_metadata_per_layer[layer_idx], new_meta)
    key, value = kv
    assert key.shape[2] == candidate_meta.final_score.numel()
    layer_budget = int(self._vln_layer_budgets[layer_idx])
    candidate_meta.final_score, _ = compute_candidate_final_score(candidate_meta, score_weights, boost)
    if key.shape[2] > layer_budget:
        keep = torch.topk(candidate_meta.final_score, k=layer_budget,
                          largest=True, sorted=False).indices.sort().values
        self._streaming_past_key_values[layer_idx] = [key.index_select(2, keep),
                                                      value.index_select(2, keep)]
        candidate_meta = gather_metadata(candidate_meta, keep)
    self._vln_metadata_per_layer[layer_idx] = candidate_meta
```

Three things to note. The score is **recomputed cache-wide every step**, so the naive weighted
sum built earlier in the same function (`merged_final`, `vggt_encoder.py:330-336`) is *not* what
governs eviction — it only seeds the new frame's metadata and feeds the probe. `keep` is sorted
so causal order is preserved. And the gather is skipped entirely while a layer is still under
budget, which is a pure no-op saving during cache fill.

Layer budgets are resolved **once per episode** (`vggt_encoder.py:196-200`), not per frame, so
the per-frame path stays free of host-device transfers.

---

## 6. End-to-end trace

```
Habitat RGB frame
  └─ evaluation.py:612       rgb_list.append(...)   ; history window = linspace(num_history=8)
  └─ evaluation.py:354-358   ONLY the current frame becomes geometry_encoder_inputs [1,3,420,560]
       └─ vggt_encoder.py:893   aggregator(frame, past_key_values=..., use_cache=True)
            └─ attention.py:1314   k = cat([past_k, k]); v = cat([past_v, v])     per global layer
            └─ (importance mode only) attention.py:1341  eviction() → per-layer top-K
       └─ vggt_encoder.py:1083  _append_ghost_frame_metadata
            └─ camera_head → camera_pose ; depth_head → depth, depth_conf ; point_head → pts3d_conf
       └─ returns aggregated_tokens_list: 24 × [1,1,1205,2048]
  └─ modeling_qwen3_5.py:765  geo_ln + geo_mlp  (the trained projector)  → [300, lang_hidden]
  └─ modeling_qwen3_5.py:753  _finalize_vln_segment_cache
       └─ vggt_encoder.py:208  finalize_vln_segment_transition
            ├─ geometry / confidence / relevance / transition → anchor
            ├─ _vln_expand_index: map merged language tokens back to source KV patches
            ├─ per layer: concat_metadata → z-score → topk(layer_budget) → index_select
            └─ RETAINED CACHE
  └─ feature fusion: residual add of the projected geometry delta onto vision-token positions
     of decoder layers 0/1/2  (deepstack_language_add, feature_fusion.py:645-657)
  └─ generate() → "STOP" | "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" → env.step()
```

**The important subtlety.** The retained cache does *not* gate what is fused into the current
step's language embeddings — that fusion uses only the just-computed current-frame features. Its
effect is indirect and cumulative: the surviving KV determines what the **next** frame's VGGT
forward pass can attend to, which shapes that frame's features, its own scores, and its fused
embedding. Eviction quality compounds through VGGT's own attention over the episode rather than
acting as a retrieval step at generation time.

One index-mapping detail: the projector merges each 2×2 patch block into one language token, so
scores computed on the merged grid must be mapped back to source KV patches by an exact
floor-divide inverse (`_vln_expand_index`, `vggt_encoder.py:466-483`). Scaling by
`aligned_h / patch_h` instead would silently shift every other row whenever the patch grid is odd.

---

## 7. Data structures and invariants

`VLNGhostTokenMetadata` (`vln_segment_transition.py:46-55`) carries, per token:
`frame_id` (int32), `geometry_score`, `confidence_score`, `instruction_score`,
`transition_score`, `final_score` (all fp32), `is_special` (bool), `best_segment_id` (int16)
— **27 bytes/token/layer**.

It is stored **per layer** (`_vln_metadata_per_layer`, one entry per aggregator layer) because
each layer evicts to its own budget and therefore retains a different subset of tokens. Metadata
must track that same subset to stay index-aligned with `key.shape[2]`.

`concat_metadata` and `gather_metadata` (`:265-287`) keep KV and metadata in lock-step across
append and prune. `concat_metadata` raises rather than returning `None` on a one-sided field,
with the reasoning stated in the source: silently dropping a field would desynchronise it from
the K/V it describes, permanently and undetectably.

`tests/test_vln_segment_transition.py` pins the design contract: z-score neutrality on constant
input, affine-rescaling invariance, strict special-token ordering, weight-proportional variance
share, and KV/metadata index consistency after pruning to a real production budget (12,389 —
layer 0 at `VGGT_TOTAL_BUDGET=360000`).

**Read-only instrumentation.** `vln_score_probe.py` (`VLNScoreProbe`) logs per-frame score
quantiles and per-layer retention statistics — retained count, special fraction, unique surviving
frames, token-age quantiles, per-component std and variance share — to JSONL. It is gated by
`VLN_SCORE_PROBE=1` and feeds nothing back into scoring. **It has never been run**; no output
exists anywhere in the tree.

---

## 8. Configuration surface

**`configs/vln_segment_transition_weights.json`** — the active scorer profile. Validated at
`vggt_encoder.py:535-570`: each weight group must sum to 1.0, all values in [0,1],
`confidence.merge == "min"`, `normalization.candidate_terms == "zscore"`.

```json
score_weights:     geometry 0.30, confidence 0.20, instruction 0.30, transition 0.20
geometry_weights:  camera_pose_change 0.50, depth_structure 0.50
transition:        recent_window_size 4, confidence_gate_weight 0.50, instruction_gate_weight 0.50
```

**`configs/importance_weights_default.json`** — GHOST's grid-searched weights (§4). Note that
`special_token_boost` and `special_token_tiebreak_eps` are read from **this** file even in
`vln_segment_transition` mode (`vggt_encoder.py:354-355`), despite that mode having its own
config file — an easy coupling to miss when tuning.

**`configs/kv_budget_proportions_cosine.json`** — the per-layer budget table (§3.3, §4). Only
the `proportions` key is read (`vggt_encoder.py:512-518`); `budgets_per_layer` is documentation.

**Environment variables** (all read in `VGGTEncoder.__init__`, `vggt_encoder.py:97-144`):

| Variable | Default | Effect |
|---|---|---|
| `USE_GHOST_KV_CACHE` | config (`False`) | Enable eviction at all |
| `GHOST_SCORE_MODE` | `importance` | `importance` \| `vln_segment_transition`; anything else raises |
| `VGGT_TOTAL_BUDGET` | 1,200,000 | Total tokens **summed across all 24 layers**; must be > 0 |
| `VLN_SEGMENT_TRANSITION_WEIGHTS_PATH` | config | Scorer profile |
| `VGGT_KV_START` / `VGGT_KV_RECENT` | 8 / 48 | `StartRecentKVCache` window when eviction is off |
| `VLN_SCORE_PROBE`, `_DIR`, `_EVERY`, `_LAYERS` | off, `vln_score_probe`, 10, `0,11,23` | Instrumentation |

Launchers: `scripts/evaluation/eval_janus_vln_scene_segment_transition_ghost.sh` (this method),
`eval_janus_vln_scene_ghost.sh` (GHOST baseline), `eval_janus_vln_scene.sh` (no eviction).

---

## 9. Known deviations and caveats

These are recorded so they are known rather than discovered by a reviewer.

**Geometry has no within-frame discriminative power.** `geometry_score` is a per-frame scalar
broadcast to all 1205 tokens (`vggt_encoder.py:372`), yet carries the joint-largest weight
(0.30). After z-scoring it acts purely as a whole-frame include/exclude bias; it cannot
distinguish two tokens from the same frame. GHOST avoided this by pairing `s_frame` with a
genuinely per-token `s_token` at equal weight (Eq. 6).

**The temporal recency term was dropped.** GHOST's `s_frame` includes `w_temp · σ(t / T_cur)`.
This method has no recency, decay, or age term at all, and all four components are frozen at
insertion and never rescored. The only time-dependence is through each component's σ in the
z-score — and that σ is measured over the **survivor** population, which was itself selected to
maximise `Σ w_c z_c`. Selection truncates variance along the dominant direction, shrinking its σ
and *raising* its effective weight `w_c/σ_c` on the next step. That is positive feedback toward
single-component ossification. It is a structural property of the design, not yet measured; the
probe's `candidate_variance_proxy_share` field exists precisely to test it.

**Special-token privilege is uncapped.** Every special token from every frame outranks every
patch forever (`vln_segment_transition.py:100`). At 5 tokens/frame and 400 steps that is 2,000
permanently unevictable tokens per layer — 6.5% of the smallest layer at 900k, 16% at 360k, and
32% at 180k. It becomes the binding constraint at aggressive budgets.

**Cached K was fp32 — fixed.** `qk_norm=True` makes `q_norm`/`k_norm` `nn.LayerNorm`
(`attention.py:55-56`), and autocast promotes LayerNorm to fp32, so `k` left the norm as fp32
while `v` stayed bf16, and the fp32 K used to be cached directly. That cost 6144 bytes/token/
layer instead of 4096 (`1024×4 + 1024×2`) for no benefit, since `F.scaled_dot_product_attention`
under autocast downcasts K to bf16 at use time regardless — the stored precision never reached
the attention computation. `attention.py:1319-1323` now casts `k` to `v.dtype` immediately after
RoPE and before the cache `torch.cat`:

```python
if k.dtype != v.dtype:
    k = k.to(v.dtype)
```

Placement is load-bearing both directions: *after* RoPE (not after `k_norm`) because
`RotaryPositionEmbedding2D` computes cos/sin in `tokens.dtype`, so casting earlier would rotate
in bf16 and add a second rounding on top of the one SDPA already applies; *before* the cache
`torch.cat` (not at `new_kv = (k, v)`) because `torch.cat` type-promotes, so downcasting later
would let the next frame's concatenation of a bf16 past with an fp32 new key silently promote
the whole cache back to fp32. Verified bit-identical on a controlled A/B (identical weights,
10 streamed frames, autocast bf16): attention output `max|diff| = 0`, and the cached K after
downcast equals the pre-fix fp32 K downcast to bf16 — one rounding, not two. The seven sites in
`Attention.eviction` that rank importance scores via `topk` are pinned to `.float()` rather than
following `k.dtype`, so the now-bf16 K cannot quantize the ranking into ties. Measured on a real
run at `VGGT_TOTAL_BUDGET=900000`: `peak_vggt_kv_mb` 5273.3 → 3515.6, exactly the predicted
−33.3%; `peak_alloc_mb` 21062 → 19314.

**Camera-head KV was unbounded — now bounded.** `CameraHead.trunk_fn` runs 4 refinement
iterations × 4 trunk layers per frame with `use_cache=True`, appending one token per
iteration per layer, and nothing pruned that cache: it grew linearly with episode length to
~50 MB at 400 steps, the last CUDA component still growing without limit (`Attention`'s own
cache carries K/V but the camera trunk is a separate 4-layer transformer with its own
attention state). `_trim_camera_kv()` (`vggt_encoder.py:1102-1140`) now runs every frame right
after `camera_head`, keeping only `VGGT_CAMERA_KV_START` frames (default 1, the reference-camera
frame) plus `VGGT_CAMERA_KV_RECENT` frames (default 64) and discarding the rest; the tokens-per-
frame count is read off the first frame's cache rather than assumed from `num_iterations`, so it
stays correct if the head's defaults change. `_reset_vggt_attention_cache_state` was also fixed
to reset the camera trunk's blocks alongside the 24 global blocks — it previously walked only
`aggregator.global_blocks`, so the trunk's cache state leaked across episodes.

Measured with the real `CameraHead`, 140 streamed frames: 4 tokens/frame, frame-aligned; at 400
frames the cache is 52.4 MB untrimmed (matching the previously measured ~50 MB) versus **8.5 MB
constant** trimmed. Unlike the K-cache fix, this one is **not bit-identical**: the only value
consumed downstream is `sigmoid(_pose_change_score(prev, cur).mean())`, and trimming the trunk's
attention context changes that scalar by up to ~1e-3 once the current frame falls outside the
window — small, but enough to occasionally flip a marginal eviction decision and diverge a
trajectory. Verified on 268 episodes shared between a pre-trim and post-trim run: **143/143**
episodes ≤65 steps (inside the window) are trajectory-identical; **50/125** episodes beyond 65
steps diverge. Treat this as a small, bounded approximation, not a free fix — report it as its
own arm (G1 in `ABLATION_PLAN.md`) rather than folding it silently into other comparisons.

**The budget table is not reproducible from source.** The generating script
(`compute_kv_budget_from_cosine_sim.py`, referenced at `aggregator.py:203, 960`) is not in the
repo. The JSON's internal arithmetic checks out (§4), but what was actually profiled — which
dataset, how many samples, which tensors were compared — cannot be recovered from this tree.

**`StartRecentKVCache` units.** `vggt_encoder.py:29-56` slices `dim=2`, the **token** dimension,
with `start_size=8`/`recent_size=48`, while its log message and comments describe these as
*frames*. That path retains 56 tokens, not 56 frames. It only affects the no-eviction baseline,
but it means the repo currently has no valid full-cache reference to compare against.

**Frame-0 metadata aliasing.** `concat_metadata(None, new)` returns `new` itself
(`:266-267`), so on the first frame — when nobody is over budget and no `gather_metadata` copy
occurs — all 24 entries of `_vln_metadata_per_layer` are the same object. Benign today because
the next frame's `cat` allocates fresh tensors.

**`importance_weights_from_hyperparams.py`** — see §4. Post-hoc algebraic fits presented as
structural derivations, reproducing a legacy constant set that is not the current default.

---

## 10. Measured behaviour

All numbers from completed runs in `evaluation_gate_scale_fix_eval/`, checkpoint
`spatialstack_janus_vln_train-gate-scale-4B-loss-3`, R2R `val_unseen`, single GPU.

**Headline result** — `budget_900000_steps_400`, full 1,839 episodes, **fp32 K, unbounded
camera KV** (the numbers below predate both fixes in §9; still the only *complete* full run,
so still the reference until the re-run below finishes):

```
SR 54.16   SPL 50.02   OS 60.36   NE 5.35
```

**Memory** (per-episode peaks, instrumented at `src/evaluation.py:701-719`):

| | budget 900k | budget 360k |
|---|---:|---:|
| Peak CUDA allocated | 21,065 MB | 17,949 MB |
| VGGT KV cache | 5,273 MB (25%) | 2,109 MB (12%) |
| Camera-head KV | 16 MB avg, 50 MB @400 steps *(unbounded — pre-fix)* | 12 / 25 MB |
| Eviction metadata | 23.2 MB | 9.3 MB |
| **Fixed floor (weights + activations)** | **~15,806 MB** | **~15,819 MB** |
| Mean step latency | 671 ms (VGGT 81 ms, 12%) | 643 ms (VGGT 65 ms, 10%) |

The ~15.8 GB floor is identical at both budgets, so eviction addresses at most a quarter of peak
CUDA. Retained history depth is 25.7–53.6 frames at 900k and 10.3–21.4 frames at 360k
(budget ÷ 1205, min to max layer).

**Re-run with both §9 fixes (bf16 K + bounded camera KV), in progress.** Same checkpoint,
budget, and step cap; `evaluation_gate_scale_fix_eval/scene_segment_transition_bf16k/
budget_900000_steps_400/`. SR/SPL are not reported here until the full 1,839 episodes complete —
the camera-KV trim is not bit-identical (see §9), so a partial-episode subset is not a valid
stand-in for the number above. What is already confirmed, independent of completion, since it
depends only on cache footprint per step, not on which episodes have finished:

| | budget 900k, fp32 K (headline above) | budget 900k, both fixes |
|---|---:|---:|
| `peak_vggt_kv_mb` | 5,273.3 | **3,515.6** (−33.3%, exact match to the bf16-K prediction) |
| `peak_alloc_mb` | 21,065 | **19,314** |

**Budget sensitivity.** On the 589 episodes shared by the 360k and 900k runs:

| | SR | SPL | NE | KV MB |
|---|---:|---:|---:|---:|
| 360,000 | 48.90 | 44.79 | 5.84 | 2,109 |
| 900,000 | 50.42 | 45.46 | 5.99 | 5,273 |

Bootstrap 95% CI on ΔSR is [−1.53, +4.75] and McNemar z = 0.96 — **not significant** — and NE
moved the wrong way. The comparison is additionally biased toward 900k, which ran with a 2×
larger step cap. On the 416 episodes where neither run hit its cap the picture is unchanged
(ΔSR +1.68, CI [−1.44, +4.81]). At present, 2.5× the KV memory buys no measurable navigation
accuracy.

**Accuracy versus episode length** (full run):

| steps | n | SR% |
|---:|---:|---:|
| 25–50 | 533 | 70.4 |
| 75–100 | 216 | 52.3 |
| 150–175 | 49 | 36.7 |
| 400 (cap) | 261 | 10.7 |

---

## 11. References

- **GHOST** — *Geometry-Hierarchical Online Streaming Token Eviction for Efficient 3D
  Reconstruction*. [arXiv:2605.15852](https://arxiv.org/abs/2605.15852) — the base method.
- **VGGT** — *Visual Geometry Grounded Transformer* — the joint-attention backbone vendored at
  `src/qwen_vl/model/vggt/`.
- **StreamVGGT** — the causal streaming inference paradigm this cache implements.
- **InfiniteVGGT** — GHOST's attention-heuristic baseline; scores tokens by current-query/key
  similarity. Explicitly *not* the direction taken here.
- **Evict3R** — *Training-Free Token Eviction for Memory-Bounded Streaming Visual Geometry
  Transformers*. [arXiv:2509.17650](https://arxiv.org/abs/2509.17650)
- **STAC** — *Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D
  Reconstruction*. [arXiv:2603.20284](https://arxiv.org/abs/2603.20284)
- **StreamCacheVGGT** — *Streaming Visual Geometry Transformers with Robust Scoring and Hybrid
  Cache Compression*. [arXiv:2604.15237](https://arxiv.org/abs/2604.15237)
- **OVGGT** — *O(1) Constant-Cost Streaming Visual Geometry Transformer*.
  [arXiv:2603.05959](https://arxiv.org/abs/2603.05959)

---

## 12. Report summary

*Prose, for lifting directly into a short write-up. Not anchored to file/line references —
those are in §1–11 above if a claim here needs to be traced back to code.*

**The problem.** Streaming VGGT attends causally over a growing key-value cache: every frame
adds tokens that all later frames attend to, so memory grows without bound over an episode. A
vision-language-navigation policy needs that geometry memory, but not an unbounded one — it
needs to keep the tokens that matter for *deciding where to go next*, and discard the rest.
GHOST solved a version of this problem for offline 3D reconstruction, scoring tokens by how much
they contribute to reconstruction quality. That is the wrong criterion for navigation: a token
can be geometrically uninformative — a flat, low-texture wall — yet decision-critical, because
it is the landmark the instruction just named.

**The core new idea: instruction-conditioned relevance.** This is the centerpiece of the
contribution and the one component that has no GHOST analogue at all. The instruction is first
**split into chunks** along its action boundaries — clause and punctuation breaks, and
sequencing words such as "then" or "after that" (deliberately *not* splitting on the spatial
phrase "next to", so "stop next to the plant" stays one action). Each chunk's tokens are then
**mean-pooled and L2-normalized into a single embedding per segment**, computed once and frozen
for the whole episode. Every visual token sitting in the KV cache is then scored by **cosine
similarity against whichever instruction segment it matches best** — not against the instruction
as one aggregate vector. That last point is the mechanism's whole reason for existing: pooling
the entire instruction into one embedding would wash a multi-step instruction into a single
average meaning, and an early-route landmark ("turn at the kitchen") would lose its match score
as soon as later clauses ("then walk down the hall") entered that average. Scoring against the
*best-matching segment instead* means a token stays relevant for as long as any part of the
instruction still needs it, independent of how far the episode has progressed. This comparison
only makes sense in the trained language projector's embedding space — visual features have to
be projected into the same space as the text before a cosine similarity is meaningful — which is
why eviction had to be restructured to run *after* that projector, rather than inside VGGT's
attention layers the way GHOST's does. That restructuring is a consequence of this idea, not a
separate contribution.

**The second new signal: transition anchoring.** Each frame is reduced to one confidence-
weighted descriptor; a token's score is boosted when its frame looks unlike the last few frames
seen — a sharp visual change, the kind produced by a doorway, a turn, or entering a new room —
gated by both geometric confidence and instruction relevance, so a "transition" only counts when
it is trustworthy and relevant, not just visually different.

**What was kept from GHOST.** The layer-wise KV budget allocation (cosine-similarity-profiled,
non-uniform across the 24 attention layers), the special-token privilege that protects camera
and register tokens from eviction, and the fully training-free operation are all inherited
unchanged. The contribution is best read as *a new scoring signal inside an already-working
skeleton*, not a redesign from scratch.

**Where the evidence currently stands.** Against a no-eviction reference, the method holds SPL
essentially flat (+0.07) while SR drops slightly (−1.09), at a 12.8% reduction in peak GPU
memory — parity, not yet a clear win, and the headline number so far is memory savings at
matched accuracy rather than an accuracy gain. A full-scale comparison against GHOST's own
scoring has not been run yet (only 299 of 1,839 episodes so far), and no ablation has yet
isolated which of the four score components — geometry, confidence, instruction, transition —
is actually responsible for the result; a full ablation matrix for exactly this question is
planned in `ABLATION_PLAN.md`. Two honest limitations worth naming in a short report: the
geometry term is a single scalar per frame despite carrying the largest configured weight, so it
cannot distinguish two tokens from the same frame; and the design has no recency or decay term
at all (GHOST's has one), with a structural argument — not yet directly measured — for why that
could cause long episodes to over-concentrate on whichever score component happens to dominate
early on.

**Two engineering fixes, orthogonal to the method.** Independent of anything above, two memory
issues were found and fixed without changing what gets scored or evicted. The key cache was
being stored in full precision due to a normalization-layer side effect, wasting a third of its
memory for no benefit — verified bit-for-bit identical after the fix, a pure efficiency gain.
Separately, a small internal cache inside the camera-pose head was never being pruned and grew
without bound over an episode; it is now capped to a bounded window, closing the last unbounded
memory component in the pipeline, at the cost of a small (~0.1%) numerical drift in long
episodes that is being tracked as its own experimental arm rather than absorbed silently into
other results.
