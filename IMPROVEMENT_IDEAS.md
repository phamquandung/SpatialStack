# Improvement Ideas — VLN Segment-Transition KV Eviction

**What this document is.** Where this eviction method can still gain, from re-reading the
eviction path and surveying the 2026 literature. Each idea states what it costs, what evidence
supports it, and what would kill it. Companion to `METHOD.md` (what the method is) and
`ABLATION_PLAN.md` (how to measure it) — arms already scheduled there are referenced, not
restated.

---

## 1. Framing: compression is not the bottleneck — episode length is

This is the single most important thing to internalise before choosing what to build.

**More KV budget buys nothing measurable.** On the 589 episodes shared by the 360k and 900k
runs, 2.5× the memory gave ΔSR +1.53 with bootstrap 95% CI `[-1.53, +4.75]` and McNemar
z = 0.96 — not significant — and NE moved the *wrong* way. Restricting to the 416 episodes where
neither run hit its step cap changes nothing (ΔSR +1.68, CI `[-1.44, +4.81]`).

**And the ceiling on compression is low anyway.** The ~15.8 GB fixed floor (Qwen3.5-4B +
VGGT-1B + heads + per-step activations) is identical at every budget, so eviction addresses **at
most 25% of peak CUDA** no matter how good the scorer gets.

**Meanwhile accuracy collapses with episode length:**

| steps | n | SR% |
|---:|---:|---:|
| 25–50 | 533 | **70.4** |
| 75–100 | 216 | 52.3 |
| 150–175 | 49 | 36.7 |
| 400 (cap) | 261 | **10.7** |

The 261 cap-hitting episodes are 14% of the set at 10.7% SR. Lifting them to even 30% is
**+2.7 SR overall** — larger than anything available in the budget sweep.

**Therefore:** rank ideas by whether they attack the long-horizon collapse, not by how much
memory they save. Every idea below is ordered on that basis. Evaluate each on the
length-stratified table above, **not on aggregate SR** — aggregate SR is dominated by short
episodes that already work, and will mask a real gain on long ones.

---

## 2. Gaps found by re-reading the code

Three things the current implementation leaves on the table.

**Gap 1 — `best_segment_id` is computed, stored, and never used for scoring.** It is produced by
`compute_instruction_segment_relevance` (`vggt_encoder.py:297`), stored per token in
`VLNGhostTokenMetadata` (`:391`), and read **only** by `vln_score_probe`. Meanwhile eviction is a
single global `topk` over `final_score` with no guarantee the surviving cache still covers every
instruction segment. Running `split_instruction_with_spans` over all 1,839 episode instructions
gives a **median of 4 segments**, with **96.5% of episodes having ≥2**:

| segments | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| episodes | 65 | 228 | 411 | 440 | 335 | 203 | 157 |
| share | 3.5% | 12.4% | 22.3% | 23.9% | 18.2% | 11.0% | 8.5% |

A signal that is already computed, already stored, and already free is doing nothing.

**Gap 2 — one score table drives all 24 layers.** `new_meta` is built once per frame
(`vggt_encoder.py:383`) and `concat_metadata` feeds the *identical* score vector into every
layer's prune loop (`:419`). Only `layer_budget` differs. Layers therefore retain different
*numbers* of tokens but rank them by an identical criterion — the layer-wise budget table is
doing all the per-layer differentiation, and the scorer none.

**Gap 3 — no redundancy handling at all.** Grepping the `vln_segment_transition` path for
merge / diversity / cluster / dedup returns nothing. A static scene observed for 50 frames fills
the cache with near-duplicate tokens that all score similarly well and all survive. GHOST ships
`_compute_diversity_scores` (cosine of each key to the mean key — about ten lines) in
`GHOST/src/streamvggt/quantization/diversity_aware_quantization.py`, but `quantization/` was
**not vendored** into `src/`, so adopting this is a small port, not a subsystem.

---

## 3. Sequencing: run the probe before building anything

Ideas **A2** and **A3** rest on hypotheses that have **never been measured**: that the cache
collapses onto a few instruction segments, and that it accumulates near-duplicates.
`VLN_SCORE_PROBE` has still never been run — confirmed again, no output exists anywhere in the
tree.

The probe already logs exactly the fields that test these premises:

| Field | Tests |
|---|---|
| `retained_per_segment` | Does the cache still cover all segments? → **A2** |
| `unique_frames`, `age_quantiles` | Is the cache collapsing onto few frames? → **A1, A3** |
| `candidate_variance_proxy_share` | Which component actually dominates, over time? → **A1** |
| `special_fraction` | How much budget the privilege eats → `ABLATION_PLAN.md` F3 |
| `distinct_score_levels` | Tie collapse |

Cost is one already-planned arm plus one device sync per logged step. Building A2 or A3 first
risks engineering against a premise that turns out to be false.

---

## 4. Tier A — cheap, targets the collapse, reuses signals already present

### A1 — Recency / decay term

GHOST's frame score includes `w_temp · σ(t / T_cur)`. **This method dropped it entirely**, and
all four components are frozen at insertion and never recomputed. The only time-dependence is
through each component's σ in the z-score — and that σ is measured over the **survivor**
population, which was itself selected to maximise `Σ w_c z_c`. Selection truncates variance
along the dominant direction, shrinking its σ and *raising* its effective weight `w_c/σ_c` on
the next step: positive feedback toward single-component ossification (`METHOD.md` §9).

`frame_id` is already stored per token, so an age term costs nothing to compute. This is the
cheapest item on this list and the most direct counter to the measured decay.

Already scheduled as arm **F6** in `ABLATION_PLAN.md` — but it is listed there as a control
among many. It should be promoted: on the evidence above it is a likely *fix*, not a
sanity check.

**What would kill it:** if the probe shows `age_quantiles` staying broad and
`candidate_variance_proxy_share` staying balanced over long episodes, the ossification story is
wrong and a decay term is just another knob.

### A2 — Segment-coverage quota *(the most promising new idea)*

Replace the single global `topk` with a per-segment floor. Each of the S instruction segments is
guaranteed a minimum share of the layer budget — e.g. `B/(2S)` slots — with the remainder
allocated globally as now. `best_segment_id` already assigns every cached token to a segment
(Gap 1), so this is pure allocation logic over data that already exists.

**Why it should work.** The method's stated advantage over a single pooled instruction embedding
is that a token stays relevant if it matches *any* clause. But nothing enforces that at eviction
time: a global `topk` can legitimately drop every token belonging to segment 3 if segments 1 and
2 happen to score higher, and once dropped they never return. The longer the episode, the more
opportunities to lose a segment entirely — which lines up with the observed collapse profile.
A quota converts "any clause *may* survive" into "every clause *does* survive".

This also strengthens the method's own research story rather than being a tuning knob: it makes
segment structure load-bearing at eviction, not just at scoring.

**What would kill it:** if `retained_per_segment` from the probe shows coverage already stays
balanced across segments through long episodes, the problem does not exist and the quota only
costs flexibility.

### A3 — Redundancy / diversity penalty

Two implementations, cheapest first:

1. **Frame-level (near-free).** `build_frame_descriptor` and `RecentTransitionState` already
   produce L2-normalised per-frame descriptors. Detect near-duplicate *frames* by cosine over
   stored descriptors and down-weight their tokens. No new machinery whatsoever.
2. **Token-level.** Port GHOST's `_compute_diversity_scores` (~10 lines, cosine of each key to
   the mean key; lower similarity = more diverse) and add it as a penalty term.

Either frees budget for genuinely new content **without raising the budget** — which matters
precisely because §1 shows raising the budget does nothing.

Related literature: **GraphKV** propagates decay through a similarity graph specifically to keep
selected tokens diverse and avoid retaining redundant neighbours.

**What would kill it:** if `unique_frames` stays high through long episodes, the cache is not
actually filling with duplicates.

---

## 5. Tier B — literature-backed, higher cost

### B1 — Three-tier retain / merge / evict

**StreamCacheVGGT** ([arXiv:2604.15237](https://arxiv.org/html/2604.15237)) replaces binary
eviction with three tiers: tokens above `τ_evict` are **retained**; those between `τ_merge` and
`τ_evict` are **merged** into their nearest retained anchor (cosine similarity on keys, fused as
an importance-weighted average); only those below `τ_merge` are **discarded**. Reported: merge
ratio **0.15** optimal; KITTI @ 500 frames abs-rel **0.135 → 0.124 (8.1%)**; NRGBD @ 300 frames
completeness 0.0147 → 0.0135.

**Honest caveat:** those are *reconstruction* metrics. Our objective is navigation SR, and
merging KV entries perturbs attention in a way that is much harder to reason about for a
downstream policy than for a geometry head. The transfer is plausible, not guaranteed. Treat the
8.1% as motivation to try it, not as an expected gain.

### B2 — Rescore retained tokens as the active segment advances

**VLN-Cache** ([arXiv:2603.07080](https://arxiv.org/html/2603.07080v3)) models
**instruction-conditioned task-stage shifts** — relevance moves as the agent progresses through
the instruction. Here all four components are frozen at insertion; instruction relevance in
particular could be re-evaluated against the currently-active segment. Composes naturally with
A2: the quota decides *how many* slots each segment gets, this decides *how* their scores age.

### B3 — Per-layer score differentiation / CLCES

Attacks Gap 2 directly. StreamCacheVGGT's **CLCES** scores tokens by cross-layer ranking
stability, `s_i = s_i × (1 + λ · Cons_i)`, rewarding tokens whose importance is consistent
across a sliding window of layers. Alternative variant: let each layer's own K contribute a
layer-specific term so the 24 layers stop sharing one ranking.

Highest implementation cost here, and the most speculative in this setting — the current design
deliberately computes the score once, outside the attention loop, and per-layer scoring gives
that up.

### B4 — Query-conditioned scoring

**History-Conditioned Spatio-Temporal Token Pruning for VLN**
([arXiv:2603.06480](https://arxiv.org/pdf/2603.06480)) selects history tokens conditioned on the
**current-frame query**, not only on insertion-time properties.

**Tension to note:** GHOST deliberately avoids attention-based scoring — that is its explicit
contrast with InfiniteVGGT — and query conditioning re-introduces the per-step O(cache) work
GHOST's design exists to avoid. Adopting it is a departure from the method's stated basis, not
an extension of it.

---

## 6. Explicitly not worth pursuing

**Adaptive per-step budget.** The objective is *peak* CUDA. A scheme that lowers the average
while leaving `B_max` unchanged does not meet the goal. If pursued anyway, both average and peak
must be reported.

**Sub-bf16 KV quantization.** The free 33% is already taken (bf16 K, `METHOD.md` §9). Going
further means dequantising a multi-GB cache every layer every step, with real accuracy risk, for
a component that is only 25% of peak.

**Raising the budget above 900k.** §1 — the curve is already flat there.

**The real ceiling.** If further *memory* reduction is the actual goal, the only remaining large
lever is the 15.8 GB floor — quantizing or offloading Qwen3.5-4B itself. That is outside token
eviction entirely, and it bounds this whole line of work.

---

## 7. Suggested order

```
probe  →  A1 (recency)  →  A2 (segment quota)  →  A3 (diversity)
       →  re-measure the length-stratified SR table
       →  only then consider Tier B
```

Evaluate every Tier-A change on the **length-stratified breakdown** (25–50 / 75–100 / 150–175 /
400-cap), not on aggregate SR. A change that lifts the 400-cap bucket from 10.7% to 30% while
leaving short episodes untouched is worth +2.7 SR overall and would barely register as
significant in an aggregate paired test at n=1839 (CI ≈ ±1.4).

---

## 8. Sources

- **StreamCacheVGGT** — *Streaming Visual Geometry Transformers with Robust Scoring and Hybrid
  Cache Compression*. [arXiv:2604.15237](https://arxiv.org/html/2604.15237)
- **History-Conditioned Spatio-Temporal Visual Token Pruning for Vision-Language Navigation**.
  [arXiv:2603.06480](https://arxiv.org/pdf/2603.06480)
- **VLN-Cache** — *Enabling Token Caching for VLN Models with Visual/Semantic Dynamics
  Awareness*. [arXiv:2603.07080](https://arxiv.org/html/2603.07080v3)
- **STAC** — *Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D
  Reconstruction*. [arXiv:2603.20284](https://arxiv.org/html/2603.20284)
- **GHOST** — *Geometry-Hierarchical Online Streaming Token Eviction for Efficient 3D
  Reconstruction*. [arXiv:2605.15852](https://arxiv.org/pdf/2605.15852)
- **FreqCache** — *Accelerating Embodied VLN Models with Adaptive Frequency-Guided Token
  Caching*. [arXiv:2604.24391](https://arxiv.org/pdf/2604.24391)
