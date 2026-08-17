# Ablation Plan — VLN Segment-Transition KV Eviction

## Context

The method (`GHOST_SCORE_MODE=vln_segment_transition`) keeps GHOST's skeleton — layer-wise
cosine budget allocation, special-token privilege (Eq. 7), training-free streaming KV — and
**replaces GHOST's scoring function** with four components (geometry, confidence, instruction,
transition), each z-scored over the candidate cache, with eviction moved to run *after* the
language projector so the instruction term can exist at all.

Budget ablations (600k / 800k / 900k / 1M / 1.2M) are already running. What does not exist is
any evidence for **which part of the method earns the result**. The instruction term — the
actual novelty relative to GHOST — has never been isolated, and `VLN_SCORE_PROBE` has never
been run. Current numbers cannot distinguish three very different stories:

1. the language signal helps,
2. the z-score normalization helps (and would help GHOST's own components equally),
3. nothing helps and the deltas are noise.

Group C and Group D exist to settle this. Everything else is supporting evidence.

Supersedes the Part C/D/E experiment matrix in `KV_EVICTION_REVIEW_PLAN.md`, whose cost model
is wrong by ~3× (it assumes ~3 h/arm and ~16.7 h full; measured is 6.8 h and **45.8 h**).

---

## 1. Protocol — fixed across every arm

Deviating from any of these invalidates the paired comparison.

| Setting | Value | Why |
|---|---|---|
| `MAX_STEPS` | `400` | The existing 360k-vs-900k comparison is confounded by a 200-vs-400 cap. Never mix. |
| `SCENE_IDS` | `none` | Full `val_unseen`, 1839 episodes, 11 scenes, ~233k steps |
| `CONFIG` | `config/vln_r2r_local.yaml` | |
| Checkpoint | `spatialstack_janus_vln_train-gate-scale-4B-loss-3` | |
| `SAVE_VIDEO` | `0` | Disk + time, no analytic value |
| bf16 K cache | always on | Proven bit-identical (`max|diff| = 0`), −33% KV |
| `VGGT_CAMERA_KV_RECENT` | `64` (default) | The one applied change that is *not* bit-identical — see G1 |

**Cost:** ~45.8 h/arm at the measured 706 ms/step. One arm per GPU: peak is 19.3 GB of
32.6 GB and the ~15.8 GB floor (Qwen3.5-4B + VGGT-1B + heads + activations) is
budget-independent, so two jobs never fit — not even at the smallest budget.

**Resume is automatic.** `src/evaluation.py:534-573` reads `result.json` from `OUTPUT_PATH`
and skips episodes already present, keyed on `scene_id + episode_id + episode_instruction`.
Point `OUTPUT_PATH` at an existing run to continue it. Back the file up first.

**Smoke-test every new arm** with `MAX_EPISODES=2` (`src/evaluation.py:560`) before committing
45.8 h. Confirm the startup banner shows the intended config.

**Report per arm:** SR, SPL, OS, NE, mean steps, `peak_vggt_kv_mb`, `peak_alloc_mb`,
`mean_step_ms`, `mean_vggt_ms`.

**Statistics:** paired on shared episodes against the reference arm, McNemar + bootstrap 95%
CI. At n=1839 the paired ΔSR CI is **≈ ±1.4**. Quote it. Most arms will land inside it, and an
arm that does is *not* evidence of equivalence — it is absence of evidence.

---

## 2. What already exists

| Episodes | Scenes | SR | SPL | KV MB | Run |
|---:|---:|---:|---:|---:|---|
| 1839 | 11 | 54.16 | 50.02 | 5273 | `zscore_fp32/budget_900000_steps_400` (fp32 K) |
| 849+ | 5 | 53.71 | 48.32 | 3516 | `bf16k/budget_900000_steps_400` **(running)** |
| 657 | 4 | 49.77 | 44.75 | 4687 | `zscore_fp32/budget_800000_steps_400` |
| 589 | 4 | 48.90 | 44.79 | 2109 | `zscore_fp32/budget_360000_steps_200` (200-cap, do not mix) |
| 344 | 3 | 54.07 | 48.35 | 5273 | `zscore_fp32/budget_900000_steps_300` (300-cap, do not mix) |
| 299 | 3 | 49.16 | 44.45 | 2109 | `ghost/budget_360000_steps_200` — **GHOST baseline, only 299 ep** |

Reference (no eviction): SR 0.5525 / SPL 0.4995 / OS 0.6145 / NE 5.2189 / KV 5767.6 MB /
alloc 24161.9 MB. Measured with fp32 K — multiply its KV by 2/3 to compare against bf16 arms.

Current standing vs that reference (1839 ep, paired): ΔSR −1.09, **ΔSPL +0.07**, at −12.8%
peak CUDA. The accuracy claim is *parity*, not a gain.

---

## 3. How each knob is driven

Verified against the code, not assumed.

**Config-only (no code change):**

| Knob | Mechanism |
|---|---|
| Score weights | `VLN_SEGMENT_TRANSITION_WEIGHTS_PATH` env → variant JSON in `configs/` |
| Total budget | `VGGT_TOTAL_BUDGET` env |
| GHOST baseline | `GHOST_SCORE_MODE=importance` |
| JanusVLN path | `USE_GHOST_KV_CACHE=0` |
| Probe | `VLN_SCORE_PROBE=1`, `VLN_SCORE_PROBE_DIR`, `_EVERY`, `_LAYERS` |
| Camera KV window | `VGGT_CAMERA_KV_START`, `VGGT_CAMERA_KV_RECENT` (0 = disable) |
| Episode limit | `MAX_EPISODES` (smoke tests only) |

**Validator constraints** (`_validate_vln_segment_transition_weights`, `vggt_encoder.py`):
`score_weights` must sum to 1.0 ±1e-6; likewise `geometry_weights` and the two `transition`
gate weights. **Leave-one-out arms must therefore renormalize the survivors**, not just zero
one entry.

Renormalizing is sound: `zscore_normalize` gives every component unit variance and `topk` is
scale-invariant, so scaling all weights by a constant cannot change the ranking. A renormalized
LOO arm means exactly "drop this component, keep the others in the same proportion".

**Hard-pinned by the validator — need code to vary:** `confidence.merge` must be `"min"`;
`normalization.candidate_terms` must be `"zscore"`.

**Not yet overridable:** `config.vggt_budget_proportions_path` has **no env override**. F5
needs a 1-line addition mirroring the `VLN_SEGMENT_TRANSITION_WEIGHTS_PATH` pattern in
`VGGTEncoder.__init__`.

---

## 4. Budget reference table

Per-layer budgets from `configs/kv_budget_proportions_cosine.json` (24 layers, min proportion
3.44%, max 7.18%). "special %" is the share of the *smallest* layer consumed by unevictable
special tokens at 400 steps (5/frame × 400 = 2000).

| Budget | KV bf16 (MiB) | smallest layer | largest layer | special % of smallest |
|---:|---:|---:|---:|---:|
| 1,200,000 | 4687.5 | 41,299 | 86,143 | 4.8% |
| 1,100,000 | 4296.9 | 37,857 | 78,965 | 5.3% |
| 1,000,000 | 3906.2 | 34,416 | 71,786 | 5.8% |
| 900,000 | 3515.6 | 30,974 | 64,608 | 6.5% |
| 800,000 | 3125.0 | 27,533 | 57,429 | 7.3% |
| 600,000 | 2343.8 | 20,650 | 43,072 | 9.7% |
| 360,000 | 1406.2 | 12,390 | 25,843 | 16.1% |
| 180,000 | 703.1 | 6,195 | 12,922 | **32.3%** |
| 120,000 | 468.8 | 4,130 | 8,614 | **48.4%** |
| 90,000 | 351.6 | 3,097 | 6,461 | **64.6%** |
| 45,000 | 175.8 | 1,549 | 3,230 | **129.1%** |

At 45k the smallest layer cannot hold its own special tokens. Below ~360k, a budget arm run
without F3/F4 measures the *privilege*, not the budget.

---

## 5. The arms

Groups are independent — assign one group per machine.

### Group A — Baselines (4 arms)

Without A2 there is no "we beat GHOST" claim at all.

| # | Arm | How | Code? |
|---|---|---|---|
| A1 | No-eviction, bf16 K | `USE_GHOST_KV_CACHE=0 VGGT_KV_START=0 VGGT_KV_RECENT=1000000` | no |
| A2 | **GHOST original @900k** | `GHOST_SCORE_MODE=importance VGGT_TOTAL_BUDGET=900000` | no |
| A3 | JanusVLN start+recent | `USE_GHOST_KV_CACHE=0` + script defaults `VGGT_KV_START=8 VGGT_KV_RECENT=56` | **blocked** |
| A4 | Random-score eviction | replace `final_score` with `randn` | small |
| A5 | Recency-only (`score = frame_id`) | one-line scorer swap | small |

**A1 needs the huge window on purpose — there is no "disable trim" switch.** On the non-GHOST
path `vggt_encoder.py:948` runs `self._kv_cache_trim(...)` unconditionally every step;
`StartRecentKVCache.__call__` is bypassed only by its early return
`if seq_len <= self.cache_size` (`vggt_encoder.py:42-43`), where
`cache_size = start_size + recent_size`. A genuine no-eviction arm therefore needs a window
larger than the longest sequence the run can reach: 400 frames × 1205 tokens/frame =
**482,000 tokens**, hence 1,000,000. Left at the defaults, A1 would silently run a 56-token
cache and would not be a no-eviction baseline at all.

**A3 is blocked by the same units bug.** `StartRecentKVCache` (`vggt_encoder.py:28-56`) slices
`dim=2` — the *token* dimension — while its log line and comments call the units *frames*. With
`VGGT_KV_START=8 VGGT_KV_RECENT=56` it retains **64 tokens, not 64 frames**. Fix the units
before running, or the baseline is meaningless. A1 and A3 differ only in these window values,
not in any flag.

**A5 matters more than it looks.** The method dropped GHOST's `w_temporal` entirely; there is
no recency or decay term anywhere. If plain recency matches the full scorer, the four-component
design is not earning its complexity.

### Group B — Budget frontier (5 arms, config-only)

Downward sweep to complement the 600k–1.2M sweep already running: **360k / 180k / 120k / 90k /
45k**. Only `VGGT_TOTAL_BUDGET` changes.

Prior evidence (589 shared episodes, 360k vs 900k): ΔSR +1.53, CI `[-1.53, +4.75]`, McNemar
z = 0.96 — **not significant**, and NE moved the wrong way. 2.5× the KV bought nothing
measurable. The interesting direction is down.

Pair every arm below 360k with F3 or F4 (see the table in §4).

### Group C — Score-component isolation (8 arms, config-only)

The core ablation. C1–C4 remove one component (renormalized); C5–C8 keep exactly one.

| Arm | geometry | confidence | instruction | transition |
|---|---:|---:|---:|---:|
| base | 0.30 | 0.20 | 0.30 | 0.20 |
| C1 − geometry | 0 | 0.285714 | 0.428572 | 0.285714 |
| C2 − confidence | 0.375 | 0 | 0.375 | 0.25 |
| C3 − instruction | 0.428572 | 0.285714 | 0 | 0.285714 |
| C4 − transition | 0.375 | 0.25 | 0.375 | 0 |
| C5 geometry only | 1.0 | 0 | 0 | 0 |
| C6 confidence only | 0 | 1.0 | 0 | 0 |
| C7 instruction only | 0 | 0 | 1.0 | 0 |
| C8 transition only | 0 | 0 | 0 | 1.0 |

Values checked against the validator. Naive 6-decimal rounding gives 0.999999 for C1/C3 and
**fails** — the residue is pushed into the largest weight. A zero weight is safe:
`compute_candidate_final_score` uses `add_(values, alpha=w)`, and `zscore_normalize` returns
zeros for a degenerate component rather than NaN.

C3 (necessity) and C7 (sufficiency) are the pair that isolates the novelty.

Known confound to report: `geometry_score` is a per-frame **scalar broadcast to all 1205
tokens** (`vggt_encoder.py`), so it has zero within-frame discriminative power despite carrying
the joint-largest weight. After z-scoring it acts as a whole-frame include/exclude gate. C1 and
C5 measure exactly that.

### Group D — Is the instruction term doing what we claim? (3 arms, small code changes)

Cheapest and most convincing evidence in this document.

| # | Arm | Purpose |
|---|---|---|
| D1 | Single pooled instruction embedding (no segmentation) | Isolates the segmentation, not just "some language signal" |
| D2 | **Wrong instruction** — another episode's text | If SR is unchanged, the instruction term is not grounding on the instruction |
| D3 | Shuffled word order | Weaker variant of D2 |

**D2 is the decisive control.** The whole claim is that scoring visual tokens against
instruction *segments* retains what the policy will need. If a randomly swapped instruction
performs the same, that claim fails regardless of what C3 shows. Run it early.

Implementation: `build_instruction_segment_state` (`vln_segment_transition.py`) is the single
entry point — D1 forces one span covering the whole string, D2/D3 substitute the text before
tokenization.

### Group E — Sub-component weights (7 arms, config-only)

| # | Change |
|---|---|
| E1 | `geometry_weights` = camera 1.0 / depth 0.0 |
| E2 | `geometry_weights` = camera 0.0 / depth 1.0 |
| E3 | transition gate = confidence 1.0 / instruction 0.0 |
| E4 | transition gate = confidence 0.0 / instruction 1.0 |
| E5–E7 | `transition.recent_window_size` ∈ {1, 2, 8} (default 4) |

### Group F — Design choices (8 arms, code changes)

| # | Arm | Note |
|---|---|---|
| F1 | Rank/percentile normalization | Scale-free, outlier-immune, and breaks the survivor-variance feedback loop |
| F2 | **GHOST-style `sigmoid` + `/max`** | Splits "new components" from "new normalization" |
| F3 | Special-token privilege capped to last N frames | Prerequisite for B below 360k |
| F4 | Privilege removed entirely | Upper bound on what the privilege costs |
| F5 | Uniform layer budget | Control for GHOST's cosine table |
| F6 | Recency / decay term added | `frame_id` is already stored; targets the long-episode collapse |
| F7 | `confidence` = depth only | Also removes a DPT forward per step |
| F8 | `confidence` = point only | |

**F2 is Tier 0.** Measured on plausible component distributions, GHOST's `sigmoid` + `/max`
gives `geometry` **73.6%** of the realized ranking influence and `instruction` **2.0%**, versus
the configured 0.30/0.30. (`/max` is applied after the sum, so it is a monotone rescale and
changes no ranking at all.) Under z-score the realized shares match the configured weights.
It is entirely possible the z-score change, not the language signal, is doing the work — F2
paired with C3 separates them.

**F5 note:** `configs/kv_budget_proportions_cosine.json` gives layer 14 about 2.1× the flat
layers, and the generating script is **absent from the repo**, so the table is not reproducible.
If uniform matches, the table is not earning its complexity. Needs the env-override one-liner.

**F7 caveat:** changes `confidence` from `min(depth, point)` to depth only, so it is a method
change, not just an efficiency win. `importance_eviction.py:208` does
`pc = meta.get("pts3d_conf") or meta.get("conf")` — `or` on a multi-element tensor raises, so
F7/F8 must **omit the key**, not store `None`.

### Group G — Engineering (1 arm)

| # | Arm |
|---|---|
| G1 | `VGGT_CAMERA_KV_RECENT=0` (trim off) vs `64` (default) |

The camera-KV trim is the only applied change that is not bit-identical: the pose scalar drifts
up to ~0.1% past frame 64, enough to flip a marginal eviction decision. Measured on 268 shared
episodes: **143/143 short (≤65 step) episodes identical, 50/125 long ones diverged.** It bounds
a leak (52.4 MB → 8.5 MB constant at 400 steps) but must be reported as an arm, not folded in
as a free engineering fix.

---

## 6. Instrumentation

Enable `VLN_SCORE_PROBE=1` on the reference arm and on C3, C7, F1, F2. It has **never been
run** — no `vln_score_probe/` output exists anywhere in the repo, so §"does the score select
the right tokens" is currently unanswerable.

```bash
VLN_SCORE_PROBE=1 VLN_SCORE_PROBE_DIR=<output>/probe
```

Cost: one device sync per logged step (default 1 in 10). Read with
`python -m qwen_vl.model.vggt.eviction.vln_score_probe <path>.jsonl`.

Fields that matter:

- `candidate_variance_proxy_share` — which component actually dominates, over time
- `unique_frames`, `age_quantiles` — is the cache collapsing onto a few frames?
- `special_fraction` — how much budget the privilege eats
- `distinct_score_levels` — tie collapse

**Hypothesis under test:** σ in `zscore_normalize` is computed over the *survivor* population,
which was selected to maximize `Σ w·z`. Selection truncates variance along the dominant
direction, shrinking its σ and *raising* its effective weight `w/σ` next step — positive
feedback toward single-component ossification. This predicts the observed SR decay with episode
length (70.4% at 25–50 steps → 52.3% at 75–100 → **10.7% at the 400 cap**, where the 261 capped
episodes are 14% of the set). If true, `candidate_variance_proxy_share` drifts toward one
component and `unique_frames` collapses. **Do not change the scoring (F1/F6/E) before this is
measured.**

---

## 7. Priority and cost

~36 arms × 45.8 h ≈ **1650 GPU-hours**. Tiered so the matrix can be cut without losing the paper.

**Tier 0 — without these there is no claim** (4 arms, ~183 h)

| # | Question it answers |
|---|---|
| A2 | Do we actually beat GHOST at full scale? (only 299 ep today) |
| C3 | Is the instruction term necessary? |
| D2 | Is the instruction term grounding on the instruction at all? |
| F2 | Is the gain the new components, or just the new normalization? |

**Tier 1 — reviewers will demand these** (~12 arms, ~550 h)
C1, C2, C4, C7 · A4, A5 · B down-sweep (360k/180k/120k/90k/45k) · F3

**Tier 2 — completeness** (~20 arms)
C5, C6, C8 · D1, D3 · Group E · F1, F4, F5, F6, F7, F8 · G1 · A1, A3

Run the probe alongside Tier 0 — it costs nothing extra and gates every Tier-2 scoring change.

---

## 8. Running an arm

```bash
cd /home/tripnv/project/Token_eviction/SpatialStack

VGGT_TOTAL_BUDGET=900000 \
MAX_STEPS=400 \
CONFIG=config/vln_r2r_local.yaml \
SCENE_IDS=none \
SAVE_VIDEO=0 \
VLN_SEGMENT_TRANSITION_WEIGHTS_PATH=configs/ablation/C3_no_instruction.json \
OUTPUT_PATH=evaluation_gate_scale_fix_eval/ablation/C3_no_instruction/ \
nohup bash scripts/evaluation/eval_janus_vln_scene_segment_transition_ghost.sh \
  > evaluation_gate_scale_fix_eval/ablation/C3_no_instruction.log 2>&1 &
```

Smoke first: prepend `MAX_EPISODES=2` and confirm the banner. Then relaunch full — resume will
keep the 2 episodes.

Summarize: `python scripts/evaluation/summarize_vln_results.py <OUTPUT_PATH>/result.json`

Suggested layout: variant weight files in `configs/ablation/<ARM_ID>_<name>.json`, outputs in
`evaluation_gate_scale_fix_eval/ablation/<ARM_ID>_<name>/`.

---

## 9. Reporting template

| Arm | n | SR | SPL | OS | NE | steps | KV MB | alloc MB | step ms | ΔSR vs ref [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ref (900k) | 1839 | | | | | | 3515.6 | 19314 | | — |
| A2 GHOST | | | | | | | | | | |
| C3 − instruction | | | | | | | | | | |
| D2 wrong instruction | | | | | | | | | | |
| F2 GHOST norm | | | | | | | | | | |

Always report paired ΔSR/ΔSPL with bootstrap CI **on shared episodes**, never a difference of
two independent means over different episode sets. Two arms that stopped at different episode
counts must be compared on the intersection only — several comparisons in this repo's history
were confounded exactly this way.

Also stratify the headline arms by episode length (25–50 / 75–100 / 150–175 / 225–250 / 400
cap). The 261 cap-hitting episodes sit at 10.7% SR and are 14% of the set; pulling them to even
30% is **+2.7 SR overall**, a larger prize than anything in the budget sweep.
