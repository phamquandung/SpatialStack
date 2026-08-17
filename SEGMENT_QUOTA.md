# A2 — Segment-Coverage Quota

**Status:** implemented, **not yet run**. Default is OFF (`fraction = 0.0`), so behavior is
unchanged until explicitly enabled. Nothing was executed during implementation — a training run
was occupying 31.5 / 32.6 GB of GPU at the time.

Tracking document for the change: the problem, the design and why this design over the
alternatives, the exact code touched, and how to enable and verify it.

---

## 1. The problem

The method's stated advantage over pooling the whole instruction into one embedding is that a
token survives if it matches **any** clause of the instruction — so a landmark named in clause 1
stays useful after the agent has moved on to clause 3.

**Eviction does not enforce that.** `compute_instruction_segment_relevance` computes
`best_segment_id` for every token, stores it in `VLNGhostTokenMetadata`, and then it is read
**only by the score probe** — never by the eviction decision. Selection is a single global
`torch.topk` over `final_score`:

```python
keep = torch.topk(candidate_meta.final_score, k=layer_budget, largest=True, sorted=False)
```

A global top-K can legitimately drop *every* token belonging to segment 3 if segments 1 and 2
happen to score higher, and once dropped they never come back. The longer the episode, the more
chances to lose a segment outright — which matches the measured collapse:

| steps | n | SR% |
|---:|---:|---:|
| 25–50 | 533 | 70.4 |
| 75–100 | 216 | 52.3 |
| 150–175 | 49 | 36.7 |
| 400 (cap) | 261 | **10.7** |

Instruction segmentation is real and non-trivial: running `split_instruction_with_spans` over
all 1,839 `val_unseen` instructions gives a **median of 4 segments**, and **96.5% of episodes
have ≥ 2**.

**This premise is a hypothesis, not a measurement.** The probe's `retained_per_segment` field
tests it directly and has never been run. If coverage turns out to already stay balanced, this
change only costs flexibility and should be reverted.

## 2. The fix

Give each instruction segment a guaranteed floor of the layer budget, then allocate the
remainder globally by score as before.

```
q = floor(patch_budget × fraction / S)        # slots reserved per segment
patch_budget = max(0, layer_budget − n_special)
```

With `fraction ≤ 1`, `q·S ≤ patch_budget`, so the reserved portion can never overflow the
budget. `fraction = 0.5` means half the budget is distributed evenly as a per-segment floor and
half is still competitive.

## 3. Why implemented as a score bonus rather than two-stage selection

The obvious implementation is: take top-`q` within each segment, then fill the remaining slots
globally. That requires knowing how many slots the quota actually consumed — a
`.sum().item()` — which is a **device sync, 24 per frame** on top of the ~120/frame the scorer
already incurs.

Instead, the quota is applied as an additive bonus to the scores of the quota winners, and the
existing single global `topk` is left untouched:

```
bonus = (max(patch_final) − min(patch_final)) + 1        # kept as a tensor, never .item()
patch_final[in_quota] += bonus
```

Because `bonus > (max − min)`, **every** quota token outranks **every** non-quota token, so a
single global `topk(layer_budget)` selects exactly "all quota winners first, then the best of
the rest" — the intended semantics, by construction, with no extra sync and no set arithmetic.

Three further properties this buys:

- **The special-token invariant is preserved for free.** The bonus is applied to patch scores
  *before* specials are set to `patch_final.max() + boost`, so specials remain strictly above
  every patch and `test_special_tokens_score_above_every_patch` still holds.
- **Graceful degradation.** If the quota ever did overflow the budget, `topk` falls back to
  ranking quota tokens among themselves by their original score, since the bonus is uniform.
- **Minimal diff.** The prune loop keeps its single `topk`; only two arguments are added.

### Rank-within-segment, without a sync

```python
order   = argsort(patch_final, descending=True)          # global score order
seg_ord = best_segment_id[patch][order]                  # segment at each position
onehot  = one_hot(seg_ord, S).to(int32)                  # [P, S]
rank    = onehot.cumsum(0).gather(1, seg_ord[:, None]).squeeze(1) - 1
in_quota_ordered = rank < q
```

`rank` is each token's position **within its own segment**, in descending score order. Taking
`rank < q` is exactly "top-q of this segment", and it handles segments holding fewer than `q`
tokens automatically — the rank simply never reaches `q`. Entirely tensor-side.

Memory: `[P, S]` int32. Worst case P = 64,607 (largest layer at budget 900k) and S = 13 (largest
instruction observed) → 3.4 MB, allocated and freed once per layer.

## 4. Code changed

| File | Change |
|---|---|
| `src/qwen_vl/model/vggt/eviction/vln_segment_transition.py` | New `_segment_quota_bonus()` helper; `compute_candidate_final_score` gains `segment_quota_fraction`, `num_segments`, `patch_budget` as **keyword args with defaults**, so all existing 3-positional-arg call sites keep working |
| `src/qwen_vl/model/geometry_encoders/vggt_encoder.py` | Read `segment_quota.fraction` from config with `VLN_SEGMENT_QUOTA_FRACTION` env override; pass quota args into the prune loop's `compute_candidate_final_score` call; validate the new config section |
| `configs/vln_segment_transition_weights.json` | New optional `segment_quota` section, `fraction: 0.0` → **OFF by default** |
| `configs/ablation/A2_segment_quota.json` | New variant with `fraction: 0.5` to enable the arm |

Not touched: the prune loop's `topk`/`index_select`, `concat_metadata`/`gather_metadata`, the
metadata layout, the four score components, and the z-score normalization.

## 5. Degenerate cases, all guarded

| Case | Behavior |
|---|---|
| `fraction == 0` (default) | Skipped entirely — bit-identical to before |
| `S <= 1` | Skipped — a single segment means the quota is the global ranking |
| `q <= 0` (budget too small for S segments) | Skipped |
| `best_segment_id is None` | Skipped |
| segment holds fewer than `q` tokens | Handled by construction; unused slots flow to the global fill |
| `patch_budget <= 0` (specials alone exceed budget) | Skipped |

## 6. How to enable

```bash
VLN_SEGMENT_QUOTA_FRACTION=0.5 ...
# or
VLN_SEGMENT_TRANSITION_WEIGHTS_PATH=configs/ablation/A2_segment_quota.json ...
```

Default without either: unchanged behavior.

## 7. Verification — not yet done

**Nothing below has been executed.** No test was run and no evaluation was launched, because the
GPU was at 31.5 / 32.6 GB with an active run.

Before trusting this change:

1. `python -m unittest discover tests` — the 6 existing tests must stay green **unchanged**.
   They call `compute_candidate_final_score(metadata, weights, 0.3)` with three positional
   arguments and `best_segment_id` all `-1`; with the default `fraction = 0.0` the new path is
   skipped, so they should be unaffected.
2. Smoke test with `MAX_EPISODES=2` and `VLN_SEGMENT_QUOTA_FRACTION=0.5`, confirming no crash
   and no shape assertion in the prune loop.
3. Confirm the quota is actually binding: run with `VLN_SCORE_PROBE=1` and check
   `retained_per_segment` is more evenly distributed than with `fraction = 0.0`. If it is
   already even at 0.0, **the premise in §1 is wrong and this should be reverted.**
4. Evaluate on the **length-stratified** SR table, not aggregate SR — the hypothesis is
   specifically about long episodes, and aggregate SR is dominated by short ones that already
   work. A gain confined to the 400-cap bucket would be worth +2.7 SR overall yet barely move
   the aggregate.

## 8. Open question worth measuring

`fraction` is untuned. `0.5` is a guess, not a derived value. Once the probe confirms the
premise, sweep it — `0.25 / 0.5 / 0.75` — since too high starves the competitive portion of the
budget and too low fails to guarantee coverage.
