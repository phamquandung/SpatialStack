# Fusion migration — ablation log

Record every step here. Fill metrics from `eval_janus_vln.sh` output. Keep a
step only if the eval delta is non-negative.

## Fingerprint (forward-parity golden)

| Date | Model / step | logits sum | logits mean | logits std | last-token argmax | top-5 ids | matches golden? |
|---|---|---|---|---|---|---|---|
| _tbd_ | Step 0 baseline (golden) | | | | | | (golden) |

## Navigation eval

| Step | Flag config | Episodes | SR ↑ | SPL ↑ | OSR ↑ | NE ↓ | Notes |
|---|---|---|---|---|---|---|---|
| 0 | baseline (deepstack_language_add, broadcast, scale 0.5) | 20 (subset) | | | | | fast baseline |
| 0 | baseline | full val_unseen | | | | | locked baseline |
| 1 | + FUSION_FRAME_STRICT | | | | | | |
| 2 | + deepstack_language_cross_attn | | | | | | |
| 3 | + FUSION_IMPORTANCE_GATE | | | | | | |
| 4 | + FUSION_SPATIAL_BIAS | | | | | | |

## Init parity checks (loss/logits at training step 0, flag ON, fresh weights)

| Step | Flag ON, fresh init == baseline? | Notes |
|---|---|---|
| 1 | | should be exact (frame-strict + zero-MLP) |
| 2 | | should be exact via tanh(gate)=0 |
| 3 | | importance≈1 at init ≈ no-op |
| 4 | | bias-gate≈0 at init ≈ no-op |
