# Fusion migration — ablation log

Record every run here. Fill metrics from `eval_janus_vln.sh` output. Keep a
change only if the eval delta is non-negative. Baseline = the released
`spatialstack_vln_fix_0.5_full` recipe: `deepstack_language_add`, last-frame
broadcast, `fusion_scale=0.5`, no gates.

## Flag reference (all default OFF unless noted)

Operator is chosen by `FEATURE_FUSION_METHOD`; sub-flags are env / config
(`--geometry_*`). Env overrides config.

| Flag (env) | Config key | Step | Applies to operator | Effect |
|---|---|---|---|---|
| `FUSION_FRAME_STRICT` | `geometry_frame_strict` | 1 | both | per-frame geometry vs last-frame broadcast (scripts default: **true**) |
| `FUSION_IMPORTANCE_GATE` | `geometry_importance_gate` | 2′ / 3 | both | suppress background geometry (add: ×gate; sgf: +log-imp on logits) |
| `FUSION_LEARNABLE_SCALE` | `geometry_learnable_scale` | 2″ | add | learnable per-layer scale (init = `fusion_scale`) |
| `FEATURE_FUSION_METHOD=deepstack_language_sgf` | `feature_fusion_method` | 2 | — | cross-attention operator (vs `deepstack_language_add`) |
| `FUSION_SPATIAL_BIAS` | `geometry_spatial_bias` | 4 | sgf | spatial-distance bias on half the heads (**exploratory — beyond GeoThinker's trained recipe**) |

Fixed for all runs: `geometry_fusion_layers=[0,1,2]`, `geometry_encoder_layers=[11,17,23]`,
`geometry_encoder_streaming=true`, `include_camera_token=false`, `fusion_scale=0.5`.

## Fingerprint (forward-parity golden)

`smoke_forward_parity.py`. Flag-OFF runs must match the golden; flag-ON runs are
expected to differ (behavior change), listed for reference.

| Date | Run | logits sum | logits mean | logits std | last argmax | top-5 ids | matches golden? |
|---|---|---|---|---|---|---|---|
| _tbd_ | Step 0 baseline (GOLDEN) | | | | | | (golden) |
| _tbd_ | add, all flags OFF (regression) | | | | | | expect ✅ |

## Init parity (step-0 training loss, flag ON, fresh weights == baseline)

All should hold: geometry contributes 0 at init (zero-init `geo_mlp` for add;
`tanh(gate)=0` for sgf). SGF block verified in isolation ✅ (test_sgf_block.py).

| Config | step-0 loss == baseline? | Notes |
|---|---|---|
| add + FRAME_STRICT | | zero-init geo_mlp |
| add + IMPORTANCE_GATE | | delta=0 regardless of gate |
| add + LEARNABLE_SCALE | | scale init 0.5, delta=0 |
| sgf (bare) | | tanh(gate)=0 |
| sgf + IMPORTANCE_GATE + SPATIAL_BIAS | | tanh(gate)=0 |

## Navigation eval

Record SR / SPL / OSR / NE. Run a `MAX_EPISODES=20` subset first, then full
`val_unseen`. One row per config; note the OUTPUT_DIR.

### Baseline

| Run | Episodes | SR ↑ | SPL ↑ | OSR ↑ | NE ↓ | OUTPUT_DIR / notes |
|---|---|---|---|---|---|---|
| baseline (add, broadcast, 0.5) | 20 | | | | | fast baseline |
| baseline | full | | | | | locked baseline |

### Additive path (`FEATURE_FUSION_METHOD=deepstack_language_add`)

| FRAME_STRICT | IMPORTANCE_GATE | LEARNABLE_SCALE | Episodes | SR ↑ | SPL ↑ | OSR ↑ | NE ↓ | OUTPUT_DIR / notes |
|---|---|---|---|---|---|---|---|---|
| true | false | false | | | | | | Step 1 |
| true | true | false | | | | | | Step 1 + 2′ |
| true | true | true | | | | | | Step 1 + 2′ + 2″ |

### SGF cross-attention path (`FEATURE_FUSION_METHOD=deepstack_language_sgf`)

| FRAME_STRICT | IMPORTANCE_GATE | SPATIAL_BIAS | Episodes | SR ↑ | SPL ↑ | OSR ↑ | NE ↓ | OUTPUT_DIR / notes |
|---|---|---|---|---|---|---|---|---|
| true | false | false | | | | | | Step 2 (SGF only) |
| true | true | false | | | | | | Step 2 + 3 — **GeoThinker-parity** (their trained recipe: cross-attn + importance, NO spatial bias) |
| true | true | true | | | | | | Step 2 + 3 + 4 — **exploratory, beyond GeoThinker** (spatial bias is coded in GeoThinker but never enabled in their training) |

> **GeoThinker's actual trained SGF** = `geo_cross_attn=True` + `geo_importance_gate=True` + `feature_fusion_method="zero"`.
> `geo_spatial_bias` has no CLI arg and is never set → `False`. So the spatial-distance bias, though present in
> `qwen_interaction.py`, is **not part of GeoThinker's released model**. Treat `FUSION_SPATIAL_BIAS=true` as a research
> extension, not a reproduction.

## Observations / decisions

- _(log conclusions here: which axis moved the metric, what to keep, what to drop)_