# SpatialStack → GeoThinker fusion migration

Incremental migration of the **VLN streaming** SpatialStack fusion toward
GeoThinker's Spatial-Grounded Fusion (SGF). One axis per step, each behind a
flag defaulting to current behavior, each validated before the next.

Target checkpoint: `model_checkpoint/spatialstack_vln_fix_0.5_full`
(`geometry_fusion_layers=[0,1,2]`, `geometry_encoder_layers=[11,17,23]`,
`feature_fusion_method=deepstack_language_add`, `geometry_fusion_scale=0.5`,
`geometry_encoder_streaming=true`, `include_camera_token=false`).

## Debugging discipline (every step)

1. **One axis per step**, behind its own flag, **default OFF** → flag-off must
   reproduce the previous model exactly.
2. **No-op initialization**: zero-init the new output projection (or `tanh(gate=0)`)
   so at training step 0 the model is bit-identical to the prior baseline. Any
   divergence at init = wiring bug.
3. **Regression smoke after every step**: `smoke_forward_parity.py` with the new
   flag OFF must match the golden fingerprint. Then a short train + subset eval
   to measure the delta. Record in `ablation_log.md`. Keep the change only if it helps.

## What SpatialStack already has (reduces the work)

- **Cross-attention operator already coded**: `deepstack_language_cross_attn`
  (`feature_fusion.py:339-365` build, `:451-472` forward) with a per-frame
  reshape and sincos2d position embeds — but **guard-blocked** for Qwen3.5 in
  `modeling_qwen3_5.py:489-493`. Step 2 = lift the guard + wire `initialize_geometry_modules`.
- **sincos2d position embeds already plumbed** into `fusion_module(... vis_pos_embed, geo_pos_embed)`.

## What GeoThinker adds (to port)

- Frame-strict per-frame geometry (we currently broadcast the last frame).
- Importance gate (background suppression) — `qwen_interaction.py:104-110,219-226`.
- Spatial-distance bias, half the heads — `qwen_interaction.py:115-139,286-306`.
- `tanh(gate)` no-op residual.

## Step ladder

| Step | Axis | Flag (proposed) | Default | Notes |
|---|---|---|---|---|
| 0 | Safety net | — | — | flags scaffolding, baseline, this harness. NO model change. |
| 1 | Frame handling | `FUSION_FRAME_STRICT` | off (broadcast) | keep per-frame geometry; may also fix train/eval streaming mismatch |
| 2 | Operator | `feature_fusion_method=deepstack_language_cross_attn` | add | un-guard + wire; `tanh(gate)=0` init |
| 3 | Selectivity | `FUSION_IMPORTANCE_GATE` | off | port importance_net into cross-attn block |
| 4 | Locality | `FUSION_SPATIAL_BIAS` | off | port get_spatial_bias + bias_gate |
| 5 (opt) | Placement | `GEOMETRY_FUSION_LAYERS` | `0 1 2` | try first-¾ layers; big compute change, ablate last |

## Baseline (Step 0) — run on the GPU box

```bash
# 1. Forward-parity golden fingerprint (fast, single forward)
python scripts/dev/fusion_migration/smoke_forward_parity.py \
  --model-path model_checkpoint/spatialstack_vln_fix_0.5_full \
  --write-golden

# 2. Fast navigation baseline (subset via MAX_EPISODES)
MAX_EPISODES=20 CHECKPOINT=model_checkpoint/spatialstack_vln_fix_0.5_full \
  bash scripts/evaluation/eval_janus_vln.sh

# 3. Full baseline (val_unseen) — record SR / SPL / etc. into ablation_log.md
CHECKPOINT=model_checkpoint/spatialstack_vln_fix_0.5_full \
  bash scripts/evaluation/eval_janus_vln.sh
```

After each later step, rerun (1) with the new flag OFF (must match golden) and
(2)/(3) with it ON to measure the delta.
