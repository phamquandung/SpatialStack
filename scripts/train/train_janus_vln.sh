#!/bin/bash
# Train or fine-tune SpatialStack (Qwen3.5) on JanusVLN VLN data.
# Upload only SpatialStack to the server — streaming VGGT and data-prep script are bundled.
#
# Modes (set VLN_TRAIN_MODE):
#   train    — JanusVLN-style: start from Qwen/Qwen3.5-4B + VGGT (default)
#   finetune — continue from Journey9ni/SpatialStack-Qwen3.5-4B
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPATIAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SPATIAL_ROOT"

# ======================
# Data paths (relative to SpatialStack root by default)
# ======================
export VLN_DATA_ROOT="${VLN_DATA_ROOT:-/mnt/data/vmo-ai-task/anhdh35/JanusVLN}"
export VLN_ANNOTATION="${VLN_ANNOTATION:-/mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json}"

if [[ "$VLN_ANNOTATION" != /* ]]; then
    ANNOTATION_FILE="$SPATIAL_ROOT/$VLN_ANNOTATION"
else
    ANNOTATION_FILE="$VLN_ANNOTATION"
fi

if [ ! -f "$ANNOTATION_FILE" ]; then
    echo "ERROR: Annotation not found: $ANNOTATION_FILE"
    echo ""
    echo "Prepare data on the server (no JanusVLN repo needed):"
    echo "  python scripts/data/create_janus_vln_data.py --data_root . --use_extra_data"
    echo ""
    echo "Or set VLN_ANNOTATION to your pre-built JSON path."
    exit 1
fi

bash scripts/verify_streaming_vggt.sh

# ======================
# Distributed Configuration
# ======================
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-22223}

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
fi

NODE_RANK=${NODE_RANK:-${SLURM_PROCID:-0}}
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a __CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES// /}"
    NPROC_PER_NODE=${#__CUDA_DEVICE_LIST[@]}
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
else
    NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi
[ "${NPROC_PER_NODE:-0}" -gt 0 ] || NPROC_PER_NODE=1
[ "${NNODES:-0}" -gt 0 ] || NNODES=1

WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export WORLD_SIZE NODE_RANK

# ======================
# Model paths (train vs finetune)
# ======================
VLN_TRAIN_MODE="${VLN_TRAIN_MODE:-train}"
VLN_TRAIN_MODE="${VLN_TRAIN_MODE,,}"

if [[ "$VLN_TRAIN_MODE" == "finetune" ]]; then
    MODEL_PATH="${MODEL_PATH:-Journey9ni/SpatialStack-Qwen3.5-4B}"
    OUTPUT_DIR="${OUTPUT_DIR:-./output/spatialstack_janus_vln_finetune}"
else
    if [[ "$VLN_TRAIN_MODE" != "train" ]]; then
        echo "WARNING: unknown VLN_TRAIN_MODE='$VLN_TRAIN_MODE', using train (Qwen3.5 base)"
    fi
    MODEL_PATH="${MODEL_PATH:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/Qwen3.5-4B}"
    OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/spatialstack_janus_vln_train}"
fi

GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/VGGT-1B}"
CACHE_DIR="${CACHE_DIR:-./cache}"
mkdir -p "$OUTPUT_DIR"

echo ">>>>> VLN_TRAIN_MODE: $VLN_TRAIN_MODE"
echo ">>>>> MODEL_PATH:      $MODEL_PATH"

# ======================
# Hyperparameters (VLN-tuned)
# ======================
LR="${LR:-2e-5}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS=$(( TOTAL_BATCH_SIZE / WORLD_SIZE ))
[ "$GRADIENT_ACCUMULATION_STEPS" -gt 0 ] || GRADIENT_ACCUMULATION_STEPS=1
echo ">>>>> VLN annotation: $ANNOTATION_FILE"
echo ">>>>> VLN data root:    $VLN_DATA_ROOT"
echo ">>>>> grad accum = $GRADIENT_ACCUMULATION_STEPS  (world_size=$WORLD_SIZE)"

DATASETS="${DATASETS:-train_r2r_rxr%100}"
USE_GEOMETRY_ENCODER="${USE_GEOMETRY_ENCODER:-true}"
GEOMETRY_ENCODER_STREAMING="${GEOMETRY_ENCODER_STREAMING:-true}"
FEATURE_FUSION_METHOD="${FEATURE_FUSION_METHOD:-deepstack_language_add}"
GEOMETRY_FUSION_LAYERS="${GEOMETRY_FUSION_LAYERS:-0 1 2}"
GEOMETRY_ENCODER_LAYERS="${GEOMETRY_ENCODER_LAYERS:-11 17 23}"
REFERENCE_FRAME="${REFERENCE_FRAME:-first}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
if [[ "${VLN_DEBUG:-}" == "1" || "${VLN_DEBUG,,}" == "true" ]]; then
    DATALOADER_NUM_WORKERS=0
    echo ">>>>> VLN_DEBUG=1: logging shapes + saving frames to ${OUTPUT_DIR}/debug_vln"
fi

train_args=(
    --model_name_or_path "$MODEL_PATH"
    --tune_mm_llm True
    --tune_mm_vision False
    --tune_mm_mlp True
    --dataset_use "$DATASETS"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --bf16
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --learning_rate "$LR"
    --mm_projector_lr 1e-5
    --vision_tower_lr 1e-6
    --optim adamw_torch
    --model_max_length 163840
    --data_flatten False
    --max_pixels $((576*28*28))
    --min_pixels $((16*28*28))
    --base_interval 2
    --video_max_frames 8
    --video_min_frames 4
    --video_max_frame_pixels $((1664*28*28))
    --video_min_frame_pixels $((256*28*28))
    --num_train_epochs 1
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --logging_steps 10
    --save_steps 1000
    --save_total_limit 5
    --deepspeed scripts/zero2_opt.json
    --gradient_checkpointing
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --group_by_modality_length true
    --seed 42
    --report_to none
    --use_geometry_encoder "$USE_GEOMETRY_ENCODER"
    --geometry_encoder_streaming "$GEOMETRY_ENCODER_STREAMING"
    --reference_frame "$REFERENCE_FRAME"
)

if [[ "${USE_GEOMETRY_ENCODER,,}" == "true" ]]; then
    train_args+=(
        --geometry_encoder_type vggt
        --geometry_encoder_path "$GEOMETRY_ENCODER_PATH"
        --feature_fusion_method "$FEATURE_FUSION_METHOD"
        --geometry_fusion_layers ${GEOMETRY_FUSION_LAYERS}
        --geometry_encoder_layers ${GEOMETRY_ENCODER_LAYERS}
    )
fi

if [[ "${VLN_DEBUG:-}" == "1" || "${VLN_DEBUG,,}" == "true" ]]; then
    train_args+=(
        --debug_vln True
        --debug_vln_save_dir "${OUTPUT_DIR}/debug_vln"
        --debug_vln_save_interval "${VLN_DEBUG_SAVE_INTERVAL:-100}"
        --debug_vln_save_geo_layers True
    )
    if [[ "${VLN_DEBUG_DEPTH:-}" == "1" || "${VLN_DEBUG_DEPTH,,}" == "true" ]]; then
        train_args+=( --debug_vln_save_depth True )
        echo ">>>>> VLN_DEBUG_DEPTH=1: will run VGGT DPT depth head (extra forward)"
    fi
fi

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    "${train_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
