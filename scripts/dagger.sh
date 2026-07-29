#!/usr/bin/env bash
# Collect DAgger trajectories with a trained SpatialStack (Qwen3.5) checkpoint.
# Mirrors JanusVLN's scripts/dagger.sh, adapted to SpatialStack's eval launcher style.
# Output is drop-in compatible with scripts/data/create_janus_vln_data.py
# (--use_extra_data reads data/dagger_data/{R2R,RxR}/{images,annotations.json}).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- conda env (spatialstack-qwen35) ----
CONDA_ENV="${CONDA_ENV:-spatialstack-qwen35}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || echo "warn: could not activate conda env '${CONDA_ENV}'"
fi

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MASTER_PORT=$((RANDOM % 101 + 20000))
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
if [ "${NPROC_PER_NODE}" -lt 1 ]; then
  NPROC_PER_NODE=1
fi

# ---- model / geometry (SpatialStack checkpoint — NOT a JanusVLN one) ----
CHECKPOINT="${CHECKPOINT:-/mnt/samsung/Project/CoRL-ICRA/SpatialStack/model-checkpoint/spatialstack_janus_vln_train-gate-scale-4B-loss-3}"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-/mnt/samsung/Project/CoRL-ICRA/SpatialStack/model-checkpoint/VGGT-1B}"
export GEOMETRY_ENCODER_PATH
# Streaming VGGT KV window (must match how the checkpoint was trained/evaluated).
export VGGT_KV_START="${VGGT_KV_START:-8}"
export VGGT_KV_RECENT="${VGGT_KV_RECENT:-56}"
# Frame-strict per-frame geometry: leave unset to follow the checkpoint config.
# export FUSION_FRAME_STRICT=1

# ---- dataset selection ----
# R2R (default). For RxR, override the three DAGGER_* vars below.
DAGGER_DATASET="${DAGGER_DATASET:-R2R}"
DAGGER_DATA_PATH="${DAGGER_DATA_PATH:-/mnt/samsung/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}"
DAGGER_GT_ANNOTATIONS_PATH="${DAGGER_GT_ANNOTATIONS_PATH:-/mnt/samsung/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz}"

# RxR example:
#   DAGGER_DATASET=RxR
#   DAGGER_DATA_PATH=.../rxr/train/train_guide_en.json.gz
#   DAGGER_GT_ANNOTATIONS_PATH=.../rxr/train/train_guide_gt.json.gz

# ---- DAgger hyper-params ----
DAGGER_UPDATE_SIZE="${DAGGER_UPDATE_SIZE:-160000}"   # max episodes to collect (across ranks)
DAGGER_COMMIT_FREQ="${DAGGER_COMMIT_FREQ:-50}"        # flush annotations every N saved episodes
DAGGER_P="${DAGGER_P:-0}"                             # 0 = pure model rollout w/ expert correction
DAGGER_DATA_IT="${DAGGER_DATA_IT:-3}"                 # unused when DAGGER_P=0
CONFIG="${CONFIG:-config/vln_dagger.yaml}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"

DAGGER_OUTPUT_PATH="${DAGGER_OUTPUT_PATH:-data/dagger_data/${DAGGER_DATASET}}"
OUTPUT_PATH="${OUTPUT_PATH:-results/dagger/${DAGGER_DATASET}}"
mkdir -p "${DAGGER_OUTPUT_PATH}" "${OUTPUT_PATH}"

echo "CHECKPOINT:            ${CHECKPOINT}"
echo "GEOMETRY_ENCODER_PATH: ${GEOMETRY_ENCODER_PATH}"
echo "DAGGER_DATASET:        ${DAGGER_DATASET}"
echo "DAGGER_DATA_PATH:      ${DAGGER_DATA_PATH}"
echo "DAGGER_GT_PATH:        ${DAGGER_GT_ANNOTATIONS_PATH}"
echo "DAGGER_OUTPUT_PATH:    ${DAGGER_OUTPUT_PATH}"
echo "DAGGER_P:              ${DAGGER_P}"
echo "NPROC_PER_NODE:        ${NPROC_PER_NODE}"
echo "VGGT_KV:               start=${VGGT_KV_START} recent=${VGGT_KV_RECENT}"

extra_args=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  extra_args+=(--dagger_save_video --save_video)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" src/dagger.py \
  --model_path "${CHECKPOINT}" \
  --geometry_encoder_path "${GEOMETRY_ENCODER_PATH}" \
  --habitat_config_path "${CONFIG}" \
  --eval_split train \
  --output_path "${OUTPUT_PATH}" \
  --dagger_dataset "${DAGGER_DATASET}" \
  --dagger_data_path "${DAGGER_DATA_PATH}" \
  --dagger_gt_annotations_path "${DAGGER_GT_ANNOTATIONS_PATH}" \
  --dagger_output_path "${DAGGER_OUTPUT_PATH}" \
  --dagger_update_size "${DAGGER_UPDATE_SIZE}" \
  --dagger_commit_freq "${DAGGER_COMMIT_FREQ}" \
  --dagger_p "${DAGGER_P}" \
  --dagger_data_it "${DAGGER_DATA_IT}" \
  "${extra_args[@]}"
