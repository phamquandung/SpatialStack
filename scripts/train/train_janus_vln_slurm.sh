#!/bin/bash -e
#SBATCH --job-name=spatialstack-janus-vln
#SBATCH --output=logs/spatialstack_janus_vln_%j.log
#SBATCH --error=logs/spatialstack_janus_vln_%j.err
#SBATCH --nodelist=worker-0
#SBATCH --gpus=8
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=8192
#
#SBATCH --container-image=/mnt/data/vmo-ai-task/dungpq6/ubuntu22-cuda128-conda-janusvln-spatialstack.sqsh
#SBATCH --container-mounts=/mnt/data/:/mnt/data/,/home/dungpq6/Project:/home/dungpq6/Project

set -euo pipefail

source /home/dungpq6/anaconda3/etc/profile.d/conda.sh
conda activate spatialstack-qwen35

# SLURM may execute a copy under /var/spool/slurmd — use an explicit project root.
PROJECT_ROOT="${PROJECT_ROOT:-/home/dungpq6/Project/SpatialStack}"
cd "${PROJECT_ROOT}"
mkdir -p logs

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"

NUM_NODES="${NUM_NODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}}"

if [[ "${NUM_NODES}" -gt 1 ]]; then
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
    # Uncomment if cross-node NCCL hangs, e.g. eth0 or ib0:
    # export NCCL_SOCKET_IFNAME=eth0
fi

# --- VLN data (JanusVLN layout on shared storage) ---
export VLN_DATA_ROOT="${VLN_DATA_ROOT:-/mnt/data/vmo-ai-task/anhdh35/JanusVLN}"
export VLN_ANNOTATION="${VLN_ANNOTATION:-/mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json}"

# --- Model checkpoints ---
export VLN_TRAIN_MODE="${VLN_TRAIN_MODE:-train}"   # train | finetune
export MODEL_PATH="${MODEL_PATH:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/Qwen3.5-4B}"
export GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/VGGT-1B}"

# --- Training hyperparameters (passed through to train_janus_vln.sh) ---
export DATASETS="${DATASETS:-train_r2r_rxr%100}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-64}"
export LR="${LR:-2e-5}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/spatialstack_janus_vln_train}"
export CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache}"

# Optional debug (rank-0 dumps to ${OUTPUT_DIR}/debug_vln)
export VLN_DEBUG="${VLN_DEBUG:-}"
export VLN_DEBUG_SAVE_INTERVAL="${VLN_DEBUG_SAVE_INTERVAL:-100}"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

# Resolve master for torchrun (container may not have scontrol on PATH).
if [[ "${NUM_NODES}" -gt 1 ]]; then
    if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
        mapfile -t _slurm_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
        export MASTER_ADDR="${MASTER_ADDR:-${_slurm_nodes[0]}}"
    else
        export MASTER_ADDR="${MASTER_ADDR:-${SLURM_JOB_NODELIST%%,*}}"
        export MASTER_ADDR="${MASTER_ADDR//[\[\]]/}"
    fi
else
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_NODES=${NUM_NODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "VLN_TRAIN_MODE=${VLN_TRAIN_MODE}"
echo "VLN_DATA_ROOT=${VLN_DATA_ROOT}"
echo "VLN_ANNOTATION=${VLN_ANNOTATION}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "GEOMETRY_ENCODER_PATH=${GEOMETRY_ENCODER_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
echo "DATASETS=${DATASETS}"

if [[ "${NUM_NODES}" -gt 1 ]]; then
    # One task per node; train_janus_vln.sh reads SLURM_PROCID / SLURM_JOB_NUM_NODES for torchrun.
    srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
        bash scripts/train/train_janus_vln.sh
else
    if command -v srun >/dev/null 2>&1; then
        srun bash scripts/train/train_janus_vln.sh
    else
        bash scripts/train/train_janus_vln.sh
    fi
fi
