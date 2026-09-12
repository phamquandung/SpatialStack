#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPATIALSTACK_ROOT="${SPATIALSTACK_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CONFIG_FILE="${SCRIPT_DIR}/../config/inference.yaml"

MODEL_PATH="${MODEL_PATH:-/storage/guest01/vinhld8/SpatialStack/checkpoints/spatialstack_janus_vln_train-gate-scale-4B-loss-3}"
GEOMETRY_ENCODER_PATH="${GEOMETRY_ENCODER_PATH:-/storage/guest01/vinhld8/SpatialStack/checkpoints/VGGT-1B}"

export MODEL_PATH GEOMETRY_ENCODER_PATH

export PYTHONPATH="${SPATIALSTACK_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# VGGT temporal KV cache
export VGGT_KV_START="${VGGT_KV_START:-8}"
export VGGT_KV_RECENT="${VGGT_KV_RECENT:-48}"

# Cache geometry sau geo_ln/MLP/gate/scale
export VLN_PROJECTED_GEOMETRY_CACHE="${VLN_PROJECTED_GEOMETRY_CACHE:-1}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

echo "MODEL_PATH: ${MODEL_PATH}"
echo "GEOMETRY_ENCODER_PATH: ${GEOMETRY_ENCODER_PATH}"
echo "VGGT_KV: start=${VGGT_KV_START} recent=${VGGT_KV_RECENT}"
echo "VLN_PROJECTED_GEOMETRY_CACHE: ${VLN_PROJECTED_GEOMETRY_CACHE}"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dagger

# Source ROS 2 workspace
source "${SPATIALSTACK_ROOT}/ros2_ws/install/setup.bash"

# --params-file first so any -p overrides passed in "$@" win over the file.
exec ros2 run ros2_inference_model inference_node.py --ros-args --params-file "${CONFIG_FILE}" "$@"
