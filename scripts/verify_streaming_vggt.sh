#!/bin/bash
# Verify that SpatialStack includes the bundled streaming VGGT (KV-cache) implementation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGGREGATOR="$SCRIPT_DIR/../src/qwen_vl/model/vggt/models/aggregator.py"

if grep -q "use_cache" "$AGGREGATOR" && grep -q "past_key_values" "$AGGREGATOR"; then
    echo "OK: streaming VGGT is bundled in SpatialStack (aggregator supports KV cache)."
else
    echo "ERROR: streaming VGGT not found in $AGGREGATOR"
    exit 1
fi
