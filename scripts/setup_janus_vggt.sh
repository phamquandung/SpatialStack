#!/bin/bash
# Deprecated: streaming VGGT is now bundled inside SpatialStack.
# This script only verifies the installation — no JanusVLN repo is required.
echo "NOTE: setup_janus_vggt.sh is deprecated. Streaming VGGT is bundled in SpatialStack."
exec bash "$(dirname "$0")/verify_streaming_vggt.sh"
