#!/usr/bin/env bash
# Watch a training OUTPUT_DIR and run a cheap rollout eval on each checkpoint-*,
# so you can see SR / voluntary-stop climb DURING training and stop early.
#
# The whole point: the LM loss saturates in ~100 steps but rollout SR is what we
# care about. Eval intermediate checkpoints and kill training once SR plateaus.
#
# Usage:
#   OUTPUT_DIR=./output/spatialstack_vln_fix \
#   GEOMETRY_ENCODER_PATH=model-checkpoint/VGGT-1B \
#   MAX_EPISODES=6 SCENE_IDS=EU6Fwq7SyZv LOOP=1 INTERVAL=900 \
#   bash scripts/evaluation/watch_eval_checkpoints.sh
#
# Env:
#   OUTPUT_DIR   training dir containing checkpoint-*           (required)
#   MAX_EPISODES episodes per checkpoint (cheap signal)         (default 6)
#   SCENE_IDS    scene(s) to eval                               (default EU6Fwq7SyZv)
#   EVAL_SPLIT                                                  (default val_unseen)
#   LOOP         1 = keep polling for new checkpoints           (default 0)
#   INTERVAL     seconds between polls when LOOP=1              (default 900)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the training dir with checkpoint-*}"
MAX_EPISODES="${MAX_EPISODES:-6}"
SCENE_IDS="${SCENE_IDS:-EU6Fwq7SyZv}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
LOOP="${LOOP:-0}"
INTERVAL="${INTERVAL:-900}"
EVAL_ROOT="${OUTPUT_DIR}/eval_ckpts"
SUMMARY="${EVAL_ROOT}/summary.tsv"
mkdir -p "${EVAL_ROOT}"
[ -f "${SUMMARY}" ] || printf "checkpoint\tstep\tn\tSR\tOSR\tNE\tmeanSteps\tvolStop\n" > "${SUMMARY}"

eval_one() {
  local ckpt="$1" name step out
  name="$(basename "${ckpt}")"
  step="${name#checkpoint-}"
  out="${EVAL_ROOT}/${name}"
  if grep -q "^${name}	" "${SUMMARY}" 2>/dev/null; then return; fi
  echo ">>> evaluating ${name} (MAX_EPISODES=${MAX_EPISODES})"
  CHECKPOINT="${ckpt}" SCENE_IDS="${SCENE_IDS}" EVAL_SPLIT="${EVAL_SPLIT}" \
    MAX_EPISODES="${MAX_EPISODES}" OUTPUT_PATH="${out}/${SCENE_IDS}" \
    bash scripts/evaluation/eval_janus_vln_scene.sh || { echo "  (eval failed for ${name})"; return; }
  python3 - "$out/$SCENE_IDS/result.json" "$name" "$step" "$MAX_EPISODES" >> "${SUMMARY}" <<'PY'
import json, os, sys
p, name, step, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
rows = [r for r in rows if 'success' in r][:n]
if not rows:
    print(f"{name}\t{step}\t0\tNA\tNA\tNA\tNA\tNA"); sys.exit()
k=len(rows)
sr=sum(r['success'] for r in rows)/k; osr=sum(r['os'] for r in rows)/k
ne=sum(r['ne'] for r in rows)/k; ms=sum(r['steps'] for r in rows)/k
vs=sum(1 for r in rows if r['steps']<400)
print(f"{name}\t{step}\t{k}\t{sr:.2f}\t{osr:.2f}\t{ne:.2f}\t{ms:.0f}\t{vs}/{k}")
PY
  echo "----- summary so far -----"; column -t "${SUMMARY}"
}

scan() {
  # numeric sort by step
  for ckpt in $(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -d "${ckpt}" ] && eval_one "${ckpt}"
  done
}

scan
while [ "${LOOP}" = "1" ]; do
  echo "... waiting ${INTERVAL}s for new checkpoints (Ctrl-C to stop)"
  sleep "${INTERVAL}"
  scan
done
echo "DONE. Full table:"; column -t "${SUMMARY}"