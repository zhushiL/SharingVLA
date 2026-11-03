#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

if [[ $# -lt 1 ]]; then
  echo "Usage: bash train.sh tasks/train_{action_policy,vla}.py configs/<policy>/<stage>.yaml [overrides]" >&2
  exit 2
fi
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
  else
    IFS=',' read -ra TRAIN_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE=${#TRAIN_GPU_IDS[@]}
  fi
fi
if (( NPROC_PER_NODE < 1 )); then
  echo "Training requires at least one visible CUDA GPU." >&2
  exit 2
fi

torchrun --nnodes="${NNODES:-1}" --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="${NODE_RANK:-0}" --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-62500}" "$@" 2>&1 | tee "${TRAIN_LOG_PATH:-log.txt}"
