#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1

ENV_NAME="shilinvla"
RECREATE=0
RESUME=0
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"

usage() {
  cat <<'USAGE'
Usage: bash create_env.sh [--env-name NAME] [--recreate] [--resume] [--flash-attn-wheel PATH]

Creates a clean Python 3.12 conda environment for xense0vla training.
If --flash-attn-wheel or FLASH_ATTN_WHEEL is provided, flash-attn is installed
from that wheel. Otherwise flash-attn==2.8.3 is installed from pip.
Use --resume to continue installing into an existing environment.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:?--env-name requires a value}"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --flash-attn-wheel)
      FLASH_ATTN_WHEEL="${2:?--flash-attn-wheel requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

CONDA_BASE="$(conda info --base)"
eval "$(conda shell.bash hook)"

ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
ENV_EXISTS=0
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  ENV_EXISTS=1
fi

if [[ "${ENV_EXISTS}" == "1" || -d "${ENV_PREFIX}" ]]; then
  if [[ "${RECREATE}" == "1" ]]; then
    if [[ "${ENV_EXISTS}" == "1" ]]; then
      conda env remove -n "${ENV_NAME}" -y
    fi
    if [[ -d "${ENV_PREFIX}" ]]; then
      case "${ENV_PREFIX}" in
        "${CONDA_BASE}/envs/"*) rm -rf "${ENV_PREFIX}" ;;
        *) echo "Refusing to remove unexpected env prefix: ${ENV_PREFIX}" >&2; exit 1 ;;
      esac
    fi
    ENV_EXISTS=0
  elif [[ "${RESUME}" == "1" ]]; then
    echo "Resuming install in existing conda env: ${ENV_NAME}"
  else
    echo "Conda env already exists: ${ENV_NAME}" >&2
    echo "Pass --resume to continue installing into it, or --recreate to remove and rebuild it." >&2
    exit 1
  fi
fi

if [[ "${RESUME}" != "1" || "${ENV_EXISTS}" != "1" ]]; then
  conda create -n "${ENV_NAME}" python=3.12 pip -y
fi
conda activate "${ENV_NAME}"

# TorchCodec dynamically links against FFmpeg's shared libraries. Pin FFmpeg 7
# because the torchcodec 0.6.0 wheels below provide adapters for FFmpeg 4-7.
conda install -n "${ENV_NAME}" -c conda-forge ffmpeg=7 -y

python -m pip install -U pip setuptools wheel

assert_torch_stack() {
  python - <<'PY'
import torch

expected = "2.8.0"
actual = torch.__version__.split("+", 1)[0]
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert actual == expected, f"torch version changed: expected {expected}, got {torch.__version__}"
assert torch.cuda.is_available(), "torch CUDA is not available"
PY
}

assert_media_stack() {
  python - <<'PY'
import shutil
import subprocess

import torchcodec
from torchcodec.decoders import VideoDecoder

ffmpeg = shutil.which("ffmpeg")
assert ffmpeg is not None, "ffmpeg executable is not available in the environment"
ffmpeg_version = subprocess.check_output(
    [ffmpeg, "-version"], text=True
).splitlines()[0]
print(ffmpeg_version)
print("torchcodec", torchcodec.__version__, "VideoDecoder", VideoDecoder.__name__)
PY
}

python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  torchdata==0.11.0 torchcodec==0.6.0
assert_torch_stack
assert_media_stack

python -m pip install -r "${REPO_ROOT}/requirements.txt"
assert_torch_stack

python -m pip install numpydantic==1.9.0 --no-deps
assert_torch_stack

if [[ -n "${FLASH_ATTN_WHEEL}" ]]; then
  if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    echo "flash-attn wheel not found: ${FLASH_ATTN_WHEEL}" >&2
    exit 1
  fi
  python -m pip install --no-deps "${FLASH_ATTN_WHEEL}"
else
  python -m pip install --no-build-isolation flash-attn==2.8.3
fi
assert_torch_stack
python - <<PY
import flash_attn
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
PY

# Qwen3.5's 18 Gated DeltaNet layers have a correct PyTorch fallback, but the
# CUDA kernels are strongly recommended for practical training throughput.
# Transformers 5.14.1 imports the pre-0.3 ``fla.modules`` API; FLA 0.3+
# installs successfully but removes that API and makes Qwen3.5 fail to import.
if python -m pip install --no-build-isolation --no-deps \
    causal-conv1d==1.6.2.post1 flash-linear-attention==0.2.2 \
  && python - <<'PY'
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

assert all(
    (
        causal_conv1d_fn,
        causal_conv1d_update,
        FusedRMSNormGated,
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
)
PY
then
  python - <<'PY'
import causal_conv1d
import fla
print("Qwen3.5 linear-attention CUDA kernels are available")
PY
else
  echo "[WARN] Qwen3.5 linear-attention CUDA kernels failed to build; using the slower PyTorch fallback." >&2
  python -m pip uninstall -y causal-conv1d flash-linear-attention >/dev/null 2>&1 || true
fi
assert_torch_stack

# 0.4.4 supplies the generic subtask interface and supports our Torch 2.8 stack.
# Install only its dataset API: its Hub <0.36 metadata conflicts with the Hub
# >=1.5 required by Qwen3.5/Transformers 5. Keep the tested Qwen stack intact.
python -m pip install --no-deps lerobot==0.4.4
assert_torch_stack
python - <<'PY'
from importlib.metadata import version

import datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import load_subtasks

assert version("lerobot") == "0.4.4"
assert datasets.__version__ == "4.4.1"
print("LeRobot dataset/subtask API ready", LeRobotDataset.__name__, LeRobotDatasetMetadata.__name__)
PY

python -m pip install -e "${REPO_ROOT}" --no-deps
assert_torch_stack

if ! python -m pip check; then
  echo "[WARN] pip check reported dependency metadata issues." >&2
  echo "[WARN] lerobot is installed with --no-deps intentionally to preserve training pins." >&2
fi

assert_media_stack
echo "Environment ready: ${ENV_NAME}"
