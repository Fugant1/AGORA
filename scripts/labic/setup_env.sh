#!/usr/bin/env bash
# ==============================================================================
# AGORA — Safe One-Time Environment Setup for LABIC (ICMC-USP)
# ==============================================================================
# Safety guarantees:
#   - Never uses sudo or touches system packages.
#   - Never modifies or stops any other user's processes.
#   - Redirects ALL uv, HuggingFace, PyTorch, and vLLM caches into ~/exp/.cache
#     (shared & persistent across LABIC machines; keeps local /home clean).
#   - Idempotent: safe to re-run without duplicating ~/.bashrc entries.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_DIR="${HOME}/exp"

if [[ ! -d "${EXP_DIR}" ]]; then
  echo "[ERROR] Expected LABIC shared directory '${EXP_DIR}' (/exp/${USER}) not found."
  echo "        Verify you are logged into a LABIC machine and ~/exp exists."
  exit 1
fi

if [[ "${PROJECT_DIR}" != "${EXP_DIR}"* && "$(readlink -f "${PROJECT_DIR}")" != "$(readlink -f "${EXP_DIR}")"* ]]; then
  echo "[WARN] Project directory '${PROJECT_DIR}' is outside '${EXP_DIR}'."
  echo "       LABIC recommends keeping projects inside ~/exp/projetos/AGORA for persistence."
fi

# 1. Configure persistent caches in ~/exp (idempotent ~/.bashrc block)
CACHE_BLOCK_START="# >>> AGORA LABIC CACHES >>>"
CACHE_BLOCK_END="# <<< AGORA LABIC CACHES <<<"
if ! grep -Fq "${CACHE_BLOCK_START}" "${HOME}/.bashrc" 2>/dev/null; then
  cat << EOF >> "${HOME}/.bashrc"

${CACHE_BLOCK_START}
export EXP_DIR="\${HOME}/exp"
export UV_PYTHON_INSTALL_DIR="\${EXP_DIR}/.local/share/uv/python"
export UV_CACHE_DIR="\${EXP_DIR}/.cache/uv"
export UV_TOOL_DIR="\${EXP_DIR}/.local/share/uv/tools"
export HF_HOME="\${EXP_DIR}/.cache/huggingface"
export TORCH_HOME="\${EXP_DIR}/.cache/torch"
export VLLM_CACHE_ROOT="\${EXP_DIR}/.cache/vllm"
${CACHE_BLOCK_END}
EOF
  echo "[OK] Added persistent ~/exp cache exports to ~/.bashrc"
else
  echo "[OK] ~/.bashrc already has AGORA LABIC cache exports"
fi

export UV_PYTHON_INSTALL_DIR="${EXP_DIR}/.local/share/uv/python"
export UV_CACHE_DIR="${EXP_DIR}/.cache/uv"
export UV_TOOL_DIR="${EXP_DIR}/.local/share/uv/tools"
export HF_HOME="${EXP_DIR}/.cache/huggingface"
export TORCH_HOME="${EXP_DIR}/.cache/torch"
export VLLM_CACHE_ROOT="${EXP_DIR}/.cache/vllm"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/data" "${PROJECT_DIR}/runs"

cd "${PROJECT_DIR}"

# 2. Create isolated virtual environment inside the project directory (/exp)
VENV_DIR="${PROJECT_DIR}/.venv"
if command -v uv >/dev/null 2>&1; then
  echo "[INFO] Creating/using virtual environment with uv at ${VENV_DIR}..."
  if [[ ! -d "${VENV_DIR}" ]]; then
    uv venv "${VENV_DIR}" --python 3.11
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  echo "[INFO] Installing PyTorch (CUDA 12.4) and AGORA + vLLM dependencies..."
  uv pip install torch --index-url https://download.pytorch.org/whl/cu124
  uv pip install \
    "transformers==5.20.2" "trl==1.15.0" vllm peft accelerate bitsandbytes \
    datasets soundfile pandas scipy scikit-learn matplotlib seaborn tqdm pytest gdown
else
  echo "[INFO] 'uv' not found; creating/using standard python3 venv at ${VENV_DIR}..."
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  pip install \
    "transformers==5.20.2" "trl==1.15.0" vllm peft accelerate bitsandbytes \
    datasets soundfile pandas scipy scikit-learn matplotlib seaborn tqdm pytest gdown
fi

# 3. Verify CPU mathematical parity tests (zero GPU usage)
echo "[INFO] Running AGORA parity self-check (CPU only)..."
pytest -q tests/test_src_parity.py

echo ""
echo "================================================================================"
echo "[SUCCESS] LABIC environment is ready at ${VENV_DIR}"
echo "Next steps:"
echo "  1. Check free GPUs:     nvidia-smi"
echo "  2. Launch GRPO safely:  GPU_ID=<FREE_GPU> bash scripts/labic/run_grpo.sh"
echo "================================================================================"
