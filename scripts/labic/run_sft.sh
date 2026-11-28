#!/usr/bin/env bash
# ==============================================================================
# OPERA — Safe Multi-User SFT-MT Launcher for LABIC (24 GB NVIDIA GPU)
# ==============================================================================
# Usage:
#   GPU_ID=0 bash scripts/labic/run_sft.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_DIR="${HOME}/exp"

GPU_ID="${GPU_ID:-}"
SFT_RUN_NAME="${SFT_RUN_NAME:-full_sft_gender_wxrate_multitask_alpha05_semi25_seed42_v5}"
SPLIT_DIR="${SPLIT_DIR:-data/sft_grpo_gender_wxrate_60_40_semi25_v5}"
PARTITION_MODE="${PARTITION_MODE:-semi_balanced}"
NEUTRAL_FRACTION="${NEUTRAL_FRACTION:-0.25}"
DATA_DIR="${DATA_DIR:-data/meld_16k_mono}"
SEED="${SEED:-42}"

if [[ -z "${GPU_ID}" ]]; then
  echo "[ERROR] Specify an idle GPU explicitly: GPU_ID=0 bash scripts/labic/run_sft.sh"
  exit 1
fi

GPU_INFO="$(nvidia-smi --id="${GPU_ID}" --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -z "${GPU_INFO}" ]]; then
  echo "[ERROR] Invalid GPU_ID=${GPU_ID}."
  exit 1
fi

IFS=',' read -r GPU_NAME MEM_USED MEM_FREE MEM_TOTAL <<< "${GPU_INFO}"
MEM_USED="$(echo "${MEM_USED}" | xargs)"
MEM_FREE="$(echo "${MEM_FREE}" | xargs)"

ACTIVE_PIDS="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs || true)"
if [[ -n "${ACTIVE_PIDS}" || "${MEM_USED}" -gt 1024 || "${MEM_FREE}" -lt 20000 ]]; then
  echo "[ABORT] GPU ${GPU_ID} (${GPU_NAME}) is busy (pids='${ACTIVE_PIDS}', used=${MEM_USED} MiB, free=${MEM_FREE} MiB)."
  echo "        Refusing to start so other experiments are NEVER affected."
  exit 1
fi

cd "${PROJECT_DIR}"
VENV_PY="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "[ERROR] Virtualenv not found at ${VENV_PY}. Run: bash scripts/labic/setup_env.sh"
  exit 1
fi

for req in \
  "${DATA_DIR}/emo_tag_clean/clean_train_set_with_cues.csv" \
  "${DATA_DIR}/emo_tag_clean/clean_val_set_with_cues.csv" \
  "${DATA_DIR}/emo_tag_clean/gender_acoustic_quartiles.json"; do
  if [[ ! -f "${req}" ]]; then
    echo "[ERROR] Missing required file: ${PROJECT_DIR}/${req}"
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${EXP_DIR}/.cache/huggingface"
export TORCH_HOME="${EXP_DIR}/.cache/torch"

mkdir -p "${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_DIR}/logs/${SFT_RUN_NAME}_${TIMESTAMP}.log"
PID_FILE="${PROJECT_DIR}/logs/${SFT_RUN_NAME}.pid"

nohup nice -n 10 "${VENV_PY}" -u -m src.pipeline sft \
  --project-dir "${PROJECT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --split-dir "${SPLIT_DIR}" \
  --partition-mode "${PARTITION_MODE}" \
  --neutral-fraction "${NEUTRAL_FRACTION}" \
  --sft-run-name "${SFT_RUN_NAME}" \
  --seed "${SEED}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &
JOB_PID=$!
echo "${JOB_PID}" > "${PID_FILE}"

echo "[LAUNCHED] SFT-MT (${SFT_RUN_NAME}) on GPU ${GPU_ID} | PID=${JOB_PID} | Log=${LOG_FILE}"
echo "  Monitor : tail -f ${LOG_FILE}"
echo "  Stop    : kill -TERM ${JOB_PID}"
