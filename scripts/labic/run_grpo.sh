#!/usr/bin/env bash
# ==============================================================================
# OPERA — Safe Multi-User GRPO v3 (vLLM) Launcher for LABIC (24 GB NVIDIA GPU)
# ==============================================================================
# Usage:
#   GPU_ID=0 bash scripts/labic/run_grpo.sh
#
# Optional overrides:
#   SFT_RUN_NAME=full_sft_gender_wxrate_multitask_alpha05_semi25_seed42_v5
#   GRPO_RUN_NAME=grpo_v3_hybrid_invfreq_4cue_mt_sft_semi25_seed42_cuekl005_v5
#   PARTITION_MODE=semi_balanced  NEUTRAL_FRACTION=0.25
#   BETA=0.05  ALPHA=1.0  R1_ALPHA=0.5  R2_WEIGHT=1.0
#   BATCH_SIZE=4  STEPS_PER_GEN=8  GRAD_ACCUM=8
#   USE_VLLM=1  VLLM_QUANT=bitsandbytes  VLLM_GPU_UTIL=0.35
#
# Safety guarantees (never disrupts other lab users):
#   1. Requires explicit GPU_ID and ABORTS if ANY process is on that GPU or
#      if free VRAM < 20,000 MiB (or used VRAM > 1,024 MiB).
#   2. Isolates execution strictly to CUDA_VISIBLE_DEVICES=$GPU_ID.
#   3. Caps CPU threads (OMP/MKL=4) and runs with `nice -n 10`.
#   4. Uses a user- & GPU-scoped /dev/shm directory for vLLM LoRA sync.
#   5. Runs under `nohup` so closing SSH/Guacamole never kills your run.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_DIR="${HOME}/exp"

# --- Configurable experiment parameters ---
GPU_ID="${GPU_ID:-}"
SFT_RUN_NAME="${SFT_RUN_NAME:-full_sft_gender_wxrate_multitask_alpha05_semi25_seed42_v5}"
GRPO_RUN_NAME="${GRPO_RUN_NAME:-grpo_v3_hybrid_invfreq_4cue_mt_sft_semi25_seed42_cuekl005_v5}"
SPLIT_DIR="${SPLIT_DIR:-data/sft_grpo_gender_wxrate_60_40_semi25_v5}"
PARTITION_MODE="${PARTITION_MODE:-semi_balanced}"
NEUTRAL_FRACTION="${NEUTRAL_FRACTION:-0.25}"
DATA_DIR="${DATA_DIR:-data/meld_16k_mono}"
SEED="${SEED:-42}"
BETA="${BETA:-0.05}"
ALPHA="${ALPHA:-1.0}"
R1_ALPHA="${R1_ALPHA:-0.5}"
R2_WEIGHT="${R2_WEIGHT:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS_PER_GEN="${STEPS_PER_GEN:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
USE_VLLM="${USE_VLLM:-1}"
VLLM_QUANT="${VLLM_QUANT:-bitsandbytes}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.35}"
RESUME_DIR="${RESUME_DIR:-}"

# --- 1. Mandatory GPU selection & strict free-GPU safety check ---
if [[ -z "${GPU_ID}" ]]; then
  echo "[ERROR] You MUST specify which GPU to use so you never collide with another user."
  echo "        Run 'nvidia-smi' first, pick an IDLE 24 GB GPU, then run:"
  echo "        GPU_ID=0 bash scripts/labic/run_grpo.sh"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found. Are you on a GPU machine?"
  exit 1
fi

GPU_INFO="$(nvidia-smi --id="${GPU_ID}" --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -z "${GPU_INFO}" ]]; then
  echo "[ERROR] Invalid GPU_ID=${GPU_ID}. Available GPUs:"
  nvidia-smi --list-gpus
  exit 1
fi

IFS=',' read -r GPU_NAME MEM_USED MEM_FREE MEM_TOTAL <<< "${GPU_INFO}"
MEM_USED="$(echo "${MEM_USED}" | xargs)"
MEM_FREE="$(echo "${MEM_FREE}" | xargs)"
MEM_TOTAL="$(echo "${MEM_TOTAL}" | xargs)"

ACTIVE_PIDS="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs || true)"
if [[ -n "${ACTIVE_PIDS}" ]]; then
  echo "[ABORT] GPU ${GPU_ID} (${GPU_NAME}) already has active compute process(es): ${ACTIVE_PIDS}"
  echo "        Refusing to start so other experiments are NEVER affected."
  echo "        Pick a different idle GPU or wait until it is free."
  exit 1
fi

if (( MEM_USED > 1024 )); then
  echo "[ABORT] GPU ${GPU_ID} (${GPU_NAME}) has ${MEM_USED} MiB already in use (> 1024 MiB safety threshold)."
  echo "        Refusing to start so other experiments are NEVER affected."
  exit 1
fi

if (( MEM_FREE < 20000 )); then
  echo "[ABORT] GPU ${GPU_ID} (${GPU_NAME}) only has ${MEM_FREE} MiB free (< 20,000 MiB required for 24 GB setup)."
  exit 1
fi

echo "[SAFE] GPU ${GPU_ID} (${GPU_NAME}): ${MEM_USED} MiB used / ${MEM_FREE} MiB free / ${MEM_TOTAL} MiB total — IDLE."

# --- 2. Pre-flight check: venv, data, split, and SFT adapter ---
cd "${PROJECT_DIR}"
VENV_PY="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "[ERROR] Virtualenv not found at ${VENV_PY}. Run: bash scripts/labic/setup_env.sh"
  exit 1
fi

REQUIRED_FILES=(
  "${DATA_DIR}/emo_tag_clean/clean_train_set_with_cues.csv"
  "${DATA_DIR}/emo_tag_clean/clean_val_set_with_cues.csv"
  "${DATA_DIR}/emo_tag_clean/gender_acoustic_quartiles.json"
  "${SPLIT_DIR}/grpo_train.csv"
  "${SPLIT_DIR}/split_manifest.json"
  "runs/ser_sft/${SFT_RUN_NAME}/final_sft_adapter/adapter_config.json"
)
for req in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${req}" ]]; then
    echo "[ERROR] Missing required file: ${PROJECT_DIR}/${req}"
    echo "        Upload/pull your data and SFT adapter first before launching."
    exit 1
  fi
done

# --- 3. Resource-polite environment exports (isolated caches & CPU thread caps) ---
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${EXP_DIR}/.cache/huggingface"
export TORCH_HOME="${EXP_DIR}/.cache/torch"
export VLLM_CACHE_ROOT="${EXP_DIR}/.cache/vllm"

# --- 4. Build CLI command ---
CMD=(
  "${VENV_PY}" -u -m src.pipeline grpo
  --project-dir "${PROJECT_DIR}"
  --data-dir "${DATA_DIR}"
  --split-dir "${SPLIT_DIR}"
  --partition-mode "${PARTITION_MODE}"
  --neutral-fraction "${NEUTRAL_FRACTION}"
  --sft-run-name "${SFT_RUN_NAME}"
  --grpo-run-name "${GRPO_RUN_NAME}"
  --seed "${SEED}"
  --beta "${BETA}"
  --alpha "${ALPHA}"
  --r1-alpha "${R1_ALPHA}"
  --r2-weight "${R2_WEIGHT}"
  --batch-size "${BATCH_SIZE}"
  --steps-per-gen "${STEPS_PER_GEN}"
  --grad-accum "${GRAD_ACCUM}"
  --vllm-quantization "${VLLM_QUANT}"
  --vllm-gpu-util "${VLLM_GPU_UTIL}"
)

if [[ "${USE_VLLM}" == "0" ]]; then
  CMD+=(--no-vllm)
fi
if [[ -n "${RESUME_DIR}" ]]; then
  CMD+=(--resume "${RESUME_DIR}")
fi

mkdir -p "${PROJECT_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_DIR}/logs/${GRPO_RUN_NAME}_${TIMESTAMP}.log"
PID_FILE="${PROJECT_DIR}/logs/${GRPO_RUN_NAME}.pid"

# --- 5. Launch safely in background with nohup + nice ---
nohup nice -n 10 "${CMD[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &
JOB_PID=$!
echo "${JOB_PID}" > "${PID_FILE}"

echo ""
echo "================================================================================"
echo "[LAUNCHED] OPERA GRPO v3 started safely in background!"
echo "  Run Name  : ${GRPO_RUN_NAME}"
echo "  SFT Base  : ${SFT_RUN_NAME}"
echo "  GPU       : ${GPU_ID} (${GPU_NAME}) [CUDA_VISIBLE_DEVICES=${GPU_ID}]"
echo "  PID       : ${JOB_PID} (saved in ${PID_FILE})"
echo "  Log File  : ${LOG_FILE}"
echo ""
echo "Useful commands (only affect YOUR job):"
echo "  Monitor log  : tail -f ${LOG_FILE}"
echo "  Check status : ps -fp ${JOB_PID}"
echo "  Check GPU    : nvidia-smi"
echo "  Stop ONLY me : kill -TERM ${JOB_PID}"
echo "================================================================================"
