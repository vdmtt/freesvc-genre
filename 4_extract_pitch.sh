#!/bin/bash
#SBATCH --job-name=freesvc_pitch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/slurm_pitch_%j.log
#SBATCH --error=logs/slurm_pitch_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
log() { echo "[$(date '+%F %T')] $*"; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/logs" "${DATASET_DIR}/pitch_features"

log "Extracting pitch: ${AUDIO_ROOT} -> ${DATASET_DIR}/pitch_features"
python scripts/preprocess_pitch.py \
    --in-dir "${AUDIO_ROOT}" \
    --out-dir "${DATASET_DIR}/pitch_features" \
    --pitch-predictor "${PITCH_PREDICTOR}" \
    --sampling-rate "${SAMPLE_RATE}" \
    --hop-length 320 \
    --device cuda \
    --num-workers "${PITCH_NUM_WORKERS}" \
    --skip-existing

log "Pitch extraction done."
