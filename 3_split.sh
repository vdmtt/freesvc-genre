#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
log() { echo "[$(date '+%F %T')] $*"; }

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

mkdir -p "${REPO_DIR}/logs" "${DATASET_DIR}"

extra_flags=()
[ -n "${TRAIN_SINGERS_FILE}" ] && extra_flags+=(--train-singers-file "${TRAIN_SINGERS_FILE}")
[ -n "${VAL_SINGERS_FILE}" ] && extra_flags+=(--valid-singers-file "${VAL_SINGERS_FILE}")
[ -n "${TEST_SINGERS_FILE}" ] && extra_flags+=(--test-singers-file "${TEST_SINGERS_FILE}")

log "Creating singer-level train/valid/test CSVs"
python "${REPO_DIR}/split_by_singer.py" \
    --audio-root "${AUDIO_ROOT}" \
    --out-dir "${DATASET_DIR}" \
    --language "${LANGUAGE}" \
    --valid-ratio "${VAL_SINGER_RATIO}" \
    --test-ratio "${TEST_SINGER_RATIO}" \
    --seed "${SEED}" \
    --min-files-per-singer "${MIN_FILES_PER_SINGER}" \
    "${extra_flags[@]}"

log "Verifying dataset"
python "${REPO_DIR}/verify_freesvc_dataset.py" \
    --dataset-dir "${DATASET_DIR}" \
    --sample-rate "${SAMPLE_RATE}" \
    --max-audio-check 500

log "Split files are ready: ${DATASET_DIR}/train.csv, valid.csv, test.csv"
