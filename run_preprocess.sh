#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
log() {
    echo "[$(date '+%H:%M:%S')] === $1 ==="
}
mkdir -p "${SCRIPT_DIR}/logs"

log "FreeSVC CPU prepare pipeline"
log "This stage does NOT require GPU"
log "REPO_DIR: ${REPO_DIR}"
log "DATASET_DIR: ${DATASET_DIR}"
log "LANGUAGE: ${LANGUAGE}"

bash "${SCRIPT_DIR}/1_setup.sh"

bash "${SCRIPT_DIR}/2_download_data_and_sort.sh"

bash "${SCRIPT_DIR}/3_split.sh"

log "Checking generated split files"

if [ ! -f "${TRAIN_CSV}" ]; then
    echo "[ERROR] Missing TRAIN_CSV: ${TRAIN_CSV}" >&2
    exit 1
fi

if [ ! -f "${VALID_CSV}" ]; then
    echo "[ERROR] Missing VALID_CSV: ${VALID_CSV}" >&2
    exit 1
fi

if [ ! -f "${TEST_CSV}" ]; then
    echo "[ERROR] Missing TEST_CSV: ${TEST_CSV}" >&2
    exit 1
fi

log "Preview train.csv"
head -n 5 "${TRAIN_CSV}" || true

log "Preview valid.csv"
head -n 5 "${VALID_CSV}" || true

log "Preview test.csv"
head -n 5 "${TEST_CSV}" || true

log "Checking singer split files"

TRAIN_SINGERS="${DATASET_DIR}/singer_splits/train_singers.txt"
VALID_SINGERS="${DATASET_DIR}/singer_splits/valid_singers.txt"
TEST_SINGERS="${DATASET_DIR}/singer_splits/test_singers.txt"

if [ ! -f "${TRAIN_SINGERS}" ] || [ ! -f "${VALID_SINGERS}" ] || [ ! -f "${TEST_SINGERS}" ]; then
    echo "[ERROR] Missing singer split files under ${DATASET_DIR}/singer_splits" >&2
    exit 1
fi

log "Checking singer leakage"

LEAK_TRAIN_VALID="$(comm -12 <(sort "${TRAIN_SINGERS}") <(sort "${VALID_SINGERS}") || true)"
LEAK_TRAIN_TEST="$(comm -12 <(sort "${TRAIN_SINGERS}") <(sort "${TEST_SINGERS}") || true)"
LEAK_VALID_TEST="$(comm -12 <(sort "${VALID_SINGERS}") <(sort "${TEST_SINGERS}") || true)"

if [ -n "${LEAK_TRAIN_VALID}" ] || [ -n "${LEAK_TRAIN_TEST}" ] || [ -n "${LEAK_VALID_TEST}" ]; then
    echo "[ERROR] Singer leakage detected!" >&2
    echo "train ∩ valid:"
    echo "${LEAK_TRAIN_VALID}"
    echo "train ∩ test:"
    echo "${LEAK_TRAIN_TEST}"
    echo "valid ∩ test:"
    echo "${LEAK_VALID_TEST}"
    exit 1
fi


MISSING_AUDIO=0

awk -F'|' '{print $1}' "${TRAIN_CSV}" "${VALID_CSV}" "${TEST_CSV}" | while read -r wav_path; do
    if [ ! -f "${wav_path}" ]; then
        echo "[MISSING_AUDIO] ${wav_path}"
        exit 2
    fi
done || MISSING_AUDIO=1

if [ "${MISSING_AUDIO}" = "1" ]; then
    echo "[ERROR] Some audio paths in CSV do not exist." >&2
    exit 1
fi

log "PREPARE DONE"
log "Processed audio:"
log "  ${AUDIO_ROOT}/${LANGUAGE}/<Singer>/<Singer>_XXXXXXXX.wav"

log "Split files:"
log "  ${TRAIN_CSV}"
log "  ${VALID_CSV}"
log "  ${TEST_CSV}"
