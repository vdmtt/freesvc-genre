#!/bin/bash
# Thin wrapper around the original stage scripts. It does NOT reimplement them;
# it just calls 1_setup.sh / 2_..sh / 3_..sh in order. Edit those files as usual.
#
# Usage:
#   bash run_prepare.sh                   # run 1 -> 2 -> 3
#   bash run_prepare.sh --steps split     # only 3
#   bash run_prepare.sh --steps sort,split# only 2 then 3
#   bash run_prepare.sh setup sort        # positional also works
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
log() { echo "[$(date '+%H:%M:%S')] === $* ==="; }

# Map stage name -> script file. Adjust filenames here if yours differ.
SETUP_SH="${SCRIPT_DIR}/1_setup.sh"
SORT_SH="${SCRIPT_DIR}/2_download_data_and_sort.sh"
SPLIT_SH="${SCRIPT_DIR}/3_split.sh"

# Parse steps (default = all). Accept --steps a,b,c  OR  positional  setup sort.
STEPS="setup,sort,split"
if [ $# -gt 0 ]; then
    case "$1" in
        --steps) STEPS="$2" ;;
        --steps=*) STEPS="${1#*=}" ;;
        *) STEPS="$(echo "$*" | tr ' ' ',')" ;;
    esac
fi
want() { case ",${STEPS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

mkdir -p "${SCRIPT_DIR}/logs"

if want setup; then log "Setup environment + pretrained checkpoints (needs internet)"; bash "${SETUP_SH}"; fi
if want sort;  then log "Download & reorganize data (needs internet)";               bash "${SORT_SH}";  fi
if want split; then log "Split + verify dataset";                                    bash "${SPLIT_SH}"; fi

# ---- post-split verification (merged from run_preprocess.sh) ----
if want split; then
    log "Verifying split outputs"

    for f in "${TRAIN_CSV}" "${VALID_CSV}" "${TEST_CSV}"; do
        [ -f "${f}" ] || { echo "[ERROR] Missing split CSV: ${f}" >&2; exit 1; }
    done

    log "Preview train.csv"; head -n 5 "${TRAIN_CSV}" || true
    log "Preview valid.csv"; head -n 5 "${VALID_CSV}" || true
    log "Preview test.csv";  head -n 5 "${TEST_CSV}"  || true

    log "Singer counts per split"
    for name in train valid test; do
        f="${DATASET_DIR}/${name}.csv"
        [ -f "${f}" ] && echo "  ${name}: $(cut -d'|' -f3 "${f}" | sort -u | wc -l) singer(s)"
    done

    TRAIN_SINGERS="${DATASET_DIR}/singer_splits/train_singers.txt"
    VALID_SINGERS="${DATASET_DIR}/singer_splits/valid_singers.txt"
    TEST_SINGERS="${DATASET_DIR}/singer_splits/test_singers.txt"
    if [ ! -f "${TRAIN_SINGERS}" ] || [ ! -f "${VALID_SINGERS}" ] || [ ! -f "${TEST_SINGERS}" ]; then
        echo "[ERROR] Missing singer split files under ${DATASET_DIR}/singer_splits" >&2
        exit 1
    fi

    log "Checking singer leakage between splits"
    LEAK_TV="$(comm -12 <(sort "${TRAIN_SINGERS}") <(sort "${VALID_SINGERS}") || true)"
    LEAK_TT="$(comm -12 <(sort "${TRAIN_SINGERS}") <(sort "${TEST_SINGERS}")  || true)"
    LEAK_VT="$(comm -12 <(sort "${VALID_SINGERS}") <(sort "${TEST_SINGERS}")  || true)"
    if [ -n "${LEAK_TV}" ] || [ -n "${LEAK_TT}" ] || [ -n "${LEAK_VT}" ]; then
        echo "[ERROR] Singer leakage detected!" >&2
        echo "train ∩ valid: ${LEAK_TV}"
        echo "train ∩ test : ${LEAK_TT}"
        echo "valid ∩ test : ${LEAK_VT}"
        exit 1
    fi
    log "No singer leakage."

    log "Checking that audio paths in CSVs exist"
    MISSING_AUDIO=0
    while IFS= read -r wav_path; do
        [ -z "${wav_path}" ] && continue
        if [ ! -f "${wav_path}" ]; then
            echo "[MISSING_AUDIO] ${wav_path}" >&2
            MISSING_AUDIO=1
        fi
    done < <(awk -F'|' '{print $1}' "${TRAIN_CSV}" "${VALID_CSV}" "${TEST_CSV}")
    if [ "${MISSING_AUDIO}" = "1" ]; then
        echo "[ERROR] Some audio paths in CSV do not exist." >&2
        exit 1
    fi
    log "All audio paths exist."
fi

log "run_prepare DONE (steps: ${STEPS})"