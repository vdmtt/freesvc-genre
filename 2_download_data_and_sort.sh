#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
log() { echo "[$(date '+%H:%M:%S')] $1"; }

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

mkdir -p "${REPO_DIR}/logs" "${AUDIO_ROOT}"
EXTRACT_BASE="${DATASET_DIR}/_chunks"
mkdir -p "${EXTRACT_BASE}"

extract_archive() {
    local src="$1"
    local dest="$2"
    mkdir -p "$dest"
    case "${CHUNK_FORMAT}" in
        zip) unzip -q -o "$src" -d "$dest" ;;
        rar) unrar x -y "$src" "$dest/" ;;
        tar) tar -xf "$src" -C "$dest" ;;
        tar.gz|tgz) tar -xzf "$src" -C "$dest" ;;
        7z) 7z x -y "$src" -o"$dest" >/dev/null ;;
        plain) cp -r "$src"/* "$dest"/ ;;
        *) echo "Unknown CHUNK_FORMAT=${CHUNK_FORMAT}" >&2; exit 1 ;;
    esac
}
run_sort() {
    local src_dir="$1"
    local extra_flags=()

    if [ "${COPY_ONLY}" = "1" ]; then
        extra_flags+=(--copy-only)
    fi

    log "Sorting singer data from ${src_dir}"

    python "${REPO_DIR}/sort_singer.py" \
        --input-dir "${src_dir}" \
        --output-dir "${AUDIO_ROOT}" \
        --language "${LANGUAGE}" \
        --sample-rate "${SAMPLE_RATE}" \
        --singer-source "${SORT_SINGER_MODE:-parent}" \
        --manifest "${SORT_MANIFEST}" \
        "${extra_flags[@]}"
}

if [ -n "${SOURCE_DATA_DIR}" ]; then
    run_sort "${SOURCE_DATA_DIR}"
elif [ "${#DATA_CHUNK_IDS[@]}" -gt 0 ]; then
    for FILE_ID in "${DATA_CHUNK_IDS[@]}"; do
        CHUNK_FILE="${EXTRACT_BASE}/${FILE_ID}.${CHUNK_FORMAT}"
        EXTRACT_DIR="${EXTRACT_BASE}/extracted_${FILE_ID}"
        if [ ! -f "${CHUNK_FILE}" ]; then
            log "Downloading chunk ${FILE_ID}"
            gdown "${FILE_ID}" -O "${CHUNK_FILE}" --fuzzy
        else
            log "Chunk ${FILE_ID} already exists"
        fi
        if [ ! -f "${EXTRACT_DIR}/.done" ]; then
            log "Extracting ${FILE_ID}"
            rm -rf "${EXTRACT_DIR}"
            mkdir -p "${EXTRACT_DIR}"
            extract_archive "${CHUNK_FILE}" "${EXTRACT_DIR}"
            touch "${EXTRACT_DIR}/.done"
        fi
        run_sort "${EXTRACT_DIR}"
    done
else
    echo "No SOURCE_DATA_DIR and DATA_CHUNK_IDS is empty. Edit config.sh first." >&2
    exit 1
fi

# Cleanup at the VERY END of sort: all chunks are downloaded, extracted and
# sorted into AUDIO_ROOT by now, so the whole raw staging area (archives +
# extracted folders) can be removed in one pass. Doing it here (instead of
# per-chunk) means nothing is deleted until the full sort has succeeded.
if [ "${DELETE_CHUNK_AFTER_SORT:-1}" = "1" ] && [ -d "${EXTRACT_BASE}" ]; then
    log "Sort finished. Removing raw chunk staging dir to save disk: ${EXTRACT_BASE}"
    rm -rf "${EXTRACT_BASE}"
fi

log "Data sorted into ${AUDIO_ROOT}/${LANGUAGE}/<singer>"
