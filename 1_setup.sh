#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "REPO_DIR resolved to: ${REPO_DIR}"
log "=== Setting up Conda environment: ${CONDA_ENV_NAME} ==="

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    log "Env '${CONDA_ENV_NAME}' already exists, skipping."
else
    conda create -n "${CONDA_ENV_NAME}" python=3.8 -y
    log "Env '${CONDA_ENV_NAME}' created."
fi

conda activate "${CONDA_ENV_NAME}"

log "=== Installing system dependencies ==="

if ! command -v unrar >/dev/null 2>&1; then
    log "unrar not found. Trying to install unrar..."
    sudo apt-get update || true
    sudo apt-get install -y unrar || sudo apt-get install -y unrar-free || true
else
    log "unrar already installed."
fi

if ! command -v unzip >/dev/null 2>&1; then
    log "unzip not found. Trying to install unzip..."
    sudo apt-get install -y unzip || true
else
    log "unzip already installed."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg not found. Trying to install ffmpeg (required for audio conversion)..."
    sudo apt-get install -y ffmpeg || true
else
    log "ffmpeg already installed."
fi

log "=== Installing Python dependencies ==="

pip install -U pip wheel setuptools -q
pip install -r "${REPO_DIR}/requirements.txt" -q
pip install gdown requests tqdm huggingface_hub -q

log "=== Creating checkpoint directories ==="

mkdir -p "${PRETRAIN_DIR}"
mkdir -p "$(dirname "${SPIN_CKPT}")"
mkdir -p "$(dirname "${RMVPE_CKPT}")"
mkdir -p "${REPO_DIR}/logs"

download_url() {
    local url="$1"
    local out="$2"
    local name="$3"

    if [ -f "${out}" ]; then
        log "${name} already exists: ${out}"
        return 0
    fi

    if [ -z "${url}" ]; then
        log "${name} URL is empty, skip URL download."
        return 1
    fi

    log "Downloading ${name}..."
    log "URL: ${url}"
    log "OUT: ${out}"

    mkdir -p "$(dirname "${out}")"

    if command -v wget >/dev/null 2>&1; then
        wget -O "${out}" "${url}"
    else
        curl -L "${url}" -o "${out}" --retry 3 --retry-delay 5
    fi

    log "${name} done."
}

download_gdrive() {
    local file_id="$1"
    local out="$2"
    local name="$3"

    if [ -f "${out}" ]; then
        log "${name} already exists: ${out}"
        return 0
    fi

    if [ -z "${file_id}" ]; then
        log "${name} Google Drive ID is empty, skip gdown download."
        return 1
    fi

    log "Downloading ${name} from Google Drive..."
    log "ID : ${file_id}"
    log "OUT: ${out}"

    mkdir -p "$(dirname "${out}")"

    gdown "https://drive.google.com/uc?id=${file_id}" -O "${out}"

    log "${name} done."
}

check_file() {
    local path="$1"
    local name="$2"

    if [ -f "${path}" ]; then
        log "[OK] ${name}: ${path}"
    else
        log "[MISSING] ${name}: ${path}"
        return 1
    fi
}
onedrive_to_direct_url() {
    local page_url="$1"

    local cid
    local resid

    cid="$(echo "$page_url" | sed -n 's/.*[?&]cid=\([^&]*\).*/\1/p')"
    resid="$(echo "$page_url" | sed -n 's/.*[?&]id=\([^&]*\).*/\1/p')"

    if [ -z "$cid" ] || [ -z "$resid" ]; then
        echo ""
        return 1
    fi

    resid="${resid//!/%21}"

    echo "https://onedrive.live.com/download?cid=${cid}&resid=${resid}"
}

download_onedrive_file() {
    local page_url="$1"
    local out="$2"
    local name="$3"

    if [ -f "$out" ]; then
        log "${name} already exists: ${out}"
        return 0
    fi

    if [ -z "$page_url" ]; then
        log "${name} OneDrive URL empty, skip."
        return 1
    fi

    local direct_url
    direct_url="$(onedrive_to_direct_url "$page_url")"

    if [ -z "$direct_url" ]; then
        log "ERROR: Cannot parse OneDrive URL for ${name}"
        log "URL: ${page_url}"
        return 1
    fi

    log "Downloading ${name} from OneDrive..."
    log "Page URL  : ${page_url}"
    log "Direct URL: ${direct_url}"
    log "Output    : ${out}"

    mkdir -p "$(dirname "$out")"

    if command -v wget >/dev/null 2>&1; then
        wget -O "$out" "$direct_url"
    else
        curl -L "$direct_url" -o "$out" --retry 3 --retry-delay 5
    fi

    if head -c 100 "$out" | grep -qi "<html"; then
        log "ERROR: ${name} download looks like HTML, not a .pth checkpoint."
        log "Try downloading manually from browser and copy to: ${out}"
        rm -f "$out"
        return 1
    fi

    log "${name} downloaded: ${out}"
}

log "=== Downloading FreeSVC auxiliary checkpoints ==="

download_url \
    "https://huggingface.co/alefiury/free-svc/resolve/main/spin.ckpt" \
    "${SPIN_CKPT}" \
    "SPIN checkpoint"

download_url \
    "https://huggingface.co/alefiury/free-svc/resolve/main/rmvpe.pt" \
    "${RMVPE_CKPT}" \
    "RMVPE checkpoint"
log "=== Preparing FreeVC generator/discriminator ==="

# Generator: Google Drive ID -> HF URL -> OneDrive (first one that works wins).
if [ ! -f "${FREEVC_G_CKPT}" ]; then
    download_gdrive "${FREEVC_G_GDRIVE_ID:-}"   "${FREEVC_G_CKPT}" "FreeVC generator (gdrive)"   || true
fi
if [ ! -f "${FREEVC_G_CKPT}" ]; then
    download_url    "${FREEVC_G_HF_URL:-}"      "${FREEVC_G_CKPT}" "FreeVC generator (HF mirror)" || true
fi
if [ ! -f "${FREEVC_G_CKPT}" ]; then
    download_onedrive_file "${FREEVC_G_ONEDRIVE_URL:-}" "${FREEVC_G_CKPT}" "FreeVC generator (onedrive)" || true
fi

if [ ! -f "${FREEVC_D_CKPT}" ]; then
    download_gdrive "${FREEVC_D_GDRIVE_ID:-}"   "${FREEVC_D_CKPT}" "FreeVC discriminator (gdrive)"   || true
fi
if [ ! -f "${FREEVC_D_CKPT}" ]; then
    download_url    "${FREEVC_D_HF_URL:-}"      "${FREEVC_D_CKPT}" "FreeVC discriminator (HF mirror)" || true
fi
if [ ! -f "${FREEVC_D_CKPT}" ]; then
    download_onedrive_file "${FREEVC_D_ONEDRIVE_URL:-}" "${FREEVC_D_CKPT}" "FreeVC discriminator (onedrive)" || true
fi

log "=== Final checkpoint check ==="

MISSING=0

check_file "${SPIN_CKPT}" "SPIN_CKPT" || MISSING=1
check_file "${RMVPE_CKPT}" "RMVPE_CKPT" || MISSING=1
check_file "${FREEVC_G_CKPT}" "FREEVC_G_CKPT" || MISSING=1

if check_file "${FREEVC_D_CKPT}" "FREEVC_D_CKPT"; then
    :
else
    if [ "${FREEVC_D_REQUIRED:-0}" = "1" ]; then
        MISSING=1
    else
        log "[OPTIONAL] Discriminator not found; training will start D from scratch."
    fi
fi

if [ "${MISSING}" = "1" ]; then
    log "=== Setup failed: missing required checkpoint(s) ==="
    echo
    echo "Required files:"
    echo "  ${SPIN_CKPT}"
    echo "  ${RMVPE_CKPT}"
    echo "  ${FREEVC_G_CKPT}"
    echo "  ${FREEVC_D_CKPT}"
    echo "  mkdir -p ${PRETRAIN_DIR}"
    echo "  cp /path/to/freevc-24.pth ${FREEVC_G_CKPT}"
    echo "  cp /path/to/D-freevc-24.pth ${FREEVC_D_CKPT}"
    exit 1
fi

log "=== Setup complete! ==="