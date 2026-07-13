#!/bin/bash
# Local CPU end-to-end smoke test. Forces CPU even if a GPU is present.
# Assumes you already have a tiny dataset_custom/train.csv + valid.csv
# (run 2_download_data_and_sort.sh + 3_split.sh on a few wavs, or hand-write them).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Force CPU: makes torch.cuda.is_available() False -> cpu_compat shim activates.
export CUDA_VISIBLE_DEVICES=""
# Keep thread count modest so a laptop stays responsive.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[cpu-dummy] Running 1-epoch CPU smoke test with config_cpu_dummy ..."
python train.py --config-dir configs --config-name config_cpu_dummy

echo "[cpu-dummy] Done. Check checkpoints in: dataset_custom/checkpoints_cpu_dummy"