REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="freesvc"
PYTHON_VERSION="3.8"

CHUNK_FORMAT="rar"  # zip | rar | tar | tar.gz | tgz | 7z | plain

SOURCE_DATA_DIR=""

DATA_CHUNK_IDS=(
    "17u0IBwdwZnJU6X0Nh_zEd6a7oz2EnyIZ"
    "1Zjair5o_bP_K6TKV9sTOfvzHabB-Gsbt"
    "1BlPioErdIGdjnXTXy0PTn_gX7pkajVOs"
    "10rDUESLQ5QXSW32MbqlXfEOyb1z6kO1R"
    "1xBBc_Cz4weihvb5O_Q5V4xryUZD3f4Fu"
    "1kztz57Vrrldj_9HaUnuLyws0pt-lwIpr"
    "1L-GhttMDX2Qb3Vpi8BQSko-Ii5A6Tbzq"
)

# ===== Dataset =====
DATASET_DIR="${REPO_DIR}/dataset_custom"
AUDIO_ROOT="${DATASET_DIR}/audio"
SORT_MANIFEST="${DATASET_DIR}/sort_manifest.csv"

LANGUAGE="english"
SAMPLE_RATE=24000
COPY_ONLY=0

PRETRAIN_DIR="${REPO_DIR}/pretrained"

FREEVC_G_GDRIVE_ID="1Qcg8hrfGQi_nFeC9VDZbtSqShTrMjNHW"
FREEVC_D_GDRIVE_ID="1qjmwHN1-EnLX21ynbUmWo2Hdjac-sKuU"

FREEVC_G_HF_URL="https://huggingface.co/spaces/OlaWod/FreeVC/resolve/main/checkpoints/freevc-24.pth"

FREEVC_D_HF_URL=""

# Legacy OneDrive fallbacks (flaky; kept only as last resort).
FREEVC_G_ONEDRIVE_URL="https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbnZ1a1ZubFEzWlR4MXJqck9aMmFiQ3d1QkFoP2U9VWxoUlI1&cid=537643E55991EE7B&sb=name&sd=1&id=537643E55991EE7B%2110737&parId=537643E55991EE7B%2110086&o=OneUp"
FREEVC_D_ONEDRIVE_URL="https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbnZ1a1ZubFEzWlR4MXJqck9aMmFiQ3d1QkFoP2U9VWxoUlI1&cid=537643E55991EE7B&sb=name&sd=1&id=537643E55991EE7B%2110738&parId=537643E55991EE7B%2110086&o=OneUp"

FREEVC_D_REQUIRED=1

FREEVC_G_CKPT="${PRETRAIN_DIR}/freevc-24.pth"
FREEVC_D_CKPT="${PRETRAIN_DIR}/D-freevc-24.pth"

SPIN_CKPT="${REPO_DIR}/models/spin/spin.ckpt"
RMVPE_CKPT="${REPO_DIR}/models/f0_predictor/ckpt/rmvpe.pt"

# Used by 5_train.sh
PRETRAIN_G="${FREEVC_G_CKPT}"
PRETRAIN_D="${FREEVC_D_CKPT}"

# ===== Singer-level zero-shot split =====
VAL_SINGER_RATIO=0.10
TEST_SINGER_RATIO=0.10
MIN_FILES_PER_SINGER=1
SEED=1806

TRAIN_SINGERS_FILE=""
VAL_SINGERS_FILE=""
TEST_SINGERS_FILE=""

TRAIN_CSV="${DATASET_DIR}/train.csv"
VALID_CSV="${DATASET_DIR}/valid.csv"
TEST_CSV="${DATASET_DIR}/test.csv"
ALL_CSV="${DATASET_DIR}/all.csv"

# ===== Pitch extraction =====
PITCH_PREDICTOR="rmvpe"
PITCH_NUM_WORKERS=1

# ===== Training =====
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_NAME="config_our"

TRAIN_BATCH_SIZE=16
EPOCHS=50
FP16_RUN=true
NUM_CPU_PROCESSES="${SLURM_CPUS_PER_TASK:-8}"

SAVE_STEPS_INTERVAL=5000
VALID_STEPS_INTERVAL=1000
SAVE_EPOCH_INTERVAL=5

WEIGHTED_BATCH_SPEAKER_SAMPLING=0.5
WEIGHTED_BATCH_LANG_SAMPLING=0.0