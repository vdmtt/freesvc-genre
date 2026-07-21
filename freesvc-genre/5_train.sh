#!/bin/bash
#SBATCH --job-name=freesvc_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/slurm_train_%j.log
#SBATCH --error=logs/slurm_train_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
log() { echo "[$(date '+%F %T')] $*"; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/logs" "${DATASET_DIR}/checkpoints"

if [ ! -f "${DATASET_DIR}/train.csv" ] || [ ! -f "${DATASET_DIR}/valid.csv" ]; then
    echo "Missing train.csv/valid.csv. Run prepare.sh" >&2
    exit 1
fi

OVERRIDES=(
    data.dataset_dir="${DATASET_DIR}"
    data.training_files="${DATASET_DIR}/train.csv"
    data.validation_files="${DATASET_DIR}/valid.csv"
    data.pitch_features_dir="${DATASET_DIR}/pitch_features"
    data.sampling_rate="${SAMPLE_RATE}"
    data.num_workers="${NUM_CPU_PROCESSES}"
    train.batch_size="${TRAIN_BATCH_SIZE}"
    train.epochs="${EPOCHS}"
    train.fp16_run="${FP16_RUN}"
    train.distributed=false
    train.use_multiprocessing=false
    train.save_steps_interval="${SAVE_STEPS_INTERVAL}"
    train.valid_steps_interval="${VALID_STEPS_INTERVAL}"
    train.save_epoch_interval="${SAVE_EPOCH_INTERVAL}"
    train.weighted_batch_speaker_sampling="${WEIGHTED_BATCH_SPEAKER_SAMPLING}"
    train.weighted_batch_lang_sampling="${WEIGHTED_BATCH_LANG_SAMPLING}"
    model.save_dir="${DATASET_DIR}/checkpoints"
)

if [ -n "${PRETRAIN_G}" ] && [ -f "${PRETRAIN_G}" ]; then
    log "Finetune generator from ${PRETRAIN_G}"
    OVERRIDES+=(model.finetune_from_model.generator="${PRETRAIN_G}")
else
    log "No PRETRAIN_G found; training generator from scratch"
    OVERRIDES+=(model.finetune_from_model.generator=null)
fi

if [ -n "${PRETRAIN_D}" ] && [ -f "${PRETRAIN_D}" ]; then
    log "Finetune discriminator from ${PRETRAIN_D}"
    OVERRIDES+=(model.finetune_from_model.discriminator="${PRETRAIN_D}")
else
    OVERRIDES+=(model.finetune_from_model.discriminator=null)
fi

log "Starting FreeSVC training with config ${CONFIG_NAME} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python train.py --config-dir configs --config-name "${CONFIG_NAME}" "${OVERRIDES[@]}"

log "Training done. Checkpoints: ${DATASET_DIR}/checkpoints"
