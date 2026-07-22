#!/usr/bin/env bash
set -euo pipefail

gpu_index="${1:-2}"
controller_mode="${2:-observe}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"

case "${controller_mode}" in
  baseline|observe|active) ;;
  *)
    echo "controller mode must be 'baseline', 'observe', or 'active'" >&2
    exit 2
    ;;
esac

source "${project_dir}/.venv/bin/activate"
cd "${project_dir}"

data_dir="${ILC_FASTSIM_DIR:-/data/suehara/mldata/flavortag/20250218_ilc_nnqq_sgvnew}"
epochs="${EPOCHS:-20}"
samples_per_epoch="${SAMPLES_PER_EPOCH:-2400000}"
samples_per_epoch_val="${SAMPLES_PER_EPOCH_VAL:-150000}"
seed="${SEED:-12345}"
run_name="${RUN_NAME:-${controller_mode}}"
run_dir="runs/sgv_3cat/${run_name}"

mkdir -p "${run_dir}"

controller_args=()
if [[ "${controller_mode}" != "baseline" ]]; then
  controller_args=(
    --training-controller "configs/controllers/linucb_lr_${controller_mode}.yaml"
  )
fi

weaver \
  --run-mode train,val \
  --data-train \
    "nnbb:${data_dir}/*_bb_train800k.root" \
    "nncc:${data_dir}/*_cc_train800k.root" \
    "nndd:${data_dir}/*_dd_train800k.root" \
  --data-val \
    "nnbb:${data_dir}/*_bb_val50k.root" \
    "nncc:${data_dir}/*_cc_val50k.root" \
    "nndd:${data_dir}/*_dd_val50k.root" \
  --data-config configs/data/ilc_nnqq_sgvnew_3cat.yaml \
  --network-config networks/particle_transformer_ee.py \
  --model-prefix "${run_dir}/net" \
  --log-file "${run_dir}/train.log" \
  "${controller_args[@]}" \
  --lr-scheduler none \
  --seed "${seed}" \
  --optimizer AdamW \
  --optimizer-option weight_decay 1e-4 \
  --batch-size 256 \
  --start-lr 1e-3 \
  --samples-per-epoch "${samples_per_epoch}" \
  --samples-per-epoch-val "${samples_per_epoch_val}" \
  --num-epochs "${epochs}" \
  --num-workers 1 \
  --fetch-step 0.01 \
  --gpus "${gpu_index}"
