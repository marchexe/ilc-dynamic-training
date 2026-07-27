#!/usr/bin/env bash
set -euo pipefail

gpu_index="${1:-0}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/../.." && pwd)"

source "${project_dir}/.venv/bin/activate"
cd "${project_dir}"

data_dir="${ILC_FASTSIM_DIR:-${project_dir}/datasets/20250218_ilc_nnqq_sgvnew}"
checkpoint="${PRETRAINED_WEIGHTS:-${project_dir}/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt}"
log_file="${VALIDATION_LOG:-${project_dir}/runs/sgv_3cat/pretrained_validation.log}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Pretrained checkpoint not found: ${checkpoint}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${log_file}")"

weaver \
  --run-mode test \
  --data-test \
    "${data_dir}/ml_flavtag_6cat_bb_val50k.root" \
    "${data_dir}/ml_flavtag_6cat_cc_val50k.root" \
    "${data_dir}/ml_flavtag_6cat_dd_val50k.root" \
  --data-config /data/suehara/part/data/ilc_nnqq_sgvnew_3cat_cut.217feb3dc9ed1ee6978db1c04604f81b.auto.yaml \
  --network-config networks/pretrained_sgv_particle_transformer.py \
  --model-prefix "${checkpoint}" \
  --log-file "${log_file}" \
  --batch-size 256 \
  --use-amp \
  --amp-dtype fp16 \
  --num-workers 1 \
  --fetch-step 0.01 \
  --gpus "${gpu_index}" \
  --predict-gpus "${gpu_index}"
