#!/usr/bin/env bash
set -euo pipefail

gpu_index="${1:-0}"
repeats="${2:-3}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/../.." && pwd)"
checkpoint="${PRETRAINED_WEIGHTS:-${project_dir}/checkpoints/pretrained/ilc_nnqq_sgvnew_3cat_cut/net_best_epoch_state.pt}"
output_dir="${VALIDATION_STABILITY_DIR:-${project_dir}/runs/validation_stability/$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "${output_dir}"

for index in $(seq 1 "${repeats}"); do
  VALIDATION_LOG="${output_dir}/validation_${index}.log" \
  PRETRAINED_WEIGHTS="${checkpoint}" \
    "${script_dir}/validate_pretrained_sgv_3cat.sh" "${gpu_index}"
done

echo "validation stability logs: ${output_dir}"
