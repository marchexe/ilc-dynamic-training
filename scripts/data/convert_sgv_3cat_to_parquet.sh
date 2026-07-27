#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/../.." && pwd)"

source "${project_dir}/.venv/bin/activate"
cd "${project_dir}"

input_dir="${ILC_FASTSIM_DIR:-${project_dir}/datasets/20250218_ilc_nnqq_sgvnew}"
output_dir="${ILC_FASTSIM_PARQUET_DIR:-${project_dir}/datasets/20250218_ilc_nnqq_sgvnew_parquet}"
workers="${PARQUET_CONVERT_WORKERS:-2}"
row_group_size="${PARQUET_ROW_GROUP_SIZE:-1000}"
compression="${PARQUET_COMPRESSION:-lz4}"

python -m weaver.utils.convert_to_parquet \
  "${input_dir}"/*_bb_train800k.root \
  "${input_dir}"/*_cc_train800k.root \
  "${input_dir}"/*_dd_train800k.root \
  "${input_dir}"/*_bb_val50k.root \
  "${input_dir}"/*_cc_val50k.root \
  "${input_dir}"/*_dd_val50k.root \
  --output-dir "${output_dir}" \
  --compression "${compression}" \
  --row-group-size "${row_group_size}" \
  --workers "${workers}"
