#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
source "${env_file}"

pilot_id="${PILOT_ID:-zarr_compare_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${ZARR_COMPARE_LOG_DIR:-${BENCH_OUTPUT_ROOT}/${pilot_id}}"
mkdir -p "${output_dir}"

submit_args=(
    --parsable
    --output="${output_dir}/slurm-%x-%j.out"
    --error="${output_dir}/slurm-%x-%j.err"
    --export="ALL,PILOT_ID=${pilot_id},OUTPUT_DIR=${output_dir},ZARR_DIR=${ZARR_DIR}"
)
if [[ -n "${BENCH_NODELIST:-}" ]]; then
    submit_args+=(--nodelist="${BENCH_NODELIST}")
fi

submitted="$(sbatch "${submit_args[@]}" "$@" "${script_dir}/zarr_compare_pilot.sbatch")"
job_id="${submitted%%;*}"
echo "submitted Raw/Zarr correctness pilot=${pilot_id} job=${job_id}"
echo "All logs and results: ${output_dir}"
