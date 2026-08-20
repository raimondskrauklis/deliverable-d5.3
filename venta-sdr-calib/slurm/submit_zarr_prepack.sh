#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
source "${env_file}"

ZARR_DIR="${ZARR_DIR:-/var/scratch/${USER}/rlmi_f1666_ir_24_zarr}"
pack_id="${ZARR_PACK_ID:-zarr_pack_$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${ZARR_LOG_DIR:-${BENCH_OUTPUT_ROOT}/${pack_id}}"
mkdir -p "${output_dir}"

submit_args=(
    --parsable
    --output="${output_dir}/slurm-%x-%j.out"
    --error="${output_dir}/slurm-%x-%j.err"
    --export="ALL,MATRIX_ID=${pack_id},OUTPUT_DIR=${output_dir},ZARR_DIR=${ZARR_DIR},ZARR_GROUPS=${ZARR_GROUPS:-4},ZARR_GROUP_OFFSET=${ZARR_GROUP_OFFSET:-0}"
)
if [[ -n "${BENCH_NODELIST:-}" ]]; then
    submit_args+=(--nodelist="${BENCH_NODELIST}")
fi

submitted="$(sbatch "${submit_args[@]}" "$@" "${script_dir}/zarr_prepack.sbatch")"
job_id="${submitted%%;*}"
echo "submitted zarr preparation=${pack_id} groups=${ZARR_GROUPS:-4} job=${job_id}"
echo "Slurm logs: ${output_dir}"
echo "Zarr archives: ${ZARR_DIR}"
