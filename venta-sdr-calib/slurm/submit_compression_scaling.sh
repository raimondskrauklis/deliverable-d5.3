#!/bin/bash
# Submit the bounded, verified Raw-to-Zarr compression scaling experiment.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${env_file}"

experiment_id="${COMPRESS_EXPERIMENT_ID:-compression_scaling_$(date -u +%Y%m%dT%H%M%SZ)}"
# Ignore a stale generic OUTPUT_DIR from a previous benchmark.
output_dir="${COMPRESS_LOG_DIR:-${BENCH_OUTPUT_ROOT}/${experiment_id}}"
scratch_root="${COMPRESS_SCRATCH_ROOT:-/var/scratch/${USER}/framework_benchmark_compression_scaling}"
mkdir -p "${output_dir}"

submit_args=(
    --parsable
    --output="${output_dir}/slurm-%x-%j.out"
    --error="${output_dir}/slurm-%x-%j.err"
    --export="ALL,MATRIX_ID=${experiment_id},OUTPUT_DIR=${output_dir},COMPRESS_SCRATCH_ROOT=${scratch_root}"
)
if [[ -n "${BENCH_NODELIST:-}" ]]; then
    submit_args+=(--nodelist="${BENCH_NODELIST}")
fi

submitted="$(sbatch "${submit_args[@]}" "$@" "${script_dir}/compression_scaling.sbatch")"
job_id="${submitted%%;*}"
echo "submitted compression scaling=${experiment_id} job=${job_id}"
echo "durable results: ${output_dir}"
echo "temporary archives: ${scratch_root}/${experiment_id}"
echo "current condition archives are deleted after durable logs/metrics are copied"
echo "stop safely before the next condition: touch '${output_dir}/STOP'"
