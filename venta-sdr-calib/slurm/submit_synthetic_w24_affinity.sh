#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
source "${env_file}"

read -r -a repetitions <<< "${BENCH_REPETITIONS:-1 2 3 4 5}"
matrix_id="${MATRIX_ID:-synthetic_w24_affinity_$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-${BENCH_OUTPUT_ROOT}/${matrix_id}}"
stop_file="${BENCH_STOP_FILE:-${output_dir}/STOP}"
mkdir -p "${output_dir}"
if [[ -e "${stop_file}" ]]; then
    echo "ERROR: stop sentinel already exists: ${stop_file}" >&2
    exit 2
fi

previous_job=""
job_count=0
for repetition in "${repetitions[@]}"; do
    if ! [[ "${repetition}" =~ ^[1-5]$ ]]; then
        echo "ERROR: invalid repetition: ${repetition}" >&2
        exit 2
    fi
    submit_args=(
        --parsable
        --ntasks=25
        --hint=nomultithread
        --array="${repetition}"
        --output="${output_dir}/slurm-%x-%A_%a.out"
        --error="${output_dir}/slurm-%x-%A_%a.err"
        --export="ALL,MATRIX_ID=${matrix_id},OUTPUT_DIR=${output_dir},BENCH_STOP_FILE=${stop_file}"
    )
    if [[ -n "${BENCH_NODELIST:-}" ]]; then
        submit_args+=(--nodelist="${BENCH_NODELIST}")
    fi
    if [[ -n "${previous_job}" ]]; then
        submit_args+=(--dependency="afterany:${previous_job}")
    fi
    submitted="$(sbatch "${submit_args[@]}" "${script_dir}/synthetic_w24_affinity.sbatch")"
    job_id="${submitted%%;*}"
    echo "submitted matrix=${matrix_id} rep=${repetition} job=${job_id} dependency=${previous_job:-none}"
    previous_job="${job_id}"
    job_count=$((job_count + 1))
done

echo "Submitted ${job_count} serial W24 affinity diagnostic jobs (${matrix_id})."
echo "All logs and results: ${output_dir}"
echo "Graceful stop sentinel: ${stop_file}"
