#!/bin/bash
# Submit a fresh paired Numba-on matrix at 64 and 71 SMT workers.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${env_file}"

matrix_id="${MATRIX_ID:-real_high_smt_numba_on_$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-${BENCH_OUTPUT_ROOT}/${matrix_id}}"
stop_file="${BENCH_STOP_FILE:-${output_dir}/STOP}"
mkdir -p "${output_dir}"
if [[ -e "${stop_file}" ]]; then
    echo "ERROR: stop sentinel already exists: ${stop_file}" >&2
    exit 2
fi

read -r -a repetitions <<< "${BENCH_REPETITIONS:-1 2 3 4 5}"
read -r -a worker_counts <<< "${BENCH_HIGH_SMT_WORKERS:-64 71}"
previous_job=""
job_count=0

for repetition in "${repetitions[@]}"; do
    if ! [[ "${repetition}" =~ ^[1-5]$ ]]; then
        echo "ERROR: invalid repetition: ${repetition}" >&2
        exit 2
    fi
    rotation=$((repetition % ${#worker_counts[@]}))
    for ((position = 0; position < ${#worker_counts[@]}; position++)); do
        workers="${worker_counts[$(((position + rotation) % ${#worker_counts[@]}))]}"
        if ((workers != 64 && workers != 71)); then
            echo "ERROR: BENCH_HIGH_SMT_WORKERS may contain only 64 and 71" >&2
            exit 2
        fi
        tasks=$((workers + 1))
        submit_args=(
            --parsable
            --ntasks="${tasks}"
            --hint=multithread
            --array="${repetition}"
            --output="${output_dir}/slurm-%x-%A_%a.out"
            --error="${output_dir}/slurm-%x-%A_%a.err"
            --export="ALL,WORKERS=${workers},CPU_REGIME=smt,MATRIX_ID=${matrix_id},OUTPUT_DIR=${output_dir},BENCH_STOP_FILE=${stop_file}"
        )
        if [[ -n "${BENCH_NODELIST:-}" ]]; then
            submit_args+=(--nodelist="${BENCH_NODELIST}")
        fi
        if [[ -n "${previous_job}" ]]; then
            submit_args+=(--dependency="afterany:${previous_job}")
        fi
        submitted="$(sbatch "${submit_args[@]}" "${script_dir}/real_high_smt_numba_on.sbatch")"
        job_id="${submitted%%;*}"
        job_count=$((job_count + 1))
        echo "submitted matrix=${matrix_id} rep=${repetition} workers=${workers} numba=on job=${job_id} dependency=${previous_job:-none}"
        previous_job="${job_id}"
    done
done

echo "Submitted ${job_count} paired high-SMT jobs (${matrix_id})."
echo "All logs and results: ${output_dir}"
echo "Graceful stop sentinel: ${stop_file}"
