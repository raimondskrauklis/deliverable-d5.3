#!/bin/bash
# Submit exact-size, serially dependent, counterbalanced matrix jobs.

set -euo pipefail

if (($# != 1)) || [[ "$1" != "real" && "$1" != "synthetic" ]]; then
    echo "Usage: $0 real|synthetic" >&2
    exit 2
fi

workload="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submission_script="${script_dir}/${workload}_matrix.sbatch"

# Override with, for example, BENCH_WORKERS="4 8 16".
read -r -a worker_counts <<< "${BENCH_WORKERS:-4 8 16 24 32}"
# A full publication run uses 1-5. A continuation after a failed dependency
# chain may use, for example, BENCH_REPETITIONS="2 3 4 5".
read -r -a repetitions <<< "${BENCH_REPETITIONS:-1 2 3 4 5}"

for workers in "${worker_counts[@]}"; do
    if ! [[ "${workers}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid worker count: ${workers}" >&2
        exit 2
    fi
done

for repetition in "${repetitions[@]}"; do
    if ! [[ "${repetition}" =~ ^[1-5]$ ]]; then
        echo "ERROR: invalid repetition: ${repetition}; expected 1 through 5" >&2
        exit 2
    fi
done

matrix_id="${MATRIX_ID:-${workload}_$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="${BENCH_OUTPUT_ROOT:-${PWD}/benchmark_results}"
output_dir="${OUTPUT_DIR:-${output_root}/${matrix_id}}"
stop_file="${BENCH_STOP_FILE:-${output_dir}/STOP}"
dependency_mode="${BENCH_DEPENDENCY_MODE:-afterany}"
if [[ "${dependency_mode}" != "afterany" && "${dependency_mode}" != "afterok" ]]; then
    echo "ERROR: BENCH_DEPENDENCY_MODE must be afterany or afterok" >&2
    exit 2
fi
mkdir -p "${output_dir}"
if [[ -e "${stop_file}" ]]; then
    echo "ERROR: stop sentinel already exists; remove it or choose a new matrix directory: ${stop_file}" >&2
    exit 2
fi
# An outer campaign can place this complete matrix after another serial chain.
# The value is a Slurm job ID, not a full dependency expression; this script
# applies the selected BENCH_DEPENDENCY_MODE consistently.
previous_job="${BENCH_INITIAL_DEPENDENCY:-}"
job_count=0

# A cyclic order gives every worker count each order position exactly once
# across five repetitions, without introducing nondeterministic submissions.
for repetition in "${repetitions[@]}"; do
    rotation=$((repetition - 1))
    for ((position = 0; position < ${#worker_counts[@]}; position++)); do
        index=$(((position + rotation) % ${#worker_counts[@]}))
        workers="${worker_counts[$index]}"
        tasks=$((workers + 1))
        submit_args=(
            --parsable
            --ntasks="${tasks}"
            --array="${repetition}"
            --output="${output_dir}/slurm-%x-%A_%a.out"
            --error="${output_dir}/slurm-%x-%A_%a.err"
            --export="ALL,WORKERS=${workers},MATRIX_ID=${matrix_id},OUTPUT_DIR=${output_dir},BENCH_STOP_FILE=${stop_file}"
        )
        if [[ -n "${BENCH_NODELIST:-}" ]]; then
            submit_args+=(--nodelist="${BENCH_NODELIST}")
        fi
        if [[ -n "${previous_job}" ]]; then
            submit_args+=(--dependency="${dependency_mode}:${previous_job}")
        fi

        submitted="$(sbatch "${submit_args[@]}" "${submission_script}")"
        job_id="${submitted%%;*}"
        job_count=$((job_count + 1))
        echo "submitted matrix=${matrix_id} rep=${repetition} order=$((position + 1)) workload=${workload} workers=${workers} tasks=${tasks} job=${job_id} dependency_mode=${dependency_mode} dependency=${previous_job:-none}"
        previous_job="${job_id}"
    done
done

echo "Submitted ${job_count} ${workload} jobs in one serial ${dependency_mode} chain (${matrix_id})."
echo "All Slurm logs and benchmark result files: ${output_dir}"
echo "Dependency policy: ${dependency_mode} (afterany continues past failed blocks)."
echo "Graceful stop sentinel: ${stop_file}"
echo "LAST_JOB_ID=${previous_job}"
