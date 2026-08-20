#!/bin/bash
# Submit the physical-core endpoint and SMT extension as one safe serial chain.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${env_file}"

read -r -a repetitions <<< "${BENCH_REPETITIONS:-1 2 3 4 5}"
read -r -a extension_points <<< "${BENCH_EXTENSION_POINTS:-35:physical-core 48:smt 64:smt 71:smt}"

matrix_id="${MATRIX_ID:-real_extension_$(date -u +%Y%m%dT%H%M%SZ)}"
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
    rotation=$((repetition - 1))
    for ((position = 0; position < ${#extension_points[@]}; position++)); do
        index=$(((position + rotation) % ${#extension_points[@]}))
        point="${extension_points[$index]}"
        workers="${point%%:*}"
        cpu_regime="${point##*:}"
        if ! [[ "${workers}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: invalid extension point: ${point}" >&2
            exit 2
        fi
        case "${cpu_regime}" in
            physical-core) hint=nomultithread ;;
            smt) hint=multithread ;;
            *) echo "ERROR: invalid extension point: ${point}" >&2; exit 2 ;;
        esac
        tasks=$((workers + 1))
        submit_args=(
            --parsable
            --ntasks="${tasks}"
            --hint="${hint}"
            --array="${repetition}"
            --output="${output_dir}/slurm-%x-%A_%a.out"
            --error="${output_dir}/slurm-%x-%A_%a.err"
            --export="ALL,WORKERS=${workers},CPU_REGIME=${cpu_regime},MATRIX_ID=${matrix_id},OUTPUT_DIR=${output_dir},BENCH_STOP_FILE=${stop_file}"
        )
        if [[ -n "${BENCH_NODELIST:-}" ]]; then
            submit_args+=(--nodelist="${BENCH_NODELIST}")
        fi
        if [[ -n "${previous_job}" ]]; then
            submit_args+=(--dependency="afterany:${previous_job}")
        fi
        submitted="$(sbatch "${submit_args[@]}" "${script_dir}/real_extension.sbatch")"
        job_id="${submitted%%;*}"
        job_count=$((job_count + 1))
        echo "submitted matrix=${matrix_id} rep=${repetition} workers=${workers} cpu_regime=${cpu_regime} tasks=${tasks} job=${job_id} dependency=${previous_job:-none}"
        previous_job="${job_id}"
    done
done

echo "Submitted ${job_count} extension jobs in one serial afterany chain (${matrix_id})."
echo "All logs and results: ${output_dir}"
echo "Graceful stop sentinel: ${stop_file}"
