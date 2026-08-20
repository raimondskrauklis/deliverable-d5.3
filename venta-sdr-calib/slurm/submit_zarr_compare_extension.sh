#!/bin/bash
# Submit the matched physical-core endpoint and Raw/Zarr SMT extension.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
source "${env_file}"

read -r -a repetitions <<< "${BENCH_REPETITIONS:-1 2 3 4 5}"
read -r -a extension_points <<< \
    "${ZARR_EXTENSION_POINTS:-35:physical-core 48:smt 64:smt 71:smt}"

matrix_id="${MATRIX_ID:-zarr_compare_extension_$(date -u +%Y%m%dT%H%M%SZ)}"
# Deliberately ignore a stale generic OUTPUT_DIR from an earlier campaign.
output_dir="${ZARR_COMPARE_LOG_DIR:-${BENCH_OUTPUT_ROOT}/${matrix_id}}"
stop_file="${output_dir}/STOP"
mkdir -p "${output_dir}"
if [[ -e "${stop_file}" ]]; then
    echo "ERROR: stop sentinel already exists: ${stop_file}" >&2
    exit 2
fi
if compgen -G "${output_dir}/*.json" >/dev/null || \
        compgen -G "${output_dir}/*.failed.txt" >/dev/null; then
    echo "ERROR: output directory already contains benchmark records: ${output_dir}" >&2
    exit 2
fi

dependency_mode="${BENCH_DEPENDENCY_MODE:-afterany}"
if [[ "${dependency_mode}" != "afterany" && "${dependency_mode}" != "afterok" ]]; then
    echo "ERROR: BENCH_DEPENDENCY_MODE must be afterany or afterok" >&2
    exit 2
fi
previous_job="${BENCH_INITIAL_DEPENDENCY:-}"
job_count=0

for repetition in "${repetitions[@]}"; do
    [[ "${repetition}" =~ ^[1-5]$ ]] || {
        echo "ERROR: invalid repetition: ${repetition}" >&2
        exit 2
    }
    rotation=$((repetition - 1))
    for ((position = 0; position < ${#extension_points[@]}; position++)); do
        point="${extension_points[$(((position + rotation) % ${#extension_points[@]}))]}"
        workers="${point%%:*}"
        cpu_regime="${point##*:}"
        [[ "${workers}" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: invalid extension point: ${point}" >&2
            exit 2
        }
        case "${cpu_regime}" in
            physical-core)
                ((workers <= 35)) || { echo "ERROR: ${point} exceeds 35 physical workers" >&2; exit 2; }
                hint=nomultithread
                ;;
            smt)
                ((workers >= 36 && workers <= 71)) || { echo "ERROR: invalid SMT point ${point}" >&2; exit 2; }
                hint=multithread
                ;;
            *) echo "ERROR: invalid extension point: ${point}" >&2; exit 2 ;;
        esac
        tasks=$((workers + 1))
        submit_args=(
            --parsable
            --ntasks="${tasks}"
            --hint="${hint}"
            --array="${repetition}"
            --time="${ZARR_EXTENSION_JOB_TIME:-04:00:00}"
            --output="${output_dir}/slurm-%x-%A_%a.out"
            --error="${output_dir}/slurm-%x-%A_%a.err"
            --export="ALL,WORKERS=${workers},CPU_REGIME=${cpu_regime},MATRIX_ID=${matrix_id},OUTPUT_DIR=${output_dir},BENCH_STOP_FILE=${stop_file},ZARR_DIR=${ZARR_DIR}"
        )
        if [[ -n "${BENCH_NODELIST:-}" ]]; then
            submit_args+=(--nodelist="${BENCH_NODELIST}")
        fi
        if [[ -n "${previous_job}" ]]; then
            submit_args+=(--dependency="${dependency_mode}:${previous_job}")
        fi
        submitted="$(sbatch "${submit_args[@]}" "${script_dir}/zarr_compare_extension.sbatch")"
        job_id="${submitted%%;*}"
        job_count=$((job_count + 1))
        echo "submitted matrix=${matrix_id} rep=${repetition} order=$((position + 1)) workers=${workers} cpu_regime=${cpu_regime} tasks=${tasks} job=${job_id} dependency=${previous_job:-none}"
        previous_job="${job_id}"
    done
done

echo "Submitted ${job_count} matched Raw/Zarr extension jobs in one serial ${dependency_mode} chain."
echo "All logs and results: ${output_dir}"
echo "Graceful stop sentinel: ${stop_file}"
echo "LAST_JOB_ID=${previous_job}"
