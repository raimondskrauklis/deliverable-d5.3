#!/bin/bash
# Exercise both physical-core and SMT affinity paths on a bounded group subset.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
source "${env_file}"

pilot_id="${PILOT_ID:-zarr_compare_extension_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${ZARR_COMPARE_LOG_DIR:-${BENCH_OUTPUT_ROOT}/${pilot_id}}"
stop_file="${output_dir}/STOP"
groups="${ZARR_EXTENSION_PILOT_GROUPS:-4}"
[[ "${groups}" =~ ^[1-8]$ ]] || {
    echo "ERROR: ZARR_EXTENSION_PILOT_GROUPS must be between 1 and 8" >&2
    exit 2
}
mkdir -p "${output_dir}"

points=(35:physical-core 71:smt)
previous_job=""
for point in "${points[@]}"; do
    workers="${point%%:*}"
    cpu_regime="${point##*:}"
    tasks=$((workers + 1))
    if [[ "${cpu_regime}" == "physical-core" ]]; then
        hint=nomultithread
    else
        hint=multithread
    fi
    submit_args=(
        --parsable
        --ntasks="${tasks}"
        --hint="${hint}"
        --array=1
        --time="${ZARR_EXTENSION_PILOT_TIME:-02:00:00}"
        --output="${output_dir}/slurm-%x-%A_%a.out"
        --error="${output_dir}/slurm-%x-%A_%a.err"
        --export="ALL,WORKERS=${workers},CPU_REGIME=${cpu_regime},REAL_GROUPS=${groups},REAL_CONDITION_TIMEOUT=30m,MATRIX_ID=${pilot_id},OUTPUT_DIR=${output_dir},BENCH_STOP_FILE=${stop_file},ZARR_DIR=${ZARR_DIR}"
    )
    if [[ -n "${BENCH_NODELIST:-}" ]]; then
        submit_args+=(--nodelist="${BENCH_NODELIST}")
    fi
    if [[ -n "${previous_job}" ]]; then
        submit_args+=(--dependency="afterok:${previous_job}")
    fi
    submitted="$(sbatch "${submit_args[@]}" "${script_dir}/zarr_compare_extension.sbatch")"
    job_id="${submitted%%;*}"
    echo "submitted pilot=${pilot_id} workers=${workers} cpu_regime=${cpu_regime} groups=${groups} job=${job_id} dependency=${previous_job:-none}"
    previous_job="${job_id}"
done

echo "Pilot logs and results: ${output_dir}"
echo "Submit the full extension only after both pilot blocks pass equivalence and affinity validation."
echo "LAST_JOB_ID=${previous_job}"
