#!/bin/bash
# Submit a two-group, 71-worker SMT affinity and correctness pilot.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${env_file}"

pilot_id="${PILOT_ID:-smt71_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-${BENCH_OUTPUT_ROOT}/${pilot_id}}"
stop_file="${output_dir}/STOP"
mkdir -p "${output_dir}"

submit_args=(
    --parsable
    --ntasks=72
    --hint=multithread
    --array=1
    --time=00:30:00
    --output="${output_dir}/slurm-%x-%A_%a.out"
    --error="${output_dir}/slurm-%x-%A_%a.err"
    --export="ALL,WORKERS=71,CPU_REGIME=smt,REAL_GROUPS=2,MATRIX_ID=${pilot_id},OUTPUT_DIR=${output_dir},BENCH_STOP_FILE=${stop_file}"
)
if [[ -n "${BENCH_NODELIST:-}" ]]; then
    submit_args+=(--nodelist="${BENCH_NODELIST}")
fi

# Forward optional sbatch arguments; later values override earlier defaults.
submitted="$(sbatch "${submit_args[@]}" "$@" "${script_dir}/real_extension.sbatch")"
job_id="${submitted%%;*}"
echo "submitted pilot=${pilot_id} workers=71 cpu_regime=smt job=${job_id}"
echo "All logs and results: ${output_dir}"
