#!/bin/bash
# Queue the short synthetic MPI/Dask matrix, followed by a compute-matched pilot.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${BENCH_ENV_FILE:-${script_dir}/cluster_env.sh}"
if [[ ! -f "${env_file}" ]]; then
    echo "ERROR: cluster environment file not found: ${env_file}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${env_file}"

campaign_id="${SYNTH_OVERNIGHT_ID:-synthetic_overnight_$(date -u +%Y%m%dT%H%M%SZ)}"
short_id="${campaign_id}_short"
pilot_id="${campaign_id}_compute_pilot"
short_dir="${BENCH_OUTPUT_ROOT}/${short_id}"
pilot_dir="${BENCH_OUTPUT_ROOT}/${pilot_id}"

mkdir -p "${short_dir}" "${pilot_dir}"
for stop_file in "${short_dir}/STOP" "${pilot_dir}/STOP"; do
    if [[ -e "${stop_file}" ]]; then
        echo "ERROR: stop sentinel already exists: ${stop_file}" >&2
        exit 2
    fi
done

echo "Submitting short synthetic matrix on ${BENCH_NODELIST:-any matching node}..."
short_submission="$({
    MATRIX_ID="${short_id}" \
    OUTPUT_DIR="${short_dir}" \
    BENCH_STOP_FILE="${short_dir}/STOP" \
    BENCH_INITIAL_DEPENDENCY="" \
    BENCH_DEPENDENCY_MODE="afterany" \
    BENCH_WORKERS="4 8 16 24 32" \
    BENCH_REPETITIONS="1 2 3 4 5" \
    SYNTH_LABEL="short" \
    SYNTH_GROUPS="1024" \
    SYNTH_SAMPLES_PER_FILE="32768" \
    SYNTH_GENERATION="cached" \
    NUMBA_MODE="off" \
    PRECISION="f64" \
    bash "${script_dir}/submit_matrix.sh" synthetic
})"
printf '%s\n' "${short_submission}"

short_last_job="$(printf '%s\n' "${short_submission}" | awk -F= '/^LAST_JOB_ID=/{print $2}' | tail -n 1)"
if [[ ! "${short_last_job}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not recover the final short-matrix job ID" >&2
    exit 2
fi

echo "Submitting the compute-matched pilot after job ${short_last_job}..."
pilot_submission="$({
    MATRIX_ID="${pilot_id}" \
    OUTPUT_DIR="${pilot_dir}" \
    BENCH_STOP_FILE="${pilot_dir}/STOP" \
    BENCH_INITIAL_DEPENDENCY="${short_last_job}" \
    BENCH_DEPENDENCY_MODE="afterany" \
    BENCH_WORKERS="32" \
    BENCH_REPETITIONS="1" \
    SYNTH_LABEL="compute_matched_pilot" \
    SYNTH_GROUPS="32" \
    SYNTH_SAMPLES_PER_FILE="94076928" \
    SYNTH_GENERATION="cached" \
    NUMBA_MODE="off" \
    PRECISION="f64" \
    bash "${script_dir}/submit_matrix.sh" synthetic
})"
printf '%s\n' "${pilot_submission}"

pilot_last_job="$(printf '%s\n' "${pilot_submission}" | awk -F= '/^LAST_JOB_ID=/{print $2}' | tail -n 1)"
if [[ ! "${pilot_last_job}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not recover the compute-pilot job ID" >&2
    exit 2
fi

echo
echo "Submitted synthetic overnight campaign ${campaign_id}."
echo "Short matrix results: ${short_dir}"
echo "Compute pilot results: ${pilot_dir}"
echo "Final queued job: ${pilot_last_job}"
echo "Monitor: squeue -u \"${USER}\" -n cal-framework-synth"
echo "Graceful stop: touch \"${short_dir}/STOP\"; touch \"${pilot_dir}/STOP\""
