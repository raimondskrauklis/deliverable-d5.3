#!/bin/bash
#SBATCH --job-name=cal-framework-pilot
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -n "${BENCH_DIR:-}" ]]; then
    COMMON_SH="${BENCH_DIR}/slurm/common.sh"
elif [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    COMMON_SH="${SLURM_SUBMIT_DIR}/slurm/common.sh"
elif [[ -f "${SLURM_SUBMIT_DIR}/framework_benchmark/slurm/common.sh" ]]; then
    COMMON_SH="${SLURM_SUBMIT_DIR}/framework_benchmark/slurm/common.sh"
else
    echo "ERROR: submit from framework_benchmark/ or export BENCH_DIR" >&2
    exit 2
fi
# shellcheck disable=SC1091
source "${COMMON_SH}"
setup_benchmark_environment

PILOT_WORKERS="${PILOT_WORKERS:-2}"
REAL_GROUPS="${REAL_GROUPS:-2}"
SYNTH_GROUPS="${SYNTH_GROUPS:-8}"
SYNTH_SAMPLES_PER_FILE="${SYNTH_SAMPLES_PER_FILE:-32768}"
SYNTH_GENERATION="cached"

BLOCK_ID="${PILOT_ID:-pilot_${SLURM_JOB_ID}}"
ORDER_INDEX=0

run_pilot_condition() {
    local framework="$1"
    local workload="$2"
    local numba_mode="$3"
    local precision="$4"
    local label="$5"
    ORDER_INDEX=$((ORDER_INDEX + 1))
    run_condition \
        "${framework}" "${workload}" "${PILOT_WORKERS}" \
        "${numba_mode}" "${precision}" \
        "${BLOCK_ID}_${label}" "${BLOCK_ID}" 1 "${ORDER_INDEX}"
}

# Baseline and Numba equivalence, with no I/O throttling.
IO_LEASES=0
run_pilot_condition mpi  real off f64 mpi_real_f64_off
run_pilot_condition dask real off f64 dask_real_f64_off
run_pilot_condition dask real on  f64 dask_real_f64_on
run_pilot_condition mpi  real on  f64 mpi_real_f64_on

compare_matched_runs \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_real_f64_off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_real_f64_off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_real_f64_on.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_real_f64_on.json"

# Explicit float32 optimization path.
run_pilot_condition mpi  real off f32 mpi_real_f32_off
run_pilot_condition dask real off f32 dask_real_f32_off
compare_matched_runs \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_real_f32_off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_real_f32_off.json"

# Exercise both lease implementations without making leases part of the main result.
IO_LEASES=1
run_pilot_condition dask real off f64 dask_real_f64_lease1
run_pilot_condition mpi  real off f64 mpi_real_f64_lease1
compare_matched_runs \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_real_f64_off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_real_f64_off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_real_f64_lease1.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_real_f64_lease1.json"

# Short cached synthetic scheduling path.
IO_LEASES=0
run_pilot_condition mpi  synthetic off f64 mpi_synth_f64_off
run_pilot_condition dask synthetic off f64 dask_synth_f64_off
compare_matched_runs \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_synth_f64_off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_synth_f64_off.json"

echo "All pilot conditions and equivalence checks passed."
