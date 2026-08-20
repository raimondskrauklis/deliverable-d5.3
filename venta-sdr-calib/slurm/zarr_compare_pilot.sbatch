#!/bin/bash
#SBATCH --job-name=cal-zarr-compare-pilot
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
source "${COMMON_SH}"
setup_benchmark_environment

# Use pilot-specific names so a general REAL_GROUPS=0 (meaning all groups in
# publication jobs) cannot silently turn this correctness check into a full run.
PILOT_WORKERS=2
REAL_GROUPS="${ZARR_PILOT_GROUPS:-2}"
REAL_GROUP_OFFSET="${ZARR_PILOT_GROUP_OFFSET:-0}"
if ! [[ "${REAL_GROUPS}" =~ ^[1-4]$ ]]; then
    echo "ERROR: ZARR_PILOT_GROUPS must be between 1 and 4" >&2
    exit 2
fi
if ! [[ "${REAL_GROUP_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ZARR_PILOT_GROUP_OFFSET must be non-negative" >&2
    exit 2
fi
IO_LEASES=0
PRECISION=f64
BLOSC_THREADS_PER_WORKER=1
DASK_WORKER_AFFINITY=distinct-cores

block_id="${PILOT_ID:-zarr_compare_pilot_${SLURM_JOB_ID}}"
order_index=0

run_pilot() {
    local framework="$1"
    local input_format="$2"
    local numba_mode="$3"
    order_index=$((order_index + 1))
    local run_id="${block_id}_${framework}_${input_format}_numba-${numba_mode}"
    run_condition \
        "${framework}" real "${PILOT_WORKERS}" "${numba_mode}" "${PRECISION}" \
        "${run_id}" "${block_id}" 1 "${order_index}" "${input_format}"
}

# Exercise both numerical paths. The pilot is a correctness gate, not a
# performance result, so the straightforward alternating order is sufficient.
run_pilot mpi  raw  off
run_pilot dask zarr off
run_pilot dask raw  off
run_pilot mpi  zarr off
run_pilot mpi  zarr on
run_pilot dask raw  on
run_pilot dask zarr on
run_pilot mpi  raw  on

compare_matched_runs \
    "${OUTPUT_DIR}/${block_id}_mpi_raw_numba-off.json" \
    "${OUTPUT_DIR}/${block_id}_dask_zarr_numba-off.json" \
    "${OUTPUT_DIR}/${block_id}_dask_raw_numba-off.json" \
    "${OUTPUT_DIR}/${block_id}_mpi_zarr_numba-off.json" \
    "${OUTPUT_DIR}/${block_id}_mpi_zarr_numba-on.json" \
    "${OUTPUT_DIR}/${block_id}_dask_raw_numba-on.json" \
    "${OUTPUT_DIR}/${block_id}_dask_zarr_numba-on.json" \
    "${OUTPUT_DIR}/${block_id}_mpi_raw_numba-on.json"

echo "All Raw/Zarr, MPI/Dask, and Numba equivalence checks passed."
