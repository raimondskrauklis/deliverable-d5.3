#!/bin/bash
#SBATCH --job-name=cal-zarr-compare
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --array=1-5%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

if [[ -n "${BENCH_STOP_FILE:-}" && -e "${BENCH_STOP_FILE}" ]]; then
    echo "Skipping job because the matrix stop sentinel exists: ${BENCH_STOP_FILE}"
    exit 0
fi

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

workers="${WORKERS:-32}"
repetition="${SLURM_ARRAY_TASK_ID}"
if ((repetition < 1 || repetition > 5)); then
    echo "ERROR: repetition must be 1 through 5" >&2
    exit 2
fi

REAL_GROUPS="${REAL_GROUPS:-0}"
REAL_GROUP_OFFSET="${REAL_GROUP_OFFSET:-0}"
IO_LEASES=0
PRECISION=f64
BLOSC_THREADS_PER_WORKER=1
DASK_WORKER_AFFINITY=distinct-cores
NUMBA_MODE="${ZARR_NUMBA_MODE:-on}"
if [[ "${NUMBA_MODE}" != "on" && "${NUMBA_MODE}" != "off" ]]; then
    echo "ERROR: ZARR_NUMBA_MODE must be on or off" >&2
    exit 2
fi

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
matrix_id="${MATRIX_ID:-zarr_compare_${array_job_id}}"
block_id="${matrix_id}_w${workers}_rep${repetition}"

# Rotate all four framework x storage conditions to counterbalance cache/order.
CONDITIONS=(mpi:raw dask:zarr dask:raw mpi:zarr)
rotation=$(((repetition - 1) % ${#CONDITIONS[@]}))
for ((position = 0; position < ${#CONDITIONS[@]}; position++)); do
    condition="${CONDITIONS[$(((position + rotation) % ${#CONDITIONS[@]}))]}"
    framework="${condition%%:*}"
    input_format="${condition##*:}"
    run_id="${block_id}_${framework}_${input_format}_numba-${NUMBA_MODE}"
    run_condition \
        "${framework}" real "${workers}" "${NUMBA_MODE}" "${PRECISION}" \
        "${run_id}" "${block_id}" "${repetition}" "$((position + 1))" \
        "${input_format}"
done

compare_matched_runs \
    "${OUTPUT_DIR}/${block_id}_mpi_raw_numba-${NUMBA_MODE}.json" \
    "${OUTPUT_DIR}/${block_id}_dask_zarr_numba-${NUMBA_MODE}.json" \
    "${OUTPUT_DIR}/${block_id}_dask_raw_numba-${NUMBA_MODE}.json" \
    "${OUTPUT_DIR}/${block_id}_mpi_zarr_numba-${NUMBA_MODE}.json"

echo "Completed matched Raw/Zarr block ${block_id} on $(hostname)."
