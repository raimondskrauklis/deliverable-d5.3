#!/bin/bash
#SBATCH --job-name=cal-framework-real
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --array=1-5%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

if [[ -n "${BENCH_STOP_FILE:-}" && -e "${BENCH_STOP_FILE}" ]]; then
    echo "Skipping job because the matrix stop sentinel exists: ${BENCH_STOP_FILE}"
    exit 0
fi

# The 33-task directive is the direct 32-worker default. submit_matrix.sh
# overrides it with exactly WORKERS+1 one-CPU tasks for every submitted array.
# The Dask step clears the inherited SLURM_CPUS_PER_TASK input before reshaping
# that same CPU allocation into one task with WORKERS+1 CPUs.

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

# One worker count x five repetitions. submit_matrix.sh submits one exact-size
# array for each of 4, 8, 16, 24, and 32 workers.
workers="${WORKERS:-32}"
repetition="${SLURM_ARRAY_TASK_ID}"

if ((repetition < 1 || repetition > 5)); then
    echo "ERROR: array index ${repetition} must be a repetition from 1 through 5" >&2
    exit 2
fi
REAL_GROUPS="${REAL_GROUPS:-0}"
REAL_GROUP_OFFSET="${REAL_GROUP_OFFSET:-0}"
IO_LEASES=0
PRECISION=f64

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
matrix_id="${MATRIX_ID:-real_${array_job_id}}"
BLOCK_ID="${matrix_id}_w${workers}_rep${repetition}"

# Rotate method order by repetition. This balances cache/order effects while
# retaining a deterministic order that is also stored in every JSON record.
CONDITIONS=(mpi:off dask:off dask:on mpi:on)
rotation=$(((repetition - 1) % ${#CONDITIONS[@]}))

for ((position = 0; position < ${#CONDITIONS[@]}; position++)); do
    condition="${CONDITIONS[$(((position + rotation) % ${#CONDITIONS[@]}))]}"
    framework="${condition%%:*}"
    numba_mode="${condition##*:}"
    run_id="${BLOCK_ID}_${framework}_numba-${numba_mode}"
    run_condition \
        "${framework}" real "${workers}" "${numba_mode}" "${PRECISION}" \
        "${run_id}" "${BLOCK_ID}" "${repetition}" "$((position + 1))"
done

compare_matched_runs \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_numba-off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_numba-off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_numba-on.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_numba-on.json"

echo "Completed matched real-data block ${BLOCK_ID} on $(hostname)."
