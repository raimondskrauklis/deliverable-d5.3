#!/bin/bash
#SBATCH --job-name=cal-framework-synth
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=03:00:00
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

workers="${WORKERS:-32}"
repetition="${SLURM_ARRAY_TASK_ID}"

if ((repetition < 1 || repetition > 5)); then
    echo "ERROR: array index ${repetition} must be a repetition from 1 through 5" >&2
    exit 2
fi
PRECISION="${PRECISION:-f64}"
NUMBA_MODE="${NUMBA_MODE:-off}"
SYNTH_LABEL="${SYNTH_LABEL:-short}"
SYNTH_GROUPS="${SYNTH_GROUPS:-1024}"
SYNTH_SAMPLES_PER_FILE="${SYNTH_SAMPLES_PER_FILE:-32768}"
SYNTH_GENERATION="${SYNTH_GENERATION:-cached}"

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
matrix_id="${MATRIX_ID:-synth_${array_job_id}}"
BLOCK_ID="${matrix_id}_${SYNTH_LABEL}_w${workers}_rep${repetition}"

# Alternate which framework runs first in successive repetitions.
if ((repetition % 2)); then
    FRAMEWORKS=(mpi dask)
else
    FRAMEWORKS=(dask mpi)
fi

for ((position = 0; position < ${#FRAMEWORKS[@]}; position++)); do
    framework="${FRAMEWORKS[$position]}"
    run_id="${BLOCK_ID}_${framework}_numba-${NUMBA_MODE}_${PRECISION}"
    run_condition \
        "${framework}" synthetic "${workers}" "${NUMBA_MODE}" "${PRECISION}" \
        "${run_id}" "${BLOCK_ID}" "${repetition}" "$((position + 1))"
done

compare_matched_runs \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_numba-${NUMBA_MODE}_${PRECISION}.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_numba-${NUMBA_MODE}_${PRECISION}.json"

echo "Completed matched synthetic block ${BLOCK_ID} on $(hostname)."
