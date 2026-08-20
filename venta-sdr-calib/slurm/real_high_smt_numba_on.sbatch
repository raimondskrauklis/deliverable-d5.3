#!/bin/bash
#SBATCH --job-name=cal-high-smt
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --hint=multithread
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --array=1-5%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

if [[ -n "${BENCH_STOP_FILE:-}" && -e "${BENCH_STOP_FILE}" ]]; then
    echo "Skipping job because the matrix stop sentinel exists: ${BENCH_STOP_FILE}"
    exit 0
fi

: "${WORKERS:?WORKERS must be exported by the submission helper}"
CPU_REGIME=smt

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

workers="${WORKERS}"
repetition="${SLURM_ARRAY_TASK_ID}"
if ((workers != 64 && workers != 71)); then
    echo "ERROR: high-SMT follow-up supports 64 or 71 workers" >&2
    exit 2
fi
if ((repetition < 1 || repetition > 5)); then
    echo "ERROR: repetition must be 1 through 5" >&2
    exit 2
fi

REAL_GROUPS="${REAL_GROUPS:-0}"
REAL_GROUP_OFFSET="${REAL_GROUP_OFFSET:-0}"
IO_LEASES=0
PRECISION=f64
NUMBA_MODE=on

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
matrix_id="${MATRIX_ID:-real_high_smt_numba_on_${array_job_id}}"
BLOCK_ID="${matrix_id}_smt_w${workers}_rep${repetition}"

# Alternate framework order across repetitions. This is a fresh paired design;
# it does not mix order-biased fragments from the interrupted four-condition run.
if ((repetition % 2)); then
    FRAMEWORKS=(mpi dask)
else
    FRAMEWORKS=(dask mpi)
fi

for ((position = 0; position < ${#FRAMEWORKS[@]}; position++)); do
    framework="${FRAMEWORKS[$position]}"
    run_id="${BLOCK_ID}_${framework}_numba-on"
    run_condition \
        "${framework}" real "${workers}" "${NUMBA_MODE}" "${PRECISION}" \
        "${run_id}" "${BLOCK_ID}" "${repetition}" "$((position + 1))"
done

compare_matched_runs \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_numba-on.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_numba-on.json"

python "${BENCH_DIR}/validate_cpu_affinity.py" \
    --cpu-regime smt --expected-workers "${workers}" \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_numba-on.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_numba-on.json"

echo "Completed paired Numba-on high-SMT block ${BLOCK_ID} on $(hostname)."
