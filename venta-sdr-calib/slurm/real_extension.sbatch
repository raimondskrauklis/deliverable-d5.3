#!/bin/bash
#SBATCH --job-name=cal-framework-ext
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --hint=multithread
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

: "${WORKERS:?WORKERS must be exported by the submission helper}"
: "${CPU_REGIME:?CPU_REGIME must be exported by the submission helper}"

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

if ((repetition < 1 || repetition > 5)); then
    echo "ERROR: array index ${repetition} must be a repetition from 1 through 5" >&2
    exit 2
fi
case "${CPU_REGIME}" in
    physical-core)
        if ((workers > 35)); then
            echo "ERROR: cbt20x has 36 physical cores; reserve one for the controller" >&2
            exit 2
        fi
        ;;
    smt)
        if ((workers > 71)); then
            echo "ERROR: cbt20x has 72 hardware threads; reserve one for the controller" >&2
            exit 2
        fi
        ;;
    *)
        echo "ERROR: CPU_REGIME must be physical-core or smt" >&2
        exit 2
        ;;
esac

REAL_GROUPS="${REAL_GROUPS:-0}"
REAL_GROUP_OFFSET="${REAL_GROUP_OFFSET:-0}"
IO_LEASES=0
PRECISION=f64

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
matrix_id="${MATRIX_ID:-real_extension_${array_job_id}}"
BLOCK_ID="${matrix_id}_${CPU_REGIME}_w${workers}_rep${repetition}"

# Preserve the counterbalanced four-condition design used by the baseline.
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

python "${BENCH_DIR}/validate_cpu_affinity.py" \
    --cpu-regime "${CPU_REGIME}" --expected-workers "${workers}" \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_numba-off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_numba-off.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_mpi_numba-on.json" \
    "${OUTPUT_DIR}/${BLOCK_ID}_dask_numba-on.json"

echo "Completed ${CPU_REGIME} extension block ${BLOCK_ID} on $(hostname)."
