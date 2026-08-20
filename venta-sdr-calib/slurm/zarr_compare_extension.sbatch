#!/bin/bash
#SBATCH --job-name=cal-zarr-ext
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
    echo "Skipping job because the extension stop sentinel exists: ${BENCH_STOP_FILE}"
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
source "${COMMON_SH}"
setup_benchmark_environment

workers="${WORKERS}"
repetition="${SLURM_ARRAY_TASK_ID}"
if ((repetition < 1 || repetition > 5)); then
    echo "ERROR: repetition must be 1 through 5" >&2
    exit 2
fi

case "${CPU_REGIME}" in
    physical-core)
        if ((workers > 35)); then
            echo "ERROR: physical-core runs support at most 35 workers plus control" >&2
            exit 2
        fi
        DASK_WORKER_AFFINITY=distinct-cores
        ;;
    smt)
        if ((workers < 36 || workers > 71)); then
            echo "ERROR: SMT extension requires 36--71 workers plus control" >&2
            exit 2
        fi
        DASK_WORKER_AFFINITY=distinct-cpus
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
BLOSC_THREADS_PER_WORKER=1
NUMBA_MODE=on

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
matrix_id="${MATRIX_ID:-zarr_compare_extension_${array_job_id}}"
block_id="${matrix_id}_${CPU_REGIME}_w${workers}_rep${repetition}"

# Rotate the four framework x format conditions within every matched block.
CONDITIONS=(mpi:raw dask:zarr dask:raw mpi:zarr)
rotation=$(((repetition - 1) % ${#CONDITIONS[@]}))
for ((position = 0; position < ${#CONDITIONS[@]}; position++)); do
    condition="${CONDITIONS[$(((position + rotation) % ${#CONDITIONS[@]}))]}"
    framework="${condition%%:*}"
    input_format="${condition##*:}"
    run_id="${block_id}_${framework}_${input_format}_numba-on"
    run_condition \
        "${framework}" real "${workers}" on "${PRECISION}" \
        "${run_id}" "${block_id}" "${repetition}" "$((position + 1))" \
        "${input_format}"
done

records=(
    "${OUTPUT_DIR}/${block_id}_mpi_raw_numba-on.json"
    "${OUTPUT_DIR}/${block_id}_dask_zarr_numba-on.json"
    "${OUTPUT_DIR}/${block_id}_dask_raw_numba-on.json"
    "${OUTPUT_DIR}/${block_id}_mpi_zarr_numba-on.json"
)
compare_matched_runs "${records[@]}"
python "${BENCH_DIR}/validate_cpu_affinity.py" \
    --cpu-regime "${CPU_REGIME}" --expected-workers "${workers}" \
    "${records[@]}"

echo "Completed matched Raw/Zarr extension block ${block_id} on $(hostname)."
