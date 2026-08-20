#!/bin/bash
#SBATCH --job-name=cal-synth-affinity
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --array=1-5%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

if [[ -n "${BENCH_STOP_FILE:-}" && -e "${BENCH_STOP_FILE}" ]]; then
    echo "Skipping job because the diagnostic stop sentinel exists: ${BENCH_STOP_FILE}"
    exit 0
fi

if [[ -n "${BENCH_DIR:-}" ]]; then
    COMMON_SH="${BENCH_DIR}/slurm/common.sh"
elif [[ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]]; then
    COMMON_SH="${SLURM_SUBMIT_DIR}/slurm/common.sh"
else
    echo "ERROR: submit from framework_benchmark/ or export BENCH_DIR" >&2
    exit 2
fi
source "${COMMON_SH}"
setup_benchmark_environment

workers=24
repetition="${SLURM_ARRAY_TASK_ID}"
if ((repetition < 1 || repetition > 5)); then
    echo "ERROR: repetition must be 1 through 5" >&2
    exit 2
fi

CPU_REGIME=physical-core
PRECISION=f64
NUMBA_MODE=off
SYNTH_GROUPS=320
SYNTH_SAMPLES_PER_FILE=94076928
SYNTH_GENERATION=cached
SYNTH_CONDITION_TIMEOUT="${SYNTH_CONDITION_TIMEOUT:-75m}"

array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
matrix_id="${MATRIX_ID:-synthetic_w24_affinity_${array_job_id}}"
BLOCK_ID="${matrix_id}_w24_rep${repetition}"

CONDITIONS=("mpi:shared" "dask:shared" "dask:distinct-cores")
rotation=$(((repetition - 1) % ${#CONDITIONS[@]}))
RECORDS=()

for ((position = 0; position < ${#CONDITIONS[@]}; position++)); do
    condition="${CONDITIONS[$(((position + rotation) % ${#CONDITIONS[@]}))]}"
    framework="${condition%%:*}"
    affinity="${condition##*:}"
    if [[ "${framework}" == "mpi" ]]; then
        label="mpi_baseline"
    elif [[ "${affinity}" == "shared" ]]; then
        label="dask_shared"
    else
        label="dask_pinned"
    fi
    run_id="${BLOCK_ID}_${label}"
    DASK_WORKER_AFFINITY="${affinity}" run_condition \
        "${framework}" synthetic "${workers}" "${NUMBA_MODE}" "${PRECISION}" \
        "${run_id}" "${BLOCK_ID}" "${repetition}" "$((position + 1))"
    RECORDS+=("${OUTPUT_DIR}/${run_id}.json")
done

compare_matched_runs "${RECORDS[@]}"
python "${BENCH_DIR}/validate_cpu_affinity.py" \
    --cpu-regime physical-core --expected-workers "${workers}" \
    "${RECORDS[@]}"

echo "Completed W24 affinity diagnostic block ${BLOCK_ID} on $(hostname)."
