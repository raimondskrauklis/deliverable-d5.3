#!/bin/bash
#SBATCH --job-name=cal-zarr-scale
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=12:00:00
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
python -c "import zarr, numcodecs; assert zarr.__version__.split('.')[0] == '2'; print({'zarr': zarr.__version__, 'numcodecs': numcodecs.__version__})"

COMPRESS_CONFIGS="${COMPRESS_CONFIGS:-1x8,1x16,1x32,2x8,4x8,8x4}"
COMPRESS_GROUPS_PER_CONDITION="${COMPRESS_GROUPS_PER_CONDITION:-8}"
COMPRESS_REPETITIONS="${COMPRESS_REPETITIONS:-3}"
COMPRESS_GROUP_OFFSET="${COMPRESS_GROUP_OFFSET:-0}"
COMPRESS_GROUP_OFFSET_STEP="${COMPRESS_GROUP_OFFSET_STEP:-8}"
COMPRESS_LEVEL="${COMPRESS_LEVEL:-9}"
COMPRESS_MIN_FREE_GIB="${COMPRESS_MIN_FREE_GIB:-50}"
COMPRESS_CONDITION_TIMEOUT_S="${COMPRESS_CONDITION_TIMEOUT_S:-5400}"
COMPRESS_CLEANUP="${COMPRESS_CLEANUP:-always}"
COMPRESS_SCRATCH_ROOT="${COMPRESS_SCRATCH_ROOT:-/var/scratch/${USER}/framework_benchmark_compression_scaling}"
COMPRESS_STOP_FILE="${COMPRESS_STOP_FILE:-${OUTPUT_DIR}/STOP}"

echo "=== COMPRESSION SCALING ==="
echo "experiment_id=${MATRIX_ID}"
echo "raw_dir=${DATA_DIR}"
echo "protected_zarr_dir=${ZARR_DIR}"
echo "temporary_scratch_root=${COMPRESS_SCRATCH_ROOT}"
echo "durable_output_dir=${OUTPUT_DIR}"
echo "configs=${COMPRESS_CONFIGS}"
echo "groups_per_condition=${COMPRESS_GROUPS_PER_CONDITION}"
echo "repetitions=${COMPRESS_REPETITIONS}"
echo "group_offset=${COMPRESS_GROUP_OFFSET} step=${COMPRESS_GROUP_OFFSET_STEP}"
echo "compression_level=${COMPRESS_LEVEL} cleanup=${COMPRESS_CLEANUP}"
echo "stop_file=${COMPRESS_STOP_FILE}"
df -hT "$(dirname "${COMPRESS_SCRATCH_ROOT}")"

timer=()
if [[ -x /usr/bin/time ]]; then
    timer=(/usr/bin/time -v -o "${OUTPUT_DIR}/experiment.time.txt")
fi

srun --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
    --hint=nomultithread --cpu-bind=cores --kill-on-bad-exit=1 \
    "${timer[@]}" python -u "${BENCH_DIR}/compression_scaling.py" \
    "${DATA_DIR}" "${COMPRESS_SCRATCH_ROOT}" "${OUTPUT_DIR}" \
    --protected-zarr-dir "${ZARR_DIR}" \
    --experiment-id "${MATRIX_ID}" \
    --configs "${COMPRESS_CONFIGS}" \
    --groups-per-condition "${COMPRESS_GROUPS_PER_CONDITION}" \
    --repetitions "${COMPRESS_REPETITIONS}" \
    --group-offset "${COMPRESS_GROUP_OFFSET}" \
    --group-offset-step "${COMPRESS_GROUP_OFFSET_STEP}" \
    --compression-level "${COMPRESS_LEVEL}" \
    --min-free-gib "${COMPRESS_MIN_FREE_GIB}" \
    --condition-timeout-s "${COMPRESS_CONDITION_TIMEOUT_S}" \
    --cleanup "${COMPRESS_CLEANUP}" \
    --stop-file "${COMPRESS_STOP_FILE}"

echo "=== COMPRESSION SCALING COMPLETE ==="
du -sh "${COMPRESS_SCRATCH_ROOT}/${MATRIX_ID}" || true
df -hT "${COMPRESS_SCRATCH_ROOT}"
