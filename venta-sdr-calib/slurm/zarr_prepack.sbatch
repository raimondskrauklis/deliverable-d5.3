#!/bin/bash
#SBATCH --job-name=cal-zarr-prepack
#SBATCH --partition=defq
#SBATCH --constraint=skylake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
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
source "${COMMON_SH}"
setup_benchmark_environment

ZARR_DIR="${ZARR_DIR:-/var/scratch/${USER}/rlmi_f1666_ir_24_zarr}"
ZARR_GROUPS="${ZARR_GROUPS:-4}"
ZARR_GROUP_OFFSET="${ZARR_GROUP_OFFSET:-0}"
ZARR_BLOSC_THREADS="${ZARR_BLOSC_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
ZARR_COMPRESSION_LEVEL="${ZARR_COMPRESSION_LEVEL:-9}"
ZARR_MIN_FREE_GIB="${ZARR_MIN_FREE_GIB:-20}"
ZARR_TIMEOUT="${ZARR_TIMEOUT:-11.5h}"

python -c "import zarr, numcodecs; assert zarr.__version__.split('.')[0] == '2'; print({'zarr': zarr.__version__, 'numcodecs': numcodecs.__version__})"
mkdir -p "${ZARR_DIR}"

echo "=== ZARR PREPARATION ==="
echo "raw_dir=${DATA_DIR}"
echo "zarr_dir=${ZARR_DIR}"
echo "groups=${ZARR_GROUPS} offset=${ZARR_GROUP_OFFSET}"
echo "blosc_threads=${ZARR_BLOSC_THREADS} compression_level=${ZARR_COMPRESSION_LEVEL}"
echo "timeout=${ZARR_TIMEOUT}"
df -hT "${ZARR_DIR}"
du -sh "${ZARR_DIR}" || true

timeout --signal=TERM --kill-after=2m "${ZARR_TIMEOUT}" \
    srun --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --hint=nomultithread --cpu-bind=cores --kill-on-bad-exit=1 \
    python -u "${BENCH_DIR}/prepare_zarr.py" "${DATA_DIR}" "${ZARR_DIR}" \
    --n-groups "${ZARR_GROUPS}" --group-offset "${ZARR_GROUP_OFFSET}" \
    --compressor zstd --compression-level "${ZARR_COMPRESSION_LEVEL}" \
    --blosc-threads "${ZARR_BLOSC_THREADS}" --min-free-gib "${ZARR_MIN_FREE_GIB}"

echo "=== ZARR PREPARATION COMPLETE ==="
du -sh "${ZARR_DIR}"
df -hT "${ZARR_DIR}"
