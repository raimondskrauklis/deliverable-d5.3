#!/bin/bash -l
#
# NOTE: the #SBATCH options below (partition, GPU/CPU sizing, QOS) and the
# --interface bond1 passed to `dask worker` further down are specific to
# the EPFL Kuma GPU cluster. Tune them for your own site before running
# this elsewhere.
#
# Worker count isn't fixed here — submit via submit_workers.sh, which
# requires --ntasks (and accepts --ntasks-per-node, --zarr3).

#SBATCH --partition h100
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-gpu 94400
#SBATCH --time 00-08:00:00
# SBATCH --account gr-et-radioastro
#SBATCH --qos scitas
# --output is set by submit_workers.sh (via DASKPOC_DASK_LOGS_DIR), not hardcoded here

# SBATCH --comment scitas_hwperf
# SBATCH --exclusive

set -euo pipefail

trap 'rc=$?; echo "Error: command failed with exit code $rc at line $LINENO: $BASH_COMMAND" >&2' ERR


SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

USE_ZARR3=0

for arg in "$@"; do
    case "$arg" in
        --zarr3)
            USE_ZARR3=1
            ;;
        *)
            echo "Error: unrecognized argument '$arg'" >&2
            exit 1
            ;;
    esac
done

if (( USE_ZARR3 )); then
    echo "Activating Zarr 3 environment"
    source "$SCRIPT_DIR/set_dask-cuda_26.04_zarr_3_environment.sh"
else
    echo "Using default Zarr 2 environment"
    source "$SCRIPT_DIR/set_dask-cuda_26.04_environment.sh"
fi


srun hostname

srun bash -lc \
     'echo "task=$SLURM_PROCID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'


: "${SCHEDULER_FILE:?SCHEDULER_FILE is not set}"
[ -f "$SCHEDULER_FILE" ] || { echo "-E- SCHEDULER_FILE $SCHEDULER_FILE not found." >&2; exit 1; }

# /!\ Don't use $OMP_NUM_THREADS for --nthreads in order to keep multithreaded worker as working unit
export UCX_LOG_LEVEL=info

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


srun \
  --ntasks="$SLURM_NTASKS" \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus-per-task=1 \
  --cpu-bind=cores \
  --gpu-bind=single:1 \
  bash -c '
    echo "rank=$SLURM_PROCID localid=$SLURM_LOCALID host=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "parent affinity:"
    taskset -cp $$

    dask worker \
      --scheduler-file "$SCHEDULER_FILE" \
      --nthreads 1 \
      --name "gpu-${SLURM_PROCID}" \
      --memory-limit 90GB \
      --protocol ucx \
      --interface bond1
  '
