#!/bin/bash
#
# NOTE: the #SBATCH options below (partition, account, mem/cpu/gpu sizing)
# and the --interface bond1 passed to `dask scheduler` further down are
# specific to the EPFL Kuma GPU cluster. Tune them for your own site
# before running this elsewhere.

#SBATCH --ntasks 1
#SBATCH --time 00-08:00:00
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-task 1
#SBATCH --mem 94400
#SBATCH --partition h100
#SBATCH --account gr-et-radioastro
# --output is set by submit_scheduler.sh (via DASKPOC_DASK_LOGS_DIR), not hardcoded here

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


echo "-I- Python:" $(which python) "->" $(realpath "$(which python)")"," $(python -V)


# dask scheduler overwrites SCHEDULER_FILE itself on startup, so we only
# need the directory to exist, not to be wiped first.
: "${SCHEDULER_DIRECTORY:?SCHEDULER_DIRECTORY is not set}"
mkdir -p "$SCHEDULER_DIRECTORY"

dask scheduler \
     --scheduler-file "$SCHEDULER_FILE" \
     --protocol ucx \
     --interface bond1
