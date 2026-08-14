#!/bin/bash
#
# Submit wrapper for launch_dask-cuda_26.04_scheduler.sh.
#
# #SBATCH directives can't read shell/environment variables, so the SLURM
# output log path is resolved here (from DASKPOC_DASK_LOGS_DIR) and passed to
# sbatch on the command line instead, and the target directory is created
# up front since SLURM won't create it itself.
#
# Usage:
#   ./submit_scheduler.sh [--zarr3]
#
# Env (see daskpoc.env.example — daskpoc.env, if present, is sourced
# automatically):
#   DASKPOC_ZARR3          1 = Zarr 3, 0/unset = Zarr 2 (default unless --zarr3 given)
#   DASKPOC_DASK_LOGS_DIR   Directory for scheduler SLURM logs (default: .)
#   DASKPOC_INSTALL_DIR    Venv install directory for the chosen flavor (required)
#   DASKPOC_SCHEDULER_DIR  Directory for scheduler state (required)

set -euo pipefail

trap 'rc=$?; echo "Error: command failed with exit code $rc at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$SCRIPT_DIR/daskpoc.env" ] && source "$SCRIPT_DIR/daskpoc.env"

: "${DASKPOC_INSTALL_DIR:?DASKPOC_INSTALL_DIR must be set (venv install directory) — export it before running this script}"
: "${DASKPOC_SCHEDULER_DIR:?DASKPOC_SCHEDULER_DIR must be set (scheduler state directory) — export it before running this script}"

if [[ "${DASKPOC_ZARR3:-0}" == "1" ]]; then
    set -- --zarr3 "$@"
fi

DASKPOC_DASK_LOGS_DIR="${DASKPOC_DASK_LOGS_DIR:-.}"
mkdir -p "$DASKPOC_DASK_LOGS_DIR"

echo "-I- Scheduler SLURM logs -> $DASKPOC_DASK_LOGS_DIR"

sbatch \
    --output="$DASKPOC_DASK_LOGS_DIR/slurm-%j_scheduler.out" \
    "$SCRIPT_DIR/launch_dask-cuda_26.04_scheduler.sh" "$@"
