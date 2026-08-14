#!/bin/bash
#
# Submit wrapper for launch_dask-cuda_26.04_workers.sh.
#
# #SBATCH directives can't read shell/environment variables, so the SLURM
# output log path is resolved here (from DASKPOC_DASK_LOGS_DIR) and passed to
# sbatch on the command line instead, and the target directory is created
# up front since SLURM won't create it itself.
#
# Worker count isn't fixed in the job script's own #SBATCH header, so
# --ntasks is required here and forwarded to sbatch (--ntasks-per-node
# defaults to 1).
#
# Usage:
#   ./submit_workers.sh --ntasks N [--ntasks-per-node N] [--zarr3]
#
# Env (see daskpoc.env.example — daskpoc.env, if present, is sourced
# automatically):
#   DASKPOC_ZARR3           1 = Zarr 3, 0/unset = Zarr 2 (default unless --zarr3 given)
#   DASKPOC_NTASKS          Default for --ntasks (still overridable on the command line)
#   DASKPOC_NTASKS_PER_NODE Default for --ntasks-per-node (default: 1)
#   DASKPOC_DASK_LOGS_DIR   Directory for worker SLURM logs (default: .)
#   DASKPOC_INSTALL_DIR    Venv install directory for the chosen flavor (required)
#   DASKPOC_SCHEDULER_DIR  Directory for scheduler state (required)

set -euo pipefail

trap 'rc=$?; echo "Error: command failed with exit code $rc at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$SCRIPT_DIR/daskpoc.env" ] && source "$SCRIPT_DIR/daskpoc.env"

: "${DASKPOC_INSTALL_DIR:?DASKPOC_INSTALL_DIR must be set (venv install directory) — export it before running this script}"
: "${DASKPOC_SCHEDULER_DIR:?DASKPOC_SCHEDULER_DIR must be set (scheduler state directory) — export it before running this script}"

NTASKS="${DASKPOC_NTASKS:-}"
NTASKS_PER_NODE="${DASKPOC_NTASKS_PER_NODE:-1}"
WORKER_ARGS=()

if [[ "${DASKPOC_ZARR3:-0}" == "1" ]]; then
    WORKER_ARGS+=("--zarr3")
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ntasks)
            NTASKS="$2"
            shift 2
            ;;
        --ntasks-per-node)
            NTASKS_PER_NODE="$2"
            shift 2
            ;;
        --zarr3)
            WORKER_ARGS+=("--zarr3")
            shift
            ;;
        *)
            echo "Error: unrecognized argument '$1'" >&2
            exit 1
            ;;
    esac
done

: "${NTASKS:?--ntasks is required, e.g. --ntasks 2}"

DASKPOC_DASK_LOGS_DIR="${DASKPOC_DASK_LOGS_DIR:-.}"
mkdir -p "$DASKPOC_DASK_LOGS_DIR"

echo "-I- Worker SLURM logs -> $DASKPOC_DASK_LOGS_DIR"

sbatch \
    --ntasks="$NTASKS" \
    --ntasks-per-node="$NTASKS_PER_NODE" \
    --output="$DASKPOC_DASK_LOGS_DIR/slurm-%j_workers.out" \
    "$SCRIPT_DIR/launch_dask-cuda_26.04_workers.sh" ${WORKER_ARGS[@]+"${WORKER_ARGS[@]}"}
