#!/bin/bash
#
# Smoke test for submit_scheduler.sh: submits a real scheduler job with
# known-good values, to sanity-check the submission path end-to-end
# (preflight checks, sbatch invocation, log path resolution).
#
# Usage:
#   ./test_submit_scheduler.sh [--zarr3]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# submit_scheduler.sh sources daskpoc.env (if present) before checking
# required variables, and daskpoc.env only fills in variables that aren't
# already set — so if you need this smoke test to use values other than
# what's in your daskpoc.env, uncomment and edit the exports below; being
# exported here beforehand, they take precedence over daskpoc.env.
#export DASKPOC_INSTALL_DIR=/work/scitas-ge/orliac/KUMA_VENVS/DASK-CUDA_26.04
#export DASKPOC_SCHEDULER_DIR=/scratch/orliac/daskpoc/scheduler
#export DASKPOC_DASK_LOGS_DIR=/scratch/orliac/daskpoc/output

"sh" "$SCRIPT_DIR/submit_scheduler.sh" "$@"
