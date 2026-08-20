#!/bin/bash
# Site configuration for the single-station benchmark on the current cluster.
# The submission helpers source this file automatically. Values already set in
# the shell take precedence, so temporary overrides do not require editing it.

export BENCH_DIR="${BENCH_DIR:-${HOME}/paper_runs/framework_benchmark}"
export BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-${HOME}/paper_runs/results}"
export DATA_DIR="${DATA_DIR:-/var/scratch/${USER}/rlmi_f1666_ir_24}"
export LOG_FILE="${LOG_FILE:-/var/scratch/${USER}/rlmi_f1666_ir_24.log}"
export ZARR_DIR="${ZARR_DIR:-/var/scratch/${USER}/rlmi_f1666_ir_24_zarr}"
export VENV_ACTIVATE="${VENV_ACTIVATE:-${HOME}/paper_runs/benchmark-venv/bin/activate}"
export BENCH_MODULES="${BENCH_MODULES:-spack/2026.05.13 python/3.13.8 openmpi/5.0.10}"
export BENCH_NODELIST="${BENCH_NODELIST:-cbt203}"

export NS="${NS:-4096}"
export FFT_BACKEND="${FFT_BACKEND:-scipy}"
export CACHE_REGIME="${CACHE_REGIME:-uncontrolled/mixed}"
export REAL_GROUPS="${REAL_GROUPS:-0}"
export REAL_GROUP_OFFSET="${REAL_GROUP_OFFSET:-0}"
export REAL_CONDITION_TIMEOUT="${REAL_CONDITION_TIMEOUT:-2h}"
export DASK_STARTUP_MAX_ATTEMPTS="${DASK_STARTUP_MAX_ATTEMPTS:-2}"
export DASK_STARTUP_RETRY_WINDOW_S="${DASK_STARTUP_RETRY_WINDOW_S:-120}"
export DASK_STARTUP_RETRY_DELAY_S="${DASK_STARTUP_RETRY_DELAY_S:-5}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This file configures the current shell when sourced:"
    echo "  source ${BASH_SOURCE[0]}"
    echo "The extension submission helpers source it automatically."
fi
