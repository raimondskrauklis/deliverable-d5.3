#!/bin/bash -l
#
# Shared Dask-CUDA 26.04 environment setup. Not meant to be sourced
# directly — sourced by set_dask-cuda_26.04_environment.sh (Zarr 2) and
# set_dask-cuda_26.04_zarr_3_environment.sh (Zarr 3), which each set
# FLAVOR before delegating here.
#
# Requires (caller/user-provided, no defaults):
#   DASKPOC_INSTALL_DIR   Venv install directory for this flavor
#   DASKPOC_SCHEDULER_DIR Directory for scheduler state (dask-scheduler.json)

set -euo pipefail

: "${FLAVOR:?FLAVOR must be set by the caller (zarr2/zarr3)}"
: "${DASKPOC_INSTALL_DIR:?DASKPOC_INSTALL_DIR must be set (venv install directory)}"
: "${DASKPOC_SCHEDULER_DIR:?DASKPOC_SCHEDULER_DIR must be set (scheduler state directory)}"

echo
echo "__info__ ${BASH_SOURCE[0]} start (flavor=$FLAVOR)"

ulimit -n 131072

VENV=$DASKPOC_INSTALL_DIR/VENV


# Unset Spack system config path
#
if [[ -v SPACK_SYSTEM_CONFIG_PATH ]]; then
    val_=$SPACK_SYSTEM_CONFIG_PATH
    unset SPACK_SYSTEM_CONFIG_PATH
    echo "__info__ SPACK_SYSTEM_CONFIG_PATH ($val_) unset"
else
    echo "__info__ SPACK_SYSTEM_CONFIG_PATH was not set"
fi

source ~/MYSPACK/v1.1.1/environments/kuma_111_aoflagger/activate.sh

unset UCX_HOME
UCX_ROOT=${SPACK_ENV}/.spack-env/view
export UCX_ROOT
echo "__info__ UCX_ROOT = $UCX_ROOT"

export RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true

SP=$VENV/lib/python3.13/site-packages
echo "__info__ SP = $SP"

export LD_LIBRARY_PATH="$UCX_ROOT/lib/ucx:$UCX_ROOT/lib:$UCX_ROOT/lib64:$SP/libucxx/lib64:$SP/librmm/lib64:$SP/rapids_logger/lib64:$LD_LIBRARY_PATH"
export LD_PRELOAD="$UCX_ROOT/lib/libucs.so.0:$UCX_ROOT/lib/libucm.so.0:$UCX_ROOT/lib/libuct.so.0:$UCX_ROOT/lib/libucp.so.0"

echo "-I- VENV = $VENV"
[ ! -d "$VENV" ] && python -m venv "$VENV"
source "$VENV/bin/activate"

echo "-I- Python:" $(which python) "->" $(realpath "$(which python)")"," $(python -V)


# Dask setup
#
# NOTE: intentionally not flavor-specific (same for zarr2/zarr3) for now.
export SCHEDULER_DIRECTORY="$DASKPOC_SCHEDULER_DIR"
SCHEDULER_FILE=$SCHEDULER_DIRECTORY/dask-scheduler.json
export SCHEDULER_FILE=$SCHEDULER_FILE
echo "-I- SCHEDULER_FILE = $SCHEDULER_FILE (exported, only used for distributed Dask clusters)"


# Get Spack env ucx library (needs > 1.18 accordings to RAPIDS)
export RAPIDS_NO_INITIALIZE=1
export RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true

export PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED=0

unset UCX_MEMTYPE_CACHE
export UCX_LOG_LEVEL=info
export UCX_TLS=rc,tcp,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1
export UCX_MAX_RNDV_RAILS=1
export UCX_MEMTYPE_CACHE=n

export DASK_DISTRIBUTED_UCXX__CREATE_CUDA_CONTEXT=True
export DASK_DISTRIBUTED_UCXX__CUDA_COPY=True
export DASK_DISTRIBUTED_UCXX__TCP=True
export DASK_DISTRIBUTED_UCXX__NVLINK=True
export DASK_DISTRIBUTED_UCXX__INFINIBAND=True
export DASK_DISTRIBUTED_UCXX__RDMACM=True
#export DASK_DISTRIBUTED_UCXX__RMM__POOL_SIZE=20GB
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=False


env | egrep ^DASK_DISTRIBUTED_UCXX
env | egrep ^UCX
env | egrep ^DASK

echo "__info__ ${BASH_SOURCE[0]} end (flavor=$FLAVOR)"
echo
