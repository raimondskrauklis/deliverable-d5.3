#!/bin/bash -l
#
# Zarr 3 Dask-CUDA 26.04 environment.
# Actual setup lives in set_dask-cuda_26.04_common_environment.sh, which
# requires DASKPOC_INSTALL_DIR and DASKPOC_SCHEDULER_DIR to already be
# set in the environment (no defaults) before this is sourced.

FLAVOR=zarr3

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/set_dask-cuda_26.04_common_environment.sh"
