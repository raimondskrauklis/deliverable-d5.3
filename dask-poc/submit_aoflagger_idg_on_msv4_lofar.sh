#!/bin/bash
#
# NOTE: the srun flags used below to probe the Dask cluster (partition,
# --gres=gpu:1) are specific to the EPFL Kuma GPU cluster. Tune them for
# your own site before running this elsewhere.
#
# Requires (export before running, or set in daskpoc.env — see
# daskpoc.env.example, sourced automatically if daskpoc.env is present):
#   DASKPOC_ZARR3          1 = Zarr 3, 0/unset = Zarr 2
#   DASKPOC_INSTALL_DIR    Venv install directory for the chosen flavor
#                          (same value used to launch the scheduler/workers — see README.md)
#   DASKPOC_SCHEDULER_DIR  Directory holding the running scheduler's state
#                          (same value used to launch the scheduler/workers — see README.md)
#   DASKPOC_OUTPUT_DIR     Where AOFlagger/IDG results get written
#   DASKPOC_DATA_REPO      Base directory holding the MSv4/LOFAR Zarr datasets (read-only input)
#   DASKPOC_DATA_PROC      Scratch base directory to copy/process datasets into

set -euo pipefail

trap 'rc=$?; echo "Error: command failed with exit code $rc at line $LINENO: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$SCRIPT_DIR/daskpoc.env" ] && source "$SCRIPT_DIR/daskpoc.env"

: "${DASKPOC_INSTALL_DIR:?DASKPOC_INSTALL_DIR must be set (venv install directory) — export it before running this script}"
: "${DASKPOC_SCHEDULER_DIR:?DASKPOC_SCHEDULER_DIR must be set (scheduler state directory) — export it before running this script}"
: "${DASKPOC_OUTPUT_DIR:?DASKPOC_OUTPUT_DIR must be set (AOFlagger/IDG results directory) — export it before running this script}"
: "${DASKPOC_DATA_REPO:?DASKPOC_DATA_REPO must be set (MSv4/LOFAR Zarr dataset repository) — export it before running this script}"
: "${DASKPOC_DATA_PROC:?DASKPOC_DATA_PROC must be set (scratch directory for processing copies) — export it before running this script}"

PRECISION='single'
TELESCOPE='LOFAR'
USE_ZARR3="${DASKPOC_ZARR3:-0}"

if (( USE_ZARR3 )); then
    echo "Activating Zarr 3 environment"
    source "$SCRIPT_DIR/set_dask-cuda_26.04_zarr_3_environment.sh"
else
    echo "Using default Zarr 2 environment"
    source "$SCRIPT_DIR/set_dask-cuda_26.04_environment.sh"
fi


# --------------------------------------------#
# Get information on Dask-CUDA cluster worker #
# --------------------------------------------#

#source ./dask_worker_info.sh $SCHEDULER_FILE

# Get number of workers.
# n_workers=-1 is required; otherwise scheduler_info may return only 5 workers.

N_DASK_WORKERS=4

if [ 1 == 1 ]; then
    N_DASK_WORKERS="$(
    srun \
        --partition=h100 \
        --time 00:03:00 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --gres=gpu:1 \
        /usr/bin/env UCX_LOG_LEVEL=warn UCX_WARN_UNUSED_ENV_VARS=n \
        python - "$SCHEDULER_FILE" <<'PY'
import sys
from distributed import Client

with Client(scheduler_file=sys.argv[1]) as client:
    # Change 1 to the minimum number required before continuing.
    client.wait_for_workers(n_workers=1, timeout="120s")

    workers = client.scheduler_info(n_workers=-1)["workers"]
    print(len(workers))
PY
)"
fi

echo "N_DASK_WORKERS = >>>$N_DASK_WORKERS<<<"


OUT_REPO="$DASKPOC_OUTPUT_DIR"

DATA_REPO="$DASKPOC_DATA_REPO/${TELESCOPE}"
DATA_PROC="$DASKPOC_DATA_PROC/${TELESCOPE}"
if (( USE_ZARR3 )); then
    XRADIO_DIR="XRADIO_1.2.0"
    DATA_REPO+="/${XRADIO_DIR}"
    DATA_PROC+="/${XRADIO_DIR}"
fi

XDS_NAME="epoch_5026988680.9055605_concatenated"

prev=""

#for COMPRESSION in 'none' 'zstd_level2'; do
for COMPRESSION in 'none'; do

    # Chunks for NCHUNKS=1,2 are too large > 2GB for the codec
    #for NCHUNKS in 4 8 16 32 64; do
    for NCHUNKS in 16; do

        if (( NCHUNKS < N_DASK_WORKERS )); then
            echo "-W- Skipping case $NCHUNKS < $N_DASK_WORKERS"
            continue
        fi

        echo
        echo "$N_DASK_WORKERS, $NCHUNKS, $COMPRESSION"

        ZARR_PATTERN="${XDS_NAME}_nchunks${NCHUNKS}_cst*_csc*_csb*_${PRECISION}_${COMPRESSION}.ps.zarr"

        shopt -s nullglob
        matches=( ${DATA_REPO}/$ZARR_PATTERN )

        if (( ${#matches[@]} != 1 )); then
            printf 'Error: expected exactly one directory matching pattern:\n  %s in %s\n' "$ZARR_PATTERN" "$DATA_REPO" >&2
            printf 'Found %d matches:\n' "${#matches[@]}" >&2
            printf '  %s\n' "${matches[@]}" >&2
            exit 1
        fi

        MSV4_REPO="${matches[0]}"

        [ -d "$MSV4_REPO" ] || { echo "-E- $MSV4_REPO not found!" >&2; exit 1; }
        echo "MSV4_REPO: $MSV4_REPO"

        ZARR_NAME="$(basename "$MSV4_REPO")"
        echo "ZARR_NAME: $ZARR_NAME"

        MSV4_PROC=${DATA_PROC}/${ZARR_NAME}

        # Get info on Zarr shape and chunks
        if (( USE_ZARR3 )); then
            source "$SCRIPT_DIR/zarr_3_metrics.sh" "$MSV4_REPO/${XDS_NAME}_0" VISIBILITY
        else
            source "$SCRIPT_DIR/zarr_metrics_v2.sh" "$MSV4_REPO/${XDS_NAME}_0" VISIBILITY
        fi


        # For shorter tests, limit the number of baselines to process
        #N_BASELINES=600
        if [[ -v N_BASELINES ]]; then
            if (( N_BASELINES > ZARR_SHAPE_1 )); then
                N_BASELINES=$ZARR_SHAPE_1
            fi
        else
            N_BASELINES=$ZARR_SHAPE_1
        fi

        echo "-I- N_BASELINES = $N_BASELINES"


        # Ceiling division: ceil(a / b) = (a + b - 1) / b
        N_ZARR_CHUNKS=$(
            echo $(( (N_BASELINES + ZARR_CHUNK_1 - 1) / ZARR_CHUNK_1 ))
                     )
        
        ZARR_CHUNKS_PER_PROC=$(
            echo $(( (N_ZARR_CHUNKS + N_DASK_WORKERS - 1) / N_DASK_WORKERS ))
                            )
        
        (( ZARR_CHUNKS_PER_PROC < 1 )) && ZARR_CHUNKS_PER_PROC=1
        
        PROC_CHUNK_1=$((ZARR_CHUNKS_PER_PROC * ZARR_CHUNK_1))
        
        export PROC_CHUNK_1
        
        printf '%-24s %d\n' \
               "N_BASELINES" "$N_BASELINES" \
               "ZARR_CHUNK_1" "$ZARR_CHUNK_1" \
               "N_DASK_WORKERS" "$N_DASK_WORKERS" \
               "N_ZARR_CHUNKS" "$N_ZARR_CHUNKS" \
               "ZARR_CHUNKS_PER_PROC" "$ZARR_CHUNKS_PER_PROC" \
               "PROC_CHUNK_1" "$PROC_CHUNK_1"


        PROC_CHUNKING="$ZARR_SHAPE_0 $PROC_CHUNK_1 $ZARR_SHAPE_2 $ZARR_SHAPE_3"
        PROC_CHUNKING_TAG="${PROC_CHUNKING// /_}"
        echo "-I- PROC_CHUNKING = $PROC_CHUNKING derived from Zarr"

        OUT_DIR=$OUT_REPO/w${N_DASK_WORKERS}/zc${NCHUNKS}/pc${PROC_CHUNKING_TAG}
        
        #for WSC_SIZE in 2048 4096 8192 16384; do
        for WSC_SIZE in 4096; do

            # Sweep the AOFlagger thread count per chunk, e.g. '' 1 2 4 8 16
            # (leave '' to let aoflagger_idg_on_msv4.py auto-size it from
            # SLURM_CPUS_PER_TASK on the worker instead of overriding it).
            for N_THREADS in ''; do

                OUT_TAG="w${N_DASK_WORKERS}_zc${NCHUNKS}_cs${PROC_CHUNKING_TAG}_${COMPRESSION}_px${WSC_SIZE}_nt${N_THREADS:-auto}"

                opt_pars="--parsable "
                if [ ! -z "$prev" ]; then
                    opt_pars+="--dependency=afterok:$prev "
                fi

                CMD=""
                CMD+="sbatch "
                CMD+="--qos scitas "
                CMD+="--chdir $OUT_DIR "
                CMD+="--output slurm-%j_${OUT_TAG}.out "
                CMD+="$opt_pars "
                CMD+="$SCRIPT_DIR/aoflagger_idg_on_msv4_lofar.sh "
                if (( USE_ZARR3 )); then
                    CMD+="--zarr3 "
                fi
                CMD+="--compression $COMPRESSION "
                CMD+="--nchunks $NCHUNKS "
                CMD+="--msv4_repo $MSV4_REPO "
                CMD+="--data_proc $DATA_PROC "
                CMD+="--zarr_name $ZARR_NAME "
                CMD+="--xds_name $XDS_NAME "
                CMD+="--proc_chunking \"$PROC_CHUNKING\" "
                CMD+="--out_dir $OUT_DIR "
                CMD+="--out_tag $OUT_TAG "
                CMD+="--nb $N_BASELINES "
                CMD+="--wsc_size $WSC_SIZE"
                if [ -n "$N_THREADS" ]; then
                    CMD+=" --n_threads $N_THREADS"
                fi

                echo "CMD = $CMD"

                jid=$(eval "$CMD") || exit 1
                echo "jid >>>$jid<<<"

                prev="$jid"

            done

        done
        
    done
    
done
