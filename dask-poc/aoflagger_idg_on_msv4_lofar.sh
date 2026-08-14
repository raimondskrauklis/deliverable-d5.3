#!/bin/bash
#
# NOTE: the #SBATCH options below (partition, GPU/CPU/mem sizing) are
# specific to the EPFL Kuma GPU cluster. Tune them for your own site
# before running this elsewhere.

#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 94400
#SBATCH --time 00-01:00:00
# SBATCH --account gr-et-radioastro
# SBATCH --exclusive
# SBATCH --qos scitas
# SBATCH --chdir /work/scitas-ge/orliac/RADIOBLOCKS/papers/AA_2026/garbage
# SBATCH --output debug-client.out

env | grep -E "^SLURM_"

set -euo pipefail

trap 'rc=$?; echo "Error: command failed with exit code $rc at line $LINENO: $BASH_COMMAND" >&2' ERR


# Notes
# ------------------------------------------------------------------------------------------
# ! Codec does not support buffers of > 2147483647 bytes
# ------------------------------------------------------------------------------------------
USE_ZARR3=0
NCHUNKS="" # Follows the MSv2 -> MSv4 multi-chunking strategy that splits each 3 dimensions in NCHUNKS
COMPRESSION=""
MSV4_REPO=""
DATA_PROC=""
ZARR_NAME=""
PROC_CHUNKING=""
XDS_NAME=""
OUT_DIR=""
OUT_TAG=""
NB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zarr3)
            USE_ZARR3=1
            shift
            ;;

        --nchunks)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --nchunks requires a value" >&2
                exit 1
            }

            NCHUNKS="$2"

            [[ "$NCHUNKS" =~ ^[0-9]+$ ]] || {
                echo "ERROR: --nchunks must be an integer >= 1" >&2
                exit 1
            }

            (( NCHUNKS >= 1 )) || {
                echo "ERROR: --nchunks must be >= 1" >&2
                exit 1
            }

            shift 2
            ;;
        
        --nb)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --nb requires a value" >&2
                exit 1
            }

            NB="$2"

            [[ "$NB" =~ ^[0-9]+$ ]] || {
                echo "ERROR: --nb must be an integer >= 1" >&2
                exit 1
            }

            (( NB >= 1 )) || {
                echo "ERROR: --nb must be >= 1" >&2
                exit 1
            }

            shift 2
            ;;

        --compression)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --compression requires a value: none or zstd_level2" >&2
                exit 1
            }

            COMPRESSION="$2"

            case "$COMPRESSION" in
                none|zstd_level2)
                    ;;
                *)
                    echo "ERROR: --compression must be either 'none' or 'zstd_level2'" >&2
                    exit 1
                    ;;
            esac

            shift 2
            ;;

        --msv4_repo)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --msv4_repo requires a value" >&2
                exit 1
            }
            MSV4_REPO="$2"
            shift 2
            ;;

        --data_proc)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --data_proc requires a value" >&2
                exit 1
            }
            DATA_PROC="$2"
            shift 2
            ;;

        --zarr_name)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --zarr_name requires a value" >&2
                exit 1
            }
            ZARR_NAME="$2"
            shift 2
            ;;

        --proc_chunking)
            echo ">>>$2<<<"
            [[ $# -ge 2 ]] || {
                echo "ERROR: --proc_chunking requires exactly 4 integers > 0" >&2
                exit 1
            }

            PROC_CHUNKING="$2"

            read -r -a proc_chunks <<< "$PROC_CHUNKING"

            [[ "${#proc_chunks[@]}" -eq 4 ]] || {
                echo "ERROR: --proc_chunking must contain exactly 4 integers > 0" >&2
                exit 1
            }

            for value in "${proc_chunks[@]}"; do
                [[ "$value" =~ ^[0-9]+$ ]] || {
                    echo "ERROR: --proc_chunking values must be integers > 0" >&2
                    exit 1
                }

                (( value > 0 )) || {
                    echo "ERROR: --proc_chunking values must be > 0" >&2
                    exit 1
                }
            done

            shift 2
            ;;

        --xds_name)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --xds_name requires a value" >&2
                exit 1
            }
            XDS_NAME="$2"
            shift 2
            ;;

        --out_dir)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --out_dir requires a value" >&2
                exit 1
            }
            OUT_DIR="$2"
            shift 2
            ;;
        
         --out_tag)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --out_tag requires a value" >&2
                exit 1
            }
            OUT_TAG="$2"
            shift 2
            ;;

        --wsc_size)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --wsc_size requires a value" >&2
                exit 1
            }

            WSC_SIZE="$2"

            [[ "$WSC_SIZE" =~ ^[0-9]+$ ]] || {
                echo "ERROR: --wsc_size must be an integer >= 1" >&2
                exit 1
            }

            (( WSC_SIZE >= 1 )) || {
                echo "ERROR: --wsc_size must be >= 1" >&2
                exit 1
            }

            shift 2
            ;;

        --n_threads)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --n_threads requires a value" >&2
                exit 1
            }

            N_THREADS="$2"

            [[ "$N_THREADS" =~ ^[0-9]+$ ]] || {
                echo "ERROR: --n_threads must be an integer >= 1" >&2
                exit 1
            }

            (( N_THREADS >= 1 )) || {
                echo "ERROR: --n_threads must be >= 1" >&2
                exit 1
            }

            shift 2
            ;;


        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

[[ -n "$NCHUNKS" ]] || {
    echo "ERROR: missing --nchunks" >&2
    exit 1
}

[[ -n "$COMPRESSION" ]] || {
    echo "ERROR: missing --compression" >&2
    exit 1
}

[[ -n "$MSV4_REPO" ]] || {
    echo "ERROR: missing --msv4_repo" >&2
    exit 1
}

[[ -n "$DATA_PROC" ]] || {
    echo "ERROR: missing --data_proc" >&2
    exit 1
}

[[ -n "$ZARR_NAME" ]] || {
    echo "ERROR: missing --zarr_name" >&2
    exit 1
}

[[ -d "$MSV4_REPO" ]] || {
    echo "ERROR: --msv4_repo is not an existing directory: $MSV4_REPO" >&2
    exit 1
}

[[ -d "$DATA_PROC" ]] || {
    echo "ERROR: --data_proc is not an existing directory: $DATA_PROC" >&2
    exit 1
}

[[ -n "$PROC_CHUNKING" ]] || {
    echo "ERROR: missing --proc_chunking" >&2
    exit 1
}

[[ -n "$XDS_NAME" ]] || {
    echo "ERROR: missing --xds_name" >&2
    exit 1
}

[[ -n "$OUT_DIR" ]] || {
    echo "ERROR: missing --out_dir" >&2
    exit 1
}

mkdir -p "$OUT_DIR"

[[ -d "$OUT_DIR" ]] || {
    echo "ERROR: failed to create output directory: $OUT_DIR" >&2
    exit 1
}

[[ -n "$OUT_TAG" ]] || {
    echo "ERROR: missing --out_tag" >&2
    exit 1
}


SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "-I- SCRIPT_DIR = $SCRIPT_DIR"

if (( USE_ZARR3 )); then
    echo "Activating Zarr 3 environment"
    source "$SCRIPT_DIR/set_dask-cuda_26.04_zarr_3_environment.sh"
else
    echo "Using default Zarr 2 environment"
    source "$SCRIPT_DIR/set_dask-cuda_26.04_environment.sh"
fi

FOV_DEG=20
#WSC_SIZE=4096

MSV4_PROC=${DATA_PROC}/${ZARR_NAME}

if [ ! -d "$MSV4_PROC" ]; then
    echo "-I- $MSV4_PROC not found. Copying from $MSV4_REPO"
    time cp -a "${MSV4_REPO}" "${DATA_PROC}/"
fi

echo "-I- Content of DATA_PROC $DATA_PROC:"
ls -l "$DATA_PROC"

echo "-I- MSV4_PROC: $MSV4_PROC"

du -sh "$MSV4_PROC"

PREFIX="${SLURM_JOB_ID:+slurm-${SLURM_JOB_ID}_}"
# Build command
CMD="python ${SCRIPT_DIR}/aoflagger_idg_on_msv4.py"
CMD+=" --scheduler-file $SCHEDULER_FILE"
CMD+=" --msv4 $MSV4_PROC"
CMD+=" --xds_name ${XDS_NAME}_0"
CMD+=" --proc_chunking ${PROC_CHUNKING}"
CMD+=" --vis_type VISIBILITY"
CMD+=" --lua_strategy ${HOME}/RADIOBLOCKS/aoflagger_on_msv4/my_lua/my-generic-default.lua"
CMD+=" --fov_deg $FOV_DEG"
CMD+=" --grid_size $WSC_SIZE"
CMD+=" --subgrid_size 32"
CMD+=" --kernel_size 16"
CMD+=" --mode gpu"
CMD+=" --timeslot_size 1"
CMD+=" --out_dir ${OUT_DIR}"
CMD+=" --out_tag ${PREFIX}${OUT_TAG}"
[[ -v NB ]] && CMD+=" --nb $NB"
[[ -v N_THREADS ]] && CMD+=" --n_threads $N_THREADS"
#CMD+=" --pin_cpu_threads"

# Run command
$CMD

echo "__skipping_flag_stats__"
exit

# Compare FLAG and FLAG2 to check against AOFlagger command line on MSv2
# See: ~/AOFLAGGER_MSV2/aoflagger_on_msv2.sh
# ----------------------------------------------------------------------
find $MSV4_PROC/${XDS_NAME}_0/time -type f -exec sha256sum {} \;
echo

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    outname=${SLURM_JOB_ID}
else
    outname="cmdline"
fi

python ${SCRIPT_DIR}/compare_flag_flag2_msv4.py \
       $MSV4_PROC \
       --group ${XDS_NAME}_0 \
       --out ${DATA_PROC}/${outname}_${XDS_NAME}_aoflagger.csv

find $MSV4_PROC/${XDS_NAME}_0/time -type f -exec sha256sum {} \;
echo
