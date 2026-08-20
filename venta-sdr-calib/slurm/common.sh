#!/bin/bash
# Shared functions for the framework benchmark sbatch scripts.
# Source this file; do not execute it directly.

SLURM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${BENCH_DIR:-$(cd "${SLURM_SCRIPT_DIR}/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${DATADIR:-/var/scratch/${USER}/rlmi_f1666_ir_24}}"
ZARR_DIR="${ZARR_DIR:-/var/scratch/${USER}/rlmi_f1666_ir_24_zarr}"
LOG_FILE="${LOG_FILE:-${LOGFILE:-/var/scratch/${USER}/rlmi_f1666_ir_24.log}}"
SUBMISSION_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-${SUBMISSION_DIR}/benchmark_results}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MATRIX_ID:-job_${SLURM_JOB_ID:-manual}}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${HOME}/paper_runs/benchmark-venv/bin/activate}"

NS="${NS:-4096}"
FFT_BACKEND="${FFT_BACKEND:-scipy}"
CACHE_REGIME="${CACHE_REGIME:-uncontrolled/mixed}"
CPU_REGIME="${CPU_REGIME:-physical-core}"


setup_benchmark_environment() {
    module purge

    # Current cluster module stack. The Spack module exposes OpenMPI in Lmod.
    # Override the space-separated list with BENCH_MODULES if the cluster
    # environment changes, but use the same modules for MPI and Dask.
    local modules_string
    modules_string="${BENCH_MODULES:-spack/2026.05.13 python/3.13.8 openmpi/5.0.10}"
    local module_name
    for module_name in ${modules_string}; do
        module load "${module_name}"
    done

    if [[ ! -f "${VENV_ACTIVATE}" ]]; then
        echo "ERROR: virtual environment activation file not found: ${VENV_ACTIVATE}" >&2
        exit 2
    fi
    source "${VENV_ACTIVATE}"

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export MPLBACKEND=Agg

    # This benchmark is deliberately single-node. Force OpenMPI's OB1/shared-
    # memory path before the mpi4py dependency check so singleton MPI_Init does
    # not initialize UCX/RDMA and require a large locked-memory allowance.
    export OMPI_MCA_pml=ob1
    export OMPI_MCA_btl=self,sm

    local temporary_base
    temporary_base="${SLURM_TMPDIR:-/tmp}"
    DASK_LOCAL_DIRECTORY="${DASK_LOCAL_DIRECTORY:-${temporary_base%/}/calibration-dask-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}}"
    NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${temporary_base%/}/calibration-numba-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}}"
    export DASK_LOCAL_DIRECTORY NUMBA_CACHE_DIR

    mkdir -p "${OUTPUT_DIR}" "${DASK_LOCAL_DIRECTORY}" "${NUMBA_CACHE_DIR}"
    ulimit -n 65535 || true

    python -c "import dask, distributed, mpi4py, numba, numpy, scipy; from mpi4py import MPI; print('dependency import check passed:', MPI.Get_library_version().strip())"

    echo "=== BENCHMARK ENVIRONMENT ==="
    echo "host=$(hostname)"
    echo "job_id=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-none}"
    echo "node_list=${SLURM_JOB_NODELIST}"
    echo "python=$(python --version 2>&1)"
    echo "mpirun=$(command -v mpirun)"
    mpirun --version
    echo "bench_dir=${BENCH_DIR}"
    echo "data_dir=${DATA_DIR}"
    echo "zarr_dir=${ZARR_DIR}"
    echo "experiment_log=${LOG_FILE} (provenance only)"
    echo "output_dir=${OUTPUT_DIR}"
    echo "dask_local_directory=${DASK_LOCAL_DIRECTORY}"
    echo "numba_cache_dir=${NUMBA_CACHE_DIR}"
    echo "cpu_regime=${CPU_REGIME}"
    echo "ompi_mca_pml=${OMPI_MCA_pml}"
    echo "ompi_mca_btl=${OMPI_MCA_btl}"
    echo "cpu_allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
    echo "ulimit_n=$(ulimit -n)"
    module list 2>&1
    if command -v spack >/dev/null 2>&1; then
        spack find --loaded --long || true
    fi
    lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|CPU\(s\)'
    python -c "import dask, distributed, mpi4py, numba, numpy, scipy; print({'numpy': numpy.__version__, 'scipy': scipy.__version__, 'numba': numba.__version__, 'mpi4py': mpi4py.__version__, 'dask': dask.__version__, 'distributed': distributed.__version__})"
}


run_condition() {
    local framework="$1"
    local workload="$2"
    local workers="$3"
    local numba_mode="$4"
    local precision="$5"
    local run_id="$6"
    local block_id="$7"
    local repetition="$8"
    local order_index="$9"
    local input_format="${10:-raw}"

    local required_cpus=$((workers + 1))
    local allocated_cpus="${SLURM_NTASKS:-0}"
    if ((required_cpus > allocated_cpus)); then
        echo "ERROR: ${workers} workers require ${required_cpus} allocated CPUs, but SLURM_NTASKS=${allocated_cpus}" >&2
        exit 2
    fi

    local script="${BENCH_DIR}/${framework}_${workload}.py"
    if [[ ! -f "${script}" ]]; then
        echo "ERROR: benchmark launcher not found: ${script}" >&2
        exit 2
    fi

    local workload_args=()
    if [[ "${workload}" == "real" ]]; then
        local input_directory
        if [[ "${input_format}" == "raw" ]]; then
            input_directory="${DATA_DIR}"
        elif [[ "${input_format}" == "zarr" ]]; then
            input_directory="${ZARR_DIR}"
            python -c "import zarr, numcodecs; assert zarr.__version__.split('.')[0] == '2'; print({'zarr': zarr.__version__, 'numcodecs': numcodecs.__version__})"
        else
            echo "ERROR: input format must be raw or zarr: ${input_format}" >&2
            return 2
        fi
        if [[ ! -d "${input_directory}" ]]; then
            echo "ERROR: real-data directory not found: ${input_directory}" >&2
            exit 2
        fi
        if [[ ! -f "${LOG_FILE}" ]]; then
            echo "ERROR: observation log not found: ${LOG_FILE}" >&2
            exit 2
        fi
        workload_args=(
            "${input_directory}"
            --input-format "${input_format}"
            --blosc-threads "${BLOSC_THREADS_PER_WORKER:-1}"
            --experiment-log "${LOG_FILE}"
            --n-groups "${REAL_GROUPS:-0}"
            --group-offset "${REAL_GROUP_OFFSET:-0}"
            --io-leases "${IO_LEASES:-0}"
            --cache-regime "${CACHE_REGIME}"
        )
    else
        workload_args=(
            --groups "${SYNTH_GROUPS:-1024}"
            --samples-per-file "${SYNTH_SAMPLES_PER_FILE:-32768}"
            --generation "${SYNTH_GENERATION:-cached}"
            --seed "${SYNTH_SEED:-0}"
            --noise-std "${SYNTH_NOISE_STD:-800.0}"
        )
    fi

    local common_args=(
        --ns "${NS}"
        --fft-backend "${FFT_BACKEND}"
        --precision "${precision}"
        --numba "${numba_mode}"
        --cpu-regime "${CPU_REGIME}"
        --block-id "${block_id}"
        --repetition "${repetition}"
        --order-index "${order_index}"
        --run-id "${run_id}"
        --output-dir "${OUTPUT_DIR}"
    )

    local time_file="${OUTPUT_DIR}/${run_id}.time.txt"
    local wall_file="${OUTPUT_DIR}/${run_id}.launcher_wall.txt"
    local failure_file="${OUTPUT_DIR}/${run_id}.failed.txt"
    local condition_timeout
    if [[ "${workload}" == "real" ]]; then
        condition_timeout="${REAL_CONDITION_TIMEOUT:-2h}"
    else
        condition_timeout="${SYNTH_CONDITION_TIMEOUT:-30m}"
    fi
    export BENCH_CONDITION_TIMEOUT="${condition_timeout}"
    if ! command -v timeout >/dev/null 2>&1; then
        echo "ERROR: GNU timeout is required for bounded benchmark conditions" >&2
        return 2
    fi
    local timeout_command=(
        timeout --signal=TERM --kill-after="${CONDITION_KILL_GRACE:-60s}"
        "${condition_timeout}"
    )
    local timer=()
    if [[ -x /usr/bin/time ]]; then
        timer=(/usr/bin/time -v -o "${time_file}")
    fi

    echo
    echo "=== RUN ${order_index}: ${run_id} ==="
    echo "framework=${framework} workload=${workload} input_format=${input_format} workers=${workers} numba=${numba_mode} precision=${precision}"

    local launcher_start_ns launcher_end_ns launcher_status launcher_attempts
    launcher_attempts=1
    launcher_start_ns="$(date +%s%N)"

    if [[ "${framework}" == "mpi" ]]; then
        local ranks="${required_cpus}"
        local mpi_binding=()
        case "${CPU_REGIME}" in
            physical-core)
                mpi_binding=(--bind-to core --map-by core)
                ;;
            smt)
                # Map across physical cores before using sibling threads. A
                # plain `--map-by hwthread` fills both siblings of early cores
                # first and can leave other physical cores idle at W < 71.
                mpi_binding=(--map-by core:HWTCPUS --bind-to hwthread --rank-by span)
                ;;
            *)
                echo "ERROR: CPU_REGIME must be physical-core or smt" >&2
                return 2
                ;;
        esac
        if "${timer[@]}" "${timeout_command[@]}" mpirun -np "${ranks}" \
                "${mpi_binding[@]}" \
                --mca pml ob1 --mca btl self,sm \
                python -u "${script}" "${workload_args[@]}" "${common_args[@]}" \
                --expected-workers "${workers}"; then
            launcher_status=0
        else
            launcher_status=$?
        fi
    elif [[ "${framework}" == "dask" ]]; then
        # SLURM_CPUS_PER_TASK=1 describes the job's MPI-shaped allocation and
        # is also treated by srun as an input option. Clear it only for this
        # step, whose explicit shape is one process spanning the same total
        # WORKERS+1 CPUs.
        local max_attempts retry_window_s retry_delay_s attempt_started_s
        local attempt_elapsed_s attempt_local_directory
        max_attempts="${DASK_STARTUP_MAX_ATTEMPTS:-2}"
        retry_window_s="${DASK_STARTUP_RETRY_WINDOW_S:-120}"
        retry_delay_s="${DASK_STARTUP_RETRY_DELAY_S:-5}"
        for retry_value in "${max_attempts}" "${retry_window_s}" "${retry_delay_s}"; do
            if ! [[ "${retry_value}" =~ ^[0-9]+$ ]]; then
                echo "ERROR: Dask retry settings must be non-negative integers" >&2
                return 2
            fi
        done
        if ((max_attempts < 1)); then
            echo "ERROR: DASK_STARTUP_MAX_ATTEMPTS must be at least 1" >&2
            return 2
        fi

        launcher_attempts=0
        while ((launcher_attempts < max_attempts)); do
            launcher_attempts=$((launcher_attempts + 1))
            attempt_started_s="${SECONDS}"
            attempt_local_directory="${DASK_LOCAL_DIRECTORY}"
            if ((launcher_attempts > 1)); then
                attempt_local_directory="${DASK_LOCAL_DIRECTORY}/retry-${launcher_attempts}"
                mkdir -p "${attempt_local_directory}"
            fi

            local dask_binding=()
            case "${CPU_REGIME}" in
                physical-core)
                    dask_binding=(--hint=nomultithread --cpu-bind=cores)
                    ;;
                smt)
                    # The one Dask controller must span the entire exact step
                    # cpuset so its spawned worker processes inherit all W+1
                    # logical CPUs. Binding the controller to one thread would
                    # accidentally confine every worker to that same thread.
                    dask_binding=(--cpu-bind=none)
                    ;;
                *)
                    echo "ERROR: CPU_REGIME must be physical-core or smt" >&2
                    return 2
                    ;;
            esac

            if env -u SLURM_CPUS_PER_TASK \
                    DASK_LAUNCH_ATTEMPT="${launcher_attempts}" \
                    DASK_STARTUP_MAX_ATTEMPTS="${max_attempts}" \
                    srun --exact --exclusive --nodes=1 --ntasks=1 \
                    --cpus-per-task="$((workers + 1))" \
                    "${dask_binding[@]}" --kill-on-bad-exit=1 \
                    "${timer[@]}" "${timeout_command[@]}" python -u "${script}" \
                    "${workload_args[@]}" "${common_args[@]}" \
                    --workers "${workers}" \
                    --worker-affinity "${DASK_WORKER_AFFINITY:-shared}" \
                    --local-directory "${attempt_local_directory}"; then
                launcher_status=0
                break
            else
                launcher_status=$?
            fi

            attempt_elapsed_s=$((SECONDS - attempt_started_s))
            if ((launcher_attempts >= max_attempts)) || \
                    ((attempt_elapsed_s > retry_window_s)) || \
                    [[ -e "${OUTPUT_DIR}/${run_id}.json" ]]; then
                break
            fi

            echo "Dask launch attempt ${launcher_attempts} failed after ${attempt_elapsed_s}s before producing a result; retrying once in a fresh process."
            sleep "${retry_delay_s}"
        done
    else
        echo "ERROR: unknown framework: ${framework}" >&2
        exit 2
    fi

    launcher_end_ns="$(date +%s%N)"
    awk -v start="${launcher_start_ns}" -v end="${launcher_end_ns}" \
        -v status="${launcher_status}" -v attempts="${launcher_attempts}" \
        -v condition_timeout="${condition_timeout}" \
        'BEGIN {
            printf "launcher_wall_s=%.9f\n", (end - start) / 1000000000
            printf "exit_status=%d\n", status
            printf "launcher_attempts=%d\n", attempts
            printf "condition_timeout=%s\n", condition_timeout
            print "scope=framework process launch through process exit"
            print "clock=GNU date CLOCK_REALTIME"
        }' > "${wall_file}"

    if ((launcher_status != 0)); then
        {
            echo "run_id=${run_id}"
            echo "framework=${framework}"
            echo "workload=${workload}"
            echo "workers=${workers}"
            echo "exit_status=${launcher_status}"
            echo "launcher_attempts=${launcher_attempts}"
            echo "condition_timeout=${condition_timeout}"
            echo "slurm_job_id=${SLURM_JOB_ID:-}"
            echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-}"
        } > "${failure_file}"
        return "${launcher_status}"
    fi
}


compare_matched_runs() {
    python "${BENCH_DIR}/compare_results.py" "$@"
}
