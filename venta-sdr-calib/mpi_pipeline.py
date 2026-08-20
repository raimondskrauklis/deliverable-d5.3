#!/usr/bin/env python3
"""Process and calibrate one complete observation with MPI."""

import sys
import time
import traceback

from launcher_setup import limit_math_libraries_to_one_thread


# Set numerical-library thread limits before importing NumPy, SciPy, or mpi4py.
limit_math_libraries_to_one_thread()

from pipeline_common import (  # noqa: E402
    pipeline_parser,
    processing_configuration,
    spectra_by_state,
    validate_complete_observation,
    validate_paths_and_options,
)


def main():
    args = pipeline_parser("mpi").parse_args()

    from mpi4py import MPI

    from mpi_runtime import _coordinator_loop, _worker_loop
    from workloads import discover_real_groups, prewarm_worker

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    worker_count = comm.Get_size() - 1
    start_time = time.localtime()

    if worker_count < 1:
        if rank == 0:
            print("MPI pipeline requires a coordinator and at least one worker", file=sys.stderr)
        return 2
    if args.expected_workers and args.expected_workers != worker_count:
        if rank == 0:
            print(
                "expected %d workers, but MPI started %d"
                % (args.expected_workers, worker_count),
                file=sys.stderr,
            )
        return 2

    logs = None
    frequency = None
    groups = None
    setup = None
    if rank == 0:
        try:
            validate_paths_and_options(args)
            from pipeline_output import load_observation_log, observation_parameters

            logs = load_observation_log(args.experiment_log, args.output_dir)
            fft_size, frequency = observation_parameters(logs)
            groups = discover_real_groups(args.data_dir, args.input_format)
            expected_scans = int(logs["header"]["N_scans,N_cal"][0])
            validate_complete_observation(groups, expected_scans)
            setup = {
                "processing": processing_configuration(args, fft_size),
                "error": "",
            }
        except Exception:
            setup = {"processing": None, "error": traceback.format_exc()}

    setup = comm.bcast(setup, root=0)
    if setup["error"]:
        if rank == 0:
            print("[ERROR] setup failed:\n%s" % setup["error"], file=sys.stderr)
        return 2
    processing = setup["processing"]

    prewarm_error = ""
    if rank != 0:
        try:
            prewarm_worker(processing)
        except Exception:
            prewarm_error = traceback.format_exc()
    prewarm_errors = comm.gather(prewarm_error, root=0)
    first_error = next((value for value in prewarm_errors if value), "") if rank == 0 else None
    first_error = comm.bcast(first_error, root=0)
    if first_error:
        if rank == 0:
            print("[ERROR] worker setup failed:\n%s" % first_error, file=sys.stderr)
        return 1

    use_lease_gate = 0 < args.io_leases < worker_count
    lease_limit = args.io_leases if use_lease_gate else worker_count
    processing_started = time.perf_counter()
    if rank == 0:
        results = _coordinator_loop(
            comm,
            groups,
            worker_count,
            lease_limit,
            processing["precision"],
        )
    else:
        _worker_loop(comm, "real", processing, None, use_lease_gate)
        results = None
    comm.barrier()

    if rank != 0:
        return 0

    processing_seconds = time.perf_counter() - processing_started
    try:
        from pipeline_output import calibrate_and_write_outputs

        spectra = spectra_by_state(results)
        calibrate_and_write_outputs(
            logs=logs,
            output_dir=args.output_dir,
            frequency=frequency,
            spectra=spectra,
            files_total=len(groups) * 8,
            start_time=start_time,
            framework="MPI",
            configuration={
                **processing,
                "workers": worker_count,
                "io_leases": args.io_leases,
            },
        )
    except Exception:
        print("[ERROR] calibration or output failed:\n%s" % traceback.format_exc(), file=sys.stderr)
        return 1

    print(
        "Completed %d scan groups with %d MPI workers in %.3f s (worker stage)"
        % (len(groups), worker_count, processing_seconds),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
