#!/usr/bin/env python3
"""MPI scheduling runtime shared by the real and synthetic launchers."""

import sys
import time
import traceback
from collections import deque

import mpi4py
import numpy as np
from mpi4py import MPI

from benchmark_cli import processing_options, synthetic_options, validate_arguments
from reporting import (
    build_run_record,
    create_run_id,
    print_run_summary,
    write_run_files,
)
from workloads import (
    create_virtual_groups,
    discover_real_groups,
    prewarm_worker,
    process_real_group,
    process_synthetic_group,
)


TAG_WORK = 101
TAG_STOP = 102
TAG_RESULT_META = 103
TAG_ERROR = 104
TAG_LEASE_REQUEST = 105
TAG_LEASE_GRANT = 106
TAG_LEASE_RELEASE = 107
TAG_RESULT_DATA = 108


class MPIReadGate(object):
    def __init__(self, comm):
        self.comm = comm

    def __enter__(self):
        self.comm.send(None, dest=0, tag=TAG_LEASE_REQUEST)
        self.comm.recv(source=0, tag=TAG_LEASE_GRANT)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.comm.send(None, dest=0, tag=TAG_LEASE_RELEASE)
        return False


def _process_group(workload, group, processing, synthetic, io_gate):
    if workload == "real":
        return process_real_group(group, processing, io_gate=io_gate)
    return process_synthetic_group(group, processing, synthetic)


def _worker_loop(comm, workload, processing, synthetic, use_lease_gate):
    gate = MPIReadGate(comm) if use_lease_gate else None
    result_dtype = np.float64 if processing["precision"] == "f64" else np.float32
    mpi_result_dtype = MPI.DOUBLE if processing["precision"] == "f64" else MPI.FLOAT
    while True:
        status = MPI.Status()
        group = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            return
        if status.Get_tag() != TAG_WORK:
            raise RuntimeError("worker received unexpected MPI tag %d" % status.Get_tag())
        try:
            result = _process_group(workload, group, processing, synthetic, gate)
            spectra = np.ascontiguousarray(result.pop("spectra"), dtype=result_dtype)
            result["spectra_shape"] = tuple(int(value) for value in spectra.shape)
            result["spectra_dtype"] = spectra.dtype.str
            comm.send(result, dest=0, tag=TAG_RESULT_META)
            comm.Send([spectra, mpi_result_dtype], dest=0, tag=TAG_RESULT_DATA)
        except Exception:
            comm.send(traceback.format_exc(), dest=0, tag=TAG_ERROR)
            return


def _coordinator_loop(comm, groups, workers, lease_limit, precision):
    results = []
    next_group = 0
    active_workers = 0
    leases_available = lease_limit
    lease_waiters = deque()
    pending_results = {}
    result_dtype = np.float64 if precision == "f64" else np.float32
    mpi_result_dtype = MPI.DOUBLE if precision == "f64" else MPI.FLOAT
    expected_dtype = np.dtype(result_dtype).str

    def dispatch_or_stop(worker):
        nonlocal next_group, active_workers
        if next_group < len(groups):
            comm.send(groups[next_group], dest=worker, tag=TAG_WORK)
            next_group += 1
        else:
            comm.send(None, dest=worker, tag=TAG_STOP)
            active_workers -= 1

    def finish_result(source, wait):
        pending = pending_results.get(source)
        if pending is None:
            return False
        request, message, spectra = pending
        if wait:
            request.Wait()
        elif not request.Test():
            return False
        del pending_results[source]
        message["spectra"] = spectra
        results.append(message)
        return True

    def handle_control_message(message, source, tag):
        nonlocal leases_available

        if tag == TAG_LEASE_REQUEST:
            if leases_available > 0:
                leases_available -= 1
                comm.send(None, dest=source, tag=TAG_LEASE_GRANT)
            else:
                lease_waiters.append(source)
            return

        if tag == TAG_LEASE_RELEASE:
            leases_available += 1
            if lease_waiters:
                waiter = lease_waiters.popleft()
                leases_available -= 1
                comm.send(None, dest=waiter, tag=TAG_LEASE_GRANT)
            return

        if tag == TAG_ERROR:
            print("[ERROR] worker rank %d failed:\n%s" % (source, message), file=sys.stderr)
            comm.Abort(1)
            raise RuntimeError("MPI.Abort returned unexpectedly")

        if tag != TAG_RESULT_META:
            comm.Abort(1)
            raise RuntimeError("coordinator received unexpected MPI tag %d" % tag)

        # A worker cannot send its next metadata message until its previous
        # blocking payload send has completed. Finalize that payload if the
        # completion was not observed by the event loop yet.
        if source in pending_results:
            finish_result(source, wait=True)

        shape = tuple(int(value) for value in message.pop("spectra_shape"))
        received_dtype = message.pop("spectra_dtype")
        if received_dtype != expected_dtype:
            comm.Abort(1)
            raise RuntimeError(
                "worker returned dtype %s, expected %s"
                % (received_dtype, expected_dtype)
            )
        spectra = np.empty(shape, dtype=result_dtype)
        request = comm.Irecv(
            [spectra, mpi_result_dtype],
            source=source,
            tag=TAG_RESULT_DATA,
        )
        pending_results[source] = (request, message, spectra)

        # Queue the worker's next assignment as soon as its metadata arrives.
        # The nonblocking receive lets all workers' payload transfers progress
        # instead of serializing them behind one blocking coordinator receive.
        dispatch_or_stop(source)

    for worker in range(1, workers + 1):
        if next_group < len(groups):
            comm.send(groups[next_group], dest=worker, tag=TAG_WORK)
            next_group += 1
            active_workers += 1
        else:
            comm.send(None, dest=worker, tag=TAG_STOP)

    while active_workers or pending_results:
        completed_sources = [
            source
            for source, (request, _message, _spectra) in pending_results.items()
            if request.Test()
        ]
        for source in completed_sources:
            finish_result(source, wait=True)

        drained_message = False
        while True:
            status = MPI.Status()
            if not comm.iprobe(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status,
            ):
                break
            source = status.Get_source()
            tag = status.Get_tag()
            message = comm.recv(source=source, tag=tag)
            handle_control_message(message, source, tag)
            drained_message = True

        if drained_message or completed_sources:
            continue

        if pending_results:
            sources = list(pending_results)
            requests = [pending_results[source][0] for source in sources]
            completed_index = MPI.Request.Waitany(requests)
            if completed_index != MPI.UNDEFINED:
                finish_result(sources[completed_index], wait=True)
            continue

        if active_workers:
            status = MPI.Status()
            message = comm.recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status,
            )
            handle_control_message(
                message,
                status.Get_source(),
                status.Get_tag(),
            )

    if lease_waiters:
        comm.Abort(1)
        raise RuntimeError("lease queue is non-empty after all workers stopped")
    return results


def run_mpi(workload, args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    workers = size - 1
    main_started = time.perf_counter()

    if workers < 1:
        if rank == 0:
            print("MPI benchmark requires at least two ranks", file=sys.stderr)
        return 2

    try:
        validate_arguments(args, workload)
    except Exception as error:
        if rank == 0:
            print("argument error: %s" % error, file=sys.stderr)
        return 2

    if args.expected_workers and args.expected_workers != workers:
        if rank == 0:
            print(
                "expected %d workers but MPI size provides %d"
                % (args.expected_workers, workers),
                file=sys.stderr,
            )
        return 2

    groups = None
    setup_error = None
    setup_started = time.perf_counter()
    if rank == 0:
        try:
            if workload == "real":
                groups = discover_real_groups(
                    args.data_dir,
                    args.input_format,
                    group_offset=args.group_offset,
                    group_count=args.n_groups,
                )
            else:
                groups = create_virtual_groups(args.groups)
        except Exception:
            setup_error = traceback.format_exc()

    setup_error = comm.bcast(setup_error, root=0)
    if setup_error:
        if rank == 0:
            print("[ERROR] input setup failed:\n%s" % setup_error, file=sys.stderr)
        return 2
    input_setup_s = time.perf_counter() - setup_started

    processing = processing_options(args)
    synthetic = synthetic_options(args) if workload == "synthetic" else None

    prewarm_started = time.perf_counter()
    worker_synthetic_config = synthetic if rank != 0 else None
    prewarm_info = prewarm_worker(
        processing,
        worker_synthetic_config,
    )
    prewarm_info["mpi_rank"] = rank
    prewarm_info["role"] = "coordinator" if rank == 0 else "worker"
    all_prewarm_info = comm.gather(prewarm_info, root=0)
    comm.barrier()
    prewarm_s = time.perf_counter() - prewarm_started

    use_lease_gate = (
        workload == "real" and args.io_leases > 0 and args.io_leases < workers
    )
    lease_limit = args.io_leases if use_lease_gate else workers

    processing_started = time.perf_counter()
    if rank == 0:
        results = _coordinator_loop(
            comm,
            groups,
            workers,
            lease_limit,
            processing["precision"],
        )
    else:
        _worker_loop(
            comm,
            workload,
            processing,
            synthetic,
            use_lease_gate,
        )
        results = None
    if rank == 0:
        processing_s = time.perf_counter() - processing_started
    teardown_started = time.perf_counter()
    comm.barrier()
    teardown_s = time.perf_counter() - teardown_started

    if rank != 0:
        return 0

    run_id = create_run_id("mpi", workload, args.run_id)
    phases = {
        "input_setup_s": input_setup_s,
        "prewarm_s": prewarm_s,
        "processing_s": processing_s,
        "teardown_s": teardown_s,
        "benchmark_to_results_s": time.perf_counter() - main_started,
    }
    extra = {
        "mpi4py_version": mpi4py.__version__,
        "mpi_standard_version": list(MPI.Get_version()),
        "mpi_library_version": MPI.Get_library_version().strip(),
        "prewarm_workers": all_prewarm_info,
        "io_gate_active": use_lease_gate,
    }
    record = build_run_record(
        "mpi",
        workload,
        args,
        workers,
        groups,
        results,
        phases,
        run_id,
        framework_details=extra,
    )
    paths = write_run_files(record, results, args.output_dir)
    print_run_summary(record, paths)
    return 0
