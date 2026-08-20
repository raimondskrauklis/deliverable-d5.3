#!/usr/bin/env python3
"""Dask scheduling runtime shared by the real and synthetic launchers."""

import logging
import os
import time
import uuid

import dask
import distributed
from dask.distributed import Client, LocalCluster, Semaphore

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


_WORKER_SEMAPHORES = {}


def _parse_cpu_list(value):
    """Expand Linux CPU-list syntax such as ``0-3,8,10-11``."""
    cpus = set()
    for item in value.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(item))
    return cpus


def _physical_core_masks(allowed_cpus):
    """Return sibling masks for every physical core in a Linux cpuset."""
    allowed = set(allowed_cpus)
    masks = set()
    for cpu in sorted(allowed):
        topology = "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu
        try:
            with open(topology, "r", encoding="ascii") as stream:
                siblings = _parse_cpu_list(stream.read())
        except OSError as exc:
            raise RuntimeError(
                "cannot discover physical-core topology from %s: %s"
                % (topology, exc)
            ) from exc
        mask = tuple(sorted(allowed.intersection(siblings)))
        if not mask:
            raise RuntimeError("CPU %d has no sibling in the allowed cpuset" % cpu)
        masks.add(mask)
    return sorted(masks, key=lambda mask: (min(mask), mask))


def _set_worker_affinity(assignments, dask_worker=None):
    """Pin one worker process to its precomputed physical-core mask."""
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("distinct-core affinity requires Linux sched_setaffinity")
    address = str(dask_worker.address)
    if address not in assignments:
        raise RuntimeError("no CPU assignment for Dask worker %s" % address)
    requested = set(int(cpu) for cpu in assignments[address])
    os.sched_setaffinity(0, requested)
    actual = sorted(os.sched_getaffinity(0))
    if set(actual) != requested:
        raise RuntimeError(
            "worker %s affinity mismatch: requested=%s actual=%s"
            % (address, sorted(requested), actual)
        )
    return {"pid": os.getpid(), "cpu_affinity": actual}


def _pin_workers_to_distinct_cores(client, worker_count):
    """Reserve one core for control work and pin each worker to another core."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("distinct-core affinity requires Linux CPU-affinity APIs")

    allowed = sorted(os.sched_getaffinity(0))
    core_masks = _physical_core_masks(allowed)
    required = int(worker_count) + 1
    if len(core_masks) != required:
        raise RuntimeError(
            "distinct-core mode expected exactly %d physical cores in the Slurm "
            "cpuset, found %d (%s)"
            % (required, len(core_masks), core_masks)
        )

    # MPI rank zero occupies the first mapped core. Reserve the corresponding
    # core for the in-process Dask client/scheduler and use the rest for workers.
    controller_mask = core_masks[0]
    worker_masks = core_masks[1:]
    addresses = sorted(client.scheduler_info().get("workers", {}))
    if len(addresses) != int(worker_count):
        raise RuntimeError(
            "scheduler reports %d workers, expected %d"
            % (len(addresses), int(worker_count))
        )
    assignments = {
        address: list(mask) for address, mask in zip(addresses, worker_masks)
    }
    responses = client.run(_set_worker_affinity, assignments)
    os.sched_setaffinity(0, set(controller_mask))

    return {
        "worker_affinity_mode": "distinct-cores",
        "controller_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "worker_affinity_assignments": assignments,
        "worker_affinity_responses": responses,
    }


def _logical_cpus_physical_first(allowed_cpus):
    """Order logical CPUs across cores before using sibling hardware threads."""
    core_masks = _physical_core_masks(allowed_cpus)
    ordered = []
    maximum_siblings = max((len(mask) for mask in core_masks), default=0)
    for sibling_index in range(maximum_siblings):
        for mask in core_masks:
            if sibling_index < len(mask):
                ordered.append(mask[sibling_index])
    return ordered


def _pin_workers_to_distinct_cpus(client, worker_count):
    """Reserve one logical CPU for control and pin each worker to another."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("distinct-cpus affinity requires Linux CPU-affinity APIs")

    allowed = sorted(os.sched_getaffinity(0))
    ordered_cpus = _logical_cpus_physical_first(allowed)
    required = int(worker_count) + 1
    if len(ordered_cpus) < required:
        raise RuntimeError(
            "distinct-cpus mode requires at least %d logical CPUs in the Slurm "
            "cpuset, found %d (%s)"
            % (required, len(ordered_cpus), ordered_cpus)
        )
    # Some exclusive Slurm configurations expose the full-node cpuset even to
    # an exact W+1-CPU step. Restrict explicitly to the requested CPU count so
    # Dask uses the same number of hardware threads as the MPI ranks.
    ordered_cpus = ordered_cpus[:required]

    controller_cpu = ordered_cpus[0]
    worker_cpus = ordered_cpus[1:]
    addresses = sorted(client.scheduler_info().get("workers", {}))
    if len(addresses) != int(worker_count):
        raise RuntimeError(
            "scheduler reports %d workers, expected %d"
            % (len(addresses), int(worker_count))
        )
    assignments = {
        address: [cpu] for address, cpu in zip(addresses, worker_cpus)
    }
    responses = client.run(_set_worker_affinity, assignments)
    os.sched_setaffinity(0, {controller_cpu})

    return {
        "worker_affinity_mode": "distinct-cpus",
        "controller_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "worker_affinity_assignments": assignments,
        "worker_affinity_responses": responses,
    }


def _dask_worker_settings(dask_worker=None):
    """Return settings that must be verified on the worker, not the client."""
    return {
        "worker_profiling_enabled": bool(
            dask.config.get("distributed.worker.profile.enabled")
        ),
        "local_directory": str(getattr(dask_worker, "local_directory", "")),
    }


def _worker_semaphore(name, max_leases):
    if not name:
        return None
    key = (name, int(max_leases))
    semaphore = _WORKER_SEMAPHORES.get(key)
    if semaphore is None:
        semaphore = Semaphore(name=name, max_leases=int(max_leases))
        _WORKER_SEMAPHORES[key] = semaphore
    return semaphore


def _dask_real_task(group, task_config):
    gate = _worker_semaphore(
        task_config["semaphore_name"],
        task_config["max_leases"],
    )
    return process_real_group(group, task_config["processing"], io_gate=gate)


def _dask_synthetic_task(group, task_config):
    return process_synthetic_group(
        group,
        task_config["processing"],
        task_config["synthetic"],
    )


def run_dask(workload, args):
    validate_arguments(args, workload)
    main_started = time.perf_counter()

    input_started = time.perf_counter()
    if workload == "real":
        groups = discover_real_groups(
            args.data_dir,
            args.input_format,
            group_offset=args.group_offset,
            group_count=args.n_groups,
        )
    else:
        groups = create_virtual_groups(args.groups)
    input_setup_s = time.perf_counter() - input_started

    processing = processing_options(args)
    synthetic = synthetic_options(args) if workload == "synthetic" else None

    local_directory = args.local_directory or None
    # This launcher owns its process, so a process-wide setting is clearer than
    # a deeply indented context around the cluster lifecycle.
    dask.config.set({"distributed.worker.profile.enabled": False})
    cluster = None
    client = None
    controller_semaphore = None
    try:
        setup_started = time.perf_counter()
        cluster = LocalCluster(
            n_workers=args.workers,
            threads_per_worker=1,
            processes=True,
            memory_limit=args.memory_limit,
            local_directory=local_directory,
            dashboard_address=None,
            silence_logs=logging.ERROR,
        )
        client = Client(cluster, direct_to_workers=True)
        client.wait_for_workers(args.workers)
        framework_setup_s = time.perf_counter() - setup_started

        affinity_started = time.perf_counter()
        affinity_details = {
            "worker_affinity_mode": args.worker_affinity,
            "controller_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "worker_affinity_assignments": {},
        }
        if args.worker_affinity == "distinct-cores":
            affinity_details = _pin_workers_to_distinct_cores(
                client,
                args.workers,
            )
        elif args.worker_affinity == "distinct-cpus":
            affinity_details = _pin_workers_to_distinct_cpus(
                client,
                args.workers,
            )
        affinity_setup_s = time.perf_counter() - affinity_started

        prewarm_started = time.perf_counter()
        prewarm_info = client.run(
            prewarm_worker,
            processing,
            synthetic,
        )
        worker_settings = client.run(_dask_worker_settings)
        for address, settings in worker_settings.items():
            prewarm_info[address].update(settings)
        if any(
            settings["worker_profiling_enabled"]
            for settings in worker_settings.values()
        ):
            raise RuntimeError("Dask worker profiling is still enabled")
        prewarm_s = time.perf_counter() - prewarm_started

        use_lease_gate = (
            workload == "real" and args.io_leases > 0 and args.io_leases < args.workers
        )
        semaphore_name = "calibration-io-%s" % uuid.uuid4().hex if use_lease_gate else ""
        lease_limit = args.io_leases if use_lease_gate else args.workers
        if use_lease_gate:
            controller_semaphore = Semaphore(
                name=semaphore_name,
                max_leases=lease_limit,
            )
            controller_semaphore.acquire()
            controller_semaphore.release()

        task_config = {
            "processing": processing,
            "synthetic": synthetic,
            "semaphore_name": semaphore_name,
            "max_leases": lease_limit,
        }
        task_config_future = client.scatter([task_config], broadcast=True)[0]

        processing_started = time.perf_counter()
        if workload == "real":
            futures = client.map(
                _dask_real_task,
                groups,
                task_config=task_config_future,
                pure=False,
            )
        else:
            futures = client.map(
                _dask_synthetic_task,
                groups,
                task_config=task_config_future,
                pure=False,
            )
        results = client.gather(futures)
        processing_s = time.perf_counter() - processing_started

        run_id = create_run_id("dask", workload, args.run_id)
        phases = {
            "input_setup_s": input_setup_s,
            "framework_setup_s": framework_setup_s,
            "affinity_setup_s": affinity_setup_s,
            "prewarm_s": prewarm_s,
            "processing_s": processing_s,
            "benchmark_to_results_s": time.perf_counter() - main_started,
        }
        scheduler_info = client.scheduler_info()
        extra = {
            "dask_version": dask.__version__,
            "distributed_version": distributed.__version__,
            "scheduler_address": scheduler_info.get("address", ""),
            "scheduler_worker_count": len(scheduler_info.get("workers", {})),
            "prewarm_workers": prewarm_info,
            "worker_local_directories": sorted(
                set(
                    item.get("local_directory", "")
                    for item in prewarm_info.values()
                    if item.get("local_directory", "")
                )
            ),
            "memory_limit": args.memory_limit,
            "requested_local_directory": local_directory or "",
            "worker_profiling_enabled": False,
            "io_gate_active": use_lease_gate,
            "launcher_attempt": int(os.environ.get("DASK_LAUNCH_ATTEMPT", "1")),
            "startup_max_attempts": int(
                os.environ.get("DASK_STARTUP_MAX_ATTEMPTS", "1")
            ),
        }
        extra.update(affinity_details)
        record = build_run_record(
            "dask",
            workload,
            args,
            args.workers,
            groups,
            results,
            phases,
            run_id,
            framework_details=extra,
        )
        paths = write_run_files(record, results, args.output_dir)
        print_run_summary(record, paths)
        return 0
    finally:
        if controller_semaphore is not None:
            try:
                controller_semaphore.close()
            except Exception:
                pass
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
