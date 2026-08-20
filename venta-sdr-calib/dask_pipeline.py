#!/usr/bin/env python3
"""Process and calibrate one complete observation with local Dask workers."""

import logging
import sys
import time
import traceback
import uuid

from launcher_setup import limit_math_libraries_to_one_thread


# Set numerical-library thread limits before importing NumPy, SciPy, or Dask.
limit_math_libraries_to_one_thread()

from pipeline_common import (  # noqa: E402
    pipeline_parser,
    processing_configuration,
    spectra_by_state,
    validate_complete_observation,
    validate_paths_and_options,
)


def main():
    args = pipeline_parser("dask").parse_args()
    start_time = time.localtime()

    try:
        validate_paths_and_options(args)
        from pipeline_output import (
            calibrate_and_write_outputs,
            load_observation_log,
            observation_parameters,
        )
        from workloads import discover_real_groups, prewarm_worker

        logs = load_observation_log(args.experiment_log, args.output_dir)
        fft_size, frequency = observation_parameters(logs)
        groups = discover_real_groups(args.data_dir, args.input_format)
        expected_scans = int(logs["header"]["N_scans,N_cal"][0])
        validate_complete_observation(groups, expected_scans)
        processing = processing_configuration(args, fft_size)
    except Exception:
        print("[ERROR] setup failed:\n%s" % traceback.format_exc(), file=sys.stderr)
        return 2

    import dask
    from dask.distributed import Client, LocalCluster, Semaphore

    from dask_runtime import (
        _dask_real_task,
        _dask_worker_settings,
        _pin_workers_to_distinct_cores,
        _pin_workers_to_distinct_cpus,
    )

    dask.config.set({"distributed.worker.profile.enabled": False})
    cluster = None
    client = None
    controller_semaphore = None
    processing_seconds = 0.0
    results = None
    try:
        cluster = LocalCluster(
            n_workers=args.workers,
            threads_per_worker=1,
            processes=True,
            memory_limit=args.memory_limit,
            local_directory=args.local_directory or None,
            dashboard_address=None,
            silence_logs=logging.ERROR,
        )
        client = Client(cluster, direct_to_workers=True)
        client.wait_for_workers(args.workers)

        if args.worker_affinity == "distinct-cores":
            _pin_workers_to_distinct_cores(client, args.workers)
        elif args.worker_affinity == "distinct-cpus":
            _pin_workers_to_distinct_cpus(client, args.workers)

        client.run(prewarm_worker, processing, None)
        settings = client.run(_dask_worker_settings)
        if any(value["worker_profiling_enabled"] for value in settings.values()):
            raise RuntimeError("Dask worker profiling is still enabled")

        use_lease_gate = 0 < args.io_leases < args.workers
        lease_limit = args.io_leases if use_lease_gate else args.workers
        semaphore_name = "calibration-io-%s" % uuid.uuid4().hex if use_lease_gate else ""
        if use_lease_gate:
            controller_semaphore = Semaphore(name=semaphore_name, max_leases=lease_limit)
            # Materialize the named semaphore before workers receive tasks.
            controller_semaphore.acquire()
            controller_semaphore.release()

        task_config = {
            "processing": processing,
            "synthetic": None,
            "semaphore_name": semaphore_name,
            "max_leases": lease_limit,
        }
        task_config_future = client.scatter([task_config], broadcast=True)[0]
        processing_started = time.perf_counter()
        futures = client.map(
            _dask_real_task,
            groups,
            task_config=task_config_future,
            pure=False,
        )
        results = client.gather(futures)
        processing_seconds = time.perf_counter() - processing_started
    except Exception:
        print("[ERROR] Dask processing failed:\n%s" % traceback.format_exc(), file=sys.stderr)
        return 1
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

    try:
        spectra = spectra_by_state(results)
        calibrate_and_write_outputs(
            logs=logs,
            output_dir=args.output_dir,
            frequency=frequency,
            spectra=spectra,
            files_total=len(groups) * 8,
            start_time=start_time,
            framework="Dask",
            configuration={
                **processing,
                "workers": args.workers,
                "io_leases": args.io_leases,
                "worker_affinity": args.worker_affinity,
            },
        )
    except Exception:
        print("[ERROR] calibration or output failed:\n%s" % traceback.format_exc(), file=sys.stderr)
        return 1

    print(
        "Completed %d scan groups with %d Dask workers in %.3f s (worker stage)"
        % (len(groups), args.workers, processing_seconds),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
