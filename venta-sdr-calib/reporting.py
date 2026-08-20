"""Human-readable, machine-readable benchmark result reporting."""

import csv
import hashlib
import json
import os
import platform
import re
import socket
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from workloads import numba_version


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_CACHE_DIR",
    "BENCH_CONDITION_TIMEOUT",
)

MPI_ENVIRONMENT = (
    "OMPI_MCA_pml",
    "OMPI_MCA_btl",
)

SLURM_ENVIRONMENT = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_ON_NODE",
    "SLURM_CPUS_PER_TASK",
    "SLURM_NTASKS",
    "SLURM_HINT",
    "SLURM_CPU_BIND",
)


def create_run_id(framework, workload, requested_id):
    if requested_id:
        run_id = requested_id
    else:
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        run_id = "%s_%s_%s_%s" % (
            framework,
            workload,
            timestamp,
            uuid.uuid4().hex[:8],
        )

    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    if not run_id:
        raise ValueError("run ID is empty after sanitization")
    return run_id


def group_set_hash(groups, include_file_sizes):
    digest = hashlib.sha256()
    for group_id, members in groups:
        digest.update(group_id.encode("utf-8"))
        if isinstance(members, dict):
            logical_members = members["members"]
        else:
            logical_members = tuple(
                (Path(member).name, Path(member).stat().st_size if include_file_sizes else 0)
                for member in members
            )
        for name, size in logical_members:
            digest.update(str(name).encode("utf-8"))
            if include_file_sizes:
                digest.update(str(int(size)).encode("ascii"))
    return digest.hexdigest()


def result_value_hash(results):
    """Hash every returned value and dtype in deterministic group order."""
    digest = hashlib.sha256()
    for result in sorted(results, key=lambda item: item["group_id"]):
        spectra = np.ascontiguousarray(result["spectra"])
        digest.update(result["group_id"].encode("utf-8"))
        digest.update(str(spectra.shape).encode("ascii"))
        digest.update(spectra.dtype.str.encode("ascii"))
        digest.update(spectra.tobytes(order="C"))
    return digest.hexdigest()


def summarize_group_timings(results):
    summary = {}
    for field in (
        "generation_s",
        "wait_s",
        "read_s",
        "convert_s",
        "fft_s",
        "group_s",
    ):
        values = [float(result["timings"][field]) for result in results]
        ordered = sorted(values)
        percentile_95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        summary[field] = {
            "sum": float(sum(values)),
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "p95": float(ordered[percentile_95_index]),
            "max": float(max(values)),
        }
    return summary


def system_information():
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba_version(),
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT
        },
        "mpi_environment": {
            name: os.environ.get(name, "") for name in MPI_ENVIRONMENT
        },
        "slurm_environment": {
            name: os.environ.get(name, "") for name in SLURM_ENVIRONMENT
        },
    }


def build_run_record(
    framework,
    workload,
    args,
    worker_count,
    groups,
    results,
    phases,
    run_id,
    framework_details=None,
):
    processing_seconds = float(phases["processing_s"])
    group_count = len(groups)
    file_count = group_count * 8
    input_bytes = int(sum(result["input_bytes"] for result in results))
    storage_bytes = int(sum(result["storage_bytes"] for result in results))
    group_seconds_sum = float(
        sum(result["timings"]["group_s"] for result in results)
    )
    result_dtypes = sorted(
        set(np.asarray(result["spectra"]).dtype.str for result in results)
    )
    workers = Counter(
        "%s:%s" % (result["worker_host"], result["worker_pid"])
        for result in results
    )

    record = {
        "schema_version": 4,
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework": framework,
        "workload": workload,
        "worker_count": int(worker_count),
        "group_count": group_count,
        "file_count": file_count,
        "fft_size": int(args.ns),
        "fft_backend": args.fft_backend,
        "precision": args.precision,
        "numba": args.numba,
        "cache_regime": args.cache_regime,
        "cpu_regime": args.cpu_regime,
        "experiment_order": {
            "block_id": args.block_id,
            "repetition": int(args.repetition),
            "order_index": int(args.order_index),
        },
        "group_set_hash": group_set_hash(groups, workload == "real"),
        "output_hash": result_value_hash(results),
        "phases": {name: float(value) for name, value in phases.items()},
        "throughput_groups_per_hour": group_count / processing_seconds * 3600.0,
        "throughput_groups_per_second": group_count / processing_seconds,
        "throughput_files_per_second": file_count / processing_seconds,
        "effective_input_mib_per_second": (
            input_bytes / (1024.0 * 1024.0) / processing_seconds
        ),
        "effective_logical_input_mib_per_second": (
            input_bytes / (1024.0 * 1024.0) / processing_seconds
        ),
        "effective_storage_mib_per_second": (
            storage_bytes / (1024.0 * 1024.0) / processing_seconds
        ),
        "input_bytes": input_bytes,
        "logical_input_bytes": input_bytes,
        "storage_bytes": storage_bytes,
        "samples": int(sum(result["samples"] for result in results)),
        "result_payload_bytes": int(
            sum(result["spectra"].nbytes for result in results)
        ),
        "result_dtypes": result_dtypes,
        "group_timing": summarize_group_timings(results),
        "task_slot_occupancy": group_seconds_sum
        / (int(worker_count) * processing_seconds),
        "worker_distribution": dict(sorted(workers.items())),
        "system": system_information(),
        "correctness": {
            "status": "not-compared",
            "meaning": "compare output_hash with the matched reference run",
        },
    }

    if workload == "real":
        real_details = {
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
            "input_format": args.input_format,
            "blosc_threads_per_worker": (
                int(args.blosc_threads) if args.input_format == "zarr" else 0
            ),
            "experiment_log": (
                str(Path(getattr(args, "experiment_log", "")).expanduser().resolve())
                if getattr(args, "experiment_log", "")
                else ""
            ),
            "experiment_log_role": "provenance-only; not parsed by benchmark",
            "group_offset": int(args.group_offset),
            "io_leases": int(args.io_leases),
        }
        if args.input_format == "zarr":
            real_details["zarr_configuration"] = groups[0][1]["zarr_configuration"]
        record.update(real_details)
    else:
        record.update(
            {
                "synthetic_profile": (
                    "short" if args.samples_per_file == 8 * args.ns else "custom"
                ),
                "samples_per_file": int(args.samples_per_file),
                "generator_seed": int(args.seed),
                "noise_std": float(args.noise_std),
                "generation": args.generation,
            }
        )

    if framework_details:
        record.update(framework_details)
    return record


def write_run_files(record, results, output_directory):
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / (record["run_id"] + ".json")
    csv_path = destination / (record["run_id"] + "_groups.csv")

    if json_path.exists() or csv_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing run ID %s" % record["run_id"]
        )

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")

    fieldnames = (
        "group_id",
        "worker_host",
        "worker_pid",
        "generation_s",
        "wait_s",
        "read_s",
        "convert_s",
        "fft_s",
        "group_s",
        "input_bytes",
        "storage_bytes",
        "samples",
        "result_payload_bytes",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["group_id"]):
            row = {
                "group_id": result["group_id"],
                "worker_host": result["worker_host"],
                "worker_pid": result["worker_pid"],
                "input_bytes": result["input_bytes"],
                "storage_bytes": result["storage_bytes"],
                "samples": result["samples"],
                "result_payload_bytes": result["spectra"].nbytes,
            }
            row.update(result["timings"])
            writer.writerow(row)
    return str(json_path), str(csv_path)


def print_run_summary(record, output_paths):
    print(
        "[RESULT] framework=%s workload=%s input_format=%s numba=%s workers=%d "
        "groups=%d processing_s=%.6f throughput_gph=%.3f"
        % (
            record["framework"],
            record["workload"],
            record.get("input_format", "not-applicable"),
            record["numba"],
            record["worker_count"],
            record["group_count"],
            record["phases"]["processing_s"],
            record["throughput_groups_per_hour"],
        ),
        flush=True,
    )
    print("[RESULT] output_hash=%s" % record["output_hash"], flush=True)
    print("[RESULT] json=%s" % output_paths[0], flush=True)
    print("[RESULT] groups_csv=%s" % output_paths[1], flush=True)
