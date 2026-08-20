#!/usr/bin/env python3
"""Measure Raw-to-Zarr compression scaling without retaining test archives.

The production Zarr directory is never written by this program.  Every
condition gets a uniquely named, marker-owned directory below an explicitly
configured scratch root.  ``prepare_zarr.py`` performs exact verification;
only after the manifests and summary have been copied to the durable result
directory may the condition directory be removed.

Parallelism has two levels: several independent scan groups can be packed at
once, and Blosc may use several threads inside each packer process.  This is
important because adding Blosc threads alone does not test the generally more
scalable group-level parallelism.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from workloads import discover_raw_groups

try:
    import resource
except ImportError:  # pragma: no cover - the benchmark runs on Linux.
    resource = None


CONFIG_RE = re.compile(r"^(?P<processes>[1-9][0-9]*)x(?P<threads>[1-9][0-9]*)$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
OWNER_FILE = ".compression-scaling-owner.json"
CONDITION_OWNER_FILE = ".compression-scaling-condition.json"
PREPARE_FIELDS = (
    "utc_time",
    "group_id",
    "status",
    "raw_bytes",
    "zarr_bytes",
    "compression_ratio",
    "compressor",
    "compression_level",
    "shuffle",
    "chunk_samples",
    "blosc_threads",
    "read_copy_s",
    "write_compress_s",
    "pack_s",
    "pack_unattributed_s",
    "verify_s",
    "total_s",
    "pack_input_mib_s",
    "compress_input_mib_s",
    "archive",
)
GROUP_RESULT_FIELDS = (
    "experiment_id",
    "condition_id",
    "repetition",
    "order_index",
    "concurrent_groups",
    "threads_per_group",
    "total_compression_cores",
    "lane",
) + PREPARE_FIELDS
SUMMARY_FIELDS = (
    "utc_time",
    "experiment_id",
    "condition_id",
    "status",
    "error",
    "repetition",
    "order_index",
    "group_offset",
    "group_count",
    "first_group",
    "last_group",
    "concurrent_groups",
    "threads_per_group",
    "total_compression_cores",
    "compression_level",
    "chunk_samples",
    "wall_s",
    "raw_bytes",
    "zarr_bytes",
    "compression_ratio",
    "input_mib_s",
    "output_mib_s",
    "groups_per_hour",
    "sum_pack_s",
    "sum_write_compress_s",
    "sum_verify_s",
    "sum_user_cpu_s",
    "sum_system_cpu_s",
    "peak_lane_rss_kib",
    "sum_lane_peak_rss_kib",
    "resource_measurement",
    "temporary_bytes_before_cleanup",
    "filesystem_free_bytes_before",
    "filesystem_free_bytes_after",
    "cleanup",
)


class ExperimentStopped(RuntimeError):
    """Raised when the stop file requests a clean boundary stop."""


class SchedulerTermination(RuntimeError):
    """Raised when Slurm or the user asks the active job to terminate."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial-%d" % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_csv(path: Path, fields: tuple[str, ...], row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def parse_configs(text: str) -> list[tuple[int, int]]:
    tokens = [token.strip() for token in text.replace(" ", ",").split(",") if token.strip()]
    if not tokens:
        raise ValueError("at least one compression configuration is required")
    configurations = []
    for token in tokens:
        match = CONFIG_RE.fullmatch(token.lower())
        if not match:
            raise ValueError(
                "invalid configuration %r; use concurrent-groups x threads, for example 4x8"
                % token
            )
        configuration = (int(match.group("processes")), int(match.group("threads")))
        if configuration in configurations:
            raise ValueError("duplicate compression configuration: %s" % token)
        configurations.append(configuration)
    return configurations


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_roots(
    scratch_root: Path,
    raw_directory: Path,
    protected_zarr: Path,
    output_directory: Path,
) -> None:
    """Reject any layout in which temporary deletion could reach real data."""
    forbidden_exact = {
        resolved("/"),
        resolved(Path.home()),
        raw_directory,
        protected_zarr,
        output_directory,
    }
    if scratch_root in forbidden_exact:
        raise ValueError("unsafe compression scratch root: %s" % scratch_root)
    for protected in (raw_directory, protected_zarr, output_directory):
        if paths_overlap(scratch_root, protected):
            raise ValueError(
                "compression scratch root must be separate from protected path: %s and %s"
                % (scratch_root, protected)
            )
    if scratch_root.parent == scratch_root:
        raise ValueError("filesystem root cannot be used as compression scratch")


def initialise_run_root(scratch_root: Path, experiment_id: str) -> tuple[Path, str]:
    if not ID_RE.fullmatch(experiment_id) or experiment_id in (".", ".."):
        raise ValueError("unsafe experiment ID: %r" % experiment_id)
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch_root = scratch_root.resolve()
    run_root = (scratch_root / experiment_id).resolve(strict=False)
    if run_root.parent != scratch_root:
        raise ValueError("experiment directory escaped the configured scratch root")
    run_root.mkdir(exist_ok=True)
    owner_path = run_root / OWNER_FILE
    if owner_path.exists():
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        if owner.get("experiment_id") != experiment_id or owner.get("run_root") != str(run_root):
            raise ValueError("scratch ownership marker does not match this experiment")
        token = str(owner.get("token", ""))
        if not token:
            raise ValueError("scratch ownership marker has no token")
    else:
        unexpected = [path for path in run_root.iterdir()]
        if unexpected:
            raise ValueError("refusing unowned non-empty experiment directory: %s" % run_root)
        token = uuid.uuid4().hex
        atomic_json(
            owner_path,
            {
                "format": "compression-scaling-owner-v1",
                "experiment_id": experiment_id,
                "run_root": str(run_root),
                "token": token,
                "created_utc": utc_now(),
            },
        )
    (run_root / "conditions").mkdir(exist_ok=True)
    return run_root, token


def create_condition_directory(run_root: Path, token: str, condition_id: str) -> Path:
    conditions_root = (run_root / "conditions").resolve()
    condition = (conditions_root / condition_id).resolve(strict=False)
    if condition.parent != conditions_root:
        raise ValueError("condition directory escaped the owned run directory")
    if condition.exists():
        raise ValueError(
            "temporary condition directory already exists; inspect or remove it explicitly: %s"
            % condition
        )
    condition.mkdir()
    atomic_json(
        condition / CONDITION_OWNER_FILE,
        {
            "format": "compression-scaling-condition-v1",
            "condition_id": condition_id,
            "condition_path": str(condition),
            "run_token": token,
            "created_utc": utc_now(),
        },
    )
    return condition


def remove_owned_condition(run_root: Path, token: str, condition: Path) -> None:
    """Remove one exact condition tree, with redundant containment checks."""
    conditions_root = (run_root / "conditions").resolve()
    if condition.is_symlink():
        raise ValueError("refusing to remove a symlinked condition directory")
    condition = condition.resolve()
    if condition.parent != conditions_root or condition == conditions_root:
        raise ValueError("refusing deletion outside the owned conditions directory: %s" % condition)
    marker_path = condition / CONDITION_OWNER_FILE
    if not marker_path.is_file():
        raise ValueError("refusing deletion without a condition ownership marker: %s" % condition)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("format") != "compression-scaling-condition-v1"
        or marker.get("condition_path") != str(condition)
        or marker.get("run_token") != token
        or marker.get("condition_id") != condition.name
    ):
        raise ValueError("refusing deletion because the condition marker does not match")
    shutil.rmtree(condition)


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def find_temporary_payloads(scratch_root: Path) -> list[Path]:
    """Find archive-sized leftovers without deleting or following other roots."""
    if not scratch_root.exists():
        return []
    payloads = set(scratch_root.rglob("*.zarr.zip"))
    payloads.update(scratch_root.rglob("*.partial-*"))
    return sorted(path.resolve(strict=False) for path in payloads if path.is_file())


def require_no_temporary_payloads(scratch_root: Path, stage: str) -> None:
    payloads = find_temporary_payloads(scratch_root)
    if not payloads:
        return
    total_bytes = sum(path.stat().st_size for path in payloads)
    sample = "\n".join("  %s" % path for path in payloads[:10])
    if len(payloads) > 10:
        sample += "\n  ... and %d more" % (len(payloads) - 10)
    raise RuntimeError(
        "%s scratch audit found %d retained archive/partial files (%.2f GiB). "
        "Inspect them before starting another experiment:\n%s"
        % (stage, len(payloads), total_bytes / 1024**3, sample)
    )


def lane_assignments(group_offset: int, group_count: int, lanes: int) -> list[tuple[int, int]]:
    quotient, remainder = divmod(group_count, lanes)
    assignments = []
    offset = group_offset
    for lane in range(lanes):
        count = quotient + (1 if lane < remainder else 0)
        if count:
            assignments.append((offset, count))
            offset += count
    return assignments


def parse_gnu_time(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.is_file():
        return values
    keys = {
        "User time (seconds)": "user_s",
        "System time (seconds)": "system_s",
        "Maximum resident set size (kbytes)": "rss_kib",
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for label, key in keys.items():
            prefix = label + ":"
            if line.strip().startswith(prefix):
                try:
                    values[key] = float(line.strip()[len(prefix) :].strip())
                except ValueError:
                    pass
    return values


def child_usage_snapshot():
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_CHILDREN)


def child_usage_difference(before, after) -> dict[str, float]:
    if before is None or after is None:
        return {}
    return {
        "sum_user_cpu_s": max(0.0, after.ru_utime - before.ru_utime),
        "sum_system_cpu_s": max(0.0, after.ru_stime - before.ru_stime),
    }


def process_high_water_rss_kib(pid: int) -> float:
    """Read a live packer's Linux high-water RSS; return zero if it exited."""
    try:
        for line in Path("/proc/%d/status" % pid).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0.0


def sample_process_memory(
    processes: list[subprocess.Popen], peaks_by_pid: dict[int, float]
) -> None:
    for process in processes:
        peak = process_high_water_rss_kib(process.pid)
        if peak:
            peaks_by_pid[process.pid] = max(peaks_by_pid.get(process.pid, 0.0), peak)


def terminate_processes(processes: list[subprocess.Popen], grace_seconds: float = 10.0) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def read_lane_results(
    condition: Path,
    expected_group_ids: set[str],
    context: dict,
) -> tuple[list[dict], dict]:
    rows = []
    cpu_rows = []
    for lane_directory in sorted(condition.glob("lane-*")):
        lane_number = int(lane_directory.name.split("-", 1)[1])
        manifest = lane_directory / "conversion_manifest.csv"
        if not manifest.is_file():
            raise RuntimeError("lane %d did not produce a conversion manifest" % lane_number)
        with manifest.open(newline="", encoding="utf-8") as handle:
            for source_row in csv.DictReader(handle):
                if source_row.get("status") != "created":
                    raise RuntimeError(
                        "lane %d returned unexpected status %r"
                        % (lane_number, source_row.get("status"))
                    )
                archive = lane_directory / source_row["archive"]
                marker = archive.with_name(archive.name + ".verified.json")
                if not archive.is_file() or not marker.is_file():
                    raise RuntimeError("verified archive or marker is missing: %s" % archive)
                row = dict(context)
                row["lane"] = lane_number
                row.update(source_row)
                rows.append(row)
        cpu_rows.append(parse_gnu_time(context["logs_directory"] / ("lane-%02d.time.txt" % lane_number)))

    actual_group_ids = {row["group_id"] for row in rows}
    if actual_group_ids != expected_group_ids or len(rows) != len(expected_group_ids):
        raise RuntimeError(
            "condition produced the wrong groups: expected=%s actual=%s"
            % (sorted(expected_group_ids), sorted(actual_group_ids))
        )
    measured_cpu_rows = [item for item in cpu_rows if item]
    cpu = {}
    if measured_cpu_rows:
        cpu = {
            "sum_user_cpu_s": sum(item.get("user_s", 0.0) for item in measured_cpu_rows),
            "sum_system_cpu_s": sum(
                item.get("system_s", 0.0) for item in measured_cpu_rows
            ),
            "peak_lane_rss_kib": max(
                (item.get("rss_kib", 0.0) for item in measured_cpu_rows), default=0.0
            ),
            "sum_lane_peak_rss_kib": sum(
                item.get("rss_kib", 0.0) for item in measured_cpu_rows
            ),
            "resource_measurement": "gnu-time-per-lane",
        }
    return rows, cpu


def run_condition(
    args: argparse.Namespace,
    run_root: Path,
    token: str,
    all_groups: list,
    repetition: int,
    order_index: int,
    group_offset: int,
    concurrent_groups: int,
    threads_per_group: int,
) -> dict:
    selected = all_groups[group_offset : group_offset + args.groups_per_condition]
    if len(selected) != args.groups_per_condition:
        raise ValueError(
            "not enough groups at offset %d: requested %d, available %d"
            % (group_offset, args.groups_per_condition, len(selected))
        )
    group_ids = [group_id for group_id, _paths in selected]
    raw_bytes_expected = sum(
        Path(path).stat().st_size for _group_id, paths in selected for path in paths
    )
    total_cores = concurrent_groups * threads_per_group
    condition_id = "r%02d-o%02d-c%d-t%d-off%d" % (
        repetition,
        order_index,
        concurrent_groups,
        threads_per_group,
        group_offset,
    )
    logs_directory = args.output_dir / "conditions" / condition_id
    free_before = shutil.disk_usage(run_root).free
    reserve = int(args.min_free_gib * 1024**3)
    conservative_output = int(raw_bytes_expected * 1.05)
    if free_before < reserve + conservative_output:
        raise OSError(
            "insufficient scratch headroom: free %.1f GiB, require %.1f GiB plus %.1f GiB reserve"
            % (
                free_before / 1024**3,
                conservative_output / 1024**3,
                args.min_free_gib,
            )
        )
    logs_directory.mkdir(parents=True, exist_ok=False)
    condition = create_condition_directory(run_root, token, condition_id)

    print(
        "[%s] rep=%d order=%d groups=%s..%s concurrency=%d threads=%d cores=%d"
        % (
            condition_id,
            repetition,
            order_index,
            group_ids[0],
            group_ids[-1],
            concurrent_groups,
            threads_per_group,
            total_cores,
        ),
        flush=True,
    )
    assignments = lane_assignments(group_offset, args.groups_per_condition, concurrent_groups)
    processes: list[subprocess.Popen] = []
    peak_rss_by_pid: dict[int, float] = {}
    handles = []
    usage_before = child_usage_snapshot()
    started = time.perf_counter()
    status = "complete"
    error = ""
    try:
        for lane_index, (lane_offset, lane_count) in enumerate(assignments, start=1):
            lane_directory = condition / ("lane-%02d" % lane_index)
            lane_directory.mkdir()
            stdout_handle = (logs_directory / ("lane-%02d.stdout.log" % lane_index)).open(
                "w", encoding="utf-8"
            )
            stderr_handle = (logs_directory / ("lane-%02d.stderr.log" % lane_index)).open(
                "w", encoding="utf-8"
            )
            handles.extend((stdout_handle, stderr_handle))
            command = [
                sys.executable,
                "-u",
                str(args.prepare_script),
                str(args.data_dir),
                str(lane_directory),
                "--n-groups",
                str(lane_count),
                "--group-offset",
                str(lane_offset),
                "--compressor",
                "zstd",
                "--compression-level",
                str(args.compression_level),
                "--shuffle",
                "bitshuffle",
                "--blosc-threads",
                str(threads_per_group),
                "--chunk-samples",
                str(args.chunk_samples),
                "--min-free-gib",
                str(args.min_free_gib),
            ]
            if Path("/usr/bin/time").is_file():
                command = [
                    "/usr/bin/time",
                    "-v",
                    "-o",
                    str(logs_directory / ("lane-%02d.time.txt" % lane_index)),
                ] + command
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
            processes.append(process)

        deadline = time.monotonic() + args.condition_timeout_s
        while any(process.poll() is None for process in processes):
            sample_process_memory(processes, peak_rss_by_pid)
            failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError("packer lane failed with exit status %d" % failed[0])
            if time.monotonic() >= deadline:
                raise TimeoutError("condition exceeded %d seconds" % args.condition_timeout_s)
            time.sleep(0.25)
        failures = [process.returncode for process in processes if process.returncode != 0]
        if failures:
            raise RuntimeError("packer lane failed with exit status %d" % failures[0])
    except BaseException as exc:
        sample_process_memory(processes, peak_rss_by_pid)
        terminate_processes(processes)
        status = "failed"
        error = "%s: %s" % (type(exc).__name__, exc)
    finally:
        for handle in handles:
            handle.close()
    wall_seconds = time.perf_counter() - started
    usage_after = child_usage_snapshot()

    context = {
        "experiment_id": args.experiment_id,
        "condition_id": condition_id,
        "repetition": repetition,
        "order_index": order_index,
        "concurrent_groups": concurrent_groups,
        "threads_per_group": threads_per_group,
        "total_compression_cores": total_cores,
        "logs_directory": logs_directory,
    }
    rows = []
    cpu = {}
    if status == "complete":
        try:
            rows, cpu = read_lane_results(condition, set(group_ids), context)
        except BaseException as exc:
            status = "failed"
            error = "%s: %s" % (type(exc).__name__, exc)

    if not cpu:
        cpu = child_usage_difference(usage_before, usage_after)
        observed_peaks = list(peak_rss_by_pid.values())
        if observed_peaks:
            cpu["peak_lane_rss_kib"] = max(observed_peaks)
            cpu["sum_lane_peak_rss_kib"] = sum(observed_peaks)
        elif resource is not None and usage_after is not None:
            # ru_maxrss is the maximum among all children seen by this
            # controller. It is a useful fallback but may include an earlier
            # condition if /proc sampling missed a very short-lived lane.
            cpu["peak_lane_rss_kib"] = float(usage_after.ru_maxrss)
            cpu["sum_lane_peak_rss_kib"] = ""
        if cpu:
            cpu["resource_measurement"] = "python-resource-and-proc"

    manifest_path = logs_directory / "group_results.csv"
    if rows:
        for row in rows:
            row.pop("logs_directory", None)
            append_csv(manifest_path, GROUP_RESULT_FIELDS, row)

    raw_bytes = sum(int(row["raw_bytes"]) for row in rows)
    zarr_bytes = sum(int(row["zarr_bytes"]) for row in rows)
    temporary_bytes = tree_size(condition)
    summary = {
        "utc_time": utc_now(),
        "experiment_id": args.experiment_id,
        "condition_id": condition_id,
        "status": status,
        "error": error,
        "repetition": repetition,
        "order_index": order_index,
        "group_offset": group_offset,
        "group_count": len(rows),
        "first_group": group_ids[0],
        "last_group": group_ids[-1],
        "concurrent_groups": concurrent_groups,
        "threads_per_group": threads_per_group,
        "total_compression_cores": total_cores,
        "compression_level": args.compression_level,
        "chunk_samples": args.chunk_samples,
        "wall_s": "%.6f" % wall_seconds,
        "raw_bytes": raw_bytes,
        "zarr_bytes": zarr_bytes,
        "compression_ratio": "%.9f" % (zarr_bytes / raw_bytes) if raw_bytes else "",
        "input_mib_s": "%.6f" % (raw_bytes / 1024**2 / wall_seconds) if raw_bytes else "",
        "output_mib_s": "%.6f" % (zarr_bytes / 1024**2 / wall_seconds) if zarr_bytes else "",
        "groups_per_hour": "%.6f" % (len(rows) * 3600 / wall_seconds) if rows else "",
        "sum_pack_s": "%.6f" % sum(float(row["pack_s"]) for row in rows),
        "sum_write_compress_s": "%.6f"
        % sum(float(row["write_compress_s"]) for row in rows),
        "sum_verify_s": "%.6f" % sum(float(row["verify_s"]) for row in rows),
        "sum_user_cpu_s": "%.6f" % cpu.get("sum_user_cpu_s", 0.0) if cpu else "",
        "sum_system_cpu_s": "%.6f" % cpu.get("sum_system_cpu_s", 0.0) if cpu else "",
        "peak_lane_rss_kib": "%.0f" % cpu.get("peak_lane_rss_kib", 0.0) if cpu else "",
        "sum_lane_peak_rss_kib": "%.0f"
        % cpu.get("sum_lane_peak_rss_kib", 0.0)
        if cpu and cpu.get("sum_lane_peak_rss_kib", "") != ""
        else "",
        "resource_measurement": cpu.get("resource_measurement", "") if cpu else "",
        "temporary_bytes_before_cleanup": temporary_bytes,
        "filesystem_free_bytes_before": free_before,
        "filesystem_free_bytes_after": "",
        "cleanup": "retained",
    }
    # Preserve all measurements before any deletion. If the scheduler stops
    # the job during cleanup, this per-condition record remains recoverable.
    condition_summary_path = logs_directory / "condition_summary.json"
    atomic_json(condition_summary_path, summary)

    should_cleanup = args.cleanup == "always" or (
        status == "complete" and args.cleanup == "success"
    )
    if should_cleanup:
        try:
            remove_owned_condition(run_root, token, condition)
            summary["cleanup"] = "deleted-owned-condition"
        except BaseException as exc:
            summary["status"] = "cleanup-failed"
            cleanup_error = "%s: %s" % (type(exc).__name__, exc)
            summary["error"] = (
                summary["error"] + "; cleanup " + cleanup_error
                if summary["error"]
                else cleanup_error
            )
    summary["filesystem_free_bytes_after"] = shutil.disk_usage(run_root).free
    atomic_json(condition_summary_path, summary)
    append_csv(args.output_dir / "compression_scaling_summary.csv", SUMMARY_FIELDS, summary)

    if summary["status"] != "complete":
        temporary_state = (
            "retained at %s" % condition
            if condition.exists()
            else "removed after durable logs were saved"
        )
        raise RuntimeError(
            "condition %s failed; temporary files %s; %s"
            % (condition_id, temporary_state, summary["error"])
        )
    print(
        "  %.1f GiB -> %.1f GiB in %.1fs (%.1f MiB/s, cleanup=%s)"
        % (
            raw_bytes / 1024**3,
            zarr_bytes / 1024**3,
            wall_seconds,
            raw_bytes / 1024**2 / wall_seconds,
            summary["cleanup"],
        ),
        flush=True,
    )
    return summary


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=resolved, help="Read-only directory containing Raw files")
    parser.add_argument("scratch_root", type=resolved, help="Dedicated temporary experiment root")
    parser.add_argument("output_dir", type=resolved, help="Durable result/log directory")
    parser.add_argument("--protected-zarr-dir", type=resolved, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--configs",
        default="1x8,1x16,1x32,2x8,4x8,8x4",
        help="Comma-separated concurrent-groups x Blosc-threads configurations",
    )
    parser.add_argument("--groups-per-condition", type=int, default=8)
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument(
        "--group-offset-step",
        type=int,
        default=8,
        help="Advance the matched group window between repetitions",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--compression-level", type=int, choices=range(10), default=9)
    parser.add_argument("--chunk-samples", type=int, default=1_000_000)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    parser.add_argument("--condition-timeout-s", type=int, default=5400)
    parser.add_argument(
        "--cleanup",
        choices=("always", "success", "never"),
        default="always",
        help=(
            "Delete the current marker-owned condition after durable logs are saved; "
            "the full matrix defaults to cleanup after success, failure, or interruption"
        ),
    )
    parser.add_argument("--stop-file", type=resolved)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.configurations = parse_configs(args.configs)
    if args.groups_per_condition < 1 or args.repetitions < 1:
        parser.error("group and repetition counts must be positive")
    if args.group_offset < 0 or args.group_offset_step < 0:
        parser.error("group offsets must be non-negative")
    if args.chunk_samples < 1 or args.min_free_gib < 0 or args.condition_timeout_s < 1:
        parser.error("chunk size and timeout must be positive; free-space reserve cannot be negative")
    for concurrent_groups, _threads in args.configurations:
        if concurrent_groups > args.groups_per_condition:
            parser.error("concurrent groups cannot exceed groups per condition")
    args.prepare_script = resolved(Path(__file__).with_name("prepare_zarr.py"))
    return args


def main() -> int:
    args = parse_args()

    def terminate_on_signal(signum, _frame):
        raise SchedulerTermination("received signal %d" % signum)

    signal.signal(signal.SIGTERM, terminate_on_signal)
    signal.signal(signal.SIGINT, terminate_on_signal)
    try:
        if not args.data_dir.is_dir():
            raise ValueError("Raw data directory does not exist: %s" % args.data_dir)
        if not args.protected_zarr_dir.is_dir():
            raise ValueError("protected completed Zarr directory does not exist: %s" % args.protected_zarr_dir)
        validate_roots(args.scratch_root, args.data_dir, args.protected_zarr_dir, args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        maximum_cores = max(processes * threads for processes, threads in args.configurations)
        if maximum_cores > available_cpus():
            raise ValueError(
                "configuration requires %d CPUs but this process can use only %d"
                % (maximum_cores, available_cpus())
            )
        all_groups = discover_raw_groups(args.data_dir, 0, 0)
        last_required = (
            args.group_offset
            + (args.repetitions - 1) * args.group_offset_step
            + args.groups_per_condition
        )
        if last_required > len(all_groups):
            raise ValueError(
                "experiment needs groups through offset %d, but only %d complete groups exist"
                % (last_required, len(all_groups))
            )

        orders = []
        for repetition in range(1, args.repetitions + 1):
            order = list(args.configurations)
            random.Random(args.seed + repetition).shuffle(order)
            orders.append(order)
        plan = {
            "format": "compression-scaling-plan-v1",
            "experiment_id": args.experiment_id,
            "raw_directory": str(args.data_dir),
            "protected_zarr_directory": str(args.protected_zarr_dir),
            "scratch_root": str(args.scratch_root),
            "output_directory": str(args.output_dir),
            "configurations": ["%dx%d" % item for item in args.configurations],
            "orders": [["%dx%d" % item for item in order] for order in orders],
            "groups_per_condition": args.groups_per_condition,
            "group_offset": args.group_offset,
            "group_offset_step": args.group_offset_step,
            "repetitions": args.repetitions,
            "compression_level": args.compression_level,
            "chunk_samples": args.chunk_samples,
            "min_free_gib": args.min_free_gib,
            "condition_timeout_s": args.condition_timeout_s,
            "cleanup": args.cleanup,
            "available_cpus": available_cpus(),
            "cache_regime": os.environ.get("CACHE_REGIME", "uncontrolled/mixed"),
        }
        plan_path = args.output_dir / "compression_scaling_plan.json"
        if plan_path.exists():
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
            if existing != plan:
                raise ValueError("existing result directory contains a different experiment plan")
        else:
            atomic_json(plan_path, plan)

        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        if args.dry_run:
            print("DRY RUN: plan validated; no temporary directory was created.")
            return 0

        require_no_temporary_payloads(args.scratch_root, "Preflight")
        run_root, token = initialise_run_root(args.scratch_root, args.experiment_id)
        stop_file = args.stop_file or (args.output_dir / "STOP")
        print("Owned temporary run root: %s" % run_root)
        print("Create %s to stop before the next condition." % stop_file, flush=True)
        for repetition, order in enumerate(orders, start=1):
            group_offset = args.group_offset + (repetition - 1) * args.group_offset_step
            for order_index, (concurrent_groups, threads_per_group) in enumerate(order, start=1):
                if stop_file.exists():
                    raise ExperimentStopped("stop file found: %s" % stop_file)
                run_condition(
                    args,
                    run_root,
                    token,
                    all_groups,
                    repetition,
                    order_index,
                    group_offset,
                    concurrent_groups,
                    threads_per_group,
                )
        require_no_temporary_payloads(args.scratch_root, "Final")
        print("Compression scaling experiment complete.")
        return 0
    except ExperimentStopped as exc:
        print("STOPPED CLEANLY: %s" % exc)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
