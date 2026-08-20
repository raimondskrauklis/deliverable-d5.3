#!/usr/bin/env python3
"""Create verified, resumable Zarr ZipStore copies of Raw scan groups.

This is a preparation step, not part of the timed Raw-versus-Zarr workload.
Raw inputs are opened read-only.  Each output is built under a unique partial
name, compared exactly with all eight source files, and only then renamed to
its final name.  A small sidecar records successful verification so rerunning
the command can cheaply skip completed groups.

The archive layout is deliberately simple: one ``<group>.zarr.zip`` file per
scan group and one ``(samples, 2)`` int16 array per source Raw file.  Columns
zero and one are the real and imaginary components respectively.
"""

import argparse
import csv
import errno
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from workloads import RAW_DTYPE, discover_raw_groups


FORMAT_NAME = "telescope-raw-iq-zarr-zip-v1"
MANIFEST_FIELDS = (
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


def stop_cleanly(signum, _frame):
    """Turn a scheduler/timeout termination into normal partial-file cleanup."""
    raise KeyboardInterrupt("received signal %d" % signum)


def load_zarr_dependencies():
    """Load the pinned Zarr 2 API with an actionable error message."""
    try:
        import zarr
        from numcodecs import Blosc, blosc
    except ImportError as exc:
        raise RuntimeError(
            "Zarr preparation requires zarr 2 and numcodecs. Install them with "
            "`python -m pip install -r requirements-zarr.txt`."
        ) from exc

    major = int(zarr.__version__.split(".", 1)[0])
    if major != 2:
        raise RuntimeError(
            "This archive format is pinned to Zarr 2, but zarr %s is installed. "
            "Install requirements-zarr.txt." % zarr.__version__
        )
    return zarr, Blosc, blosc


def source_records(paths):
    records = []
    for path_string in paths:
        path = Path(path_string)
        stat = path.stat()
        records.append(
            {
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return records


def marker_path(archive_path):
    return archive_path.with_name(archive_path.name + ".verified.json")


def marker_matches(archive_path, paths):
    sidecar = marker_path(archive_path)
    if not archive_path.is_file() or not sidecar.is_file():
        return False
    try:
        marker = json.loads(sidecar.read_text(encoding="utf-8"))
        archive_stat = archive_path.stat()
        return (
            marker.get("format") == FORMAT_NAME
            and marker.get("archive_size") == int(archive_stat.st_size)
            and marker.get("archive_mtime_ns") == int(archive_stat.st_mtime_ns)
            and marker.get("sources") == source_records(paths)
        )
    except (OSError, ValueError, TypeError):
        return False


def write_marker(archive_path, group_id, paths):
    stat = archive_path.stat()
    marker = {
        "format": FORMAT_NAME,
        "group_id": group_id,
        "archive": archive_path.name,
        "archive_size": int(stat.st_size),
        "archive_mtime_ns": int(stat.st_mtime_ns),
        "sources": source_records(paths),
        "verification": "exact-int16-chunk-comparison",
    }
    destination = marker_path(archive_path)
    temporary = destination.with_name(destination.name + ".partial-%d" % os.getpid())
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def create_archive(group_id, paths, partial_path, args, zarr, Blosc, blosc):
    blosc.set_nthreads(args.blosc_threads)
    shuffle = Blosc.BITSHUFFLE if args.shuffle == "bitshuffle" else Blosc.SHUFFLE
    compressor = Blosc(
        cname=args.compressor,
        clevel=args.compression_level,
        shuffle=shuffle,
    )

    read_copy_seconds = 0.0
    write_compress_seconds = 0.0
    started = time.perf_counter()
    store = zarr.ZipStore(str(partial_path), mode="w")
    write_compress_seconds += time.perf_counter() - started
    try:
        started = time.perf_counter()
        root = zarr.group(store=store, overwrite=True)
        records = source_records(paths)
        root.attrs.update(
            {
                "format": FORMAT_NAME,
                "group_id": group_id,
                "raw_dtype": "little-endian structured int16 (re, im)",
                "member_order": [record["name"] for record in records],
                "sources": records,
                "compressor": args.compressor,
                "compression_level": args.compression_level,
                "shuffle": args.shuffle,
                "chunk_samples": args.chunk_samples,
            }
        )
        write_compress_seconds += time.perf_counter() - started

        for path_string in paths:
            path = Path(path_string)
            raw = np.memmap(path, mode="r", dtype=RAW_DTYPE)
            sample_count = int(raw.shape[0])
            started = time.perf_counter()
            output = root.create_dataset(
                path.name,
                shape=(sample_count, 2),
                chunks=(min(args.chunk_samples, sample_count), 2),
                dtype=np.int16,
                compressor=compressor,
                overwrite=False,
            )
            write_compress_seconds += time.perf_counter() - started
            for start in range(0, sample_count, args.chunk_samples):
                end = min(start + args.chunk_samples, sample_count)
                started = time.perf_counter()
                chunk = np.empty((end - start, 2), dtype=np.int16)
                chunk[:, 0] = raw["re"][start:end]
                chunk[:, 1] = raw["im"][start:end]
                read_copy_seconds += time.perf_counter() - started

                started = time.perf_counter()
                output[start:end, :] = chunk
                write_compress_seconds += time.perf_counter() - started
            del output, raw
    finally:
        started = time.perf_counter()
        store.close()
        write_compress_seconds += time.perf_counter() - started
    return read_copy_seconds, write_compress_seconds


def verify_archive(group_id, paths, archive_path, chunk_samples, zarr):
    """Compare every stored int16 value with its source Raw file."""
    store = zarr.ZipStore(str(archive_path), mode="r")
    try:
        root = zarr.open_group(store=store, mode="r")
        if root.attrs.get("format") != FORMAT_NAME:
            raise ValueError("unexpected archive format in %s" % archive_path)
        if root.attrs.get("group_id") != group_id:
            raise ValueError("group ID mismatch in %s" % archive_path)

        expected_names = [Path(path).name for path in paths]
        if sorted(root.array_keys()) != sorted(expected_names):
            raise ValueError("archive member list does not match Raw group %s" % group_id)

        for path_string in paths:
            path = Path(path_string)
            raw = np.memmap(path, mode="r", dtype=RAW_DTYPE)
            stored = root[path.name]
            expected_shape = (int(raw.shape[0]), 2)
            if stored.shape != expected_shape or np.dtype(stored.dtype) != np.dtype(np.int16):
                raise ValueError(
                    "%s has shape/dtype %s/%s; expected %s/int16"
                    % (path.name, stored.shape, stored.dtype, expected_shape)
                )
            for start in range(0, raw.shape[0], chunk_samples):
                end = min(start + chunk_samples, raw.shape[0])
                actual = stored[start:end, :]
                if not np.array_equal(actual[:, 0], raw["re"][start:end]):
                    raise ValueError("real-component mismatch: %s at sample %d" % (path.name, start))
                if not np.array_equal(actual[:, 1], raw["im"][start:end]):
                    raise ValueError("imaginary-component mismatch: %s at sample %d" % (path.name, start))
            del stored, raw
    finally:
        store.close()


def append_manifest(path, row):
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def ensure_free_space(output_directory, minimum_free_gib):
    free = shutil.disk_usage(output_directory).free
    minimum = int(minimum_free_gib * (1024 ** 3))
    if free < minimum:
        raise OSError(
            errno.ENOSPC,
            "filesystem free space %.1f GiB is below --min-free-gib %.1f"
            % (free / 1024 ** 3, minimum_free_gib),
        )


def prepare_group(group_id, paths, output_directory, args, zarr, Blosc, blosc):
    archive = output_directory / (group_id + ".zarr.zip")
    raw_bytes = sum(Path(path).stat().st_size for path in paths)

    if marker_matches(archive, paths):
        return (
            "verified-existing",
            archive,
            raw_bytes,
            archive.stat().st_size,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    if archive.exists():
        print("  Existing archive lacks a current marker; verifying it exactly.", flush=True)
        verify_started = time.perf_counter()
        verify_archive(group_id, paths, archive, args.chunk_samples, zarr)
        verify_seconds = time.perf_counter() - verify_started
        write_marker(archive, group_id, paths)
        return (
            "verified-existing",
            archive,
            raw_bytes,
            archive.stat().st_size,
            0.0,
            0.0,
            0.0,
            verify_seconds,
            verify_seconds,
        )

    ensure_free_space(output_directory, args.min_free_gib)
    partial = archive.with_name(archive.name + ".partial-%d" % os.getpid())
    started = time.perf_counter()
    try:
        pack_started = time.perf_counter()
        read_copy_seconds, write_compress_seconds = create_archive(
            group_id, paths, partial, args, zarr, Blosc, blosc
        )
        pack_seconds = time.perf_counter() - pack_started
        verify_started = time.perf_counter()
        verify_archive(group_id, paths, partial, args.chunk_samples, zarr)
        verify_seconds = time.perf_counter() - verify_started
        # Hard-link publication is atomic on this one filesystem and, unlike
        # os.replace(), cannot overwrite a final archive created concurrently.
        os.link(partial, archive)
        partial.unlink()
        write_marker(archive, group_id, paths)
    except BaseException:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    return (
        "created",
        archive,
        raw_bytes,
        archive.stat().st_size,
        read_copy_seconds,
        write_compress_seconds,
        pack_seconds,
        verify_seconds,
        time.perf_counter() - started,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create exact, verified Zarr ZipStore copies of Raw scan groups."
    )
    parser.add_argument("data_dir", help="Directory containing observation .raw files")
    parser.add_argument("zarr_dir", help="New scratch directory for .zarr.zip archives")
    parser.add_argument("--n-groups", type=int, default=0, help="Groups to prepare; 0 means all")
    parser.add_argument("--group-offset", type=int, default=0)
    parser.add_argument("--compressor", default="zstd", choices=("zstd", "lz4", "blosclz"))
    parser.add_argument("--compression-level", type=int, default=9, choices=range(0, 10))
    parser.add_argument("--shuffle", default="bitshuffle", choices=("bitshuffle", "shuffle"))
    parser.add_argument("--blosc-threads", type=int, default=8)
    parser.add_argument("--chunk-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=20.0,
        help="Stop before starting another group if filesystem free space is below this value",
    )
    args = parser.parse_args()
    if args.n_groups < 0 or args.group_offset < 0:
        parser.error("group counts and offsets must be non-negative")
    if args.blosc_threads < 1 or args.chunk_samples < 1:
        parser.error("thread and chunk counts must be positive")
    if args.min_free_gib < 0:
        parser.error("--min-free-gib must be non-negative")
    return args


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, stop_cleanly)
    signal.signal(signal.SIGINT, stop_cleanly)
    try:
        zarr, Blosc, blosc = load_zarr_dependencies()
        groups = discover_raw_groups(args.data_dir, args.group_offset, args.n_groups)
        output_directory = Path(args.zarr_dir).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest = output_directory / "conversion_manifest.csv"

        print("Raw directory:  %s" % Path(args.data_dir).expanduser().resolve())
        print("Zarr directory: %s" % output_directory)
        print("Groups:         %d" % len(groups))
        print(
            "Codec:          Blosc %s level=%d %s, threads=%d"
            % (args.compressor, args.compression_level, args.shuffle, args.blosc_threads)
        )
        print("Verification:   exact int16 comparison before final rename", flush=True)

        total_raw = 0
        total_zarr = 0
        created_raw = 0
        created_pack_seconds = 0.0
        created_write_compress_seconds = 0.0
        for index, (group_id, paths) in enumerate(groups, start=1):
            print("[%d/%d] %s" % (index, len(groups), group_id), flush=True)
            result = prepare_group(
                group_id, paths, output_directory, args, zarr, Blosc, blosc
            )
            (
                status,
                archive,
                raw_bytes,
                zarr_bytes,
                read_copy_s,
                write_compress_s,
                pack_s,
                verify_s,
                total_s,
            ) = result
            ratio = zarr_bytes / raw_bytes
            pack_unattributed_s = max(0.0, pack_s - read_copy_s - write_compress_s)
            pack_input_mib_s = raw_bytes / 1024 ** 2 / pack_s if pack_s else 0.0
            compress_input_mib_s = (
                raw_bytes / 1024 ** 2 / write_compress_s if write_compress_s else 0.0
            )
            total_raw += raw_bytes
            total_zarr += zarr_bytes
            if status == "created":
                created_raw += raw_bytes
                created_pack_seconds += pack_s
                created_write_compress_seconds += write_compress_s
            row = {
                "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "group_id": group_id,
                "status": status,
                "raw_bytes": raw_bytes,
                "zarr_bytes": zarr_bytes,
                "compression_ratio": "%.9f" % ratio,
                "compressor": args.compressor,
                "compression_level": args.compression_level,
                "shuffle": args.shuffle,
                "chunk_samples": args.chunk_samples,
                "blosc_threads": args.blosc_threads,
                "read_copy_s": "%.6f" % read_copy_s,
                "write_compress_s": "%.6f" % write_compress_s,
                "pack_s": "%.6f" % pack_s,
                "pack_unattributed_s": "%.6f" % pack_unattributed_s,
                "verify_s": "%.6f" % verify_s,
                "total_s": "%.6f" % total_s,
                "pack_input_mib_s": "%.6f" % pack_input_mib_s if pack_s else "",
                "compress_input_mib_s": (
                    "%.6f" % compress_input_mib_s if write_compress_s else ""
                ),
                "archive": archive.name,
            }
            append_manifest(manifest, row)
            print(
                "  %s: %.2f GiB -> %.2f GiB (%.1f%%); "
                "read/copy %.1fs, compress/write %.1fs, verify %.1fs, total %.1fs"
                % (
                    status,
                    raw_bytes / 1024 ** 3,
                    zarr_bytes / 1024 ** 3,
                    ratio * 100.0,
                    read_copy_s,
                    write_compress_s,
                    verify_s,
                    total_s,
                ),
                flush=True,
            )

        print(
            "Complete: %.2f GiB Raw -> %.2f GiB Zarr (%.1f%%)"
            % (total_raw / 1024 ** 3, total_zarr / 1024 ** 3, total_zarr / total_raw * 100.0)
        )
        if created_raw and created_pack_seconds:
            print(
                "New conversion throughput: %.1f MiB/s end-to-end pack; "
                "%.1f MiB/s during compression/write"
                % (
                    created_raw / 1024 ** 2 / created_pack_seconds,
                    created_raw / 1024 ** 2 / created_write_compress_seconds,
                )
            )
        return 0
    except OSError as exc:
        if exc.errno in (errno.EDQUOT, errno.ENOSPC):
            print("STOPPED SAFELY: scratch quota or free-space limit reached: %s" % exc, file=sys.stderr)
            return 3
        raise
    except (RuntimeError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
