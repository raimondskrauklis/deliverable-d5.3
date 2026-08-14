from __future__ import annotations

import os
import sys
import threading
import uuid
import copy
import logging
import numpy as np
import pandas as pd
import xarray as xr
import dask
import math
from concurrent.futures import ThreadPoolExecutor
import queue
from pathlib import Path
import itertools
from distributed import MultiLock
from dask.distributed import Client
from dask.distributed import performance_report
from dask.distributed import get_worker
from distributed import print as dask_print
from xradio.measurement_set import open_processing_set
import dask.array as da
import zarr
import time
from datetime import datetime
import argparse
import idg
import idg.util
import json
from collections.abc import Mapping, Sequence
from typing import TypeAlias
from pprint import pformat
import contextlib


@contextlib.contextmanager
def capture_native_output(path: str):
    """Context manager: redirect this process's OS-level stdout/stderr (fd 1/2) to path, restoring them on exit."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    with open(path, "ab", buffering=0) as output:
        try:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)

            os.close(old_stdout)
            os.close(old_stderr)


#EO: check this one...
## Disable retries ()
##dask.config.set({"distributed.scheduler.allowed-failures": 0})

# Make flush systematic
from functools import partial
print = partial(print, flush=True)


# Log with millisecond info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def json_safe(obj):
    """Recursively convert obj (slices, numpy scalars/arrays/dtypes, zarr dtype objects, tuples/lists/dicts) into something json.dumps() can serialize."""
    # Python slice, e.g. slice(0, 10, None)
    if isinstance(obj, slice):
        return {
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        }

    # NumPy scalar, e.g. np.int64, np.float32
    if isinstance(obj, np.generic):
        return obj.item()

    # NumPy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # NumPy dtype
    if isinstance(obj, np.dtype):
        return str(obj)

    # Zarr v3 dtype objects, e.g. Complex64DType
    if obj.__class__.__name__.endswith("DType"):
        if hasattr(obj, "to_native_dtype"):
            return str(obj.to_native_dtype())
        return str(obj)

    # Tuple/list, including tuple of slices
    if isinstance(obj, tuple):
        return [json_safe(x) for x in obj]

    if isinstance(obj, list):
        return [json_safe(x) for x in obj]

    # Dict
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    return obj


def _append_stat_record(record, path):
    """Append record as one JSON line to path, shared by append_worker_stat()/append_client_stat()."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(", ", ": "),
                default=str,
            )
            + "\n"
        )


def append_worker_stat(record, out_dir=".", out_tag="worker_stats"):
    """Append record (augmented with time/worker/pid) to this worker's own JSONL stats file under out_dir."""
    worker = get_worker()

    os.makedirs(out_dir, exist_ok=True)

    safe_name = (
        worker.address
        .replace("://", "_")
        .replace("/", "_")
        .replace(":", "_")
    )

    path = os.path.join(out_dir, f"{out_tag}_{safe_name}_pid{os.getpid()}.jsonl")

    record = {
        "time": time.time(),
        "worker": worker.address,
        "pid": os.getpid(),
        **record,
    }

    _append_stat_record(record, path)


def append_client_stat(record, out_dir=".", out_tag="client_stats"):
    """Append record (augmented with time/pid) to the client's JSONL stats file under out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{out_tag}.jsonl")

    record = {
        "time": time.time(),
        "pid": os.getpid(),
        **record,
    }

    _append_stat_record(record, path)


def guess_zarr_format(path):
    """
    Guess whether a Zarr store/array/group is v2 or v3 from filesystem metadata.

    Returns:
        2, 3, or None
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(p)

    # Zarr v3 metadata
    zarr_json = p / "zarr.json"
    if zarr_json.exists():
        try:
            with zarr_json.open("r") as f:
                meta = json.load(f)
            if meta.get("zarr_format") == 3:
                return 3
        except Exception:
            # zarr.json exists, so it is still very likely v3
            return 3

    # Zarr v2 group metadata
    if (p / ".zgroup").exists():
        try:
            with (p / ".zgroup").open("r") as f:
                meta = json.load(f)
            if meta.get("zarr_format") == 2:
                return 2
        except Exception:
            return 2

    # Zarr v2 array metadata
    if (p / ".zarray").exists():
        try:
            with (p / ".zarray").open("r") as f:
                meta = json.load(f)
            if meta.get("zarr_format") == 2:
                return 2
        except Exception:
            return 2

    # Consolidated v2 metadata at root
    if (p / ".zmetadata").exists():
        try:
            with (p / ".zmetadata").open("r") as f:
                meta = json.load(f)
            zmeta = meta.get("metadata", {})
            if any(k.endswith(".zarray") or k.endswith(".zgroup") for k in zmeta):
                return 2
        except Exception:
            return 2

    return None


def format_cpu_list(cpus):
    """Format an iterable of CPU ids as a compact range string, e.g. [0,1,2,4] -> "0-2,4"."""
    cpus = sorted(cpus)
    if not cpus:
        return "[]"

    ranges = []
    start = prev = cpus[0]

    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = c

    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def check_nan_inf(x, name="array"):
    """Print a NaN/inf diagnostic summary for array x (labeled name) and return True if any non-finite value is present."""
    x = np.asarray(x)

    nan_mask = np.isnan(x)
    inf_mask = np.isinf(x)
    bad_mask = ~np.isfinite(x)   # True for NaN, +inf, -inf

    print(f"{name}: shape={x.shape}, dtype={x.dtype}")
    print(f"  has NaN: {nan_mask.any()}  count={nan_mask.sum()}")
    print(f"  has inf: {inf_mask.any()}  count={inf_mask.sum()}")
    print(f"  has bad: {bad_mask.any()}  count={bad_mask.sum()}")

    if bad_mask.any():
        first_bad = np.argwhere(bad_mask)[0]
        print(f"  first bad index: {tuple(first_bad)}")
        print(f"  first bad value: {x[tuple(first_bad)]}")

    return bad_mask.any()


def _idg_init_proxy_cache(proxy, subgrid_size, cell_size, w_step, shift):
    """Call proxy.init_cache() once per worker (cached in w._ps_cache); a no-op on later calls. Requires reset_worker_cache() to have run first so cache["initialized"] exists."""
    w = get_worker()
    pid = os.getpid()
    cache = getattr(w, "_ps_cache", None)
    if cache is None:
        cache = w._ps_cache = {}
    if not cache["initialized"]:
        proxy.init_cache(subgrid_size, cell_size, w_step, shift)
        cache["initialized"] = True
        dask_print(f"{wlog_prefix(w.name, w.address, pid)} Cache initialized on worker {w}", flush=True)


def _get_grid(grid_size):
    """Return this worker's cached (4, grid_size, grid_size) complex64 IDG grid, allocating it on first use."""
    w = get_worker()
    cache = getattr(w, "_ps_cache", None)
    if cache is None:
        cache = w._ps_cache = {}
    if "grid" not in cache:
        cache["grid"] = np.zeros((4, grid_size, grid_size), dtype=np.complex64)
    return cache["grid"]


def _get_idg_proxy(proxy_mode):
    """Return this worker's cached IDG proxy for proxy_mode ("cpu"/"hybrid"/"gpu"), creating it on first use."""
    w = get_worker()
    pid = os.getpid()
    cache = getattr(w, "_ps_cache", None)

    if cache is None:
        cache = w._ps_cache = {}

    if proxy_mode not in cache:
        if proxy_mode == "cpu":
            proxy = idg.CPU.Optimized()
        elif proxy_mode == "hybrid":
            proxy = idg.HybridCUDA.GenericOptimized()
        elif proxy_mode == "gpu":
            proxy = idg.CUDA.Generic()
        else:
            raise ValueError(f"Unexpected value provided for IDG proxy mode: {proxy_mode}. Valid are cpu, hybrid and gpu")
        dask_print(f"{wlog_prefix(w.name, w.address, pid)} IDG get proxy =", proxy, flush=True)
        cache[proxy_mode] = proxy
    else:
        dask_print(f"{wlog_prefix(w.name, w.address, pid)} IDG get proxy (from cache) =", cache[proxy_mode], flush=True)

    return cache[proxy_mode]


def reset_worker_cache(dask_worker=None):
    """Runs on each worker via client.run(): wipe its per-worker IDG/processing-set cache."""
    w = dask_worker
    cache = getattr(w, "_ps_cache", None)
    if cache is not None:
        del w._ps_cache
    w._ps_cache = {}
    cache = w._ps_cache
    cache["initialized"] = False


def reset_ps(msv4: str, dask_worker=None):
    """Runs on each worker via client.run(): reopen and cache the processing set for msv4."""
    w = dask_worker
    cache = getattr(w, "_ps_cache", None)
    if cache is not None:
        cache["msv4"] = open_processing_set(ps_store=msv4)


def blockinfo_to_slice_tuple(block_info):
    """Convert a da.map_blocks() block_info[...]["array-location"] list of (start, stop) pairs into a tuple of slices."""
    al = block_info
    t = tuple(slice(s, e) for s, e in al)
    return t


def now_ms_string() -> str:
    """Current local time as "YYYY-MM-DD HH:MM:SS.mmm"."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def wlog_prefix(name, address, pid, level="I"):
    """Standard "-<level>- <timestamp> worker <name> (<pid>, <address>)" log prefix."""
    return f"-{level}- {now_ms_string()} worker {name} ({pid:7d}, {address:24s})"


#####
import ctypes


def get_current_cpu():
    """Return the CPU core the calling thread is currently running on (via libc sched_getcpu())."""
    # Lazily opened per-process (not at module import time) so that no
    # ctypes.CDLL handle — which holds raw C function pointers and cannot
    # be pickled — ever needs to be captured when this function is shipped
    # to workers via cloudpickle as part of a dask task graph.
    if not hasattr(get_current_cpu, "_libc"):
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.sched_getcpu.restype = ctypes.c_int
        get_current_cpu._libc = libc

    cpu = get_current_cpu._libc.sched_getcpu()
    if cpu < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "sched_getcpu failed")
    return cpu


def _process_baseline_slice(
        start,
        vis_slice,
        flag_slice,
        strategy_path: str,
        existing_flags_slice=None,
        *,
        worker_address: str = None,
        worker_name: str = None,
        pin_cpu_threads: bool = False,
        pinned_cpu_by_tid=None,
):
    """
    Run AOFlagger on one baseline-axis slice (one ThreadPoolExecutor work item).

    Called from flag_me() via pool.submit(), one call per baseline sub-range
    of the chunk being processed. start is the baseline offset of this slice
    within the chunk (used by the caller to place the result back).

    Returns (start, flags_slice, prt), where flags_slice is the new flag
    array for this slice and prt is a one-line timing/diagnostics string to
    be logged by the caller. dask_print()/get_worker() rely on thread-local
    state scoped to Dask's own task-execution thread, which this function
    does not run on (it runs in a separate ThreadPoolExecutor spawned by
    flag_me()) — hence worker_name/worker_address are passed in explicitly
    and logging is deferred to the caller instead of done here directly.
    """
    t0 = time.perf_counter()

    native_tid = threading.get_native_id()

    pinned_cpu = None
    if pinned_cpu_by_tid is not None:
        pinned_cpu = pinned_cpu_by_tid.get(native_tid)

    affinity_set = os.sched_getaffinity(0)
    affinity_list = sorted(affinity_set)

    if pin_cpu_threads and len(affinity_set) != 1:
        raise RuntimeError(
            f"CPU pinning is enabled, but expected single-CPU affinity, "
            f"got {affinity_list}"
        )

    pid = os.getpid()
    py_tid = threading.get_ident()

    import aoflagger

    t0 = time.perf_counter()

    # Each thread creates its own AOFlagger Strategy.
    flagger = aoflagger.AOFlagger()
    strategy = flagger.load_strategy_file(strategy_path)

    n_bsl, n_time, n_chan, n_pol = vis_slice.shape

    flags_list = []

    imageset = flagger.make_image_set(n_time, n_chan, n_pol * 2)
    flagset  = flagger.make_flag_mask(n_time, n_chan)

    t3 = time.perf_counter()

    cpu_run_before = get_current_cpu()

    for i in range(n_bsl):

        b_vis  = vis_slice[i]
        b_flag = flag_slice[i]

        flagset.set_buffer(np.transpose(b_flag))

        for j in range(0, 4):
            p_vis = b_vis[:,:,j]
            imageset.set_image_buffer(2*j,   np.transpose(np.real(p_vis).data))
            imageset.set_image_buffer(2*j+1, np.transpose(np.imag(p_vis).data))

        flags = strategy.run(imageset, flagset)

        flags_values = flags.get_buffer()

        flags_list.append(flags_values)

    cpu_run_after = get_current_cpu()

    t4 = time.perf_counter()

    flags_slice = np.stack(flags_list, axis=0)

    t5 = time.perf_counter()

    prt  = f"{wlog_prefix(worker_name, worker_address, pid)} "
    prt += f"native_tid={native_tid:7d} "
    prt += f"py_tid={py_tid} "
    if pinned_cpu is None:
        prt += "pinned_cpu=None "
    else:
        prt += f"pinned_cpu={pinned_cpu:2d} "
    prt += f"cpu_run_before={cpu_run_before:2d} "
    prt += f"cpu_run_after={cpu_run_after:2d} "
    prt += f"affinity={format_cpu_list(affinity_list)} "
    prt += f"strategy_id={id(strategy)} "
    prt += f"[n_bsl={n_bsl} n_time={n_time} n_chan={n_chan} n_pol={n_pol}] "
    prt += f"proc={t4-t3:.3f}s "
    prt += f"total={t5-t0:.3f}s"

    return start, flags_slice, prt


def chunk_indices_for_slice(start, stop, chunk_size):
    """Return the range of chunk indices (along one axis, chunk width chunk_size) touched by [start, stop)."""
    first = start // chunk_size
    last = (stop - 1) // chunk_size
    return range(first, last + 1)


def chunk_coords_touched(shape, chunks, loc):
    """Return the list of chunk-coordinate tuples touched by loc (a tuple of ints/slices) on an array of the given shape/chunks."""
    coords_per_axis = []

    for axis, (size, chunk, idx) in enumerate(zip(shape, chunks, loc)):
        if isinstance(idx, int):
            start = idx
            stop = idx + 1
        else:
            start, stop, step = idx.indices(size)
            if step != 1:
                raise ValueError("Only step=1 slices supported")

        coords_per_axis.append(chunk_indices_for_slice(start, stop, chunk))

    return list(itertools.product(*coords_per_axis))


def human_bytes(n):
    """Format a byte count n as a human-readable string, e.g. 1536 -> "1.50 KiB"."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    x = float(n)

    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0


def normalize_loc(loc, ndim):
    """Expand loc (an int/slice/tuple, possibly with an Ellipsis or fewer axes than ndim) into a full ndim-length tuple."""
    if not isinstance(loc, tuple):
        loc = (loc,)

    # Expand ellipsis
    if Ellipsis in loc:
        ellipsis_index = loc.index(Ellipsis)
        n_missing = ndim - (len(loc) - 1)
        loc = (
            loc[:ellipsis_index]
            + tuple(slice(None) for _ in range(n_missing))
            + loc[ellipsis_index + 1 :]
        )

    # Fill missing axes
    if len(loc) < ndim:
        loc = loc + tuple(slice(None) for _ in range(ndim - len(loc)))

    if len(loc) != ndim:
        raise ValueError(f"loc has {len(loc)} axes, array has ndim={ndim}")

    return loc


def slice_from_indexer(indexer, size):
    """
    Normalize int/slice/None to a concrete slice.

    This helper assumes step == 1.
    """
    if indexer is None:
        return slice(0, size, 1)

    if isinstance(indexer, int):
        if indexer < 0:
            indexer += size
        return slice(indexer, indexer + 1, 1)

    if isinstance(indexer, slice):
        start, stop, step = indexer.indices(size)

        if step != 1:
            raise NotImplementedError(
                f"Only step=1 slices are supported here, got {indexer}"
            )

        return slice(start, stop, 1)

    raise TypeError(f"Unsupported indexer type: {type(indexer)}")


def axis_chunk_sizes(size, chunk):
    """
    Return actual chunk sizes along one axis, including smaller edge chunk.
    """
    n_full = size // chunk
    rem = size % chunk

    chunks = [chunk] * n_full

    if rem:
        chunks.append(rem)

    return tuple(chunks)


def chunks_touched_1d(axis_chunks, slc):
    """
    Return chunk indices touched by a 1D slice.
    """
    touched = []
    pos = 0

    for ci, csize in enumerate(axis_chunks):
        c0 = pos
        c1 = pos + csize
        pos = c1

        if slc.start < c1 and slc.stop > c0:
            touched.append(ci)

    return touched


def count_zarr_chunks_touched(arr, loc):
    """
    Count logical Zarr chunks touched by arr[loc].

    Parameters
    ----------
    arr:
        zarr Array
    loc:
        tuple of slices/ints, same ndim as arr

    Returns
    -------
    dict with chunk information.
    """
    loc = normalize_loc(loc, arr.ndim)

    axis_chunk_lists = [
        axis_chunk_sizes(size, chunk)
        for size, chunk in zip(arr.shape, arr.chunks)
    ]

    concrete_slices = []
    touched_per_axis = []
    selected_shape = []

    for axis_chunks, indexer, size in zip(axis_chunk_lists, loc, arr.shape):
        slc = slice_from_indexer(indexer, size)
        concrete_slices.append(slc)
        selected_shape.append(max(0, slc.stop - slc.start))
        touched_per_axis.append(chunks_touched_1d(axis_chunks, slc))

    n_chunks = math.prod(len(x) for x in touched_per_axis)

    return {
        "shape": arr.shape,
        "chunks": arr.chunks,
        "dtype": arr.dtype,
        "loc": loc,
        "concrete_slices": tuple(concrete_slices),
        "selected_shape": tuple(selected_shape),
        "axis_chunk_lists": axis_chunk_lists,
        "touched_per_axis": touched_per_axis,
        "n_chunks": n_chunks,
    }


def touched_chunk_keys(info, separator="."):
    """
    Generate Zarr v2 chunk keys like 0.0.0.0.
    """
    for coord in itertools.product(*info["touched_per_axis"]):
        yield separator.join(map(str, coord))


def count_existing_chunk_files_with_size(zarr_path, xds_name, var_name, info):

    """
    Count existing chunk files and their compressed size on disk.

    Assumes Zarr v2 directory layout:
        zarr_path / xds_name / var_name / chunk_key

    st_size: physical byte size of that specific compressed chunk file in storage
    """

    arr_dir = Path(zarr_path) / xds_name / var_name

    existing = []
    missing = []
    total_bytes_on_disk = 0

    for key in touched_chunk_keys(info):
        path = arr_dir / key

        if path.exists():
            size = path.stat().st_size
            existing.append((key, path, size))
            total_bytes_on_disk += size
            #print("key, path, size =", key, path, size," => total_bytes_on_disk =", total_bytes_on_disk)
        else:
            print("-E- Missing chunk with key =", key)
            sys.exit(1)
            missing.append(key)

    return existing, missing, total_bytes_on_disk


def zarr3_chunk_path(arr_dir: Path, key: str) -> Path:
    """
    Convert a logical chunk key/index into an on-disk Zarr v3 chunk path.

    Accepts keys like:
        "0.1.2"
        "0/1/2"
        "c/0/1/2"
        "c.0.1.2"

    Handles common Zarr v3 chunk_key_encoding cases:
        default + "/"  -> c/0/1/2
        default + "."  -> c.0.1.2
        v2             -> 0.1.2
    """

    meta_path = arr_dir / "zarr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Zarr v3 metadata: {meta_path}")

    with meta_path.open("r") as f:
        meta = json.load(f)

    if meta.get("zarr_format") != 3:
        raise ValueError(f"Expected Zarr v3 array metadata in {meta_path}, got {meta.get('zarr_format')!r}")

    enc = meta.get("chunk_key_encoding", {})
    enc_name = enc.get("name", "default")

    # Zarr metadata spelling differs slightly across examples/versions.
    cfg = enc.get("configuration", enc.get("config", {}))
    sep = enc.get("separator", cfg.get("separator", "/"))

    # Normalize incoming key to chunk coordinate parts.
    k = str(key).strip()

    if k.startswith("c/"):
        parts = k[2:].split("/")
    elif k.startswith("c."):
        parts = k[2:].split(".")
    elif "/" in k:
        parts = k.split("/")
    else:
        parts = k.split(".")

    parts = [p for p in parts if p != ""]

    if enc_name == "default":
        if sep == "/":
            return arr_dir / "c" / Path(*parts)
        elif sep == ".":
            return arr_dir / ("c." + ".".join(parts))
        else:
            raise NotImplementedError(f"Unsupported Zarr v3 default chunk separator: {sep!r}")

    elif enc_name == "v2":
        # v3 can use a v2-compatible key encoding: no c/ prefix.
        if sep == "/":
            return arr_dir / Path(*parts)
        else:
            return arr_dir / sep.join(parts)

    else:
        raise NotImplementedError(f"Unsupported Zarr v3 chunk_key_encoding: {enc!r}")


def count_existing_chunk_files_with_size_zarr3(zarr_path, xds_name, var_name, info):
    """
    Count existing Zarr v3 chunk files and their compressed size on disk.

    Zarr v3 layout is normally:
        zarr_path / xds_name / var_name / zarr.json
        zarr_path / xds_name / var_name / c / chunk_i / chunk_j / ...

    But the exact chunk key is controlled by chunk_key_encoding in zarr.json.

    st_size is the physical byte size of that specific stored chunk object/file.
    """

    arr_dir = Path(zarr_path) / xds_name / var_name

    existing = []
    missing = []
    total_bytes_on_disk = 0

    print("------------------------------------------")
    print("info =", info)
    print("------------------------------------------")
    print("touched_chunk_keys(info) =", touched_chunk_keys(info))
    print("------------------------------------------")
    print("arr_dir =", arr_dir)
    print("------------------------------------------")

    meta_path = arr_dir / "zarr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Zarr v3 array metadata: {meta_path}")

    for key in touched_chunk_keys(info):
        path = zarr3_chunk_path(arr_dir, key)

        if path.exists():
            size = path.stat().st_size
            #print("key, path, size =", key, path, size)
            existing.append((key, path, size))
            total_bytes_on_disk += size
        else:
            print("-E- Missing chunk with key =", key)
            print("-E- Expected Zarr v3 path =", path)
            missing.append(key)

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} chunk files under {arr_dir}; "
            f"first missing key={missing[0]!r}"
        )

    return existing, missing, total_bytes_on_disk


def estimate_touched_uncompressed_nbytes(arr, info):
    """
    Full decompressed payload size of all touched chunks.
    """
    total = 0

    for coord in itertools.product(*info["touched_per_axis"]):
        n_elem = 1

        for axis, chunk_index in enumerate(coord):
            #n_ = n_elem
            n_elem *= info["axis_chunk_lists"][axis][chunk_index]
            #print(axis, chunk_index, n_ , "*", info["axis_chunk_lists"][axis][chunk_index], "=", n_elem)

        total += n_elem * arr.dtype.itemsize
        #print("total touched_uncompressed_nbytes =", total)


    return int(total)


def estimate_selected_bytes(arr, info):
    """
    Size of the final selected output array.
    """
    n_elem = 1

    for s in info["selected_shape"]:
        n_elem *= s

    return int(n_elem * arr.dtype.itemsize)


def read_zarr_with_chunk_count(root, zarr_path, zarr_version: int, xds_name: str, var_name: str, loc,
                               out_dir: str, out_tag: str, n_threads: int = 1):
    """
    Read root[xds_name][var_name][loc] from Zarr, reporting how many on-disk
    chunks were touched and at what compressed/uncompressed size.

    Logs a summary and appends a "zarr_read" record via append_worker_stat().

    n_threads sets Dask's threaded-scheduler num_workers for the compute()
    call that performs the actual read.

    Returns (data, info), where info is the dict built by
    count_zarr_chunks_touched(), augmented with size/byte-count fields.
    """

    pid = os.getpid()
    w = get_worker()
    w_name = w.name
    w_address = w.address

    arr = root[xds_name][var_name]

    info = count_zarr_chunks_touched(arr, loc)

    if zarr_version == 3:
        print("Zarr v3")
        existing, missing, touched_compressed_bytes_on_disk = count_existing_chunk_files_with_size_zarr3(
            zarr_path=zarr_path,
            xds_name=xds_name,
            var_name=var_name,
            info=info)
    elif zarr_version == 2:
        print("Zarr v2")
        existing, missing, touched_compressed_bytes_on_disk = count_existing_chunk_files_with_size(
            zarr_path=zarr_path,
            xds_name=xds_name,
            var_name=var_name,
            info=info)
    else:
        print("Unknown / not a Zarr store")
        sys.exit(1)


    touched_uncompressed_bytes = estimate_touched_uncompressed_nbytes(arr, info)

    selected_bytes = estimate_selected_bytes(arr, info)

    shape = tuple(s.stop - s.start for s in info["concrete_slices"])

    selected_elements = math.prod(shape)

    dask_print(f"{wlog_prefix(w_name, w_address, pid)} {var_name} selection info:\n"
               ">>>>\n",
               f"{info}\n",
               "<<<<\n")

    data = None

    # Parallel read + I/O process monitoring
    # --------------------------------------
    darr = da.from_zarr(arr)
    ts = time.perf_counter()
    io_before = process_io()
    with dask.config.set(scheduler="threads", num_workers=n_threads):
        data = darr[loc].compute()
    io_after = process_io()
    te = time.perf_counter()

    assert data.flags["C_CONTIGUOUS"], f"{var_name} is not contiguous!"

    load_time = te - ts

    read_bytes = io_after["read_bytes"] - io_before["read_bytes"]
    rchar      = io_after["rchar"]      - io_before["rchar"]

    selected_size = selected_elements * data.dtype.itemsize


    # Augment info
    info["touched_compressed_bytes_on_disk"] = touched_compressed_bytes_on_disk
    info["touched_uncompressed_bytes"]       = touched_uncompressed_bytes
    info["selected_bytes"]                   = selected_bytes
    info["selected_elements"]                = selected_elements
    info["selected_size_B"]                  = selected_size


    msg = pformat(info)
    msg += "\n"
    msg += f"-I- reading {xds_name}/{var_name}\n"
    msg += f"-I- shape                           : {arr.shape}\n"
    msg += f"-I- chunks size                     : {arr.chunks}\n"
    msg += f"-I- dtype                           : {arr.dtype}\n"
    msg += f"-I- loc                             : {loc}\n"
    msg += f"-I- concrete_slices                 : {info['concrete_slices']}\n"
    msg += f"-I- selected_shape                  : {info['selected_shape']}\n"
    msg += f"-I- touched_per_axis                : {info['touched_per_axis']}\n"
    msg += f"-I- logical chunks touched          : {info['n_chunks']}\n"
    msg += f"-I- existing chunk files            : {len(existing)}\n"
    msg += f"-I- touched compressed bytes on disk: {touched_compressed_bytes_on_disk} ({human_bytes(touched_compressed_bytes_on_disk)})\n"
    msg += f"-I- touched uncompressed bytes      : {touched_uncompressed_bytes} ({human_bytes(touched_uncompressed_bytes)})\n"
    msg += f"-I- selected output bytes           : {selected_bytes} ({human_bytes(selected_bytes)})"
    msg += "\n"
    msg += f"-I- load time      (read+uncomp)    : {load_time:.3f} s\n"
    msg += f"-I- load bandwidth (read+uncomp)    : {(float(touched_uncompressed_bytes) / 10**9)     / load_time:.3f} GiB/s == " \
        f"{(float(touched_uncompressed_bytes) / (1024)**3) / load_time:.3f} GB/s\n"
    msg += f"-I- loaded data shape               : {data.shape}\n"
    msg += f"-I- loaded data dtype               : {data.dtype}\n"
    msg += f"-I- loaded data bytes               : {data.nbytes} ({human_bytes(data.nbytes)})\n"
    msg += f"-I- selected_elements               : {selected_elements}\n"
    msg += f"-I- selected_size                   : {selected_size} B"
    #msg += f"-I- needed to read              : {info['n_chunks'] * np.prod(arr.chunks) * data.dtype.itemsize} B\n"

    print(msg)

    dask_print(msg)

    append_worker_stat({
        "event": "zarr_read",
        "xds_name": xds_name,
        "loc": str(loc),
        var_name: {
            "array_shape": arr.shape,
            "array_chunks": arr.chunks,
            "dtype": str(data.dtype),
            "dtype_bytes": data.dtype.itemsize,
            "loc": json_safe(loc),
            "concrete_slices": json_safe(info["concrete_slices"]),
            "selected_shape": json_safe(info["selected_shape"]),
            "touched_per_axis": json_safe(info["touched_per_axis"]),
            "n_chunks": info["n_chunks"],
            "selected_elements": selected_elements,
            "load_time_s": f"{load_time:.3f}",
            "read_bytes_B": read_bytes,
            "rchar_B": rchar,
            "touched_compressed_bytes_on_disk": touched_compressed_bytes_on_disk,
            "touched_uncompressed_bytes": touched_uncompressed_bytes,
            "rchar_minus_touched_compressed_bytes_on_disk": rchar - touched_compressed_bytes_on_disk,
            "load_bandwith_GBs":  f"{(float(touched_uncompressed_bytes) / (1024)**3) / load_time:.3f}",
            "load_bandwith_GiBs": f"{(float(touched_uncompressed_bytes) / 10**9)     / load_time:.3f}",
            "selected_bytes": selected_bytes,
            "over_load": f"{float(touched_uncompressed_bytes) / float(selected_bytes):.3f}"
        },
    },
    out_dir=out_dir,
    out_tag=out_tag)

    return data, info


def make_thread_pool(n_threads, pin_threads=False):
    """
    Return a ThreadPoolExecutor.

    If pin_threads=True, each pool thread is pinned to one CPU from the
    current process affinity mask.
    """
    n_threads = min(n_threads, len(os.sched_getaffinity(0)))

    if not pin_threads:
        return ThreadPoolExecutor(max_workers=n_threads), {}

    allowed_cpus = sorted(os.sched_getaffinity(0))[:n_threads]

    cpu_queue = queue.Queue()
    for cpu in allowed_cpus:
        cpu_queue.put(cpu)

    pinned_cpu_by_tid = {}

    def pin_thread():
        """ThreadPoolExecutor initializer: pin the calling pool thread to the next free CPU from cpu_queue."""
        cpu = cpu_queue.get()

        # Applies to the current OS thread on Linux
        os.sched_setaffinity(0, {cpu})

        tid = threading.get_native_id()
        pinned_cpu_by_tid[tid] = cpu

        dask_print(
            f"-I- pinned thread "
            f"pid={os.getpid():7d} "
            f"native_tid={tid} "
            f"cpu={cpu} "
            f"affinity={sorted(os.sched_getaffinity(0))}",
            flush=True,
        )

    pool = ThreadPoolExecutor(
        max_workers=n_threads,
        initializer=pin_thread,
    )

    return pool, pinned_cpu_by_tid


def shared_baseline_chunk_ids(start: int, stop: int, chunk_size: int, nb: int) -> set[int]:
    """
    Baseline chunk IDs that may also be touched by an adjacent partition.

    The worker owns baseline indices [start, stop).
    """
    if stop <= start:
        return set()

    shared = set()

    # Partition starts in the middle of a disk chunk:
    # previous worker may touch this same chunk.
    if start % chunk_size != 0:
        shared.add(start // chunk_size)

    # Partition ends in the middle of a disk chunk:
    # next worker may touch this same chunk.
    if stop % chunk_size != 0 and stop != nb:
        shared.add((stop - 1) // chunk_size)

    return shared


def process_io():
    """Return this process's /proc/<pid>/io counters (rchar, wchar, read_bytes, write_bytes, ...) as a dict."""
    io_file = Path(f"/proc/{os.getpid()}/io")
    values = {}
    for line in io_file.read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip())
    return values


def flag_me(dummy_bvis, *,
            msv4_path: str, xds_name: str, pin_cpu_threads: bool, compute_mode: str,
            block_info=None, block_id=None, strategy_path: str = None,
            kernel_size: int, subgrid_size: int, vis_type: str, cell_size: float,
            zarr_version: int, out_dir: str, out_tag: str,
            n_threads: int = None,
            timeslot_size: int, grid_size: int,
            nb: int):
    """
    da.map_blocks() kernel: process one chunk of the MSv4 dataset end to end.

    For the chunk identified by block_info (dummy_bvis is an untouched
    placeholder array — the real VISIBILITY/FLAG/UVW/etc. data is read
    straight from msv4_path here, not passed in through the dask graph):

      1. Reads VISIBILITY, FLAG, UVW, frequency, time and baseline/antenna
         metadata for this chunk from the on-disk Zarr store.
      2. Runs AOFlagger on each baseline in parallel (ThreadPoolExecutor,
         see _process_baseline_slice()), producing updated flags.
      3. Writes the updated flags back into FLAG2 at msv4_path, taking a
         MultiLock only on the baseline chunks shared with a neighboring
         partition.
      4. Runs IDG gridding + FFT on the flagged visibilities.

    Returns the real part of the gridded image for this chunk, shaped
    (1, 1, 1, grid_size, grid_size) to match the output chunking declared
    by the map_blocks() call in run_pipeline().
    """
    ts_func = time.perf_counter()

    pid = os.getpid()

    w = get_worker()
    w_name = w.name
    w_address = w.address

    dask_print(f"{wlog_prefix(w_name, w_address, pid)} block_info[0][\"chunk-location\"]    (input)  =", block_info[0]["chunk-location"], block_info[0]["array-location"])
    dask_print(f"{wlog_prefix(w_name, w_address, pid)} block_info[None][\"chunk-location\"] (output) =", block_info[None]["chunk-location"], block_info[None]["array-location"])

    root = zarr.open(msv4_path, mode="a")

    # Slice location as slices
    loc = blockinfo_to_slice_tuple(block_info[0]["array-location"])

    ts_load = time.perf_counter()

    CPUS_PER_WORKER = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    if n_threads is None:
        n_threads = CPUS_PER_WORKER

    dask_print(f"{wlog_prefix(w_name, w_address, pid)} START loading data on {CPUS_PER_WORKER} cpus")

    bvis, bvis_info = read_zarr_with_chunk_count(
        root=root,
        zarr_path=msv4_path,
        zarr_version=zarr_version,
        xds_name=xds_name,
        var_name=vis_type,
        loc=loc,
        out_dir=out_dir,
        out_tag=out_tag,
        n_threads=n_threads)

    bflag, bflag_info = read_zarr_with_chunk_count(
        root=root,
        zarr_path=msv4_path,
        zarr_version=zarr_version,
        xds_name=xds_name,
        var_name="FLAG",
        loc=loc,
        out_dir=out_dir,
        out_tag=out_tag,
        n_threads=n_threads)

    dask_print("bflag_info =\n", bflag_info)

    ts_load_buvw = time.perf_counter()
    buvw = root[xds_name]["UVW"][loc[0], loc[1], 0:3].astype(np.float32)
    te_load_buvw = time.perf_counter()
    t_load_buvw = te_load_buvw - ts_load_buvw
    assert buvw.flags["C_CONTIGUOUS"], "buvw is not contiguous!"
    print(f"-T- buvw: {type(buvw)} in {t_load_buvw:.3f} s")

    bfreq = root[xds_name]["frequency"][loc[2]].astype(np.float32)
    assert bfreq.flags["C_CONTIGUOUS"], "buvw is not contiguous!"

    btime = root[xds_name]["time"][loc[0]]
    assert btime.flags["C_CONTIGUOUS"], "btime is not contiguous!"

    # Baselines' antennae 1 & 2 (ban1, ban2)
    if zarr_version == 2:
        ban1 = root[xds_name].baseline_antenna1_name[loc[1]]
        ban2 = root[xds_name].baseline_antenna2_name[loc[1]]
        ant_names = root[xds_name]["antenna_xds"].antenna_name
    elif zarr_version == 3:
        ban1 = root[xds_name]["baseline_antenna1_name"][loc[1]]
        ban2 = root[xds_name]["baseline_antenna2_name"][loc[1]]
        ant_names = root[xds_name]["antenna_xds"]["antenna_name"]

    idx_ban1 = pd.Index(ant_names).get_indexer(ban1)
    idx_ban2 = pd.Index(ant_names).get_indexer(ban2)

    baselines = np.column_stack((idx_ban1,idx_ban2)).astype(np.intc)

    te_load = time.perf_counter()

    msg  = f"{wlog_prefix(w_name, w_address, pid, 'T')} END loading data chunk {block_info[0]['chunk-location']} on {w.address} on {CPUS_PER_WORKER} CPU cores to produce {block_info[None]['chunk-location']}\n"
    msg += f"{wlog_prefix(w_name, w_address, pid, 'T')} All data loaded in {te_load - ts_load:.3f} s on {CPUS_PER_WORKER} CPU cores, {w.address}."

    dask_print(msg)

    bntimes, bnchans = bvis.shape[0], bvis.shape[2]
    assert bntimes == len(btime)
    assert bnchans == len(bfreq)

    buf = np.ones(bvis.shape, dtype=bool)
    buf[:] = False

    # Transpose to match IDG expectations
    bvis  = bvis.transpose(1, 0, 2, 3)
    bflag = bflag.transpose(1, 0, 2, 3)
    buvw  = buvw.transpose(1, 0, 2)

    # Transpose does break C contiguity, fix that
    ts_contiguity = time.perf_counter()
    if not bvis.flags["C_CONTIGUOUS"]:
        #print("-I- bvis was not C_CONTIGUOUS!")
        bvis = np.ascontiguousarray(bvis)
    if not bflag.flags["C_CONTIGUOUS"]:
        #print("-I- bflag was not C_CONTIGUOUS!")
        bflag = np.ascontiguousarray(bflag)
    if not buvw.flags["C_CONTIGUOUS"]:
        #print("-I- buvw was not C_CONTIGUOUS!")
        buvw = np.ascontiguousarray(buvw)
    te_contiguity = time.perf_counter()
    dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} bvis, bflag and buvw made contiguous in {te_contiguity - ts_contiguity:.3f} sec")

    n_threads = min(int(n_threads), bvis.shape[0])
    dask_print(f"{wlog_prefix(w_name, w_address, pid)} n_threads = {n_threads} as min({int(n_threads)}, {bvis.shape[0]})")

    # Split baseline axis into nearly equal chunks
    boundaries = np.linspace(0, bvis.shape[0], n_threads + 1, dtype=int)

    work_items = []
    for tid in range(n_threads):
        start = boundaries[tid]
        stop = boundaries[tid + 1]

        if start == stop:
            continue

        vis_slice  = bvis[start:stop, :, :, :]

        flag_slice = np.any(bflag[start:stop, :, :, :], axis=3)

        work_items.append((start, vis_slice, flag_slice, strategy_path))

    print(f"{wlog_prefix(w_name, w_address, pid)} work_items has {len(work_items)} items")

    # Flags has no pol dimension
    flags_block = np.empty((bvis.shape[0], bvis.shape[2], bvis.shape[1]), dtype=np.bool_)


    tp_s = time.perf_counter()

    pool, pinned_cpu_by_tid = make_thread_pool(
        n_threads=n_threads,
        pin_threads=pin_cpu_threads,
    )
    tp_i = time.perf_counter()

    msg = ""
    with pool:
        futures = [
            pool.submit(_process_baseline_slice,
                        *item,
                        worker_name=w_name,
                        worker_address=w_address,
                        pin_cpu_threads=pin_cpu_threads,
                        pinned_cpu_by_tid=pinned_cpu_by_tid)
            for item in work_items
        ]

        for fut in futures:
            start, flags_slice, prt = fut.result()
            msg += prt + "\n"
            stop = start + flags_slice.shape[0]
            flags_block[start:stop, :, :] = flags_slice

    tp_e = time.perf_counter()
    tp = tp_e - tp_s

    dask_print(msg)

    dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} ThreadPoolExecutor (aoflagger) on {n_threads} threads took {tp:.3f} s; make_thread_pool in {tp_i - tp_s:.3f} s")


    append_worker_stat({
        "event": "aoflagger",
        "xds_name": xds_name,
        "loc": str(loc),
        "aoflagger": {
            "time_s": f"{tp:.3f}",
        },
    },
    out_dir=out_dir,
    out_tag=out_tag)


    # Store new flags in FLAG2 (need to acquire locks on chunks when
    #  chunk processed by multiple workers)
    # --------------------------------------------------------------
    flags4 = flags_block.transpose(2, 0, 1)
    flags4 = np.broadcast_to(flags4[...,None], bflag.transpose(1, 0, 2, 3).shape)

    # Theoretical size
    flags4_elements = int(np.prod(flags4.shape))
    flags4_bytes = flags4_elements * flags4.dtype.itemsize
    assert flags4_elements == bflag_info["selected_elements"], f"flags4_elements {flags4_elements} not equal to expected {bflag_info['selected_elements']}"
    assert flags4_bytes == bflag_info["selected_bytes"], f"flags4_elements {flags4_bytes} not equal to expected {bflag_info['selected_bytes']}"

    dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} bflag touched compressed (?) bytes on disk {bflag_info['touched_compressed_bytes_on_disk']} B, touched uncompressed bytes on disk {bflag_info['touched_uncompressed_bytes']} B, theoretical size {flags4_bytes}, n elements {flags4_elements} of type {flags4.dtype} {flags4.dtype.itemsize} B")

    arr = root[xds_name]["FLAG2"]
    coords = chunk_coords_touched(arr.shape, arr.chunks, loc)

    baseline_axis = 1

    baseline_start = loc[baseline_axis].start
    baseline_stop = loc[baseline_axis].stop
    baseline_chunk_size = arr.chunks[baseline_axis]

    shared_bl_chunks = shared_baseline_chunk_ids(
        baseline_start,
        baseline_stop,
        baseline_chunk_size,
        nb
    )
    dask_print(f"{wlog_prefix(w_name, w_address, pid)} shared_bl_chunks = {shared_bl_chunks}")

    coords_to_lock = sorted(
        c for c in coords
        if c[baseline_axis] in shared_bl_chunks
    )

    lock_names = sorted({
        f"zarr-write:{msv4_path}:{xds_name}:FLAG2:{'.'.join(map(str, coord))}"
        for coord in coords_to_lock
    })

    lock = MultiLock(lock_names) if lock_names else None

    store_s = time.perf_counter()
    t_lock = 0
    t_rel = 0
    try:
        if lock is not None:
            ts_lock_acq = time.perf_counter()
            lock.acquire()
            te_lock_acq = time.perf_counter()
            t_lock = te_lock_acq - ts_lock_acq
            dask_print(f"{wlog_prefix(w_name, w_address, pid)} MultiLock acquired {len(lock_names)} locks in "
                       f"{t_lock:.3f} s")
        else:
            dask_print(f"{wlog_prefix(w_name, w_address, pid)} no shared chunks => no lock required")


        io_before = process_io()

        s_write = time.perf_counter()
        arr[loc] = flags4
        e_write = time.perf_counter()
        write_time = e_write - s_write

        io_after = process_io()

        write_bytes = io_after["write_bytes"] - io_before["write_bytes"]
        wchars = io_after["wchar"] - io_before["wchar"]

    finally:
        if lock is not None:
            ts_lock_rel = time.perf_counter()
            lock.release()
            te_lock_rel = time.perf_counter()
            t_rel = te_lock_rel - ts_lock_rel
            dask_print(f"{wlog_prefix(w_name, w_address, pid)} released {len(lock_names)} locks in "
                       f"{t_rel:.3f} s")
    store_e = time.perf_counter()
    store_time = store_e - store_s

    flag2_write_bandwidth_GBs = float(bflag_info["touched_uncompressed_bytes"]) / (1024)**3 / write_time
    flag2_write_bandwith_GiBs = float(bflag_info["touched_uncompressed_bytes"]) / 10**9     / write_time
    flag2_store_bandwidth_GBs = float(bflag_info["touched_uncompressed_bytes"]) / (1024)**3 / store_time
    flag2_store_bandwith_GiBs = float(bflag_info["touched_uncompressed_bytes"]) / 10**9     / store_time

    dask_print(f"{wlog_prefix(w_name, w_address, pid)} FLAG2 written in {write_time:.3f} s, stored in {store_time} s, write_bytes {write_bytes} B, wchar {wchars} B, touched_uncompressed_bytes {bflag_info['touched_uncompressed_bytes']}, flag2_write_bandwidth_GBs {flag2_write_bandwidth_GBs:.3f} GB/s, flag2_write_bandwith_GiBs {flag2_write_bandwith_GiBs:.3f}, flag2_store_bandwidth_GBs {flag2_store_bandwidth_GBs:.3f} GB/s, flag2_store_bandwith_GiBs {flag2_store_bandwith_GiBs:.3f}")

    append_worker_stat({
        "event": "zarr_store",
        "xds_name": xds_name,
        "loc": str(loc),
        "flag2": {
            "array_shape": arr.shape,
            "array_chunks": arr.chunks,
            "dtype": str(flags4.dtype),
            "dtype_bytes": flags4.dtype.itemsize,
            "loc": json_safe(loc),
            "concrete_slices": json_safe(bflag_info["concrete_slices"]),
            "selected_shape": json_safe(bflag_info["selected_shape"]),
            "touched_per_axis": json_safe(bflag_info["touched_per_axis"]),
            "n_chunks": bflag_info["n_chunks"],
            "lock_time_s": f"{t_lock:.3f}",
            "release_time_s": f"{t_rel:.3f}",
            "store_time_s": f"{store_time:.3f}",
            "write_time_s": f"{write_time:.3f}",
            "write_bytes": write_bytes,
            "wchar": wchars,
            "selected_elements": bflag_info["selected_elements"],
            "selected_bytes": bflag_info["selected_bytes"],
            "touched_compressed_bytes_on_disk": bflag_info["touched_compressed_bytes_on_disk"],
            "touched_uncompressed_bytes":       bflag_info["touched_uncompressed_bytes"],
            "selected_minus_touched_uncompressed_bytes": bflag_info["selected_bytes"] - bflag_info["touched_uncompressed_bytes"],
            "flag2_write_bandwidth_GBs": f"{flag2_write_bandwidth_GBs:.3f}",
            "flag2_write_bandwith_GiBs": f"{flag2_write_bandwith_GiBs:.3f}",
            "flag2_store_bandwidth_GBs": f"{flag2_store_bandwidth_GBs:.3f}",
            "flag2_store_bandwith_GiBs": f"{flag2_store_bandwith_GiBs:.3f}",
            "over_write": f"{float(bflag_info['touched_uncompressed_bytes']) / float(bflag_info['selected_bytes']):.3f}"
        },
    },
    out_dir=out_dir,
    out_tag=out_tag)

    #dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} bflag size on disk (touched chunks, comp?) {bflag_tcbod} B, estimated decompressed size {bflag_tubod} B, theoretical size {flags4_theoretical_bytes}, n elements {flags4_elements} of type {flags4.dtype} {flags4.dtype.itemsize} B")


    # Look at baseline statistics as per in AOFlagger, should match command line results
    # ----------------------------------------------------------------------------------
    msg = ""
    for i in range(min(10, bflag.shape[0])):
        noldc = sum(sum(np.any(np.logical_not(bflag[i,]), axis=2)))
        noldf = sum(sum(np.any(bflag[i,], axis=2)))
        nnewc = sum(sum(np.logical_not(flags_block[i,])))
        nnewf = sum(sum(flags_block[i,]))
        new = nnewf - noldf

        msg += f"{wlog_prefix(w_name, w_address, pid)} bsl {i} - OLD: count {noldc:8d} rfi {noldf:6d}, NEW: delta {new:6d} count {nnewc:8d} rfi {nnewf:6d}, TOTAL: {nnewc+nnewf:6d}\n"
    dask_print(msg)


    #return np.ones((1, 1, 1, grid_size, grid_size), dtype=np.float32)

    # Apply updated flags onto visibilities
    # -------------------------------------
    flags4 = flags4.transpose(1, 0, 2, 3)
    dask_print(f"{wlog_prefix(w_name, w_address, pid)} {bvis.shape} vs {flags4.shape}")

    bvis = bvis * np.logical_not(flags4)
    bvis[np.isnan(bvis)] = 0
    #bvis = bvis * np.logical_not(bflag)


    # ---- IDG ---- #

    # From IDG example =================================================
    #flags_block     = table.getcol('FLAG',
    #                               startrow = nr_rows_read,
    #                               nrow = nr_rows_per_batch)
    #vis_block = vis_block * ~flags_block
    #vis_block[np.isnan(vis_block)] = 0
    #===================================================================

    safe_name = (w.address
                 .replace("://", "_")
                 .replace("/", "_")
                 .replace(":", "_"))

    idg_worker_log = os.path.join(out_dir, f"{out_tag}_{safe_name}_pid{os.getpid()}_idg.log")

    tidg_s = time.perf_counter()

    with capture_native_output(idg_worker_log):

        dask_print(f"{wlog_prefix(w_name, w_address, pid)} IDG kernel_size = {kernel_size}, subgrid_size = {subgrid_size}")

        ts_proxy = time.perf_counter()
        proxy = _get_idg_proxy(compute_mode)
        te_proxy = time.perf_counter()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} IDG _get_idg_proxy() {(te_proxy - ts_proxy):.3f} s")

        ###########
        # Init data
        ###########

        bntimeslots = int((bntimes + timeslot_size - 1) / timeslot_size)
        bnstations = ant_names.shape[0]
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'D')} bntimes = {bntimes}, timeslot_size = {timeslot_size} => bntimeslots = {bntimeslots}, bnstations = {bnstations}")

        ts_aterms = time.perf_counter()
        aterms = idg.util.get_identity_aterms(bntimeslots, bnstations, subgrid_size, 4, info=True);
        te_aterms = time.perf_counter()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} {block_info[0]['chunk-location']} IDG get_identity_aterms() {(te_aterms - ts_aterms):.3f} s")
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'D')} aterms =", aterms.shape, aterms.dtype)

        aterm_offsets = idg.util.get_example_aterm_offsets(bntimeslots, bntimes, info=True)
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'D')} aterm_offsets =", aterm_offsets.shape, aterm_offsets.dtype, aterm_offsets)

        taper = idg.util.get_identity_taper(subgrid_size)
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'D')} taper =", taper.shape, taper.dtype)

        w_step = 0.0

        shift = np.zeros(2, np.float32)
        dask_print(f"{wlog_prefix(w_name, w_address, pid)} shift =", shift)


        ##################################
        # .set_grid() THEN .init_cache()
        ##################################


        # Init grid always because cached
        time_get_grid = -time.time()
        grid = _get_grid(grid_size)
        grid[:] = 0
        time_get_grid += time.time()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'D')} {block_info[0]['chunk-location']} IDG get_grid took {time_get_grid:.3f} s, grid {grid.shape}, {grid.dtype}")

        time_set_grid  = -time.time()
        proxy.set_grid(grid)
        time_set_grid += time.time()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'D')} {block_info[0]['chunk-location']} IDG set_grid took {time_set_grid:.3f} s")

        # Then init cache
        time_cache = -time.time()
        _idg_init_proxy_cache(proxy, subgrid_size, cell_size, w_step, shift)
        time_cache += time.time()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} {block_info[0]['chunk-location']} _idg_init_proxy_cache took {time_cache:.3f} s")

        time_gridding = -time.time()
        proxy.gridding(kernel_size, bfreq, bvis, buvw, baselines, aterms, aterm_offsets, taper)
        # TODO: This one is needed, check why. Bug?
        proxy.get_final_grid()
        time_gridding += time.time()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} {block_info[0]['chunk-location']} gridding took {time_gridding:.3f} s")

        time_fft = -time.time()
        proxy.transform(idg.FourierDomainToImageDomain)
        time_fft += time.time()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} {block_info[0]['chunk-location']} transform took {time_fft:.3f} s")

        time_r0 = -time.time()
        img_real = np.real(grid[0,:,:])
        time_r0 += time.time()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} {block_info[0]['chunk-location']} real took {time_r0:.3f} sec")

        time_r1 = -time.time()
        img_real = img_real[None,None,None,:,:]
        time_r1 += time.time()
        dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} {block_info[0]['chunk-location']} img dim expansion {time_r1:.3f} sec")

    tidg_e = time.perf_counter()

    tidg = tidg_e - tidg_s
    dask_print(f"{wlog_prefix(w_name, w_address, pid)} {block_info[0]['chunk-location']} whole idg in {tidg:.3f} s")

    dask_print(f"{wlog_prefix(w_name, w_address, pid)} {block_info[0]['chunk-location']} img_real: {type(img_real)}, {img_real.shape}, {img_real.dtype}\n{img_real}")

    te_func = time.perf_counter()
    func_time = te_func - ts_func

    dask_print(f"{wlog_prefix(w_name, w_address, pid, 'T')} total time = {func_time:.3f} sec at {datetime.now()}")

    append_worker_stat({
        "event": "idg",
        "xds_name": xds_name,
        "loc": str(loc),
        "times": {
            "total_s": f"{tidg:.3f}",
            "aterms_s": f"{te_aterms - ts_aterms:.3f}",
            "get_grid_s": f"{time_get_grid:.3f}",
            "gridding_s": f"{time_gridding:.3f}",
            "cache_s": f"{time_cache:.3f}",
            "set_grid_s": f"{time_set_grid:.3f}",
            "fft_s": f"{time_fft:.3f}",
            "real_s": f"{time_r0:.3f}"
        },
    },
    out_dir=out_dir,
    out_tag=out_tag)

    append_worker_stat({
        "event": "flag_me",
        "xds_name": xds_name,
        "loc": str(loc),
        "flag_me": {
            "total_time_s": f"{func_time:.3f}",
        },
    },
    out_dir=out_dir,
    out_tag=out_tag)


    check_nan_inf(img_real, "img_real")

    return img_real


def print_banner(run_id, dask_worker=None) -> None:
    """Runs on a worker via client.run(): print a banner marking the start of run_id in that worker's log."""
    import socket
    import os
    import time

    msg = (
        "\n" * 3 + "=" * 80 + "\n"
        f"NEW RUN {run_id}\n"
        f"worker : {dask_worker.address if dask_worker else 'unknown'}\n"
        f"name   : {getattr(dask_worker, 'name', 'unknown')}\n"
        f"host   : {socket.gethostname()}\n"
        f"pid    : {os.getpid():7d}\n"
        f"time   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "=" * 80 + "\n"
    )

    dask_print(msg, flush=True)


def add_flag2(ps_zarr_path: str, ms_name: str) -> None:
    """
    Add an empty FLAG2 variable (a copy of FLAG) to ms_name in the MSv4 Zarr
    store at ps_zarr_path, and register it under a new "aoflagger" entry in
    the group's data_groups attrs so xradio picks it up as a valid flag
    variable. No-op if FLAG2 is already present. AOFlagger results get
    written into FLAG2 later (in flag_me()), leaving the original FLAG
    untouched for comparison.
    """

    # Open with xradio to respect the MSv4 layout
    ps_xdt = open_processing_set(ps_zarr_path)

    if ms_name not in ps_xdt:
        raise KeyError(f"{ms_name!r} not found. Available nodes: {list(ps_xdt.children)}")

    # Convert DatasetView -> mutable Dataset
    ms_xds = ps_xdt[ms_name].to_dataset().copy()

    if "FLAG" not in ms_xds:
        raise KeyError(
            f"FLAG not found in {ms_name}. "
            f"Available variables: {list(ms_xds.data_vars)}"
        )

    if "FLAG2" in ms_xds:
        print("-I- FLAG2 already visible through xradio")
        return

    # Create FLAG2 as a true xarray variable, same dims/chunks/coords as FLAG
    flag2 = ms_xds["FLAG"].copy(deep=True).rename("FLAG2")
    flag2.attrs.update({
        "description": "Copy of FLAG updated by AOFlagger",
        "meaning": "True means flagged / invalid sample",
    })

    # Write only the new variable to the existing MS group
    xr.Dataset({"FLAG2": flag2}).to_zarr(
        ps_zarr_path,
        group=ms_name,
        mode="a",
        consolidated=False,
    )

    # Patch data_groups metadata for xradio/MSv4
    root = zarr.open_group(ps_zarr_path, mode="a")
    grp = root[ms_name]

    data_groups = copy.deepcopy(ms_xds.attrs.get("data_groups", {}))

    base = data_groups.get("base", {}).copy()
    if not base:
        base = {
            "correlated_data": "VISIBILITY",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }

    data_groups["aoflagger"] = base.copy()
    data_groups["aoflagger"]["flag"] = "FLAG2"

    grp.attrs["data_groups"] = data_groups

    # Refresh consolidated metadata if present/used
    zarr.consolidate_metadata(ps_zarr_path)

    print(f"-I- Added FLAG2 to {ps_zarr_path}/{ms_name}")
    print("-I- data_groups =", data_groups)

    # Verify with plain xarray
    ds_check = xr.open_zarr(
        ps_zarr_path,
        group=ms_name,
        chunks={},
        consolidated=False,
    )
    print("-I- xarray vars:", list(ds_check.data_vars))
    print("-I- FLAG2 visible in xarray:", "FLAG2" in ds_check.data_vars)

    # Verify with xradio again
    ps_check = open_processing_set(ps_zarr_path)
    ms_check = ps_check[ms_name].to_dataset()
    print("-I- xradio vars:", list(ms_check.data_vars))
    print("-I- FLAG2 visible in xradio:", "FLAG2" in ms_check.data_vars)


def report_affinity(dask_worker=None):
    """Runs on a worker via client.run(): report its host/pid/CPU affinity mask."""
    import os
    import socket

    return {
        "name": str(getattr(dask_worker, "name", "")),
        "address": str(getattr(dask_worker, "address", "")),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "n_affinity": len(os.sched_getaffinity(0)),
    }


def run_pipeline(
    client: Client,
    args: argparse.Namespace,
) -> None:
    """
    Run one AOFlagger + IDG pass over the MSv4 dataset at args.msv4.

    Resets workers' per-worker cache, ensures FLAG2 exists (or resets it to
    all-False if reusing an existing one), then for each MS in the
    processing set: rechunks VISIBILITY/FLAG from their on-disk chunking to
    args.proc_chunking if needed, dispatches flag_me() over the dataset via
    da.map_blocks(), and sums the per-chunk gridded images into a final
    image (printed for diagnostics only, not saved — the durable outputs of
    a run are the FLAG2 array written by flag_me() and the per-run
    stats/dask-report files under args.out_dir).
    """

    print("===== run_pipeline =====")
    print(client)
    print(args)
    print("========================")

    # Sanity checks
    #assert args.num_timesteps > 0
    #assert args.num_channels > 0
    assert args.fov_deg > 0
    assert args.fov_deg < 180

    run_id = str(uuid.uuid4())[:8]

    # Field of view
    fov_rad = args.fov_deg / 180.0 * np.pi
    print(f"-I- FoV = {args.fov_deg} [deg] = {fov_rad} [rad]")

    cell_size = fov_rad / args.grid_size
    print(f"-I- cell size = {fov_rad} / {args.grid_size} = {cell_size:.7f} [rad]")

    workers = list(client.scheduler_info()["workers"])
    n_workers = len(workers)

    # Print a banner to indicate new run in workers-log
    client.run(print_banner, run_id, workers=[workers[0]])

    # Print CPU affinity
    aff = client.run(report_affinity)
    print(f"-I- {now_ms_string()}")
    print(f"Dask workers ({n_workers}) CPU affinity ")
    print("=" * 100)
    for addr, r in sorted(aff.items()):
        cpus = r.get("affinity", [])
        print(f"addr={addr:23s} "
              f"{r.get('host', ''):5s} "
              f"pid={r.get('pid', ''):<8} "
              f"name={str(r.get('name', '')):<8} "
              f"ncpus={len(cpus):<3} "
              f"cpus={format_cpu_list(cpus)} ")


    # Reset workers
    print(f"-I- {now_ms_string()} resetting workers' cache")
    client.run(reset_worker_cache)

    print(f"-I- {now_ms_string()} resetting workers' ps")
    client.run(reset_ps, msv4=args.msv4)


    root = zarr.open(args.msv4, mode="a")
    grp = root[args.xds_name]
    flag = grp["FLAG"]


    # Create FLAG2, always -- assume AOFlagger to be run
    # --------------------------------------------------
    #if "FLAG2" in grp:
    #    del grp["FLAG2"]
    #    zarr.consolidate_metadata(args.msv4)
    #    print("-I- Deleted old raw FLAG2")
    #add_flag2(ps_zarr_path=args.msv4, ms_name=args.xds_name)
    if "FLAG2" not in grp:
        s_addf2 = time.perf_counter()
        add_flag2(ps_zarr_path=args.msv4, ms_name=args.xds_name)
        e_addf2 = time.perf_counter()
        print("-I- Created FLAG2")
        append_client_stat({
            "event": "run_pipeline",
            "xds_name": args.xds_name,
            "add_flag2_to_zarr": {
                "time_s": f"{e_addf2 - s_addf2:.3f}",
            }},
        out_tag=args.out_tag)
    else:
        print("-I- Reusing existing FLAG2: resetting whole thing to False")
        s_resetf2 = time.perf_counter()
        grp["FLAG2"][:] = False
        e_resetf2 = time.perf_counter()
        append_client_stat({
            "event": "run_pipeline",
            "xds_name": args.xds_name,
            "reset_flag2": {
                "time_s": f"{e_resetf2 - s_resetf2:.3f}",
            }},
        out_tag=args.out_tag)


    # See https://xradio.readthedocs.io/en/latest/measurement_set/overview.html
    ps_xdt = open_processing_set(ps_store=args.msv4)
    print("-I- len(ps_xdt) =", len(ps_xdt))


    zarr_version = guess_zarr_format(args.msv4)

    for ms_name in ps_xdt:

        print(f"\n\n====================== ms {ms_name} ======================\n")
        ms_xdt = ps_xdt[ms_name]
        print("-I- ms_xdt type:", type(ms_xdt), "\n", ms_xdt)

        NB = args.nb if args.nb is not None else flag.shape[1]
        print("-I- NB =", NB)

        flag = ms_xdt.FLAG[:,0:NB,]
        print("-I- type(flag)      =", type(flag))
        print("-I- type(flag.data) =", type(flag.data))
        print("-I- flat.data.dtype =", flag.data.dtype)


        vis = ms_xdt.VISIBILITY
        print("-I- type(ms_xdt.VISIBILITY) =", type(ms_xdt.VISIBILITY), flush=True)
        vis = ms_xdt.VISIBILITY[:,0:NB,]
        print("-I- type(ms_xdt.VISIBILITY[:,0:NB,]) =", type(ms_xdt.VISIBILITY[:,0:NB,]), flush=True)
        print("-I- vis =\n", vis)
        print("----------------------------------------------------------------")
        print("-I- type(vis)      =", type(vis))
        print("-I- type(vis.data) =", type(vis.data))
        print("-I- vis.data.dtype =", vis.data.dtype)
        print("-I- vis.dims       =", vis.dims)
        print("-I- vis.shape      =", vis.shape)
        #print("-I- vis.data.chunksize =", vis.data.chunksize)

        zarr_chunking = dict(zip(vis.dims, vis.data.chunksize))
        proc_chunking = dict(zip(vis.dims, tuple(args.proc_chunking)))

        #EO: check if all that can't be done through map_blocks directly
        if proc_chunking != zarr_chunking:
            t_chunk0 = time.perf_counter()
            vis = vis.chunk(proc_chunking)
            t_chunk1 = time.perf_counter()
            flag = flag.chunk(proc_chunking)
            t_chunk2 = time.perf_counter()
            print(f"-T- t_chunk vis  = {t_chunk1-t_chunk0:.3f} sec")
            print(f"-T- t_chunk flag = {t_chunk2-t_chunk1:.3f} sec")
            print("-I- new proc_chunking =", proc_chunking)
        else:
            print("-I- Rechunking not required.")

        for d in ("time", "baseline_id", "frequency"):
            if proc_chunking[d] % zarr_chunking[d] != 0 and zarr_chunking[d] % proc_chunking[d] != 0:
                print("-W- Expect different loading times due to misaling chunking "
                      f"between disk and processing on dimension \"{d}\" (disk:{zarr_chunking[d]} vs proc:{proc_chunking[d]})")


        #nt, nb, nf, npol = vis.shape

        #TODO: understand the impact of creating a dummy structure to cancel
        #      getitem calls and therefore transfers between workers
        dummy = xr.DataArray(
            da.empty_like(vis.data, dtype=np.int8),
            dims=vis.dims,
            coords=vis.coords,
            name="dummy",
            attrs={},
        )

        # Select between vis (data transfers) and dummy (no data transfers)
        #mapb_input = vis
        mapb_input = dummy
        print(f"-I- {now_ms_string()}", mapb_input.data)


        print(f"-I- {now_ms_string()} map_blocks() START")
        t_map0 =  time.perf_counter()
        with performance_report(filename=f"{args.out_tag}_dask-report.html"):
            partials = da.map_blocks(flag_me,
                                     mapb_input.data,
                                     msv4_path=args.msv4,
                                     pin_cpu_threads=args.pin_cpu_threads,
                                     xds_name=args.xds_name,
                                     compute_mode=args.mode,
                                     vis_type=args.vis_type,
                                     zarr_version=zarr_version,
                                     out_dir=args.out_dir,
                                     out_tag=args.out_tag,
                                     # aoflagger
                                     strategy_path=args.lua_strategy,
                                     # idg
                                     subgrid_size = args.subgrid_size,
                                     kernel_size = args.kernel_size,
                                     cell_size = cell_size,
                                     timeslot_size = args.timeslot_size,
                                     grid_size = args.grid_size,
                                     # n_threads=None (default) lets flag_me()
                                     # resolve it per-worker from SLURM_CPUS_PER_TASK
                                     # (this client job's own cpus-per-task can differ
                                     # from the workers').
                                     n_threads = args.n_threads,
                                     # output
                                     # float32, not configurable: IDG (idg.idgtypes) fixes
                                     # visibilities/grid/aterms/taper to complex64/float32
                                     # throughout, so there is no double-precision path to
                                     # plumb a --precision option into (see README.md).
                                     dtype=np.float32,
                                     chunks=(1, 1, 1, args.grid_size, args.grid_size),
                                     drop_axis=(3,),
                                     new_axis=(3, 4),
                                     nb=NB
                                     )

            print("-I- partials repr :", partials)
            print("-I- partials type :", type(partials))
            print("-I- partials shape:", partials.shape)
            print("-I- partials ndim :", partials.ndim)
            print("-I- partials chunks:", getattr(partials, "chunks", None))
            print("-I- partials sum attr:", partials.sum)

            final = partials.sum(axis=(0, 1, 2))
            #final = partials.mean(axis=(0, 1, 2))

            print("-I- final repr  :", final)
            print("-I- final type  :", type(final))
            print("-I- final shape :", final.shape)
            print("-I- final ndim  :", final.ndim)
            print("-I- final chunks:", getattr(final, "chunks", None))

            final_image = final.compute()

            print("-I- final_image type :", type(final_image))
            print("-I- final_image shape:", getattr(final_image, "shape", None))
            print("-I- final_image dtype:", getattr(final_image, "dtype", None))
            print("-I- final_image =\n", final_image)

        t_map1 = time.perf_counter()
        print(f"-I- {now_ms_string()} map_blocks() END")
        print(f"-T- t_map  = {t_map1-t_map0:.3f} s")
        append_client_stat({
            "event": "run_pipeline",
            "xds_name": args.xds_name,
            "map_blocks": {
                "time_s": f"{t_map1 - t_map0:.3f}",
            }},
        out_tag=args.out_tag)


LogRecord: TypeAlias = tuple[str, str]
WorkerLogs: TypeAlias = Mapping[str, Sequence[LogRecord]]


def new_log_records(
    before: Sequence[LogRecord],
    after: Sequence[LogRecord],
) -> list[LogRecord]:
    """
    Return records added after `before`.

    Handles Dask's bounded log buffer by finding the largest overlap between
    the end of the baseline and the beginning of the final snapshot.
    """
    before = list(before)
    after = list(after)

    if not before:
        return after

    # Normal case: the original snapshot remains an exact prefix.
    if len(after) >= len(before) and after[: len(before)] == before:
        return after[len(before) :]

    # Older records may have been evicted from Dask's bounded log buffer.
    for overlap in range(min(len(before), len(after)), 0, -1):
        if before[-overlap:] == after[:overlap]:
            return after[overlap:]

    # No reliable overlap found. Return the final snapshot rather than
    # silently losing potentially relevant messages.
    return after


def capture_log_baseline(
    client: Client,
) -> tuple[list[LogRecord], dict[str, list[LogRecord]]]:
    """Snapshot current scheduler/worker logs so print_logs_since_baseline() can later show only what's new."""
    scheduler_logs = list(client.get_scheduler_logs())

    worker_logs = {
        worker: list(records)
        for worker, records in client.get_worker_logs().items()
    }

    return scheduler_logs, worker_logs


def print_logs_since_baseline(
    client: Client,
    scheduler_before: Sequence[LogRecord],
    workers_before: WorkerLogs,
) -> None:
    """
    Print scheduler and worker records generated during the client workload.

    Output goes to the client process stdout and is therefore captured in the
    client's Slurm output file.
    """
    scheduler_after = list(client.get_scheduler_logs())

    workers_after = {
        worker: list(records)
        for worker, records in client.get_worker_logs().items()
    }

    print("\n================ DASK SCHEDULER LOGS ================")

    scheduler_window = new_log_records(
        scheduler_before,
        scheduler_after,
    )

    if scheduler_window:
        for level, message in scheduler_window:
            print(f"[scheduler][{level}] {message}", flush=True)
    else:
        print("[scheduler] No new retained log records.", flush=True)

    print("\n================ DASK WORKER LOGS ===================")

    for worker in sorted(workers_after):
        print(f"\n----- {worker} -----", flush=True)

        worker_window = new_log_records(
            workers_before.get(worker, []),
            workers_after[worker],
        )

        if worker_window:
            for level, message in worker_window:
                print(
                    f"[worker={worker}][{level}] {message}",
                    flush=True,
                )
        else:
            print("No new retained log records.", flush=True)


def parse_args():
    """Parse command-line arguments for this script."""
    p = argparse.ArgumentParser()
    p.add_argument("--scheduler-file", type=str,   required=True,  help="Path to dask scheduler.json written by dask-scheduler")
    p.add_argument("--msv4",           type=str,   required=True,  help="Path to input MSV4 dataset")
    p.add_argument("--xds_name",       type=str,   required=True,  help="Xarray dataset name to process")
    p.add_argument("--proc_chunking",  type=int,   required=True,  help="Chunking used for processing the data (rechunking from Zarr on disk chunking)", nargs=4, metavar=("i","j","k","l"), )
    p.add_argument("--vis_type",       type=str,   required=True,  help="Visibility type to use", choices=["VISIBILITY", "VISIBILITY_CORRECTED", "VISIBILITY_MODEL"])
    p.add_argument("--lua_strategy",   type=str,   required=True,  help="Path to Aoflagger lua flagging strategy")
    # idg
    p.add_argument("--fov_deg",        type=float, required=True,  help="FoV in degrees")
    p.add_argument("--grid_size",      type=int,   required=True,  help="Image size in pixels")
    p.add_argument("--subgrid_size",   type=int,   required=True,  help="Subgrid size in pixels")
    p.add_argument("--kernel_size",    type=int,   required=True,  help="Kernel size in pixels")
    p.add_argument("--mode",           type=str,   required=True,  help="Proxy to be used", choices=["cpu", "hybrid", "gpu"])
    p.add_argument("--timeslot_size",  type=int,   required=True,  help="Number of timestep per timeslot")
    p.add_argument("--out_dir",        type=str,   required=True,  help="Output directory")
    p.add_argument("--out_tag",        type=str,   required=True,  help="Output tag")
    p.add_argument("--nb",             type=int,   required=True,  help="Limit number of baselines to process, for quick tests")
    p.add_argument("--pin_cpu_threads",action="store_true",        help="Pin each ThreadPoolExecutor thread to one CPU from the worker affinity mask")
    p.add_argument("--n_threads",      type=int,   default=None,   help="Override the number of AOFlagger threads per chunk (default: SLURM_CPUS_PER_TASK on the worker running that chunk)")
    return p.parse_args()


def main() -> None:
    """CLI entrypoint: connect to the Dask cluster, run run_pipeline(), and print scheduler/worker logs emitted during the run."""

    args = parse_args()
    print("-I- args =", args)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)-8s "
            "[%(name)s] %(message)s"
        ),
    )

    # Dask cluster
    # ------------
    s_dask = time.perf_counter()
    client = Client(scheduler_file=args.scheduler_file)
    print(f"-I- {now_ms_string()} Dask client =", client)
    e_dask = time.perf_counter()
    print(f"-T- {now_ms_string()} Dask client = {e_dask - s_dask:.3f} sec")


    append_client_stat({
        "event": "dask_client",
        "xds_name": args.xds_name,
        "setup": {
            "time_s": f"{e_dask - s_dask:.3f}",
        }},
        out_tag=args.out_tag)

    scheduler_before, workers_before = capture_log_baseline(client)

    s_run = time.perf_counter()
    try:
        print("Starting Dask client workload", flush=True)
        run_pipeline(client, args)
        print("Dask client workload completed", flush=True)

    finally:
        # Run while the scheduler and workers are still alive.
        try:
            print_logs_since_baseline(
                client,
                scheduler_before,
                workers_before,
            )
        except Exception:
            logging.exception(
                "Could not retrieve scheduler/worker logs"
            )

        client.close()

    e_run = time.perf_counter()
    append_client_stat({
        "event": "dask_client",
        "xds_name": args.xds_name,
        "run": {
            "time_s": f"{e_run - s_run:.3f}",
        }},
        out_tag=args.out_tag)


if __name__ == "__main__":
    main()
