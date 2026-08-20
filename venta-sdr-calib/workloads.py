"""Scientific workloads shared by MPI and Dask.

The functions in this module know nothing about either scheduling framework.
Both runtimes call the same group-processing functions with the same options.
"""

import hashlib
import json
import os
import re
import socket
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import fft as scipy_fft
from scipy import signal

try:
    import numba
except ImportError:  # Numba remains an optional benchmark dependency.
    numba = None


RAW_DTYPE = np.dtype([("re", np.int16), ("im", np.int16)])

# Preserve the historical order used by MPI_calibr_v2.py.
GROUP_MEMBER_ORDER = (
    ("s0", 1),
    ("s0", 2),
    ("r0", 1),
    ("r0", 2),
    ("s1", 1),
    ("s1", 2),
    ("r1", 1),
    ("r1", 2),
)

RAW_FILENAME = re.compile(
    r"^(?P<prefix>.+)_no(?P<scan>\d+)(?P<state>[sr][01])_ch(?P<channel>[12])\.raw$",
    re.IGNORECASE,
)
ZARR_FORMAT_NAME = "telescope-raw-iq-zarr-zip-v1"


def _select_groups(groups, group_offset, group_count):
    if group_offset >= len(groups):
        raise ValueError(
            "group offset %d leaves no groups (available=%d)"
            % (group_offset, len(groups))
        )
    selected = groups[group_offset:]
    if group_count:
        selected = selected[:group_count]
        if len(selected) != group_count:
            raise ValueError(
                "requested %d groups after offset, found %d"
                % (group_count, len(selected))
            )
    return selected


def discover_raw_groups(data_directory, group_offset=0, group_count=0):
    """Discover and strictly validate complete eight-file observation groups."""
    directory = Path(data_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("data directory does not exist: %s" % directory)

    raw_paths = sorted(directory.glob("*.raw"))
    if not raw_paths:
        raise ValueError("no .raw files found in %s" % directory)

    members = {}
    prefixes = set()
    unmatched = []
    for path in raw_paths:
        match = RAW_FILENAME.match(path.name)
        if match is None:
            unmatched.append(path.name)
            continue

        prefix = match.group("prefix")
        scan = int(match.group("scan"))
        state = match.group("state").lower()
        channel = int(match.group("channel"))
        key = (scan, state, channel)

        if key in members:
            raise ValueError("duplicate group member: %r" % (key,))
        prefixes.add(prefix)
        members[key] = str(path)

    if unmatched:
        raise ValueError(
            "raw filenames do not match the expected pattern: %s"
            % ", ".join(unmatched[:5])
        )
    if len(prefixes) != 1:
        raise ValueError("expected one observation prefix, found %s" % sorted(prefixes))

    complete_groups = []
    incomplete_groups = []
    for scan in sorted(set(key[0] for key in members)):
        missing = [
            item
            for item in GROUP_MEMBER_ORDER
            if (scan, item[0], item[1]) not in members
        ]
        if missing:
            incomplete_groups.append((scan, missing))
            continue

        paths = tuple(
            members[(scan, state, channel)]
            for state, channel in GROUP_MEMBER_ORDER
        )
        complete_groups.append(("no%03d" % scan, paths))

    if incomplete_groups:
        details = "; ".join(
            "no%03d missing %s" % (scan, missing)
            for scan, missing in incomplete_groups[:5]
        )
        raise ValueError("incomplete scan groups: %s" % details)

    return _select_groups(complete_groups, group_offset, group_count)


def _load_zarr2():
    try:
        import zarr
        from numcodecs import blosc
    except ImportError as exc:
        raise RuntimeError(
            "Zarr input requires the versions in requirements-zarr.txt"
        ) from exc
    if int(zarr.__version__.split(".", 1)[0]) != 2:
        raise RuntimeError("Zarr input requires zarr 2, found %s" % zarr.__version__)
    return zarr, blosc


def _ordered_zarr_members(group_id, sources):
    members = {}
    prefixes = set()
    for source in sources:
        name = str(source.get("name", ""))
        size = int(source.get("size", -1))
        match = RAW_FILENAME.match(name)
        if match is None or size <= 0 or size % RAW_DTYPE.itemsize:
            raise ValueError("invalid source metadata in %s: %r" % (group_id, source))
        scan = int(match.group("scan"))
        expected_group_id = "no%03d" % scan
        if expected_group_id != group_id:
            raise ValueError(
                "source %s belongs to %s, not %s" % (name, expected_group_id, group_id)
            )
        state = match.group("state").lower()
        channel = int(match.group("channel"))
        key = (state, channel)
        if key in members:
            raise ValueError("duplicate Zarr member %r in %s" % (key, group_id))
        prefixes.add(match.group("prefix"))
        members[key] = (name, size)

    if len(prefixes) != 1 or set(members) != set(GROUP_MEMBER_ORDER):
        raise ValueError("Zarr archive %s does not describe one complete group" % group_id)
    return tuple(members[key] for key in GROUP_MEMBER_ORDER)


def discover_zarr_groups(data_directory, group_offset=0, group_count=0):
    """Discover verified Zarr 2 ZipStore groups without reading IQ payloads."""
    zarr, _blosc = _load_zarr2()
    directory = Path(data_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("data directory does not exist: %s" % directory)
    archives = sorted(directory.glob("no*.zarr.zip"))
    if not archives:
        raise ValueError("no no*.zarr.zip archives found in %s" % directory)

    groups = []
    seen = set()
    common_configuration = None
    for archive in archives:
        store = zarr.ZipStore(str(archive), mode="r")
        try:
            root = zarr.open_group(store=store, mode="r")
            group_id = str(root.attrs.get("group_id", ""))
            if not re.match(r"^no\d{3,}$", group_id):
                raise ValueError("invalid group_id in %s" % archive)
            if root.attrs.get("format") != ZARR_FORMAT_NAME:
                raise ValueError("unexpected Zarr format in %s" % archive)
            if archive.name != group_id + ".zarr.zip":
                raise ValueError("archive name and group_id disagree: %s" % archive)
            if group_id in seen:
                raise ValueError("duplicate Zarr group: %s" % group_id)
            sources = list(root.attrs.get("sources", []))
            members = _ordered_zarr_members(group_id, sources)
            configuration = {
                "format": ZARR_FORMAT_NAME,
                "zarr_store": "ZipStore",
                "compressor": str(root.attrs.get("compressor", "")),
                "compression_level": int(root.attrs.get("compression_level", -1)),
                "shuffle": str(root.attrs.get("shuffle", "")),
                "chunk_samples": int(root.attrs.get("chunk_samples", -1)),
                "array_dtype": "int16",
            }
            if (
                not configuration["compressor"]
                or configuration["compression_level"] < 0
                or not configuration["shuffle"]
                or configuration["chunk_samples"] <= 0
            ):
                raise ValueError("incomplete storage configuration in %s" % archive)
            if common_configuration is None:
                common_configuration = configuration
            elif configuration != common_configuration:
                raise ValueError("mixed Zarr configurations found in %s" % directory)
            if sorted(root.array_keys()) != sorted(name for name, _size in members):
                raise ValueError("array keys do not match source metadata in %s" % archive)
            for name, size in members:
                array = root[name]
                expected_shape = (size // RAW_DTYPE.itemsize, 2)
                if array.shape != expected_shape or np.dtype(array.dtype) != np.dtype(np.int16):
                    raise ValueError(
                        "%s in %s has shape/dtype %s/%s, expected %s/int16"
                        % (name, archive.name, array.shape, array.dtype, expected_shape)
                    )
        finally:
            store.close()

        marker_path = archive.with_name(archive.name + ".verified.json")
        if not marker_path.is_file():
            raise ValueError("verification marker is missing: %s" % marker_path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        archive_stat = archive.stat()
        if (
            marker.get("format") != ZARR_FORMAT_NAME
            or marker.get("group_id") != group_id
            or marker.get("archive_size") != int(archive_stat.st_size)
            or marker.get("archive_mtime_ns") != int(archive_stat.st_mtime_ns)
            or marker.get("sources") != sources
        ):
            raise ValueError("verification marker does not match archive: %s" % archive)

        seen.add(group_id)
        groups.append(
            (
                group_id,
                {
                    "input_format": "zarr",
                    "archive": str(archive),
                    "members": members,
                    "zarr_configuration": configuration,
                },
            )
        )

    groups.sort(key=lambda item: int(item[0][2:]))
    return _select_groups(groups, group_offset, group_count)


def discover_real_groups(data_directory, input_format, group_offset=0, group_count=0):
    if input_format == "raw":
        return discover_raw_groups(data_directory, group_offset, group_count)
    if input_format == "zarr":
        return discover_zarr_groups(data_directory, group_offset, group_count)
    raise ValueError("unknown input format: %s" % input_format)


def create_virtual_groups(group_count):
    """Create deterministic filenames without creating filesystem objects."""
    groups = []
    for scan in range(1, group_count + 1):
        group_id = "synthetic_no%06d" % scan
        names = tuple(
            "%s%s_ch%d.raw" % (group_id, state, channel)
            for state, channel in GROUP_MEMBER_ORDER
        )
        groups.append((group_id, names))
    return groups


@lru_cache(maxsize=32)
def blackman_harris_window(fft_size, precision):
    """Return the production float64 window or the explicit float32 variant."""
    dtype = np.float64 if precision == "f64" else np.float32
    window = signal.windows.blackmanharris(fft_size).astype(dtype)
    window.setflags(write=False)
    return window


def _python_overlap_indices(fft_size, sample_count):
    segment_count = sample_count // fft_size
    indices = np.empty((segment_count, 2), dtype=np.int64)
    pointer = fft_size
    bin_pointer = 0

    for index in range(segment_count):
        start = bin_pointer * fft_size + pointer - fft_size
        end = bin_pointer * fft_size + pointer
        indices[index, 0] = start
        indices[index, 1] = end

        pointer = fft_size + int(fft_size * (0.661 if index % 2 else 0.339))
        if index % 3 == 2 and index != 0:
            bin_pointer += 1
            pointer = fft_size
    return indices


if numba is not None:

    @numba.njit(cache=True, nogil=True)
    def _numba_overlap_indices(fft_size, sample_count):
        segment_count = sample_count // fft_size
        indices = np.empty((segment_count, 2), dtype=np.int64)
        pointer = fft_size
        bin_pointer = 0

        for index in range(segment_count):
            start = bin_pointer * fft_size + pointer - fft_size
            end = bin_pointer * fft_size + pointer
            indices[index, 0] = start
            indices[index, 1] = end

            if index % 2:
                pointer = fft_size + int(fft_size * 0.661)
            else:
                pointer = fft_size + int(fft_size * 0.339)
            if index % 3 == 2 and index != 0:
                bin_pointer += 1
                pointer = fft_size
        return indices


    @numba.njit(cache=True, nogil=True)
    def _numba_iq_conversion_f32(real, imaginary):
        output = np.empty(real.size, dtype=np.complex64)
        scale = np.float32(1.0e-3)
        for index in range(real.size):
            output[index] = complex(real[index] * scale, imaginary[index] * scale)
        return output


    @numba.njit(cache=True, nogil=True)
    def _numba_iq_conversion_f64(real, imaginary):
        output = np.empty(real.size, dtype=np.complex128)
        scale = np.float64(1.0e-3)
        for index in range(real.size):
            output[index] = complex(real[index] * scale, imaginary[index] * scale)
        return output

else:
    _numba_overlap_indices = None
    _numba_iq_conversion_f32 = None
    _numba_iq_conversion_f64 = None


def require_numba():
    if numba is None:
        raise RuntimeError("--numba on was requested, but Numba is not installed")


@lru_cache(maxsize=128)
def overlap_indices(fft_size, sample_count, use_numba):
    """Build the historical 0.339/0.661 overlap index sequence."""
    segment_count = sample_count // fft_size
    if segment_count <= 0:
        raise ValueError("input contains fewer samples than one FFT")

    if use_numba:
        require_numba()
        indices = _numba_overlap_indices(fft_size, sample_count)
    else:
        indices = _python_overlap_indices(fft_size, sample_count)

    if int(indices[-1, 1]) > sample_count:
        raise ValueError("historical overlap indices exceed the input array")
    indices.setflags(write=False)
    return indices


def convert_iq_components(real_component, imaginary_component, use_numba, precision):
    """Convert separate int16 I/Q components using the selected precision."""
    if use_numba:
        require_numba()
        converter = (
            _numba_iq_conversion_f64
            if precision == "f64"
            else _numba_iq_conversion_f32
        )
        return converter(real_component, imaginary_component)

    if precision == "f64":
        real = real_component.astype(np.float64)
        imaginary = imaginary_component.astype(np.float64)
        return (real + 1j * imaginary) * np.float64(1.0e-3)

    real = real_component.astype(np.float32)
    imaginary = imaginary_component.astype(np.float32)
    iq = (real + 1j * imaginary) * np.float32(1.0e-3)
    return iq.astype(np.complex64, copy=False)


def convert_raw_iq(raw_samples, use_numba, precision):
    """Convert structured Raw int16 I/Q samples."""
    return convert_iq_components(
        raw_samples["re"],
        raw_samples["im"],
        use_numba,
        precision,
    )


def calculate_power_spectrum(iq_samples, processing):
    """Apply the production window/overlap FFT accumulation to one file."""
    fft_size = processing["fft_size"]
    precision = processing["precision"]
    indices = overlap_indices(
        fft_size,
        int(iq_samples.shape[0]),
        processing["use_numba"],
    )
    window = blackman_harris_window(fft_size, precision)
    accumulator = np.zeros(fft_size, dtype=np.float64)
    segment_dtype = np.complex128 if precision == "f64" else np.complex64
    result_dtype = np.float64 if precision == "f64" else np.float32
    segment = np.empty(fft_size, dtype=segment_dtype)

    for start, end in indices:
        np.multiply(iq_samples[start:end], window, out=segment)
        if processing["fft_backend"] == "scipy":
            transformed = scipy_fft.fft(segment, workers=1)
        else:
            transformed = np.fft.fft(segment)
        accumulator += (
            transformed.real * transformed.real
            + transformed.imag * transformed.imag
        )

    spectrum = np.fft.fftshift(accumulator) / float(indices.shape[0])
    return spectrum.astype(result_dtype, copy=False)


def read_raw_file(path, io_gate=None):
    """Read one file while measuring lease wait and actual filesystem I/O."""
    if io_gate is None:
        wait_seconds = 0.0
        read_started = time.perf_counter()
        raw = np.fromfile(path, dtype=RAW_DTYPE)
        read_seconds = time.perf_counter() - read_started
    else:
        wait_started = time.perf_counter()
        with io_gate:
            wait_seconds = time.perf_counter() - wait_started
            read_started = time.perf_counter()
            raw = np.fromfile(path, dtype=RAW_DTYPE)
            read_seconds = time.perf_counter() - read_started

    if raw.size == 0:
        raise ValueError("empty raw file: %s" % path)
    return raw, wait_seconds, read_seconds


def process_raw_group(group, processing, io_gate=None):
    """Read and process all eight members of one real scan group."""
    group_id, paths = group
    group_started = time.perf_counter()
    spectra = []
    wait_seconds = 0.0
    read_seconds = 0.0
    conversion_seconds = 0.0
    fft_seconds = 0.0
    input_bytes = 0
    sample_count = 0

    for path in paths:
        raw, waited, elapsed = read_raw_file(path, io_gate)
        wait_seconds += waited
        read_seconds += elapsed
        input_bytes += int(raw.nbytes)
        sample_count += int(raw.shape[0])

        started = time.perf_counter()
        iq = convert_raw_iq(
            raw,
            processing["use_numba"],
            processing["precision"],
        )
        conversion_seconds += time.perf_counter() - started
        del raw

        started = time.perf_counter()
        spectra.append(calculate_power_spectrum(iq, processing))
        fft_seconds += time.perf_counter() - started
        del iq

    return _group_result(
        group_id,
        spectra,
        wait_seconds=wait_seconds,
        read_seconds=read_seconds,
        conversion_seconds=conversion_seconds,
        fft_seconds=fft_seconds,
        generation_seconds=0.0,
        group_seconds=time.perf_counter() - group_started,
        input_bytes=input_bytes,
        storage_bytes=input_bytes,
        sample_count=sample_count,
    )


def _read_zarr_member(root, member_name, io_gate):
    if io_gate is None:
        waited = 0.0
        started = time.perf_counter()
        matrix = root[member_name][:]
        elapsed = time.perf_counter() - started
    else:
        wait_started = time.perf_counter()
        with io_gate:
            waited = time.perf_counter() - wait_started
            started = time.perf_counter()
            matrix = root[member_name][:]
            elapsed = time.perf_counter() - started
    return matrix, waited, elapsed


def process_zarr_group(group, processing, io_gate=None):
    """Read one verified ZipStore and run the shared kernel on its eight arrays."""
    zarr, blosc = _load_zarr2()
    blosc.set_nthreads(int(processing["blosc_threads"]))
    group_id, descriptor = group
    archive = descriptor["archive"]
    members = descriptor["members"]
    group_started = time.perf_counter()
    spectra = []
    wait_seconds = 0.0
    read_seconds = 0.0
    conversion_seconds = 0.0
    fft_seconds = 0.0
    logical_bytes = 0
    sample_count = 0

    started = time.perf_counter()
    store = zarr.ZipStore(archive, mode="r")
    try:
        root = zarr.open_group(store=store, mode="r")
        read_seconds += time.perf_counter() - started
        for member_name, expected_bytes in members:
            matrix, waited, elapsed = _read_zarr_member(root, member_name, io_gate)
            wait_seconds += waited
            read_seconds += elapsed
            if matrix.ndim != 2 or matrix.shape[1] != 2 or matrix.dtype != np.int16:
                raise ValueError("invalid Zarr IQ array: %s in %s" % (member_name, archive))
            if int(matrix.nbytes) != int(expected_bytes):
                raise ValueError("Zarr IQ byte count changed: %s in %s" % (member_name, archive))
            logical_bytes += int(matrix.nbytes)
            sample_count += int(matrix.shape[0])

            started = time.perf_counter()
            iq = convert_iq_components(
                matrix[:, 0],
                matrix[:, 1],
                processing["use_numba"],
                processing["precision"],
            )
            conversion_seconds += time.perf_counter() - started
            del matrix

            started = time.perf_counter()
            spectra.append(calculate_power_spectrum(iq, processing))
            fft_seconds += time.perf_counter() - started
            del iq
    finally:
        started = time.perf_counter()
        store.close()
        read_seconds += time.perf_counter() - started

    return _group_result(
        group_id,
        spectra,
        wait_seconds=wait_seconds,
        read_seconds=read_seconds,
        conversion_seconds=conversion_seconds,
        fft_seconds=fft_seconds,
        generation_seconds=0.0,
        group_seconds=time.perf_counter() - group_started,
        input_bytes=logical_bytes,
        storage_bytes=int(os.path.getsize(archive)),
        sample_count=sample_count,
    )


def process_real_group(group, processing, io_gate=None):
    """Dispatch one real group to the selected storage-format reader."""
    if processing.get("input_format", "raw") == "raw":
        return process_raw_group(group, processing, io_gate=io_gate)
    return process_zarr_group(group, processing, io_gate=io_gate)


_SYNTHETIC_CACHE = {}


def deterministic_seed(name, base_seed):
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(base_seed).encode("ascii"))
    digest.update(b"|")
    digest.update(name.encode("utf-8"))
    return int.from_bytes(digest.digest()[:4], "little", signed=False)


def generate_synthetic_raw(sample_count, seed, noise_standard_deviation):
    """Generate deterministic Gaussian int16 IQ without writing a file."""
    random = np.random.default_rng(seed)
    raw = np.empty(sample_count, dtype=RAW_DTYPE)
    chunk_size = 2_000_000

    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        count = end - start
        real = random.normal(0.0, noise_standard_deviation, size=count)
        imaginary = random.normal(0.0, noise_standard_deviation, size=count)
        raw["re"][start:end] = np.clip(np.rint(real), -32768, 32767).astype(np.int16)
        raw["im"][start:end] = np.clip(np.rint(imaginary), -32768, 32767).astype(np.int16)
    return raw


def cached_synthetic_raw(synthetic):
    key = (
        synthetic["samples_per_file"],
        synthetic["seed"],
        synthetic["noise_std"],
    )
    raw = _SYNTHETIC_CACHE.get(key)
    if raw is None:
        raw = generate_synthetic_raw(*key)
        raw.setflags(write=False)
        _SYNTHETIC_CACHE[key] = raw
    return raw


def process_synthetic_group(group, processing, synthetic):
    """Process eight deterministic in-memory virtual files."""
    group_id, filenames = group
    group_started = time.perf_counter()
    spectra = []
    generation_seconds = 0.0
    conversion_seconds = 0.0
    fft_seconds = 0.0

    for filename in filenames:
        started = time.perf_counter()
        if synthetic["generation"] == "cached":
            raw = cached_synthetic_raw(synthetic)
        else:
            seed = deterministic_seed(filename, synthetic["seed"])
            raw = generate_synthetic_raw(
                synthetic["samples_per_file"],
                seed,
                synthetic["noise_std"],
            )
        generation_seconds += time.perf_counter() - started

        started = time.perf_counter()
        iq = convert_raw_iq(
            raw,
            processing["use_numba"],
            processing["precision"],
        )
        conversion_seconds += time.perf_counter() - started

        started = time.perf_counter()
        spectra.append(calculate_power_spectrum(iq, processing))
        fft_seconds += time.perf_counter() - started
        del iq
        if synthetic["generation"] == "per-file":
            del raw

    return _group_result(
        group_id,
        spectra,
        wait_seconds=0.0,
        read_seconds=0.0,
        conversion_seconds=conversion_seconds,
        fft_seconds=fft_seconds,
        generation_seconds=generation_seconds,
        group_seconds=time.perf_counter() - group_started,
        input_bytes=0,
        storage_bytes=0,
        sample_count=synthetic["samples_per_file"] * len(filenames),
    )


def _group_result(
    group_id,
    spectra,
    wait_seconds,
    read_seconds,
    conversion_seconds,
    fft_seconds,
    generation_seconds,
    group_seconds,
    input_bytes,
    storage_bytes,
    sample_count,
):
    return {
        "group_id": group_id,
        "worker_host": socket.gethostname(),
        "worker_pid": os.getpid(),
        "spectra": np.stack(spectra, axis=0),
        "timings": {
            "generation_s": generation_seconds,
            "wait_s": wait_seconds,
            "read_s": read_seconds,
            "convert_s": conversion_seconds,
            "fft_s": fft_seconds,
            "group_s": group_seconds,
        },
        "input_bytes": int(input_bytes),
        "storage_bytes": int(storage_bytes),
        "samples": int(sample_count),
    }


def prewarm_worker(processing, synthetic=None, dask_worker=None):
    """Compile Numba and initialize FFT/cache state before the timed section."""
    started = time.perf_counter()
    fft_size = processing["fft_size"]

    zarr_info = {}
    if processing.get("input_format") == "zarr":
        zarr, blosc = _load_zarr2()
        blosc.set_nthreads(int(processing["blosc_threads"]))
        zarr_info = {
            "zarr_version": zarr.__version__,
            "blosc_threads": int(processing["blosc_threads"]),
        }

    blackman_harris_window(fft_size, processing["precision"])
    overlap_indices(fft_size, 8 * fft_size, processing["use_numba"])

    probe_raw = np.zeros(fft_size, dtype=RAW_DTYPE)
    probe_iq = convert_raw_iq(
        probe_raw,
        processing["use_numba"],
        processing["precision"],
    )
    if processing["fft_backend"] == "scipy":
        scipy_fft.fft(probe_iq, workers=1)
    else:
        np.fft.fft(probe_iq)

    if synthetic and synthetic["generation"] == "cached":
        cached_synthetic_raw(synthetic)
        overlap_indices(
            fft_size,
            synthetic["samples_per_file"],
            processing["use_numba"],
        )

    try:
        cpu_affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_affinity = []

    result = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cpu_affinity": cpu_affinity,
        "local_directory": str(getattr(dask_worker, "local_directory", "")),
        "prewarm_s": time.perf_counter() - started,
        "numba_enabled": processing["use_numba"],
    }
    result.update(zarr_info)
    return result


def numba_version():
    return numba.__version__ if numba is not None else "not-installed"
