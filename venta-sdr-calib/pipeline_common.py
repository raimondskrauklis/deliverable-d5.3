"""Shared setup for the operational MPI and Dask calibration pipelines."""

import argparse
import os
import re
from pathlib import Path


GROUP_ID = re.compile(r"^no(?P<scan>[0-9]+)$")


def add_pipeline_arguments(parser, framework):
    parser.add_argument("experiment_log", help="SDR observation log")
    parser.add_argument("data_dir", help="Raw-file or verified Zarr input directory")
    parser.add_argument(
        "output_dir",
        help="New directory for calibrated arrays, DAT/JSON files, and plots",
    )
    parser.add_argument(
        "--input-format",
        choices=("raw", "zarr"),
        default="raw",
        help="Input representation (default: raw)",
    )
    parser.add_argument(
        "--precision",
        choices=("f64", "f32"),
        default="f64",
        help="Worker arithmetic precision; f64 matches the production pipeline",
    )
    parser.add_argument(
        "--fft-backend",
        choices=("scipy", "numpy"),
        default="scipy",
        help="FFT implementation used by every worker",
    )
    parser.add_argument(
        "--numba",
        choices=("off", "on"),
        default="on",
        help="Use Numba for IQ conversion and overlap-index construction",
    )
    parser.add_argument(
        "--blosc-threads",
        type=int,
        default=1,
        help="Blosc decompression threads per worker for Zarr input",
    )
    parser.add_argument(
        "--io-leases",
        type=int,
        default=0,
        help="Maximum simultaneous input reads; 0 means unrestricted",
    )
    if framework == "mpi":
        parser.add_argument(
            "--expected-workers",
            type=int,
            default=0,
            help="Fail unless MPI provides this many worker ranks",
        )
    else:
        parser.add_argument(
            "--workers",
            type=int,
            required=True,
            help="Number of Dask worker processes",
        )
        parser.add_argument(
            "--worker-affinity",
            choices=("shared", "distinct-cores", "distinct-cpus"),
            default="shared",
            help="Optional Linux CPU placement for Dask workers",
        )
        parser.add_argument(
            "--local-directory",
            default=os.environ.get("SLURM_TMPDIR", ""),
            help="Dask temporary directory (default: SLURM_TMPDIR)",
        )
        parser.add_argument(
            "--memory-limit",
            default=0,
            help="Dask per-worker memory limit; 0 disables spilling and pausing",
        )


def pipeline_parser(framework):
    parser = argparse.ArgumentParser(
        description=(
            "Process and calibrate one complete observation with %s" % framework.upper()
        )
    )
    add_pipeline_arguments(parser, framework)
    return parser


def validate_paths_and_options(args):
    log_path = Path(args.experiment_log).expanduser().resolve()
    data_path = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output_dir).expanduser().resolve()

    if not log_path.is_file():
        raise ValueError("experiment log does not exist: %s" % log_path)
    if not data_path.is_dir():
        raise ValueError("input directory does not exist: %s" % data_path)
    if output_path == data_path or data_path in output_path.parents:
        raise ValueError("output directory must not be inside the input directory")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError("output directory is not empty: %s" % output_path)
    if args.blosc_threads < 1:
        raise ValueError("--blosc-threads must be positive")
    if args.io_leases < 0:
        raise ValueError("--io-leases cannot be negative")
    if args.input_format == "raw" and args.blosc_threads != 1:
        raise ValueError("--blosc-threads only applies to Zarr input")
    if hasattr(args, "workers") and args.workers < 1:
        raise ValueError("--workers must be positive")
    if hasattr(args, "expected_workers") and args.expected_workers < 0:
        raise ValueError("--expected-workers cannot be negative")

    output_path.mkdir(parents=True, exist_ok=True)
    args.experiment_log = str(log_path)
    args.data_dir = str(data_path)
    args.output_dir = str(output_path)
    return output_path


def processing_configuration(args, fft_size):
    return {
        "fft_size": int(fft_size),
        "precision": args.precision,
        "fft_backend": args.fft_backend,
        "use_numba": args.numba == "on",
        "input_format": args.input_format,
        "blosc_threads": int(args.blosc_threads),
    }


def validate_complete_observation(groups, expected_scans):
    if len(groups) != int(expected_scans):
        raise ValueError(
            "observation log expects %d scan groups, input contains %d"
            % (int(expected_scans), len(groups))
        )

    scan_numbers = []
    for group_id, _descriptor in groups:
        match = GROUP_ID.fullmatch(group_id)
        if match is None:
            raise ValueError("invalid scan-group identifier: %s" % group_id)
        scan_numbers.append(int(match.group("scan")))
    expected = list(range(1, len(groups) + 1))
    if scan_numbers != expected:
        raise ValueError(
            "full calibration requires consecutive scan groups no001..no%03d; found %s"
            % (len(groups), scan_numbers[:10])
        )


def spectra_by_state(results):
    """Convert ordered eight-spectrum group results to legacy calibration lists."""
    buckets = {"s0": [], "s1": [], "r0": [], "r1": []}
    index = {
        "s0": (0, 1),
        "r0": (2, 3),
        "s1": (4, 5),
        "r1": (6, 7),
    }
    ordered = sorted(results, key=lambda result: result["group_id"])
    for result in ordered:
        spectra = result["spectra"]
        if spectra.ndim != 2 or spectra.shape[0] != 8:
            raise ValueError(
                "group %s returned shape %s instead of (8, Ns)"
                % (result["group_id"], spectra.shape)
            )
        for state, positions in index.items():
            for channel, position in enumerate(positions, start=1):
                buckets[state].append(
                    {
                        "file": "%s%s_ch%d" % (
                            result["group_id"], state, channel
                        ),
                        "data": spectra[position],
                    }
                )
    return buckets
