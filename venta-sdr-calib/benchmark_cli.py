"""Command-line interfaces shared by the four public benchmark launchers."""

import argparse
import os


def _add_processing_options(parser):
    parser.add_argument("--ns", type=int, default=4096, help="FFT length")
    parser.add_argument(
        "--precision",
        choices=("f64", "f32"),
        default="f64",
        help=(
            "Numerical precision: f64 matches MPI_calibr_v2.py; f32 is an "
            "explicit optimized variant"
        ),
    )
    parser.add_argument(
        "--fft-backend",
        choices=("scipy", "numpy"),
        default="scipy",
        help="Single-thread FFT implementation used by every worker",
    )
    parser.add_argument(
        "--numba",
        choices=("off", "on"),
        default="off",
        help=(
            "Use Numba for int16-to-complex conversion and overlap-index "
            "construction. Numba does not accelerate the filesystem read itself."
        ),
    )


def _add_output_options(parser):
    parser.add_argument(
        "--output-dir",
        default="framework_benchmark_results",
        help="Directory for one JSON summary and per-group CSV per run",
    )
    parser.add_argument("--run-id", default="", help="Unique run label")
    parser.add_argument(
        "--block-id",
        default="",
        help="Matched randomized experiment block label recorded in the JSON",
    )
    parser.add_argument(
        "--repetition",
        type=int,
        default=0,
        help="Repetition number within the experiment design (0 means unspecified)",
    )
    parser.add_argument(
        "--order-index",
        type=int,
        default=0,
        help="One-based execution order within a block (0 means unspecified)",
    )
    parser.add_argument(
        "--cache-regime",
        default="uncontrolled/mixed",
        choices=(
            "client-cold/server-uncontrolled",
            "storage-cold",
            "warm",
            "uncontrolled/mixed",
            "not-applicable",
        ),
        help="Recorded cache condition; this option does not evict caches",
    )
    parser.add_argument(
        "--cpu-regime",
        choices=("physical-core", "smt", "unspecified"),
        default="unspecified",
        help=(
            "CPU placement regime recorded for provenance. physical-core uses "
            "one hardware thread per core; smt may use both hardware threads."
        ),
    )


def _add_dask_options(parser):
    parser.add_argument("--workers", type=int, required=True, help="Dask worker processes")
    parser.add_argument(
        "--worker-affinity",
        choices=("shared", "distinct-cores", "distinct-cpus"),
        default="shared",
        help=(
            "CPU placement inside the Slurm allocation. shared preserves the "
            "LocalCluster default; distinct-cores pins every worker to a "
            "different physical core; distinct-cpus pins every worker to a "
            "different logical CPU, spreading across physical cores before "
            "using SMT siblings. Both pinned modes reserve one CPU for control."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        default=0,
        help="Dask per-worker memory limit; 0 disables spilling and pausing",
    )
    parser.add_argument(
        "--local-directory",
        default=os.environ.get("SLURM_TMPDIR", ""),
        help="Dask temporary directory; defaults to SLURM_TMPDIR when set",
    )


def real_data_parser(framework):
    parser = argparse.ArgumentParser(
        description="Fair %s benchmark of real telescope Raw or Zarr groups"
        % framework.upper()
    )
    parser.add_argument("data_dir", help="Directory containing the selected input format")
    parser.add_argument(
        "--input-format",
        choices=("raw", "zarr"),
        default="raw",
        help="Storage format; both paths feed the identical conversion/FFT kernel",
    )
    parser.add_argument(
        "--blosc-threads",
        type=int,
        default=1,
        help="Blosc decompression threads per worker for Zarr input (default: 1)",
    )
    parser.add_argument(
        "--experiment-log",
        default="",
        help=(
            "Observation log recorded for provenance. The benchmark kernel does "
            "not parse this file; processing parameters such as --ns remain explicit."
        ),
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=0,
        help="Complete groups to process; 0 uses every available group",
    )
    parser.add_argument(
        "--group-offset",
        type=int,
        default=0,
        help="Skip this many numerically ordered groups before --n-groups",
    )
    parser.add_argument(
        "--io-leases",
        type=int,
        default=0,
        help="Maximum concurrent file reads; 0 means unrestricted",
    )
    _add_processing_options(parser)
    _add_output_options(parser)
    _add_framework_options(parser, framework)
    return parser


def synthetic_data_parser(framework):
    parser = argparse.ArgumentParser(
        description="Fair %s benchmark of deterministic synthetic groups"
        % framework.upper()
    )
    parser.add_argument("--groups", type=int, default=256, help="Virtual groups")
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=0,
        help="Samples per virtual file; 0 selects the short 8*Ns profile",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base deterministic seed")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=800.0,
        help="Gaussian IQ standard deviation in int16 units",
    )
    parser.add_argument(
        "--generation",
        choices=("cached", "per-file"),
        default="cached",
        help="Reuse a worker-local array or generate data for every virtual file",
    )
    _add_processing_options(parser)
    _add_output_options(parser)
    parser.set_defaults(cache_regime="not-applicable")
    _add_framework_options(parser, framework)
    return parser


def _add_framework_options(parser, framework):
    if framework == "dask":
        _add_dask_options(parser)
    else:
        parser.add_argument(
            "--expected-workers",
            type=int,
            default=0,
            help="Fail unless MPI size minus one equals this value",
        )


def validate_arguments(args, workload):
    """Normalize arguments and reject ambiguous benchmark configurations."""
    if args.ns <= 0:
        raise ValueError("--ns must be positive")
    if args.repetition < 0 or args.order_index < 0:
        raise ValueError("--repetition and --order-index cannot be negative")

    if workload == "real":
        if args.n_groups < 0 or args.group_offset < 0:
            raise ValueError("--n-groups and --group-offset cannot be negative")
        if args.io_leases < 0:
            raise ValueError("--io-leases cannot be negative")
        if args.blosc_threads <= 0:
            raise ValueError("--blosc-threads must be positive")
        if args.input_format == "raw" and args.blosc_threads != 1:
            raise ValueError("--blosc-threads only applies to --input-format zarr")
    else:
        if args.groups <= 0:
            raise ValueError("--groups must be positive")
        if args.samples_per_file < 0:
            raise ValueError("--samples-per-file cannot be negative")
        if args.samples_per_file == 0:
            args.samples_per_file = 8 * args.ns
        if args.samples_per_file < args.ns:
            raise ValueError("--samples-per-file must be at least --ns")
        if args.samples_per_file % args.ns:
            raise ValueError("--samples-per-file must be an integer multiple of --ns")
        if args.noise_std <= 0:
            raise ValueError("--noise-std must be positive")

    if hasattr(args, "workers") and args.workers <= 0:
        raise ValueError("--workers must be positive")


def processing_options(args):
    """Small serializable configuration passed unchanged to every worker."""
    options = {
        "fft_size": int(args.ns),
        "fft_backend": args.fft_backend,
        "precision": args.precision,
        "use_numba": args.numba == "on",
    }
    if hasattr(args, "input_format"):
        options.update(
            {
                "input_format": args.input_format,
                "blosc_threads": int(args.blosc_threads),
            }
        )
    return options


def synthetic_options(args):
    return {
        "samples_per_file": int(args.samples_per_file),
        "seed": int(args.seed),
        "noise_std": float(args.noise_std),
        "generation": args.generation,
    }
