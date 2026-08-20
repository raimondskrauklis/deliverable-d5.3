#!/usr/bin/env python3
"""Benchmark real Raw or verified Zarr groups with a local Dask scheduler."""

from launcher_setup import limit_math_libraries_to_one_thread

limit_math_libraries_to_one_thread()

from benchmark_cli import real_data_parser
from dask_runtime import run_dask


def main():
    parser = real_data_parser("dask")
    return run_dask("real", parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
