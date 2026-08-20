#!/usr/bin/env python3
"""Benchmark deterministic in-memory synthetic groups with Dask."""

from launcher_setup import limit_math_libraries_to_one_thread

limit_math_libraries_to_one_thread()

from benchmark_cli import synthetic_data_parser
from dask_runtime import run_dask


def main():
    parser = synthetic_data_parser("dask")
    return run_dask("synthetic", parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
