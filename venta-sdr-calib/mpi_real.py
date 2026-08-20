#!/usr/bin/env python3
"""Benchmark real Raw or verified Zarr groups with MPI."""

from launcher_setup import limit_math_libraries_to_one_thread

limit_math_libraries_to_one_thread()

from benchmark_cli import real_data_parser
from mpi_runtime import run_mpi


def main():
    parser = real_data_parser("mpi")
    return run_mpi("real", parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
