#!/usr/bin/env python3
"""Plot the primary MPI/Dask by Raw/Zarr benchmark for the article.

Only run-level JSON records are used. The script validates that every
worker/framework/format cell has the same number of repetitions and that all
runs produced the same group-set and output hashes before creating the plot.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FRAMEWORKS = ("mpi", "dask")
FORMATS = ("raw", "zarr")
STYLES = {
    ("mpi", "raw"): dict(
        color="#D55E00", marker="o", linestyle="-",
        markerfacecolor="white", label="MPI, Raw",
    ),
    ("dask", "raw"): dict(
        color="#D55E00", marker="s", linestyle="--",
        markerfacecolor="white", label="Dask, Raw",
    ),
    ("mpi", "zarr"): dict(
        color="#0072B2", marker="o", linestyle="-",
        markerfacecolor="#0072B2", label="MPI, Zarr",
    ),
    ("dask", "zarr"): dict(
        color="#0072B2", marker="s", linestyle="--",
        markerfacecolor="#0072B2", label="Dask, Zarr",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Primary result directory")
    parser.add_argument("output_dir", type=Path, help="Directory for PDF and PNG output")
    parser.add_argument(
        "--additional-results-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional complete matched matrix to append, such as the SMT extension",
    )
    return parser.parse_args()


def load_and_validate(results_dirs: list[Path]) -> tuple[dict, list[int], int, dict]:
    cells: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    cell_repetitions: dict[tuple[int, str, str], set[int]] = defaultdict(set)
    group_hashes, output_hashes, group_counts = set(), set(), set()
    regimes: dict[int, set[str]] = defaultdict(set)

    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("workload") != "real" or record.get("numba") != "on":
                continue
            framework = record.get("framework")
            input_format = record.get("input_format")
            if framework not in FRAMEWORKS or input_format not in FORMATS:
                continue
            worker = int(record["worker_count"])
            key = (worker, framework, input_format)
            repetition = int(record["experiment_order"]["repetition"])
            if repetition in cell_repetitions[key]:
                raise ValueError(
                    f"Duplicate repetition {repetition} for matrix cell {key}"
                )
            cell_repetitions[key].add(repetition)
            cells[key].append(float(record["throughput_groups_per_hour"]))
            regimes[worker].add(record.get("cpu_regime", "unspecified"))
            group_hashes.add(record["group_set_hash"])
            output_hashes.add(record["output_hash"])
            group_counts.add(int(record["group_count"]))

    if not cells:
        raise ValueError(f"No primary real-data records found in {results_dirs}")
    if len(group_hashes) != 1 or len(output_hashes) != 1 or len(group_counts) != 1:
        raise ValueError("Runs disagree on group set, output values, or group count")

    workers = sorted({key[0] for key in cells})
    expected = {
        (worker, framework, input_format)
        for worker in workers
        for framework in FRAMEWORKS
        for input_format in FORMATS
    }
    if set(cells) != expected:
        raise ValueError(f"Incomplete primary matrix: missing {sorted(expected - set(cells))}")

    repetition_sets = {tuple(sorted(values)) for values in cell_repetitions.values()}
    if len(repetition_sets) != 1 or len(next(iter(repetition_sets))) < 2:
        raise ValueError(f"Unequal or insufficient repetitions: {repetition_sets}")
    ambiguous_regimes = {
        worker: values for worker, values in regimes.items() if len(values) != 1
    }
    if ambiguous_regimes:
        raise ValueError(f"Worker counts mix CPU regimes: {ambiguous_regimes}")
    worker_regimes = {worker: next(iter(values)) for worker, values in regimes.items()}
    return cells, workers, len(next(iter(repetition_sets))), worker_regimes


def main() -> int:
    args = parse_args()
    results_dirs = [args.results_dir.resolve()] + [
        path.resolve() for path in args.additional_results_dir
    ]
    cells, workers, repetitions, worker_regimes = load_and_validate(results_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
        }
    )

    fig, axis = plt.subplots(figsize=(3.45, 2.65))
    for framework, input_format in (
        ("mpi", "raw"),
        ("dask", "raw"),
        ("mpi", "zarr"),
        ("dask", "zarr"),
    ):
        values = [cells[(worker, framework, input_format)] for worker in workers]
        means = [statistics.mean(items) for items in values]
        deviations = [statistics.stdev(items) for items in values]
        axis.errorbar(
            workers,
            means,
            yerr=deviations,
            linewidth=1.45,
            markersize=3.0,
            capsize=2.2,
            markeredgewidth=0.7,
            **STYLES[(framework, input_format)],
        )

    smt_workers = [worker for worker in workers if worker_regimes[worker] == "smt"]
    physical_workers = [
        worker for worker in workers if worker_regimes[worker] == "physical-core"
    ]
    if smt_workers and physical_workers:
        smt_start = (max(physical_workers) + min(smt_workers)) / 2.0
        axis.axvspan(
            smt_start,
            max(workers) + 1,
            color="#777777",
            alpha=0.07,
            linewidth=0,
        )
        axis.text(
            (smt_start + max(workers)) / 2.0,
            0.05,
            "SMT",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Throughput (scan groups per hour)")
    axis.set_xticks(workers)
    if len(workers) > 6:
        axis.tick_params(axis="x", labelrotation=45)
    axis.set_xlim(min(workers) - 1, max(workers) + 1)
    axis.set_ylim(bottom=0)
    axis.legend(
        frameon=False, ncol=2, loc="upper left",
        columnspacing=0.9, handlelength=2.4,
    )
    fig.tight_layout(pad=0.35)

    stem = args.output_dir / "single_station_primary_scaling"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Validated {len(cells)} cells x {repetitions} repetitions")
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
