#!/usr/bin/env python3
"""Create reproducible tables and figures from a completed real-data matrix.

The script reads only the run-level JSON records. Launcher wall times are not
used because a successful Dask retry may include a failed startup attempt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


FRAMEWORKS = ("mpi", "dask")
NUMBA_MODES = ("off", "on")
FRAMEWORK_STYLE = {
    "mpi": {"color": "#0072B2", "marker": "o", "label": "MPI"},
    "dask": {"color": "#D55E00", "marker": "s", "label": "Dask"},
}
NUMBA_STYLE = {
    "off": {"color": "#7A5195", "marker": "o", "label": "Numba off"},
    "on": {"color": "#009E73", "marker": "s", "label": "Numba on"},
}
PHASE_STYLE = {
    "read_s": {"color": "#0072B2", "marker": "o", "label": "Raw read"},
    "convert_s": {"color": "#E69F00", "marker": "s", "label": "IQ conversion"},
    "fft_s": {"color": "#009E73", "marker": "^", "label": "FFT accumulation"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dirs",
        type=Path,
        nargs="+",
        help="One or more directories containing run JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RESULTS_DIR/figures)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Ignore records above this worker count (useful after a partial extension)",
    )
    return parser.parse_args()


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float, float]:
    mean, sd = mean_sd(values)
    if len(values) < 2:
        return mean, mean, mean
    df = len(values) - 1
    critical = float(student_t.ppf(0.975, df))
    half_width = critical * sd / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_records(results_dirs: list[Path]) -> list[dict]:
    records = []
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("workload") == "real":
                record["_path"] = str(path)
                records.append(record)
    if not records:
        raise ValueError(f"No real-workload JSON records found in {results_dirs}")
    return records


def validate_matrix(records: list[dict]) -> tuple[list[int], list[int], str]:
    keys = set()
    workers = sorted({int(record["worker_count"]) for record in records})
    repetitions = sorted(
        {int(record["experiment_order"]["repetition"]) for record in records}
    )
    expected = {
        (worker, repetition, framework, numba)
        for worker in workers
        for repetition in repetitions
        for framework in FRAMEWORKS
        for numba in NUMBA_MODES
    }
    for record in records:
        key = (
            int(record["worker_count"]),
            int(record["experiment_order"]["repetition"]),
            record["framework"],
            record["numba"],
        )
        if key in keys:
            raise ValueError(f"Duplicate matrix cell: {key}")
        keys.add(key)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"Incomplete matrix; missing={missing}, extra={extra}")

    invariant_fields = ("group_set_hash", "output_hash", "group_count", "precision")
    for field in invariant_fields:
        values = {record[field] for record in records}
        if len(values) != 1:
            raise ValueError(f"Records disagree on {field}: {values}")
    nodes = {record["system"]["hostname"] for record in records}
    if len(nodes) != 1:
        raise ValueError(f"Records were collected on multiple nodes: {nodes}")
    return workers, repetitions, next(iter(nodes))


def index_records(records: list[dict]) -> dict[tuple[int, int, str, str], dict]:
    return {
        (
            int(record["worker_count"]),
            int(record["experiment_order"]["repetition"]),
            record["framework"],
            record["numba"],
        ): record
        for record in records
    }


def build_tables(records: list[dict], workers: list[int], repetitions: list[int]):
    index = index_records(records)
    scaling_rows = []
    phase_rows = []
    framework_rows = []
    numba_rows = []

    for worker in workers:
        for framework in FRAMEWORKS:
            for numba in NUMBA_MODES:
                subset = [index[(worker, rep, framework, numba)] for rep in repetitions]
                processing = [float(item["phases"]["processing_s"]) for item in subset]
                throughput = [float(item["throughput_groups_per_hour"]) for item in subset]
                processing_mean, processing_sd = mean_sd(processing)
                throughput_mean, throughput_sd = mean_sd(throughput)
                scaling_rows.append(
                    {
                        "workers": worker,
                        "framework": framework,
                        "numba": numba,
                        "repetitions": len(subset),
                        "processing_mean_s": f"{processing_mean:.9f}",
                        "processing_sd_s": f"{processing_sd:.9f}",
                        "processing_cv_percent": f"{processing_sd / processing_mean * 100:.6f}",
                        "throughput_mean_groups_per_hour": f"{throughput_mean:.9f}",
                        "throughput_sd_groups_per_hour": f"{throughput_sd:.9f}",
                    }
                )
                phase_row = {
                    "workers": worker,
                    "framework": framework,
                    "numba": numba,
                    "repetitions": len(subset),
                }
                for phase in PHASE_STYLE:
                    values = [float(item["group_timing"][phase]["mean"]) for item in subset]
                    phase_mean, phase_sd = mean_sd(values)
                    phase_row[f"{phase}_mean"] = f"{phase_mean:.9f}"
                    phase_row[f"{phase}_sd"] = f"{phase_sd:.9f}"
                phase_rows.append(phase_row)

        for numba in NUMBA_MODES:
            differences = []
            for rep in repetitions:
                mpi_s = float(index[(worker, rep, "mpi", numba)]["phases"]["processing_s"])
                dask_s = float(index[(worker, rep, "dask", numba)]["phases"]["processing_s"])
                differences.append((dask_s / mpi_s - 1.0) * 100.0)
            mean, low, high = ci95(differences)
            framework_rows.append(
                {
                    "workers": worker,
                    "numba": numba,
                    "repetitions": len(differences),
                    "dask_minus_mpi_mean_percent": f"{mean:.9f}",
                    "ci95_low_percent": f"{low:.9f}",
                    "ci95_high_percent": f"{high:.9f}",
                }
            )

        for framework in FRAMEWORKS:
            benefits = []
            for rep in repetitions:
                off_s = float(index[(worker, rep, framework, "off")]["phases"]["processing_s"])
                on_s = float(index[(worker, rep, framework, "on")]["phases"]["processing_s"])
                benefits.append((off_s / on_s - 1.0) * 100.0)
            mean, low, high = ci95(benefits)
            numba_rows.append(
                {
                    "workers": worker,
                    "framework": framework,
                    "repetitions": len(benefits),
                    "numba_speedup_mean_percent": f"{mean:.9f}",
                    "ci95_low_percent": f"{low:.9f}",
                    "ci95_high_percent": f"{high:.9f}",
                }
            )
    return scaling_rows, framework_rows, numba_rows, phase_rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def mark_smt_region(axis, records: list[dict]) -> None:
    smt_workers = sorted(
        {
            int(record["worker_count"])
            for record in records
            if record.get("cpu_regime") == "smt"
        }
    )
    if not smt_workers:
        return
    lower = (35 + smt_workers[0]) / 2.0
    upper = max(smt_workers) + 1.5
    axis.axvspan(lower, upper, color="#999999", alpha=0.10, zorder=0)
    axis.text(
        0.985,
        0.04,
        "SMT",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#666666",
    )


def save_figure(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scaling(rows, workers, output_dir, subtitle, records):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for axis, numba in zip(axes, NUMBA_MODES):
        mark_smt_region(axis, records)
        for framework in FRAMEWORKS:
            subset = [
                row for row in rows
                if row["framework"] == framework and row["numba"] == numba
            ]
            style = FRAMEWORK_STYLE[framework]
            axis.errorbar(
                workers,
                [float(row["throughput_mean_groups_per_hour"]) for row in subset],
                yerr=[float(row["throughput_sd_groups_per_hour"]) for row in subset],
                color=style["color"], marker=style["marker"], label=style["label"],
                linewidth=1.7, markersize=4.5, capsize=3,
            )
        axis.set_title(f"Numba {numba}")
        axis.set_xlabel("Worker processes")
        axis.set_xticks(workers)
    axes[0].set_ylabel("Throughput (groups/hour)")
    axes[0].legend(frameon=False)
    fig.suptitle("Real-data processing throughput", y=1.04, fontsize=11)
    fig.text(0.5, -0.01, subtitle, ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout()
    save_figure(fig, output_dir, "real_framework_scaling")


def plot_framework_difference(rows, workers, output_dir, subtitle, records):
    fig, axis = plt.subplots(figsize=(5.4, 3.25))
    axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    mark_smt_region(axis, records)
    for numba in NUMBA_MODES:
        subset = [row for row in rows if row["numba"] == numba]
        means = [float(row["dask_minus_mpi_mean_percent"]) for row in subset]
        lows = [float(row["ci95_low_percent"]) for row in subset]
        highs = [float(row["ci95_high_percent"]) for row in subset]
        style = NUMBA_STYLE[numba]
        axis.errorbar(
            workers, means,
            yerr=[[mean - low for mean, low in zip(means, lows)],
                  [high - mean for mean, high in zip(means, highs)]],
            color=style["color"], marker=style["marker"], label=style["label"],
            linewidth=1.7, markersize=4.5, capsize=3,
        )
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Dask processing time relative to MPI (%)")
    axis.set_xticks(workers)
    axis.legend(frameon=False)
    axis.text(0.02, 0.96, "Dask slower", transform=axis.transAxes, va="top", color="#555555")
    axis.text(0.02, 0.04, "Dask faster", transform=axis.transAxes, va="bottom", color="#555555")
    fig.suptitle("Paired framework difference (95% CI)", y=1.02, fontsize=11)
    fig.text(0.5, -0.015, subtitle, ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout()
    save_figure(fig, output_dir, "real_framework_paired_difference")


def plot_numba_benefit(rows, workers, output_dir, subtitle, records):
    fig, axis = plt.subplots(figsize=(5.4, 3.25))
    axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    mark_smt_region(axis, records)
    for framework in FRAMEWORKS:
        subset = [row for row in rows if row["framework"] == framework]
        means = [float(row["numba_speedup_mean_percent"]) for row in subset]
        lows = [float(row["ci95_low_percent"]) for row in subset]
        highs = [float(row["ci95_high_percent"]) for row in subset]
        style = FRAMEWORK_STYLE[framework]
        axis.errorbar(
            workers, means,
            yerr=[[mean - low for mean, low in zip(means, lows)],
                  [high - mean for mean, high in zip(means, highs)]],
            color=style["color"], marker=style["marker"], label=style["label"],
            linewidth=1.7, markersize=4.5, capsize=3,
        )
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Processing-time improvement from Numba (%)")
    axis.set_xticks(workers)
    axis.legend(frameon=False)
    fig.suptitle("Numba benefit (paired repetitions, 95% CI)", y=1.02, fontsize=11)
    fig.text(0.5, -0.015, subtitle, ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout()
    save_figure(fig, output_dir, "real_numba_benefit")


def plot_phases(rows, workers, output_dir, subtitle, records):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), sharex=True, sharey=True)
    for row_index, framework in enumerate(FRAMEWORKS):
        for column_index, numba in enumerate(NUMBA_MODES):
            axis = axes[row_index, column_index]
            mark_smt_region(axis, records)
            subset = [
                row for row in rows
                if row["framework"] == framework and row["numba"] == numba
            ]
            for phase, style in PHASE_STYLE.items():
                axis.errorbar(
                    workers,
                    [float(row[f"{phase}_mean"]) for row in subset],
                    yerr=[float(row[f"{phase}_sd"]) for row in subset],
                    color=style["color"], marker=style["marker"], label=style["label"],
                    linewidth=1.5, markersize=4, capsize=2.5,
                )
            axis.set_title(f"{FRAMEWORK_STYLE[framework]['label']}, Numba {numba}")
            axis.set_xticks(workers)
    axes[0, 0].set_ylabel("Mean seconds per group")
    axes[1, 0].set_ylabel("Mean seconds per group")
    axes[1, 0].set_xlabel("Worker processes")
    axes[1, 1].set_xlabel("Worker processes")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Per-group phase inflation under concurrency", y=1.01, fontsize=11)
    fig.text(0.5, -0.005, subtitle, ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout()
    save_figure(fig, output_dir, "real_phase_scaling")


def main() -> int:
    args = parse_args()
    results_dirs = [path.resolve() for path in args.results_dirs]
    output_dir = (args.output_dir or results_dirs[0] / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(results_dirs)
    if args.max_workers is not None:
        records = [
            record
            for record in records
            if int(record["worker_count"]) <= args.max_workers
        ]
        if not records:
            raise ValueError(f"No records remain after --max-workers {args.max_workers}")
    workers, repetitions, node = validate_matrix(records)
    scaling, framework, numba, phases = build_tables(records, workers, repetitions)

    write_csv(
        output_dir / "real_scaling_summary.csv",
        list(scaling[0]), scaling,
    )
    write_csv(
        output_dir / "real_framework_paired_difference.csv",
        list(framework[0]), framework,
    )
    write_csv(
        output_dir / "real_numba_benefit.csv",
        list(numba[0]), numba,
    )
    write_csv(
        output_dir / "real_phase_summary.csv",
        list(phases[0]), phases,
    )

    configure_style()
    group_count = records[0]["group_count"]
    subtitle = (
        f"{len(repetitions)} repetitions; {group_count} groups/run; node {node}; "
        "error bars are sample SD unless marked 95% CI"
    )
    plot_scaling(scaling, workers, output_dir, subtitle, records)
    plot_framework_difference(framework, workers, output_dir, subtitle, records)
    plot_numba_benefit(numba, workers, output_dir, subtitle, records)
    plot_phases(phases, workers, output_dir, subtitle, records)

    print(f"Validated {len(records)} records on {node}.")
    print(f"Wrote 4 tables and 8 figure files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
