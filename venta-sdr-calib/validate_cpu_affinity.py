#!/usr/bin/env python3
"""Validate recorded MPI and Dask affinity for a matched Slurm block."""

import argparse
import json
from pathlib import Path


def load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--cpu-regime", choices=("physical-core", "smt"), required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    args = parser.parse_args()

    failures = []
    for path in args.records:
        record = load(path)
        framework = record.get("framework")
        workers = int(record.get("worker_count", -1))
        if workers != args.expected_workers:
            failures.append(f"{path}: worker_count={workers}")
        if record.get("cpu_regime") != args.cpu_regime:
            failures.append(
                f"{path}: cpu_regime={record.get('cpu_regime')!r}, "
                f"expected {args.cpu_regime!r}"
            )
        prewarm = record.get("prewarm_workers", {})

        if framework == "mpi":
            if len(prewarm) != workers + 1:
                failures.append(
                    f"{path}: MPI prewarm has {len(prewarm)} ranks, expected {workers + 1}"
                )
                continue
            affinities = [tuple(item.get("cpu_affinity", [])) for item in prewarm]
            if args.cpu_regime == "smt":
                if any(len(affinity) != 1 for affinity in affinities):
                    failures.append(f"{path}: SMT MPI ranks are not bound one per hardware thread")
                bound_cpus = {affinity[0] for affinity in affinities if len(affinity) == 1}
                if len(bound_cpus) != workers + 1:
                    failures.append(
                        f"{path}: MPI uses {len(bound_cpus)} unique hardware threads, "
                        f"expected {workers + 1}"
                    )
            elif len(set(affinities)) != workers + 1:
                failures.append(f"{path}: physical-core MPI rank masks overlap")

        elif framework == "dask":
            values = list(prewarm.values()) if isinstance(prewarm, dict) else prewarm
            if len(values) != workers:
                failures.append(
                    f"{path}: Dask prewarm has {len(values)} workers, expected {workers}"
                )
                continue
            affinity_mode = record.get("worker_affinity_mode", "shared")
            affinities = [tuple(item.get("cpu_affinity", [])) for item in values]
            if affinity_mode == "shared":
                affinity_sizes = {len(affinity) for affinity in affinities}
                if not affinity_sizes or min(affinity_sizes) < workers + 1:
                    failures.append(
                        f"{path}: shared Dask worker affinity sizes "
                        f"{sorted(affinity_sizes)} do not cover at least "
                        f"{workers + 1} CPUs"
                    )
            elif affinity_mode in ("distinct-cores", "distinct-cpus"):
                if any(not affinity for affinity in affinities):
                    failures.append(f"{path}: a pinned Dask worker has empty affinity")
                if affinity_mode == "distinct-cpus" and any(
                    len(affinity) != 1 for affinity in affinities
                ):
                    failures.append(
                        f"{path}: distinct-cpus workers are not bound one per logical CPU"
                    )
                if len(set(affinities)) != workers:
                    failures.append(
                        f"{path}: pinned Dask workers do not have {workers} unique masks"
                    )
                occupied = set()
                for affinity in affinities:
                    overlap = occupied.intersection(affinity)
                    if overlap:
                        failures.append(
                            f"{path}: pinned Dask worker masks overlap on CPUs "
                            f"{sorted(overlap)}"
                        )
                        break
                    occupied.update(affinity)
                controller = set(record.get("controller_cpu_affinity", []))
                if not controller:
                    failures.append(f"{path}: pinned Dask controller mask is empty")
                elif affinity_mode == "distinct-cpus" and len(controller) != 1:
                    failures.append(
                        f"{path}: distinct-cpus controller is not bound to one logical CPU"
                    )
                elif controller.intersection(occupied):
                    failures.append(
                        f"{path}: Dask controller overlaps worker CPUs "
                        f"{sorted(controller.intersection(occupied))}"
                    )
            else:
                failures.append(
                    f"{path}: unknown Dask worker_affinity_mode={affinity_mode!r}"
                )
        else:
            failures.append(f"{path}: unknown framework {framework!r}")

    if failures:
        print("[FAIL] CPU affinity validation")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"[PASS] {args.cpu_regime} affinity for {len(args.records)} records "
        f"at {args.expected_workers} workers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
