#!/usr/bin/env python3
"""Check that matched benchmark JSON files produced identical scientific output."""

import argparse
import json
from pathlib import Path


MATCHED_FIELDS = (
    "workload",
    "group_count",
    "group_set_hash",
    "fft_size",
    "fft_backend",
    "precision",
    "cpu_regime",
    "result_dtypes",
    "result_payload_bytes",
)


def load_record(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(
        description="Compare output hashes and scientific configuration of matched runs"
    )
    parser.add_argument("records", nargs="+", help="Two or more benchmark JSON files")
    args = parser.parse_args()

    if len(args.records) < 2:
        parser.error("provide at least two JSON records")

    loaded = [(path, load_record(path)) for path in args.records]
    reference_path, reference = loaded[0]
    failures = []

    for path, record in loaded[1:]:
        for field in MATCHED_FIELDS:
            if record.get(field) != reference.get(field):
                failures.append(
                    "%s: %s differs (%r != %r)"
                    % (path, field, record.get(field), reference.get(field))
                )
        if record.get("output_hash") != reference.get("output_hash"):
            failures.append("%s: output_hash differs from %s" % (path, reference_path))

    if failures:
        print("[FAIL] matched benchmark outputs are not equivalent")
        for failure in failures:
            print("  - %s" % failure)
        return 1

    print(
        "[PASS] %d matched records have identical configuration and output hash"
        % len(loaded)
    )
    print("[PASS] output_hash=%s" % reference["output_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
