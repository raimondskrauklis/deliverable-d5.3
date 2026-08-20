#!/bin/bash
# Create one labelled local result directory and submit the mandatory pilot.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pilot_id="${PILOT_ID:-pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="${BENCH_OUTPUT_ROOT:-${PWD}/benchmark_results}"
output_dir="${OUTPUT_DIR:-${output_root}/${pilot_id}}"
mkdir -p "${output_dir}"

submit_args=(
    --parsable
    --output="${output_dir}/slurm-%x-%j.out"
    --error="${output_dir}/slurm-%x-%j.err"
    --export="ALL,PILOT_ID=${pilot_id},OUTPUT_DIR=${output_dir}"
)

# Forward optional sbatch arguments, for example --nodelist or --partition.
submitted="$(sbatch "${submit_args[@]}" "$@" "${script_dir}/pilot.sbatch")"
job_id="${submitted%%;*}"

echo "submitted pilot=${pilot_id} job=${job_id}"
echo "All Slurm logs and benchmark result files: ${output_dir}"
