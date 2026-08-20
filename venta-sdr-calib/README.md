# Single-station calibration benchmark

This directory contains the scripts used to compare MPI and Dask execution of
the worker stage of a single-station spectral calibration pipeline. The same
worker implementation is used by both frameworks. The experiments also compare
Raw and compressed Zarr input and optional Numba acceleration.

The directory also contains two operational launchers for processing a complete
observation. Unlike the benchmark launchers, they continue through
system-temperature calibration and write the final flux-density products and
plots.

A scan group contains eight IQ files: four observing states (`s0`, `s1`, `r0`,
`r1`) and two polarization channels. Each worker reads or reconstructs the
integer IQ samples, converts them to complex values, forms overlapping segments,
and accumulates windowed FFT power spectra.

The main reported time is `processing_s`, measured from task dispatch until all
spectra have reached the coordinator. System-temperature calibration, plotting,
and final product writing are outside this interval.

## Python files

- `mpi_pipeline.py` -- complete MPI processing and calibration pipeline.
- `dask_pipeline.py` -- complete Dask processing and calibration pipeline.
- `pipeline_common.py` -- shared operational arguments, input validation, and
  conversion of group results into calibration states.
- `pipeline_output.py` -- shared log parsing, Tsys correction, flux calibration,
  DAT/JSON/NumPy output, and plot generation.
- `mpi_real.py` -- MPI launcher for real Raw or Zarr scan groups.
- `dask_real.py` -- Dask launcher for the same real-data workload.
- `mpi_synthetic.py` -- MPI launcher for deterministic in-memory data.
- `dask_synthetic.py` -- Dask launcher for the same synthetic workload.
- `workloads.py` -- group discovery, Raw/Zarr reading, IQ conversion, overlap
  construction, FFT processing, and synthetic-data generation.
- `mpi_runtime.py` -- MPI task distribution and result collection.
- `dask_runtime.py` -- Dask cluster setup, task distribution, worker placement,
  and result collection.
- `benchmark_cli.py` -- command-line arguments shared by the launchers.
- `launcher_setup.py` -- limits numerical libraries to one thread before NumPy
  and SciPy are imported.
- `reporting.py` -- JSON and CSV output, timing summaries, configuration
  metadata, and correctness hashes.
- `compare_results.py` -- checks matched result files for identical inputs and
  numerical output.
- `validate_cpu_affinity.py` -- checks recorded MPI and Dask processor
  placement.
- `prepare_zarr.py` -- creates resumable and exactly verified Zarr ZipStore
  copies of complete Raw scan groups.
- `compression_scaling.py` -- measures group-level and Blosc-thread compression
  scaling using temporary verified archives.
- `plot_real_results.py` -- creates figures and source tables for the real-data
  MPI/Dask and Numba experiments.
- `plot_article_primary.py` -- creates the Raw/Zarr by MPI/Dask scaling figure.
- `requirements-zarr.txt` -- pins the Zarr 2 and numcodecs versions required by
  the archive format used in the experiments.

## Processing a complete observation

Both operational launchers require an SDR experiment log, an input directory,
and a new or empty output directory. The input must contain every complete scan
group named in the log. Raw inputs and verified Zarr archives are supported.
They use the project's `ExperimentsLogReader` package plus NumPy, SciPy,
Matplotlib, PyWavelets, Numba, and either mpi4py or Dask Distributed.

MPI uses one coordinator rank in addition to the requested workers. This
example starts 32 workers:

```bash
mpirun -np 33 --bind-to core --map-by core \
  python -u mpi_pipeline.py OBSERVATION.log RAW_DIRECTORY OUTPUT_DIRECTORY \
  --expected-workers 32 --input-format raw --numba on
```

Dask runs a local scheduler and 32 single-threaded worker processes:

```bash
python -u dask_pipeline.py OBSERVATION.log RAW_DIRECTORY OUTPUT_DIRECTORY \
  --workers 32 --input-format raw --numba on
```

On a Slurm node, launch the Dask command in a step with one CPU for the
controller plus one CPU per worker. If the step exposes exactly those physical
cores, add `--worker-affinity distinct-cores` to assign each process to its own
core. For Zarr, replace the input directory and add `--input-format zarr`; the
directory must contain the matching `.zarr.zip` files and verification markers.

The output directory receives per-scan LCP/RCP arrays, averaged DAT and JSON
spectra, raw-spectrum and Tsys diagnostics, calibrated-spectrum plots, a parsed
copy of the observation log, and a calibration run log. The scripts refuse to
write inside the input directory or into a non-empty output directory.

## Slurm files

The files under `slurm/` are configured for the DAS-6 Skylake nodes. Site
paths, modules, and the preferred node are defined in `cluster_env.sh` and can
be overridden with environment variables.

- `cluster_env.sh` -- DAS-6 paths, modules, benchmark defaults, and node choice.
- `common.sh` -- shared environment setup, timeouts, Dask retries, CPU binding,
  launch functions, and result checks.
- `pilot.sbatch` / `submit_pilot.sh` -- small Raw MPI/Dask correctness pilot.
- `real_matrix.sbatch` / `submit_matrix.sh` -- original Raw real-data or
  synthetic worker-count matrices.
- `synthetic_matrix.sbatch` / `submit_synthetic_overnight.sh` -- synthetic
  scaling matrix and full-length compute pilot.
- `real_extension.sbatch` / `submit_extension.sh` -- physical-core endpoint and
  simultaneous-multithreading extension.
- `submit_extension_pilot.sh` -- small affinity check before the extension.
- `real_high_smt_numba_on.sbatch` / `submit_high_smt_numba_on.sh` -- paired
  high-worker Numba-on follow-up.
- `synthetic_w24_affinity.sbatch` / `submit_synthetic_w24_affinity.sh` --
  targeted Dask worker-placement experiment at 24 workers.
- `zarr_prepack.sbatch` / `submit_zarr_prepack.sh` -- resumable Raw-to-Zarr
  conversion and verification.
- `zarr_compare_pilot.sbatch` / `submit_zarr_compare_pilot.sh` -- matched
  Raw/Zarr and MPI/Dask correctness pilot.
- `zarr_compare_matrix.sbatch` / `submit_zarr_compare_matrix.sh` -- primary
  Raw/Zarr by MPI/Dask scaling matrix.
- `zarr_compare_extension.sbatch` / `submit_zarr_compare_extension.sh` --
  35-worker endpoint and 48/64/71-worker SMT extension for both formats.
- `submit_zarr_compare_extension_pilot.sh` -- small physical-core and SMT
  affinity pilot.
- `compression_scaling.sbatch` / `submit_compression_scaling.sh` -- verified
  Raw-to-Zarr compression scaling matrix.

## Running on DAS-6

Edit or override the values in `slurm/cluster_env.sh`, then source it:

```bash
source slurm/cluster_env.sh
```

The principal Raw/Zarr experiment is run in three stages:

```bash
bash "$BENCH_DIR/slurm/submit_zarr_prepack.sh"
bash "$BENCH_DIR/slurm/submit_zarr_compare_pilot.sh"
bash "$BENCH_DIR/slurm/submit_zarr_compare_matrix.sh"
```

The first command converts four groups by default. Set `ZARR_GROUPS=0` to
resume and convert all available groups after checking the pilot. The comparison
pilot must pass before submitting the full matrix.

Every benchmark condition writes a JSON summary and a per-group CSV. Output
files use unique run identifiers and are not overwritten. Raw inputs are opened
read-only. Zarr preparation writes partial archives first and only promotes an
archive after exact verification. The compression-scaling experiment deletes
only temporary condition directories carrying its matching ownership marker.
