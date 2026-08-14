# Running AOFlagger + IDG on a Dask-CUDA 26.04 cluster 

This describes how to run a Dask-CUDA 26.04 cluster on SLURM: start the
scheduler, start workers, then run the client, which runs AOFlagger
(CPU RFI flagging) and IDG (GPU gridding) on a LOFAR MSv4 dataset. It covers
`submit_scheduler.sh` / `submit_workers.sh` / `submit_aoflagger_idg_on_msv4_lofar.sh`
(the recommended entry points) and
the underlying SLURM job scripts they wrap. Order matters — the
scheduler must be running before workers can connect, and workers must
be running before the client has anything to submit work to.

**NOTE**: some of the scripts needs tuning to fit the user installation and HPC
cluster setup. Provided scripts reflects what was used on Kuma, a GPU cluster from
EPFL.

## Files involved

| File | Role |
|---|---|
| `submit_scheduler.sh` | Submission wrapper for the scheduler. Resolves the SLURM log path and calls `sbatch`. **Use this to launch the scheduler.** |
| `submit_workers.sh` | Submission wrapper for workers. Same as above, plus requires `--ntasks`. **Use this to launch workers.** |
| `launch_dask-cuda_26.04_scheduler.sh` | The scheduler's SLURM batch script (job body). Not normally invoked directly. |
| `launch_dask-cuda_26.04_workers.sh` | The workers' SLURM batch script (job body). Not normally invoked directly. |
| `set_dask-cuda_26.04_environment.sh` | Thin entry point for the default (Zarr 2) environment. Sourced automatically by both launch scripts. |
| `set_dask-cuda_26.04_zarr_3_environment.sh` | Thin entry point for the Zarr 3 environment. Sourced automatically when `--zarr3` is passed. |
| `set_dask-cuda_26.04_common_environment.sh` | Shared implementation both entry points delegate to (Spack/venv activation, UCX tuning, RAPIDS settings, `SCHEDULER_DIRECTORY`/`SCHEDULER_FILE`). Not sourced directly — edit this file, not the two entry points, to change actual behavior. |
| `test_submit_scheduler.sh` | Smoke test: submits a real scheduler job with known-good values baked in. |
| `test_submit_workers.sh` | Smoke test: submits real worker jobs with known-good values baked in (same `DASKPOC_SCHEDULER_DIR` as the scheduler test). |
| `submit_aoflagger_idg_on_msv4_lofar.sh` | Client driver. Probes the running cluster's worker count, then submits a chain of dependent AOFlagger/IDG SLURM jobs sweeping over MSv4/LOFAR Zarr chunking and compression combinations. |
| `aoflagger_idg_on_msv4_lofar.sh` | The AOFlagger/IDG SLURM batch script (job body) submitted per combination by the driver above. Not normally invoked directly. |
| `aoflagger_idg_on_msv4.py` | Python client: connects to the scheduler via `--scheduler-file`, runs AOFlagger (CPU RFI flagging) and IDG (GPU gridding) on the MSv4 data, distributed across the connected workers. |
| `zarr_metrics_v2.sh` / `zarr_3_metrics.sh` | Read shape/chunking metadata off an on-disk Zarr array (Zarr 2 / Zarr 3), used by the driver to size the processing chunks per run. |
| `daskpoc.env.example` | Template for `daskpoc.env`, the config file capturing a whole run's configuration in one place — see Configuration below. |

## Configuration (`daskpoc.env`)

All the `DASKPOC_*` variables below are read from `daskpoc.env` if it
exists next to the scripts — `submit_scheduler.sh`, `submit_workers.sh`,
and `submit_aoflagger_idg_on_msv4_lofar.sh` all source it automatically,
before checking anything is set. Every assignment in `daskpoc.env` uses
`${VAR:-default}` rather than a plain `export`, so it only fills in
variables that aren't already set — a plain shell `export` done *before*
calling one of the `submit_*.sh` scripts still wins, for a one-off
override without editing the file (this is how the `test_submit_*.sh`
smoke tests pin their own known-good values regardless of what's in
`daskpoc.env`).

```bash
cp daskpoc.env.example daskpoc.env
# edit daskpoc.env with your paths, then:
./submit_scheduler.sh
./submit_workers.sh
./submit_aoflagger_idg_on_msv4_lofar.sh
```

Because it's one file, `daskpoc.env` is also a convenient record of what
a given run actually used — copy it alongside a run's output/logs if you
want to be able to reconstruct later exactly which paths and flavor
produced it.

| Variable | Required by | Meaning |
|---|---|---|
| `DASKPOC_ZARR3` | all three | `1` = Zarr 3, `0`/unset = Zarr 2. Equivalent to always passing `--zarr3`; an explicit `--zarr3` on the command line still works (adds to this, doesn't turn it back off). |
| `DASKPOC_INSTALL_DIR` | all three (**required**) | Venv install directory for the chosen flavor. |
| `DASKPOC_SCHEDULER_DIR` | all three (**required**) | Scheduler state directory; must be the same value for scheduler, workers, and client to find each other. |
| `DASKPOC_DASK_LOGS_DIR` | scheduler, workers | SLURM job log directory (default: `.`). |
| `DASKPOC_NTASKS` | workers | Default for `--ntasks` (still overridable on the command line). |
| `DASKPOC_NTASKS_PER_NODE` | workers | Default for `--ntasks-per-node` (default: `1`). |
| `DASKPOC_OUTPUT_DIR` | client (**required**) | Where AOFlagger/IDG results get written. |
| `DASKPOC_DATA_REPO` | client (**required**) | Base directory holding the MSv4/LOFAR Zarr datasets (read-only input). |
| `DASKPOC_DATA_PROC` | client (**required**) | Scratch base directory to copy/process datasets into; must already exist. |

`daskpoc.env` is gitignored (see `.gitignore` in this directory) since it
holds personal paths — only `daskpoc.env.example` is meant to be
committed.

## Environment scripts

`set_dask-cuda_26.04_environment.sh` (default, Zarr 2) and
`set_dask-cuda_26.04_zarr_3_environment.sh` (Zarr 3) are **sourced**, not
executed, by the launch script, so everything they `export` becomes part
of the job's environment. Each one only sets `FLAVOR` (`zarr2`/`zarr3`,
used for logging) for its flavor, then sources
`set_dask-cuda_26.04_common_environment.sh`, which holds the actual setup
logic once, shared by both flavors. This is what to edit for any change
that should apply to both (UCX tuning, RAPIDS settings, Spack activation,
etc.); the two entry points should stay minimal.

The common script **requires** two environment variables to already be
set — no defaults, no auto-detected paths:

- `DASKPOC_INSTALL_DIR` — path to the venv to activate for this flavor
  (e.g. a Zarr 2 vs. Zarr 3 install are different directories, so this
  must be set to the right one per flavor before submission).
- `DASKPOC_SCHEDULER_DIR` — directory for scheduler state; exported
  onward as `SCHEDULER_DIRECTORY`/`SCHEDULER_FILE`
  (`$DASKPOC_SCHEDULER_DIR/dask-scheduler.json`) for the launch/worker
  scripts to consume. **Intentionally not flavor-specific** — Zarr 2 and
  Zarr 3 jobs share whatever single directory you set here (see Known
  limitations below).

Set both via `daskpoc.env` (see Configuration above — the recommended
way, since it's one file used by every script) or `export` them by hand
before calling `submit_scheduler.sh` / `submit_workers.sh` — **the same
values for scheduler and workers**, so they agree on `SCHEDULER_FILE`:

```bash
export DASKPOC_INSTALL_DIR=/work/scitas-ge/$USER/KUMA_VENVS/DASK-CUDA_26.04
export DASKPOC_SCHEDULER_DIR=/scratch/$USER/dask-cuda_26.04_scheduler_directory
./submit_scheduler.sh
./submit_workers.sh --ntasks 2
```

If either is unset, the job fails immediately with a clear
`DASKPOC_INSTALL_DIR must be set ...` / `DASKPOC_SCHEDULER_DIR must be
set ...` error rather than silently falling back to someone else's path.
`submit_scheduler.sh`/`submit_workers.sh` also check this upfront, before
even calling `sbatch`.

## Launching the scheduler

```bash
./submit_scheduler.sh            # default (Zarr 2) environment
```

This submits `launch_dask-cuda_26.04_scheduler.sh` (which needs to be adapted to
the user environment, see header) via `sbatch`.

What the job does:

1. Activates the chosen environment, which exports
   `SCHEDULER_DIRECTORY` and `SCHEDULER_FILE`.
2. Ensures `SCHEDULER_DIRECTORY` exists (`mkdir -p`) — it does **not**
   delete any pre-existing contents. `dask scheduler` overwrites
   `SCHEDULER_FILE` itself on startup, so no cleanup step is needed
   (and running two schedulers against the same directory at once isn't
   meaningful anyway — the fixed path only makes sense as one instance
   at a time).
3. Starts `dask scheduler` over UCX on the `bond1` interface, writing
   connection info to `SCHEDULER_FILE` for workers to pick up.

## Launching workers

```bash
./submit_workers.sh --ntasks 2                      # 2 GPU workers, default (Zarr 2)
./submit_workers.sh --ntasks 2 --ntasks-per-node 1   # 1 worker/node
```

Unlike the scheduler, worker count isn't fixed in
`launch_dask-cuda_26.04_workers.sh`'s own `#SBATCH` header, so
`--ntasks` is **required** (`--ntasks-per-node` defaults to `1`).
`submit_workers.sh` forwards both to `sbatch`. Note that `launch_dask-cuda_26.04_workers.sh`
also needs to be adapted to the user environment, see its header.

The launch script checks `SCHEDULER_FILE` exists before starting workers and fails
if it doesn't (i.e. if the scheduler isn't running yet).


## Running the client (AOFlagger + IDG on LOFAR MSv4)

With a scheduler and workers already running (see above) and
`daskpoc.env` filled in (see Configuration above — this script needs
`DASKPOC_INSTALL_DIR`/`DASKPOC_SCHEDULER_DIR` set to the **same values**
used for them, plus its own `DASKPOC_OUTPUT_DIR`/`DASKPOC_DATA_REPO`/
`DASKPOC_DATA_PROC`, all required, no defaults):

```bash
./submit_aoflagger_idg_on_msv4_lofar.sh
```

(`DASKPOC_DATA_REPO`/`DASKPOC_DATA_PROC` get `/${TELESCOPE}` appended
automatically.)

This script is used to submit a single job or a series of jobs. It to probe
the cluster's worker count via a short `srun` +
`distributed.Client`) and then submits a **chain of dependent SLURM
jobs** (`--dependency=afterok:...`), one per (compression, chunk count)
combination, each running `aoflagger_idg_on_msv4_lofar.sh`.

For each combination, that job script:

1. Copies the MSv4 Zarr dataset from `DASKPOC_DATA_REPO` to
   `DASKPOC_DATA_PROC` if not already there.
2. Builds the AOFlagger/IDG command line and runs
   `aoflagger_idg_on_msv4.py --scheduler-file $SCHEDULER_FILE ...`, which
   connects to the running scheduler and distributes 
   the workload across the connected workers.
3. Writes output/logs under a `--chdir`-ed, per-combination directory
   inside `DASKPOC_OUTPUT_DIR` (`--output slurm-%j_<tag>.out`, not
   `DASKPOC_DASK_LOGS_DIR` — that one's scheduler/workers-only).

By default, `aoflagger_idg_on_msv4.py` sizes each chunk's AOFlagger thread
pool from `SLURM_CPUS_PER_TASK` on the worker that processes it, so it
automatically matches whatever `--cpus-per-task` the workers job was
started with. Pass `--n_threads N` on its command line to override this
for a given run — e.g. to sweep 1, 2, 4, ... up to `SLURM_CPUS_PER_TASK`
and measure how AOFlagger performance scales with thread count.

`submit_aoflagger_idg_on_msv4_lofar.sh` has a matching `N_THREADS` sweep
loop (innermost, alongside the existing `COMPRESSION`/`NCHUNKS`/`WSC_SIZE`
loops):

```bash
for N_THREADS in ''; do
```

The default single empty value (`''`) submits one job per combination
with `--n_threads` omitted, i.e. the auto-`SLURM_CPUS_PER_TASK` behavior.
Edit it to a list of values, e.g. `'' 1 2 4 8 16`, to submit one job per
thread count as part of the same dependency chain — each job's
`--n_threads` is passed through only when non-empty, and `OUT_TAG` gets a
`_nt<value|auto>` suffix so sweep runs land in distinctly-named output
directories.

## SLURM log output

Both wrappers write the job's SLURM log under `DASKPOC_DASK_LOGS_DIR`:

```
$DASKPOC_DASK_LOGS_DIR/slurm-%j_scheduler.out   # submit_scheduler.sh
$DASKPOC_DASK_LOGS_DIR/slurm-%j_workers.out     # submit_workers.sh
```

`DASKPOC_DASK_LOGS_DIR` defaults to `.` (the directory you run the wrapper
from) if not set, and is created automatically if it doesn't exist. To
send logs elsewhere:

```bash
DASKPOC_DASK_LOGS_DIR=/scratch/$USER/dask-cuda-logs ./submit_scheduler.sh
```

`#SBATCH` directives can't read shell/environment variables directly,
which is why this is handled by the wrappers (passing `--output` to
`sbatch` on the command line) rather than hardcoded in the launch scripts.
