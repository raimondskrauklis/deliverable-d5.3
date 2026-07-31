# Deliverable D5.3

**RADIOBLOCKS Deliverable 5.3 — Prototype processing workflow functionality using software modules under the DASK framework**

This repository hosts the software components associated with Deliverable 5.3 of the [RADIOBLOCKS](https://www.radioblocks.eu/) project, reporting on the progress achieved in Work Package 5 (WP5), Task 5.1. It contains software modules and a DASK-based prototype workflow that are available and ready for further integration and development within the WP5 processing framework.

## Contents

The repository aggregates individual software components as Git submodules. Each component is developed and maintained by the project partners.

Current submodules:

- [`fastpredict`](fastpredict/) — A LOFAR package for predicting visibilities from sky model components using direct prediction methods.

## Getting started

Clone the repository including its submodules:

```bash
git clone --recurse-submodules <repository-url>
```

To update the submodules after a clone:

```bash
git submodule update --init --recursive
```

Refer to each submodule's own README for build and usage instructions.

## License

Individual submodules are released under their own licenses (see each submodule's repository for details).
