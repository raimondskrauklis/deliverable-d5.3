#!/usr/bin/env bash
set -Eeuo pipefail

ZARR_PATH="${1:?Usage: $0 <dataset.zarr> [array_name]}"
ARRAY_NAME="${2:-VISIBILITY}"

ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT

echo
echo "Checking version of Zarr:"
python - <<'PY'
import zarr
print(zarr.__version__)
print(zarr.__file__)
PY

echo

python - "$ZARR_PATH" "$ARRAY_NAME" "$ENV_FILE" <<'PY'
import sys
import shlex
import math
import zarr

zarr_path = sys.argv[1]
array_name = sys.argv[2]
env_file = sys.argv[3]

root = zarr.open_group(zarr_path, mode="r")

try:
    arr = root[array_name]
except Exception as e:
    raise RuntimeError(
        f"Cannot find array {array_name!r} in {zarr_path!r}. "
        f"Available keys: {list(root.keys())}"
    ) from e

if not hasattr(arr, "shape"):
    raise RuntimeError(
        f"{array_name!r} exists but is not an array. "
        f"It is probably a group. Available children: {list(arr.keys())}"
    )

shape = tuple(int(x) for x in arr.shape)
dtype = str(arr.dtype)

chunks = getattr(arr, "chunks", None)

if chunks is None:
    meta = arr.metadata
    chunk_grid = meta.chunk_grid
    config = chunk_grid.configuration

    if isinstance(config, dict):
        chunks = config["chunk_shape"]
    else:
        chunks = config.chunk_shape

chunks = tuple(int(x) for x in chunks)

if len(shape) != len(chunks):
    raise RuntimeError(f"shape/chunks rank mismatch: shape={shape}, chunks={chunks}")

chunks_per_dim = tuple(
    int(math.ceil(s / c))
    for s, c in zip(shape, chunks)
)

n_chunks_total = math.prod(chunks_per_dim)

meta = getattr(arr, "metadata", None)
zarr_format = getattr(meta, "zarr_format", None)

if zarr_format is None:
    # Zarr v3 arrays have chunk_grid metadata.
    zarr_format = "3" if hasattr(meta, "chunk_grid") else "unknown"

lines = []

def emit(name, value):
    lines.append(f"{name}={shlex.quote(str(value))}")

emit("ZARR_PATH", zarr_path)
emit("ARRAY_NAME", array_name)
emit("ZARR_FORMAT", zarr_format)
emit("ZARR_NDIM", len(shape))
emit("ZARR_DTYPE", dtype)

for i, value in enumerate(shape):
    emit(f"ZARR_SHAPE_{i}", value)

for i, value in enumerate(chunks):
    emit(f"ZARR_CHUNK_{i}", value)

for i, value in enumerate(chunks_per_dim):
    emit(f"ZARR_NCHUNKS_{i}", value)

emit("ZARR_NCHUNKS_TOTAL", n_chunks_total)
emit("ZARR_SHAPE_CSV", ",".join(map(str, shape)))
emit("ZARR_CHUNKS_CSV", ",".join(map(str, chunks)))
emit("ZARR_NCHUNKS_CSV", ",".join(map(str, chunks_per_dim)))

with open(env_file, "w") as f:
    f.write("\n".join(lines))
    f.write("\n")
PY

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${ZARR_DTYPE:?missing ZARR_DTYPE}"
: "${ZARR_NDIM:?missing ZARR_NDIM}"
: "${ZARR_SHAPE_CSV:?missing ZARR_SHAPE_CSV}"
: "${ZARR_CHUNKS_CSV:?missing ZARR_CHUNKS_CSV}"

echo "array          : $ARRAY_NAME"
echo "zarr format    : $ZARR_FORMAT"
echo "dtype          : $ZARR_DTYPE"
echo "ndim           : $ZARR_NDIM"
echo "shape          : $ZARR_SHAPE_CSV"
echo "chunks         : $ZARR_CHUNKS_CSV"
echo "chunks per dim : $ZARR_NCHUNKS_CSV"
echo "total chunks   : $ZARR_NCHUNKS_TOTAL"

echo "axis 0 size    : $ZARR_SHAPE_0"
echo "axis 0 chunk   : $ZARR_CHUNK_0"
echo "axis 0 nchunks : $ZARR_NCHUNKS_0"
echo

