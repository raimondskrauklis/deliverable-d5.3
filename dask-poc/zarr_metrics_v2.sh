#!/usr/bin/env bash
set -Eeuo pipefail

ZARR_PATH="${1:?Usage: $0 <dataset.zarr> [array_name]}"
ARRAY_NAME="${2:-VISIBILITY}"

eval "$(
python - "$ZARR_PATH" "$ARRAY_NAME" <<'PY'
import sys
import shlex
import math
import zarr

zarr_path = sys.argv[1]
array_name = sys.argv[2]

root = zarr.open_group(zarr_path, mode="r")
arr = root[array_name]

shape = tuple(int(x) for x in arr.shape)
chunks = tuple(int(x) for x in arr.chunks)
chunks_per_dim = tuple(
    int(math.ceil(s / c))
    for s, c in zip(shape, chunks)
)

n_chunks_total = math.prod(chunks_per_dim)

print(f"ZARR_PATH={shlex.quote(zarr_path)}")
print(f"ARRAY_NAME={shlex.quote(array_name)}")
print(f"ZARR_NDIM={len(shape)}")
print(f"ZARR_DTYPE={shlex.quote(str(arr.dtype))}")

for i, value in enumerate(shape):
    print(f"ZARR_SHAPE_{i}={value}")

for i, value in enumerate(chunks):
    print(f"ZARR_CHUNK_{i}={value}")

for i, value in enumerate(chunks_per_dim):
    print(f"ZARR_NCHUNKS_{i}={value}")

print(f"ZARR_NCHUNKS_TOTAL={n_chunks_total}")

print(f"ZARR_SHAPE_CSV={shlex.quote(','.join(map(str, shape)))}")
print(f"ZARR_CHUNKS_CSV={shlex.quote(','.join(map(str, chunks)))}")
print(f"ZARR_NCHUNKS_CSV={shlex.quote(','.join(map(str, chunks_per_dim)))}")
PY
)"

echo "array          : $ARRAY_NAME"
echo "dtype          : $ZARR_DTYPE"
echo "ndim           : $ZARR_NDIM"
echo "shape          : $ZARR_SHAPE_CSV"
echo "chunks         : $ZARR_CHUNKS_CSV"
echo "chunks per dim : $ZARR_NCHUNKS_CSV"
echo "total chunks   : $ZARR_NCHUNKS_TOTAL"

# Example access:
echo "axis 0 size    : $ZARR_SHAPE_0"
echo "axis 0 chunk   : $ZARR_CHUNK_0"
echo "axis 0 nchunks : $ZARR_NCHUNKS_0"
