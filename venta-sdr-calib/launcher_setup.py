"""Process-level settings that must be applied before importing NumPy/SciPy."""

import os


MATH_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def limit_math_libraries_to_one_thread():
    """Prevent hidden FFT/BLAS threads from oversubscribing worker processes."""
    for variable in MATH_THREAD_VARIABLES:
        os.environ[variable] = "1"
