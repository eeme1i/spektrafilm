"""Working precision for image-sized arrays.

The runtime pipeline carries images as float32: it halves memory use and
memory traffic, which dominates most per-pixel stages. Small arrays
(spectra, curves, LUTs, matrices) stay float64. Kernels therefore keep a
float32 input as float32 and promote anything else to float64.
"""

import numpy as np

IMAGE_DTYPE = np.float32


def as_float_array(x) -> np.ndarray:
    """Return ``x`` as a float32 or float64 array, avoiding copies.

    float32 input is kept as float32; every other input (float64, ints,
    lists, scalars) becomes float64.
    """
    x = np.asarray(x)
    if x.dtype == np.float32:
        return x
    return x.astype(np.float64, copy=False)
