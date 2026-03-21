"""Derivative helpers for one-dimensional signals."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gradient(
    values: ArrayLike, x: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Return the numerical gradient of a one-dimensional signal.

    Args:
        values: Signal values to differentiate.
        x: Optional x-axis coordinates. When omitted, unit spacing is assumed.

    Returns:
        Float64 NumPy array containing the numerical gradient.
    """

    y = np.asarray(values, dtype=np.float64)
    if x is None:
        return np.gradient(y).astype(np.float64)
    x_array = np.asarray(x, dtype=np.float64)
    return np.gradient(y, x_array).astype(np.float64)
