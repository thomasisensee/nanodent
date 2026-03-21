"""Filtering helpers for one-dimensional signals."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter


def savgol(
    values: ArrayLike,
    *,
    window_length: int = 31,
    polyorder: int = 3,
    mode: str = "interp",
) -> NDArray[np.float64]:
    """Apply a Savitzky-Golay filter with window coercion for short signals.

    Args:
        values: One-dimensional signal to smooth.
        window_length: Preferred smoothing window length.
        polyorder: Polynomial order used by the filter.
        mode: Edge-handling mode forwarded to SciPy.

    Returns:
        Smoothed float64 NumPy array with the same shape as the input.
    """

    array = np.asarray(values, dtype=np.float64)
    adjusted_window = _coerce_window_length(
        len(array), window_length, polyorder
    )
    if adjusted_window <= polyorder:
        return array.copy()
    return savgol_filter(
        array, window_length=adjusted_window, polyorder=polyorder, mode=mode
    )


def _coerce_window_length(
    length: int, window_length: int, polyorder: int
) -> int:
    """Adjust the requested window length to a valid odd value.

    Args:
        length: Length of the input signal.
        window_length: Requested smoothing window length.
        polyorder: Polynomial order used by the filter.

    Returns:
        Window length accepted by `scipy.signal.savgol_filter`.
    """

    if length <= polyorder + 1:
        return length
    adjusted = min(window_length, length)
    if adjusted % 2 == 0:
        adjusted -= 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    if adjusted < minimum:
        adjusted = minimum
    if adjusted > length:
        adjusted = length if length % 2 == 1 else length - 1
    return max(adjusted, polyorder + 1)
