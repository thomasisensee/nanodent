"""Alignment helpers for onset detection and x-axis shifting."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol

AlignmentMethod = Literal["none", "force_threshold", "savgol_force_slope"]


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Result of an alignment calculation."""

    method: AlignmentMethod
    anchor_index: int
    anchor_x: float
    shifted_x: NDArray[np.float64]


def align_curve(
    x: ArrayLike,
    y: ArrayLike,
    *,
    method: AlignmentMethod = "none",
    force_threshold: float = 10.0,
    slope_threshold: float | None = None,
    window_length: int = 31,
    polyorder: int = 3,
) -> AlignmentResult:
    """Align a curve by returning its shifted x-values and chosen anchor.

    Args:
        x: Original x-axis values.
        y: Signal values used to detect the onset anchor.
        method: Alignment strategy to apply.
        force_threshold: Force threshold used by `force_threshold`.
        slope_threshold: Optional explicit slope threshold for
            `savgol_force_slope`.
        window_length: Savitzky-Golay window length for slope-based alignment.
        polyorder: Savitzky-Golay polynomial order for slope-based alignment.

    Returns:
        Alignment result containing the detected anchor and shifted x-values.
    """

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape for alignment.")
    if x_array.ndim != 1:
        raise ValueError("align_curve only supports one-dimensional signals.")

    match method:
        case "none":
            anchor_index = 0
            shifted_x = x_array.copy()
        case "force_threshold":
            anchor_index = _first_index(y_array >= force_threshold)
            shifted_x = x_array - x_array[anchor_index]
        case "savgol_force_slope":
            smoothed = savgol(
                y_array, window_length=window_length, polyorder=polyorder
            )
            slope = gradient(smoothed, x_array)
            positive_max = float(np.max(slope))
            threshold = (
                slope_threshold
                if slope_threshold is not None
                else max(positive_max * 0.15, 1e-12)
            )
            anchor_index = _first_index(slope >= threshold)
            shifted_x = x_array - x_array[anchor_index]
        case _:
            raise ValueError(f"Unsupported alignment method {method!r}")

    return AlignmentResult(
        method=method,
        anchor_index=anchor_index,
        anchor_x=float(x_array[anchor_index]),
        shifted_x=shifted_x.astype(np.float64, copy=False),
    )


def _first_index(mask: NDArray[np.bool_]) -> int:
    """Return the first `True` index in a boolean mask.

    Args:
        mask: Boolean mask to inspect.

    Returns:
        Index of the first `True` value, or `0` if the mask is empty.
    """

    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return 0
    return int(indices[0])
