"""Unloading-branch detection helpers for nanoindentation test curves."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class UnloadingDetectionResult:
    """Result of unloading-start detection on one experiment curve."""

    success: bool
    method: str = "max_force"
    reason: str | None = None
    start_index: int | None = None
    start_time_s: float | None = None
    start_disp_nm: float | None = None
    start_force_uN: float | None = None
    end_disp_nm: float | None = None

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary row."""

        return {
            "success": self.success,
            "method": self.method,
            "reason": self.reason,
            "start_index": self.start_index,
            "start_time_s": self.start_time_s,
            "start_disp_nm": self.start_disp_nm,
            "start_force_uN": self.start_force_uN,
            "end_disp_nm": self.end_disp_nm,
        }


def detect_unloading(
    force_uN: ArrayLike,
    *,
    time_s: ArrayLike | None = None,
    disp_nm: ArrayLike | None = None,
    method: Literal["max_force"] = "max_force",
) -> UnloadingDetectionResult:
    """Detect the start of the unloading branch in one test curve.

    Args:
        force_uN: Force values in acquisition order.
        time_s: Optional time values aligned with `force_uN`.
        disp_nm: Optional displacement values aligned with `force_uN`.
        method: Detection strategy. `max_force` uses the global maximum force
            sample as unloading start.

    Returns:
        Result object containing the unloading-start coordinates. The current
        default method always succeeds for non-empty, well-formed signals.
    """

    force_array = np.asarray(force_uN, dtype=np.float64)
    if force_array.ndim != 1:
        raise ValueError("Unloading detection requires a 1D signal.")
    if len(force_array) == 0:
        raise ValueError("Unloading detection requires at least one sample.")
    time_array = _optional_signal_array(
        time_s, name="time_s", expected_shape=force_array.shape
    )
    disp_array = _optional_signal_array(
        disp_nm, name="disp_nm", expected_shape=force_array.shape
    )
    if method != "max_force":
        raise ValueError("method must be 'max_force'.")

    start_index = int(np.argmax(force_array))
    return UnloadingDetectionResult(
        success=True,
        method=method,
        reason=None,
        start_index=start_index,
        start_time_s=None
        if time_array is None
        else float(time_array[start_index]),
        start_disp_nm=None
        if disp_array is None
        else float(disp_array[start_index]),
        start_force_uN=float(force_array[start_index]),
        end_disp_nm=None if disp_array is None else float(disp_array[-1]),
    )


def _optional_signal_array(
    values: ArrayLike | None,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Validate an optional 1D signal aligned with the force signal."""

    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have the same shape as force_uN.")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D signal.")
    return array
