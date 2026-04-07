"""Force-peak detection helpers for nanoindentation force signals."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks


@dataclass(frozen=True, slots=True)
class ForcePeakPosition:
    """One detected force peak with aligned experiment coordinates."""

    index: int
    time_s: float | None
    disp_nm: float | None
    force_uN: float
    prominence_uN: float | None

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary row."""

        return {
            "index": self.index,
            "time_s": self.time_s,
            "disp_nm": self.disp_nm,
            "force_uN": self.force_uN,
            "prominence_uN": self.prominence_uN,
        }


@dataclass(frozen=True, slots=True)
class ForcePeakDetectionResult:
    """Result of raw-force peak detection on one experiment."""

    success: bool
    reason: str | None = None
    peaks: tuple[ForcePeakPosition, ...] = ()
    peak_count: int = 0
    prominence: float = 100.0
    threshold: float | None = 1.0

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary row."""

        return {
            "success": self.success,
            "reason": self.reason,
            "peak_count": self.peak_count,
            "prominence": self.prominence,
            "threshold": self.threshold,
            "peaks": tuple(peak.summary() for peak in self.peaks),
        }


def detect_force_peaks(
    force_uN: ArrayLike,
    *,
    time_s: ArrayLike | None = None,
    disp_nm: ArrayLike | None = None,
    prominence: float = 100.0,
    threshold: float | None = 1.0,
) -> ForcePeakDetectionResult:
    """Detect up to two strongest force peaks in a raw force signal.

    Args:
        force_uN: Force values in acquisition order.
        time_s: Optional time values aligned with `force_uN`.
        disp_nm: Optional displacement values aligned with `force_uN`.
        prominence: Minimum peak prominence passed to `find_peaks`.
        threshold: Minimum peak threshold passed to `find_peaks`.

    Returns:
        Result object containing up to two strongest peaks. Signals with no
        detected peaks return an unsuccessful result instead of raising,
        except for malformed input arguments.
    """

    force_array = np.asarray(force_uN, dtype=np.float64)
    if force_array.ndim != 1:
        raise ValueError("Force-peak detection requires a 1D signal.")
    if len(force_array) == 0:
        raise ValueError("Force-peak detection requires at least one sample.")
    time_array = _optional_signal_array(
        time_s, name="time_s", expected_shape=force_array.shape
    )
    disp_array = _optional_signal_array(
        disp_nm, name="disp_nm", expected_shape=force_array.shape
    )

    peak_indices, peak_properties = find_peaks(
        force_array,
        prominence=prominence,
        threshold=threshold,
    )
    if len(peak_indices) == 0:
        return ForcePeakDetectionResult(
            success=False,
            reason="no_force_peaks_detected",
            peaks=(),
            peak_count=0,
            prominence=float(prominence),
            threshold=None if threshold is None else float(threshold),
        )

    prominences = np.asarray(
        peak_properties.get(
            "prominences", np.full(len(peak_indices), np.nan, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    selected_order = np.argsort(prominences)[-2:]
    selected_indices = np.asarray(peak_indices[selected_order], dtype=np.int64)
    selected_prominences = prominences[selected_order]
    sort_order = np.argsort(selected_indices)
    selected_indices = selected_indices[sort_order]
    selected_prominences = selected_prominences[sort_order]

    peaks = tuple(
        ForcePeakPosition(
            index=int(index),
            time_s=None if time_array is None else float(time_array[index]),
            disp_nm=None if disp_array is None else float(disp_array[index]),
            force_uN=float(force_array[index]),
            prominence_uN=float(prominence_value)
            if np.isfinite(prominence_value)
            else None,
        )
        for index, prominence_value in zip(
            selected_indices, selected_prominences, strict=False
        )
    )
    return ForcePeakDetectionResult(
        success=True,
        reason=None,
        peaks=peaks,
        peak_count=len(peaks),
        prominence=float(prominence),
        threshold=None if threshold is None else float(threshold),
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
