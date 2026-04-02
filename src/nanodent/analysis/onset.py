"""Onset-detection helpers for nanoindentation force signals."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from nanodent.analysis.filters import savgol


@dataclass(frozen=True, slots=True)
class OnsetDetectionResult:
    """Result of sustained-threshold onset detection on one force signal."""

    success: bool
    reason: str | None = None
    onset_index: int | None = None
    onset_time_s: float | None = None
    onset_disp_nm: float | None = None
    baseline_points: int = 0
    baseline_mean_uN: float | None = None
    baseline_std_uN: float | None = None
    threshold_uN: float | None = None
    k: float = 4.0
    consecutive: int = 1
    used_smoothing: bool = False
    smoothing: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary row."""

        return {
            "success": self.success,
            "reason": self.reason,
            "onset_index": self.onset_index,
            "onset_time_s": self.onset_time_s,
            "onset_disp_nm": self.onset_disp_nm,
            "baseline_points": self.baseline_points,
            "baseline_mean_uN": self.baseline_mean_uN,
            "baseline_std_uN": self.baseline_std_uN,
            "threshold_uN": self.threshold_uN,
            "k": self.k,
            "consecutive": self.consecutive,
        }


def detect_onset(
    force_uN: ArrayLike,
    *,
    time_s: ArrayLike | None = None,
    disp_nm: ArrayLike | None = None,
    baseline_points: int = 100,
    k: float = 4.0,
    consecutive: int = 5,
    smoothing: Mapping[str, Any] | None = None,
) -> OnsetDetectionResult:
    """Detect the first sustained threshold crossing in one force signal.

    Args:
        force_uN: Force values in acquisition order.
        time_s: Optional time values aligned with `force_uN`.
        disp_nm: Optional displacement values aligned with `force_uN`.
        baseline_points: Number of leading samples used to estimate the
            baseline statistics. The effective count is clipped to the signal
            length.
        k: Number of baseline standard deviations above the mean required for
            the detection threshold.
        consecutive: Number of consecutive samples above the threshold needed
            to accept an onset.
        smoothing: Optional keyword arguments forwarded to `nanodent.savgol`.

    Returns:
        Result object containing the onset index and baseline diagnostics.
        Signals with no sustained crossing return an unsuccessful result
        instead of raising, except for malformed input arguments.
    """

    force_array = np.asarray(force_uN, dtype=np.float64)
    if force_array.ndim != 1:
        raise ValueError("Onset detection requires a 1D signal.")
    if len(force_array) == 0:
        raise ValueError("Onset detection requires at least one sample.")
    time_array = _optional_signal_array(
        time_s, name="time_s", expected_shape=force_array.shape
    )
    disp_array = _optional_signal_array(
        disp_nm, name="disp_nm", expected_shape=force_array.shape
    )
    if baseline_points < 1:
        raise ValueError("baseline_points must be at least 1.")
    if consecutive < 1:
        raise ValueError("consecutive must be at least 1.")

    frozen_smoothing = _freeze_mapping(smoothing)
    active_force = (
        force_array.copy()
        if frozen_smoothing is None
        else savgol(force_array, **dict(frozen_smoothing))
    )

    effective_baseline_points = min(baseline_points, len(active_force))
    baseline = active_force[:effective_baseline_points]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))
    threshold = float(baseline_mean + k * baseline_std)
    above_threshold = active_force > threshold

    run_length = 0
    for index, is_above in enumerate(above_threshold):
        run_length = run_length + 1 if bool(is_above) else 0
        if run_length >= consecutive:
            onset_index = index - consecutive + 1
            return OnsetDetectionResult(
                success=True,
                reason=None,
                onset_index=onset_index,
                onset_time_s=None
                if time_array is None
                else float(time_array[onset_index]),
                onset_disp_nm=None
                if disp_array is None
                else float(disp_array[onset_index]),
                baseline_points=effective_baseline_points,
                baseline_mean_uN=baseline_mean,
                baseline_std_uN=baseline_std,
                threshold_uN=threshold,
                k=float(k),
                consecutive=consecutive,
                used_smoothing=frozen_smoothing is not None,
                smoothing=frozen_smoothing,
            )

    return OnsetDetectionResult(
        success=False,
        reason="no_onset_detected",
        onset_index=None,
        onset_time_s=None,
        onset_disp_nm=None,
        baseline_points=effective_baseline_points,
        baseline_mean_uN=baseline_mean,
        baseline_std_uN=baseline_std,
        threshold_uN=threshold,
        k=float(k),
        consecutive=consecutive,
        used_smoothing=frozen_smoothing is not None,
        smoothing=frozen_smoothing,
    )


def _freeze_mapping(
    mapping: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return an immutable copy of an optional keyword mapping."""

    if mapping is None:
        return None
    return MappingProxyType(dict(mapping))


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
