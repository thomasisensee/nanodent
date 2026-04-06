"""Onset-detection helpers for nanoindentation force signals."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from nanodent.analysis.filters import savgol


@dataclass(frozen=True, slots=True)
class OnsetDetectionResult:
    """Result of sustained-threshold onset detection on one force signal."""

    success: bool
    mode: str = "relative"
    reason: str | None = None
    onset_index: int | None = None
    onset_time_s: float | None = None
    onset_disp_nm: float | None = None
    baseline_points: int = 0
    baseline_start_index: int | None = None
    baseline_end_index: int | None = None
    baseline_mean_uN: float | None = None
    baseline_offset_uN: float | None = None
    baseline_std_uN: float | None = None
    threshold_uN: float | None = None
    absolute_threshold_uN: float | None = None
    k: float = 4.0
    consecutive: int = 1
    used_smoothing: bool = False
    smoothing: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary row."""

        return {
            "success": self.success,
            "mode": self.mode,
            "reason": self.reason,
            "onset_index": self.onset_index,
            "onset_time_s": self.onset_time_s,
            "onset_disp_nm": self.onset_disp_nm,
            "baseline_points": self.baseline_points,
            "baseline_start_index": self.baseline_start_index,
            "baseline_end_index": self.baseline_end_index,
            "baseline_mean_uN": self.baseline_mean_uN,
            "baseline_offset_uN": self.baseline_offset_uN,
            "baseline_std_uN": self.baseline_std_uN,
            "threshold_uN": self.threshold_uN,
            "absolute_threshold_uN": self.absolute_threshold_uN,
            "k": self.k,
            "consecutive": self.consecutive,
        }


def detect_onset(
    force_uN: ArrayLike,
    *,
    time_s: ArrayLike | None = None,
    disp_nm: ArrayLike | None = None,
    mode: Literal["relative", "absolute"] = "relative",
    baseline_points: int = 100,
    baseline_start_index: int | None = None,
    baseline_end_index: int | None = None,
    k: float = 4.0,
    absolute_threshold_uN: float | None = None,
    consecutive: int = 5,
    smoothing: Mapping[str, Any] | None = None,
) -> OnsetDetectionResult:
    """Detect the first sustained threshold crossing in one force signal.

    Args:
        force_uN: Force values in acquisition order.
        time_s: Optional time values aligned with `force_uN`.
        disp_nm: Optional displacement values aligned with `force_uN`.
        mode: Thresholding mode. `relative` uses baseline mean plus
            `k * baseline_std`; `absolute` compares directly against
            `absolute_threshold_uN`.
        baseline_points: Number of leading samples used to estimate the
            baseline statistics. The effective count is clipped to the signal
            length.
        baseline_start_index: Optional inclusive start index of the baseline
            window.
        baseline_end_index: Optional exclusive end index of the baseline
            window.
        k: Number of baseline standard deviations above the mean required for
            the detection threshold.
        absolute_threshold_uN: Absolute threshold used in `absolute` mode.
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
    if mode not in {"relative", "absolute"}:
        raise ValueError("mode must be 'relative' or 'absolute'.")
    if (baseline_start_index is None) ^ (baseline_end_index is None):
        raise ValueError(
            "baseline_start_index and baseline_end_index must be provided "
            "together."
        )
    if mode == "absolute" and absolute_threshold_uN is None:
        raise ValueError(
            "absolute_threshold_uN is required when mode='absolute'."
        )

    frozen_smoothing = _freeze_mapping(smoothing)
    active_force = (
        force_array.copy()
        if frozen_smoothing is None
        else savgol(force_array, **dict(frozen_smoothing))
    )

    (
        baseline,
        baseline_window_start,
        baseline_window_end,
        effective_baseline_points,
    ) = _baseline_window(
        active_force,
        baseline_points=baseline_points,
        baseline_start_index=baseline_start_index,
        baseline_end_index=baseline_end_index,
    )
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))
    threshold = (
        float(baseline_mean + k * baseline_std)
        if mode == "relative"
        else float(absolute_threshold_uN)
    )
    above_threshold = active_force > threshold

    run_length = 0
    for index in range(baseline_window_end, len(above_threshold)):
        is_above = above_threshold[index]
        run_length = run_length + 1 if bool(is_above) else 0
        if run_length >= consecutive:
            onset_index = index - consecutive + 1
            return OnsetDetectionResult(
                success=True,
                mode=mode,
                reason=None,
                onset_index=onset_index,
                onset_time_s=None
                if time_array is None
                else float(time_array[onset_index]),
                onset_disp_nm=None
                if disp_array is None
                else float(disp_array[onset_index]),
                baseline_points=effective_baseline_points,
                baseline_start_index=baseline_window_start,
                baseline_end_index=baseline_window_end,
                baseline_mean_uN=baseline_mean,
                baseline_offset_uN=baseline_mean,
                baseline_std_uN=baseline_std,
                threshold_uN=threshold,
                absolute_threshold_uN=None
                if mode == "relative"
                else float(absolute_threshold_uN),
                k=float(k),
                consecutive=consecutive,
                used_smoothing=frozen_smoothing is not None,
                smoothing=frozen_smoothing,
            )

    return OnsetDetectionResult(
        success=False,
        mode=mode,
        reason="no_onset_detected",
        onset_index=None,
        onset_time_s=None,
        onset_disp_nm=None,
        baseline_points=effective_baseline_points,
        baseline_start_index=baseline_window_start,
        baseline_end_index=baseline_window_end,
        baseline_mean_uN=baseline_mean,
        baseline_offset_uN=baseline_mean,
        baseline_std_uN=baseline_std,
        threshold_uN=threshold,
        absolute_threshold_uN=None
        if mode == "relative"
        else float(absolute_threshold_uN),
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


def _baseline_window(
    active_force: np.ndarray,
    *,
    baseline_points: int,
    baseline_start_index: int | None,
    baseline_end_index: int | None,
) -> tuple[np.ndarray, int, int, int]:
    """Return the selected baseline window and its slice indices."""

    if baseline_start_index is None or baseline_end_index is None:
        effective_baseline_points = min(baseline_points, len(active_force))
        return (
            active_force[:effective_baseline_points],
            0,
            effective_baseline_points,
            effective_baseline_points,
        )
    if baseline_start_index < 0 or baseline_end_index > len(active_force):
        raise ValueError("Baseline indices must lie within the force signal.")
    if baseline_start_index >= baseline_end_index:
        raise ValueError("Baseline indices require start < end.")
    baseline = active_force[baseline_start_index:baseline_end_index]
    if len(baseline) == 0:
        raise ValueError("Baseline window must contain at least one sample.")
    return (
        baseline,
        baseline_start_index,
        baseline_end_index,
        len(baseline),
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
