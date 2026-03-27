"""Heuristic quality checks for nanoindentation curves."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

from nanodent.analysis.filters import savgol


@dataclass(frozen=True, slots=True)
class QualityCheckResult:
    """Result of a heuristic experiment-quality check."""

    enabled: bool
    reason: str | None = None
    onset_fraction: float | None = None
    onset_disp_nm: float | None = None
    rise_width_fraction: float | None = None


def classify_high_displacement(
    disp_nm: ArrayLike,
    *,
    max_disp_nm: float = 1000.0,
) -> QualityCheckResult:
    """Classify runs whose test displacement exceeds a hard limit.

    Args:
        disp_nm: Displacement values from the test section.
        max_disp_nm: Maximum allowed displacement in nanometers.

    Returns:
        Quality classification result. Curves that exceed the displacement
        limit are marked with reason `high_disp`.
    """

    x = np.asarray(disp_nm, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(
            "High-displacement classification requires a 1D signal."
        )
    if len(x) == 0:
        return QualityCheckResult(enabled=True)

    enabled = bool(np.max(x) <= max_disp_nm)
    return QualityCheckResult(
        enabled=enabled,
        reason=None if enabled else "high_disp",
    )


def classify_outlier_jumps(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    disp_z_threshold: float = 100.0,
    force_z_threshold: float = 70.0,
) -> QualityCheckResult:
    """Classify experiments with isolated local spikes in displacement
    or force.

    Args:
        disp_nm: Displacement values from the test section.
        force_uN: Force values from the test section.
        disp_z_threshold: Robust z-score threshold for isolated displacement
            spikes relative to neighboring samples.
        force_z_threshold: Robust z-score threshold for isolated force spikes
            relative to neighboring samples.

    Returns:
        Quality classification result with reason `outlier_disp` or
        `outlier_force` when an isolated local spike is detected.
    """

    x = np.asarray(disp_nm, dtype=np.float64)
    y = np.asarray(force_uN, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("disp_nm and force_uN must have the same shape.")
    if x.ndim != 1:
        raise ValueError("Outlier-jump classification requires 1D signals.")
    if len(x) < 3:
        return QualityCheckResult(enabled=True)

    disp_score = _max_local_outlier_score(x)
    if disp_score >= disp_z_threshold:
        return QualityCheckResult(enabled=False, reason="outlier_disp")

    force_score = _max_local_outlier_score(y)
    if force_score >= force_z_threshold:
        return QualityCheckResult(enabled=False, reason="outlier_force")

    return QualityCheckResult(enabled=True)


def classify_flat_force(
    force_uN: ArrayLike,
    *,
    min_robust_force_span_uN: float = 200.0,
    low_quantile: float = 0.05,
    high_quantile: float = 0.95,
) -> QualityCheckResult:
    """Classify runs whose force signal remains nearly flat throughout.

    Args:
        force_uN: Force values from the test section.
        min_robust_force_span_uN: Minimum acceptable robust force span between
            the selected quantiles.
        low_quantile: Lower quantile used for the robust force span.
        high_quantile: Upper quantile used for the robust force span.

    Returns:
        Quality classification result. Flat signals are marked with reason
        `flat_force`.
    """

    y = np.asarray(force_uN, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("Flat-force classification requires a 1D signal.")
    if len(y) == 0:
        return QualityCheckResult(enabled=True)

    low_value, high_value = np.quantile(y, [low_quantile, high_quantile])
    robust_span = float(high_value - low_value)
    enabled = robust_span >= min_robust_force_span_uN
    return QualityCheckResult(
        enabled=enabled,
        reason=None if enabled else "flat_force",
    )


def classify_gradual_onset(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    bin_count: int = 24,
    baseline_bin_count: int = 4,
    onset_force_fraction: float = 0.05,
    target_force_fraction: float = 0.5,
    sustained_bins: int = 2,
    max_rise_width_fraction: float = 0.2,
) -> QualityCheckResult:
    """Classify curves whose onset rises too gradually in displacement.

    The heuristic sorts the force-displacement samples by displacement,
    averages them onto coarse displacement bins, and measures how much
    displacement the coarse curve needs to rise from an early threshold to a
    mid-force threshold. Curves whose rise width is too broad are classified
    as gradual onset.

    Args:
        disp_nm: Displacement values from the test section.
        force_uN: Force values from the test section.
        bin_count: Number of coarse displacement bins used to suppress
            point-to-point loops and noise.
        baseline_bin_count: Number of leftmost bins used to estimate the
            initial baseline force level.
        onset_force_fraction: Lower force fraction used to define onset.
        target_force_fraction: Upper force fraction used to define where the
            onset rise is considered complete.
        sustained_bins: Number of consecutive bins that must exceed the lower
            onset threshold before the onset is accepted.
        max_rise_width_fraction: Maximum allowed relative displacement width
            between the lower and upper force thresholds.

    Returns:
        Quality classification result with optional onset diagnostics.
    """

    x = np.asarray(disp_nm, dtype=np.float64)
    y = np.asarray(force_uN, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("disp_nm and force_uN must have the same shape.")
    if x.ndim != 1:
        raise ValueError("Gradual-onset classification requires 1D signals.")
    if len(x) == 0:
        return QualityCheckResult(enabled=True)

    sorted_x, coarse_disp, coarse_force = _coarse_force_curve(
        x, y, bin_count=bin_count
    )
    if coarse_force.size == 0:
        return QualityCheckResult(enabled=True)

    min_x = float(sorted_x[0])
    max_x = float(sorted_x[-1])
    if np.isclose(min_x, max_x):
        return QualityCheckResult(enabled=True)

    baseline_count = min(max(baseline_bin_count, 1), len(coarse_force))
    baseline = float(np.median(coarse_force[:baseline_count]))
    dynamic_range = float(np.max(coarse_force) - baseline)
    if dynamic_range <= 0.0:
        return QualityCheckResult(enabled=True)

    threshold = baseline + onset_force_fraction * dynamic_range
    target_threshold = baseline + target_force_fraction * dynamic_range
    required_bins = min(max(sustained_bins, 1), len(coarse_force))

    onset_index: int | None = None
    for index in range(len(coarse_force) - required_bins + 1):
        if np.all(coarse_force[index : index + required_bins] >= threshold):
            onset_index = index
            break

    if onset_index is None:
        return QualityCheckResult(enabled=True)

    onset_disp = float(coarse_disp[onset_index])
    onset_fraction = (onset_disp - min_x) / (max_x - min_x)
    target_index = next(
        (
            index
            for index in range(onset_index, len(coarse_force))
            if coarse_force[index] >= target_threshold
        ),
        None,
    )
    if target_index is None:
        return QualityCheckResult(
            enabled=False,
            reason="gradual_onset",
            onset_fraction=float(onset_fraction),
            onset_disp_nm=onset_disp,
            rise_width_fraction=1.0,
        )

    rise_width_fraction = float(
        coarse_disp[target_index] - coarse_disp[onset_index]
    ) / (max_x - min_x)
    enabled = rise_width_fraction <= max_rise_width_fraction
    return QualityCheckResult(
        enabled=enabled,
        reason=None if enabled else "gradual_onset",
        onset_fraction=float(onset_fraction),
        onset_disp_nm=onset_disp,
        rise_width_fraction=float(rise_width_fraction),
    )


def classify_peak_balance(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    peak_bin_count: int = 48,
    peak_prominence_fraction: float = 0.05,
    min_secondary_peak_fraction: float = 0.1,
    require_two_peaks: bool = False,
) -> QualityCheckResult:
    """Classify curves whose second resolved peak is too small.

    The heuristic sorts the force-displacement samples by displacement,
    averages them onto coarse displacement bins, smooths the coarse force
    curve, and compares the two strongest resolved peaks.

    If fewer than two prominent peaks are resolved after smoothing, the
    heuristic abstains by default. Set ``require_two_peaks=True`` to disable
    runs unless two resolved peaks are present.

    Args:
        disp_nm: Displacement values from the test section.
        force_uN: Force values from the test section.
        peak_bin_count: Number of coarse displacement bins used before peak
            detection.
        peak_prominence_fraction: Minimum prominence for resolved peaks,
            expressed as a fraction of the coarse-force dynamic range.
        min_secondary_peak_fraction: Minimum allowed ratio between the
            second-highest and highest resolved peaks.
        require_two_peaks: When true, disable curves that do not resolve at
            least two prominent peaks after smoothing.

    Returns:
        Quality classification result. Curves with an undersized second peak,
        or without two resolved peaks when ``require_two_peaks`` is true, are
        marked with reason `weak_second_peak`.
    """

    x = np.asarray(disp_nm, dtype=np.float64)
    y = np.asarray(force_uN, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("disp_nm and force_uN must have the same shape.")
    if x.ndim != 1:
        raise ValueError("Peak-balance classification requires 1D signals.")
    if len(x) == 0:
        return QualityCheckResult(enabled=True)

    _, _, coarse_force = _coarse_force_curve(x, y, bin_count=peak_bin_count)
    if coarse_force.size < 5:
        return QualityCheckResult(enabled=True)

    smoothed_force = savgol(coarse_force, window_length=7, polyorder=1)
    dynamic_range = float(np.ptp(smoothed_force))
    if dynamic_range <= 0.0:
        return QualityCheckResult(enabled=True)

    baseline = float(np.min(smoothed_force))
    prominence = max(dynamic_range * peak_prominence_fraction, 1.0)
    peak_indices, _ = find_peaks(
        smoothed_force,
        prominence=prominence,
        distance=max(len(smoothed_force) // 8, 1),
    )
    if len(peak_indices) < 2:
        if require_two_peaks:
            return QualityCheckResult(enabled=False, reason="weak_second_peak")
        return QualityCheckResult(enabled=True)

    peak_heights = sorted(
        (float(smoothed_force[index] - baseline) for index in peak_indices),
        reverse=True,
    )
    if peak_heights[0] <= 0.0:
        return QualityCheckResult(enabled=True)

    secondary_peak_fraction = peak_heights[1] / peak_heights[0]
    enabled = secondary_peak_fraction >= min_secondary_peak_fraction
    return QualityCheckResult(
        enabled=enabled,
        reason=None if enabled else "weak_second_peak",
    )


def classify_quality(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    min_robust_force_span_uN: float = 200.0,
    low_quantile: float = 0.40,
    high_quantile: float = 0.999,
    max_disp_nm: float = 1000.0,
    peak_bin_count: int = 48,
    peak_prominence_fraction: float = 0.05,
    min_secondary_peak_fraction: float = 0.1,
    require_two_peaks: bool = False,
    disp_z_threshold: float = 100.0,
    force_z_threshold: float = 70.0,
    bin_count: int = 24,
    baseline_bin_count: int = 4,
    onset_force_fraction: float = 0.05,
    target_force_fraction: float = 0.5,
    sustained_bins: int = 2,
    max_rise_width_fraction: float = 0.2,
) -> QualityCheckResult:
    """Run the enabled quality heuristics in order and return the first match.

    Args:
        disp_nm: Displacement values from the test section.
        force_uN: Force values from the test section.
        min_robust_force_span_uN: Minimum acceptable robust force span for the
            flat-force check.
        low_quantile: Lower quantile used for the robust force span.
        high_quantile: Upper quantile used for the robust force span.
        max_disp_nm: Maximum allowed displacement before disabling the
            experiment.
        peak_bin_count: Number of coarse displacement bins used for the
            peak-balance heuristic.
        peak_prominence_fraction: Minimum prominence used to resolve peaks,
            relative to the coarse-force dynamic range.
        min_secondary_peak_fraction: Minimum allowed ratio between the
            second-highest and highest resolved peaks.
        require_two_peaks: When true, disable curves that do not resolve at
            least two peaks after smoothing.
        disp_z_threshold: Robust z-score threshold for isolated displacement
            spikes.
        force_z_threshold: Robust z-score threshold for isolated force spikes.
        bin_count: Number of coarse displacement bins for gradual-onset
            detection.
        baseline_bin_count: Number of early bins used for baseline force.
        onset_force_fraction: Lower force fraction used to define onset.
        target_force_fraction: Upper force fraction used to define where the
            onset rise is considered complete.
        sustained_bins: Number of consecutive bins that must exceed the onset
            threshold.
        max_rise_width_fraction: Maximum allowed displacement width between the
            lower and upper onset thresholds before disabling the experiment.

    Returns:
        First disabling classification that matches, otherwise an enabled
        result.
    """

    flat_force_result = classify_flat_force(
        force_uN,
        min_robust_force_span_uN=min_robust_force_span_uN,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    if not flat_force_result.enabled:
        return flat_force_result

    outlier_result = classify_outlier_jumps(
        disp_nm,
        force_uN,
        disp_z_threshold=disp_z_threshold,
        force_z_threshold=force_z_threshold,
    )
    if not outlier_result.enabled:
        return outlier_result

    high_disp_result = classify_high_displacement(
        disp_nm, max_disp_nm=max_disp_nm
    )
    if not high_disp_result.enabled:
        return high_disp_result

    gradual_onset_result = classify_gradual_onset(
        disp_nm,
        force_uN,
        bin_count=bin_count,
        baseline_bin_count=baseline_bin_count,
        onset_force_fraction=onset_force_fraction,
        target_force_fraction=target_force_fraction,
        sustained_bins=sustained_bins,
        max_rise_width_fraction=max_rise_width_fraction,
    )
    if not gradual_onset_result.enabled:
        return gradual_onset_result

    return classify_peak_balance(
        disp_nm,
        force_uN,
        peak_bin_count=peak_bin_count,
        peak_prominence_fraction=peak_prominence_fraction,
        min_secondary_peak_fraction=min_secondary_peak_fraction,
        require_two_peaks=require_two_peaks,
    )


def _coarse_force_curve(
    disp_nm: np.ndarray, force_uN: np.ndarray, *, bin_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return displacement-sorted samples and a coarse binned force curve."""

    order = np.argsort(disp_nm)
    sorted_x = disp_nm[order]
    sorted_y = force_uN[order]
    min_x = float(sorted_x[0])
    max_x = float(sorted_x[-1])
    edges = np.linspace(min_x, max_x, max(bin_count, 2) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned_force: list[float] = []
    binned_disp: list[float] = []
    for index, (lower, upper, center) in enumerate(
        zip(edges[:-1], edges[1:], centers, strict=False)
    ):
        if index == len(edges) - 2:
            mask = (sorted_x >= lower) & (sorted_x <= upper)
        else:
            mask = (sorted_x >= lower) & (sorted_x < upper)
        if not np.any(mask):
            continue
        binned_force.append(float(np.mean(sorted_y[mask])))
        binned_disp.append(float(center))

    return (
        sorted_x,
        np.asarray(binned_disp, dtype=np.float64),
        np.asarray(binned_force, dtype=np.float64),
    )


def _max_local_outlier_score(values: np.ndarray) -> float:
    """Return the strongest local outlier score against neighboring samples."""

    if len(values) < 3:
        return 0.0
    residual = values[1:-1] - 0.5 * (values[:-2] + values[2:])
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    max_deviation = float(np.max(np.abs(residual - median)))
    if mad <= 0.0:
        return (
            0.0 if np.isclose(max_deviation, 0.0, atol=1e-12) else float("inf")
        )
    return max_deviation / (1.4826 * mad)
