"""Heuristic quality checks for nanoindentation curves."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class QualityCheckResult:
    """Result of a heuristic experiment-quality check."""

    enabled: bool
    reason: str | None = None
    onset_fraction: float | None = None
    onset_disp_nm: float | None = None


def classify_delayed_onset(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    bin_count: int = 24,
    baseline_bin_count: int = 4,
    onset_force_fraction: float = 0.1,
    sustained_bins: int = 2,
    max_onset_fraction: float = 0.7,
) -> QualityCheckResult:
    """Classify curves whose useful rise begins too late in displacement.

    The heuristic sorts the force-displacement samples by displacement,
    averages them onto coarse displacement bins, and treats the onset as the
    first sustained rise above the initial baseline plus a fraction of the
    dynamic range.

    Args:
        disp_nm: Displacement values from the test section.
        force_uN: Force values from the test section.
        bin_count: Number of coarse displacement bins used to suppress
            point-to-point loops and noise.
        baseline_bin_count: Number of leftmost bins used to estimate the
            initial baseline force level.
        onset_force_fraction: Fraction of the binned dynamic range used to
            define onset.
        sustained_bins: Number of consecutive bins that must exceed the onset
            threshold before the onset is accepted.
        max_onset_fraction: Maximum allowed relative onset position within the
            displacement span. Larger values are classified as delayed onset.

    Returns:
        Quality classification result with an optional onset location.
    """

    x = np.asarray(disp_nm, dtype=np.float64)
    y = np.asarray(force_uN, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("disp_nm and force_uN must have the same shape.")
    if x.ndim != 1:
        raise ValueError("Delayed-onset classification requires 1D signals.")
    if len(x) == 0:
        return QualityCheckResult(enabled=True)

    order = np.argsort(x)
    sorted_x = x[order]
    sorted_y = y[order]
    min_x = float(sorted_x[0])
    max_x = float(sorted_x[-1])
    if np.isclose(min_x, max_x):
        return QualityCheckResult(enabled=True)

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

    if not binned_force:
        return QualityCheckResult(enabled=True)

    coarse_force = np.asarray(binned_force, dtype=np.float64)
    coarse_disp = np.asarray(binned_disp, dtype=np.float64)
    baseline_count = min(max(baseline_bin_count, 1), len(coarse_force))
    baseline = float(np.median(coarse_force[:baseline_count]))
    dynamic_range = float(np.max(coarse_force) - baseline)
    if dynamic_range <= 0.0:
        return QualityCheckResult(enabled=True)

    threshold = baseline + onset_force_fraction * dynamic_range
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
    enabled = onset_fraction <= max_onset_fraction
    return QualityCheckResult(
        enabled=enabled,
        reason=None if enabled else "delayed_onset",
        onset_fraction=float(onset_fraction),
        onset_disp_nm=onset_disp,
    )
