"""Oliver-Pharr-style unloading analysis helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nanodent.analysis.filters import savgol
from nanodent.analysis.fit import curve_fit_model

if TYPE_CHECKING:
    from nanodent.study import Study

FloatArray = NDArray[np.float64]
FrozenMapping = Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class OliverPharrExperimentResult:
    """Result of a straight-line Oliver-Pharr unloading fit."""

    stem: str = ""
    success: bool = False
    reason: str | None = None
    peak_index: int = 0
    peak_force_uN: float | None = None
    peak_disp_nm: float | None = None
    unloading_start_index: int = 0
    unloading_end_index: int = 0
    fit_point_count: int = 0
    used_smoothing: bool = False
    smoothing: FrozenMapping = None
    stiffness_uN_per_nm: float | None = None
    force_intercept_uN: float | None = None
    depth_intercept_nm: float | None = None
    r_squared: float | None = None
    x_fit: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    y_fit: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary row."""

        return {
            "stem": self.stem,
            "success": self.success,
            "reason": self.reason,
            "peak_index": self.peak_index,
            "peak_force_uN": self.peak_force_uN,
            "peak_disp_nm": self.peak_disp_nm,
            "stiffness_uN_per_nm": self.stiffness_uN_per_nm,
            "force_intercept_uN": self.force_intercept_uN,
            "depth_intercept_nm": self.depth_intercept_nm,
            "r_squared": self.r_squared,
            "fit_point_count": self.fit_point_count,
        }


@dataclass(frozen=True, slots=True)
class OliverPharrBatchResult:
    """Immutable batch container for per-experiment Oliver-Pharr results."""

    study: Study
    results: tuple[OliverPharrExperimentResult, ...]
    unloading_fraction: float
    smoothing: FrozenMapping = None
    fit_num_points: int = 200

    def __post_init__(self) -> None:
        """Freeze optional batch-level smoothing settings."""

        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "smoothing", _freeze_mapping(self.smoothing))

    def __len__(self) -> int:
        """Return the number of analyzed experiments."""

        return len(self.results)

    def __iter__(self) -> Iterator[OliverPharrExperimentResult]:
        """Iterate over per-experiment results in study order."""

        return iter(self.results)

    def by_stem(self, stem: str) -> OliverPharrExperimentResult:
        """Return the result for one experiment stem."""

        for result in self.results:
            if result.stem == stem:
                return result
        raise KeyError(f"No Oliver-Pharr result for stem {stem!r}.")

    def summary(self) -> list[dict[str, Any]]:
        """Return compact rows suitable for notebook inspection."""

        return [result.summary() for result in self.results]


def analyze_oliver_pharr(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    unloading_fraction: float = 0.2,
    smoothing: Mapping[str, Any] | None = None,
    fit_num_points: int = 2,
    stem: str = "",
) -> OliverPharrExperimentResult:
    """Fit a straight line to the early unloading branch of one test curve.

    Args:
        disp_nm: Displacement values from the test section in acquisition
            order.
        force_uN: Force values from the test section in acquisition order.
        unloading_fraction: Fraction of the post-peak unloading branch used
            for the straight-line fit. Must lie in ``(0, 1]``.
        smoothing: Optional keyword arguments forwarded to `nanodent.savgol`.
            When provided, the same filter is applied to displacement and
            force before peak detection and fitting.
        fit_num_points: Number of points used to evaluate the dense fitted
            straight line for plotting.
        stem: Optional experiment label propagated by higher-level wrappers.

    Returns:
        Result object containing the fitted unloading diagnostics. Invalid or
        incomplete unloading segments are returned as unsuccessful results
        instead of raising, except for malformed input arguments.
    """

    disp_array = np.asarray(disp_nm, dtype=np.float64)
    force_array = np.asarray(force_uN, dtype=np.float64)
    if disp_array.shape != force_array.shape:
        raise ValueError("disp_nm and force_uN must have the same shape.")
    if disp_array.ndim != 1:
        raise ValueError("Oliver-Pharr analysis requires 1D signals.")
    if len(disp_array) == 0:
        raise ValueError("Oliver-Pharr analysis requires at least one sample.")
    if not 0.0 < unloading_fraction <= 1.0:
        raise ValueError("unloading_fraction must lie in the interval (0, 1].")
    if fit_num_points < 2:
        raise ValueError("fit_num_points must be at least 2.")

    frozen_smoothing = _freeze_mapping(smoothing)
    if frozen_smoothing is None:
        active_disp = disp_array.copy()
        active_force = force_array.copy()
    else:
        smoothing_kwargs = dict(frozen_smoothing)
        active_disp = savgol(disp_array, **smoothing_kwargs)
        active_force = savgol(force_array, **smoothing_kwargs)

    peak_index = int(np.argmax(active_force))
    peak_force = float(active_force[peak_index])
    peak_disp = float(active_disp[peak_index])
    if peak_index >= len(active_force) - 1:
        return _failed_result(
            stem=stem,
            reason="no_unloading_branch",
            peak_index=peak_index,
            peak_force_uN=peak_force,
            peak_disp_nm=peak_disp,
            unloading_start_index=peak_index,
            unloading_end_index=peak_index,
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
        )

    unloading_length = len(active_force) - peak_index
    fit_point_count = max(
        int(np.ceil(unloading_length * unloading_fraction)), 1
    )
    unloading_end_index = min(
        peak_index + fit_point_count - 1,
        len(active_force) - 1,
    )
    fit_disp = active_disp[peak_index : unloading_end_index + 1]
    fit_force = active_force[peak_index : unloading_end_index + 1]
    if len(fit_disp) < 5:
        return _failed_result(
            stem=stem,
            reason="too_few_unloading_points",
            peak_index=peak_index,
            peak_force_uN=peak_force,
            peak_disp_nm=peak_disp,
            unloading_start_index=peak_index,
            unloading_end_index=unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
        )
    if not np.isfinite(fit_disp).all() or not np.isfinite(fit_force).all():
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            peak_index=peak_index,
            peak_force_uN=peak_force,
            peak_disp_nm=peak_disp,
            unloading_start_index=peak_index,
            unloading_end_index=unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
        )

    try:
        fit_result = curve_fit_model(
            fit_disp,
            fit_force,
            _linear_model,
            num_points=fit_num_points,
        )
    except (RuntimeError, ValueError):
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            peak_index=peak_index,
            peak_force_uN=peak_force,
            peak_disp_nm=peak_disp,
            unloading_start_index=peak_index,
            unloading_end_index=unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
        )

    slope = float(fit_result.parameters[0])
    intercept = float(fit_result.parameters[1])
    if not np.isfinite([slope, intercept]).all():
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            peak_index=peak_index,
            peak_force_uN=peak_force,
            peak_disp_nm=peak_disp,
            unloading_start_index=peak_index,
            unloading_end_index=unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
        )
    if np.isclose(slope, 0.0, atol=1e-12):
        return _failed_result(
            stem=stem,
            reason="zero_stiffness",
            peak_index=peak_index,
            peak_force_uN=peak_force,
            peak_disp_nm=peak_disp,
            unloading_start_index=peak_index,
            unloading_end_index=unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
        )

    depth_intercept = float(-intercept / slope)
    if not np.isfinite(depth_intercept):
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            peak_index=peak_index,
            peak_force_uN=peak_force,
            peak_disp_nm=peak_disp,
            unloading_start_index=peak_index,
            unloading_end_index=unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
        )

    fitted_window = _linear_model(fit_disp, slope, intercept)
    r_squared = _r_squared(fit_force, fitted_window)
    return OliverPharrExperimentResult(
        stem=stem,
        success=True,
        reason=None,
        peak_index=peak_index,
        peak_force_uN=peak_force,
        peak_disp_nm=peak_disp,
        unloading_start_index=peak_index,
        unloading_end_index=unloading_end_index,
        fit_point_count=len(fit_disp),
        used_smoothing=frozen_smoothing is not None,
        smoothing=frozen_smoothing,
        stiffness_uN_per_nm=slope,
        force_intercept_uN=intercept,
        depth_intercept_nm=depth_intercept,
        r_squared=r_squared,
        x_fit=np.asarray(fit_result.x_fit, dtype=np.float64),
        y_fit=np.asarray(fit_result.y_fit, dtype=np.float64),
    )


def _failed_result(
    *,
    stem: str,
    reason: str,
    peak_index: int,
    peak_force_uN: float | None,
    peak_disp_nm: float | None,
    unloading_start_index: int,
    unloading_end_index: int,
    fit_point_count: int = 0,
    used_smoothing: bool,
    smoothing: FrozenMapping,
) -> OliverPharrExperimentResult:
    """Build a standardized unsuccessful analysis result."""

    return OliverPharrExperimentResult(
        stem=stem,
        success=False,
        reason=reason,
        peak_index=peak_index,
        peak_force_uN=peak_force_uN,
        peak_disp_nm=peak_disp_nm,
        unloading_start_index=unloading_start_index,
        unloading_end_index=unloading_end_index,
        fit_point_count=fit_point_count,
        used_smoothing=used_smoothing,
        smoothing=smoothing,
        stiffness_uN_per_nm=None,
        force_intercept_uN=None,
        depth_intercept_nm=None,
        r_squared=None,
        x_fit=np.empty(0, dtype=np.float64),
        y_fit=np.empty(0, dtype=np.float64),
    )


def _freeze_mapping(
    mapping: Mapping[str, Any] | None,
) -> FrozenMapping:
    """Return an immutable copy of an optional keyword mapping."""

    if mapping is None:
        return None
    return MappingProxyType(dict(mapping))


def _linear_model(
    x_values: NDArray[np.float64], slope: float, intercept: float
) -> NDArray[np.float64]:
    """Return a straight-line model."""

    return slope * x_values + intercept


def _r_squared(observed: FloatArray, predicted: FloatArray) -> float:
    """Return the coefficient of determination for one fitted window."""

    residual_sum = float(np.sum((observed - predicted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if np.isclose(total_sum, 0.0, atol=1e-12):
        return 1.0 if np.isclose(residual_sum, 0.0, atol=1e-12) else 0.0
    return float(1.0 - residual_sum / total_sum)
