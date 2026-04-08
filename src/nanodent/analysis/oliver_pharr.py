"""Oliver-Pharr-style unloading analysis helpers."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from nanodent.analysis.filters import savgol
from nanodent.models import TipAreaFunction

_LINEAR_FIT_MODEL = "linear_fraction"
_POWER_LAW_FIT_MODEL = "power_law_full"
_DEFAULT_TIP_AREA_FUNCTION = TipAreaFunction(c0=24.5)


@dataclass(frozen=True, slots=True)
class OliverPharrExperimentResult:
    """Result of one Oliver-Pharr unloading fit."""

    stem: str = ""
    success: bool = False
    reason: str | None = None
    fit_model: str = _LINEAR_FIT_MODEL
    evaluation_index: int = 0
    evaluation_force_uN: float | None = None
    evaluation_disp_nm: float | None = None
    unloading_start_index: int = 0
    unloading_end_index: int = 0
    fit_point_count: int = 0
    used_smoothing: bool = False
    smoothing: Mapping[str, Any] | None = None
    disp_correction_nm: float | None = None
    force_correction_uN: float | None = None
    stiffness_uN_per_nm: float | None = None
    r_squared: float | None = None
    linear_slope_uN_per_nm: float | None = None
    linear_intercept_uN: float | None = None
    linear_depth_intercept_nm: float | None = None
    power_law_k: float | None = None
    power_law_m: float | None = None
    power_law_hf_nm: float | None = None
    hardness_success: bool = False
    hardness_reason: str | None = None
    epsilon: float | None = None
    onset_disp_nm: float | None = None
    hmax_nm: float | None = None
    contact_depth_nm: float | None = None
    contact_area_nm2: float | None = None
    hardness_uN_per_nm2: float | None = None
    reduced_modulus_uN_per_nm2: float | None = None
    tip_area_function: TipAreaFunction | None = None
    x_fit: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    y_fit: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary row."""

        return {
            "stem": self.stem,
            "success": self.success,
            "reason": self.reason,
            "fit_model": self.fit_model,
            "evaluation_index": self.evaluation_index,
            "evaluation_force_uN": self.evaluation_force_uN,
            "evaluation_disp_nm": self.evaluation_disp_nm,
            "unloading_start_index": self.unloading_start_index,
            "unloading_end_index": self.unloading_end_index,
            "fit_point_count": self.fit_point_count,
            "disp_correction_nm": self.disp_correction_nm,
            "force_correction_uN": self.force_correction_uN,
            "stiffness_uN_per_nm": self.stiffness_uN_per_nm,
            "r_squared": self.r_squared,
            "linear_slope_uN_per_nm": self.linear_slope_uN_per_nm,
            "linear_intercept_uN": self.linear_intercept_uN,
            "linear_depth_intercept_nm": self.linear_depth_intercept_nm,
            "power_law_k": self.power_law_k,
            "power_law_m": self.power_law_m,
            "power_law_hf_nm": self.power_law_hf_nm,
            "hardness_success": self.hardness_success,
            "hardness_reason": self.hardness_reason,
            "epsilon": self.epsilon,
            "onset_disp_nm": self.onset_disp_nm,
            "hmax_nm": self.hmax_nm,
            "contact_depth_nm": self.contact_depth_nm,
            "contact_area_nm2": self.contact_area_nm2,
            "hardness_uN_per_nm2": self.hardness_uN_per_nm2,
            "reduced_modulus_uN_per_nm2": self.reduced_modulus_uN_per_nm2,
            "tip_area_function": self.tip_area_function,
        }


def analyze_oliver_pharr(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    unloading_start_trace_index: int = 0,
    fit_model: Literal["linear_fraction", "power_law_full"] = (
        _POWER_LAW_FIT_MODEL
    ),
    unloading_fraction: float | None = None,
    smoothing: Mapping[str, Any] | None = None,
    fit_num_points: int = 200,
    onset_disp_nm: float | None = None,
    baseline_offset_uN: float | None = None,
    epsilon: float = 0.75,
    tip_area_function: TipAreaFunction | None = None,
    stem: str = "",
) -> OliverPharrExperimentResult:
    """Fit one supported Oliver-Pharr model to an unloading branch.

    Args:
        disp_nm: Displacement values from the unloading branch in
            acquisition order.
        force_uN: Force values from the unloading branch in acquisition
            order.
        unloading_start_trace_index: Absolute trace index of the first
            unloading sample. This keeps result indices aligned with the
            original experiment trace.
        fit_model: Fitting strategy. `linear_fraction` fits a straight line
            to the early unloading branch. `power_law_full` fits
            `f = k * (h - hf)^m` to the full unloading branch.
        unloading_fraction: Fraction of the post-start unloading branch used
            for `linear_fraction`. When omitted, defaults to `0.2`.
        smoothing: Optional keyword arguments forwarded to `nanodent.savgol`.
            When provided, the same filter is applied to displacement and
            force before fitting.
        fit_num_points: Number of points used to evaluate the dense fitted
            curve for plotting.
        onset_disp_nm: Optional onset displacement used to compute
            onset-corrected hardness diagnostics.
        baseline_offset_uN: Optional force baseline offset subtracted before
            evaluation and fitting.
        epsilon: Geometry factor used for contact-depth estimation.
        tip_area_function: Optional tip area function used for contact-area
            estimation. When omitted, defaults to `24.5 * hc^2`.
        stem: Optional experiment label propagated by higher-level wrappers.

    Returns:
        Result object containing fitted unloading diagnostics. Invalid or
        incomplete unloading segments are returned as unsuccessful results
        instead of raising, except for malformed input arguments.
    """

    if unloading_start_trace_index < 0:
        raise ValueError(
            "unloading_start_trace_index must refer to a valid sample."
        )
    resolved_tip_area_function = (
        _DEFAULT_TIP_AREA_FUNCTION
        if tip_area_function is None
        else tip_area_function
    )

    disp_array = np.asarray(disp_nm, dtype=np.float64)
    force_array = np.asarray(force_uN, dtype=np.float64)
    if disp_array.shape != force_array.shape:
        raise ValueError("disp_nm and force_uN must have the same shape.")
    if disp_array.ndim != 1:
        raise ValueError("Oliver-Pharr analysis requires 1D signals.")
    if len(disp_array) == 0:
        raise ValueError("Oliver-Pharr analysis requires at least one sample.")
    if fit_model not in {_LINEAR_FIT_MODEL, _POWER_LAW_FIT_MODEL}:
        raise ValueError(
            "fit_model must be 'linear_fraction' or 'power_law_full'."
        )
    if fit_num_points < 2:
        raise ValueError("fit_num_points must be at least 2.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than 0.")
    if fit_model == _LINEAR_FIT_MODEL:
        resolved_unloading_fraction = (
            0.2 if unloading_fraction is None else float(unloading_fraction)
        )
        if not 0.0 < resolved_unloading_fraction <= 1.0:
            raise ValueError(
                "unloading_fraction must lie in the interval (0, 1]."
            )
    else:
        if unloading_fraction is not None:
            raise ValueError(
                "unloading_fraction is only supported for "
                "fit_model='linear_fraction'."
            )
        resolved_unloading_fraction = None

    frozen_smoothing = _freeze_mapping(smoothing)
    if frozen_smoothing is None:
        active_disp = disp_array.copy()
        active_force = force_array.copy()
    else:
        smoothing_kwargs = dict(frozen_smoothing)
        active_disp = savgol(disp_array, **smoothing_kwargs)
        active_force = savgol(force_array, **smoothing_kwargs)

    disp_correction = None if onset_disp_nm is None else float(onset_disp_nm)
    force_correction = (
        None if baseline_offset_uN is None else float(baseline_offset_uN)
    )
    corrected_disp = _apply_correction(
        active_disp,
        correction=disp_correction,
    )
    corrected_force = _apply_correction(
        active_force,
        correction=force_correction,
    )

    evaluation_force = float(corrected_force[0])
    evaluation_disp = float(corrected_disp[0])
    if len(corrected_force) < 2:
        result = _failed_result(
            stem=stem,
            reason="no_unloading_branch",
            fit_model=fit_model,
            evaluation_index=unloading_start_trace_index,
            evaluation_force_uN=evaluation_force,
            evaluation_disp_nm=evaluation_disp,
            unloading_start_index=unloading_start_trace_index,
            unloading_end_index=unloading_start_trace_index,
            fit_point_count=1,
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
            disp_correction_nm=disp_correction,
            force_correction_uN=force_correction,
        )
        return _attach_hardness(
            result,
            onset_disp_nm=onset_disp_nm,
            epsilon=epsilon,
            tip_area_function=resolved_tip_area_function,
        )

    if fit_model == _LINEAR_FIT_MODEL:
        result = _fit_linear_fraction(
            active_disp=corrected_disp,
            active_force=corrected_force,
            stem=stem,
            evaluation_index=0,
            trace_index_offset=unloading_start_trace_index,
            evaluation_force_uN=evaluation_force,
            evaluation_disp_nm=evaluation_disp,
            unloading_fraction=resolved_unloading_fraction,
            fit_num_points=fit_num_points,
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
            disp_correction_nm=disp_correction,
            force_correction_uN=force_correction,
        )
    else:
        result = _fit_power_law_full(
            active_disp=corrected_disp,
            active_force=corrected_force,
            stem=stem,
            evaluation_index=0,
            trace_index_offset=unloading_start_trace_index,
            evaluation_force_uN=evaluation_force,
            evaluation_disp_nm=evaluation_disp,
            fit_num_points=fit_num_points,
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
            disp_correction_nm=disp_correction,
            force_correction_uN=force_correction,
        )
    return _attach_hardness(
        result,
        onset_disp_nm=onset_disp_nm,
        epsilon=epsilon,
        tip_area_function=resolved_tip_area_function,
    )


def _fit_linear_fraction(
    *,
    active_disp: NDArray[np.float64],
    active_force: NDArray[np.float64],
    stem: str,
    evaluation_index: int,
    trace_index_offset: int,
    evaluation_force_uN: float,
    evaluation_disp_nm: float,
    unloading_fraction: float,
    fit_num_points: int,
    used_smoothing: bool,
    smoothing: Mapping[str, Any] | None,
    disp_correction_nm: float | None,
    force_correction_uN: float | None,
) -> OliverPharrExperimentResult:
    """Fit a straight line to the early unloading fraction."""

    absolute_evaluation_index = trace_index_offset + evaluation_index
    unloading_length = len(active_force) - evaluation_index
    fit_point_count = max(
        int(np.ceil(unloading_length * unloading_fraction)), 1
    )
    local_unloading_end_index = min(
        evaluation_index + fit_point_count - 1,
        len(active_force) - 1,
    )
    absolute_unloading_end_index = (
        trace_index_offset + local_unloading_end_index
    )
    fit_disp = active_disp[evaluation_index : local_unloading_end_index + 1]
    fit_force = active_force[evaluation_index : local_unloading_end_index + 1]
    if len(fit_disp) < 5:
        return _failed_result(
            stem=stem,
            reason="too_few_unloading_points",
            fit_model=_LINEAR_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )
    if not np.isfinite(fit_disp).all() or not np.isfinite(fit_force).all():
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_model=_LINEAR_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )

    initial_slope = _estimate_slope(fit_disp, fit_force)
    initial_intercept = float(fit_force[0] - initial_slope * fit_disp[0])
    try:
        parameters, _ = curve_fit(
            _linear_model,
            fit_disp,
            fit_force,
            p0=(initial_slope, initial_intercept),
        )
    except (RuntimeError, ValueError):
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_model=_LINEAR_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )

    slope = float(parameters[0])
    intercept = float(parameters[1])
    if not np.isfinite([slope, intercept]).all():
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_model=_LINEAR_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )
    if np.isclose(slope, 0.0, atol=1e-12):
        return _failed_result(
            stem=stem,
            reason="zero_stiffness",
            fit_model=_LINEAR_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )

    depth_intercept = float(-intercept / slope)
    if not np.isfinite(depth_intercept):
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_model=_LINEAR_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
        )

    fitted_window = _linear_model(fit_disp, slope, intercept)
    r_squared = _r_squared(fit_force, fitted_window)
    x_fit = np.linspace(
        float(np.min(fit_disp)),
        float(np.max(fit_disp)),
        fit_num_points,
        dtype=np.float64,
    )
    y_fit = np.asarray(
        _linear_model(x_fit, slope, intercept), dtype=np.float64
    )
    return OliverPharrExperimentResult(
        stem=stem,
        success=True,
        reason=None,
        fit_model=_LINEAR_FIT_MODEL,
        evaluation_index=absolute_evaluation_index,
        evaluation_force_uN=evaluation_force_uN,
        evaluation_disp_nm=evaluation_disp_nm,
        unloading_start_index=absolute_evaluation_index,
        unloading_end_index=absolute_unloading_end_index,
        fit_point_count=len(fit_disp),
        used_smoothing=used_smoothing,
        smoothing=smoothing,
        disp_correction_nm=disp_correction_nm,
        force_correction_uN=force_correction_uN,
        stiffness_uN_per_nm=slope,
        r_squared=r_squared,
        linear_slope_uN_per_nm=slope,
        linear_intercept_uN=intercept,
        linear_depth_intercept_nm=depth_intercept,
        x_fit=x_fit,
        y_fit=y_fit,
    )


def _fit_power_law_full(
    *,
    active_disp: NDArray[np.float64],
    active_force: NDArray[np.float64],
    stem: str,
    evaluation_index: int,
    trace_index_offset: int,
    evaluation_force_uN: float,
    evaluation_disp_nm: float,
    fit_num_points: int,
    used_smoothing: bool,
    smoothing: Mapping[str, Any] | None,
    disp_correction_nm: float | None,
    force_correction_uN: float | None,
) -> OliverPharrExperimentResult:
    """Fit the full unloading branch with a power-law model."""

    absolute_evaluation_index = trace_index_offset + evaluation_index
    local_unloading_end_index = len(active_force) - 1
    absolute_unloading_end_index = (
        trace_index_offset + local_unloading_end_index
    )
    fit_disp = active_disp[evaluation_index:]
    fit_force = active_force[evaluation_index:]
    if len(fit_disp) < 5:
        return _failed_result(
            stem=stem,
            reason="too_few_unloading_points",
            fit_model=_POWER_LAW_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )
    if not np.isfinite(fit_disp).all() or not np.isfinite(fit_force).all():
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_model=_POWER_LAW_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )

    try:
        (
            fitted_k,
            fitted_m,
            fitted_hf,
            fitted_window,
        ) = _power_law_fit_parameters(
            fit_disp=fit_disp,
            fit_force=fit_force,
        )
    except ValueError as exc:
        return _failed_result(
            stem=stem,
            reason=str(exc),
            fit_model=_POWER_LAW_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )
    except (RuntimeError, ValueError):
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_model=_POWER_LAW_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )

    try:
        fitted_evaluation_disp = _power_law_evaluation_disp(
            evaluation_force_uN=evaluation_force_uN,
            k=fitted_k,
            m=fitted_m,
            hf_nm=fitted_hf,
        )
        stiffness = _power_law_stiffness(
            evaluation_disp_nm=fitted_evaluation_disp,
            k=fitted_k,
            m=fitted_m,
            hf_nm=fitted_hf,
        )
    except ValueError as exc:
        return _failed_result(
            stem=stem,
            reason=str(exc),
            fit_model=_POWER_LAW_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )
    if not np.isfinite(stiffness):
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_model=_POWER_LAW_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )
    if np.isclose(stiffness, 0.0, atol=1e-12):
        return _failed_result(
            stem=stem,
            reason="zero_stiffness",
            fit_model=_POWER_LAW_FIT_MODEL,
            evaluation_index=absolute_evaluation_index,
            evaluation_force_uN=evaluation_force_uN,
            evaluation_disp_nm=evaluation_disp_nm,
            unloading_start_index=absolute_evaluation_index,
            unloading_end_index=absolute_unloading_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=used_smoothing,
            smoothing=smoothing,
            disp_correction_nm=disp_correction_nm,
            force_correction_uN=force_correction_uN,
        )

    r_squared = _r_squared(fit_force, fitted_window)
    x_fit = np.linspace(
        float(np.min(fit_disp)),
        float(np.max(fit_disp)),
        fit_num_points,
        dtype=np.float64,
    )
    y_fit = np.asarray(
        _power_law_model(x_fit, fitted_k, fitted_m, fitted_hf),
        dtype=np.float64,
    )
    return OliverPharrExperimentResult(
        stem=stem,
        success=True,
        reason=None,
        fit_model=_POWER_LAW_FIT_MODEL,
        evaluation_index=absolute_evaluation_index,
        evaluation_force_uN=evaluation_force_uN,
        evaluation_disp_nm=fitted_evaluation_disp,
        unloading_start_index=absolute_evaluation_index,
        unloading_end_index=absolute_unloading_end_index,
        fit_point_count=len(fit_disp),
        used_smoothing=used_smoothing,
        smoothing=smoothing,
        disp_correction_nm=disp_correction_nm,
        force_correction_uN=force_correction_uN,
        stiffness_uN_per_nm=stiffness,
        r_squared=r_squared,
        power_law_k=fitted_k,
        power_law_m=fitted_m,
        power_law_hf_nm=fitted_hf,
        x_fit=x_fit,
        y_fit=y_fit,
    )


def _power_law_fit_parameters(
    *,
    fit_disp: NDArray[np.float64],
    fit_force: NDArray[np.float64],
) -> tuple[float, float, float, NDArray[np.float64]]:
    """Return fitted power-law parameters and fitted-window values."""

    fitted_k, fitted_m, fitted_hf = _fit_power_law_with_fitted_hf(
        fit_disp=fit_disp,
        fit_force=fit_force,
    )

    if not np.isfinite([fitted_k, fitted_m, fitted_hf]).all():
        raise ValueError("fit_failed")
    fitted_window = np.asarray(
        _power_law_model(fit_disp, fitted_k, fitted_m, fitted_hf),
        dtype=np.float64,
    )
    if not np.isfinite(fitted_window).all():
        raise ValueError("fit_failed")
    return fitted_k, fitted_m, fitted_hf, fitted_window


def _fit_power_law_with_fitted_hf(
    *,
    fit_disp: NDArray[np.float64],
    fit_force: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Fit `k`, `m`, and `hf` simultaneously."""

    min_disp = float(np.min(fit_disp))
    max_disp = float(np.max(fit_disp))
    span = max(max_disp - min_disp, 1e-6)
    initial_hf = min_disp - span
    initial_delta = np.asarray(fit_disp - initial_hf, dtype=np.float64)
    initial_k, initial_m = _power_law_initial_guess(
        fit_force=fit_force,
        delta=initial_delta,
    )
    parameters, _ = curve_fit(
        _power_law_model,
        fit_disp,
        fit_force,
        p0=(initial_k, initial_m, initial_hf),
        bounds=(
            (0.0, 0.1, min_disp - 100.0 * span),
            (np.inf, 10.0, min_disp),
        ),
        maxfev=20000,
    )
    return float(parameters[0]), float(parameters[1]), float(parameters[2])


def _power_law_initial_guess(
    *,
    fit_force: NDArray[np.float64],
    delta: NDArray[np.float64],
) -> tuple[float, float]:
    """Return a stable initial guess for the power-law fit."""

    initial_m = 1.5
    reference_force = max(float(np.nanmax(fit_force)), 1e-6)
    reference_delta = max(float(delta[0]), 1e-6)
    initial_k = max(reference_force / (reference_delta**initial_m), 1e-12)
    return initial_k, initial_m


def _failed_result(
    *,
    stem: str,
    reason: str,
    fit_model: str,
    evaluation_index: int,
    evaluation_force_uN: float | None,
    evaluation_disp_nm: float | None,
    unloading_start_index: int,
    unloading_end_index: int,
    fit_point_count: int = 0,
    used_smoothing: bool,
    smoothing: Mapping[str, Any] | None,
    disp_correction_nm: float | None,
    force_correction_uN: float | None,
) -> OliverPharrExperimentResult:
    """Build a standardized unsuccessful analysis result."""

    return OliverPharrExperimentResult(
        stem=stem,
        success=False,
        reason=reason,
        fit_model=fit_model,
        evaluation_index=evaluation_index,
        evaluation_force_uN=evaluation_force_uN,
        evaluation_disp_nm=evaluation_disp_nm,
        unloading_start_index=unloading_start_index,
        unloading_end_index=unloading_end_index,
        fit_point_count=fit_point_count,
        used_smoothing=used_smoothing,
        smoothing=smoothing,
        disp_correction_nm=disp_correction_nm,
        force_correction_uN=force_correction_uN,
        stiffness_uN_per_nm=None,
        r_squared=None,
        x_fit=np.empty(0, dtype=np.float64),
        y_fit=np.empty(0, dtype=np.float64),
    )


def _attach_hardness(
    result: OliverPharrExperimentResult,
    *,
    onset_disp_nm: float | None,
    epsilon: float,
    tip_area_function: TipAreaFunction,
) -> OliverPharrExperimentResult:
    """Return an Oliver-Pharr result with hardness diagnostics attached."""

    if onset_disp_nm is None:
        return replace(
            result,
            epsilon=float(epsilon),
            onset_disp_nm=None,
            tip_area_function=tip_area_function,
            hardness_success=False,
            hardness_reason="missing_onset",
        )
    if result.evaluation_force_uN is None:
        return replace(
            result,
            epsilon=float(epsilon),
            onset_disp_nm=float(onset_disp_nm),
            tip_area_function=tip_area_function,
            hardness_success=False,
            hardness_reason="missing_evaluation_force",
        )
    if result.evaluation_disp_nm is None:
        return replace(
            result,
            epsilon=float(epsilon),
            onset_disp_nm=float(onset_disp_nm),
            tip_area_function=tip_area_function,
            hardness_success=False,
            hardness_reason="missing_evaluation_disp",
        )
    if result.stiffness_uN_per_nm is None:
        return replace(
            result,
            epsilon=float(epsilon),
            onset_disp_nm=float(onset_disp_nm),
            tip_area_function=tip_area_function,
            hardness_success=False,
            hardness_reason="missing_stiffness",
        )

    hmax_nm = float(result.evaluation_disp_nm)
    if not np.isfinite(hmax_nm) or hmax_nm <= 0.0:
        return replace(
            result,
            epsilon=float(epsilon),
            onset_disp_nm=float(onset_disp_nm),
            hmax_nm=hmax_nm,
            tip_area_function=tip_area_function,
            hardness_success=False,
            hardness_reason="invalid_hmax",
        )

    contact_depth_nm = float(
        hmax_nm
        - epsilon * result.evaluation_force_uN / result.stiffness_uN_per_nm
    )
    if not np.isfinite(contact_depth_nm) or contact_depth_nm <= 0.0:
        return replace(
            result,
            epsilon=float(epsilon),
            onset_disp_nm=float(onset_disp_nm),
            hmax_nm=hmax_nm,
            contact_depth_nm=contact_depth_nm,
            tip_area_function=tip_area_function,
            hardness_success=False,
            hardness_reason="invalid_contact_depth",
        )

    contact_area_nm2 = tip_area_function.evaluate(contact_depth_nm)
    if not np.isfinite(contact_area_nm2) or contact_area_nm2 <= 0.0:
        return replace(
            result,
            epsilon=float(epsilon),
            onset_disp_nm=float(onset_disp_nm),
            hmax_nm=hmax_nm,
            contact_depth_nm=contact_depth_nm,
            contact_area_nm2=contact_area_nm2,
            tip_area_function=tip_area_function,
            hardness_success=False,
            hardness_reason="invalid_contact_area",
        )

    hardness_uN_per_nm2 = float(result.evaluation_force_uN / contact_area_nm2)
    reduced_modulus_uN_per_nm2 = float(
        0.5
        * np.sqrt(np.pi)
        * result.stiffness_uN_per_nm
        / np.sqrt(contact_area_nm2)
    )
    return replace(
        result,
        epsilon=float(epsilon),
        onset_disp_nm=float(onset_disp_nm),
        hmax_nm=hmax_nm,
        contact_depth_nm=contact_depth_nm,
        contact_area_nm2=contact_area_nm2,
        hardness_uN_per_nm2=hardness_uN_per_nm2,
        reduced_modulus_uN_per_nm2=reduced_modulus_uN_per_nm2,
        tip_area_function=tip_area_function,
        hardness_success=True,
        hardness_reason=None,
    )


def _freeze_mapping(
    mapping: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return an immutable copy of an optional keyword mapping."""

    if mapping is None:
        return None
    return MappingProxyType(dict(mapping))


def _apply_correction(
    values: NDArray[np.float64],
    *,
    correction: float | None,
) -> NDArray[np.float64]:
    """Return a corrected copy of one signal."""

    if correction is None:
        return values.copy()
    return np.asarray(values - correction, dtype=np.float64)


def _linear_model(
    x_values: NDArray[np.float64], slope: float, intercept: float
) -> NDArray[np.float64]:
    """Return a straight-line model."""

    return slope * x_values + intercept


def _power_law_model(
    x_values: NDArray[np.float64] | ArrayLike,
    k: float,
    m: float,
    hf_nm: float,
) -> NDArray[np.float64]:
    """Return the Oliver-Pharr power-law unloading model."""

    delta = np.asarray(x_values, dtype=np.float64) - float(hf_nm)
    if np.any(delta < 0.0):
        raise ValueError("invalid_power_law_domain")
    return float(k) * np.power(delta, float(m))


def _power_law_stiffness(
    *,
    evaluation_disp_nm: float,
    k: float,
    m: float,
    hf_nm: float,
) -> float:
    """Return the derivative of the power-law model at one displacement."""

    delta = float(evaluation_disp_nm - hf_nm)
    if delta <= 0.0:
        raise ValueError("invalid_power_law_domain")
    return float(k * m * np.power(delta, m - 1.0))


def _power_law_evaluation_disp(
    *,
    evaluation_force_uN: float,
    k: float,
    m: float,
    hf_nm: float,
) -> float:
    """Return the fitted displacement where the model reaches one load."""

    if not np.isfinite([evaluation_force_uN, k, m, hf_nm]).all():
        raise ValueError("fit_failed")
    if evaluation_force_uN < 0.0:
        raise ValueError("invalid_power_law_domain")
    if k <= 0.0 or m <= 0.0:
        raise ValueError("fit_failed")
    return float(hf_nm + np.power(evaluation_force_uN / k, 1.0 / m))


def _estimate_slope(
    x_values: NDArray[np.float64], y_values: NDArray[np.float64]
) -> float:
    """Estimate an initial slope from the endpoints of one fit window."""

    delta_x = float(x_values[-1] - x_values[0])
    if np.isclose(delta_x, 0.0, atol=1e-12):
        return 0.0
    return float((y_values[-1] - y_values[0]) / delta_x)


def _r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Return the coefficient of determination for one fitted window."""

    residual_sum = float(np.sum((observed - predicted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if np.isclose(total_sum, 0.0, atol=1e-12):
        return 1.0 if np.isclose(residual_sum, 0.0, atol=1e-12) else 0.0
    return float(1.0 - residual_sum / total_sum)
