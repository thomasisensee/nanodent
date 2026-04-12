"""Hertzian-style onloading analysis helpers."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from nanodent.analysis.filters import savgol


@dataclass(frozen=True, slots=True)
class HertzianExperimentResult:
    """Result of one Hertzian onloading fit."""

    stem: str = ""
    success: bool = False
    reason: str | None = None
    fit_start_index: int = 0
    fit_end_index: int = 0
    fit_point_count: int = 0
    used_smoothing: bool = False
    smoothing: Mapping[str, Any] | None = None
    initial_onset_disp_nm: float | None = None
    force_correction_uN: float | None = None
    amplitude_uN_per_nm_3_2: float | None = None
    h_onset_nm: float | None = None
    reduced_modulus_uN_per_nm2: float | None = None
    radius_nm: float | None = None
    pop_in_load_uN: float | None = None
    tau_max_uN_per_nm2: float | None = None
    r_squared: float | None = None
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
            "fit_start_index": self.fit_start_index,
            "fit_end_index": self.fit_end_index,
            "fit_point_count": self.fit_point_count,
            "initial_onset_disp_nm": self.initial_onset_disp_nm,
            "force_correction_uN": self.force_correction_uN,
            "amplitude_uN_per_nm_3_2": self.amplitude_uN_per_nm_3_2,
            "h_onset_nm": self.h_onset_nm,
            "reduced_modulus_uN_per_nm2": self.reduced_modulus_uN_per_nm2,
            "radius_nm": self.radius_nm,
            "pop_in_load_uN": self.pop_in_load_uN,
            "tau_max_uN_per_nm2": self.tau_max_uN_per_nm2,
            "r_squared": self.r_squared,
        }


def calculate_hertzian_radius(
    hertzian_amplitude_uN_per_nm_3_2: float,
    reduced_modulus_uN_per_nm2: float,
) -> float:
    """Return Hertzian tip radius implied by amplitude and reduced modulus."""

    amplitude = _positive_finite(
        hertzian_amplitude_uN_per_nm_3_2,
        name="hertzian_amplitude_uN_per_nm_3_2",
    )
    reduced_modulus = _positive_finite(
        reduced_modulus_uN_per_nm2,
        name="reduced_modulus_uN_per_nm2",
    )
    return float(np.power(3.0 * amplitude / (4.0 * reduced_modulus), 2.0))


def calculate_tau_max(
    reduced_modulus_uN_per_nm2: float,
    hertzian_amplitude_uN_per_nm_3_2: float,
    pop_in_load_uN: float,
) -> float:
    """Return the maximum shear stress from OP, Hertzian, and pop-in values."""

    reduced_modulus = _positive_finite(
        reduced_modulus_uN_per_nm2,
        name="reduced_modulus_uN_per_nm2",
    )
    amplitude = _positive_finite(
        hertzian_amplitude_uN_per_nm_3_2,
        name="hertzian_amplitude_uN_per_nm_3_2",
    )
    pop_in_load = _positive_finite(
        pop_in_load_uN,
        name="pop_in_load_uN",
    )
    inner = (
        512.0
        * pop_in_load
        * np.power(reduced_modulus, 6.0)
        / (27.0 * np.power(amplitude, 4.0))
    )
    return float((0.31 / np.pi) * np.power(inner, 1.0 / 3.0))


def analyze_hertzian(
    disp_nm: ArrayLike,
    force_uN: ArrayLike,
    *,
    fit_end_index: int,
    smoothing: Mapping[str, Any] | None = None,
    fit_num_points: int = 200,
    initial_onset_disp_nm: float | None = None,
    baseline_offset_uN: float | None = None,
    reduced_modulus_uN_per_nm2: float | None = None,
    pop_in_load_uN: float | None = None,
    stem: str = "",
) -> HertzianExperimentResult:
    """Fit a Hertzian onloading model up to a force-peak endpoint.

    Args:
        disp_nm: Displacement values in acquisition order.
        force_uN: Force values in acquisition order.
        fit_end_index: Inclusive local index of the first force peak.
        smoothing: Optional keyword arguments forwarded to `nanodent.savgol`.
        fit_num_points: Number of dense fitted points for plotting.
        initial_onset_disp_nm: Optional initial guess for the fitted onset.
        baseline_offset_uN: Optional force baseline offset subtracted before
            fitting.
        reduced_modulus_uN_per_nm2: Optional Oliver-Pharr reduced modulus used
            to derive the Hertzian radius and maximum shear stress.
        pop_in_load_uN: Optional second force-peak load used to derive maximum
            shear stress.
        stem: Optional experiment label propagated by higher-level wrappers.

    Returns:
        Result object containing fitted Hertzian diagnostics. Invalid or
        incomplete fit windows are returned as unsuccessful results instead
        of raising, except for malformed input arguments.
    """

    disp_array = np.asarray(disp_nm, dtype=np.float64)
    force_array = np.asarray(force_uN, dtype=np.float64)
    if disp_array.shape != force_array.shape:
        raise ValueError("disp_nm and force_uN must have the same shape.")
    if disp_array.ndim != 1:
        raise ValueError("Hertzian analysis requires 1D signals.")
    if len(disp_array) == 0:
        raise ValueError("Hertzian analysis requires at least one sample.")
    if fit_num_points < 2:
        raise ValueError("fit_num_points must be at least 2.")
    if fit_end_index < 0 or fit_end_index >= len(disp_array):
        raise ValueError("fit_end_index must refer to a valid sample.")

    frozen_smoothing = _freeze_mapping(smoothing)
    if frozen_smoothing is None:
        active_disp = disp_array.copy()
        active_force = force_array.copy()
    else:
        smoothing_kwargs = dict(frozen_smoothing)
        active_disp = savgol(disp_array, **smoothing_kwargs)
        active_force = savgol(force_array, **smoothing_kwargs)

    force_correction = (
        None if baseline_offset_uN is None else float(baseline_offset_uN)
    )
    corrected_force = _apply_correction(
        active_force,
        correction=force_correction,
    )
    initial_onset = (
        None if initial_onset_disp_nm is None else float(initial_onset_disp_nm)
    )

    fit_disp = active_disp[: fit_end_index + 1]
    fit_force = corrected_force[: fit_end_index + 1]
    if len(fit_disp) < 5:
        return _failed_result(
            stem=stem,
            reason="too_few_onloading_points",
            fit_end_index=fit_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
            initial_onset_disp_nm=initial_onset,
            force_correction_uN=force_correction,
        )
    if not np.isfinite(fit_disp).all() or not np.isfinite(fit_force).all():
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_end_index=fit_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
            initial_onset_disp_nm=initial_onset,
            force_correction_uN=force_correction,
        )

    try:
        amplitude, h_onset = _fit_hertzian_parameters(
            fit_disp=fit_disp,
            fit_force=fit_force,
            initial_onset_disp_nm=initial_onset,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_end_index=fit_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
            initial_onset_disp_nm=initial_onset,
            force_correction_uN=force_correction,
        )
    if not np.isfinite([amplitude, h_onset]).all() or amplitude < 0.0:
        return _failed_result(
            stem=stem,
            reason="fit_failed",
            fit_end_index=fit_end_index,
            fit_point_count=len(fit_disp),
            used_smoothing=frozen_smoothing is not None,
            smoothing=frozen_smoothing,
            initial_onset_disp_nm=initial_onset,
            force_correction_uN=force_correction,
        )

    fitted_window = _hertzian_model(fit_disp, amplitude, h_onset)
    r_squared = _r_squared(fit_force, fitted_window)
    reduced_modulus = _optional_positive_finite(
        reduced_modulus_uN_per_nm2,
        name="reduced_modulus_uN_per_nm2",
    )
    pop_in_load = _optional_positive_finite(
        pop_in_load_uN,
        name="pop_in_load_uN",
    )
    radius = _calculate_radius_or_none(
        amplitude,
        reduced_modulus,
    )
    tau_max = _calculate_tau_max_or_none(
        reduced_modulus,
        amplitude,
        pop_in_load,
    )
    fit_end_disp = float(fit_disp[-1])
    x_fit_start = min(float(h_onset), fit_end_disp)
    x_fit = np.linspace(
        x_fit_start,
        fit_end_disp,
        fit_num_points,
        dtype=np.float64,
    )
    y_fit = np.asarray(
        _hertzian_model(x_fit, amplitude, h_onset),
        dtype=np.float64,
    )
    return HertzianExperimentResult(
        stem=stem,
        success=True,
        reason=None,
        fit_start_index=0,
        fit_end_index=fit_end_index,
        fit_point_count=len(fit_disp),
        used_smoothing=frozen_smoothing is not None,
        smoothing=frozen_smoothing,
        initial_onset_disp_nm=initial_onset,
        force_correction_uN=force_correction,
        amplitude_uN_per_nm_3_2=amplitude,
        h_onset_nm=h_onset,
        reduced_modulus_uN_per_nm2=reduced_modulus,
        radius_nm=radius,
        pop_in_load_uN=pop_in_load,
        tau_max_uN_per_nm2=tau_max,
        r_squared=r_squared,
        x_fit=x_fit,
        y_fit=y_fit,
    )


def missing_force_peak_result(
    *,
    stem: str = "",
    initial_onset_disp_nm: float | None = None,
    baseline_offset_uN: float | None = None,
) -> HertzianExperimentResult:
    """Return a standardized result for a missing force-peak dependency."""

    return _failed_result(
        stem=stem,
        reason="missing_force_peak",
        fit_end_index=0,
        fit_point_count=0,
        used_smoothing=False,
        smoothing=None,
        initial_onset_disp_nm=initial_onset_disp_nm,
        force_correction_uN=baseline_offset_uN,
    )


def _fit_hertzian_parameters(
    *,
    fit_disp: NDArray[np.float64],
    fit_force: NDArray[np.float64],
    initial_onset_disp_nm: float | None,
) -> tuple[float, float]:
    """Return fitted Hertzian amplitude and onset displacement."""

    min_disp = float(np.min(fit_disp))
    max_disp = float(np.max(fit_disp))
    span = max(max_disp - min_disp, 1.0)
    initial_h_onset = (
        min_disp if initial_onset_disp_nm is None else initial_onset_disp_nm
    )
    lower_h_onset = min_disp - 10.0 * span
    upper_h_onset = max_disp
    initial_h_onset = float(
        np.clip(initial_h_onset, lower_h_onset, upper_h_onset)
    )
    reference_force = max(float(np.nanmax(fit_force)), 1e-6)
    reference_delta = max(float(max_disp - initial_h_onset), 1.0)
    initial_amplitude = max(
        reference_force / np.power(reference_delta, 1.5),
        1e-12,
    )
    parameters, _ = curve_fit(
        _hertzian_model,
        fit_disp,
        fit_force,
        p0=(initial_amplitude, initial_h_onset),
        bounds=((0.0, lower_h_onset), (np.inf, upper_h_onset)),
        maxfev=20000,
    )
    return float(parameters[0]), float(parameters[1])


def _calculate_radius_or_none(
    amplitude_uN_per_nm_3_2: float,
    reduced_modulus_uN_per_nm2: float | None,
) -> float | None:
    """Return derived Hertzian radius when its dependency is usable."""

    if reduced_modulus_uN_per_nm2 is None:
        return None
    try:
        return calculate_hertzian_radius(
            amplitude_uN_per_nm_3_2,
            reduced_modulus_uN_per_nm2,
        )
    except ValueError:
        return None


def _calculate_tau_max_or_none(
    reduced_modulus_uN_per_nm2: float | None,
    amplitude_uN_per_nm_3_2: float,
    pop_in_load_uN: float | None,
) -> float | None:
    """Return derived maximum shear stress when dependencies are usable."""

    if reduced_modulus_uN_per_nm2 is None or pop_in_load_uN is None:
        return None
    try:
        return calculate_tau_max(
            reduced_modulus_uN_per_nm2,
            amplitude_uN_per_nm_3_2,
            pop_in_load_uN,
        )
    except ValueError:
        return None


def _failed_result(
    *,
    stem: str,
    reason: str,
    fit_end_index: int,
    fit_point_count: int,
    used_smoothing: bool,
    smoothing: Mapping[str, Any] | None,
    initial_onset_disp_nm: float | None,
    force_correction_uN: float | None,
) -> HertzianExperimentResult:
    """Build a standardized unsuccessful analysis result."""

    return HertzianExperimentResult(
        stem=stem,
        success=False,
        reason=reason,
        fit_start_index=0,
        fit_end_index=fit_end_index,
        fit_point_count=fit_point_count,
        used_smoothing=used_smoothing,
        smoothing=smoothing,
        initial_onset_disp_nm=initial_onset_disp_nm,
        force_correction_uN=force_correction_uN,
        x_fit=np.empty(0, dtype=np.float64),
        y_fit=np.empty(0, dtype=np.float64),
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


def _hertzian_model(
    x_values: NDArray[np.float64] | ArrayLike,
    amplitude: float,
    h_onset_nm: float,
) -> NDArray[np.float64]:
    """Return the clamped Hertzian onloading model."""

    delta = np.maximum(
        np.asarray(x_values, dtype=np.float64) - float(h_onset_nm),
        0.0,
    )
    return float(amplitude) * np.power(delta, 1.5)


def _r_squared(
    observed: NDArray[np.float64], predicted: NDArray[np.float64]
) -> float:
    """Return the coefficient of determination for one fitted window."""

    residual_sum = float(np.sum((observed - predicted) ** 2))
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if np.isclose(total_sum, 0.0, atol=1e-12):
        return 1.0 if np.isclose(residual_sum, 0.0, atol=1e-12) else 0.0
    return float(1.0 - residual_sum / total_sum)


def _optional_positive_finite(
    value: float | None,
    *,
    name: str,
) -> float | None:
    """Return an optional dependency only when it is usable."""

    if value is None:
        return None
    try:
        return _positive_finite(value, name=name)
    except (TypeError, ValueError):
        return None


def _positive_finite(value: float, *, name: str) -> float:
    """Return a finite positive float or raise a descriptive error."""

    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and greater than 0.")
    return parsed
