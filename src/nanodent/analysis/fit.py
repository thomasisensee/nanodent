"""Curve-fitting extension surface for later analysis workflows."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

ModelFunction = Callable[..., ArrayLike]


@dataclass(frozen=True, slots=True)
class FitResult:
    """Result of a SciPy-backed curve fit."""

    parameters: NDArray[np.float64]
    covariance: NDArray[np.float64] | None
    x_fit: NDArray[np.float64]
    y_fit: NDArray[np.float64]
    success: bool
    model_name: str


def curve_fit_model(
    x: ArrayLike,
    y: ArrayLike,
    model: ModelFunction,
    *,
    p0: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    num_points: int = 200,
) -> FitResult:
    """Fit a model with SciPy and return dense fitted coordinates.

    Args:
        x: Input x-axis values.
        y: Observed y-axis values to fit.
        model: Callable model accepted by `scipy.optimize.curve_fit`.
        p0: Optional initial parameter guess.
        bounds: Lower and upper parameter bounds.
        num_points: Number of points used for the dense fitted curve.

    Returns:
        Fit result containing optimized parameters, covariance, and dense fit
        coordinates.
    """

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    parameters, covariance = curve_fit(
        model, x_array, y_array, p0=p0, bounds=bounds
    )
    x_fit = np.linspace(
        float(np.min(x_array)),
        float(np.max(x_array)),
        num_points,
        dtype=np.float64,
    )
    y_fit = np.asarray(model(x_fit, *parameters), dtype=np.float64)
    return FitResult(
        parameters=np.asarray(parameters, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        x_fit=x_fit,
        y_fit=y_fit,
        success=True,
        model_name=getattr(model, "__name__", model.__class__.__name__),
    )
