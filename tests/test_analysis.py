import numpy as np
import pytest

from nanodent.analysis.align import align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.analysis.fit import curve_fit_model


def test_savgol_preserves_shape_and_reduces_noise() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 101)
    clean = np.sin(x)
    noisy = clean + 0.2 * np.sin(11.0 * x)

    smoothed = savgol(noisy, window_length=21, polyorder=3)

    assert smoothed.shape == noisy.shape
    assert np.mean(np.abs(smoothed - clean)) < np.mean(np.abs(noisy - clean))


def test_gradient_matches_quadratic_derivative() -> None:
    x = np.linspace(-3.0, 3.0, 401)
    y = x**2

    dy_dx = gradient(y, x)

    assert dy_dx[200] == pytest.approx(0.0, abs=1e-3)
    assert dy_dx[320] == pytest.approx(2.0 * x[320], rel=1e-2)


def test_force_threshold_alignment_detects_delayed_onset() -> None:
    x = np.linspace(0.0, 10.0, 501)
    y = np.where(x < 4.0, 0.0, 5.0 * (x - 4.0))

    result = align_curve(x, y, method="force_threshold", force_threshold=1.0)

    assert result.anchor_x == pytest.approx(4.2, abs=0.05)
    assert result.shifted_x[result.anchor_index] == pytest.approx(0.0)


def test_savgol_force_slope_alignment_detects_ramp_start() -> None:
    x = np.linspace(0.0, 10.0, 1001)
    y = np.zeros_like(x)
    ramp_mask = x >= 5.0
    y[ramp_mask] = 20.0 * (x[ramp_mask] - 5.0)

    result = align_curve(
        x, y, method="savgol_force_slope", window_length=41, polyorder=2
    )

    assert result.anchor_x == pytest.approx(5.0, abs=0.2)
    assert result.shifted_x[result.anchor_index] == pytest.approx(0.0)


def test_curve_fit_model_smoke_test() -> None:
    x = np.linspace(0.0, 10.0, 101)
    y = 2.0 * x + 1.5

    def linear_model(
        x_values: np.ndarray, slope: float, intercept: float
    ) -> np.ndarray:
        return slope * x_values + intercept

    result = curve_fit_model(x, y, linear_model, p0=[1.0, 0.0])

    assert result.success is True
    assert result.parameters[0] == pytest.approx(2.0, rel=1e-3)
    assert result.parameters[1] == pytest.approx(1.5, rel=1e-3)
    assert len(result.x_fit) == 200
