import numpy as np
import pytest

from nanodent.analysis.align import align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.analysis.fit import curve_fit_model
from nanodent.analysis.quality import (
    classify_flat_force,
    classify_gradual_onset,
    classify_outlier_jumps,
    classify_quality,
)


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


def test_classify_gradual_onset_detects_gradual_rise() -> None:
    x = np.linspace(0.0, 100.0, 1201)
    y = np.zeros_like(x)
    ramp_mask = (x >= 20.0) & (x < 90.0)
    y[ramp_mask] = 3000.0 * (x[ramp_mask] - 20.0) / 70.0
    y[x >= 90.0] = 3000.0

    result = classify_gradual_onset(
        x,
        y,
        bin_count=20,
        baseline_bin_count=3,
        onset_force_fraction=0.1,
        target_force_fraction=0.5,
        sustained_bins=2,
        max_rise_width_fraction=0.2,
    )

    assert result.enabled is False
    assert result.reason == "gradual_onset"
    assert result.rise_width_fraction == pytest.approx(0.28, abs=0.06)


def test_classify_gradual_onset_keeps_sharp_rise_enabled() -> None:
    x = np.linspace(0.0, 100.0, 1201)
    y = np.zeros_like(x)
    ramp_mask = (x >= 38.0) & (x < 45.0)
    y[ramp_mask] = 3000.0 * (x[ramp_mask] - 38.0) / 7.0
    y[x >= 45.0] = 3000.0

    result = classify_gradual_onset(
        x,
        y,
        bin_count=20,
        baseline_bin_count=3,
        onset_force_fraction=0.1,
        target_force_fraction=0.5,
        sustained_bins=2,
        max_rise_width_fraction=0.2,
    )

    assert result.enabled is True
    assert result.reason is None
    assert result.rise_width_fraction == pytest.approx(0.025, abs=0.03)


def test_classify_flat_force_detects_plateau_signal() -> None:
    y = np.full(1500, 500.0) + 8.0 * np.sin(np.linspace(0.0, 8.0, 1500))

    result = classify_flat_force(y, min_robust_force_span_uN=50.0)

    assert result.enabled is False
    assert result.reason == "flat_force"


def test_classify_outlier_jumps_detects_isolated_disp_spike() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = np.linspace(0.0, 3000.0, 1200)
    x[600] = 900.0

    result = classify_outlier_jumps(x, y, disp_z_threshold=50.0)

    assert result.enabled is False
    assert result.reason == "outlier_disp"


def test_classify_outlier_jumps_detects_isolated_force_spike() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = np.linspace(0.0, 3000.0, 1200)
    y[600] = -800.0

    result = classify_outlier_jumps(x, y, force_z_threshold=50.0)

    assert result.enabled is False
    assert result.reason == "outlier_force"


def test_classify_quality_prioritizes_flat_force_before_onset() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = np.full_like(x, 500.0) + 6.0 * np.sin(x / 4.0)

    result = classify_quality(x, y, min_robust_force_span_uN=50.0)

    assert result.enabled is False
    assert result.reason == "flat_force"


def test_classify_quality_detects_outlier_before_gradual_onset() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = np.zeros_like(x)
    ramp_mask = (x >= 20.0) & (x < 90.0)
    y[ramp_mask] = 3000.0 * (x[ramp_mask] - 20.0) / 70.0
    y[x >= 90.0] = 3000.0
    x[600] = 900.0

    result = classify_quality(x, y, disp_z_threshold=50.0)

    assert result.enabled is False
    assert result.reason == "outlier_disp"
