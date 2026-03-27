import numpy as np
import pytest

from nanodent.analysis.align import align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.analysis.fit import curve_fit_model
from nanodent.analysis.oliver_pharr import analyze_oliver_pharr
from nanodent.analysis.quality import (
    classify_flat_force,
    classify_gradual_onset,
    classify_high_displacement,
    classify_outlier_jumps,
    classify_peak_balance,
    classify_quality,
)


def _make_linear_unloading_curve() -> tuple[np.ndarray, np.ndarray]:
    load_disp = np.linspace(0.0, 100.0, 101)
    load_force = 0.01 * load_disp**2
    unload_disp = np.linspace(100.0, 80.0, 41)[1:]
    unload_force = 5.0 * unload_disp - 400.0
    return (
        np.concatenate([load_disp, unload_disp]),
        np.concatenate([load_force, unload_force]),
    )


def _make_spiky_peak_curve() -> tuple[np.ndarray, np.ndarray]:
    load_disp = np.linspace(0.0, 100.0, 101)
    load_force = 0.5 * load_disp
    unload_disp = np.linspace(100.0, 60.0, 41)[1:]
    unload_force = 1.25 * unload_disp - 75.0
    disp = np.concatenate([load_disp, unload_disp])
    force = np.concatenate([load_force, unload_force])
    disp[70] += 4.0
    force[70] += 25.0
    return disp, force


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


def test_analyze_oliver_pharr_fits_linear_unloading_branch() -> None:
    x, y = _make_linear_unloading_curve()

    result = analyze_oliver_pharr(x, y, unloading_fraction=0.25)

    assert result.success is True
    assert result.reason is None
    assert result.used_smoothing is False
    assert result.peak_index == 100
    assert result.peak_force_uN == pytest.approx(100.0)
    assert result.peak_disp_nm == pytest.approx(100.0)
    assert result.unloading_start_index == 100
    assert result.unloading_end_index == 110
    assert result.fit_point_count == 11
    assert result.stiffness_uN_per_nm == pytest.approx(5.0, rel=1e-4)
    assert result.force_intercept_uN == pytest.approx(-400.0, rel=1e-4)
    assert result.depth_intercept_nm == pytest.approx(80.0, rel=1e-4)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert len(result.x_fit) == 200
    assert len(result.y_fit) == 200


def test_analyze_oliver_pharr_uses_smoothed_signals_for_peak_detection() -> (
    None
):
    x, y = _make_spiky_peak_curve()

    raw_result = analyze_oliver_pharr(x, y, unloading_fraction=0.25)
    smoothed_result = analyze_oliver_pharr(
        x,
        y,
        unloading_fraction=0.25,
        smoothing={"window_length": 21, "polyorder": 2},
    )

    assert raw_result.peak_index == 70
    assert smoothed_result.success is True
    assert smoothed_result.used_smoothing is True
    assert dict(smoothed_result.smoothing or {}) == {
        "window_length": 21,
        "polyorder": 2,
    }
    assert smoothed_result.peak_index == pytest.approx(100, abs=2)


def test_analyze_oliver_pharr_rejects_invalid_unloading_fraction() -> None:
    x, y = _make_linear_unloading_curve()

    with pytest.raises(ValueError, match="unloading_fraction"):
        analyze_oliver_pharr(x, y, unloading_fraction=0.0)

    with pytest.raises(ValueError, match="unloading_fraction"):
        analyze_oliver_pharr(x, y, unloading_fraction=1.1)


def test_analyze_oliver_pharr_marks_missing_unloading_branch() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    result = analyze_oliver_pharr(x, y)

    assert result.success is False
    assert result.reason == "no_unloading_branch"


def test_analyze_oliver_pharr_marks_too_few_unloading_points() -> None:
    x = np.array([0.0, 1.0, 2.0, 1.5, 1.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float64)

    result = analyze_oliver_pharr(x, y, unloading_fraction=1.0)

    assert result.success is False
    assert result.reason == "too_few_unloading_points"


def test_analyze_oliver_pharr_marks_zero_stiffness() -> None:
    x = np.array([0.0, 1.0, 2.0, 1.9, 1.8, 1.7, 1.6], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    result = analyze_oliver_pharr(x, y, unloading_fraction=1.0)

    assert result.success is False
    assert result.reason == "zero_stiffness"


def test_analyze_oliver_pharr_marks_non_finite_window_as_fit_failure() -> None:
    x = np.array(
        [0.0, 1.0, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4],
        dtype=np.float64,
    )
    y = np.array(
        [0.0, 1.0, 5.0, 4.0, np.nan, 2.0, 1.0, 0.5, 0.0],
        dtype=np.float64,
    )

    result = analyze_oliver_pharr(x, y, unloading_fraction=1.0)

    assert result.success is False
    assert result.reason == "fit_failed"


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


def test_classify_high_displacement_disables_curve_above_limit() -> None:
    x = np.linspace(0.0, 1200.0, 1200)

    result = classify_high_displacement(x, max_disp_nm=1000.0)

    assert result.enabled is False
    assert result.reason == "high_disp"


def test_classify_peak_balance_keeps_two_established_peaks_enabled() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = 2500.0 * np.exp(-(((x - 30.0) / 8.0) ** 2))
    y += 1200.0 * np.exp(-(((x - 70.0) / 10.0) ** 2))

    result = classify_peak_balance(x, y, min_secondary_peak_fraction=0.3)

    assert result.enabled is True
    assert result.reason is None


def test_classify_peak_balance_disables_too_small_second_peak() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = 2500.0 * np.exp(-(((x - 30.0) / 8.0) ** 2))
    y += 500.0 * np.exp(-(((x - 70.0) / 10.0) ** 2))

    result = classify_peak_balance(x, y, min_secondary_peak_fraction=0.3)

    assert result.enabled is False
    assert result.reason == "weak_second_peak"


def test_classify_peak_balance_abstains_without_two_resolved_peaks() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = 2500.0 * np.exp(-(((x - 30.0) / 8.0) ** 2))

    result = classify_peak_balance(x, y, min_secondary_peak_fraction=0.3)

    assert result.enabled is True
    assert result.reason is None


def test_classify_peak_balance_can_require_two_resolved_peaks() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = 2500.0 * np.exp(-(((x - 30.0) / 8.0) ** 2))

    result = classify_peak_balance(
        x,
        y,
        min_secondary_peak_fraction=0.3,
        require_two_peaks=True,
    )

    assert result.enabled is False
    assert result.reason == "weak_second_peak"


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


def test_classify_quality_disables_high_displacement_before_onset() -> None:
    x = np.linspace(0.0, 1200.0, 1200)
    y = 2.5 * x

    result = classify_quality(x, y, max_disp_nm=1000.0)

    assert result.enabled is False
    assert result.reason == "high_disp"


def test_classify_quality_disables_weak_second_peak_before_onset() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = 2500.0 * np.exp(-(((x - 30.0) / 8.0) ** 2))
    y += 500.0 * np.exp(-(((x - 70.0) / 10.0) ** 2))

    result = classify_quality(x, y, min_secondary_peak_fraction=0.3)

    assert result.enabled is False
    assert result.reason == "weak_second_peak"


def test_classify_quality_abstains_on_single_peak_by_default() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = 2500.0 * np.exp(-(((x - 30.0) / 8.0) ** 2))

    result = classify_quality(
        x,
        y,
        min_secondary_peak_fraction=0.3,
        force_z_threshold=float("inf"),
        max_rise_width_fraction=1.0,
    )

    assert result.enabled is True
    assert result.reason is None


def test_classify_quality_can_require_two_peaks() -> None:
    x = np.linspace(0.0, 100.0, 1200)
    y = 2500.0 * np.exp(-(((x - 30.0) / 8.0) ** 2))

    result = classify_quality(
        x,
        y,
        min_secondary_peak_fraction=0.3,
        require_two_peaks=True,
        force_z_threshold=float("inf"),
        max_rise_width_fraction=1.0,
    )

    assert result.enabled is False
    assert result.reason == "weak_second_peak"
