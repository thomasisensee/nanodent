import numpy as np
import pytest

from nanodent.analysis.filters import savgol
from nanodent.analysis.force_peaks import detect_force_peaks
from nanodent.analysis.oliver_pharr import analyze_oliver_pharr
from nanodent.analysis.onset import detect_onset
from nanodent.analysis.quality import (
    classify_flat_force,
    classify_gradual_onset,
    classify_high_displacement,
    classify_outlier_jumps,
    classify_peak_balance,
    classify_quality,
)
from nanodent.analysis.unloading import detect_unloading


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


def _make_power_law_unloading_curve(
    *,
    hf_nm: float = 40.0,
    m: float = 1.5,
    unloading_end_nm: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    load_disp = np.linspace(0.0, 100.0, 101)
    load_force = 0.01 * load_disp**2
    unload_disp = np.linspace(100.0, unloading_end_nm, 61)[1:]
    k = 100.0 / np.power(100.0 - hf_nm, m)
    unload_force = k * np.power(unload_disp - hf_nm, m)
    return (
        np.concatenate([load_disp, unload_disp]),
        np.concatenate([load_force, unload_force]),
        float(k),
    )


def _make_two_peak_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(0.0, 120.0, 1.0, dtype=np.float64)
    disp = np.linspace(0.0, 119.0, len(time), dtype=np.float64)
    force = 5.0 + 150.0 * np.exp(-(((time - 30.0) / 4.0) ** 2))
    force += 220.0 * np.exp(-(((time - 80.0) / 6.0) ** 2))
    return time, disp, force


def test_savgol_preserves_shape_and_reduces_noise() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 101)
    clean = np.sin(x)
    noisy = clean + 0.2 * np.sin(11.0 * x)

    smoothed = savgol(noisy, window_length=21, polyorder=3)

    assert smoothed.shape == noisy.shape
    assert np.mean(np.abs(smoothed - clean)) < np.mean(np.abs(noisy - clean))


def test_detect_onset_finds_first_sustained_crossing() -> None:
    force = np.array([0.0] * 8 + [1.0, 1.2, 1.4, 1.6, 1.8], dtype=np.float64)
    time = np.arange(len(force), dtype=np.float64) * 0.5
    disp = np.linspace(0.0, 12.0, len(force), dtype=np.float64)

    result = detect_onset(
        force,
        time_s=time,
        disp_nm=disp,
        baseline_points=5,
        k=0.5,
        consecutive=3,
    )

    assert result.success is True
    assert result.reason is None
    assert result.mode == "relative"
    assert result.onset_index == 8
    assert result.onset_time_s == pytest.approx(time[8])
    assert result.onset_disp_nm == pytest.approx(disp[8])
    assert result.baseline_points == 5
    assert result.baseline_start_index == 0
    assert result.baseline_end_index == 5
    assert result.baseline_offset_uN == pytest.approx(0.0)
    assert result.used_smoothing is False


def test_detect_onset_returns_unsuccessful_result_without_crossing() -> None:
    force = np.linspace(0.0, 0.3, 20, dtype=np.float64)

    result = detect_onset(
        force,
        baseline_points=10,
        k=10.0,
        consecutive=4,
    )

    assert result.success is False
    assert result.reason == "no_onset_detected"
    assert result.onset_index is None
    assert result.onset_time_s is None
    assert result.onset_disp_nm is None


def test_detect_onset_can_use_explicit_baseline_indices() -> None:
    force = np.array(
        [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 1.0],
        dtype=np.float64,
    )

    default_result = detect_onset(
        force,
        baseline_points=2,
        k=0.5,
        consecutive=1,
    )
    indexed_result = detect_onset(
        force,
        mode="relative",
        baseline_start_index=2,
        baseline_end_index=6,
        k=0.5,
        consecutive=1,
    )

    assert default_result.success is False
    assert indexed_result.success is True
    assert indexed_result.onset_index == 6
    assert indexed_result.baseline_points == 4
    assert indexed_result.baseline_start_index == 2
    assert indexed_result.baseline_end_index == 6
    assert indexed_result.baseline_mean_uN == pytest.approx(0.0)
    assert indexed_result.baseline_offset_uN == pytest.approx(0.0)


def test_detect_onset_can_use_absolute_threshold_mode() -> None:
    force = np.array(
        [5.0, 5.0, 0.0, 0.0, 0.2, 0.7, 1.2],
        dtype=np.float64,
    )
    time = np.arange(len(force), dtype=np.float64)
    disp = np.linspace(0.0, 6.0, len(force), dtype=np.float64)

    result = detect_onset(
        force,
        time_s=time,
        disp_nm=disp,
        mode="absolute",
        baseline_start_index=2,
        baseline_end_index=4,
        absolute_threshold_uN=0.5,
        consecutive=1,
    )

    assert result.success is True
    assert result.mode == "absolute"
    assert result.onset_index == 5
    assert result.onset_time_s == pytest.approx(time[5])
    assert result.onset_disp_nm == pytest.approx(disp[5])
    assert result.threshold_uN == pytest.approx(0.5)
    assert result.absolute_threshold_uN == pytest.approx(0.5)
    assert result.baseline_start_index == 2
    assert result.baseline_end_index == 4
    assert result.baseline_mean_uN == pytest.approx(0.0)


def test_detect_onset_uses_smoothed_force_when_requested() -> None:
    force = np.array(
        [
            0.0,
            0.8,
            -0.8,
            0.8,
            -0.8,
            0.8,
            -0.8,
            0.8,
            -0.8,
            0.8,
            -0.8,
            0.5,
            0.7,
            0.9,
            1.1,
            1.1,
            1.1,
        ],
        dtype=np.float64,
    )

    raw_result = detect_onset(
        force,
        baseline_points=10,
        k=2.0,
        consecutive=3,
    )
    smoothed_result = detect_onset(
        force,
        baseline_points=10,
        k=2.0,
        consecutive=3,
        smoothing={"window_length": 5, "polyorder": 1},
    )

    assert raw_result.success is False
    assert smoothed_result.success is True
    assert smoothed_result.used_smoothing is True
    assert dict(smoothed_result.smoothing or {}) == {
        "window_length": 5,
        "polyorder": 1,
    }


def test_detect_onset_validates_detection_parameters() -> None:
    force = np.linspace(0.0, 1.0, 20, dtype=np.float64)

    with pytest.raises(ValueError, match="baseline_points"):
        detect_onset(force, baseline_points=0)

    with pytest.raises(ValueError, match="consecutive"):
        detect_onset(force, consecutive=0)

    with pytest.raises(ValueError, match="time_s"):
        detect_onset(force, time_s=np.arange(19, dtype=np.float64))

    with pytest.raises(ValueError, match="disp_nm"):
        detect_onset(force, disp_nm=np.arange(19, dtype=np.float64))

    with pytest.raises(ValueError, match="mode"):
        detect_onset(force, mode="invalid")

    with pytest.raises(ValueError, match="absolute_threshold_uN"):
        detect_onset(force, mode="absolute")

    with pytest.raises(ValueError, match="provided together"):
        detect_onset(force, baseline_start_index=0)

    with pytest.raises(ValueError, match="start < end"):
        detect_onset(force, baseline_start_index=5, baseline_end_index=5)

    with pytest.raises(ValueError, match="within the force signal"):
        detect_onset(force, baseline_start_index=0, baseline_end_index=25)


def test_detect_force_peaks_detects_two_peaks_with_coordinates() -> None:
    time, disp, force = _make_two_peak_signal()

    result = detect_force_peaks(
        force,
        time_s=time,
        disp_nm=disp,
        prominence=100.0,
        threshold=1.0,
    )

    assert result.success is True
    assert result.reason is None
    assert result.peak_count == 2
    assert len(result.peaks) == 2
    assert result.peaks[0].index < result.peaks[1].index
    assert result.peaks[0].time_s == pytest.approx(time[result.peaks[0].index])
    assert result.peaks[0].disp_nm == pytest.approx(
        disp[result.peaks[0].index]
    )
    assert result.peaks[1].time_s == pytest.approx(time[result.peaks[1].index])
    assert result.peaks[1].disp_nm == pytest.approx(
        disp[result.peaks[1].index]
    )


def test_detect_force_peaks_returns_unsuccessful_result_without_peaks() -> (
    None
):
    force = np.linspace(0.0, 1.0, 30, dtype=np.float64)

    result = detect_force_peaks(force, prominence=100.0, threshold=1.0)

    assert result.success is False
    assert result.reason == "no_force_peaks_detected"
    assert result.peaks == ()
    assert result.peak_count == 0


def test_detect_force_peaks_keeps_two_strongest_peaks_in_index_order() -> None:
    time = np.arange(0.0, 150.0, 1.0, dtype=np.float64)
    disp = np.linspace(0.0, 149.0, len(time), dtype=np.float64)
    force = 3.0
    force += 110.0 * np.exp(-(((time - 20.0) / 3.0) ** 2))
    force += 260.0 * np.exp(-(((time - 60.0) / 4.0) ** 2))
    force += 180.0 * np.exp(-(((time - 110.0) / 5.0) ** 2))

    result = detect_force_peaks(
        force,
        time_s=time,
        disp_nm=disp,
        prominence=100.0,
        threshold=1.0,
    )

    peak_indices = tuple(peak.index for peak in result.peaks)
    assert result.success is True
    assert result.peak_count == 2
    assert peak_indices == (60, 110)


def test_detect_force_peaks_validates_aligned_inputs() -> None:
    force = np.linspace(0.0, 1.0, 20, dtype=np.float64)

    with pytest.raises(ValueError, match="time_s"):
        detect_force_peaks(force, time_s=np.arange(19, dtype=np.float64))

    with pytest.raises(ValueError, match="disp_nm"):
        detect_force_peaks(force, disp_nm=np.arange(19, dtype=np.float64))


def test_detect_unloading_uses_global_max_force_with_coordinates() -> None:
    time = np.arange(6, dtype=np.float64) * 0.5
    disp = np.linspace(0.0, 5.0, 6, dtype=np.float64)
    force = np.array([0.0, 1.0, 4.0, 7.0, 5.0, 3.0], dtype=np.float64)

    result = detect_unloading(
        force,
        time_s=time,
        disp_nm=disp,
    )

    assert result.success is True
    assert result.reason is None
    assert result.method == "max_force"
    assert result.start_index == 3
    assert result.start_time_s == pytest.approx(time[3])
    assert result.start_disp_nm == pytest.approx(disp[3])
    assert result.start_force_uN == pytest.approx(force[3])
    assert result.end_disp_nm == pytest.approx(disp[-1])


def test_detect_unloading_validates_inputs_and_method() -> None:
    force = np.linspace(0.0, 1.0, 20, dtype=np.float64)

    with pytest.raises(ValueError, match="time_s"):
        detect_unloading(force, time_s=np.arange(19, dtype=np.float64))

    with pytest.raises(ValueError, match="disp_nm"):
        detect_unloading(force, disp_nm=np.arange(19, dtype=np.float64))

    with pytest.raises(ValueError, match="method"):
        detect_unloading(force, method="invalid")


def test_analyze_oliver_pharr_fits_linear_unloading_branch() -> None:
    x, y = _make_linear_unloading_curve()

    result = analyze_oliver_pharr(
        x,
        y,
        unloading_fraction=0.25,
        onset_disp_nm=20.0,
    )

    assert result.success is True
    assert result.reason is None
    assert result.fit_model == "linear_fraction"
    assert result.used_smoothing is False
    assert result.evaluation_index == 100
    assert result.evaluation_force_uN == pytest.approx(100.0)
    assert result.evaluation_disp_nm == pytest.approx(100.0)
    assert result.unloading_start_index == 100
    assert result.unloading_end_index == 110
    assert result.fit_point_count == 11
    assert result.stiffness_uN_per_nm == pytest.approx(5.0, rel=1e-4)
    assert result.linear_slope_uN_per_nm == pytest.approx(5.0, rel=1e-4)
    assert result.linear_intercept_uN == pytest.approx(-400.0, rel=1e-4)
    assert result.linear_depth_intercept_nm == pytest.approx(80.0, rel=1e-4)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.hardness_success is True
    assert result.hardness_reason is None
    assert result.epsilon == pytest.approx(0.75)
    assert result.onset_disp_nm == pytest.approx(20.0)
    assert result.hmax_nm == pytest.approx(80.0)
    assert result.contact_depth_nm == pytest.approx(65.0)
    assert result.contact_area_nm2 == pytest.approx(103512.5)
    assert result.hardness_uN_per_nm2 == pytest.approx(100.0 / 103512.5)
    assert result.reduced_modulus_uN_per_nm2 == pytest.approx(
        0.5 * np.sqrt(np.pi) * 5.0 / np.sqrt(103512.5)
    )
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

    assert raw_result.evaluation_index == 70
    assert smoothed_result.success is True
    assert smoothed_result.used_smoothing is True
    assert dict(smoothed_result.smoothing or {}) == {
        "window_length": 21,
        "polyorder": 2,
    }
    assert smoothed_result.evaluation_index == pytest.approx(100, abs=2)


def test_analyze_oliver_pharr_can_reuse_precomputed_unloading_start() -> None:
    x, y = _make_spiky_peak_curve()

    result = analyze_oliver_pharr(
        x,
        y,
        unloading_fraction=0.25,
        smoothing={"window_length": 21, "polyorder": 2},
        unloading_start_index=70,
    )

    assert result.success is True
    assert result.evaluation_index == 70
    assert result.unloading_start_index == 70


def test_analyze_oliver_pharr_validates_explicit_unloading_start() -> None:
    x, y = _make_linear_unloading_curve()

    with pytest.raises(ValueError, match="unloading_start_index"):
        analyze_oliver_pharr(x, y, unloading_start_index=len(x))


def test_analyze_oliver_pharr_rejects_unloading_fraction_for_power_law() -> (
    None
):
    x, y, _ = _make_power_law_unloading_curve()

    with pytest.raises(ValueError, match="unloading_fraction"):
        analyze_oliver_pharr(
            x,
            y,
            fit_model="power_law_full",
            unloading_fraction=0.25,
            unloading_start_index=100,
        )


def test_analyze_oliver_pharr_rejects_invalid_unloading_fraction() -> None:
    x, y = _make_linear_unloading_curve()

    with pytest.raises(ValueError, match="unloading_fraction"):
        analyze_oliver_pharr(x, y, unloading_fraction=0.0)

    with pytest.raises(ValueError, match="unloading_fraction"):
        analyze_oliver_pharr(x, y, unloading_fraction=1.1)


def test_analyze_oliver_pharr_rejects_invalid_epsilon() -> None:
    x, y = _make_linear_unloading_curve()

    with pytest.raises(ValueError, match="epsilon"):
        analyze_oliver_pharr(x, y, epsilon=0.0)


def test_analyze_oliver_pharr_rejects_invalid_power_law_hf_mode() -> None:
    x, y, _ = _make_power_law_unloading_curve()

    with pytest.raises(ValueError, match="power_law_hf_mode"):
        analyze_oliver_pharr(
            x,
            y,
            fit_model="power_law_full",
            unloading_start_index=100,
            power_law_hf_mode="invalid",
        )


def test_analyze_oliver_pharr_marks_missing_onset_for_hardness() -> None:
    x, y = _make_linear_unloading_curve()

    result = analyze_oliver_pharr(x, y, unloading_fraction=0.25)

    assert result.success is True
    assert result.hardness_success is False
    assert result.hardness_reason == "missing_onset"
    assert result.epsilon == pytest.approx(0.75)
    assert result.hmax_nm is None
    assert result.contact_depth_nm is None
    assert result.contact_area_nm2 is None
    assert result.hardness_uN_per_nm2 is None
    assert result.reduced_modulus_uN_per_nm2 is None


def test_analyze_oliver_pharr_can_override_epsilon() -> None:
    x, y = _make_linear_unloading_curve()

    result = analyze_oliver_pharr(
        x,
        y,
        unloading_fraction=0.25,
        onset_disp_nm=20.0,
        epsilon=0.5,
    )

    assert result.hardness_success is True
    assert result.epsilon == pytest.approx(0.5)
    assert result.contact_depth_nm == pytest.approx(70.0)
    assert result.contact_area_nm2 == pytest.approx(24.5 * 70.0 * 70.0)
    assert result.reduced_modulus_uN_per_nm2 == pytest.approx(
        0.5
        * np.sqrt(np.pi)
        * result.stiffness_uN_per_nm
        / np.sqrt(result.contact_area_nm2)
    )


def test_analyze_oliver_pharr_marks_invalid_onset_corrected_hmax() -> None:
    x, y = _make_linear_unloading_curve()

    result = analyze_oliver_pharr(
        x,
        y,
        unloading_fraction=0.25,
        onset_disp_nm=100.0,
    )

    assert result.success is True
    assert result.hardness_success is False
    assert result.hardness_reason == "invalid_hmax"
    assert result.hmax_nm == pytest.approx(0.0)
    assert result.reduced_modulus_uN_per_nm2 is None


def test_analyze_oliver_pharr_marks_missing_unloading_branch() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    result = analyze_oliver_pharr(x, y)

    assert result.success is False
    assert result.reason == "no_unloading_branch"


def test_analyze_oliver_pharr_fits_full_power_law_with_fitted_hf() -> None:
    x, y, expected_k = _make_power_law_unloading_curve(unloading_end_nm=50.0)

    result = analyze_oliver_pharr(
        x,
        y,
        fit_model="power_law_full",
        unloading_start_index=100,
        onset_disp_nm=20.0,
    )

    assert result.success is True
    assert result.fit_model == "power_law_full"
    assert result.evaluation_index == 100
    assert result.unloading_end_index == len(x) - 1
    assert result.fit_point_count == len(x) - 100
    assert result.power_law_hf_mode == "fit"
    assert result.power_law_k == pytest.approx(expected_k, rel=5e-2)
    assert result.power_law_m == pytest.approx(1.5, rel=5e-2)
    assert result.power_law_hf_nm == pytest.approx(40.0, abs=1.0)
    expected_stiffness = expected_k * 1.5 * np.power(60.0, 0.5)
    assert result.stiffness_uN_per_nm == pytest.approx(
        expected_stiffness, rel=5e-2
    )
    assert result.linear_intercept_uN is None
    assert result.linear_depth_intercept_nm is None


def test_analyze_oliver_pharr_fits_full_power_law_with_fixed_end_disp() -> (
    None
):
    x, y, expected_k = _make_power_law_unloading_curve(unloading_end_nm=40.0)

    result = analyze_oliver_pharr(
        x,
        y,
        fit_model="power_law_full",
        unloading_start_index=100,
        unloading_end_disp_nm=40.0,
        power_law_hf_mode="fixed_end_disp",
    )

    assert result.success is True
    assert result.power_law_hf_mode == "fixed_end_disp"
    assert result.power_law_hf_nm == pytest.approx(40.0)
    assert result.power_law_k == pytest.approx(expected_k, rel=5e-2)
    assert result.power_law_m == pytest.approx(1.5, rel=5e-2)


def test_analyze_op_power_law_uses_unloading_start_as_evaluation_point() -> (
    None
):
    x, y, _ = _make_power_law_unloading_curve(unloading_end_nm=40.0)
    y[70] += 10.0

    result = analyze_oliver_pharr(
        x,
        y,
        fit_model="power_law_full",
        unloading_start_index=100,
        unloading_end_disp_nm=40.0,
        power_law_hf_mode="fixed_end_disp",
    )

    assert result.success is True
    assert result.evaluation_index == 100
    assert result.evaluation_disp_nm == pytest.approx(x[100])


def test_analyze_oliver_pharr_power_law_requires_end_disp_for_fixed_hf() -> (
    None
):
    x, y, _ = _make_power_law_unloading_curve(unloading_end_nm=40.0)

    result = analyze_oliver_pharr(
        x,
        y,
        fit_model="power_law_full",
        unloading_start_index=100,
        power_law_hf_mode="fixed_end_disp",
    )

    assert result.success is False
    assert result.reason == "missing_unloading_end_disp"


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
