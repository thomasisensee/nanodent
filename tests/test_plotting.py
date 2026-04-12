from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nanodent.analysis.force_peaks import detect_force_peaks
from nanodent.analysis.hertzian import analyze_hertzian
from nanodent.analysis.oliver_pharr import analyze_oliver_pharr
from nanodent.analysis.onset import detect_onset
from nanodent.analysis.unloading import detect_unloading
from nanodent.models import Experiment, ExperimentPaths, SignalTable
from nanodent.plotting import (
    _decorate_saved_experiment_axes,
    plot_experiments,
    save_experiment_plots,
)
from nanodent.study import Study


def _make_experiment() -> Experiment:
    return _make_experiment_with_fit()


def _make_experiment_with_fit(
    *,
    fit_model: str = "linear_fraction",
) -> Experiment:
    load_disp = np.linspace(0.0, 100.0, 101)
    load_force = 0.01 * load_disp**2
    if fit_model == "power_law_full":
        unload_disp = np.linspace(100.0, 40.0, 61)[1:]
        power_law_m = 1.5
        power_law_k = 100.0 / np.power(60.0, power_law_m)
        unload_force = power_law_k * np.power(unload_disp - 40.0, power_law_m)
    else:
        unload_disp = np.linspace(100.0, 80.0, 41)[1:]
        unload_force = 5.0 * unload_disp - 400.0
    disp = np.concatenate([load_disp, unload_disp])
    force = np.concatenate([load_force, unload_force])
    time = np.arange(len(disp), dtype=np.float64)
    test = SignalTable(
        columns={
            "time_s": time,
            "disp_nm": disp,
            "force_uN": force,
        },
        point_count=len(disp),
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = Experiment(
        paths=ExperimentPaths(
            stem="synthetic",
            hld_path=Path("synthetic.hld"),
        ),
        metadata={},
        metadata_entries=(),
        timestamp=datetime(2026, 3, 4, 13, 56, 27),
        approach=None,
        drift=None,
        test=test,
    )
    experiment = experiment.with_onset(
        detect_onset(
            force,
            time_s=time,
            disp_nm=disp,
            baseline_points=20,
            k=1.0,
            consecutive=3,
        )
    )
    experiment = experiment.with_oliver_pharr(
        analyze_oliver_pharr(
            disp[100:],
            force[100:],
            unloading_start_trace_index=100,
            fit_model=fit_model,
            unloading_fraction=0.25
            if fit_model == "linear_fraction"
            else None,
            onset_disp_nm=experiment.onset.onset_disp_nm,
            baseline_offset_uN=experiment.onset.baseline_offset_uN,
            stem="synthetic",
        )
    )
    experiment = experiment.with_unloading(
        detect_unloading(
            force,
            time_s=time,
            disp_nm=disp,
        )
    )
    experiment = experiment.with_force_peaks(
        detect_force_peaks(
            force,
            time_s=time,
            disp_nm=disp,
            prominence=10.0,
            threshold=1.0,
        )
    )
    return experiment.with_hertzian(
        analyze_hertzian(
            disp,
            force,
            fit_end_index=experiment.force_peaks.peaks[0].index,
            initial_onset_disp_nm=experiment.onset.onset_disp_nm,
            baseline_offset_uN=experiment.onset.baseline_offset_uN,
            stem="synthetic",
        )
    )


def _make_experiment_with_timestamp(timestamp: datetime) -> Experiment:
    return replace(
        _make_experiment(),
        timestamp=timestamp,
    )


def test_plot_experiments_draws_attached_oliver_pharr_overlay() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment)

    assert len(ax.lines) == 4
    assert ax.lines[0].get_label() == "synthetic"
    assert ax.lines[1].get_label() == "synthetic fit"
    assert ax.lines[2].get_label() == "_nolegend_"
    assert ax.lines[2].get_linestyle() == ":"
    assert ax.lines[3].get_label() == "synthetic Hertzian fit"
    assert np.allclose(
        ax.lines[1].get_xdata(),
        np.asarray(experiment.oliver_pharr.x_fit, dtype=np.float64)
        + float(experiment.oliver_pharr.disp_correction_nm or 0.0),
    )
    plt.close(figure)


def test_plot_experiments_can_draw_evaluation_marker_when_requested() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(
        ax,
        experiment,
        show_oliver_pharr_evaluation=True,
    )

    assert len(ax.lines) == 5
    marker_line = ax.lines[2]
    assert marker_line.get_label() == "_nolegend_"
    assert marker_line.get_marker() == "o"
    assert marker_line.get_markerfacecolor() == "none"
    assert marker_line.get_xdata()[0] == pytest.approx(
        float(experiment.oliver_pharr.evaluation_disp_nm)
        + float(experiment.oliver_pharr.disp_correction_nm or 0.0)
    )
    assert marker_line.get_ydata()[0] == pytest.approx(
        float(experiment.oliver_pharr.evaluation_force_uN)
        + float(experiment.oliver_pharr.force_correction_uN or 0.0)
    )
    plt.close(figure)


def test_plot_experiments_power_law_marker_uses_inverse_fit_point() -> None:
    experiment = _make_experiment_with_fit(fit_model="power_law_full")
    figure, ax = plt.subplots()

    plot_experiments(
        ax,
        experiment,
        show_oliver_pharr_evaluation=True,
    )

    marker_line = ax.lines[2]
    expected_x = float(experiment.oliver_pharr.evaluation_disp_nm) + float(
        experiment.oliver_pharr.disp_correction_nm or 0.0
    )
    expected_y = float(experiment.oliver_pharr.evaluation_force_uN) + float(
        experiment.oliver_pharr.force_correction_uN or 0.0
    )
    assert marker_line.get_xdata()[0] == pytest.approx(expected_x)
    assert marker_line.get_ydata()[0] == pytest.approx(expected_y)
    plt.close(figure)


def test_plot_experiments_omits_linear_extension_for_power_law_fit() -> None:
    experiment = _make_experiment_with_fit(fit_model="power_law_full")
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment)

    assert len(ax.lines) == 3
    assert ax.lines[1].get_label() == "synthetic fit"
    plt.close(figure)


def test_plot_experiments_resolves_groups_via_study() -> None:
    first = _make_experiment_with_timestamp(datetime(2026, 3, 4, 13, 56, 27))
    second = replace(
        _make_experiment_with_timestamp(datetime(2026, 3, 4, 14, 10, 0)),
        paths=ExperimentPaths(
            stem="synthetic_2",
            hld_path=Path("synthetic_2.hld"),
        ),
    )
    study = Study(experiments=(first, second))
    group = study.group_by_time_gap(max_gap=timedelta(minutes=30))[0]
    figure, ax = plt.subplots()

    plot_experiments(ax, group, study=study)

    assert len(ax.lines) == 8
    plt.close(figure)


def test_plot_experiments_group_requires_study() -> None:
    experiment = _make_experiment()
    study = Study(experiments=(experiment,))
    group = study.group_by_time_gap()[0]
    figure, ax = plt.subplots()

    with pytest.raises(TypeError, match="requires passing study"):
        plot_experiments(ax, group)

    plt.close(figure)


def test_plot_experiments_can_hide_attached_oliver_pharr_overlay() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, show_oliver_pharr=False)

    assert len(ax.lines) == 2
    assert ax.lines[1].get_label() == "synthetic Hertzian fit"
    plt.close(figure)


def test_plot_experiments_can_hide_attached_hertzian_overlay() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, show_hertzian=False)

    assert len(ax.lines) == 3
    assert all(
        line.get_label() != "synthetic Hertzian fit" for line in ax.lines
    )
    plt.close(figure)


def test_plot_experiments_hertzian_label_stays_simple() -> None:
    experiment = _make_experiment()
    experiment = experiment.with_hertzian(
        replace(
            experiment.hertzian,
            radius_nm=42.0,
            tau_max_uN_per_nm2=0.1234,
        )
    )
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment)

    assert ax.lines[3].get_label() == "synthetic Hertzian fit"
    plt.close(figure)


def test_plot_experiments_can_draw_unloading_overlay_when_requested() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, show_unloading=True)

    assert len(ax.lines) == 5
    assert np.array_equal(
        ax.lines[1].get_xdata(),
        np.asarray(experiment.test["disp_nm"], dtype=np.float64)[
            experiment.unloading.start_index :
        ],
    )
    assert ax.lines[1].get_alpha() == pytest.approx(0.75)
    plt.close(figure)


def test_plot_experiments_can_zero_displacement_at_onset() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, zero_onset=True)

    curve_x = ax.lines[0].get_xdata()
    fit_x = ax.lines[1].get_xdata()
    extension_x = ax.lines[2].get_xdata()
    onset_disp = float(experiment.onset.onset_disp_nm)

    assert curve_x[experiment.onset.onset_index] == 0.0
    assert curve_x[0] == -onset_disp
    assert np.allclose(
        fit_x,
        np.asarray(experiment.oliver_pharr.x_fit, dtype=np.float64),
    )
    assert np.allclose(
        extension_x[1],
        float(experiment.oliver_pharr.x_fit[0]),
    )
    plt.close(figure)


def test_plot_experiments_shifts_evaluation_marker_when_zeroing_onset() -> (
    None
):
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(
        ax,
        experiment,
        zero_onset=True,
        show_oliver_pharr_evaluation=True,
    )

    marker_line = ax.lines[2]
    assert marker_line.get_xdata()[0] == pytest.approx(
        float(experiment.oliver_pharr.evaluation_disp_nm)
    )
    assert marker_line.get_ydata()[0] == pytest.approx(
        float(experiment.oliver_pharr.evaluation_force_uN)
        + float(experiment.oliver_pharr.force_correction_uN or 0.0)
    )
    plt.close(figure)


def test_plot_experiments_leaves_curve_unchanged_without_usable_onset() -> (
    None
):
    experiment = replace(
        _make_experiment(),
        onset=replace(_make_experiment().onset, success=False),
    )
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, zero_onset=True)

    assert np.array_equal(
        ax.lines[0].get_xdata(),
        np.asarray(experiment.test["disp_nm"], dtype=np.float64),
    )
    assert np.array_equal(
        ax.lines[1].get_xdata(),
        np.asarray(experiment.oliver_pharr.x_fit, dtype=np.float64)
        + float(experiment.oliver_pharr.disp_correction_nm or 0.0),
    )
    plt.close(figure)


def test_saved_plot_decoration_adds_top_axis_and_analysis_box() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()
    ax.set_xlim(0.0, 120.0)
    ax.set_ylim(0.0, 550.0)

    _decorate_saved_experiment_axes(
        ax,
        experiment=experiment,
    )

    assert ax.get_title() == "synthetic\n2026-03-04 13:56:27"
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == (
        "S=5.00 uN/nm\n"
        f"H={experiment.oliver_pharr.hardness_uN_per_nm2:.3g} uN/nm^2\n"
        f"Er={experiment.oliver_pharr.reduced_modulus_uN_per_nm2:.3g} "
        "uN/nm^2\n"
        f"A={experiment.hertzian.amplitude_uN_per_nm_3_2:.3g} "
        "uN/nm^(3/2)\n"
        f"h0={experiment.hertzian.h_onset_nm:.3g} nm"
    )
    assert len(figure.axes) == 3
    top_ax = figure.axes[1]
    right_ax = figure.axes[2]
    top_tick_labels = [tick.get_text() for tick in top_ax.get_xticklabels()]
    top_tick_positions = list(top_ax.get_xticks())
    right_tick_labels = [
        tick.get_text() for tick in right_ax.get_yticklabels()
    ]
    right_tick_positions = list(right_ax.get_yticks())
    raw_positions = [float(experiment.onset.onset_disp_nm)]
    raw_positions.extend(
        float(peak.disp_nm) for peak in experiment.force_peaks.peaks
    )
    raw_positions.append(
        float(
            experiment.test["disp_nm"][np.argmax(experiment.test["force_uN"])]
        )
    )
    expected_positions: list[float] = []
    for position in sorted(raw_positions):
        if any(
            np.isclose(position, existing, atol=1e-12)
            for existing in expected_positions
        ):
            continue
        expected_positions.append(position)
    expected_labels = [f"{position:.3g}" for position in expected_positions]
    raw_forces = [
        float(experiment.test["force_uN"][experiment.onset.onset_index])
    ]
    raw_forces.extend(
        float(peak.force_uN) for peak in experiment.force_peaks.peaks
    )
    raw_forces.append(float(np.max(experiment.test["force_uN"])))
    expected_forces: list[float] = []
    for force in sorted(raw_forces):
        if any(
            np.isclose(force, existing, atol=1e-12)
            for existing in expected_forces
        ):
            continue
        expected_forces.append(force)
    expected_force_labels = [f"{force:.3g}" for force in expected_forces]

    assert top_tick_positions == expected_positions
    assert top_tick_labels == expected_labels
    assert right_tick_positions == expected_forces
    assert right_tick_labels == expected_force_labels
    plt.close(figure)


def test_saved_plot_decoration_zeroes_top_axis_positions_at_onset() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()
    ax.set_xlim(-20.0, 100.0)
    ax.set_ylim(0.0, 550.0)

    _decorate_saved_experiment_axes(
        ax,
        experiment=experiment,
        zero_onset=True,
    )

    top_ax = figure.axes[1]
    onset_disp = float(experiment.onset.onset_disp_nm)
    expected_positions: list[float] = []
    raw_positions = [float(experiment.onset.onset_disp_nm) - onset_disp]
    raw_positions.extend(
        float(peak.disp_nm) - onset_disp
        for peak in experiment.force_peaks.peaks
    )
    raw_positions.append(
        float(
            experiment.test["disp_nm"][np.argmax(experiment.test["force_uN"])]
        )
        - onset_disp
    )
    for position in sorted(raw_positions):
        if any(
            np.isclose(position, existing, atol=1e-12)
            for existing in expected_positions
        ):
            continue
        expected_positions.append(position)

    assert list(top_ax.get_xticks()) == expected_positions
    assert top_ax.get_xticklabels()[0].get_text() == "0"
    plt.close(figure)


def test_saved_plot_decoration_omits_missing_analysis_values() -> None:
    experiment = _make_experiment().with_oliver_pharr(
        replace(
            _make_experiment().oliver_pharr,
            hardness_uN_per_nm2=None,
            reduced_modulus_uN_per_nm2=None,
        )
    )
    figure, ax = plt.subplots()

    _decorate_saved_experiment_axes(
        ax,
        experiment=experiment,
    )

    assert ax.get_title() == "synthetic\n2026-03-04 13:56:27"
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == (
        "S=5.00 uN/nm\n"
        f"A={experiment.hertzian.amplitude_uN_per_nm_3_2:.3g} "
        "uN/nm^(3/2)\n"
        f"h0={experiment.hertzian.h_onset_nm:.3g} nm"
    )
    assert len(figure.axes) == 3
    plt.close(figure)


def test_save_experiment_plots_can_zero_onset(tmp_path: Path) -> None:
    experiment = _make_experiment()

    saved_paths = save_experiment_plots(
        experiment,
        tmp_path,
        zero_onset=True,
    )

    assert saved_paths == [tmp_path / "synthetic.png"]
    assert saved_paths[0].exists()


def test_save_experiment_plots_adds_radius_and_tau_max_to_analysis_box(
    tmp_path: Path,
) -> None:
    experiment = _make_experiment()
    experiment = experiment.with_hertzian(
        replace(
            experiment.hertzian,
            radius_nm=42.0,
            tau_max_uN_per_nm2=0.1234,
        )
    )
    initial_figures = set(plt.get_fignums())

    saved_paths = save_experiment_plots(
        experiment,
        tmp_path,
        close=False,
    )
    new_figures = set(plt.get_fignums()) - initial_figures
    figure = plt.figure(new_figures.pop())
    ax = figure.axes[0]

    assert saved_paths == [tmp_path / "synthetic.png"]
    assert ax.get_legend() is None
    assert "R=42 nm" in ax.texts[0].get_text()
    assert "tau_max=0.123 uN/nm^2" in ax.texts[0].get_text()
    plt.close(figure)


def test_save_experiment_plots_uses_stem_without_source_path(
    tmp_path: Path,
) -> None:
    experiment = replace(
        _make_experiment(),
        stem="generic_synthetic",
        paths=None,
        source_path=None,
        source_format=None,
    )

    saved_paths = save_experiment_plots(experiment, tmp_path)

    assert saved_paths == [tmp_path / "generic_synthetic.png"]
    assert saved_paths[0].exists()
