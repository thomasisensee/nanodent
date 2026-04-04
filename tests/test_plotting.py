from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from nanodent.analysis.force_peaks import detect_force_peaks
from nanodent.analysis.oliver_pharr import analyze_oliver_pharr
from nanodent.analysis.onset import detect_onset
from nanodent.models import Experiment, ExperimentPaths, SignalTable
from nanodent.plotting import (
    _decorate_saved_experiment_axes,
    plot_experiments,
    plot_hardness_over_time,
    plot_reduced_modulus_over_time,
    save_experiment_plots,
)


def _make_experiment() -> Experiment:
    load_disp = np.linspace(0.0, 100.0, 101)
    load_force = 0.01 * load_disp**2
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
            disp,
            force,
            unloading_fraction=0.25,
            onset_disp_nm=experiment.onset.onset_disp_nm,
            stem="synthetic",
        )
    )
    return experiment.with_force_peaks(
        detect_force_peaks(
            force,
            time_s=time,
            disp_nm=disp,
            prominence=10.0,
            threshold=1.0,
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

    assert len(ax.lines) == 3
    assert ax.lines[0].get_label() == "synthetic"
    assert ax.lines[1].get_label() == "synthetic fit"
    assert ax.lines[2].get_label() == "_nolegend_"
    assert ax.lines[2].get_linestyle() == ":"
    plt.close(figure)


def test_plot_hardness_over_time_returns_axes_and_plots_values() -> None:
    first = _make_experiment_with_timestamp(datetime(2026, 3, 4, 13, 56, 27))
    second = _make_experiment_with_timestamp(datetime(2026, 3, 4, 14, 56, 27))
    figure, ax = plt.subplots()

    returned_ax = plot_hardness_over_time(ax, [first, second])

    assert returned_ax is ax
    assert len(ax.lines) == 1
    assert np.allclose(
        ax.lines[0].get_xdata(),
        [
            mdates.date2num(first.timestamp),
            mdates.date2num(second.timestamp),
        ],
    )
    assert np.allclose(
        ax.lines[0].get_ydata(),
        [
            first.oliver_pharr.hardness_uN_per_nm2,
            second.oliver_pharr.hardness_uN_per_nm2,
        ],
    )
    assert ax.get_ylabel() == "Hardness H / uN/nm^2"
    assert ax.get_title() == "Hardness Over Time"
    plt.close(figure)


def test_plot_reduced_modulus_over_time_returns_axes_and_plots_values() -> (
    None
):
    first = _make_experiment_with_timestamp(datetime(2026, 3, 4, 13, 56, 27))
    second = _make_experiment_with_timestamp(datetime(2026, 3, 4, 14, 56, 27))
    figure, ax = plt.subplots()

    returned_ax = plot_reduced_modulus_over_time(ax, [first, second])

    assert returned_ax is ax
    assert len(ax.lines) == 1
    assert np.allclose(
        ax.lines[0].get_ydata(),
        [
            first.oliver_pharr.reduced_modulus_uN_per_nm2,
            second.oliver_pharr.reduced_modulus_uN_per_nm2,
        ],
    )
    assert ax.get_ylabel() == "Reduced modulus Er / uN/nm^2"
    assert ax.get_title() == "Reduced Modulus Over Time"
    plt.close(figure)


def test_plot_hardness_over_time_skips_missing_values() -> None:
    first = _make_experiment_with_timestamp(datetime(2026, 3, 4, 13, 56, 27))
    second = replace(
        _make_experiment_with_timestamp(datetime(2026, 3, 4, 14, 56, 27)),
        oliver_pharr=replace(
            _make_experiment().oliver_pharr,
            hardness_uN_per_nm2=None,
        ),
    )
    figure, ax = plt.subplots()

    plot_hardness_over_time(ax, [first, second])

    assert len(ax.lines) == 1
    assert np.allclose(
        ax.lines[0].get_ydata(),
        [first.oliver_pharr.hardness_uN_per_nm2],
    )
    plt.close(figure)


def test_plot_reduced_modulus_over_time_honors_selection() -> None:
    enabled_experiment = _make_experiment_with_timestamp(
        datetime(2026, 3, 4, 13, 56, 27)
    )
    disabled_experiment = replace(
        _make_experiment_with_timestamp(datetime(2026, 3, 4, 14, 56, 27)),
        enabled=False,
        disabled_reason="manual",
    )
    figure, ax = plt.subplots()

    plot_reduced_modulus_over_time(
        ax,
        [enabled_experiment, disabled_experiment],
        selection="disabled",
    )

    assert len(ax.lines) == 1
    assert np.allclose(
        ax.lines[0].get_ydata(),
        [disabled_experiment.oliver_pharr.reduced_modulus_uN_per_nm2],
    )
    plt.close(figure)


def test_plot_hardness_over_time_leaves_axes_empty_when_no_values() -> None:
    experiment = replace(
        _make_experiment(),
        oliver_pharr=replace(
            _make_experiment().oliver_pharr,
            hardness_uN_per_nm2=None,
        ),
    )
    figure, ax = plt.subplots()

    plot_hardness_over_time(ax, [experiment])

    assert len(ax.lines) == 0
    assert ax.get_ylabel() == "Hardness H / uN/nm^2"
    plt.close(figure)


def test_plot_experiments_can_hide_attached_oliver_pharr_overlay() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, show_oliver_pharr=False)

    assert len(ax.lines) == 1
    plt.close(figure)


def test_plot_experiments_ignores_attached_fit_for_other_axes() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, x="time_s", y="force_uN")

    assert len(ax.lines) == 1
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
        np.asarray(experiment.oliver_pharr.x_fit, dtype=np.float64)
        - onset_disp,
    )
    assert np.allclose(
        extension_x[1],
        float(experiment.oliver_pharr.x_fit[0]) - onset_disp,
    )
    plt.close(figure)


def test_plot_experiments_can_zero_time_at_onset() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment, x="time_s", y="force_uN", zero_onset=True)

    curve_x = ax.lines[0].get_xdata()
    onset_time = float(experiment.onset.onset_time_s)

    assert len(ax.lines) == 1
    assert curve_x[experiment.onset.onset_index] == 0.0
    assert curve_x[0] == -onset_time
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
        np.asarray(experiment.oliver_pharr.x_fit, dtype=np.float64),
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
        section="test",
        x="disp_nm",
        y="force_uN",
    )

    assert ax.get_title() == "synthetic"
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == (
        "S=5.00 uN/nm\n"
        f"H={experiment.oliver_pharr.hardness_uN_per_nm2:.3g} uN/nm^2\n"
        f"Er={experiment.oliver_pharr.reduced_modulus_uN_per_nm2:.3g} uN/nm^2"
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
        section="test",
        x="disp_nm",
        y="force_uN",
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


def test_saved_plot_decoration_skips_top_axis_for_other_axes() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    _decorate_saved_experiment_axes(
        ax,
        experiment=experiment,
        section="test",
        x="time_s",
        y="force_uN",
    )

    assert ax.get_title() == "synthetic"
    assert len(ax.texts) == 1
    assert len(figure.axes) == 1
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
        section="test",
        x="disp_nm",
        y="force_uN",
    )

    assert ax.get_title() == "synthetic"
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == "S=5.00 uN/nm"
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
