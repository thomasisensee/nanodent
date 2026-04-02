from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanodent.analysis.oliver_pharr import analyze_oliver_pharr
from nanodent.analysis.onset import detect_onset
from nanodent.models import Experiment, ExperimentPaths, SignalTable
from nanodent.plotting import (
    _decorate_saved_experiment_axes,
    plot_experiments,
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
    experiment = experiment.with_oliver_pharr(
        analyze_oliver_pharr(
            disp, force, unloading_fraction=0.25, stem="synthetic"
        )
    )
    return experiment.with_onset(
        detect_onset(
            force,
            time_s=time,
            disp_nm=disp,
            baseline_points=20,
            k=1.0,
            consecutive=3,
        )
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


def test_saved_plot_decoration_adds_top_axis_and_stiffness_title() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()
    ax.set_xlim(0.0, 120.0)

    _decorate_saved_experiment_axes(
        ax,
        experiment=experiment,
        section="test",
        x="disp_nm",
        y="force_uN",
    )

    assert ax.get_title() == "synthetic | S=5.00 uN/nm"
    assert len(figure.axes) == 2
    top_ax = figure.axes[1]
    top_tick_labels = [tick.get_text() for tick in top_ax.get_xticklabels()]
    top_tick_positions = list(top_ax.get_xticks())
    expected_onset_label = f"{float(experiment.onset.onset_disp_nm):.3g}"
    expected_max_label = f"{
        float(
            experiment.test['disp_nm'][np.argmax(experiment.test['force_uN'])]
        ):.3g}"
    assert len(top_tick_positions) == 2
    assert top_tick_labels == [expected_onset_label, expected_max_label]
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

    assert ax.get_title() == "synthetic | S=5.00 uN/nm"
    assert len(figure.axes) == 1
    plt.close(figure)


def test_saved_plot_decoration_uses_stem_only_without_stiffness() -> None:
    experiment = _make_experiment().with_oliver_pharr(None)
    figure, ax = plt.subplots()

    _decorate_saved_experiment_axes(
        ax,
        experiment=experiment,
        section="test",
        x="disp_nm",
        y="force_uN",
    )

    assert ax.get_title() == "synthetic"
    assert len(figure.axes) == 2
    plt.close(figure)
