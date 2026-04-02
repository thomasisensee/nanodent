from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanodent.analysis.oliver_pharr import analyze_oliver_pharr
from nanodent.models import Experiment, ExperimentPaths, SignalTable
from nanodent.plotting import plot_experiments


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
    return experiment.with_oliver_pharr(
        analyze_oliver_pharr(
            disp, force, unloading_fraction=0.25, stem="synthetic"
        )
    )


def test_plot_experiments_draws_attached_oliver_pharr_overlay() -> None:
    experiment = _make_experiment()
    figure, ax = plt.subplots()

    plot_experiments(ax, experiment)

    assert len(ax.lines) == 2
    assert ax.lines[0].get_label() == "synthetic"
    assert ax.lines[1].get_label() == "synthetic fit"
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
