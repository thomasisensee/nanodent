from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nanodent import (
    load_folder,
    plot_experiments,
    plot_group_timeline,
    save_experiment_plots,
)
from nanodent.analysis.filters import savgol

DATA_DIR = Path(__file__).parent / "data"
BAD_STEMS = [
    "AIrIndent10000nm 02",
    "Tritium_Retention_Study_04.03.2026_0005",
    "Tritium_Retention_Study_04.03.2026_0009",
    "Tritium_Retention_Study_04.03.2026_THU_morning_0001",
]


def test_plot_experiments_returns_passed_axes_for_one_exp() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    experiment = next(
        exp
        for exp in study.experiments
        if exp.stem == "Tritium_Retention_Study_04.03.2026_0000"
    )
    _, ax = plt.subplots()

    returned_ax = plot_experiments(ax, experiment)

    assert returned_ax is ax
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == experiment.stem
    assert ax.get_title() == ""
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
    assert ax.get_legend() is None


def test_plot_experiments_can_overlay_group_on_one_axes() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    group = study.group_by_time_gap()[0]
    _, ax = plt.subplots()

    plot_experiments(ax, group, cmap="plasma")

    assert len(ax.lines) == 2
    assert ax.lines[0].get_color() != ax.lines[1].get_color()
    assert [line.get_label() for line in ax.lines] == [
        "Tritium_Retention_Study_04.03.2026_0000",
        "Tritium_Retention_Study_04.03.2026_0001",
    ]
    assert ax.get_legend() is None


def test_plot_experiments_can_flatten_a_study_with_selection() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    _, disabled_ax = plt.subplots()
    _, all_ax = plt.subplots()

    plot_experiments(disabled_ax, study, selection="disabled")
    plot_experiments(all_ax, study, selection="both")

    assert len(disabled_ax.lines) == 4
    assert len(all_ax.lines) == 8


def test_plot_experiments_smooths_both_x_and_y() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    experiment = next(
        exp
        for exp in study.experiments
        if exp.stem == "Tritium_Retention_Study_04.03.2026_0000"
    )
    smoothing = {"window_length": 21, "polyorder": 2}
    table = experiment.section("test")
    raw_x = np.asarray(table["disp_nm"], dtype=np.float64)
    raw_y = np.asarray(table["force_uN"], dtype=np.float64)
    _, ax = plt.subplots()

    plot_experiments(ax, experiment, smoothing=smoothing)

    assert np.allclose(
        ax.lines[0].get_xdata(),
        savgol(raw_x, **smoothing),
    )
    assert np.allclose(
        ax.lines[0].get_ydata(),
        savgol(raw_y, **smoothing),
    )


def test_plot_experiments_can_overlay_oliver_pharr_fit() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    experiment = next(
        exp
        for exp in study.experiments
        if exp.stem == "Tritium_Retention_Study_04.03.2026_0000"
    )
    batch = study.analyze_oliver_pharr()
    _, ax = plt.subplots()

    plot_experiments(ax, experiment, oliver_pharr=batch)

    assert len(ax.lines) == 2
    assert ax.lines[1].get_color() == "black"
    assert ax.lines[1].get_linestyle() == "--"
    assert ax.lines[1].get_linewidth() == 2.5
    assert ax.lines[1].get_label() == f"{experiment.stem} fit"


def test_plot_experiments_skips_missing_oliver_pharr_fit() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    experiment = next(
        exp for exp in study.experiments if exp.stem == "AIrIndent10000nm 02"
    )
    batch = study.analyze_oliver_pharr()
    _, ax = plt.subplots()

    plot_experiments(ax, experiment, selection="both", oliver_pharr=batch)

    assert len(ax.lines) == 1


def test_plot_experiments_allows_fit_style_overrides() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    experiment = next(
        exp
        for exp in study.experiments
        if exp.stem == "Tritium_Retention_Study_04.03.2026_0000"
    )
    batch = study.analyze_oliver_pharr()
    _, ax = plt.subplots()

    plot_experiments(
        ax,
        experiment,
        oliver_pharr=batch,
        fit_kwargs={"color": "red", "linewidth": 4.0, "linestyle": ":"},
    )

    assert ax.lines[1].get_color() == "red"
    assert ax.lines[1].get_linewidth() == 4.0
    assert ax.lines[1].get_linestyle() == ":"


def test_plot_experiments_rejects_fit_overlay_for_nonstandard_axes() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    batch = study.analyze_oliver_pharr()
    experiment = next(
        exp
        for exp in study.experiments
        if exp.stem == "Tritium_Retention_Study_04.03.2026_0000"
    )
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="Oliver-Pharr overlays"):
        plot_experiments(
            ax, experiment, x="time_s", y="force_uN", oliver_pharr=batch
        )


def test_plot_group_timeline_supports_study_and_explicit_groups() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    figure, axes = plot_group_timeline(study)
    explicit_figure, explicit_axes = plot_group_timeline(groups)

    assert figure is axes.figure
    assert explicit_figure is explicit_axes.figure
    assert len(axes.collections) == 2
    assert len(axes.lines) == 2
    assert [tick.get_text() for tick in axes.get_yticklabels()] == [
        "Group 0 (2 exp)",
        "Group 1 (2 exp)",
    ]


def test_plot_group_timeline_can_include_disabled_experiments() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)

    _, default_axes = plot_group_timeline(study)
    _, all_axes = plot_group_timeline(study, include_disabled=True)

    assert len(default_axes.collections) == 2
    assert len(default_axes.lines) == 2
    assert len(all_axes.collections) == 5
    assert len(all_axes.lines) == 5


def test_save_experiment_plots_uses_input_filenames_for_outputs(
    tmp_path: Path,
) -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)

    saved_paths = save_experiment_plots(study, tmp_path)

    assert len(saved_paths) == 4
    assert [path.name for path in saved_paths] == [
        "Tritium_Retention_Study_04.03.2026_0000.png",
        "Tritium_Retention_Study_04.03.2026_0001.png",
        "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0059.png",
        "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0060.png",
    ]
    assert all(path.exists() for path in saved_paths)


def test_save_experiment_plots_can_include_disabled_experiments(
    tmp_path: Path,
) -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)

    saved_paths = save_experiment_plots(
        study, tmp_path, selection="both", image_format="pdf"
    )

    assert len(saved_paths) == 8
    assert saved_paths[0].name == "AIrIndent10000nm 02.pdf"
    assert saved_paths[-1].name == (
        "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0060.pdf"
    )
    assert all(path.exists() for path in saved_paths)


def test_save_experiment_plots_can_target_only_disabled_experiments(
    tmp_path: Path,
) -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)

    saved_paths = save_experiment_plots(
        study, tmp_path, selection="disabled", image_format="png"
    )

    assert len(saved_paths) == 4
    assert [path.name for path in saved_paths] == [
        "AIrIndent10000nm 02.png",
        "Tritium_Retention_Study_04.03.2026_0005.png",
        "Tritium_Retention_Study_04.03.2026_0009.png",
        "Tritium_Retention_Study_04.03.2026_THU_morning_0001.png",
    ]
    assert all(path.exists() for path in saved_paths)


def test_save_experiment_plots_can_overlay_oliver_pharr_fits(
    tmp_path: Path,
) -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    batch = study.analyze_oliver_pharr()

    save_experiment_plots(
        [
            next(
                exp
                for exp in study.experiments
                if exp.stem == "Tritium_Retention_Study_04.03.2026_0000"
            )
        ],
        tmp_path,
        oliver_pharr=batch,
        close=False,
    )

    figure = plt.figure(plt.get_fignums()[-1])
    ax = figure.axes[0]

    assert len(ax.lines) == 2
    assert ax.lines[1].get_label() == (
        "Tritium_Retention_Study_04.03.2026_0000 fit"
    )
    plt.close(figure)
