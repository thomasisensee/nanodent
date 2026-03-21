from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from nanodent import load_folder, plot_group_timeline, plot_groups

DATA_DIR = Path(__file__).parent / "data"
BAD_STEMS = [
    "AIrIndent10000nm 02",
    "Tritium_Retention_Study_04.03.2026_0005",
    "Tritium_Retention_Study_04.03.2026_0009",
]


def test_plot_groups_defaults_to_one_panel_per_group() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    figure, axes = plot_groups(groups, linewidth=2.5, cmap="plasma")

    assert figure is axes[0, 0].figure
    assert axes.shape == (2, 1)
    assert len(axes[0, 0].lines) == 2
    assert len(axes[1, 0].lines) == 2
    assert all(
        line.get_linewidth() == 2.5
        for axis in axes[:, 0]
        for line in axis.lines
    )
    assert axes[0, 0].get_xlabel() == "disp_nm"
    assert axes[0, 0].get_ylabel() == "force_uN"


def test_plot_groups_restarts_colormap_for_each_group() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    _, axes = plot_groups(groups, cmap="plasma")

    assert axes[0, 0].lines[0].get_color() == axes[1, 0].lines[0].get_color()
    assert axes[0, 0].lines[1].get_color() == axes[1, 0].lines[1].get_color()


def test_plot_groups_overlay_preserves_combined_axes_behavior() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    figure, axes = plot_groups(groups, layout="overlay", cmap="plasma")

    assert figure is axes.figure
    assert len(axes.lines) == 4
    assert axes.get_ylabel() == "force_uN"


def test_plot_groups_overlay_restarts_colormap_for_each_group() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    _, axes = plot_groups(groups, layout="overlay", cmap="plasma")

    assert axes.lines[0].get_color() == axes.lines[2].get_color()
    assert axes.lines[1].get_color() == axes.lines[3].get_color()


def test_plot_groups_can_show_separate_slope_panels() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    figure, axes = plot_groups(
        groups[:1],
        smoothing={"window_length": 21, "polyorder": 2},
        show_slope=True,
        xlim=(0.0, 1500.0),
        ylim=(0.0, 1200.0),
        slope_ylim=(-20.0, 50.0),
    )

    assert figure is axes[0, 0].figure
    assert axes.shape == (1, 2)
    assert axes[0, 0].get_ylabel() == "force_uN"
    assert axes[0, 1].get_ylabel() == "d/ddisp_nm force_uN"
    assert axes[0, 0].get_xlim() == (0.0, 1500.0)
    assert axes[0, 1].get_ylim() == (-20.0, 50.0)
    assert len(axes[0, 1].lines) == 2


def test_plot_groups_alignment_clips_negative_shifted_x_by_default() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    _, axes = plot_groups(
        groups[:1],
        alignment={"method": "force_threshold", "force_threshold": 10.0},
    )

    assert all(np.min(line.get_xdata()) >= 0.0 for line in axes[0, 0].lines)


def test_plot_groups_can_keep_negative_shifted_x_when_requested() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)
    groups = study.group_by_time_gap()

    _, axes = plot_groups(
        groups[:1],
        alignment={"method": "force_threshold", "force_threshold": 10.0},
        clip_aligned_negative=False,
    )

    assert any(np.min(line.get_xdata()) < 0.0 for line in axes[0, 0].lines)


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


def test_plot_groups_rejects_slope_panels_in_overlay_mode() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)

    with pytest.raises(ValueError, match="layout='grid'"):
        plot_groups(
            study.group_by_time_gap(), layout="overlay", show_slope=True
        )


def test_plot_groups_can_include_disabled_experiments_when_requested() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)

    _, default_axes = plot_groups(study, layout="overlay")
    _, all_axes = plot_groups(study, layout="overlay", include_disabled=True)

    assert len(default_axes.lines) == 4
    assert len(all_axes.lines) == 7


def test_plot_group_timeline_can_include_disabled_experiments() -> None:
    study = load_folder(DATA_DIR).disable_experiments(BAD_STEMS)

    _, default_axes = plot_group_timeline(study)
    _, all_axes = plot_group_timeline(study, include_disabled=True)

    assert len(default_axes.collections) == 2
    assert len(default_axes.lines) == 2
    assert len(all_axes.collections) == 4
    assert len(all_axes.lines) == 4
