from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from nanodent import load_folder, plot_group_timeline, plot_groups

DATA_DIR = Path(__file__).parent / "data"


def test_plot_groups_defaults_to_one_panel_per_group() -> None:
    study = load_folder(DATA_DIR)
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


def test_plot_groups_overlay_preserves_combined_axes_behavior() -> None:
    study = load_folder(DATA_DIR)
    groups = study.group_by_time_gap()

    figure, axes = plot_groups(groups, layout="overlay", cmap="plasma")

    assert figure is axes.figure
    assert len(axes.lines) == 4
    assert axes.get_ylabel() == "force_uN"


def test_plot_groups_can_show_separate_slope_panels() -> None:
    study = load_folder(DATA_DIR)
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
    study = load_folder(DATA_DIR)
    groups = study.group_by_time_gap()

    _, axes = plot_groups(
        groups[:1],
        alignment={"method": "force_threshold", "force_threshold": 10.0},
    )

    assert all(np.min(line.get_xdata()) >= 0.0 for line in axes[0, 0].lines)


def test_plot_groups_can_keep_negative_shifted_x_when_requested() -> None:
    study = load_folder(DATA_DIR)
    groups = study.group_by_time_gap()

    _, axes = plot_groups(
        groups[:1],
        alignment={"method": "force_threshold", "force_threshold": 10.0},
        clip_aligned_negative=False,
    )

    assert any(np.min(line.get_xdata()) < 0.0 for line in axes[0, 0].lines)


def test_plot_group_timeline_supports_study_and_explicit_groups() -> None:
    study = load_folder(DATA_DIR)
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
    study = load_folder(DATA_DIR)

    with pytest.raises(ValueError, match="layout='grid'"):
        plot_groups(
            study.group_by_time_gap(), layout="overlay", show_slope=True
        )
