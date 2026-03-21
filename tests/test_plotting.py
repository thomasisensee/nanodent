from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from nanodent import load_folder, plot_groups

DATA_DIR = Path(__file__).parent / "data"


def test_plot_groups_returns_figure_and_axes() -> None:
    study = load_folder(DATA_DIR)
    groups = study.group_by_time_gap()

    figure, axes = plot_groups(groups, linewidth=2.5, cmap="plasma")

    assert figure is axes.figure
    assert len(axes.lines) == 4
    assert all(line.get_linewidth() == 2.5 for line in axes.lines)
    assert axes.get_xlabel() == "disp_nm"
    assert axes.get_ylabel() == "force_uN"


def test_plot_groups_supports_smoothing_derivative_and_alignment() -> None:
    study = load_folder(DATA_DIR)
    groups = study.group_by_time_gap()

    figure, axes = plot_groups(
        groups[:1],
        smoothing={"window_length": 21, "polyorder": 2},
        derivative=True,
        alignment={"method": "force_threshold", "force_threshold": 10.0},
    )

    assert figure is axes.figure
    assert len(axes.lines) == 2
    assert axes.get_ylabel() == "d/ddisp_nm force_uN"
