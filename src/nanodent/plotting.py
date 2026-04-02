"""Matplotlib plotting helpers for experiment data."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nanodent.analysis.filters import savgol
from nanodent.models import Experiment
from nanodent.study import ExperimentGroup, Study

PlotSelection = (
    Study
    | Experiment
    | ExperimentGroup
    | Sequence[Experiment]
    | Sequence[ExperimentGroup]
)


def plot_group_timeline(
    groups: Study | Sequence[ExperimentGroup],
    *,
    max_gap: timedelta = timedelta(minutes=30),
    include_disabled: bool = False,
    cmap: str = "tab10",
    marker: str = "o",
    markersize: float = 6.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a static timeline view of automatically grouped experiments."""

    resolved_groups = _coerce_timeline_groups(
        groups, max_gap=max_gap, include_disabled=include_disabled
    )
    figure: Figure
    if ax is None:
        figure, ax = plt.subplots(
            figsize=(10, max(2.5, len(resolved_groups) * 0.9))
        )
    else:
        figure = ax.figure

    if not resolved_groups:
        ax.set_title("Experiment Group Timeline")
        ax.set_xlabel("Acquisition time")
        ax.set_yticks([])
        return figure, ax

    color_map = plt.get_cmap(cmap, max(len(resolved_groups), 1))
    y_positions = np.arange(len(resolved_groups), dtype=float)

    for index, group in enumerate(resolved_groups):
        color = color_map(index)
        start_num = mdates.date2num(group.start)
        end_num = mdates.date2num(group.end)
        width = max(end_num - start_num, 1e-9)
        ax.broken_barh(
            [(start_num, width)],
            (y_positions[index] - 0.35, 0.7),
            facecolors=color,
            alpha=0.25,
        )
        timestamps = [
            mdates.date2num(experiment.timestamp)
            for experiment in group.experiments
        ]
        ax.plot(
            timestamps,
            np.full(len(timestamps), y_positions[index], dtype=float),
            linestyle="",
            marker=marker,
            markersize=markersize,
            color=color,
        )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            f"Group {group.index} ({len(group.experiments)} exp)"
            for group in resolved_groups
        ]
    )
    ax.set_xlabel("Acquisition time")
    ax.set_title("Experiment Group Timeline")
    ax.set_ylim(-0.75, len(resolved_groups) - 0.25)
    ax.grid(axis="x", alpha=0.3)
    figure.autofmt_xdate()
    return figure, ax


def plot_experiments(
    ax: Axes,
    target: PlotSelection,
    *,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    smoothing: Mapping[str, Any] | None = None,
    cmap: str = "viridis",
    max_gap: timedelta = timedelta(minutes=30),
    selection: Literal["enabled", "disabled", "both"] = "enabled",
    show_oliver_pharr: bool = True,
    fit_kwargs: Mapping[str, Any] | None = None,
    **line_kwargs: Any,
) -> Axes:
    """Plot one or more experiment curves onto an existing axes."""

    experiments = _coerce_experiments(
        target, max_gap=max_gap, selection=selection
    )
    use_force_displacement = (
        section == "test" and x == "disp_nm" and y == "force_uN"
    )

    color_map = plt.get_cmap(cmap, max(len(experiments), 1))
    for index, experiment in enumerate(experiments):
        curve = _prepare_curve(
            experiment,
            section=section,
            x=x,
            y=y,
            smoothing=smoothing,
        )
        curve_kwargs = dict(line_kwargs)
        curve_kwargs.setdefault("color", color_map(index))
        ax.plot(
            curve.x_values,
            curve.y_values,
            label=experiment.stem,
            **curve_kwargs,
        )

        fit_result = (
            experiment.oliver_pharr
            if show_oliver_pharr and use_force_displacement
            else None
        )
        _plot_oliver_pharr_overlay(
            ax,
            stem=experiment.stem,
            fit_result=fit_result,
            fit_kwargs=fit_kwargs,
        )

    return ax


def save_experiment_plots(
    groups: PlotSelection,
    output_dir: str | Path,
    *,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    smoothing: Mapping[str, Any] | None = None,
    max_gap: timedelta = timedelta(minutes=30),
    selection: Literal["enabled", "disabled", "both"] = "enabled",
    show_oliver_pharr: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    image_format: str = "png",
    dpi: float = 150.0,
    close: bool = True,
    fit_kwargs: Mapping[str, Any] | None = None,
    **line_kwargs: Any,
) -> list[Path]:
    """Save one plot per experiment using output names from `.hld` files."""

    experiments = _coerce_experiments(
        groups, max_gap=max_gap, selection=selection
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    suffix = f".{image_format.lstrip('.')}"
    for experiment in experiments:
        figure, ax = plt.subplots()
        plot_experiments(
            ax,
            experiment,
            section=section,
            x=x,
            y=y,
            smoothing=smoothing,
            max_gap=max_gap,
            selection="both",
            show_oliver_pharr=show_oliver_pharr,
            fit_kwargs=fit_kwargs,
            **line_kwargs,
        )
        ax.set_title(experiment.stem)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        _decorate_saved_experiment_axes(
            ax,
            experiment=experiment,
            section=section,
            x=x,
            y=y,
        )
        ax.grid(alpha=0.2)
        figure.tight_layout()

        output_path = (
            destination / experiment.paths.hld_path.with_suffix(suffix).name
        )
        figure.savefig(output_path, dpi=dpi)
        saved_paths.append(output_path)
        if close:
            plt.close(figure)

    return saved_paths


def _decorate_saved_experiment_axes(
    ax: Axes,
    *,
    experiment: Experiment,
    section: str,
    x: str,
    y: str,
) -> None:
    """Apply saved-plot-only annotations for one experiment axes."""

    ax.set_title(_format_saved_experiment_title(experiment))
    if not _uses_force_displacement_axes(section=section, x=x, y=y):
        return
    _add_saved_plot_top_axis(ax, experiment=experiment)


def _plot_oliver_pharr_overlay(
    ax: Axes,
    *,
    stem: str,
    fit_result: Any,
    fit_kwargs: Mapping[str, Any] | None,
) -> None:
    """Plot the fitted and softly extended Oliver-Pharr lines."""

    if fit_result is None or not fit_result.success:
        return
    if len(fit_result.x_fit) == 0 or len(fit_result.y_fit) == 0:
        return

    overlay_kwargs = {
        "alpha": 0.95,
        "color": "black",
        "label": f"{stem} fit",
        "linestyle": "--",
        "linewidth": 2.5,
        "zorder": 4,
    }
    if fit_kwargs is not None:
        overlay_kwargs.update(dict(fit_kwargs))
    ax.plot(fit_result.x_fit, fit_result.y_fit, **overlay_kwargs)

    extension_segment = _oliver_pharr_extension_segment(fit_result)
    if extension_segment is None:
        return

    x_values, y_values = extension_segment
    extension_kwargs = {
        "alpha": min(float(overlay_kwargs.get("alpha", 1.0)) * 0.45, 1.0),
        "color": overlay_kwargs.get("color", "black"),
        "linestyle": ":",
        "linewidth": max(
            float(overlay_kwargs.get("linewidth", 1.0)) * 0.5, 0.75
        ),
        "zorder": max(float(overlay_kwargs.get("zorder", 4)) - 0.1, 0.0),
        "label": "_nolegend_",
    }
    ax.plot(x_values, y_values, **extension_kwargs)


def _uses_force_displacement_axes(*, section: str, x: str, y: str) -> bool:
    """Return whether the axes represent the test force-displacement view."""

    return section == "test" and x == "disp_nm" and y == "force_uN"


def _format_saved_experiment_title(experiment: Experiment) -> str:
    """Return the saved-plot title including stiffness when available."""

    title = experiment.stem
    fit_result = experiment.oliver_pharr
    if fit_result is None or not fit_result.success:
        return title
    if fit_result.stiffness_uN_per_nm is None:
        return title
    stiffness = fit_result.stiffness_uN_per_nm
    return f"{title} | S={stiffness:.2f} uN/nm"


def _add_saved_plot_top_axis(ax: Axes, *, experiment: Experiment) -> None:
    """Add a top displacement axis marking onset and maximum force."""

    tick_positions, tick_labels = _saved_plot_top_axis_ticks(experiment)
    if not tick_positions:
        return

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks(tick_positions)
    top_ax.set_xticklabels(tick_labels)
    top_ax.xaxis.set_ticks_position("top")
    top_ax.xaxis.set_label_position("top")
    top_ax.set_xlabel("")
    top_ax.grid(False)


def _saved_plot_top_axis_ticks(
    experiment: Experiment,
) -> tuple[list[float], list[str]]:
    """Return displacement positions and labels for saved-plot top ticks."""

    tick_pairs: list[tuple[float, str]] = []

    onset = experiment.onset
    if onset is not None and onset.success and onset.onset_disp_nm is not None:
        tick_pairs.append(
            (
                float(onset.onset_disp_nm),
                _format_disp_tick(onset.onset_disp_nm),
            )
        )

    max_force_disp = _max_force_disp_nm(experiment)
    if max_force_disp is not None and not any(
        np.isclose(position, max_force_disp, atol=1e-12)
        for position, _ in tick_pairs
    ):
        tick_pairs.append((max_force_disp, _format_disp_tick(max_force_disp)))

    return (
        [position for position, _ in tick_pairs],
        [label for _, label in tick_pairs],
    )


def _max_force_disp_nm(experiment: Experiment) -> float | None:
    """Return the displacement at the raw maximum-force sample."""

    force = np.asarray(experiment.test["force_uN"], dtype=np.float64)
    disp = np.asarray(experiment.test["disp_nm"], dtype=np.float64)
    if len(force) == 0 or len(disp) == 0:
        return None
    max_index = int(np.argmax(force))
    return float(disp[max_index])


def _format_disp_tick(value_nm: float) -> str:
    """Return a compact displacement label for top-axis ticks."""

    return f"{float(value_nm):.3g}"


def _oliver_pharr_extension_segment(
    fit_result: Any,
) -> tuple[list[float], list[float]] | None:
    """Return the softened extension segment from intercept to fit start."""

    if fit_result.depth_intercept_nm is None:
        return None
    if fit_result.force_intercept_uN is None:
        return None
    if fit_result.stiffness_uN_per_nm is None:
        return None
    if len(fit_result.x_fit) == 0 or len(fit_result.y_fit) == 0:
        return None

    start_x = float(fit_result.depth_intercept_nm)
    start_y = float(
        fit_result.force_intercept_uN
        + fit_result.stiffness_uN_per_nm * fit_result.depth_intercept_nm
    )
    end_x = float(fit_result.x_fit[0])
    end_y = float(fit_result.y_fit[0])
    return [start_x, end_x], [start_y, end_y]


class _PreparedCurve:
    """Prepared plotting arrays for one experiment."""

    def __init__(
        self,
        *,
        x_values: NDArray[np.float64],
        y_values: NDArray[np.float64],
    ) -> None:
        self.x_values = x_values
        self.y_values = y_values


def _prepare_curve(
    experiment: Experiment,
    *,
    section: str,
    x: str,
    y: str,
    smoothing: Mapping[str, Any] | None,
) -> _PreparedCurve:
    """Return plotting arrays for one experiment and signal selection."""

    table = experiment.section(section)
    x_values = np.asarray(table[x], dtype=np.float64)
    y_values = np.asarray(table[y], dtype=np.float64)
    if smoothing is not None:
        smoothing_kwargs = dict(smoothing)
        x_values = savgol(x_values, **smoothing_kwargs)
        y_values = savgol(y_values, **smoothing_kwargs)
    return _PreparedCurve(x_values=x_values, y_values=y_values)


def _coerce_experiments(
    target: PlotSelection,
    *,
    max_gap: timedelta,
    selection: Literal["enabled", "disabled", "both"],
) -> list[Experiment]:
    """Normalize supported plotting inputs into experiments by status."""

    if isinstance(target, Study):
        groups = target.group_by_time_gap(
            max_gap=max_gap, include_disabled=True
        )
        return _select_experiments(
            [
                experiment
                for group in groups
                for experiment in group.experiments
            ],
            selection=selection,
        )
    if isinstance(target, Experiment):
        return _select_experiments([target], selection=selection)
    if isinstance(target, ExperimentGroup):
        return _select_experiments(target.experiments, selection=selection)

    items = list(target)
    if not items:
        return []
    first_item = items[0]
    if isinstance(first_item, ExperimentGroup):
        return _select_experiments(
            [
                experiment
                for group in items
                for experiment in group.experiments
            ],
            selection=selection,
        )
    if isinstance(first_item, Experiment):
        return _select_experiments(items, selection=selection)
    raise TypeError(
        "plot_experiments expects a Study, Experiment, ExperimentGroup, "
        "or a sequence of Experiment/ExperimentGroup objects."
    )


def _select_experiments(
    experiments: Sequence[Experiment],
    *,
    selection: Literal["enabled", "disabled", "both"],
) -> list[Experiment]:
    """Return experiments filtered by enabled/disabled status."""

    if selection == "enabled":
        return [experiment for experiment in experiments if experiment.enabled]
    if selection == "disabled":
        return [
            experiment for experiment in experiments if not experiment.enabled
        ]
    if selection == "both":
        return list(experiments)
    raise ValueError(
        "selection must be one of 'enabled', 'disabled', or 'both'."
    )


def _coerce_timeline_groups(
    groups: Study | Sequence[ExperimentGroup],
    *,
    max_gap: timedelta,
    include_disabled: bool,
) -> list[ExperimentGroup]:
    """Normalize supported group inputs into timeline-ready groups."""

    if isinstance(groups, Study):
        return groups.group_by_time_gap(
            max_gap=max_gap, include_disabled=include_disabled
        )
    return _filtered_groups(groups, include_disabled=include_disabled)


def _filtered_groups(
    groups: Sequence[ExperimentGroup], *, include_disabled: bool
) -> list[ExperimentGroup]:
    """Return groups with optional filtering of disabled experiments."""

    filtered: list[ExperimentGroup] = []
    for group in groups:
        experiments = tuple(
            experiment
            for experiment in group.experiments
            if include_disabled or experiment.enabled
        )
        if not experiments:
            continue
        filtered.append(
            ExperimentGroup(experiments=experiments, index=group.index)
        )
    return filtered
