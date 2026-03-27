"""Matplotlib plotting helpers for experiment groups."""

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
from nanodent.analysis.oliver_pharr import OliverPharrBatchResult
from nanodent.models import Experiment
from nanodent.study import ExperimentGroup, Study

LineGroups = (
    Study | ExperimentGroup | Sequence[ExperimentGroup] | Sequence[Experiment]
)
CurveSelection = Experiment | ExperimentGroup | Sequence[Experiment]
AxesGrid = Axes | NDArray[np.object_]


def plot_groups(
    groups: LineGroups,
    *,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    cmap: str = "viridis",
    smoothing: Mapping[str, Any] | None = None,
    layout: Literal["grid", "overlay"] = "grid",
    max_gap: timedelta = timedelta(minutes=30),
    include_disabled: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    sharex: bool = True,
    ax: Axes | None = None,
    **line_kwargs: Any,
) -> tuple[Figure, AxesGrid]:
    """Plot experiment groups as either a grid or an overlay.

    Args:
        groups: Study, experiment group, or experiment sequence to plot.
        section: Section name to read from each experiment.
        x: Column name to plot on the x-axis.
        y: Column name to plot on the y-axis.
        cmap: Matplotlib colormap name used to color the curves.
        smoothing: Optional keyword arguments for `nanodent.savgol`.
        layout: Plot layout. `grid` creates one panel per group, while
            `overlay` combines all groups on one axes.
        max_gap: Time gap used when `groups` is a `Study`.
        include_disabled: Whether disabled experiments should remain visible in
            the plotted groups.
        xlim: Optional x-axis limits applied to every subplot.
        ylim: Optional y-axis limits applied to the main curve panels.
        sharex: Whether grid subplots should share an x-axis.
        ax: Existing axes used only in `overlay` mode.
        **line_kwargs: Additional keyword arguments passed through to
            `Axes.plot`.

    Returns:
        Figure and axes containing the plotted curves. In `grid` mode the axes
        object is a NumPy object array; in `overlay` mode it is a single axes.
    """

    resolved_groups = _coerce_groups(
        groups, max_gap=max_gap, include_disabled=include_disabled
    )
    if layout == "grid" and ax is not None:
        raise ValueError(
            "The 'ax' parameter is only supported when layout='overlay'."
        )
    if layout == "overlay":
        return _plot_groups_overlay(
            resolved_groups,
            section=section,
            x=x,
            y=y,
            cmap=cmap,
            smoothing=smoothing,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            **line_kwargs,
        )
    return _plot_groups_grid(
        resolved_groups,
        section=section,
        x=x,
        y=y,
        cmap=cmap,
        smoothing=smoothing,
        xlim=xlim,
        ylim=ylim,
        sharex=sharex,
        **line_kwargs,
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
    """Plot a static timeline view of automatically grouped experiments.

    Args:
        groups: Study to group automatically, or explicit experiment groups to
            display as-is.
        max_gap: Time gap used when `groups` is a `Study`.
        include_disabled: Whether disabled experiments should remain visible in
            the timeline.
        cmap: Matplotlib colormap name used for group colors.
        marker: Marker style used for experiment timestamps.
        markersize: Marker size used for experiment timestamps.
        ax: Existing axes to draw on. A new figure and axes are created when
            omitted.

    Returns:
        Figure and axes containing the group timeline visualization.
    """

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


def plot_force_displacement(
    target: CurveSelection,
    *,
    oliver_pharr: OliverPharrBatchResult | None = None,
    smoothing: Mapping[str, Any] | None = None,
    ax: Axes | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    **line_kwargs: Any,
) -> Axes:
    """Plot force versus displacement for explicit experiment selections.

    Args:
        target: Experiment, experiment group, or explicit experiment sequence
            to overlay on one axes.
        oliver_pharr: Optional batch result used to overlay successful fitted
            unloading lines by matching experiment stems.
        smoothing: Optional keyword arguments for `nanodent.savgol`. When
            provided, the same filter is applied to displacement and force
            before plotting.
        ax: Existing axes to draw on. A new axes is created when omitted.
        fit_kwargs: Optional keyword arguments applied only to Oliver-Pharr fit
            overlays. These override the default dashed-line styling.
        **line_kwargs: Additional keyword arguments passed through to
            `Axes.plot` for the experiment curves.

    Returns:
        Axes containing the plotted force-displacement curves.
    """

    experiments, default_title = _coerce_curve_selection(target)
    if ax is None:
        _, ax = plt.subplots()

    if not experiments:
        ax.set_title(default_title)
        ax.set_xlabel("disp_nm")
        ax.set_ylabel("force_uN")
        ax.grid(alpha=0.2)
        return ax

    fit_lookup = _oliver_pharr_lookup(oliver_pharr)
    for experiment in experiments:
        curve = _prepare_force_displacement_curve(
            experiment, smoothing=smoothing
        )
        ax.plot(
            curve.x_values,
            curve.y_values,
            label=experiment.stem,
            **line_kwargs,
        )

        fit_result = fit_lookup.get(experiment.stem)
        if fit_result is None or not fit_result.success:
            continue
        if len(fit_result.x_fit) == 0 or len(fit_result.y_fit) == 0:
            continue
        overlay_kwargs = {
            "alpha": 0.95,
            "color": "black",
            "label": f"{experiment.stem} fit",
            "linestyle": "--",
            "linewidth": 2.5,
            "zorder": 4,
        }
        if fit_kwargs is not None:
            overlay_kwargs.update(dict(fit_kwargs))
        ax.plot(fit_result.x_fit, fit_result.y_fit, **overlay_kwargs)

    ax.set_title(default_title)
    ax.set_xlabel("disp_nm")
    ax.set_ylabel("force_uN")
    ax.grid(alpha=0.2)
    if len(experiments) > 1:
        ax.legend()
    return ax


def save_experiment_plots(
    groups: LineGroups,
    output_dir: str | Path,
    *,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    smoothing: Mapping[str, Any] | None = None,
    max_gap: timedelta = timedelta(minutes=30),
    include_disabled: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    image_format: str = "png",
    dpi: float = 150.0,
    close: bool = True,
    **line_kwargs: Any,
) -> list[Path]:
    """Save one simple plot per experiment using output names from
    `.hld` files.

    Args:
        groups: Study, experiment group, or experiment sequence to plot.
        output_dir: Directory where individual plot files will be written.
        section: Section name to read from each experiment.
        x: Column name to plot on the x-axis.
        y: Column name to plot on the y-axis.
        smoothing: Optional keyword arguments for `nanodent.savgol`.
        max_gap: Time gap used when `groups` is a `Study`.
        include_disabled: Whether disabled experiments should be included.
        xlim: Optional x-axis limits applied to every saved plot.
        ylim: Optional y-axis limits applied to every saved plot.
        image_format: File format/extension passed to Matplotlib.
        dpi: Rasterization density used by `Figure.savefig`.
        close: Whether to close each figure after saving.
        **line_kwargs: Additional keyword arguments passed through to
            `Axes.plot`.

    Returns:
        Paths to the saved plot files.
    """

    experiments = _coerce_experiments(
        groups, max_gap=max_gap, include_disabled=include_disabled
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    suffix = f".{image_format.lstrip('.')}"
    for experiment in experiments:
        figure, ax = plt.subplots()
        curve = _prepare_curve(
            experiment,
            section=section,
            x=x,
            y=y,
            smoothing=smoothing,
        )
        y_values = curve.display_y
        ax.plot(curve.x_values, y_values, **line_kwargs)
        ax.set_title(experiment.stem)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
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


def _plot_groups_overlay(
    groups: list[ExperimentGroup],
    *,
    section: str,
    x: str,
    y: str,
    cmap: str,
    smoothing: Mapping[str, Any] | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    ax: Axes | None,
    **line_kwargs: Any,
) -> tuple[Figure, Axes]:
    figure: Figure
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    for group in groups:
        color_map = plt.get_cmap(cmap, max(len(group.experiments), 1))
        for color_index, experiment in enumerate(group.experiments):
            curve = _prepare_curve(
                experiment,
                section=section,
                x=x,
                y=y,
                smoothing=smoothing,
            )
            ax.plot(
                curve.x_values,
                curve.display_y,
                color=color_map(color_index),
                label=experiment.stem,
                **line_kwargs,
            )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    return figure, ax


def _plot_groups_grid(
    groups: list[ExperimentGroup],
    *,
    section: str,
    x: str,
    y: str,
    cmap: str,
    smoothing: Mapping[str, Any] | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    sharex: bool,
    **line_kwargs: Any,
) -> tuple[Figure, NDArray[np.object_]]:
    group_count = max(len(groups), 1)
    figure, axes = plt.subplots(
        group_count,
        1,
        squeeze=False,
        sharex=sharex,
        figsize=(9, max(3.5, group_count * 3.6)),
    )

    if not groups:
        main_ax = axes[0, 0]
        main_ax.set_title("No experiment groups")
        main_ax.set_xlabel(x)
        main_ax.set_ylabel(y)
        return figure, axes

    for row_index, group in enumerate(groups):
        main_ax = axes[row_index, 0]
        color_map = plt.get_cmap(cmap, max(len(group.experiments), 1))

        for color_index, experiment in enumerate(group.experiments):
            curve = _prepare_curve(
                experiment,
                section=section,
                x=x,
                y=y,
                smoothing=smoothing,
            )
            color = color_map(color_index)
            main_ax.plot(
                curve.x_values,
                curve.display_y,
                color=color,
                label=experiment.stem,
                **line_kwargs,
            )

        _decorate_group_axes(
            main_ax,
            group,
            x=x,
            y=y,
            xlim=xlim,
            ylim=ylim,
            show_legend=False,
        )

    figure.tight_layout()
    return figure, axes[: len(groups), :]


class _PreparedCurve:
    """Prepared plotting arrays for one experiment."""

    def __init__(
        self,
        *,
        x_values: NDArray[np.float64],
        display_y: NDArray[np.float64],
    ) -> None:
        self.x_values = x_values
        self.display_y = display_y


class _ForceDisplacementCurve:
    """Prepared force-displacement arrays for one experiment."""

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
    table = experiment.section(section)
    x_values = np.asarray(table[x], dtype=np.float64)
    raw_y = np.asarray(table[y], dtype=np.float64)
    display_y = (
        savgol(raw_y, **dict(smoothing))
        if smoothing is not None
        else raw_y.copy()
    )

    return _PreparedCurve(
        x_values=x_values,
        display_y=display_y,
    )


def _prepare_force_displacement_curve(
    experiment: Experiment,
    *,
    smoothing: Mapping[str, Any] | None,
) -> _ForceDisplacementCurve:
    """Return explicit force-displacement arrays for one experiment."""

    table = experiment.section("test")
    x_values = np.asarray(table["disp_nm"], dtype=np.float64)
    y_values = np.asarray(table["force_uN"], dtype=np.float64)
    if smoothing is not None:
        smoothing_kwargs = dict(smoothing)
        x_values = savgol(x_values, **smoothing_kwargs)
        y_values = savgol(y_values, **smoothing_kwargs)
    return _ForceDisplacementCurve(x_values=x_values, y_values=y_values)


def _decorate_group_axes(
    ax: Axes,
    group: ExperimentGroup,
    *,
    x: str,
    y: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    show_legend: bool,
) -> None:
    ax.set_title(f"Group {group.index} ({len(group.experiments)} experiments)")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    if show_legend:
        ax.legend()


def _coerce_groups(
    groups: LineGroups, *, max_gap: timedelta, include_disabled: bool
) -> list[ExperimentGroup]:
    """Normalize supported group inputs into experiment groups.

    Args:
        groups: Supported grouping input accepted by `plot_groups`.
        max_gap: Time gap used when `groups` is a `Study`.

    Returns:
        Concrete experiment groups ready for plotting.
    """

    if isinstance(groups, Study):
        return groups.group_by_time_gap(
            max_gap=max_gap, include_disabled=include_disabled
        )
    if isinstance(groups, ExperimentGroup):
        return _filtered_groups([groups], include_disabled=include_disabled)
    groups_list = list(groups)
    if not groups_list:
        return []
    first_item = groups_list[0]
    if isinstance(first_item, ExperimentGroup):
        return _filtered_groups(groups_list, include_disabled=include_disabled)
    if isinstance(first_item, Experiment):
        selected_experiments = tuple(
            experiment
            for experiment in groups_list
            if include_disabled or experiment.enabled
        )
        if not selected_experiments:
            return []
        return [ExperimentGroup(experiments=selected_experiments, index=0)]
    raise TypeError(
        "plot_groups expects a Study, ExperimentGroup, or a sequence "
        "of those objects."
    )


def _coerce_curve_selection(
    target: CurveSelection,
) -> tuple[tuple[Experiment, ...], str]:
    """Normalize explicit curve selections into experiments and a title."""

    if isinstance(target, Experiment):
        return (target,), target.stem
    if isinstance(target, ExperimentGroup):
        return (
            target.experiments,
            f"Group {target.index} ({len(target.experiments)} experiments)",
        )
    experiments = tuple(target)
    if not experiments:
        return (), "Force vs. Displacement"
    first_item = experiments[0]
    if not isinstance(first_item, Experiment):
        raise TypeError(
            "plot_force_displacement expects an Experiment, "
            "ExperimentGroup, or a sequence of Experiment objects."
        )
    return experiments, "Force vs. Displacement"


def _oliver_pharr_lookup(
    oliver_pharr: OliverPharrBatchResult | None,
) -> dict[str, Any]:
    """Return per-stem Oliver-Pharr results for plotting overlays."""

    if oliver_pharr is None:
        return {}
    return {result.stem: result for result in oliver_pharr.results}


def _coerce_experiments(
    groups: LineGroups, *, max_gap: timedelta, include_disabled: bool
) -> list[Experiment]:
    """Normalize supported plotting inputs into a flat experiment list."""

    resolved_groups = _coerce_groups(
        groups, max_gap=max_gap, include_disabled=include_disabled
    )
    return [
        experiment
        for group in resolved_groups
        for experiment in group.experiments
    ]


def _coerce_timeline_groups(
    groups: Study | Sequence[ExperimentGroup],
    *,
    max_gap: timedelta,
    include_disabled: bool,
) -> list[ExperimentGroup]:
    """Normalize supported group inputs into timeline-ready experiment groups.

    Args:
        groups: Study or explicit experiment groups.
        max_gap: Time gap used when `groups` is a `Study`.

    Returns:
        Concrete experiment groups ready for timeline plotting.
    """

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
