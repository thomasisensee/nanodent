"""Matplotlib plotting helpers for experiment groups."""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from nanodent.analysis.align import align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.models import Experiment
from nanodent.study import ExperimentGroup, Study

LineGroups = (
    Study | ExperimentGroup | Sequence[ExperimentGroup] | Sequence[Experiment]
)
AxesGrid = Axes | NDArray[np.object_]


def plot_groups(
    groups: LineGroups,
    *,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    cmap: str = "viridis",
    alignment: str | Mapping[str, Any] | None = None,
    smoothing: Mapping[str, Any] | None = None,
    derivative: bool = False,
    layout: Literal["grid", "overlay"] = "grid",
    show_slope: bool = False,
    clip_aligned_negative: bool = True,
    max_gap: timedelta = timedelta(minutes=30),
    include_disabled: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    slope_ylim: tuple[float, float] | None = None,
    sharex: bool = True,
    ax: Axes | None = None,
    **line_kwargs: Any,
) -> tuple[Figure, AxesGrid]:
    """Plot experiment groups as either a grid or an overlay.

    Args:
        groups: Study, experiment group, or experiment sequence to plot.
        section: Section name to read from each experiment.
        x: Column name to plot on the x-axis.
        y: Column name to plot on the y-axis before optional derivative
            processing.
        cmap: Matplotlib colormap name used to color the curves.
        alignment: Optional alignment configuration. Pass a method name or a
            mapping of keyword arguments for `align_curve`.
        smoothing: Optional keyword arguments for `nanodent.savgol`.
        derivative: If `True`, the main curve panel shows the numerical
            derivative of the chosen signal with respect to `x`.
        layout: Plot layout. `grid` creates one panel per group, while
            `overlay` combines all groups on one axes.
        show_slope: If `True` in grid mode, add a second subplot row per group
            showing slopes separately from the main curve panel.
        clip_aligned_negative: If `True`, hide aligned samples whose shifted
            x-values are negative.
        max_gap: Time gap used when `groups` is a `Study`.
        include_disabled: Whether disabled experiments should remain visible in
            the plotted groups.
        xlim: Optional x-axis limits applied to every subplot.
        ylim: Optional y-axis limits applied to the main curve panels.
        slope_ylim: Optional y-axis limits applied to slope panels.
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
    if layout == "overlay" and show_slope:
        raise ValueError(
            "Separate slope panels are only supported when layout='grid'."
        )
    if layout == "overlay":
        return _plot_groups_overlay(
            resolved_groups,
            section=section,
            x=x,
            y=y,
            cmap=cmap,
            alignment=alignment,
            smoothing=smoothing,
            derivative=derivative,
            clip_aligned_negative=clip_aligned_negative,
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
        alignment=alignment,
        smoothing=smoothing,
        derivative=derivative,
        show_slope=show_slope,
        clip_aligned_negative=clip_aligned_negative,
        xlim=xlim,
        ylim=ylim,
        slope_ylim=slope_ylim,
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


def _plot_groups_overlay(
    groups: list[ExperimentGroup],
    *,
    section: str,
    x: str,
    y: str,
    cmap: str,
    alignment: str | Mapping[str, Any] | None,
    smoothing: Mapping[str, Any] | None,
    derivative: bool,
    clip_aligned_negative: bool,
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
                alignment=alignment,
                smoothing=smoothing,
                clip_aligned_negative=clip_aligned_negative,
            )
            y_values = curve.main_derivative if derivative else curve.display_y
            ax.plot(
                curve.x_values,
                y_values,
                color=color_map(color_index),
                label=experiment.stem,
                **line_kwargs,
            )

    ax.set_xlabel(x)
    ax.set_ylabel(f"d/d{x} {y}" if derivative else y)
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
    alignment: str | Mapping[str, Any] | None,
    smoothing: Mapping[str, Any] | None,
    derivative: bool,
    show_slope: bool,
    clip_aligned_negative: bool,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    slope_ylim: tuple[float, float] | None,
    sharex: bool,
    **line_kwargs: Any,
) -> tuple[Figure, NDArray[np.object_]]:
    group_count = max(len(groups), 1)
    total_rows = group_count * (2 if show_slope else 1)
    figure, axes = plt.subplots(
        total_rows,
        1,
        squeeze=False,
        sharex=sharex,
        figsize=(9, max(3.5, group_count * (5.4 if show_slope else 3.6))),
        gridspec_kw=(
            {
                "height_ratios": [
                    ratio for _ in range(group_count) for ratio in (1.0, 0.7)
                ]
            }
            if show_slope
            else None
        ),
    )

    if not groups:
        main_ax = axes[0, 0]
        main_ax.set_title("No experiment groups")
        main_ax.set_xlabel(x)
        main_ax.set_ylabel(f"d/d{x} {y}" if derivative else y)
        return figure, axes

    for row_index, group in enumerate(groups):
        main_ax = axes[row_index * (2 if show_slope else 1), 0]
        slope_ax = axes[row_index * 2 + 1, 0] if show_slope else None
        color_map = plt.get_cmap(cmap, max(len(group.experiments), 1))

        for color_index, experiment in enumerate(group.experiments):
            curve = _prepare_curve(
                experiment,
                section=section,
                x=x,
                y=y,
                alignment=alignment,
                smoothing=smoothing,
                clip_aligned_negative=clip_aligned_negative,
            )
            color = color_map(color_index)
            main_y = curve.main_derivative if derivative else curve.display_y
            main_ax.plot(
                curve.x_values,
                main_y,
                color=color,
                label=experiment.stem,
                **line_kwargs,
            )
            if slope_ax is not None:
                slope_ax.plot(
                    curve.x_values,
                    curve.slope_y,
                    color=color,
                    label=experiment.stem,
                    **line_kwargs,
                )

        _decorate_group_axes(
            main_ax,
            group,
            x=x,
            y=y,
            derivative=derivative,
            xlim=xlim,
            ylim=ylim,
            show_legend=False,
        )
        if slope_ax is not None:
            slope_ax.set_title(f"Group {group.index} slope")
            slope_ax.set_xlabel(x)
            slope_ax.set_ylabel(f"d/d{x} {y}")
            if xlim is not None:
                slope_ax.set_xlim(*xlim)
            if slope_ylim is not None:
                slope_ax.set_ylim(*slope_ylim)
            slope_ax.grid(alpha=0.2)

    figure.tight_layout()
    if show_slope:
        grouped_axes = np.empty((len(groups), 2), dtype=object)
        for row_index in range(len(groups)):
            grouped_axes[row_index, 0] = axes[row_index * 2, 0]
            grouped_axes[row_index, 1] = axes[row_index * 2 + 1, 0]
        return figure, grouped_axes
    return figure, axes[: len(groups), :]


class _PreparedCurve:
    """Prepared plotting arrays for one experiment."""

    def __init__(
        self,
        *,
        x_values: NDArray[np.float64],
        display_y: NDArray[np.float64],
        main_derivative: NDArray[np.float64],
        slope_y: NDArray[np.float64],
    ) -> None:
        self.x_values = x_values
        self.display_y = display_y
        self.main_derivative = main_derivative
        self.slope_y = slope_y


def _prepare_curve(
    experiment: Experiment,
    *,
    section: str,
    x: str,
    y: str,
    alignment: str | Mapping[str, Any] | None,
    smoothing: Mapping[str, Any] | None,
    clip_aligned_negative: bool,
) -> _PreparedCurve:
    table = experiment.section(section)
    x_values = np.asarray(table[x], dtype=np.float64)
    raw_y = np.asarray(table[y], dtype=np.float64)
    display_y = (
        savgol(raw_y, **dict(smoothing))
        if smoothing is not None
        else raw_y.copy()
    )

    if alignment is not None:
        alignment_kwargs = (
            {"method": alignment}
            if isinstance(alignment, str)
            else dict(alignment)
        )
        aligned = align_curve(x_values, raw_y, **alignment_kwargs)
        x_values = aligned.shifted_x
        if clip_aligned_negative:
            mask = x_values >= 0.0
            x_values = x_values[mask]
            raw_y = raw_y[mask]
            display_y = display_y[mask]

    main_derivative = _gradient_or_zeros(display_y, x_values)
    slope_source = display_y if smoothing is not None else raw_y
    slope_y = _gradient_or_zeros(slope_source, x_values)
    return _PreparedCurve(
        x_values=x_values,
        display_y=display_y,
        main_derivative=main_derivative,
        slope_y=slope_y,
    )


def _decorate_group_axes(
    ax: Axes,
    group: ExperimentGroup,
    *,
    x: str,
    y: str,
    derivative: bool,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    show_legend: bool,
) -> None:
    ax.set_title(f"Group {group.index} ({len(group.experiments)} experiments)")
    ax.set_xlabel(x)
    ax.set_ylabel(f"d/d{x} {y}" if derivative else y)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)
    if show_legend:
        ax.legend()


def _gradient_or_zeros(
    values: NDArray[np.float64], x_values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return a gradient when possible, otherwise a zero-valued fallback.

    Args:
        values: Signal values to differentiate.
        x_values: X-axis coordinates associated with `values`.

    Returns:
        Gradient array, or zeros for signals that are too short.
    """

    if len(values) < 2:
        return np.zeros_like(values)
    return gradient(values, x_values)


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
