"""Matplotlib plotting helpers for experiment groups."""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from nanodent.analysis.align import align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.models import Experiment
from nanodent.study import ExperimentGroup, Study


def plot_groups(
    groups: Study
    | ExperimentGroup
    | Sequence[ExperimentGroup]
    | Sequence[Experiment],
    *,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    cmap: str = "viridis",
    alignment: str | Mapping[str, Any] | None = None,
    smoothing: Mapping[str, Any] | None = None,
    derivative: bool = False,
    ax: Axes | None = None,
    **line_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot one or more experiment groups onto a single Matplotlib axes.

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
        derivative: If `True`, plot the numerical derivative of the chosen
            signal with respect to `x`.
        ax: Existing axes to draw on. A new figure and axes are created when
            omitted.
        **line_kwargs: Additional keyword arguments passed through to
            `Axes.plot`.

    Returns:
        Figure and axes containing the plotted curves.
    """

    resolved_groups = _coerce_groups(groups)
    figure: Figure
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    total_experiments = sum(
        len(group.experiments) for group in resolved_groups
    )
    color_map = plt.get_cmap(cmap, max(total_experiments, 1))

    color_index = 0
    for group in resolved_groups:
        for experiment in group.experiments:
            table = experiment.section(section)
            x_values = table[x]
            y_values = table[y]

            if smoothing is not None:
                y_values = savgol(y_values, **dict(smoothing))
            if alignment is not None:
                alignment_kwargs = (
                    {"method": alignment}
                    if isinstance(alignment, str)
                    else dict(alignment)
                )
                aligned = align_curve(x_values, table[y], **alignment_kwargs)
                x_values = aligned.shifted_x
            if derivative:
                y_values = gradient(y_values, x_values)

            ax.plot(
                x_values,
                y_values,
                color=color_map(color_index),
                label=experiment.stem,
                **line_kwargs,
            )
            color_index += 1

    ax.set_xlabel(x)
    ax.set_ylabel(f"d/d{x} {y}" if derivative else y)
    ax.legend()
    return figure, ax


def _coerce_groups(
    groups: Study
    | ExperimentGroup
    | Sequence[ExperimentGroup]
    | Sequence[Experiment],
) -> list[ExperimentGroup]:
    """Normalize supported group inputs into experiment groups.

    Args:
        groups: Supported grouping input accepted by `plot_groups`.

    Returns:
        Concrete experiment groups ready for plotting.
    """

    if isinstance(groups, Study):
        return groups.group_by_time_gap()
    if isinstance(groups, ExperimentGroup):
        return [groups]
    groups_list = list(groups)
    if not groups_list:
        return []
    first_item = groups_list[0]
    if isinstance(first_item, ExperimentGroup):
        return list(groups_list)
    if isinstance(first_item, Experiment):
        return [ExperimentGroup(experiments=tuple(groups_list), index=0)]
    raise TypeError(
        "plot_groups expects a Study, ExperimentGroup,"
        "or a sequence of those objects."
    )
