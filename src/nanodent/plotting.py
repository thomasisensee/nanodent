"""Matplotlib plotting helpers for experiment data."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    study: Study | None = None,
    max_gap: timedelta = timedelta(minutes=30),
    include_disabled: bool = False,
    cmap: str = "tab10",
    marker: str = "o",
    markersize: float = 6.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a static timeline view of automatically grouped experiments."""

    resolved_groups = _coerce_timeline_groups(
        groups,
        study=study,
        max_gap=max_gap,
        include_disabled=include_disabled,
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
        start_num = mdates.date2num(group.experiments[0].timestamp)
        end_num = mdates.date2num(group.experiments[-1].timestamp)
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
    study: Study | None = None,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    smoothing: Mapping[str, Any] | None = None,
    cmap: str = "viridis",
    max_gap: timedelta = timedelta(minutes=30),
    selection: Literal["enabled", "disabled", "both"] = "enabled",
    zero_onset: bool = False,
    show_unloading: bool = False,
    show_oliver_pharr: bool = True,
    unloading_kwargs: Mapping[str, Any] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    **line_kwargs: Any,
) -> Axes:
    """Plot one or more experiment curves onto an existing axes."""

    experiments = _coerce_experiments(
        target,
        study=study,
        max_gap=max_gap,
        selection=selection,
    )
    use_force_displacement = (
        section == "test" and x == "disp_nm" and y == "force_uN"
    )

    color_map = plt.get_cmap(cmap, max(len(experiments), 1))
    for index, experiment in enumerate(experiments):
        onset_offset = _onset_axis_offset(
            experiment,
            section=section,
            axis=x,
            zero_onset=zero_onset,
        )
        curve = _prepare_curve(
            experiment,
            section=section,
            x=x,
            y=y,
            smoothing=smoothing,
            onset_offset=onset_offset,
        )
        curve_kwargs = dict(line_kwargs)
        curve_kwargs.setdefault("color", color_map(index))
        ax.plot(
            curve.x_values,
            curve.y_values,
            label=experiment.stem,
            **curve_kwargs,
        )
        _plot_unloading_overlay(
            ax,
            experiment=experiment,
            curve=curve,
            show_unloading=show_unloading and use_force_displacement,
            curve_kwargs=curve_kwargs,
            unloading_kwargs=unloading_kwargs,
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
            onset_offset=onset_offset if x == "disp_nm" else None,
        )

    return ax


def save_experiment_plots(
    groups: PlotSelection,
    output_dir: str | Path,
    *,
    study: Study | None = None,
    section: str = "test",
    x: str = "disp_nm",
    y: str = "force_uN",
    smoothing: Mapping[str, Any] | None = None,
    max_gap: timedelta = timedelta(minutes=30),
    selection: Literal["enabled", "disabled", "both"] = "enabled",
    zero_onset: bool = False,
    show_oliver_pharr: bool = True,
    unloading_kwargs: Mapping[str, Any] | None = None,
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
        groups,
        study=study,
        max_gap=max_gap,
        selection=selection,
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
            study=study,
            section=section,
            x=x,
            y=y,
            smoothing=smoothing,
            max_gap=max_gap,
            selection="both",
            zero_onset=zero_onset,
            show_unloading=True,
            show_oliver_pharr=show_oliver_pharr,
            unloading_kwargs=unloading_kwargs,
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
            zero_onset=zero_onset,
        )
        ax.grid(alpha=0.2)
        figure.tight_layout()

        output_path = destination / _experiment_output_name(
            experiment,
            suffix=suffix,
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
    zero_onset: bool = False,
) -> None:
    """Apply saved-plot-only annotations for one experiment axes."""

    ax.set_title(_format_saved_experiment_title(experiment))
    _add_saved_plot_analysis_box(ax, experiment=experiment)
    if not _uses_force_displacement_axes(section=section, x=x, y=y):
        return
    events = _saved_plot_annotation_events(experiment, zero_onset=zero_onset)
    _add_saved_plot_top_axis(ax, events=events)
    _add_saved_plot_right_axis(ax, events=events)


def _plot_oliver_pharr_overlay(
    ax: Axes,
    *,
    stem: str,
    fit_result: Any,
    fit_kwargs: Mapping[str, Any] | None,
    onset_offset: float | None,
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
    x_fit = np.asarray(fit_result.x_fit, dtype=np.float64)
    if onset_offset is not None:
        x_fit = x_fit - onset_offset
    ax.plot(x_fit, fit_result.y_fit, **overlay_kwargs)

    extension_segment = _oliver_pharr_extension_segment(
        fit_result, onset_offset=onset_offset
    )
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


def _plot_unloading_overlay(
    ax: Axes,
    *,
    experiment: Experiment,
    curve: "_PreparedCurve",
    show_unloading: bool,
    curve_kwargs: Mapping[str, Any],
    unloading_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Overlay the detected unloading branch for saved-plot inspection."""

    if not show_unloading:
        return
    unloading = experiment.unloading
    if unloading is None or not unloading.success:
        return
    if unloading.start_index is None:
        return

    start_index = int(unloading.start_index)
    if start_index < 0 or start_index >= len(curve.x_values):
        return

    overlay_kwargs = {
        "color": curve_kwargs.get("color", "black"),
        "alpha": 0.75,
        "linewidth": max(
            float(curve_kwargs.get("linewidth", 1.5)) * 1.15,
            1.0,
        ),
        "linestyle": curve_kwargs.get("linestyle", "-"),
        "label": "_nolegend_",
        "zorder": max(float(curve_kwargs.get("zorder", 2.0)) + 0.1, 0.0),
    }
    if unloading_kwargs is not None:
        overlay_kwargs.update(dict(unloading_kwargs))
    ax.plot(
        curve.x_values[start_index:],
        curve.y_values[start_index:],
        **overlay_kwargs,
    )


def _uses_force_displacement_axes(*, section: str, x: str, y: str) -> bool:
    """Return whether the axes represent the test force-displacement view."""

    return section == "test" and x == "disp_nm" and y == "force_uN"


def _format_saved_experiment_title(experiment: Experiment) -> str:
    """Return the saved-plot title."""

    timestamp = experiment.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return f"{experiment.stem}\n{timestamp}"


def _add_saved_plot_analysis_box(ax: Axes, *, experiment: Experiment) -> None:
    """Add a compact in-axes summary of attached scalar analysis results."""

    summary = _saved_plot_analysis_summary(experiment)
    if summary is None:
        return

    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.9,
        },
        zorder=5,
    )


def _saved_plot_analysis_summary(experiment: Experiment) -> str | None:
    """Return the saved-plot scalar summary text when values are available."""

    fit_result = experiment.oliver_pharr
    if fit_result is None or not fit_result.success:
        return None

    lines: list[str] = []
    if fit_result.stiffness_uN_per_nm is not None:
        lines.append(f"S={fit_result.stiffness_uN_per_nm:.2f} uN/nm")
    if fit_result.hardness_uN_per_nm2 is not None:
        lines.append(f"H={fit_result.hardness_uN_per_nm2:.3g} uN/nm^2")
    if fit_result.reduced_modulus_uN_per_nm2 is not None:
        lines.append(f"Er={fit_result.reduced_modulus_uN_per_nm2:.3g} uN/nm^2")
    if not lines:
        return None
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _SavedPlotAnnotationEvent:
    """One saved-plot annotation event with aligned plot coordinates."""

    disp_nm: float
    force_uN: float


def _add_saved_plot_top_axis(
    ax: Axes, *, events: Sequence[_SavedPlotAnnotationEvent]
) -> None:
    """Add a top displacement axis marking onset and maximum force."""

    tick_positions, tick_labels = _saved_plot_top_axis_ticks(events)
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
    events: Sequence[_SavedPlotAnnotationEvent],
) -> tuple[list[float], list[str]]:
    """Return displacement positions and labels for saved-plot top ticks."""

    unique_positions = _unique_sorted_tick_positions(
        [event.disp_nm for event in events]
    )
    return (
        unique_positions,
        [_format_disp_tick(position) for position in unique_positions],
    )


def _add_saved_plot_right_axis(
    ax: Axes, *, events: Sequence[_SavedPlotAnnotationEvent]
) -> None:
    """Add a right force axis aligned to saved-plot annotation events."""

    tick_positions, tick_labels = _saved_plot_right_axis_ticks(events)
    if not tick_positions:
        return

    right_ax = ax.twinx()
    right_ax.set_ylim(ax.get_ylim())
    right_ax.set_yticks(tick_positions)
    right_ax.set_yticklabels(tick_labels)
    right_ax.yaxis.set_ticks_position("right")
    right_ax.yaxis.set_label_position("right")
    right_ax.set_ylabel("")
    right_ax.grid(False)


def _saved_plot_right_axis_ticks(
    events: Sequence[_SavedPlotAnnotationEvent],
) -> tuple[list[float], list[str]]:
    """Return force positions and labels for saved-plot right ticks."""

    unique_positions = _unique_sorted_tick_positions(
        [event.force_uN for event in events]
    )
    return (
        unique_positions,
        [_format_force_tick(position) for position in unique_positions],
    )


def _saved_plot_annotation_events(
    experiment: Experiment,
    *,
    zero_onset: bool,
) -> list[_SavedPlotAnnotationEvent]:
    """Return saved-plot annotation events as displacement-force pairs."""

    force = np.asarray(experiment.trace["force_uN"], dtype=np.float64)
    disp = np.asarray(experiment.trace["disp_nm"], dtype=np.float64)
    if len(force) == 0 or len(disp) == 0:
        return []

    onset_offset = _onset_axis_offset(
        experiment,
        section="test",
        axis="disp_nm",
        zero_onset=zero_onset,
    )

    events: list[_SavedPlotAnnotationEvent] = []
    onset = experiment.onset
    if (
        onset is not None
        and onset.success
        and onset.onset_index is not None
        and onset.onset_disp_nm is not None
    ):
        onset_index = int(onset.onset_index)
        if 0 <= onset_index < len(force):
            events.append(
                _SavedPlotAnnotationEvent(
                    disp_nm=_shift_axis_value(
                        float(onset.onset_disp_nm), onset_offset
                    ),
                    force_uN=float(force[onset_index]),
                )
            )

    force_peaks = experiment.force_peaks
    if force_peaks is not None and force_peaks.success:
        events.extend(
            _SavedPlotAnnotationEvent(
                disp_nm=_shift_axis_value(float(peak.disp_nm), onset_offset),
                force_uN=float(peak.force_uN),
            )
            for peak in force_peaks.peaks
            if peak.disp_nm is not None
        )

    max_index = int(np.argmax(force))
    events.append(
        _SavedPlotAnnotationEvent(
            disp_nm=_shift_axis_value(float(disp[max_index]), onset_offset),
            force_uN=float(force[max_index]),
        )
    )
    return events


def _format_disp_tick(value_nm: float) -> str:
    """Return a compact displacement label for top-axis ticks."""

    return f"{float(value_nm):.3g}"


def _format_force_tick(value_uN: float) -> str:
    """Return a compact force label for right-axis ticks."""

    return f"{float(value_uN):.3g}"


def _unique_sorted_tick_positions(
    positions: list[float], *, atol: float = 1e-12
) -> list[float]:
    """Return sorted tick positions with near-duplicates removed."""

    unique_positions: list[float] = []
    for position in sorted(float(value) for value in positions):
        if any(
            np.isclose(position, existing, atol=atol)
            for existing in unique_positions
        ):
            continue
        unique_positions.append(position)
    return unique_positions


def _oliver_pharr_extension_segment(
    fit_result: Any,
    *,
    onset_offset: float | None,
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
    start_x = _shift_axis_value(start_x, onset_offset)
    end_x = _shift_axis_value(end_x, onset_offset)
    end_y = float(fit_result.y_fit[0])
    return [start_x, end_x], [start_y, end_y]


def _experiment_output_name(
    experiment: Experiment,
    *,
    suffix: str,
) -> str:
    """Return the output filename used for one saved experiment plot."""

    if experiment.source_path is not None:
        return experiment.source_path.with_suffix(suffix).name
    if experiment.paths is not None:
        return experiment.paths.hld_path.with_suffix(suffix).name
    return f"{experiment.stem}{suffix}"


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
    onset_offset: float | None,
) -> _PreparedCurve:
    """Return plotting arrays for one experiment and signal selection."""

    table = experiment.section(section)
    x_values = np.asarray(table[x], dtype=np.float64)
    y_values = np.asarray(table[y], dtype=np.float64)
    if smoothing is not None:
        smoothing_kwargs = dict(smoothing)
        x_values = savgol(x_values, **smoothing_kwargs)
        y_values = savgol(y_values, **smoothing_kwargs)
    if onset_offset is not None:
        x_values = x_values - onset_offset
    return _PreparedCurve(x_values=x_values, y_values=y_values)


def _onset_axis_offset(
    experiment: Experiment,
    *,
    section: str,
    axis: str,
    zero_onset: bool,
) -> float | None:
    """Return the onset-derived offset for a supported x-axis."""

    if not zero_onset or section != "test":
        return None
    onset = experiment.onset
    if onset is None or not onset.success:
        return None
    if axis == "disp_nm":
        return (
            float(onset.onset_disp_nm)
            if onset.onset_disp_nm is not None
            else None
        )
    if axis == "time_s":
        return (
            float(onset.onset_time_s)
            if onset.onset_time_s is not None
            else None
        )
    return None


def _shift_axis_value(value: float, onset_offset: float | None) -> float:
    """Return one axis value shifted by the onset offset when present."""

    if onset_offset is None:
        return float(value)
    return float(value) - onset_offset


def _coerce_experiments(
    target: PlotSelection,
    *,
    study: Study | None,
    max_gap: timedelta,
    selection: Literal["enabled", "disabled", "both"],
) -> list[Experiment]:
    """Normalize supported plotting inputs into experiments by status."""

    if isinstance(target, Study):
        return _select_experiments(target.experiments, selection=selection)
    if isinstance(target, Experiment):
        return _select_experiments([target], selection=selection)
    if isinstance(target, ExperimentGroup):
        resolved_study = _require_group_study(study=study)
        return _select_experiments(
            resolved_study.resolve_group(target, include_disabled=True),
            selection=selection,
        )

    items = list(target)
    if not items:
        return []
    first_item = items[0]
    if isinstance(first_item, ExperimentGroup):
        resolved_study = _require_group_study(study=study)
        return _select_experiments(
            [
                experiment
                for group in items
                for experiment in resolved_study.resolve_group(
                    group, include_disabled=True
                )
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
    study: Study | None,
    max_gap: timedelta,
    include_disabled: bool,
) -> list["_ResolvedExperimentGroup"]:
    """Normalize supported group inputs into timeline-ready groups."""

    if isinstance(groups, Study):
        return [
            _ResolvedExperimentGroup(
                index=group.index,
                experiments=group.resolve(
                    groups,
                    include_disabled=include_disabled,
                ),
            )
            for group in groups.group_by_time_gap(
                max_gap=max_gap,
                include_disabled=include_disabled,
            )
        ]
    return _filtered_groups(
        groups,
        study=_require_group_study(study=study),
        include_disabled=include_disabled,
    )


def _filtered_groups(
    groups: Sequence[ExperimentGroup],
    *,
    study: Study,
    include_disabled: bool,
) -> list["_ResolvedExperimentGroup"]:
    """Return groups with optional filtering of disabled experiments."""

    filtered: list[_ResolvedExperimentGroup] = []
    for group in groups:
        experiments = study.resolve_group(
            group,
            include_disabled=include_disabled,
        )
        if not experiments:
            continue
        filtered.append(
            _ResolvedExperimentGroup(
                index=group.index,
                experiments=experiments,
            )
        )
    return filtered


@dataclass(frozen=True, slots=True)
class _ResolvedExperimentGroup:
    """A plotting-ready group with concrete experiments."""

    experiments: tuple[Experiment, ...]
    index: int = 0


def _require_group_study(*, study: Study | None) -> Study:
    """Return the study used to resolve stem-based groups."""

    if study is None:
        raise TypeError(
            "Stem-based ExperimentGroup plotting requires passing study=..."
        )
    return study
