"""Public package exports for nanodent."""

from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanodent.analysis.filters import savgol
from nanodent.analysis.oliver_pharr import (
    OliverPharrBatchResult,
    OliverPharrExperimentResult,
    analyze_oliver_pharr,
)
from nanodent.analysis.quality import (
    QualityCheckResult,
    classify_flat_force,
    classify_gradual_onset,
    classify_high_displacement,
    classify_outlier_jumps,
    classify_peak_balance,
    classify_quality,
)
from nanodent.io import load_experiment, load_folder
from nanodent.models import (
    Experiment,
    ExperimentPaths,
    MetadataEntry,
    SegmentDefinition,
    SignalTable,
)
from nanodent.study import ExperimentGroup, Study

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from nanodent.plotting import AxesGrid

try:
    __version__ = metadata.version("nanodent")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Experiment",
    "ExperimentGroup",
    "ExperimentPaths",
    "OliverPharrBatchResult",
    "OliverPharrExperimentResult",
    "QualityCheckResult",
    "MetadataEntry",
    "SegmentDefinition",
    "SignalTable",
    "Study",
    "__version__",
    "analyze_oliver_pharr",
    "classify_gradual_onset",
    "classify_high_displacement",
    "classify_flat_force",
    "classify_peak_balance",
    "classify_outlier_jumps",
    "classify_quality",
    "load_experiment",
    "load_folder",
    "plot_force_displacement",
    "plot_group_timeline",
    "plot_groups",
    "savgol",
    "save_experiment_plots",
]


def plot_groups(*args: Any, **kwargs: Any) -> tuple["Figure", "AxesGrid"]:
    """Lazily import Matplotlib plotting helpers when needed.

    Args:
        *args: Positional args forwarded to `nanodent.plotting.plot_groups`.
        **kwargs: Keyword args forwarded to `nanodent.plotting.plot_groups`.

    Returns:
        Figure and axes containing the plotted curves.
    """

    from nanodent.plotting import plot_groups as _plot_groups

    return _plot_groups(*args, **kwargs)


def plot_group_timeline(*args: Any, **kwargs: Any) -> tuple["Figure", "Axes"]:
    """Lazily import the group timeline plotting helper when needed.

    Args:
        *args: Positional args forwarded to
            `nanodent.plotting.plot_group_timeline`.
        **kwargs: Keyword args forwarded to
            `nanodent.plotting.plot_group_timeline`.

    Returns:
        Figure and axes containing the group timeline visualization.
    """

    from nanodent.plotting import plot_group_timeline as _plot_group_timeline

    return _plot_group_timeline(*args, **kwargs)


def plot_force_displacement(*args: Any, **kwargs: Any) -> "Axes":
    """Lazily import the single-axes force-displacement plotting helper.

    Args:
        *args: Positional args forwarded to
            `nanodent.plotting.plot_force_displacement`.
        **kwargs: Keyword args forwarded to
            `nanodent.plotting.plot_force_displacement`.

    Returns:
        Axes containing the plotted force-displacement curves.
    """

    from nanodent.plotting import (
        plot_force_displacement as _plot_force_displacement,
    )

    return _plot_force_displacement(*args, **kwargs)


def save_experiment_plots(*args: Any, **kwargs: Any) -> list["Path"]:
    """Lazily import and save one plot file per experiment.

    Args:
        *args: Positional args forwarded to
            `nanodent.plotting.save_experiment_plots`.
        **kwargs: Keyword args forwarded to
            `nanodent.plotting.save_experiment_plots`.

    Returns:
        Paths to the saved plot files.
    """

    from nanodent.plotting import save_experiment_plots as _save_plots

    return _save_plots(*args, **kwargs)
