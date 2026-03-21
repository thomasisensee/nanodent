"""Public package exports for nanodent."""

from importlib import metadata
from typing import TYPE_CHECKING, Any

from nanodent.analysis.align import AlignmentResult, align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.analysis.fit import FitResult, curve_fit_model
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

try:
    __version__ = metadata.version("nanodent")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AlignmentResult",
    "Experiment",
    "ExperimentGroup",
    "ExperimentPaths",
    "FitResult",
    "MetadataEntry",
    "SegmentDefinition",
    "SignalTable",
    "Study",
    "__version__",
    "align_curve",
    "curve_fit_model",
    "gradient",
    "load_experiment",
    "load_folder",
    "plot_groups",
    "savgol",
]


def plot_groups(*args: Any, **kwargs: Any) -> tuple["Figure", "Axes"]:
    """Lazily import Matplotlib plotting helpers when needed.

    Args:
        *args: Positional args forwarded to `nanodent.plotting.plot_groups`.
        **kwargs: Keyword args forwarded to `nanodent.plotting.plot_groups`.

    Returns:
        Figure and axes containing the plotted curves.
    """

    from nanodent.plotting import plot_groups as _plot_groups

    return _plot_groups(*args, **kwargs)
