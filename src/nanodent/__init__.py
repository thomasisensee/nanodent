"""Public package exports for nanodent."""

from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanodent.analysis.filters import savgol
from nanodent.analysis.force_peaks import (
    ForcePeakDetectionResult,
    ForcePeakPosition,
    detect_force_peaks,
)
from nanodent.analysis.hertzian import (
    HertzianExperimentResult,
    analyze_hertzian,
    calculate_hertzian_radius,
    calculate_tau_max,
)
from nanodent.analysis.oliver_pharr import (
    OliverPharrExperimentResult,
    analyze_oliver_pharr,
)
from nanodent.analysis.onset import OnsetDetectionResult, detect_onset
from nanodent.analysis.quality import (
    QualityCheckResult,
    classify_flat_force,
    classify_gradual_onset,
    classify_high_displacement,
    classify_outlier_jumps,
    classify_peak_balance,
    classify_quality,
)
from nanodent.analysis.unloading import (
    UnloadingDetectionResult,
    detect_unloading,
)
from nanodent.io import load_experiment, load_folder
from nanodent.models import (
    Experiment,
    ExperimentPaths,
    MetadataEntry,
    SegmentDefinition,
    SignalTable,
    TipAreaFunction,
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
    "Experiment",
    "ExperimentGroup",
    "ExperimentPaths",
    "ForcePeakDetectionResult",
    "ForcePeakPosition",
    "HertzianExperimentResult",
    "OnsetDetectionResult",
    "OliverPharrExperimentResult",
    "QualityCheckResult",
    "UnloadingDetectionResult",
    "MetadataEntry",
    "SegmentDefinition",
    "SignalTable",
    "TipAreaFunction",
    "Study",
    "__version__",
    "detect_force_peaks",
    "detect_onset",
    "detect_unloading",
    "analyze_hertzian",
    "analyze_oliver_pharr",
    "calculate_hertzian_radius",
    "calculate_tau_max",
    "classify_gradual_onset",
    "classify_high_displacement",
    "classify_flat_force",
    "classify_peak_balance",
    "classify_outlier_jumps",
    "classify_quality",
    "load_experiment",
    "load_folder",
    "plot_experiments",
    "plot_group_timeline",
    "savgol",
    "save_experiment_plots",
]


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


def plot_experiments(*args: Any, **kwargs: Any) -> "Axes":
    """Lazily import the experiment force-displacement plotting helper.

    Args:
        *args: Positional args forwarded to
            `nanodent.plotting.plot_experiments`.
        **kwargs: Keyword args forwarded to
            `nanodent.plotting.plot_experiments`.

    Returns:
        Axes containing the plotted experiment force-displacement curves.
    """

    from nanodent.plotting import plot_experiments as _plot_experiments

    return _plot_experiments(*args, **kwargs)


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
