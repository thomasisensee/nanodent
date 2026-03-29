"""Signal-processing and fitting helpers."""

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

__all__ = [
    "OliverPharrBatchResult",
    "OliverPharrExperimentResult",
    "QualityCheckResult",
    "analyze_oliver_pharr",
    "classify_gradual_onset",
    "classify_high_displacement",
    "classify_flat_force",
    "classify_peak_balance",
    "classify_outlier_jumps",
    "classify_quality",
    "savgol",
]
