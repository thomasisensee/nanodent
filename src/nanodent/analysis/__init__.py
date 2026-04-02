"""Signal-processing and fitting helpers."""

from nanodent.analysis.filters import savgol
from nanodent.analysis.force_peaks import (
    ForcePeakDetectionResult,
    ForcePeakPosition,
    detect_force_peaks,
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

__all__ = [
    "ForcePeakDetectionResult",
    "ForcePeakPosition",
    "OnsetDetectionResult",
    "OliverPharrExperimentResult",
    "QualityCheckResult",
    "detect_force_peaks",
    "detect_onset",
    "analyze_oliver_pharr",
    "classify_gradual_onset",
    "classify_high_displacement",
    "classify_flat_force",
    "classify_peak_balance",
    "classify_outlier_jumps",
    "classify_quality",
    "savgol",
]
