"""Signal-processing and fitting helpers."""

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

__all__ = [
    "ForcePeakDetectionResult",
    "ForcePeakPosition",
    "HertzianExperimentResult",
    "OnsetDetectionResult",
    "OliverPharrExperimentResult",
    "QualityCheckResult",
    "UnloadingDetectionResult",
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
    "savgol",
]
