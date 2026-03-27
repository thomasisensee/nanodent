"""Signal-processing and fitting helpers."""

from nanodent.analysis.align import AlignmentResult, align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.analysis.fit import FitResult, curve_fit_model
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
    "AlignmentResult",
    "FitResult",
    "QualityCheckResult",
    "align_curve",
    "classify_gradual_onset",
    "classify_high_displacement",
    "classify_flat_force",
    "classify_peak_balance",
    "classify_outlier_jumps",
    "classify_quality",
    "curve_fit_model",
    "gradient",
    "savgol",
]
