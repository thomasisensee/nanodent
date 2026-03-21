"""Signal-processing and fitting helpers."""

from nanodent.analysis.align import AlignmentResult, align_curve
from nanodent.analysis.derivative import gradient
from nanodent.analysis.filters import savgol
from nanodent.analysis.fit import FitResult, curve_fit_model
from nanodent.analysis.quality import (
    QualityCheckResult,
    classify_delayed_onset,
    classify_flat_force,
    classify_quality,
)

__all__ = [
    "AlignmentResult",
    "FitResult",
    "QualityCheckResult",
    "align_curve",
    "classify_delayed_onset",
    "classify_flat_force",
    "classify_quality",
    "curve_fit_model",
    "gradient",
    "savgol",
]
