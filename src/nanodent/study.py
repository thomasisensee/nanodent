"""Study-level containers and grouping utilities."""

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from nanodent.analysis.force_peaks import (
    detect_force_peaks as _detect_force_peaks,
)
from nanodent.analysis.oliver_pharr import (
    analyze_oliver_pharr as _analyze_oliver_pharr,
)
from nanodent.analysis.onset import detect_onset as _detect_onset
from nanodent.analysis.quality import classify_quality as _classify_quality
from nanodent.models import Experiment


@dataclass(frozen=True, slots=True)
class ExperimentGroup:
    """A deterministic group of temporally related experiments."""

    stems: tuple[str, ...]
    index: int = 0

    def __post_init__(self) -> None:
        """Validate that the group contains at least one experiment stem."""

        if not self.stems:
            raise ValueError(
                "ExperimentGroup requires at least one experiment."
            )

    def resolve(
        self,
        study: "Study",
        *,
        include_disabled: bool = False,
    ) -> tuple[Experiment, ...]:
        """Resolve the group's stems against the current study."""

        return study.resolve_group(self, include_disabled=include_disabled)

    def summary(
        self,
        study: "Study",
        *,
        include_disabled: bool = True,
    ) -> dict[str, Any]:
        """Return a notebook-friendly summary of the group."""

        experiments = self.resolve(study, include_disabled=include_disabled)
        if not experiments:
            return {
                "index": self.index,
                "experiment_count": 0,
                "enabled_count": 0,
                "disabled_count": 0,
                "start": None,
                "end": None,
                "duration": timedelta(0),
                "stems": (),
            }

        return {
            "index": self.index,
            "experiment_count": len(experiments),
            "enabled_count": sum(
                experiment.enabled for experiment in experiments
            ),
            "disabled_count": sum(
                not experiment.enabled for experiment in experiments
            ),
            "start": experiments[0].timestamp,
            "end": experiments[-1].timestamp,
            "duration": experiments[-1].timestamp - experiments[0].timestamp,
            "stems": tuple(
                f"{experiment.stem} "
                f"({'enabled' if experiment.enabled else 'disabled'})"
                for experiment in experiments
            ),
        }


@dataclass(frozen=True, slots=True)
class Study:
    """A collection of experiments sorted by acquisition timestamp."""

    experiments: tuple[Experiment, ...]

    def __post_init__(self) -> None:
        """Store experiments in ascending timestamp order."""

        sorted_experiments = tuple(
            sorted(
                self.experiments,
                key=lambda experiment: experiment.timestamp,
            )
        )
        duplicate_stems = _duplicate_stems(sorted_experiments)
        if duplicate_stems:
            raise ValueError(
                "Study requires unique experiment stems. "
                f"Duplicate stems: {', '.join(duplicate_stems)}."
            )
        object.__setattr__(self, "experiments", sorted_experiments)

    def __len__(self) -> int:
        """Return the number of experiments in the study."""

        return len(self.experiments)

    def __iter__(self) -> Iterator[Experiment]:
        """Iterate through experiments in timestamp order."""

        return iter(self.experiments)

    def group_by_time_gap(
        self,
        max_gap: timedelta = timedelta(minutes=30),
        *,
        include_disabled: bool = False,
    ) -> list[ExperimentGroup]:
        """Group experiments by the gap between consecutive timestamps."""

        experiments = self.get_experiments(include_disabled=include_disabled)
        if not experiments:
            return []

        groups: list[list[Experiment]] = [[experiments[0]]]
        for experiment in experiments[1:]:
            gap = experiment.timestamp - groups[-1][-1].timestamp
            if gap > max_gap:
                groups.append([experiment])
                continue
            groups[-1].append(experiment)
        return [
            ExperimentGroup(
                stems=tuple(item.stem for item in group),
                index=index,
            )
            for index, group in enumerate(groups)
        ]

    def regroup(
        self,
        groups: Iterable[Sequence[Experiment | str]],
        *,
        include_disabled: bool = False,
    ) -> list[ExperimentGroup]:
        """Create explicit groups from Python-side experiment selections."""

        regrouped: list[ExperimentGroup] = []
        for group in groups:
            stems = tuple(_coerce_experiment_stem(item) for item in group)
            selected_group = self.get_experiments(
                stems=stems,
                include_disabled=include_disabled,
            )
            if not selected_group:
                continue
            sorted_group = tuple(
                sorted(
                    selected_group,
                    key=lambda experiment: experiment.timestamp,
                )
            )
            regrouped.append(
                ExperimentGroup(
                    stems=tuple(
                        experiment.stem for experiment in sorted_group
                    ),
                    index=len(regrouped),
                )
            )
        return regrouped

    def group_by_datetime_ranges(
        self,
        ranges: Iterable[tuple[datetime, datetime]],
        *,
        include_disabled: bool = False,
    ) -> list[ExperimentGroup]:
        """Create groups from explicit inclusive datetime windows."""

        validated_ranges = self._validate_datetime_ranges(ranges)
        experiments = self.get_experiments(include_disabled=include_disabled)
        grouped: list[ExperimentGroup] = []
        for start, end in validated_ranges:
            selected_group = tuple(
                experiment
                for experiment in experiments
                if start <= experiment.timestamp <= end
            )
            if not selected_group:
                continue
            grouped.append(
                ExperimentGroup(
                    stems=tuple(
                        experiment.stem for experiment in selected_group
                    ),
                    index=len(grouped),
                )
            )
        return grouped

    def describe_groups(
        self,
        max_gap: timedelta = timedelta(minutes=30),
        *,
        include_disabled: bool = True,
    ) -> list[dict[str, Any]]:
        """Summarize time-gap groups for interactive inspection."""

        return [
            group.summary(self, include_disabled=include_disabled)
            for group in self.group_by_time_gap(
                max_gap=max_gap,
                include_disabled=include_disabled,
            )
        ]

    def classify_quality(
        self,
        *,
        min_robust_force_span_uN: float = 200.0,
        low_quantile: float = 0.40,
        high_quantile: float = 0.999,
        max_disp_nm: float = 1000.0,
        peak_bin_count: int = 48,
        peak_prominence_fraction: float = 0.05,
        min_secondary_peak_fraction: float = 0.1,
        require_two_peaks: bool = False,
        disp_z_threshold: float = 100.0,
        force_z_threshold: float = 70.0,
        bin_count: int = 24,
        baseline_bin_count: int = 4,
        onset_force_fraction: float = 0.05,
        target_force_fraction: float = 0.5,
        sustained_bins: int = 2,
        max_rise_width_fraction: float = 0.2,
    ) -> "Study":
        """Return a study with experiments flagged by quality heuristics."""

        classified: list[Experiment] = []
        for experiment in self.experiments:
            test_section = experiment.section("test")
            result = _classify_quality(
                test_section["disp_nm"],
                test_section["force_uN"],
                min_robust_force_span_uN=min_robust_force_span_uN,
                low_quantile=low_quantile,
                high_quantile=high_quantile,
                max_disp_nm=max_disp_nm,
                peak_bin_count=peak_bin_count,
                peak_prominence_fraction=peak_prominence_fraction,
                min_secondary_peak_fraction=min_secondary_peak_fraction,
                require_two_peaks=require_two_peaks,
                disp_z_threshold=disp_z_threshold,
                force_z_threshold=force_z_threshold,
                bin_count=bin_count,
                baseline_bin_count=baseline_bin_count,
                onset_force_fraction=onset_force_fraction,
                target_force_fraction=target_force_fraction,
                sustained_bins=sustained_bins,
                max_rise_width_fraction=max_rise_width_fraction,
            )
            classified.append(
                experiment.with_enabled(result.enabled, reason=result.reason)
            )
        return Study(experiments=tuple(classified))

    def analyze_oliver_pharr(
        self,
        *,
        stems: Iterable[str] | str | None = None,
        unloading_fraction: float = 0.2,
        smoothing: Mapping[str, Any] | None = None,
        fit_num_points: int = 2,
        use_force_peak: bool = True,
        epsilon: float = 0.75,
        include_disabled: bool = False,
        overwrite: bool = False,
    ) -> "Study":
        """Analyze selected experiments with a straight-line unloading fit."""

        selected_stems = self._selected_stem_set(
            stems=stems,
            include_disabled=include_disabled,
        )
        skipped: list[str] = []
        experiments: list[Experiment] = []
        for experiment in self.experiments:
            if experiment.stem not in selected_stems:
                experiments.append(experiment)
                continue
            if not overwrite and _has_successful_result(
                experiment.oliver_pharr
            ):
                skipped.append(experiment.stem)
                experiments.append(experiment)
                continue
            experiments.append(
                experiment.with_oliver_pharr(
                    _analyze_oliver_pharr(
                        experiment.section("test")["disp_nm"],
                        experiment.section("test")["force_uN"],
                        unloading_fraction=unloading_fraction,
                        smoothing=smoothing,
                        fit_num_points=fit_num_points,
                        use_force_peak=use_force_peak,
                        onset_disp_nm=(
                            experiment.onset.onset_disp_nm
                            if experiment.onset is not None
                            and experiment.onset.success
                            else None
                        ),
                        epsilon=epsilon,
                        stem=experiment.stem,
                    )
                )
            )
        self._warn_skipped_results(
            analysis_name="Oliver-Pharr analysis",
            skipped_stems=skipped,
        )
        return Study(experiments=tuple(experiments))

    def detect_onset(
        self,
        *,
        stems: Iterable[str] | str | None = None,
        baseline_points: int = 100,
        k: float = 4.0,
        consecutive: int = 5,
        smoothing: Mapping[str, Any] | None = None,
        include_disabled: bool = False,
        overwrite: bool = False,
    ) -> "Study":
        """Detect onset on selected experiments using the test force signal."""

        selected_stems = self._selected_stem_set(
            stems=stems,
            include_disabled=include_disabled,
        )
        skipped: list[str] = []
        invalidated_oliver_pharr: list[str] = []
        experiments: list[Experiment] = []
        for experiment in self.experiments:
            if experiment.stem not in selected_stems:
                experiments.append(experiment)
                continue
            if not overwrite and _has_successful_result(experiment.onset):
                skipped.append(experiment.stem)
                experiments.append(experiment)
                continue

            updated = experiment.with_onset(
                _detect_onset(
                    experiment.section("test")["force_uN"],
                    time_s=experiment.section("test")["time_s"],
                    disp_nm=experiment.section("test")["disp_nm"],
                    baseline_points=baseline_points,
                    k=k,
                    consecutive=consecutive,
                    smoothing=smoothing,
                )
            )
            if experiment.oliver_pharr is not None:
                updated = updated.with_oliver_pharr(None)
                invalidated_oliver_pharr.append(experiment.stem)
            experiments.append(updated)
        self._warn_skipped_results(
            analysis_name="onset detection",
            skipped_stems=skipped,
        )
        self._warn_invalidated_results(
            dependency_name="Oliver-Pharr",
            reason="onset results were recomputed",
            stems=invalidated_oliver_pharr,
        )
        return Study(experiments=tuple(experiments))

    def detect_force_peaks(
        self,
        *,
        stems: Iterable[str] | str | None = None,
        prominence: float = 100.0,
        threshold: float | None = 1.0,
        include_disabled: bool = False,
        overwrite: bool = False,
    ) -> "Study":
        """Detect raw-force peaks on selected experiments."""

        selected_stems = self._selected_stem_set(
            stems=stems,
            include_disabled=include_disabled,
        )
        skipped: list[str] = []
        experiments: list[Experiment] = []
        for experiment in self.experiments:
            if experiment.stem not in selected_stems:
                experiments.append(experiment)
                continue
            if not overwrite and _has_successful_result(
                experiment.force_peaks
            ):
                skipped.append(experiment.stem)
                experiments.append(experiment)
                continue
            experiments.append(
                experiment.with_force_peaks(
                    _detect_force_peaks(
                        experiment.section("test")["force_uN"],
                        time_s=experiment.section("test")["time_s"],
                        disp_nm=experiment.section("test")["disp_nm"],
                        prominence=prominence,
                        threshold=threshold,
                    )
                )
            )
        self._warn_skipped_results(
            analysis_name="force-peak detection",
            skipped_stems=skipped,
        )
        return Study(experiments=tuple(experiments))

    def set_enabled(
        self,
        stems: Iterable[str] | str,
        *,
        enabled: bool,
        reason: str | None = None,
    ) -> "Study":
        """Return a study with selected stems manually enabled or disabled."""

        requested_stems = set(self._normalize_requested_stems(stems))
        updated = tuple(
            experiment.with_enabled(enabled, reason=reason)
            if experiment.stem in requested_stems
            else experiment
            for experiment in self.experiments
        )
        return Study(experiments=updated)

    def enable_experiments(self, stems: Iterable[str] | str) -> "Study":
        """Return a study with selected experiment stems enabled."""

        return self.set_enabled(stems, enabled=True)

    def disable_experiments(
        self, stems: Iterable[str] | str, *, reason: str = "manual"
    ) -> "Study":
        """Return a study with selected experiment stems disabled."""

        return self.set_enabled(stems, enabled=False, reason=reason)

    def get_experiments(
        self,
        *,
        stems: Iterable[str] | str | None = None,
        include_disabled: bool = False,
    ) -> tuple[Experiment, ...]:
        """Return study experiments, optionally filtered by stems."""

        if stems is None:
            experiments = self.experiments
        else:
            requested_stems = self._normalize_requested_stems(stems)
            experiments_by_stem = self._experiments_by_stem()
            unknown_stems = [
                stem
                for stem in requested_stems
                if stem not in experiments_by_stem
            ]
            if unknown_stems:
                raise ValueError(
                    "Unknown experiment stems requested: "
                    f"{', '.join(unknown_stems)}."
                )
            experiments = tuple(
                experiments_by_stem[stem] for stem in requested_stems
            )

        if include_disabled:
            return experiments
        return tuple(
            experiment for experiment in experiments if experiment.enabled
        )

    def resolve_group(
        self,
        group: ExperimentGroup,
        *,
        include_disabled: bool = False,
    ) -> tuple[Experiment, ...]:
        """Resolve a stem-based group against this study."""

        return self.get_experiments(
            stems=group.stems,
            include_disabled=include_disabled,
        )

    def experiment_table(
        self,
        *,
        stems: Iterable[str] | str | None = None,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        """Return one scalar summary row per selected experiment."""

        return [
            _experiment_table_row(experiment)
            for experiment in self.get_experiments(
                stems=stems,
                include_disabled=include_disabled,
            )
        ]

    def _selected_stem_set(
        self,
        *,
        stems: Iterable[str] | str | None,
        include_disabled: bool,
    ) -> set[str]:
        """Return the selected experiment stems for an analysis call."""

        return {
            experiment.stem
            for experiment in self.get_experiments(
                stems=stems,
                include_disabled=include_disabled,
            )
        }

    def _warn_skipped_results(
        self,
        *,
        analysis_name: str,
        skipped_stems: Sequence[str],
    ) -> None:
        """Warn once when successful existing results are preserved."""

        if not skipped_stems:
            return
        warnings.warn(
            f"Skipped {len(skipped_stems)} experiment(s) during "
            f"{analysis_name} because successful results already exist: "
            f"{', '.join(skipped_stems)}. Pass overwrite=True to recompute.",
            stacklevel=2,
        )

    def _warn_invalidated_results(
        self,
        *,
        dependency_name: str,
        reason: str,
        stems: Sequence[str],
    ) -> None:
        """Warn once when dependent results are cleared automatically."""

        if not stems:
            return
        warnings.warn(
            f"Cleared {dependency_name} results for {len(stems)} "
            f"experiment(s) because {reason}: {', '.join(stems)}.",
            stacklevel=2,
        )

    def _experiments_by_stem(self) -> dict[str, Experiment]:
        """Return study experiments keyed by stem."""

        return {experiment.stem: experiment for experiment in self.experiments}

    def _normalize_requested_stems(
        self,
        stems: Iterable[str] | str,
    ) -> tuple[str, ...]:
        """Return distinct requested stems preserving user order."""

        values = (stems,) if isinstance(stems, str) else tuple(stems)
        ordered_stems: list[str] = []
        seen: set[str] = set()
        for stem in values:
            if stem in seen:
                continue
            ordered_stems.append(stem)
            seen.add(stem)
        return tuple(ordered_stems)

    def _validate_datetime_ranges(
        self, ranges: Iterable[tuple[datetime, datetime]]
    ) -> tuple[tuple[datetime, datetime], ...]:
        """Return validated datetime ranges preserving input order."""

        validated_ranges = tuple(ranges)
        for start, end in validated_ranges:
            if start > end:
                raise ValueError("Datetime ranges require start <= end.")

        sorted_ranges = sorted(validated_ranges, key=lambda item: item[0])
        for previous, current in zip(sorted_ranges, sorted_ranges[1:]):
            if current[0] <= previous[1]:
                raise ValueError("Datetime ranges must not overlap.")
        return validated_ranges


def _coerce_experiment_stem(experiment_or_stem: Experiment | str) -> str:
    """Return a stem from either an Experiment instance or a string."""

    if isinstance(experiment_or_stem, Experiment):
        return experiment_or_stem.stem
    return experiment_or_stem


def _duplicate_stems(experiments: Sequence[Experiment]) -> tuple[str, ...]:
    """Return duplicate experiment stems in first-seen order."""

    duplicates: list[str] = []
    seen: set[str] = set()
    duplicate_set: set[str] = set()
    for experiment in experiments:
        if experiment.stem not in seen:
            seen.add(experiment.stem)
            continue
        if experiment.stem in duplicate_set:
            continue
        duplicates.append(experiment.stem)
        duplicate_set.add(experiment.stem)
    return tuple(duplicates)


def _has_successful_result(result: Any) -> bool:
    """Return whether an attached analysis result is successful."""

    return result is not None and bool(getattr(result, "success", False))


def _experiment_table_row(experiment: Experiment) -> dict[str, Any]:
    """Return one notebook-friendly scalar row for an experiment."""

    onset = experiment.onset
    force_peaks = experiment.force_peaks
    oliver_pharr = experiment.oliver_pharr
    return {
        "stem": experiment.stem,
        "timestamp": experiment.timestamp,
        "enabled": experiment.enabled,
        "disabled_reason": experiment.disabled_reason,
        "temperature_c": experiment.temperature_c,
        "humidity_percent": experiment.humidity_percent,
        "onset_success": None if onset is None else onset.success,
        "onset_reason": None if onset is None else onset.reason,
        "onset_index": None if onset is None else onset.onset_index,
        "onset_time_s": None if onset is None else onset.onset_time_s,
        "onset_disp_nm": None if onset is None else onset.onset_disp_nm,
        "force_peaks_success": None
        if force_peaks is None
        else force_peaks.success,
        "force_peaks_reason": None
        if force_peaks is None
        else force_peaks.reason,
        "force_peak_count": None
        if force_peaks is None
        else force_peaks.peak_count,
        "oliver_pharr_success": None
        if oliver_pharr is None
        else oliver_pharr.success,
        "oliver_pharr_reason": None
        if oliver_pharr is None
        else oliver_pharr.reason,
        "stiffness_uN_per_nm": None
        if oliver_pharr is None
        else oliver_pharr.stiffness_uN_per_nm,
        "hardness_success": None
        if oliver_pharr is None
        else oliver_pharr.hardness_success,
        "hardness_reason": None
        if oliver_pharr is None
        else oliver_pharr.hardness_reason,
        "hardness_uN_per_nm2": None
        if oliver_pharr is None
        else oliver_pharr.hardness_uN_per_nm2,
        "reduced_modulus_uN_per_nm2": None
        if oliver_pharr is None
        else oliver_pharr.reduced_modulus_uN_per_nm2,
    }
