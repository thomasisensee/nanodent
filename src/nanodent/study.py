"""Study-level containers and grouping utilities."""

import pickle
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

from nanodent.analysis.force_peaks import (
    detect_force_peaks as _detect_force_peaks,
)
from nanodent.analysis.oliver_pharr import analyze_oliver_pharr
from nanodent.analysis.onset import detect_onset as _detect_onset
from nanodent.analysis.quality import classify_quality as _classify_quality
from nanodent.analysis.unloading import (
    detect_unloading as _detect_unloading,
)
from nanodent.models import Experiment

_SESSION_FORMAT_VERSION = 1
_DEFAULT_ENABLED = True
_DEFAULT_DISABLED_REASON = None


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
            test_section = experiment.trace
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
        fit_model: str = "linear_fraction",
        unloading_fraction: float | None = None,
        smoothing: Mapping[str, Any] | None = None,
        fit_num_points: int = 200,
        power_law_hf_mode: str = "fit",
        epsilon: float = 0.75,
        include_disabled: bool = False,
        overwrite: bool = False,
    ) -> "Study":
        """Analyze selected experiments with one Oliver-Pharr fit model."""

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
            updated = experiment
            unloading = updated.unloading
            if not _has_successful_result(unloading):
                updated = updated.with_unloading(
                    _detect_unloading(
                        updated.trace["force_uN"],
                        time_s=updated.trace["time_s"],
                        disp_nm=updated.trace["disp_nm"],
                    )
                )
                unloading = updated.unloading
            experiments.append(
                updated.with_oliver_pharr(
                    analyze_oliver_pharr(
                        *updated.unloading_curve(x="disp_nm", y="force_uN"),
                        unloading_start_trace_index=0
                        if unloading is None
                        else unloading.start_index,
                        fit_model=fit_model,
                        unloading_fraction=unloading_fraction,
                        smoothing=smoothing,
                        fit_num_points=fit_num_points,
                        unloading_end_disp_nm=None
                        if unloading is None
                        else unloading.end_disp_nm,
                        power_law_hf_mode=power_law_hf_mode,
                        onset_disp_nm=(
                            None
                            if updated.onset is None
                            else updated.onset.onset_disp_nm
                        ),
                        baseline_offset_uN=(
                            None
                            if updated.onset is None
                            else updated.onset.baseline_offset_uN
                        ),
                        epsilon=epsilon,
                        stem=updated.stem,
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
        mode: str = "relative",
        baseline_points: int = 100,
        baseline_start_index: int | None = None,
        baseline_end_index: int | None = None,
        k: float = 4.0,
        absolute_threshold_uN: float | None = None,
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
                    experiment.trace["force_uN"],
                    time_s=experiment.trace["time_s"],
                    disp_nm=experiment.trace["disp_nm"],
                    mode=mode,
                    baseline_points=baseline_points,
                    baseline_start_index=baseline_start_index,
                    baseline_end_index=baseline_end_index,
                    k=k,
                    absolute_threshold_uN=absolute_threshold_uN,
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
                        experiment.trace["force_uN"],
                        time_s=experiment.trace["time_s"],
                        disp_nm=experiment.trace["disp_nm"],
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

    def detect_unloading(
        self,
        *,
        stems: Iterable[str] | str | None = None,
        include_disabled: bool = False,
        overwrite: bool = False,
    ) -> "Study":
        """Detect unloading-start indices on selected experiments."""

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
            if not overwrite and _has_successful_result(experiment.unloading):
                skipped.append(experiment.stem)
                experiments.append(experiment)
                continue

            updated = experiment.with_unloading(
                _detect_unloading(
                    experiment.trace["force_uN"],
                    time_s=experiment.trace["time_s"],
                    disp_nm=experiment.trace["disp_nm"],
                )
            )
            if experiment.oliver_pharr is not None:
                updated = updated.with_oliver_pharr(None)
                invalidated_oliver_pharr.append(experiment.stem)
            experiments.append(updated)
        self._warn_skipped_results(
            analysis_name="unloading detection",
            skipped_stems=skipped,
        )
        self._warn_invalidated_results(
            dependency_name="Oliver-Pharr",
            reason="unloading results were recomputed",
            stems=invalidated_oliver_pharr,
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

    def scalar_series(
        self,
        metric: str,
        *,
        stems: Iterable[str] | str | None = None,
        include_disabled: bool = False,
        drop_missing: bool = True,
    ) -> list[dict[str, Any]]:
        """Return timestamped scalar rows for one supported metric."""

        getter = _scalar_metric_getter(metric)
        rows: list[dict[str, Any]] = []
        for experiment in self.get_experiments(
            stems=stems,
            include_disabled=include_disabled,
        ):
            value = getter(experiment)
            if value is None and drop_missing:
                continue
            rows.append(
                {
                    "timestamp": experiment.timestamp,
                    "stem": experiment.stem,
                    "value": value,
                }
            )
        return rows

    def save_session(self, path: str | Path) -> Path:
        """Persist experiment flags and attached analysis results."""

        destination = Path(path)
        payload = {
            "format_version": _SESSION_FORMAT_VERSION,
            "package_version": _package_version(),
            "saved_at": datetime.now(),
            "experiment_count": len(self.experiments),
            "experiments": {
                experiment.stem: _session_entry(experiment)
                for experiment in self.experiments
            },
        }
        with destination.open("wb") as handle:
            pickle.dump(_make_pickle_safe(payload), handle, protocol=4)
        return destination

    def load_session(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> "Study":
        """Apply a previously saved analysis session onto this study."""

        source = Path(path)
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        experiments_payload = dict(payload.get("experiments", {}))

        missing_stems = [
            stem
            for stem in experiments_payload
            if stem not in self._experiments_by_stem()
        ]
        self._warn_missing_session_stems(missing_stems)

        timestamp_mismatches: list[str] = []
        file_mismatches: list[str] = []
        state_conflicts: dict[str, list[str]] = {
            "enabled state": [],
            "onset": [],
            "force peaks": [],
            "unloading": [],
            "Oliver-Pharr": [],
        }

        updated: list[Experiment] = []
        for experiment in self.experiments:
            saved = experiments_payload.get(experiment.stem)
            if saved is None:
                updated.append(experiment)
                continue

            if saved.get("timestamp") != experiment.timestamp:
                timestamp_mismatches.append(experiment.stem)
            saved_source_name = saved.get("source_name", saved.get("hld_name"))
            if saved_source_name != _experiment_source_name(experiment):
                file_mismatches.append(experiment.stem)

            current = experiment
            current, enabled_conflict = _apply_enabled_state(
                current,
                saved_enabled=bool(saved.get("enabled", _DEFAULT_ENABLED)),
                saved_reason=saved.get(
                    "disabled_reason", _DEFAULT_DISABLED_REASON
                ),
                overwrite=overwrite,
            )
            if enabled_conflict:
                state_conflicts["enabled state"].append(experiment.stem)

            current, onset_conflict = _apply_saved_analysis_result(
                current,
                saved_result=saved.get("onset"),
                current_result=current.onset,
                overwrite=overwrite,
                result_name="onset",
            )
            if onset_conflict:
                state_conflicts["onset"].append(experiment.stem)

            current, peaks_conflict = _apply_saved_analysis_result(
                current,
                saved_result=saved.get("force_peaks"),
                current_result=current.force_peaks,
                overwrite=overwrite,
                result_name="force_peaks",
            )
            if peaks_conflict:
                state_conflicts["force peaks"].append(experiment.stem)

            current, unloading_conflict = _apply_saved_analysis_result(
                current,
                saved_result=saved.get("unloading"),
                current_result=current.unloading,
                overwrite=overwrite,
                result_name="unloading",
            )
            if unloading_conflict:
                state_conflicts["unloading"].append(experiment.stem)

            current, oliver_conflict = _apply_saved_analysis_result(
                current,
                saved_result=saved.get("oliver_pharr"),
                current_result=current.oliver_pharr,
                overwrite=overwrite,
                result_name="oliver_pharr",
            )
            if oliver_conflict:
                state_conflicts["Oliver-Pharr"].append(experiment.stem)
            updated.append(current)

        self._warn_session_mismatches(
            mismatch_name="timestamp",
            stems=timestamp_mismatches,
        )
        self._warn_session_mismatches(
            mismatch_name="filename",
            stems=file_mismatches,
        )
        self._warn_session_conflicts(
            state_conflicts=state_conflicts,
            overwrite=overwrite,
        )
        return Study(experiments=tuple(updated))

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

    def _warn_missing_session_stems(self, stems: Sequence[str]) -> None:
        """Warn when a saved session contains stems absent from the study."""

        if not stems:
            return
        warnings.warn(
            f"Skipped {len(stems)} saved experiment(s) because the current "
            f"study does not contain those stems: {', '.join(stems)}.",
            stacklevel=2,
        )

    def _warn_session_mismatches(
        self,
        *,
        mismatch_name: str,
        stems: Sequence[str],
    ) -> None:
        """Warn when saved metadata does not match the current study."""

        if not stems:
            return
        warnings.warn(
            f"Found {mismatch_name} mismatches for {len(stems)} "
            f"experiment(s) while loading the session: {', '.join(stems)}.",
            stacklevel=2,
        )

    def _warn_session_conflicts(
        self,
        *,
        state_conflicts: Mapping[str, Sequence[str]],
        overwrite: bool,
    ) -> None:
        """Warn once per state type when saved values were not applied."""

        if overwrite:
            return
        for state_name, stems in state_conflicts.items():
            if not stems:
                continue
            warnings.warn(
                f"Kept current {state_name} for {len(stems)} experiment(s) "
                f"instead of the saved session state: {', '.join(stems)}. "
                "Pass overwrite=True to apply the saved state.",
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


def _scalar_metric_getter(metric: str) -> Any:
    """Return the accessor function for a supported scalar metric."""

    registry = {
        "hardness": lambda experiment: (
            None
            if experiment.oliver_pharr is None
            else experiment.oliver_pharr.hardness_uN_per_nm2
        ),
        "reduced_modulus": lambda experiment: (
            None
            if experiment.oliver_pharr is None
            else experiment.oliver_pharr.reduced_modulus_uN_per_nm2
        ),
        "stiffness": lambda experiment: (
            None
            if experiment.oliver_pharr is None
            else experiment.oliver_pharr.stiffness_uN_per_nm
        ),
        "onset_disp": lambda experiment: (
            None
            if experiment.onset is None
            else experiment.onset.onset_disp_nm
        ),
        "onset_time": lambda experiment: (
            None if experiment.onset is None else experiment.onset.onset_time_s
        ),
        "force_peak_count": lambda experiment: (
            None
            if experiment.force_peaks is None
            else experiment.force_peaks.peak_count
        ),
    }
    if metric not in registry:
        raise ValueError(
            "Unknown scalar metric. Supported metrics are: "
            f"{', '.join(sorted(registry))}."
        )
    return registry[metric]


def _package_version() -> str:
    """Return the installed package version when available."""

    try:
        return metadata.version("nanodent")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def _session_entry(experiment: Experiment) -> dict[str, Any]:
    """Return the persisted session state for one experiment."""

    return {
        "timestamp": experiment.timestamp,
        "source_name": _experiment_source_name(experiment),
        "hld_name": _experiment_source_name(experiment),
        "enabled": experiment.enabled,
        "disabled_reason": experiment.disabled_reason,
        "onset": _make_pickle_safe(experiment.onset),
        "force_peaks": _make_pickle_safe(experiment.force_peaks),
        "unloading": _make_pickle_safe(experiment.unloading),
        "oliver_pharr": _make_pickle_safe(experiment.oliver_pharr),
    }


def _make_pickle_safe(value: Any) -> Any:
    """Recursively replace non-picklable immutable views with plain values."""

    if value is None:
        return None
    if isinstance(value, MappingProxyType):
        return {key: _make_pickle_safe(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _make_pickle_safe(getattr(value, field.name))
                for field in fields(value)
            }
        )
    if isinstance(value, tuple):
        return tuple(_make_pickle_safe(item) for item in value)
    if isinstance(value, list):
        return [_make_pickle_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            _make_pickle_safe(key): _make_pickle_safe(item)
            for key, item in value.items()
        }
    return value


def _apply_enabled_state(
    experiment: Experiment,
    *,
    saved_enabled: bool,
    saved_reason: str | None,
    overwrite: bool,
) -> tuple[Experiment, bool]:
    """Apply saved enabled state unless current manual state should win."""

    current_is_default = (
        experiment.enabled is _DEFAULT_ENABLED
        and experiment.disabled_reason == _DEFAULT_DISABLED_REASON
    )
    saved_matches_current = (
        experiment.enabled == saved_enabled
        and experiment.disabled_reason == saved_reason
    )
    if overwrite or current_is_default or saved_matches_current:
        return (
            experiment.with_enabled(saved_enabled, reason=saved_reason),
            False,
        )
    return experiment, True


def _apply_saved_analysis_result(
    experiment: Experiment,
    *,
    saved_result: Any,
    current_result: Any,
    overwrite: bool,
    result_name: str,
) -> tuple[Experiment, bool]:
    """Apply a saved analysis result when the conflict policy allows it."""

    if saved_result is None:
        if overwrite:
            return _replace_analysis_result(
                experiment,
                result_name=result_name,
                result=None,
            ), False
        return experiment, False
    if current_result is not None and not overwrite:
        return experiment, True
    return _replace_analysis_result(
        experiment,
        result_name=result_name,
        result=saved_result,
    ), False


def _replace_analysis_result(
    experiment: Experiment,
    *,
    result_name: str,
    result: Any,
) -> Experiment:
    """Return an experiment with one attached analysis result replaced."""

    if result_name == "onset":
        return experiment.with_onset(result)
    if result_name == "force_peaks":
        return experiment.with_force_peaks(result)
    if result_name == "unloading":
        return experiment.with_unloading(result)
    if result_name == "oliver_pharr":
        return experiment.with_oliver_pharr(result)
    raise ValueError(f"Unknown analysis result {result_name!r}.")


def _experiment_source_name(experiment: Experiment) -> str | None:
    """Return the best available source identifier for one experiment."""

    if experiment.source_path is not None:
        return experiment.source_path.name
    if experiment.paths is not None:
        return experiment.paths.hld_path.name
    return None
