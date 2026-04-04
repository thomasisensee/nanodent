"""Study-level containers and grouping utilities."""

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

    experiments: tuple[Experiment, ...]
    index: int = 0

    def __post_init__(self) -> None:
        """Validate that the group contains at least one experiment."""

        if not self.experiments:
            raise ValueError(
                "ExperimentGroup requires at least one experiment."
            )

    @property
    def start(self) -> datetime:
        """Timestamp of the first experiment in the group."""

        return self.experiments[0].timestamp

    @property
    def end(self) -> datetime:
        """Timestamp of the last experiment in the group."""

        return self.experiments[-1].timestamp

    @property
    def duration(self) -> timedelta:
        """Time spanned by the grouped experiments."""

        return self.end - self.start

    @property
    def stems(self) -> tuple[str, ...]:
        """Experiment stems in group order."""

        return tuple(experiment.stem for experiment in self.experiments)

    def summary(self) -> dict[str, Any]:
        """Return a notebook-friendly summary of the group.

        Returns:
            Dictionary containing the group index, temporal extent, experiment
            count, and experiment stems annotated with enabled state.
        """

        return {
            "index": self.index,
            "experiment_count": len(self.experiments),
            "enabled_count": sum(
                experiment.enabled for experiment in self.experiments
            ),
            "disabled_count": sum(
                not experiment.enabled for experiment in self.experiments
            ),
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "stems": tuple(
                f"{experiment.stem} "
                f"({'enabled' if experiment.enabled else 'disabled'})"
                for experiment in self.experiments
            ),
        }


@dataclass(frozen=True, slots=True)
class Study:
    """A collection of experiments sorted by acquisition timestamp."""

    experiments: tuple[Experiment, ...]

    def __post_init__(self) -> None:
        """Store experiments in ascending timestamp order."""

        object.__setattr__(
            self,
            "experiments",
            tuple(
                sorted(
                    self.experiments,
                    key=lambda experiment: experiment.timestamp,
                )
            ),
        )

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
        """Group experiments by the gap between consecutive timestamps.

        Args:
            max_gap: Maximum allowed delay between neighboring experiments for
                them to remain in the same group.
            include_disabled: Whether disabled experiments should participate
                in grouping.

        Returns:
            Deterministic groups ordered by acquisition time.
        """

        experiments = self._selected_experiments(
            include_disabled=include_disabled
        )
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
            ExperimentGroup(experiments=tuple(group), index=index)
            for index, group in enumerate(groups)
        ]

    def regroup(
        self,
        groups: Iterable[Sequence[Experiment]],
        *,
        include_disabled: bool = False,
    ) -> list[ExperimentGroup]:
        """Create explicit groups from Python-side experiment selections.

        Args:
            groups: Manually chosen experiment sequences, such as selections
                made in a notebook or GUI. Use
                `group_by_datetime_ranges(...)` for timestamp-window based
                grouping.
            include_disabled: Whether disabled experiments should remain in the
                regrouped result.

        Returns:
            Explicit experiment groups with stable indices and timestamp order
            within each group.
        """

        regrouped: list[ExperimentGroup] = []
        for group in groups:
            selected_group = tuple(
                experiment
                for experiment in group
                if include_disabled or experiment.enabled
            )
            if not selected_group:
                continue
            sorted_group = tuple(
                sorted(
                    selected_group, key=lambda experiment: experiment.timestamp
                )
            )
            regrouped.append(
                ExperimentGroup(experiments=sorted_group, index=len(regrouped))
            )
        return regrouped

    def group_by_datetime_ranges(
        self,
        ranges: Iterable[tuple[datetime, datetime]],
        *,
        include_disabled: bool = False,
    ) -> list[ExperimentGroup]:
        """Create groups from explicit inclusive datetime windows.

        Args:
            ranges: Inclusive `(start, end)` datetime pairs used to select
                experiments by timestamp.
            include_disabled: Whether disabled experiments should remain in the
                grouped result.

        Returns:
            Explicit experiment groups in the order the datetime windows were
            requested.

        Raises:
            ValueError: If a datetime range has `start > end` or if requested
                ranges overlap.
        """

        validated_ranges = self._validate_datetime_ranges(ranges)
        experiments = self._selected_experiments(
            include_disabled=include_disabled
        )
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
                    experiments=selected_group,
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
        """Summarize time-gap groups for interactive inspection.

        Args:
            max_gap: Maximum allowed delay between neighboring experiments for
                them to remain in the same group.
            include_disabled: Whether disabled experiments should remain in the
                described groups. Defaults to `True` so the summary reflects
                quality filtering decisions.

        Returns:
            A list of compact dictionaries, one per group, suitable for quick
            display in notebooks or REPL sessions.
        """

        return [
            group.summary()
            for group in self.group_by_time_gap(
                max_gap=max_gap, include_disabled=include_disabled
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
        """Return a study with experiments flagged by quality heuristics.

        Args:
            min_robust_force_span_uN: Minimum acceptable robust force span for
                the flat-force check.
            low_quantile: Lower quantile used for the flat-force span.
            high_quantile: Upper quantile used for the flat-force span.
            max_disp_nm: Maximum allowed displacement before disabling the
                experiment.
            peak_bin_count: Number of coarse displacement bins used for the
                peak-balance heuristic.
            peak_prominence_fraction: Minimum prominence used to resolve
                peaks, relative to the coarse-force dynamic range.
            min_secondary_peak_fraction: Minimum allowed ratio between the
                second-highest and highest resolved peaks.
            require_two_peaks: When true, disable curves that do not resolve
                at least two peaks after smoothing.
            disp_z_threshold: Robust z-score threshold for isolated
                displacement spikes.
            force_z_threshold: Robust z-score threshold for isolated force
                spikes.
            bin_count: Number of coarse displacement bins for onset-shape
                detection.
            baseline_bin_count: Number of early bins used for baseline force.
            onset_force_fraction: Lower force fraction used to define onset.
            target_force_fraction: Upper force fraction used to define where
                the onset rise is considered complete.
            sustained_bins: Number of consecutive bins that must exceed the
                lower onset threshold.
            max_rise_width_fraction: Maximum allowed displacement width
                between the lower and upper onset thresholds before disabling
                the experiment.

        Returns:
            New study with updated `enabled` and `disabled_reason` values.
        """

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
        unloading_fraction: float = 0.2,
        smoothing: Mapping[str, Any] | None = None,
        fit_num_points: int = 2,
        use_force_peak: bool = True,
        include_disabled: bool = False,
    ) -> "Study":
        """Analyze selected experiments with a straight-line unloading fit.

        Args:
            unloading_fraction: Fraction of the post-peak unloading branch used
                for the fit.
            smoothing: Optional keyword args forwarded to `nanodent.savgol`
                and applied equally to displacement and force.
            fit_num_points: Number of points used for dense fitted-line
                coordinates.
            include_disabled: Whether disabled experiments should be analyzed
                alongside enabled ones.

        Returns:
            New study with per-experiment Oliver-Pharr fit results attached
            to selected experiments.
        """

        analyzed_stems = {
            experiment.stem
            for experiment in self._selected_experiments(
                include_disabled=include_disabled
            )
        }
        experiments = tuple(
            experiment.with_oliver_pharr(
                _analyze_oliver_pharr(
                    experiment.section("test")["disp_nm"],
                    experiment.section("test")["force_uN"],
                    unloading_fraction=unloading_fraction,
                    smoothing=smoothing,
                    fit_num_points=fit_num_points,
                    use_force_peak=use_force_peak,
                    stem=experiment.stem,
                )
                if experiment.stem in analyzed_stems
                else None
            )
            for experiment in self.experiments
        )
        return Study(experiments=experiments)

    def detect_onset(
        self,
        *,
        baseline_points: int = 100,
        k: float = 4.0,
        consecutive: int = 5,
        smoothing: Mapping[str, Any] | None = None,
        include_disabled: bool = False,
    ) -> "Study":
        """Detect onset on selected experiments using the test force signal.

        Args:
            baseline_points: Number of leading samples used to estimate the
                baseline statistics for thresholding.
            k: Number of baseline standard deviations above the mean required
                to accept an onset candidate.
            consecutive: Number of consecutive samples above the threshold
                required to accept an onset.
            smoothing: Optional keyword args forwarded to `nanodent.savgol`
                before thresholding.
            include_disabled: Whether disabled experiments should be analyzed
                alongside enabled ones.

        Returns:
            New study with per-experiment onset results attached to selected
            experiments.
        """

        analyzed_stems = {
            experiment.stem
            for experiment in self._selected_experiments(
                include_disabled=include_disabled
            )
        }
        experiments = tuple(
            experiment.with_onset(
                _detect_onset(
                    experiment.section("test")["force_uN"],
                    time_s=experiment.section("test")["time_s"],
                    disp_nm=experiment.section("test")["disp_nm"],
                    baseline_points=baseline_points,
                    k=k,
                    consecutive=consecutive,
                    smoothing=smoothing,
                )
                if experiment.stem in analyzed_stems
                else None
            )
            for experiment in self.experiments
        )
        return Study(experiments=experiments)

    def detect_force_peaks(
        self,
        *,
        prominence: float = 100.0,
        threshold: float | None = 1.0,
        include_disabled: bool = False,
    ) -> "Study":
        """Detect raw-force peaks on selected experiments.

        Args:
            prominence: Minimum peak prominence passed to `find_peaks`.
            threshold: Minimum peak threshold passed to `find_peaks`.
            include_disabled: Whether disabled experiments should be analyzed
                alongside enabled ones.

        Returns:
            New study with per-experiment force peak results attached to
            selected experiments.
        """

        analyzed_stems = {
            experiment.stem
            for experiment in self._selected_experiments(
                include_disabled=include_disabled
            )
        }
        experiments = tuple(
            experiment.with_force_peaks(
                _detect_force_peaks(
                    experiment.section("test")["force_uN"],
                    time_s=experiment.section("test")["time_s"],
                    disp_nm=experiment.section("test")["disp_nm"],
                    prominence=prominence,
                    threshold=threshold,
                )
                if experiment.stem in analyzed_stems
                else None
            )
            for experiment in self.experiments
        )
        return Study(experiments=experiments)

    def set_enabled(
        self,
        stems: Iterable[str] | str,
        *,
        enabled: bool,
        reason: str | None = None,
    ) -> "Study":
        """Return a study with selected stems manually enabled or disabled.

        Args:
            stems: One or more experiment stems to update.
            enabled: Whether matching experiments should be enabled.
            reason: Optional reason stored when disabling experiments.

        Returns:
            New study with updated experiment flags.
        """

        requested_stems = {stems} if isinstance(stems, str) else set(stems)
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

    def _selected_experiments(
        self, *, include_disabled: bool
    ) -> tuple[Experiment, ...]:
        """Return experiments filtered by enabled state when requested."""

        if include_disabled:
            return self.experiments
        return tuple(
            experiment for experiment in self.experiments if experiment.enabled
        )

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
