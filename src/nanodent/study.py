"""Study-level containers and grouping utilities."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, Sequence

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
            count, and experiment stems.
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
            "stems": self.stems,
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
        """Create explicit groups from Python-side selections.

        Args:
            groups: Manually chosen experiment sequences, such as selections
                made in a notebook or GUI.
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
