"""Study-level containers and grouping utilities."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator, Sequence

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
        self, max_gap: timedelta = timedelta(minutes=30)
    ) -> list[ExperimentGroup]:
        """Group experiments by the gap between consecutive timestamps.

        Args:
            max_gap: Maximum allowed delay between neighboring experiments for
                them to remain in the same group.

        Returns:
            Deterministic groups ordered by acquisition time.
        """

        if not self.experiments:
            return []

        groups: list[list[Experiment]] = [[self.experiments[0]]]
        for experiment in self.experiments[1:]:
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
        self, groups: Iterable[Sequence[Experiment]]
    ) -> list[ExperimentGroup]:
        """Create explicit groups from Python-side selections.

        Args:
            groups: Manually chosen experiment sequences, such as selections
                made in a notebook or GUI.

        Returns:
            Explicit experiment groups with stable indices and timestamp order
            within each group.
        """

        regrouped: list[ExperimentGroup] = []
        for index, group in enumerate(groups):
            sorted_group = tuple(
                sorted(group, key=lambda experiment: experiment.timestamp)
            )
            regrouped.append(
                ExperimentGroup(experiments=sorted_group, index=index)
            )
        return regrouped

    def describe_groups(
        self, max_gap: timedelta = timedelta(minutes=30)
    ) -> list[dict[str, Any]]:
        """Summarize time-gap groups for interactive inspection.

        Args:
            max_gap: Maximum allowed delay between neighboring experiments for
                them to remain in the same group.

        Returns:
            A list of compact dictionaries, one per group, suitable for quick
            display in notebooks or REPL sessions.
        """

        return [
            group.summary()
            for group in self.group_by_time_gap(max_gap=max_gap)
        ]
