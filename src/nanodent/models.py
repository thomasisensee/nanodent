"""Core domain models for nanoindentation experiments."""

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MetadataEntry:
    """A single raw metadata line from an input file."""

    key: str
    value: str


@dataclass(frozen=True, slots=True)
class SegmentDefinition:
    """Metadata describing one acquisition segment."""

    number: int
    segment_type: str
    duration_s: float
    begin_time_s: float
    end_time_s: float
    begin_demand: float | None
    end_demand: float | None
    points: int


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Paths belonging to one experiment stem."""

    stem: str
    hld_path: Path
    tdm_path: Path | None = None
    tdx_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SignalTable:
    """Tabular numeric section data with normalized column names."""

    columns: dict[str, FloatArray]
    point_count: int
    raw_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate column consistency and declared point count."""

        lengths = {len(values) for values in self.columns.values()}
        if not self.columns:
            raise ValueError("SignalTable requires at least one column.")
        if len(lengths) != 1:
            raise ValueError(
                "All columns in a SignalTable must have the same length."
            )
        column_length = lengths.pop()
        if self.point_count != column_length:
            raise ValueError(
                f"""SignalTable point_count={self.point_count} does not match
                column length {column_length}."""
            )

    def __len__(self) -> int:
        """Return the number of rows in the table."""

        return self.point_count

    def __getitem__(self, column_name: str) -> FloatArray:
        """Return one column by normalized column name.

        Args:
            column_name: Normalized column name to retrieve.

        Returns:
            Underlying float64 NumPy array for that column.
        """

        return self.columns[column_name]

    @property
    def column_names(self) -> tuple[str, ...]:
        """Normalized column names in source order."""

        return tuple(self.columns.keys())

    def to_dict(self) -> dict[str, FloatArray]:
        """Return a shallow copy of the column mapping.

        Returns:
            Dictionary mapping normalized column names to float64 arrays.
        """

        return dict(self.columns)


@dataclass(frozen=True, slots=True)
class Experiment:
    """One nanoindentation experiment parsed from disk."""

    paths: ExperimentPaths
    metadata: dict[str, str]
    metadata_entries: tuple[MetadataEntry, ...]
    timestamp: datetime
    approach: SignalTable | None
    drift: SignalTable | None
    test: SignalTable
    temperature_c: float | None = None
    humidity_percent: float | None = None
    segment_definitions: tuple[SegmentDefinition, ...] = ()
    enabled: bool = True
    disabled_reason: str | None = None

    @property
    def stem(self) -> str:
        """Return the experiment stem used across sibling files."""

        return self.paths.stem

    def section(self, name: str) -> SignalTable:
        """Return a named signal section.

        Args:
            name: Section name, typically `approach`, `drift`, or `test`.

        Returns:
            Requested signal table.

        Raises:
            KeyError: If the section does not exist on the experiment.
        """

        match name:
            case "approach":
                if self.approach is None:
                    raise KeyError("Experiment has no approach section.")
                return self.approach
            case "drift":
                if self.drift is None:
                    raise KeyError("Experiment has no drift section.")
                return self.drift
            case "test":
                return self.test
            case _:
                raise KeyError(f"Unknown section {name!r}.")

    def summary(self) -> dict[str, Any]:
        """Return a compact summary useful for quick inspection.

        Returns:
            Dictionary containing key metadata and section sizes.
        """

        return {
            "stem": self.stem,
            "timestamp": self.timestamp,
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "approach_points": len(self.approach)
            if self.approach is not None
            else 0,
            "drift_points": len(self.drift) if self.drift is not None else 0,
            "test_points": len(self.test),
        }

    def with_enabled(
        self, enabled: bool, *, reason: str | None = None
    ) -> "Experiment":
        """Return a copy of the experiment with updated enabled state.

        Args:
            enabled: Whether the experiment should participate in default
                grouping, plotting, and analysis flows.
            reason: Optional short reason recorded when disabling an
                experiment. Enabled experiments always clear the reason.

        Returns:
            New experiment instance with updated enabled metadata.
        """

        return replace(
            self,
            enabled=enabled,
            disabled_reason=None if enabled else reason,
        )
