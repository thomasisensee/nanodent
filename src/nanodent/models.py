"""Core domain models for nanoindentation experiments."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nanodent.analysis.force_peaks import ForcePeakDetectionResult
    from nanodent.analysis.oliver_pharr import OliverPharrExperimentResult
    from nanodent.analysis.onset import OnsetDetectionResult
    from nanodent.analysis.unloading import UnloadingDetectionResult


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

    columns: dict[str, NDArray[np.float64]]
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

    def __getitem__(self, column_name: str) -> NDArray[np.float64]:
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

    def to_dict(self) -> dict[str, NDArray[np.float64]]:
        """Return a shallow copy of the column mapping.

        Returns:
            Dictionary mapping normalized column names to float64 arrays.
        """

        return dict(self.columns)


@dataclass(frozen=True, slots=True)
class Experiment:
    timestamp: datetime
    test: SignalTable
    stem: str = ""
    paths: ExperimentPaths | None = None
    source_path: Path | None = None
    source_format: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    metadata_entries: tuple[MetadataEntry, ...] = ()
    approach: SignalTable | None = None
    drift: SignalTable | None = None
    extra_sections: Mapping[str, SignalTable] = field(default_factory=dict)
    temperature_c: float | None = None
    humidity_percent: float | None = None
    segment_definitions: tuple[SegmentDefinition, ...] = ()
    enabled: bool = True
    disabled_reason: str | None = None
    onset: "OnsetDetectionResult | None" = None
    force_peaks: "ForcePeakDetectionResult | None" = None
    unloading: "UnloadingDetectionResult | None" = None
    oliver_pharr: "OliverPharrExperimentResult | None" = None

    def __post_init__(self) -> None:
        """Normalize optional provenance and validate core experiment state."""

        if self.paths is not None:
            object.__setattr__(self, "stem", self.paths.stem)
        elif not self.stem:
            if self.source_path is not None:
                object.__setattr__(self, "stem", self.source_path.stem)
            else:
                raise ValueError(
                    "Experiment requires either stem, paths, or source_path."
                )

        if self.paths is not None:
            object.__setattr__(self, "source_path", self.paths.hld_path)
        if self.source_format is None and self.source_path is not None:
            suffix = self.source_path.suffix.lstrip(".")
            object.__setattr__(
                self,
                "source_format",
                suffix.lower() if suffix else None,
            )
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "extra_sections", dict(self.extra_sections))

    @property
    def trace(self) -> SignalTable:
        """Return the primary canonical measurement trace."""

        return self.test

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
                section = self.approach or self.extra_sections.get("approach")
                if section is None:
                    raise KeyError("Experiment has no approach section.")
                return section
            case "drift":
                section = self.drift or self.extra_sections.get("drift")
                if section is None:
                    raise KeyError("Experiment has no drift section.")
                return section
            case "test":
                return self.trace
            case _:
                try:
                    return self.extra_sections[name]
                except KeyError as exc:
                    raise KeyError(f"Unknown section {name!r}.") from exc

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
            "test_points": len(self.trace),
        }

    def unloading_curve(
        self,
        *,
        x: str = "disp_nm",
        y: str = "force_uN",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the unloading branch as aligned NumPy arrays.

        Args:
            x: Trace column used for the x-values.
            y: Trace column used for the y-values.

        Returns:
            Pair of NumPy arrays sliced from the detected unloading start
            through the end of the test trace.

        Raises:
            ValueError: If no successful unloading result is attached or the
                stored unloading start index is invalid.
            KeyError: If either requested column is absent from the test trace.
        """

        unloading = self.unloading
        if unloading is None or not unloading.success:
            raise ValueError(
                "Experiment has no successful unloading result attached."
            )
        if unloading.start_index is None:
            raise ValueError("Unloading result has no start index.")

        start_index = int(unloading.start_index)
        if start_index < 0 or start_index >= len(self.trace):
            raise ValueError(
                "Unloading start index must lie within the test trace."
            )

        x_values = np.asarray(self.trace[x], dtype=np.float64)[start_index:]
        y_values = np.asarray(self.trace[y], dtype=np.float64)[start_index:]
        return x_values, y_values

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

    def with_oliver_pharr(
        self, result: "OliverPharrExperimentResult | None"
    ) -> "Experiment":
        """Return a copy of the experiment with updated analysis results."""

        return replace(self, oliver_pharr=result)

    def with_onset(
        self, result: "OnsetDetectionResult | None"
    ) -> "Experiment":
        """Return a copy of the experiment with updated onset results."""

        return replace(self, onset=result)

    def with_force_peaks(
        self, result: "ForcePeakDetectionResult | None"
    ) -> "Experiment":
        """Return a copy of the experiment with updated force peaks."""

        return replace(self, force_peaks=result)

    def with_unloading(
        self, result: "UnloadingDetectionResult | None"
    ) -> "Experiment":
        """Return a copy of the experiment with updated unloading data."""

        return replace(self, unloading=result)

    @classmethod
    def from_measurements(
        cls,
        *,
        stem: str,
        timestamp: datetime,
        time: Any,
        displacement: Any,
        force: Any,
        time_unit: str = "s",
        displacement_unit: str = "nm",
        force_unit: str = "uN",
        raw_columns: tuple[str, str, str] | None = None,
        source_path: str | Path | None = None,
        source_format: str | None = None,
        metadata: Mapping[str, str] | None = None,
        metadata_entries: tuple[MetadataEntry, ...] = (),
        approach: SignalTable | None = None,
        drift: SignalTable | None = None,
        extra_sections: Mapping[str, SignalTable] | None = None,
        temperature_c: float | None = None,
        humidity_percent: float | None = None,
        segment_definitions: tuple[SegmentDefinition, ...] = (),
        enabled: bool = True,
        disabled_reason: str | None = None,
    ) -> "Experiment":
        """Create one experiment from array-like measurement signals."""

        time_array = _coerce_measurement_array(time, name="time")
        disp_array = _coerce_measurement_array(
            displacement,
            name="displacement",
        )
        force_array = _coerce_measurement_array(force, name="force")
        expected_shape = time_array.shape
        for name, values in (
            ("displacement", disp_array),
            ("force", force_array),
        ):
            if values.shape != expected_shape:
                raise ValueError(f"{name} must have the same shape as time.")

        canonical_columns = {
            "time_s": time_array * _time_scale_factor(time_unit),
            "disp_nm": disp_array
            * _displacement_scale_factor(displacement_unit),
            "force_uN": force_array * _force_scale_factor(force_unit),
        }
        trace = SignalTable(
            columns=canonical_columns,
            point_count=len(time_array),
            raw_columns=raw_columns
            or (
                f"time_{time_unit}",
                f"displacement_{displacement_unit}",
                f"force_{force_unit}",
            ),
        )
        return cls(
            stem=stem,
            timestamp=timestamp,
            test=trace,
            source_path=None if source_path is None else Path(source_path),
            source_format=source_format,
            metadata=dict(metadata or {}),
            metadata_entries=metadata_entries,
            approach=approach,
            drift=drift,
            extra_sections=dict(extra_sections or {}),
            temperature_c=temperature_c,
            humidity_percent=humidity_percent,
            segment_definitions=segment_definitions,
            enabled=enabled,
            disabled_reason=disabled_reason,
        )

    @classmethod
    def from_tabular_data(
        cls,
        table: Mapping[str, Any] | Any,
        *,
        stem: str,
        timestamp: datetime,
        time_column: str,
        displacement_column: str,
        force_column: str,
        time_unit: str = "s",
        displacement_unit: str = "nm",
        force_unit: str = "uN",
        source_path: str | Path | None = None,
        source_format: str | None = None,
        metadata: Mapping[str, str] | None = None,
        metadata_entries: tuple[MetadataEntry, ...] = (),
        approach: SignalTable | None = None,
        drift: SignalTable | None = None,
        extra_sections: Mapping[str, SignalTable] | None = None,
        temperature_c: float | None = None,
        humidity_percent: float | None = None,
        segment_definitions: tuple[SegmentDefinition, ...] = (),
        enabled: bool = True,
        disabled_reason: str | None = None,
    ) -> "Experiment":
        """Create one experiment from a mapping- or DataFrame-like table."""

        return cls.from_measurements(
            stem=stem,
            timestamp=timestamp,
            time=table[time_column],
            displacement=table[displacement_column],
            force=table[force_column],
            time_unit=time_unit,
            displacement_unit=displacement_unit,
            force_unit=force_unit,
            raw_columns=(time_column, displacement_column, force_column),
            source_path=source_path,
            source_format=source_format,
            metadata=metadata,
            metadata_entries=metadata_entries,
            approach=approach,
            drift=drift,
            extra_sections=extra_sections,
            temperature_c=temperature_c,
            humidity_percent=humidity_percent,
            segment_definitions=segment_definitions,
            enabled=enabled,
            disabled_reason=disabled_reason,
        )


def _coerce_measurement_array(
    values: Any, *, name: str
) -> NDArray[np.float64]:
    """Return one numeric measurement signal as a 1D float64 array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D signal.")
    return array


def _time_scale_factor(unit: str) -> float:
    """Return the factor converting one supported time unit into seconds."""

    normalized = unit.strip().lower()
    scales = {
        "s": 1.0,
        "sec": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
    }
    try:
        return scales[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported time unit {unit!r}. Supported units: s, ms."
        ) from exc


def _displacement_scale_factor(unit: str) -> float:
    """Return the factor converting supported displacement units into nm."""

    normalized = _normalize_unit(unit)
    scales = {
        "nm": 1.0,
        "um": 1e3,
        "mm": 1e6,
    }
    try:
        return scales[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unsupported displacement unit "
            f"{unit!r}. Supported units: nm, um, mm."
        ) from exc


def _force_scale_factor(unit: str) -> float:
    """Return the factor converting supported force units into uN."""

    normalized = _normalize_unit(unit)
    scales = {
        "un": 1.0,
        "mn": 1e3,
        "n": 1e6,
    }
    try:
        return scales[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported force unit {unit!r}. Supported units: uN, mN, N."
        ) from exc


def _normalize_unit(unit: str) -> str:
    """Return a normalized ASCII unit token."""

    return (
        unit.strip()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("�", "u")
        .lower()
    )
