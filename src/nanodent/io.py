"""Input and loading utilities for nanoindentation experiments."""

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from nanodent.models import (
    Experiment,
    ExperimentPaths,
    MetadataEntry,
    SegmentDefinition,
    SignalTable,
)
from nanodent.study import Study

_SECTION_NAMES = {
    "sample approach": "approach",
    "drift measurement": "drift",
    "test": "test",
}
_DATA_POINTS_PATTERN = re.compile(
    r"^(?P<label>.+?)\s+Data Points:\s*(?P<count>\d+)\s*$"
)
_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %Y"
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def load_experiment(path: str | Path) -> Experiment:
    """Load a single experiment from disk.

    Args:
        path: Path to an experiment `.hld` file or to a sibling file that
            shares the same experiment stem.

    Returns:
        Parsed experiment including metadata, numeric sections, and sibling
        file paths.
    """

    input_path = Path(path)
    experiment_paths = _resolve_experiment_paths(input_path)
    return _parse_hld_file(experiment_paths)


def load_folder(path: str | Path) -> Study:
    """Load all parseable experiments from a folder.

    Args:
        path: Directory containing one or more experiment file triplets.

    Returns:
        Study containing all experiments discovered in the folder, sorted by
        acquisition timestamp.

    Raises:
        NotADirectoryError: If `path` does not point to a directory.
    """

    folder = Path(path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory.")

    experiments = [
        _parse_hld_file(experiment_paths)
        for experiment_paths in _scan_experiment_paths(folder)
        if experiment_paths.hld_path.exists()
    ]
    return Study(experiments=tuple(experiments))


def _resolve_experiment_paths(path: Path) -> ExperimentPaths:
    """Resolve the canonical sibling paths for one experiment stem.

    Args:
        path: Candidate experiment path, typically an `.hld`, `.tdm`, or `.tdx`
            file, or a bare stem path.

    Returns:
        Resolved sibling paths for the experiment.
    """

    path = path.expanduser()
    if path.suffix.lower() == ".hld":
        hld_path = path
        stem = path.stem
    else:
        stem = path.stem if path.suffix else path.name
        hld_path = (
            path.with_suffix(".hld") if path.suffix else path / f"{stem}.hld"
        )
        if path.is_dir():
            raise IsADirectoryError(
                "load_experiment expects a file path, not a folder."
            )
    if not hld_path.exists():
        raise FileNotFoundError(
            f"Could not find canonical .hld file for {path}."
        )
    return ExperimentPaths(
        stem=stem,
        hld_path=hld_path,
        tdm_path=hld_path.with_suffix(".tdm")
        if hld_path.with_suffix(".tdm").exists()
        else None,
        tdx_path=hld_path.with_suffix(".tdx")
        if hld_path.with_suffix(".tdx").exists()
        else None,
    )


def _scan_experiment_paths(folder: Path) -> list[ExperimentPaths]:
    """Scan a folder for experiment stems with a canonical `.hld` file.

    Args:
        folder: Directory to scan.

    Returns:
        Sorted experiment path bundles for all stems that contain an `.hld`
        file.
    """

    stems: dict[str, dict[str, Path]] = {}
    for file_path in sorted(folder.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix not in {".hld", ".tdm", ".tdx"}:
            continue
        stems.setdefault(file_path.stem, {})[suffix] = file_path

    experiment_paths = [
        ExperimentPaths(
            stem=stem,
            hld_path=paths[".hld"],
            tdm_path=paths.get(".tdm"),
            tdx_path=paths.get(".tdx"),
        )
        for stem, paths in stems.items()
        if ".hld" in paths
    ]
    return sorted(experiment_paths, key=lambda item: item.stem)


def _parse_hld_file(experiment_paths: ExperimentPaths) -> Experiment:
    """Parse a canonical `.hld` file into an experiment object.

    Args:
        experiment_paths: Resolved sibling paths for one experiment stem.

    Returns:
        Parsed experiment instance.
    """

    text = experiment_paths.hld_path.read_text(encoding="iso-8859-1")
    try:
        metadata_entries, sections = _parse_hld_text(text)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse {experiment_paths.hld_path}: {exc}"
        ) from exc
    metadata = _metadata_mapping(metadata_entries)
    timestamp = datetime.strptime(metadata["Time Stamp"], _TIMESTAMP_FORMAT)
    return Experiment(
        paths=experiment_paths,
        metadata=metadata,
        metadata_entries=tuple(metadata_entries),
        timestamp=timestamp,
        approach=sections.get("approach"),
        drift=sections.get("drift"),
        test=_require_section(sections, "test"),
        temperature_c=_parse_first_float(metadata.get("Test Temp")),
        humidity_percent=_parse_first_float(metadata.get("Test Humidity")),
        segment_definitions=tuple(_parse_segments(metadata_entries)),
    )


def _parse_hld_text(
    text: str,
) -> tuple[list[MetadataEntry], dict[str, SignalTable]]:
    """Parse raw `.hld` text into metadata entries and numeric tables.

    Args:
        text: Raw file contents decoded from disk.

    Returns:
        Tuple containing metadata entries in file order and parsed signal
        tables by normalized section name.
    """

    metadata_entries: list[MetadataEntry] = []
    sections: dict[str, SignalTable] = {}
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if _DATA_POINTS_PATTERN.match(line):
            break
        key, value = _split_metadata_line(lines[index])
        metadata_entries.append(MetadataEntry(key=key, value=value))
        index += 1

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        match = _DATA_POINTS_PATTERN.match(line)
        if match is None:
            raise ValueError(
                f"Unexpected line while parsing sections: {lines[index]!r}"
            )
        label = match.group("label").strip()
        section_name = _SECTION_NAMES.get(label.lower())
        if section_name is None:
            raise ValueError(f"Unsupported section {label!r}")
        point_count = int(match.group("count"))
        index += 1
        if index >= len(lines):
            raise ValueError(f"Missing header row for section {label!r}")
        raw_columns = tuple(part.strip() for part in lines[index].split("\t"))
        index += 1
        rows: list[list[float]] = []
        while index < len(lines) and len(rows) < point_count:
            row_line = lines[index].strip()
            index += 1
            if not row_line:
                continue
            rows.append([float(cell.strip()) for cell in row_line.split("\t")])
        if len(rows) != point_count:
            raise ValueError(
                f"""Section {label!r} declared {point_count} points
                but contained {len(rows)} rows."""
            )
        sections[section_name] = _make_signal_table(raw_columns, rows)
    return metadata_entries, sections


def _split_metadata_line(line: str) -> tuple[str, str]:
    """Split a metadata line into key and value components.

    Args:
        line: Raw metadata line from the `.hld` file.

    Returns:
        Two-tuple of metadata key and value.
    """

    if ":" not in line:
        raise ValueError(f"Malformed metadata line: {line!r}")
    key, value = line.split(":", maxsplit=1)
    return key.strip(), value.strip()


def _make_signal_table(
    raw_columns: tuple[str, ...], rows: Iterable[Iterable[float]]
) -> SignalTable:
    """Build a signal table from column headers and numeric rows.

    Args:
        raw_columns: Column names exactly as written in the file.
        rows: Numeric rows belonging to the section.

    Returns:
        Signal table with normalized column names and float64 arrays.
    """

    normalized_columns = [
        _normalize_column_name(column) for column in raw_columns
    ]
    matrix = np.asarray(list(rows), dtype=np.float64)
    if matrix.size == 0:
        matrix = np.empty((0, len(normalized_columns)), dtype=np.float64)
    elif matrix.ndim != 2:
        raise ValueError(
            "Section data could not be interpreted as a "
            "rectangular numeric table."
        )
    if matrix.shape[1] != len(normalized_columns):
        raise ValueError(
            f"Section declared {len(normalized_columns)} columns "
            f"but contained {matrix.shape[1]}."
        )
    columns = {
        normalized: matrix[:, index].astype(np.float64, copy=False)
        for index, normalized in enumerate(normalized_columns)
    }
    return SignalTable(
        columns=columns, point_count=matrix.shape[0], raw_columns=raw_columns
    )


def _metadata_mapping(entries: Iterable[MetadataEntry]) -> dict[str, str]:
    """Convert ordered metadata entries into a plain mapping.

    Args:
        entries: Metadata entries from the parsed file.

    Returns:
        Dictionary keyed by metadata field name.
    """

    mapping: dict[str, str] = {}
    for entry in entries:
        mapping[entry.key] = entry.value
    return mapping


def _parse_segments(
    entries: Iterable[MetadataEntry],
) -> list[SegmentDefinition]:
    """Extract segment metadata blocks from ordered metadata entries.

    Args:
        entries: Metadata entries from the parsed file.

    Returns:
        Segment definitions in file order.
    """

    segments: list[SegmentDefinition] = []
    current: dict[str, str] = {}
    for entry in entries:
        if entry.key == "Segment Time":
            if current:
                segments.append(_build_segment(current))
                current = {}
            current[entry.key] = entry.value
            continue
        if current and entry.key.startswith("Segment "):
            current[entry.key] = entry.value
            continue
        if current:
            segments.append(_build_segment(current))
            current = {}
    if current:
        segments.append(_build_segment(current))
    return segments


def _build_segment(raw_segment: dict[str, str]) -> SegmentDefinition:
    """Create a segment definition from one raw metadata block.

    Args:
        raw_segment: Metadata key-value pairs belonging to one segment.

    Returns:
        Parsed segment definition.
    """

    return SegmentDefinition(
        number=int(raw_segment.get("Segment Number", "0")),
        segment_type=raw_segment.get("Segment Type", ""),
        duration_s=_parse_required_float(raw_segment, "Segment Time"),
        begin_time_s=_parse_required_float(raw_segment, "Segment Begin Time"),
        end_time_s=_parse_required_float(raw_segment, "Segment End Time"),
        begin_demand=_parse_first_float(
            raw_segment.get("Segment Begin Demand")
        ),
        end_demand=_parse_first_float(raw_segment.get("Segment End Demand")),
        points=int(float(raw_segment.get("Segment Points", "0"))),
    )


def _parse_required_float(mapping: dict[str, str], key: str) -> float:
    """Parse a required numeric value from a metadata mapping.

    Args:
        mapping: Source metadata mapping.
        key: Metadata key to extract.

    Returns:
        Parsed floating-point value.
    """

    value = _parse_first_float(mapping.get(key))
    if value is None:
        raise ValueError(f"Missing numeric value for {key!r}")
    return value


def _parse_first_float(value: str | None) -> float | None:
    """Extract the first floating-point token from a metadata string.

    Args:
        value: Raw metadata value that may contain units or extra text.

    Returns:
        Parsed float if one is present, otherwise `None`.
    """

    if value is None:
        return None
    match = _NUMBER_PATTERN.search(value)
    if match is None:
        return None
    return float(match.group(0))


def _normalize_column_name(column_name: str) -> str:
    """Normalize a raw column name into a Python-friendly identifier.

    Args:
        column_name: Column label as written in the input file.

    Returns:
        Snake-case column name with normalized unit suffixes.
    """

    text = column_name.strip()
    text = text.replace("µ", "u").replace("μ", "u").replace("�", "u")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    tokens = [token for token in text.split("_") if token]
    if not tokens:
        raise ValueError(f"Could not normalize column name {column_name!r}")
    if (
        len(tokens) >= 2
        and tokens[-2].lower() == "u"
        and tokens[-1].lower() == "n"
    ):
        tokens = [*tokens[:-2], "uN"]
    normalized_tokens: list[str] = []
    for index, token in enumerate(tokens):
        lower_token = token.lower()
        if index == len(tokens) - 1:
            normalized_tokens.append(
                {
                    "s": "s",
                    "mm": "mm",
                    "nm": "nm",
                    "un": "uN",
                    "v": "V",
                    "hz": "Hz",
                    "deg": "deg",
                }.get(lower_token, lower_token)
            )
        else:
            normalized_tokens.append(lower_token)
    return "_".join(normalized_tokens)


def _require_section(
    sections: dict[str, SignalTable], name: str
) -> SignalTable:
    """Return a required section or raise a descriptive error.

    Args:
        sections: Parsed section mapping.
        name: Section name to retrieve.

    Returns:
        Requested signal table.
    """

    try:
        return sections[name]
    except KeyError as exc:
        raise ValueError(f"Expected section {name!r} in .hld file.") from exc
