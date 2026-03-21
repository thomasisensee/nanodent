from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from nanodent import load_folder

DATA_DIR = Path(__file__).parent / "data"


def test_group_by_time_gap_creates_two_default_groups() -> None:
    study = load_folder(DATA_DIR)

    groups = study.group_by_time_gap()

    assert [len(group.experiments) for group in groups] == [2, 2]
    assert groups[0].stems == (
        "Tritium_Retention_Study_04.03.2026_0000",
        "Tritium_Retention_Study_04.03.2026_0001",
    )


def test_group_by_time_gap_keeps_exact_boundary_in_same_group() -> None:
    study = load_folder(DATA_DIR)
    first = replace(
        study.experiments[0], timestamp=datetime(2026, 3, 4, 13, 0, 0)
    )
    second = replace(
        study.experiments[1], timestamp=datetime(2026, 3, 4, 13, 30, 0)
    )
    third = replace(
        study.experiments[2], timestamp=datetime(2026, 3, 4, 14, 0, 1)
    )

    groups = replace(
        study, experiments=(first, second, third)
    ).group_by_time_gap(max_gap=timedelta(minutes=30))

    assert [len(group.experiments) for group in groups] == [2, 1]


def test_regroup_supports_manual_group_overrides() -> None:
    study = load_folder(DATA_DIR)

    groups = study.regroup(
        [[study.experiments[0], study.experiments[2]], [study.experiments[1]]]
    )

    assert [group.index for group in groups] == [0, 1]
    assert groups[0].stems == (
        "Tritium_Retention_Study_04.03.2026_0000",
        "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0059",
    )


def test_describe_groups_returns_group_summaries() -> None:
    study = load_folder(DATA_DIR)

    summaries = study.describe_groups()

    assert [summary["experiment_count"] for summary in summaries] == [2, 2]
    assert summaries[0]["index"] == 0
    assert summaries[0]["start"] == study.experiments[0].timestamp
    assert summaries[0]["end"] == study.experiments[1].timestamp
    assert summaries[0]["stems"] == (
        "Tritium_Retention_Study_04.03.2026_0000",
        "Tritium_Retention_Study_04.03.2026_0001",
    )
