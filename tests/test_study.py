from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from nanodent import load_folder

DATA_DIR = Path(__file__).parent / "data"
FLAT_FORCE_STEM = "AIrIndent10000nm 02"
GRADUAL_ONSET_STEM = "Tritium_Retention_Study_04.03.2026_0009"
SECOND_GRADUAL_ONSET_STEM = "Tritium_Retention_Study_04.03.2026_0005"


def test_group_by_time_gap_creates_two_default_groups() -> None:
    study = load_folder(DATA_DIR)

    groups = study.group_by_time_gap()

    assert [len(group.experiments) for group in groups] == [1, 2, 2, 2]
    assert groups[0].stems == (FLAT_FORCE_STEM,)
    assert groups[1].stems == (
        "Tritium_Retention_Study_04.03.2026_0005",
        GRADUAL_ONSET_STEM,
    )
    assert groups[2].stems == (
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
    by_stem = {experiment.stem: experiment for experiment in study.experiments}

    groups = study.regroup(
        [
            [
                by_stem["Tritium_Retention_Study_04.03.2026_0000"],
                by_stem[
                    "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0059"
                ],
            ],
            [by_stem["Tritium_Retention_Study_04.03.2026_0001"]],
        ]
    )

    assert [group.index for group in groups] == [0, 1]
    assert groups[0].stems == (
        "Tritium_Retention_Study_04.03.2026_0000",
        "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0059",
    )


def test_describe_groups_returns_group_summaries() -> None:
    study = load_folder(DATA_DIR)

    summaries = study.describe_groups()

    assert [summary["experiment_count"] for summary in summaries] == [
        1,
        2,
        2,
        2,
    ]
    assert summaries[0]["index"] == 0
    assert summaries[0]["enabled_count"] == 1
    assert summaries[0]["disabled_count"] == 0
    assert summaries[2]["start"] == study.experiments[3].timestamp
    assert summaries[2]["end"] == study.experiments[4].timestamp
    assert summaries[2]["stems"] == (
        "Tritium_Retention_Study_04.03.2026_0000",
        "Tritium_Retention_Study_04.03.2026_0001",
    )


def test_describe_groups_includes_disabled_experiments_by_default() -> None:
    study = load_folder(DATA_DIR).classify_delayed_onset()

    summaries = study.describe_groups()

    assert [summary["disabled_count"] for summary in summaries] == [1, 2, 0, 0]
    assert summaries[0]["enabled_count"] == 0
    assert summaries[1]["enabled_count"] == 0
    assert summaries[3]["enabled_count"] == 2


def test_classify_delayed_onset_disables_gradual_onset_experiment() -> None:
    study = load_folder(DATA_DIR)

    classified = study.classify_delayed_onset()
    by_stem = {
        experiment.stem: experiment for experiment in classified.experiments
    }

    assert by_stem[GRADUAL_ONSET_STEM].enabled is False
    assert by_stem[GRADUAL_ONSET_STEM].disabled_reason == "gradual_onset"
    assert by_stem["Tritium_Retention_Study_04.03.2026_0000"].enabled is True
    assert (
        by_stem["Tritium_Retention_Study_04.03.2026_0000"].disabled_reason
        is None
    )


def test_classify_delayed_onset_disables_second_gradual_onset_experiment() -> (
    None
):
    study = load_folder(DATA_DIR)

    classified = study.classify_delayed_onset()
    by_stem = {
        experiment.stem: experiment for experiment in classified.experiments
    }

    assert by_stem[SECOND_GRADUAL_ONSET_STEM].enabled is False
    assert (
        by_stem[SECOND_GRADUAL_ONSET_STEM].disabled_reason == "gradual_onset"
    )


def test_classify_delayed_onset_keeps_late_but_steep_run_enabled() -> None:
    study = load_folder(DATA_DIR)

    classified = study.classify_delayed_onset()
    by_stem = {
        experiment.stem: experiment for experiment in classified.experiments
    }

    assert (
        by_stem[
            "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0059"
        ].enabled
        is True
    )
    assert (
        by_stem[
            "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0059"
        ].disabled_reason
        is None
    )


def test_classify_delayed_onset_disables_flat_force_experiment() -> None:
    study = load_folder(DATA_DIR)

    classified = study.classify_delayed_onset()
    by_stem = {
        experiment.stem: experiment for experiment in classified.experiments
    }

    assert by_stem[FLAT_FORCE_STEM].enabled is False
    assert by_stem[FLAT_FORCE_STEM].disabled_reason == "flat_force"


def test_disabled_experiments_are_ignored_by_default_grouping() -> None:
    study = load_folder(DATA_DIR).disable_experiments(
        [
            FLAT_FORCE_STEM,
            "Tritium_Retention_Study_04.03.2026_0005",
            GRADUAL_ONSET_STEM,
        ]
    )

    assert [len(group.experiments) for group in study.group_by_time_gap()] == [
        2,
        2,
    ]
    assert [
        len(group.experiments)
        for group in study.group_by_time_gap(include_disabled=True)
    ] == [1, 2, 2, 2]


def test_manual_enable_and_disable_return_new_studies() -> None:
    study = load_folder(DATA_DIR)

    disabled = study.disable_experiments(FLAT_FORCE_STEM, reason="manual")
    restored = disabled.enable_experiments(FLAT_FORCE_STEM)

    assert study.experiments[0].enabled is True
    assert disabled.experiments[0].enabled is False
    assert disabled.experiments[0].disabled_reason == "manual"
    assert restored.experiments[0].enabled is True
    assert restored.experiments[0].disabled_reason is None
