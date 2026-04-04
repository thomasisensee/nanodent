from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from nanodent.models import SignalTable
from nanodent.study import Study

EXPERIMENT_A = "experiment_a"
EXPERIMENT_B = "experiment_b"
EXPERIMENT_C = "experiment_c"
EXPERIMENT_D = "experiment_d"


def test_group_by_time_gap_creates_two_default_groups(base_study) -> None:
    groups = base_study.group_by_time_gap()

    assert [len(group.experiments) for group in groups] == [2, 2]
    assert groups[0].stems == (EXPERIMENT_A, EXPERIMENT_B)
    assert groups[1].stems == (EXPERIMENT_C, EXPERIMENT_D)


def test_group_by_time_gap_keeps_exact_boundary_in_same_group(
    base_study,
) -> None:
    first = replace(
        base_study.experiments[0], timestamp=datetime(2026, 3, 4, 13, 0, 0)
    )
    second = replace(
        base_study.experiments[1], timestamp=datetime(2026, 3, 4, 13, 30, 0)
    )
    third = replace(
        base_study.experiments[2], timestamp=datetime(2026, 3, 4, 14, 0, 1)
    )

    groups = replace(
        base_study, experiments=(first, second, third)
    ).group_by_time_gap(max_gap=timedelta(minutes=30))

    assert [len(group.experiments) for group in groups] == [2, 1]


def test_regroup_supports_manual_group_overrides(base_study) -> None:
    by_stem = {
        experiment.stem: experiment for experiment in base_study.experiments
    }

    groups = base_study.regroup(
        [
            [
                by_stem[EXPERIMENT_A],
                by_stem[EXPERIMENT_C],
            ],
            [by_stem[EXPERIMENT_B]],
        ]
    )

    assert [group.index for group in groups] == [0, 1]
    assert groups[0].stems == (EXPERIMENT_A, EXPERIMENT_C)


def test_group_by_datetime_ranges_selects_experiments_in_range(
    base_study,
) -> None:
    groups = base_study.group_by_datetime_ranges(
        [
            (
                datetime(2026, 3, 4, 13, 56, 27),
                datetime(2026, 3, 4, 14, 2, 2),
            ),
            (
                datetime(2026, 3, 11, 12, 6, 42),
                datetime(2026, 3, 11, 12, 6, 42),
            ),
        ]
    )

    assert [group.index for group in groups] == [0, 1]
    assert groups[0].stems == (EXPERIMENT_A, EXPERIMENT_B)
    assert groups[1].stems == (EXPERIMENT_C,)


def test_group_by_datetime_ranges_skips_empty_ranges(base_study) -> None:
    groups = base_study.group_by_datetime_ranges(
        [
            (
                datetime(2026, 3, 1, 0, 0, 0),
                datetime(2026, 3, 1, 1, 0, 0),
            ),
            (
                datetime(2026, 3, 11, 12, 6, 42),
                datetime(2026, 3, 11, 12, 10, 0),
            ),
        ]
    )

    assert [group.index for group in groups] == [0]
    assert groups[0].stems == (EXPERIMENT_C, EXPERIMENT_D)


def test_group_by_datetime_ranges_excludes_disabled_by_default(
    base_study,
) -> None:
    study = base_study.disable_experiments(EXPERIMENT_B)

    groups = study.group_by_datetime_ranges(
        [
            (
                datetime(2026, 3, 4, 13, 56, 27),
                datetime(2026, 3, 4, 14, 2, 2),
            ),
        ]
    )

    assert len(groups) == 1
    assert groups[0].stems == (EXPERIMENT_A,)


def test_group_by_datetime_ranges_can_include_disabled(base_study) -> None:
    study = base_study.disable_experiments(EXPERIMENT_B)

    groups = study.group_by_datetime_ranges(
        [
            (
                datetime(2026, 3, 4, 13, 56, 27),
                datetime(2026, 3, 4, 14, 2, 2),
            ),
        ],
        include_disabled=True,
    )

    assert len(groups) == 1
    assert groups[0].stems == (EXPERIMENT_A, EXPERIMENT_B)


def test_group_by_datetime_ranges_rejects_invalid_range(base_study) -> None:
    with pytest.raises(
        ValueError, match="Datetime ranges require start <= end."
    ):
        base_study.group_by_datetime_ranges(
            [
                (
                    datetime(2026, 3, 4, 14, 2, 2),
                    datetime(2026, 3, 4, 13, 56, 27),
                ),
            ]
        )


def test_group_by_datetime_ranges_rejects_overlapping_ranges(
    base_study,
) -> None:
    with pytest.raises(ValueError, match="Datetime ranges must not overlap."):
        base_study.group_by_datetime_ranges(
            [
                (
                    datetime(2026, 3, 4, 13, 56, 27),
                    datetime(2026, 3, 4, 14, 0, 0),
                ),
                (
                    datetime(2026, 3, 4, 14, 0, 0),
                    datetime(2026, 3, 4, 14, 2, 2),
                ),
            ]
        )


def test_describe_groups_returns_group_summaries(base_study) -> None:
    summaries = base_study.describe_groups()

    assert [summary["experiment_count"] for summary in summaries] == [2, 2]
    assert summaries[0]["index"] == 0
    assert summaries[0]["enabled_count"] == 2
    assert summaries[0]["disabled_count"] == 0
    assert summaries[0]["start"] == base_study.experiments[0].timestamp
    assert summaries[1]["end"] == base_study.experiments[-1].timestamp
    assert summaries[0]["stems"] == (
        f"{EXPERIMENT_A} (enabled)",
        f"{EXPERIMENT_B} (enabled)",
    )


def test_classify_quality_keeps_stable_enabled_state_shape(base_study) -> None:
    classified = base_study.classify_quality()
    by_stem = {
        experiment.stem: experiment for experiment in classified.experiments
    }

    assert by_stem[EXPERIMENT_C].enabled is False
    assert by_stem[EXPERIMENT_C].disabled_reason == "high_disp"
    assert by_stem[EXPERIMENT_A].enabled is True
    assert by_stem[EXPERIMENT_A].disabled_reason is None
    assert sum(not exp.enabled for exp in classified.experiments) == 1


def test_manual_enable_and_disable_return_new_studies(base_study) -> None:
    disabled = base_study.disable_experiments(EXPERIMENT_A, reason="manual")
    restored = disabled.enable_experiments(EXPERIMENT_A)

    assert base_study.experiments[0].enabled is True
    assert disabled.experiments[0].enabled is False
    assert disabled.experiments[0].disabled_reason == "manual"
    assert restored.experiments[0].enabled is True
    assert restored.experiments[0].disabled_reason is None


def test_analyze_oliver_pharr_skips_disabled_experiments_by_default(
    base_study,
) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A)

    analyzed = study.analyze_oliver_pharr()
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert analyzed is not study
    assert by_stem[EXPERIMENT_A].oliver_pharr is None
    assert by_stem[EXPERIMENT_B].oliver_pharr is not None
    assert by_stem[EXPERIMENT_C].oliver_pharr is not None
    assert by_stem[EXPERIMENT_D].oliver_pharr is not None


def test_analyze_oliver_pharr_can_include_disabled_experiments(
    base_study,
) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A)

    analyzed = study.analyze_oliver_pharr(include_disabled=True)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert len(analyzed.experiments) == len(study.experiments)
    assert by_stem[EXPERIMENT_A].oliver_pharr is not None
    assert by_stem[EXPERIMENT_A].oliver_pharr.stem == EXPERIMENT_A


def test_analyze_oliver_pharr_attaches_hardness_after_onset_detection(
    base_study,
) -> None:
    analyzed = base_study.detect_onset().analyze_oliver_pharr()
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].oliver_pharr is not None
    assert by_stem[EXPERIMENT_A].oliver_pharr.success is True
    assert by_stem[EXPERIMENT_A].oliver_pharr.onset_disp_nm == pytest.approx(
        by_stem[EXPERIMENT_A].onset.onset_disp_nm
    )
    assert by_stem[EXPERIMENT_A].oliver_pharr.hardness_reason != (
        "missing_onset"
    )
    assert by_stem[EXPERIMENT_A].oliver_pharr.epsilon == pytest.approx(0.75)


def test_analyze_oliver_pharr_marks_missing_onset_when_not_detected(
    base_study,
) -> None:
    analyzed = base_study.analyze_oliver_pharr()
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].oliver_pharr is not None
    assert by_stem[EXPERIMENT_A].oliver_pharr.success is True
    assert by_stem[EXPERIMENT_A].oliver_pharr.hardness_success is False
    assert by_stem[EXPERIMENT_A].oliver_pharr.hardness_reason == (
        "missing_onset"
    )


def test_analyze_oliver_pharr_attaches_unsuccessful_results(
    base_study,
) -> None:
    short_test = SignalTable(
        columns={
            "time_s": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
            "disp_nm": np.array([0.0, 1.0, 2.0, 1.5], dtype=np.float64),
            "force_uN": np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float64),
        },
        point_count=4,
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = replace(base_study.experiments[0], test=short_test)
    analyzed = Study(experiments=(experiment,)).analyze_oliver_pharr()
    analyzed_experiment = analyzed.experiments[0]

    assert analyzed_experiment.oliver_pharr is not None
    assert analyzed_experiment.oliver_pharr.success is False
    assert (
        analyzed_experiment.oliver_pharr.reason == "too_few_unloading_points"
    )
    assert analyzed_experiment.oliver_pharr.stem == EXPERIMENT_A


def test_analyze_oliver_pharr_clears_existing_results_for_unselected_runs(
    base_study,
) -> None:
    once = base_study.analyze_oliver_pharr(include_disabled=True)
    disabled = once.disable_experiments(EXPERIMENT_A)
    analyzed = disabled.analyze_oliver_pharr(include_disabled=False)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].oliver_pharr is None


def test_detect_onset_skips_disabled_experiments_by_default(
    base_study,
) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A)

    analyzed = study.detect_onset()
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert analyzed is not study
    assert by_stem[EXPERIMENT_A].onset is None
    assert by_stem[EXPERIMENT_B].onset is not None
    assert by_stem[EXPERIMENT_C].onset is not None
    assert by_stem[EXPERIMENT_D].onset is not None


def test_detect_onset_can_include_disabled_experiments(base_study) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A)

    analyzed = study.detect_onset(include_disabled=True)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert len(analyzed.experiments) == len(study.experiments)
    assert by_stem[EXPERIMENT_A].onset is not None
    assert by_stem[EXPERIMENT_A].onset.onset_time_s is not None
    assert by_stem[EXPERIMENT_A].onset.onset_disp_nm is not None


def test_detect_onset_attaches_unsuccessful_results(base_study) -> None:
    flat_test = SignalTable(
        columns={
            "time_s": np.arange(8, dtype=np.float64),
            "disp_nm": np.linspace(0.0, 7.0, 8, dtype=np.float64),
            "force_uN": np.zeros(8, dtype=np.float64),
        },
        point_count=8,
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = replace(base_study.experiments[0], test=flat_test)

    analyzed = Study(experiments=(experiment,)).detect_onset(
        baseline_points=4,
        k=2.0,
        consecutive=3,
    )
    analyzed_experiment = analyzed.experiments[0]

    assert analyzed_experiment.onset is not None
    assert analyzed_experiment.onset.success is False
    assert analyzed_experiment.onset.reason == "no_onset_detected"
    assert analyzed_experiment.onset.onset_time_s is None
    assert analyzed_experiment.onset.onset_disp_nm is None


def test_detect_onset_attaches_time_and_displacement_values(
    base_study,
) -> None:
    experiment = base_study.experiments[0]
    analyzed = Study(experiments=(experiment,)).detect_onset()
    onset = analyzed.experiments[0].onset

    assert onset is not None
    if onset.success:
        assert onset.onset_index is not None
        assert onset.onset_time_s == pytest.approx(
            experiment.test["time_s"][onset.onset_index]
        )
        assert onset.onset_disp_nm == pytest.approx(
            experiment.test["disp_nm"][onset.onset_index]
        )


def test_detect_onset_clears_existing_results_for_unselected_runs(
    base_study,
) -> None:
    once = base_study.detect_onset(include_disabled=True)
    disabled = once.disable_experiments(EXPERIMENT_A)
    analyzed = disabled.detect_onset(include_disabled=False)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].onset is None


def test_detect_force_peaks_skips_disabled_experiments_by_default(
    base_study,
) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A)

    analyzed = study.detect_force_peaks()
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert analyzed is not study
    assert by_stem[EXPERIMENT_A].force_peaks is None
    assert by_stem[EXPERIMENT_B].force_peaks is not None
    assert by_stem[EXPERIMENT_C].force_peaks is not None
    assert by_stem[EXPERIMENT_D].force_peaks is not None


def test_detect_force_peaks_can_include_disabled_experiments(
    base_study,
) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A)

    analyzed = study.detect_force_peaks(include_disabled=True)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert len(analyzed.experiments) == len(study.experiments)
    assert by_stem[EXPERIMENT_A].force_peaks is not None


def test_detect_force_peaks_attaches_unsuccessful_results(base_study) -> None:
    flat_test = SignalTable(
        columns={
            "time_s": np.arange(8, dtype=np.float64),
            "disp_nm": np.linspace(0.0, 7.0, 8, dtype=np.float64),
            "force_uN": np.zeros(8, dtype=np.float64),
        },
        point_count=8,
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = replace(base_study.experiments[0], test=flat_test)

    analyzed = Study(experiments=(experiment,)).detect_force_peaks()
    detected = analyzed.experiments[0].force_peaks

    assert detected is not None
    assert detected.success is False
    assert detected.reason == "no_force_peaks_detected"
    assert detected.peaks == ()


def test_detect_force_peaks_attaches_time_and_displacement_values(
    base_study,
) -> None:
    experiment = base_study.experiments[0]
    analyzed = Study(experiments=(experiment,)).detect_force_peaks(
        prominence=50.0,
        threshold=1.0,
    )
    detected = analyzed.experiments[0].force_peaks

    assert detected is not None
    if detected.success:
        for peak in detected.peaks:
            assert peak.time_s == pytest.approx(
                experiment.test["time_s"][peak.index]
            )
            assert peak.disp_nm == pytest.approx(
                experiment.test["disp_nm"][peak.index]
            )


def test_detect_force_peaks_clears_existing_results_for_unselected_runs(
    base_study,
) -> None:
    once = base_study.detect_force_peaks(include_disabled=True)
    disabled = once.disable_experiments(EXPERIMENT_A)
    analyzed = disabled.detect_force_peaks(include_disabled=False)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].force_peaks is None
