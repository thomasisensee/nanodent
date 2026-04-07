from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nanodent.models import ExperimentPaths, SignalTable
from nanodent.study import Study

EXPERIMENT_A = "experiment_a"
EXPERIMENT_B = "experiment_b"
EXPERIMENT_C = "experiment_c"
EXPERIMENT_D = "experiment_d"


def test_group_by_time_gap_creates_two_default_groups(base_study) -> None:
    groups = base_study.group_by_time_gap()

    assert [len(group.stems) for group in groups] == [2, 2]
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

    assert [len(group.stems) for group in groups] == [2, 1]


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


def test_resolve_group_uses_current_study_state(base_study) -> None:
    group = base_study.group_by_time_gap()[0]

    updated_study = base_study.detect_onset(stems=group.stems)
    resolved = updated_study.resolve_group(group)

    assert tuple(experiment.stem for experiment in resolved) == group.stems
    assert all(experiment.onset is not None for experiment in resolved)


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
    if by_stem[EXPERIMENT_A].oliver_pharr.hardness_success:
        assert (
            by_stem[EXPERIMENT_A].oliver_pharr.reduced_modulus_uN_per_nm2
            is not None
        )
    else:
        assert (
            by_stem[EXPERIMENT_A].oliver_pharr.reduced_modulus_uN_per_nm2
            is None
        )


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
    assert (
        by_stem[EXPERIMENT_A].oliver_pharr.reduced_modulus_uN_per_nm2 is None
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


def test_analyze_oliver_pharr_preserves_unselected_results(
    base_study,
) -> None:
    once = base_study.analyze_oliver_pharr(include_disabled=True)
    disabled = once.disable_experiments(EXPERIMENT_A)
    with pytest.warns(UserWarning, match="successful results already exist"):
        analyzed = disabled.analyze_oliver_pharr(include_disabled=False)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].oliver_pharr is not None


def test_analyze_oliver_pharr_requires_overwrite_to_replace_success(
    base_study,
) -> None:
    once = base_study.detect_onset().analyze_oliver_pharr()

    with pytest.warns(UserWarning, match=EXPERIMENT_A):
        rerun = once.analyze_oliver_pharr(stems=EXPERIMENT_A)

    assert (
        rerun.experiments[0].oliver_pharr == once.experiments[0].oliver_pharr
    )


def test_analyze_oliver_pharr_overwrite_recomputes_selected_experiments(
    base_study,
) -> None:
    once = base_study.detect_onset().analyze_oliver_pharr()
    rerun = once.analyze_oliver_pharr(
        stems=EXPERIMENT_A,
        epsilon=0.9,
        overwrite=True,
    )
    by_stem = {experiment.stem: experiment for experiment in rerun.experiments}

    assert by_stem[EXPERIMENT_A].oliver_pharr.epsilon == pytest.approx(0.9)
    assert (
        by_stem[EXPERIMENT_B].oliver_pharr == once.experiments[1].oliver_pharr
    )


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


def test_detect_onset_forwards_mode_and_baseline_window(base_study) -> None:
    test_section = SignalTable(
        columns={
            "time_s": np.arange(7, dtype=np.float64),
            "disp_nm": np.linspace(0.0, 6.0, 7, dtype=np.float64),
            "force_uN": np.array(
                [5.0, 5.0, 0.0, 0.0, 0.2, 0.7, 1.2],
                dtype=np.float64,
            ),
        },
        point_count=7,
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = replace(base_study.experiments[0], test=test_section)

    analyzed = Study(experiments=(experiment,)).detect_onset(
        mode="absolute",
        baseline_start_index=2,
        baseline_end_index=4,
        absolute_threshold_uN=0.5,
        consecutive=1,
    )
    onset = analyzed.experiments[0].onset

    assert onset is not None
    assert onset.mode == "absolute"
    assert onset.onset_index == 5
    assert onset.baseline_start_index == 2
    assert onset.baseline_end_index == 4
    assert onset.absolute_threshold_uN == pytest.approx(0.5)


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


def test_detect_onset_preserves_unselected_results(
    base_study,
) -> None:
    once = base_study.detect_onset(include_disabled=True)
    disabled = once.disable_experiments(EXPERIMENT_A)
    with pytest.warns(UserWarning, match="successful results already exist"):
        analyzed = disabled.detect_onset(include_disabled=False)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].onset is not None


def test_detect_onset_recomputes_unsuccessful_results_by_default(
    base_study,
) -> None:
    flat_test = SignalTable(
        columns={
            "time_s": np.arange(8, dtype=np.float64),
            "disp_nm": np.linspace(0.0, 7.0, 8, dtype=np.float64),
            "force_uN": np.zeros(8, dtype=np.float64),
        },
        point_count=8,
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    detected = Study(
        experiments=(replace(base_study.experiments[0], test=flat_test),)
    ).detect_onset(
        baseline_points=4,
        k=2.0,
        consecutive=3,
    )
    rerun = detected.detect_onset(
        baseline_points=5,
        k=2.0,
        consecutive=3,
    )

    assert rerun.experiments[0].onset is not None
    assert rerun.experiments[0].onset.success is False
    assert rerun.experiments[0].onset.baseline_points == 5


def test_detect_onset_overwrite_clears_dependent_oliver_pharr(
    base_study,
) -> None:
    analyzed = base_study.detect_onset().analyze_oliver_pharr()

    with pytest.warns(UserWarning, match="Cleared Oliver-Pharr"):
        rerun = analyzed.detect_onset(stems=EXPERIMENT_A, overwrite=True)

    by_stem = {experiment.stem: experiment for experiment in rerun.experiments}
    assert by_stem[EXPERIMENT_A].onset is not None
    assert by_stem[EXPERIMENT_A].oliver_pharr is None
    assert by_stem[EXPERIMENT_B].oliver_pharr is not None


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


def test_detect_force_peaks_preserves_unselected_results(
    base_study,
) -> None:
    once = base_study.detect_force_peaks(include_disabled=True)
    disabled = once.disable_experiments(EXPERIMENT_A)
    with pytest.warns(UserWarning, match="successful results already exist"):
        analyzed = disabled.detect_force_peaks(include_disabled=False)
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].force_peaks is not None


def test_detect_unloading_attaches_results(base_study) -> None:
    analyzed = base_study.detect_unloading()
    detected = analyzed.experiments[0].unloading

    assert detected is not None
    assert detected.success is True
    assert detected.start_index is not None
    assert detected.start_force_uN == pytest.approx(
        np.max(base_study.experiments[0].test["force_uN"])
    )


def test_detect_unloading_skips_disabled_experiments_by_default(
    base_study,
) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A)

    analyzed = study.detect_unloading()
    by_stem = {
        experiment.stem: experiment for experiment in analyzed.experiments
    }

    assert analyzed is not study
    assert by_stem[EXPERIMENT_A].unloading is None
    assert by_stem[EXPERIMENT_B].unloading is not None


def test_detect_unloading_overwrite_clears_dependent_oliver_pharr(
    base_study,
) -> None:
    analyzed = base_study.detect_unloading().analyze_oliver_pharr()

    with pytest.warns(UserWarning, match="Cleared Oliver-Pharr"):
        rerun = analyzed.detect_unloading(stems=EXPERIMENT_A, overwrite=True)

    by_stem = {experiment.stem: experiment for experiment in rerun.experiments}
    assert by_stem[EXPERIMENT_A].unloading is not None
    assert by_stem[EXPERIMENT_A].oliver_pharr is None
    assert by_stem[EXPERIMENT_B].oliver_pharr is not None


def test_analyze_oliver_pharr_auto_attaches_unloading_when_missing(
    base_study,
) -> None:
    analyzed = base_study.analyze_oliver_pharr()
    experiment = analyzed.experiments[0]

    assert experiment.unloading is not None
    assert experiment.unloading.success is True
    assert experiment.oliver_pharr is not None
    assert experiment.oliver_pharr.unloading_start_index == (
        experiment.unloading.start_index
    )


def test_scalar_series_returns_timestamped_rows(base_study) -> None:
    analyzed = (
        base_study.detect_onset().detect_force_peaks().analyze_oliver_pharr()
    )
    expected = [
        {
            "timestamp": experiment.timestamp,
            "stem": experiment.stem,
            "value": experiment.oliver_pharr.hardness_uN_per_nm2,
        }
        for experiment in analyzed.experiments
        if experiment.oliver_pharr is not None
        and experiment.oliver_pharr.hardness_uN_per_nm2 is not None
    ]

    rows = analyzed.scalar_series("hardness")

    assert rows == expected


def test_scalar_series_can_keep_missing_values(base_study) -> None:
    rows = base_study.scalar_series("hardness", drop_missing=False)

    assert len(rows) == len(base_study.experiments)
    assert all(row["value"] is None for row in rows)


def test_scalar_series_respects_enabled_filter(base_study) -> None:
    study = base_study.disable_experiments(EXPERIMENT_A).detect_onset(
        include_disabled=True
    )

    enabled_rows = study.scalar_series("onset_disp")
    all_rows = study.scalar_series("onset_disp", include_disabled=True)

    assert [row["stem"] for row in enabled_rows] == [
        EXPERIMENT_B,
        EXPERIMENT_C,
        EXPERIMENT_D,
    ]
    assert [row["stem"] for row in all_rows] == [
        EXPERIMENT_A,
        EXPERIMENT_B,
        EXPERIMENT_C,
        EXPERIMENT_D,
    ]


def test_scalar_series_rejects_unknown_metrics(base_study) -> None:
    with pytest.raises(ValueError, match="Unknown scalar metric"):
        base_study.scalar_series("not_a_metric")


def test_save_and_load_session_restores_results(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "study-session.pkl"
    saved = (
        base_study.disable_experiments(EXPERIMENT_A, reason="manual")
        .detect_onset(
            mode="absolute",
            baseline_start_index=0,
            baseline_end_index=10,
            absolute_threshold_uN=1.0,
            include_disabled=True,
            smoothing={"window_length": 5, "polyorder": 1},
        )
        .detect_force_peaks(include_disabled=True)
        .analyze_oliver_pharr(include_disabled=True)
    )

    saved_path = saved.save_session(session_path)
    restored = base_study.load_session(saved_path)
    by_stem = {
        experiment.stem: experiment for experiment in restored.experiments
    }

    assert saved_path == session_path
    assert session_path.exists()
    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].disabled_reason == "manual"
    assert by_stem[EXPERIMENT_A].onset is not None
    assert dict(by_stem[EXPERIMENT_A].onset.smoothing or {}) == {
        "window_length": 5,
        "polyorder": 1,
    }
    assert by_stem[EXPERIMENT_A].onset.mode == "absolute"
    assert by_stem[EXPERIMENT_A].onset.baseline_start_index == 0
    assert by_stem[EXPERIMENT_A].onset.baseline_end_index == 10
    assert by_stem[EXPERIMENT_A].onset.absolute_threshold_uN == pytest.approx(
        1.0
    )
    assert by_stem[EXPERIMENT_A].onset.baseline_offset_uN is not None
    assert by_stem[EXPERIMENT_B].force_peaks is not None
    assert by_stem[EXPERIMENT_B].unloading is not None
    assert by_stem[EXPERIMENT_B].oliver_pharr is not None


def test_load_session_keeps_current_results_without_overwrite(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "study-session.pkl"
    saved = base_study.detect_onset(
        baseline_points=7,
        include_disabled=True,
    )
    current = base_study.detect_onset(
        baseline_points=4,
        include_disabled=True,
    )
    saved.save_session(session_path)

    with pytest.warns(UserWarning, match="Kept current onset"):
        restored = current.load_session(session_path)

    assert restored.experiments[0].onset.baseline_points == 4


def test_load_session_overwrite_applies_saved_results(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "study-session.pkl"
    saved = base_study.detect_onset(
        baseline_points=7,
        include_disabled=True,
    )
    current = base_study.detect_onset(
        baseline_points=4,
        include_disabled=True,
    )
    saved.save_session(session_path)

    restored = current.load_session(session_path, overwrite=True)

    assert restored.experiments[0].onset.baseline_points == 7


def test_load_session_warns_about_missing_saved_stems(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "study-session.pkl"
    saved = base_study.detect_onset(include_disabled=True)
    partial_study = Study(experiments=(base_study.experiments[0],))
    saved.save_session(session_path)

    with pytest.warns(
        UserWarning,
        match="current study does not contain those stems",
    ):
        restored = partial_study.load_session(session_path)

    assert len(restored.experiments) == 1
    assert restored.experiments[0].onset is not None


def test_load_session_warns_on_timestamp_and_filename_mismatch(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "study-session.pkl"
    saved = base_study.detect_onset(include_disabled=True)
    modified_first = replace(
        base_study.experiments[0],
        timestamp=datetime(2030, 1, 1, 0, 0, 0),
        paths=ExperimentPaths(
            stem=base_study.experiments[0].stem,
            hld_path=Path("renamed_file.hld"),
        ),
    )
    current = Study(experiments=(modified_first, *base_study.experiments[1:]))
    saved.save_session(session_path)

    with pytest.warns(UserWarning) as warning_records:
        current.load_session(session_path)

    messages = [str(record.message) for record in warning_records]
    assert any("timestamp mismatches" in message for message in messages)
    assert any("filename mismatches" in message for message in messages)


def test_save_and_load_session_supports_experiments_without_file_paths(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "generic-session.pkl"
    generic_experiment = replace(
        base_study.experiments[0],
        paths=None,
        source_path=None,
        source_format=None,
    )
    study = Study(experiments=(generic_experiment,)).detect_onset()

    study.save_session(session_path)
    restored = Study(experiments=(generic_experiment,)).load_session(
        session_path
    )

    assert restored.experiments[0].stem == generic_experiment.stem
    assert restored.experiments[0].onset is not None
