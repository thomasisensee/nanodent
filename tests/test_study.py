from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nanodent.models import (
    Experiment,
    ExperimentPaths,
    SignalTable,
    TipAreaFunction,
)
from nanodent.study import Study

EXPERIMENT_A = "experiment_a"
EXPERIMENT_B = "experiment_b"
EXPERIMENT_C = "experiment_c"
EXPERIMENT_D = "experiment_d"
PARSED_TIP_AREA_FUNCTION = TipAreaFunction(
    c0=24.5,
    c1=7749.44,
    c2=229988.0,
    c3=-2.17869e6,
    c4=2.49969e6,
    c5=0.0,
)


def _mean_timestamp(*timestamps: datetime) -> datetime:
    origin = timestamps[0]
    return origin + timedelta(
        seconds=float(
            np.mean(
                [
                    (timestamp - origin).total_seconds()
                    for timestamp in timestamps
                ]
            )
        )
    )


def _hertzian_test_section(*, baseline_offset_uN: float = 5.0) -> SignalTable:
    load_disp = np.linspace(0.0, 100.0, 101, dtype=np.float64)
    load_force = (
        0.5 * np.maximum(load_disp - 10.0, 0.0) ** 1.5 + baseline_offset_uN
    )
    unload_disp = np.linspace(100.0, 80.0, 41, dtype=np.float64)[1:]
    unload_force = np.linspace(
        float(load_force[-1]),
        baseline_offset_uN + 50.0,
        41,
        dtype=np.float64,
    )[1:]
    disp = np.concatenate([load_disp, unload_disp])
    force = np.concatenate([load_force, unload_force])
    return SignalTable(
        columns={
            "time_s": np.arange(len(disp), dtype=np.float64),
            "disp_nm": disp,
            "force_uN": force,
        },
        point_count=len(disp),
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )


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


def test_loaded_experiments_parse_tip_area_function(base_study) -> None:
    assert all(
        experiment.parsed_tip_area_function == PARSED_TIP_AREA_FUNCTION
        for experiment in base_study.experiments
    )
    assert all(
        experiment.tip_area_function is None
        for experiment in base_study.experiments
    )


def test_study_methods_preserve_study_tip_area_function(base_study) -> None:
    study = base_study.with_tip_area_function(TipAreaFunction(c0=31.0))

    updated = study.disable_experiments(EXPERIMENT_A).detect_onset()

    assert updated.tip_area_function == TipAreaFunction(c0=31.0)


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
    assert by_stem[EXPERIMENT_A].oliver_pharr.disp_correction_nm == (
        pytest.approx(by_stem[EXPERIMENT_A].onset.onset_disp_nm)
    )
    assert by_stem[EXPERIMENT_A].oliver_pharr.force_correction_uN == (
        pytest.approx(by_stem[EXPERIMENT_A].onset.baseline_offset_uN)
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


def test_analyze_oliver_pharr_uses_parsed_tip_area_function(
    base_study,
) -> None:
    analyzed = base_study.detect_onset().analyze_oliver_pharr(
        stems=EXPERIMENT_A
    )
    experiment = analyzed.experiments[0]
    result = experiment.oliver_pharr

    assert result is not None
    assert result.tip_area_function == PARSED_TIP_AREA_FUNCTION
    assert result.contact_depth_nm is not None
    assert result.contact_area_nm2 == pytest.approx(
        PARSED_TIP_AREA_FUNCTION.evaluate(result.contact_depth_nm)
    )


def test_analyze_oliver_pharr_uses_study_tip_area_function_fallback(
    base_study,
) -> None:
    experiment = replace(
        base_study.experiments[0],
        parsed_tip_area_function=None,
        tip_area_function=None,
    )
    study_tip_area_function = TipAreaFunction(c0=30.0, c1=100.0)
    study = Study(
        experiments=(experiment,),
        tip_area_function=study_tip_area_function,
    ).detect_onset()

    analyzed = study.analyze_oliver_pharr()
    result = analyzed.experiments[0].oliver_pharr

    assert result is not None
    assert result.tip_area_function == study_tip_area_function
    assert result.contact_depth_nm is not None
    assert result.contact_area_nm2 == pytest.approx(
        study_tip_area_function.evaluate(result.contact_depth_nm)
    )


def test_analyze_oliver_pharr_experiment_tip_area_function_overrides_study(
    base_study,
) -> None:
    experiment_tip_area_function = TipAreaFunction(
        c0=25.0, c1=7600, c2=210000, c3=-2.26e6, c4=2.4e6
    )
    study_tip_area_function = TipAreaFunction(
        c0=24.5, c1=7650, c2=220000, c3=-2.2e6, c4=2.3e6
    )
    experiment = replace(
        base_study.experiments[0],
        parsed_tip_area_function=None,
    ).with_tip_area_function(experiment_tip_area_function)
    study = (
        Study(
            experiments=(experiment,),
            tip_area_function=study_tip_area_function,
        )
        .detect_onset()
        .analyze_oliver_pharr()
    )

    result = study.experiments[0].oliver_pharr

    assert result is not None
    assert result.tip_area_function == experiment_tip_area_function
    assert result.contact_depth_nm is not None
    assert result.contact_area_nm2 == pytest.approx(
        experiment_tip_area_function.evaluate(result.contact_depth_nm)
    )


def test_analyze_oliver_pharr_uses_dense_fit_curve_by_default(
    base_study,
) -> None:
    analyzed = base_study.analyze_oliver_pharr()

    result = analyzed.experiments[0].oliver_pharr

    assert result is not None
    assert result.success is True
    assert len(result.x_fit) == 200
    assert len(result.y_fit) == 200


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


def test_analyze_hertzian_auto_detects_force_peaks_and_uses_onset(
    base_study,
) -> None:
    experiment = replace(
        base_study.experiments[0],
        test=_hertzian_test_section(),
    )
    study = Study(experiments=(experiment,)).detect_onset(
        baseline_points=10,
        k=1.0,
        consecutive=3,
    )

    analyzed = study.analyze_hertzian(peak_prominence=50.0)
    result = analyzed.experiments[0].hertzian

    assert analyzed.experiments[0].force_peaks is not None
    assert result is not None
    assert result.success is True
    assert result.stem == EXPERIMENT_A
    assert result.fit_end_index == (
        analyzed.experiments[0].force_peaks.peaks[0].index
    )
    assert result.initial_onset_disp_nm == pytest.approx(
        analyzed.experiments[0].onset.onset_disp_nm
    )
    assert result.force_correction_uN == pytest.approx(
        analyzed.experiments[0].onset.baseline_offset_uN
    )
    assert result.h_onset_nm == pytest.approx(10.0, abs=0.25)


def test_analyze_hertzian_skips_disabled_experiments_by_default(
    base_study,
) -> None:
    experiment = replace(
        base_study.experiments[0],
        test=_hertzian_test_section(),
    )
    study = Study(experiments=(experiment,)).disable_experiments(EXPERIMENT_A)

    skipped = study.analyze_hertzian(peak_prominence=50.0)
    included = study.analyze_hertzian(
        peak_prominence=50.0,
        include_disabled=True,
    )

    assert skipped.experiments[0].hertzian is None
    assert included.experiments[0].hertzian is not None


def test_analyze_hertzian_attaches_missing_force_peak_result(
    base_study,
) -> None:
    flat_test = SignalTable(
        columns={
            "time_s": np.arange(20, dtype=np.float64),
            "disp_nm": np.arange(20, dtype=np.float64),
            "force_uN": np.ones(20, dtype=np.float64),
        },
        point_count=20,
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = replace(base_study.experiments[0], test=flat_test)

    analyzed = Study(experiments=(experiment,)).analyze_hertzian()
    result = analyzed.experiments[0].hertzian

    assert result is not None
    assert result.success is False
    assert result.reason == "missing_force_peak"


def test_analyze_hertzian_requires_overwrite_to_replace_success(
    base_study,
) -> None:
    experiment = replace(
        base_study.experiments[0],
        test=_hertzian_test_section(),
    )
    once = Study(experiments=(experiment,)).analyze_hertzian(
        peak_prominence=50.0
    )

    with pytest.warns(UserWarning, match=EXPERIMENT_A):
        skipped = once.analyze_hertzian(
            stems=EXPERIMENT_A,
            fit_num_points=25,
        )
    overwritten = once.analyze_hertzian(
        stems=EXPERIMENT_A,
        fit_num_points=25,
        overwrite=True,
    )

    assert skipped.experiments[0].hertzian == once.experiments[0].hertzian
    assert len(overwritten.experiments[0].hertzian.x_fit) == 25


def test_recomputed_dependencies_clear_hertzian_results(base_study) -> None:
    experiment = replace(
        base_study.experiments[0],
        test=_hertzian_test_section(),
    )
    once = (
        Study(experiments=(experiment,))
        .detect_onset(baseline_points=10, k=1.0, consecutive=3)
        .analyze_hertzian(peak_prominence=50.0)
    )

    with pytest.warns(UserWarning, match="Hertzian"):
        onset_rerun = once.detect_onset(
            baseline_points=12,
            k=1.0,
            consecutive=3,
            overwrite=True,
        )
    with pytest.warns(UserWarning, match="Hertzian"):
        peak_rerun = once.detect_force_peaks(
            prominence=50.0,
            overwrite=True,
        )

    assert onset_rerun.experiments[0].hertzian is None
    assert peak_rerun.experiments[0].hertzian is None


def test_analyze_oliver_pharr_forwards_power_law_options(base_study) -> None:
    load_disp = np.linspace(0.0, 100.0, 101)
    load_force = 0.01 * load_disp**2
    unload_disp = np.linspace(100.0, 40.0, 61)[1:]
    unload_force = (100.0 / np.power(60.0, 1.5)) * np.power(
        unload_disp - 40.0, 1.5
    )
    test_section = SignalTable(
        columns={
            "time_s": np.arange(161, dtype=np.float64),
            "disp_nm": np.concatenate([load_disp, unload_disp]),
            "force_uN": np.concatenate([load_force, unload_force]),
        },
        point_count=161,
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = replace(base_study.experiments[0], test=test_section)

    analyzed = Study(experiments=(experiment,)).analyze_oliver_pharr(
        fit_model="power_law_full",
    )
    result = analyzed.experiments[0].oliver_pharr

    assert result is not None
    assert result.success is True
    assert result.fit_model == "power_law_full"
    assert result.power_law_hf_nm == pytest.approx(
        analyzed.experiments[0].unloading.end_disp_nm
    )


def test_analyze_oliver_pharr_uses_baseline_offset_without_successful_onset(
    base_study,
) -> None:
    disp = np.concatenate(
        [
            np.linspace(0.0, 100.0, 101),
            np.linspace(100.0, 80.0, 41)[1:],
        ]
    )
    force = np.concatenate(
        [
            0.01 * np.linspace(0.0, 100.0, 101) ** 2,
            5.0 * np.linspace(100.0, 80.0, 41)[1:] - 400.0,
        ]
    )
    force = force + 4.0
    test_section = SignalTable(
        columns={
            "time_s": np.arange(len(disp), dtype=np.float64),
            "disp_nm": disp,
            "force_uN": force,
        },
        point_count=len(disp),
        raw_columns=("Time_s", "Disp_nm", "Force_uN"),
    )
    experiment = replace(base_study.experiments[0], test=test_section)
    study = Study(experiments=(experiment,)).detect_onset(
        baseline_points=20,
        k=1000.0,
        consecutive=3,
    )

    analyzed = study.analyze_oliver_pharr()
    result = analyzed.experiments[0].oliver_pharr

    assert result is not None
    assert result.success is True
    assert study.experiments[0].onset.success is False
    assert result.force_correction_uN == pytest.approx(
        analyzed.experiments[0].onset.baseline_offset_uN
    )
    assert result.evaluation_force_uN == pytest.approx(
        float(np.max(force) - analyzed.experiments[0].onset.baseline_offset_uN)
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


def test_analyze_oliver_pharr_uses_experiment_unloading_curve(
    base_study,
    monkeypatch,
) -> None:
    original = Experiment.unloading_curve
    calls: list[tuple[str, str, str]] = []

    def wrapped(
        self: Experiment,
        *,
        x: str = "disp_nm",
        y: str = "force_uN",
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((self.stem, x, y))
        return original(self, x=x, y=y)

    monkeypatch.setattr(Experiment, "unloading_curve", wrapped)

    analyzed = base_study.detect_unloading().analyze_oliver_pharr(
        stems=EXPERIMENT_A
    )
    experiment = analyzed.experiments[0]

    assert calls == [(EXPERIMENT_A, "disp_nm", "force_uN")]
    assert experiment.oliver_pharr is not None
    assert experiment.unloading is not None
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


def test_scalar_series_returns_pop_in_load_rows(base_study) -> None:
    analyzed = base_study.detect_force_peaks()
    expected = [
        {
            "timestamp": experiment.timestamp,
            "stem": experiment.stem,
            "value": min(
                float(peak.force_uN) for peak in experiment.force_peaks.peaks
            ),
        }
        for experiment in analyzed.experiments
        if experiment.force_peaks is not None
        and experiment.force_peaks.success
        and len(experiment.force_peaks.peaks) >= 2
    ]

    rows = analyzed.scalar_series("pop_in_load")

    assert rows == expected


def test_scalar_series_can_keep_missing_values(base_study) -> None:
    rows = base_study.scalar_series("hardness", drop_missing=False)

    assert len(rows) == len(base_study.experiments)
    assert all(row["value"] is None for row in rows)


def test_scalar_series_pop_in_load_can_keep_missing_values(
    base_study,
) -> None:
    analyzed = base_study.detect_force_peaks()
    first, *rest = analyzed.experiments
    one_peak_result = replace(
        first.force_peaks,
        peaks=first.force_peaks.peaks[:1],
        peak_count=1,
    )
    partial = Study(
        experiments=(replace(first, force_peaks=one_peak_result), *rest)
    )

    rows = partial.scalar_series("pop_in_load", drop_missing=False)

    assert rows[0] == {
        "timestamp": first.timestamp,
        "stem": first.stem,
        "value": None,
    }
    assert [row["stem"] for row in rows[1:]] == [
        experiment.stem for experiment in rest
    ]
    assert all(row["value"] is not None for row in rows[1:])


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


def test_group_scalar_series_returns_grouped_rows(base_study) -> None:
    analyzed = (
        base_study.detect_onset().detect_force_peaks().analyze_oliver_pharr()
    )
    groups = analyzed.group_by_time_gap()
    expected_rows = []
    for group in groups:
        resolved = analyzed.resolve_group(group)
        values = np.asarray(
            [
                experiment.oliver_pharr.hardness_uN_per_nm2
                for experiment in resolved
                if experiment.oliver_pharr is not None
                and experiment.oliver_pharr.hardness_uN_per_nm2 is not None
            ],
            dtype=np.float64,
        )
        if len(values) == 0:
            continue
        valid_experiments = [
            experiment
            for experiment in resolved
            if experiment.oliver_pharr is not None
            and experiment.oliver_pharr.hardness_uN_per_nm2 is not None
        ]
        expected_rows.append(
            {
                "timestamp": _mean_timestamp(
                    *[experiment.timestamp for experiment in valid_experiments]
                ),
                "group_index": group.index,
                "stems": tuple(experiment.stem for experiment in resolved),
                "value": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "count": len(values),
            }
        )

    rows = analyzed.group_scalar_series("hardness")

    assert rows == expected_rows


def test_group_scalar_series_returns_grouped_pop_in_load_rows(
    base_study,
) -> None:
    analyzed = base_study.detect_force_peaks()
    groups = analyzed.group_by_time_gap()
    expected_rows = []
    for group in groups:
        resolved = analyzed.resolve_group(group)
        values = np.asarray(
            [
                min(
                    float(peak.force_uN)
                    for peak in experiment.force_peaks.peaks
                )
                for experiment in resolved
                if experiment.force_peaks is not None
                and experiment.force_peaks.success
                and len(experiment.force_peaks.peaks) >= 2
            ],
            dtype=np.float64,
        )
        if len(values) == 0:
            continue
        valid_experiments = [
            experiment
            for experiment in resolved
            if experiment.force_peaks is not None
            and experiment.force_peaks.success
            and len(experiment.force_peaks.peaks) >= 2
        ]
        expected_rows.append(
            {
                "timestamp": _mean_timestamp(
                    *[experiment.timestamp for experiment in valid_experiments]
                ),
                "group_index": group.index,
                "stems": tuple(experiment.stem for experiment in resolved),
                "value": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "count": len(values),
            }
        )

    rows = analyzed.group_scalar_series("pop_in_load")

    assert rows == expected_rows


def test_group_scalar_series_can_use_explicit_groups(base_study) -> None:
    analyzed = base_study.detect_onset()
    group = analyzed.group_by_time_gap()[0]

    rows = analyzed.group_scalar_series("onset_time", groups=[group])

    resolved = analyzed.resolve_group(group)
    expected_values = np.asarray(
        [experiment.onset.onset_time_s for experiment in resolved],
        dtype=np.float64,
    )
    assert rows == [
        {
            "timestamp": _mean_timestamp(
                *[experiment.timestamp for experiment in resolved]
            ),
            "group_index": group.index,
            "stems": group.stems,
            "value": float(np.mean(expected_values)),
            "std": float(np.std(expected_values, ddof=0)),
            "count": len(expected_values),
        }
    ]


def test_group_scalar_series_uses_available_values_within_group(
    base_study,
) -> None:
    analyzed = (
        base_study.detect_onset().detect_force_peaks().analyze_oliver_pharr()
    )
    first, second, third, fourth = analyzed.experiments
    partial = Study(
        experiments=(
            first,
            second,
            replace(third, oliver_pharr=None),
            fourth,
        )
    )
    rows = partial.group_scalar_series("hardness")

    row = next(
        row for row in rows if row["stems"] == (EXPERIMENT_C, EXPERIMENT_D)
    )

    assert row["count"] == 1
    assert row["timestamp"] == fourth.timestamp
    assert row["value"] == pytest.approx(
        fourth.oliver_pharr.hardness_uN_per_nm2
    )
    assert row["std"] == pytest.approx(0.0)


def test_group_scalar_series_pop_in_load_uses_available_values_within_group(
    base_study,
) -> None:
    analyzed = base_study.detect_force_peaks()
    first, second, third, fourth = analyzed.experiments
    one_peak_result = replace(
        third.force_peaks,
        peaks=third.force_peaks.peaks[:1],
        peak_count=1,
    )
    partial = Study(
        experiments=(
            first,
            second,
            replace(third, force_peaks=one_peak_result),
            fourth,
        )
    )
    group = partial.group_by_time_gap()[1]

    rows = partial.group_scalar_series("pop_in_load", groups=[group])

    assert rows == [
        {
            "timestamp": fourth.timestamp,
            "group_index": group.index,
            "stems": group.stems,
            "value": pytest.approx(
                min(float(peak.force_uN) for peak in fourth.force_peaks.peaks)
            ),
            "std": pytest.approx(0.0),
            "count": 1,
        }
    ]


def test_group_scalar_series_can_keep_empty_groups(base_study) -> None:
    groups = base_study.group_by_time_gap()

    rows = base_study.group_scalar_series("hardness", drop_missing=False)

    assert len(rows) == len(groups)
    assert rows[0] == {
        "timestamp": _mean_timestamp(
            base_study.experiments[0].timestamp,
            base_study.experiments[1].timestamp,
        ),
        "group_index": groups[0].index,
        "stems": groups[0].stems,
        "value": None,
        "std": None,
        "count": 0,
    }


def test_group_scalar_series_pop_in_load_can_keep_empty_groups(
    base_study,
) -> None:
    analyzed = base_study.detect_force_peaks()
    one_peak_experiments = []
    for experiment in analyzed.experiments:
        one_peak_experiments.append(
            replace(
                experiment,
                force_peaks=replace(
                    experiment.force_peaks,
                    peaks=experiment.force_peaks.peaks[:1],
                    peak_count=1,
                ),
            )
        )
    partial = Study(experiments=tuple(one_peak_experiments))
    groups = partial.group_by_time_gap()

    rows = partial.group_scalar_series("pop_in_load", drop_missing=False)

    assert len(rows) == len(groups)
    assert rows[0] == {
        "timestamp": _mean_timestamp(
            partial.experiments[0].timestamp,
            partial.experiments[1].timestamp,
        ),
        "group_index": groups[0].index,
        "stems": groups[0].stems,
        "value": None,
        "std": None,
        "count": 0,
    }


def test_group_scalar_series_respects_enabled_filter(base_study) -> None:
    study = base_study.disable_experiments(EXPERIMENT_B).detect_onset(
        include_disabled=True
    )
    groups = study.group_by_time_gap(include_disabled=True)

    enabled_rows = study.group_scalar_series("onset_disp")
    all_rows = study.group_scalar_series(
        "onset_disp",
        groups=groups,
        include_disabled=True,
    )

    assert [row["stems"] for row in enabled_rows] == [
        (EXPERIMENT_A,),
        (EXPERIMENT_C, EXPERIMENT_D),
    ]
    assert [row["stems"] for row in all_rows] == [
        (EXPERIMENT_A, EXPERIMENT_B),
        (EXPERIMENT_C, EXPERIMENT_D),
    ]


def test_group_scalar_series_rejects_unknown_metrics(base_study) -> None:
    with pytest.raises(ValueError, match="Unknown scalar metric"):
        base_study.group_scalar_series("not_a_metric")


def test_save_and_load_session_restores_results(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "study-session.pkl"
    experiment_tip_area_function = TipAreaFunction(c0=40.0, c1=5.0)
    study_tip_area_function = TipAreaFunction(c0=32.0)
    first, *rest = base_study.experiments
    saved = (
        Study(
            experiments=(
                first.with_tip_area_function(experiment_tip_area_function),
                *rest,
            ),
            tip_area_function=study_tip_area_function,
        )
        .disable_experiments(EXPERIMENT_A, reason="manual")
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
        .analyze_hertzian(include_disabled=True)
    )

    saved_path = saved.save_session(session_path)
    restored = base_study.load_session(saved_path)
    by_stem = {
        experiment.stem: experiment for experiment in restored.experiments
    }

    assert saved_path == session_path
    assert session_path.exists()
    assert restored.tip_area_function == study_tip_area_function
    assert by_stem[EXPERIMENT_A].enabled is False
    assert by_stem[EXPERIMENT_A].disabled_reason == "manual"
    assert (
        by_stem[EXPERIMENT_A].tip_area_function == experiment_tip_area_function
    )
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
    assert by_stem[EXPERIMENT_B].hertzian is not None


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
    saved = base_study.with_tip_area_function(
        TipAreaFunction(c0=28.0)
    ).detect_onset(
        baseline_points=7,
        include_disabled=True,
    )
    current_first = base_study.experiments[0].with_tip_area_function(
        TipAreaFunction(c0=29.0)
    )
    current = Study(
        experiments=(current_first, *base_study.experiments[1:]),
        tip_area_function=TipAreaFunction(c0=30.0),
    ).detect_onset(baseline_points=4, include_disabled=True)
    saved.save_session(session_path)

    restored = current.load_session(session_path, overwrite=True)

    assert restored.experiments[0].onset.baseline_points == 7
    assert restored.tip_area_function == TipAreaFunction(c0=28.0)
    assert restored.experiments[0].tip_area_function is None


def test_load_session_keeps_current_tip_area_functions_without_overwrite(
    base_study, tmp_path: Path
) -> None:
    session_path = tmp_path / "study-session.pkl"
    saved_first = base_study.experiments[0].with_tip_area_function(
        TipAreaFunction(c0=41.0)
    )
    saved = Study(
        experiments=(saved_first, *base_study.experiments[1:]),
        tip_area_function=TipAreaFunction(c0=33.0),
    )
    current_first = base_study.experiments[0].with_tip_area_function(
        TipAreaFunction(c0=42.0)
    )
    current = Study(
        experiments=(current_first, *base_study.experiments[1:]),
        tip_area_function=TipAreaFunction(c0=34.0),
    )
    saved.save_session(session_path)

    with pytest.warns(UserWarning) as warning_records:
        restored = current.load_session(session_path)

    messages = [str(record.message) for record in warning_records]
    assert any(
        "current study tip area function" in message for message in messages
    )
    assert any("current tip area function" in message for message in messages)
    assert restored.tip_area_function == TipAreaFunction(c0=34.0)
    assert restored.experiments[0].tip_area_function == TipAreaFunction(
        c0=42.0
    )


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
