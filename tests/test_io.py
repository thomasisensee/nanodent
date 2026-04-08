from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nanodent import load_experiment, load_folder
from nanodent.analysis.unloading import detect_unloading
from nanodent.io import _normalize_column_name
from nanodent.models import Experiment, TipAreaFunction

DATA_DIR = Path(__file__).parent / "data"


def _write_simple_hld(
    file_path: Path,
    *,
    timestamp: str = "Wed Mar 04 13:56:27 2026",
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "\n".join(
            [
                "File Version: Demo",
                "Test Type: Indentation",
                f"Time Stamp: {timestamp}",
                "Test Data Points: 2",
                "Time_s\tDisp_nm\tForce_uN",
                "0.0\t0.0\t0.0",
                "1.0\t2.0\t3.0",
            ]
        ),
        encoding="iso-8859-1",
    )


def test_load_experiment_parses_sections_and_metadata() -> None:
    experiment = load_experiment(DATA_DIR / "experiment_a.hld")

    assert experiment.stem == "experiment_a"
    assert experiment.timestamp == datetime(2026, 3, 4, 13, 56, 27)
    assert experiment.temperature_c is None
    assert experiment.humidity_percent is None
    assert experiment.paths.tdm_path is None
    assert experiment.paths.tdx_path is None
    assert experiment.approach is None
    assert experiment.drift is None
    assert len(experiment.test) == 2800
    assert experiment.test.column_names[:3] == (
        "time_s",
        "disp_nm",
        "force_uN",
    )
    assert experiment.test["disp_nm"].dtype.name == "float64"
    assert experiment.test["force_uN"].dtype.name == "float64"
    assert len(experiment.segment_definitions) == 0
    assert experiment.enabled is True
    assert experiment.disabled_reason is None
    assert experiment.parsed_tip_area_function == TipAreaFunction(
        c0=24.5,
        c1=7749.44,
        c2=229988.0,
        c3=-2.17869e6,
        c4=2.49969e6,
        c5=0.0,
    )
    assert experiment.tip_area_function is None
    assert experiment.onset is None
    assert experiment.force_peaks is None
    assert experiment.unloading is None
    assert experiment.oliver_pharr is None


def test_normalize_column_name_handles_micro_variants() -> None:
    assert _normalize_column_name("Force_µN") == "force_uN"
    assert _normalize_column_name("Force_�N") == "force_uN"
    assert _normalize_column_name("Force_uN") == "force_uN"
    assert _normalize_column_name("Piezo Extension_nm") == "piezo_extension_nm"


def test_load_experiment_accepts_iso_8859_1_micro_symbol(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "synthetic.hld"
    file_path.write_text(
        "\n".join(
            [
                "File Version: Demo",
                "Test Type: Indentation",
                "Time Stamp: Wed Mar 04 13:56:27 2026",
                "Test Temp: 23 deg C",
                "Test Humidity: 50 %",
                "Sample Approach Data Points: 1",
                "Time_s\tDisp_nm",
                "0.0\t0.0",
                "Drift Measurement Data Points: 1",
                "Time_s\tDisp_nm",
                "0.0\t0.0",
                "Test Data Points: 2",
                "Time_s\tDisp_nm\tForce_µN",
                "0.0\t0.0\t0.0",
                "1.0\t2.0\t3.0",
            ]
        ),
        encoding="iso-8859-1",
    )

    experiment = load_experiment(file_path)

    assert experiment.test.column_names == ("time_s", "disp_nm", "force_uN")
    assert experiment.test["force_uN"][1] == pytest.approx(3.0)


def test_load_experiment_supports_zero_point_optional_sections(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "zero_sections.hld"
    file_path.write_text(
        "\n".join(
            [
                "File Version: Demo",
                "Test Type: Indentation",
                "Time Stamp: Wed Mar 04 13:56:27 2026",
                "Test Temp: 23 deg C",
                "Test Humidity: 50 %",
                "Sample Approach Data Points: 0",
                "Time_s\tDisp_nm",
                "Drift Measurement Data Points: 0",
                "Time_s\tDisp_nm",
                "Test Data Points: 2",
                "Time_s\tDisp_nm\tForce_µN",
                "0.0\t0.0\t0.0",
                "1.0\t2.0\t3.0",
            ]
        ),
        encoding="iso-8859-1",
    )

    experiment = load_experiment(file_path)

    assert experiment.approach is not None
    assert experiment.drift is not None
    assert len(experiment.approach) == 0
    assert len(experiment.drift) == 0
    assert len(experiment.test) == 2


def test_load_experiment_keeps_tip_area_function_optional(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "no_tip_area.hld"
    file_path.write_text(
        "\n".join(
            [
                "File Version: Demo",
                "Test Type: Indentation",
                "Time Stamp: Wed Mar 04 13:56:27 2026",
                "Test Data Points: 2",
                "Time_s\tDisp_nm\tForce_uN",
                "0.0\t0.0\t0.0",
                "1.0\t2.0\t3.0",
            ]
        ),
        encoding="iso-8859-1",
    )

    experiment = load_experiment(file_path)

    assert experiment.parsed_tip_area_function is None
    assert experiment.tip_area_function is None


def test_load_experiment_converts_supported_source_units(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "converted_units.hld"
    file_path.write_text(
        "\n".join(
            [
                "File Version: Demo",
                "Test Type: Indentation",
                "Time Stamp: Wed Mar 04 13:56:27 2026",
                "Test Data Points: 2",
                "Time_ms\tDisp_um\tForce_mN",
                "0.0\t0.0\t0.0",
                "1000.0\t2.0\t0.003",
            ]
        ),
        encoding="iso-8859-1",
    )

    experiment = load_experiment(file_path)

    assert experiment.test.column_names == ("time_s", "disp_nm", "force_uN")
    assert experiment.test["time_s"][1] == pytest.approx(1.0)
    assert experiment.test["disp_nm"][1] == pytest.approx(2000.0)
    assert experiment.test["force_uN"][1] == pytest.approx(3.0)


def test_from_measurements_converts_units_to_canonical_trace() -> None:
    experiment = Experiment.from_measurements(
        stem="synthetic",
        timestamp=datetime(2026, 3, 4, 13, 56, 27),
        time=np.array([0.0, 250.0, 1000.0], dtype=np.float64),
        displacement=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        force=np.array([0.0, 0.001, 0.002], dtype=np.float64),
        time_unit="ms",
        displacement_unit="um",
        force_unit="mN",
    )

    assert experiment.stem == "synthetic"
    assert experiment.paths is None
    assert experiment.trace.column_names == ("time_s", "disp_nm", "force_uN")
    assert experiment.trace["time_s"][-1] == pytest.approx(1.0)
    assert experiment.trace["disp_nm"][-1] == pytest.approx(2000.0)
    assert experiment.trace["force_uN"][-1] == pytest.approx(2.0)


def test_from_tabular_data_accepts_mapping_like_tables() -> None:
    experiment = Experiment.from_tabular_data(
        {
            "elapsed_ms": [0.0, 500.0],
            "depth_um": [0.0, 1.5],
            "load_mN": [0.0, 0.004],
        },
        stem="table_based",
        timestamp=datetime(2026, 3, 4, 13, 56, 27),
        time_column="elapsed_ms",
        displacement_column="depth_um",
        force_column="load_mN",
        time_unit="ms",
        displacement_unit="um",
        force_unit="mN",
    )

    assert experiment.trace["time_s"][1] == pytest.approx(0.5)
    assert experiment.trace["disp_nm"][1] == pytest.approx(1500.0)
    assert experiment.trace["force_uN"][1] == pytest.approx(4.0)


def test_experiment_unloading_curve_returns_trace_slice() -> None:
    experiment = Experiment.from_measurements(
        stem="synthetic",
        timestamp=datetime(2026, 3, 4, 13, 56, 27),
        time=np.arange(6, dtype=np.float64),
        displacement=np.array(
            [0.0, 1.0, 2.0, 3.0, 2.5, 2.0], dtype=np.float64
        ),
        force=np.array([0.0, 1.0, 3.0, 5.0, 4.0, 2.0], dtype=np.float64),
    ).with_unloading(
        detect_unloading(
            np.array([0.0, 1.0, 3.0, 5.0, 4.0, 2.0], dtype=np.float64),
            time_s=np.arange(6, dtype=np.float64),
            disp_nm=np.array(
                [0.0, 1.0, 2.0, 3.0, 2.5, 2.0],
                dtype=np.float64,
            ),
        )
    )

    x_values, y_values = experiment.unloading_curve()

    assert np.array_equal(
        x_values,
        np.array([3.0, 2.5, 2.0], dtype=np.float64),
    )
    assert np.array_equal(
        y_values,
        np.array([5.0, 4.0, 2.0], dtype=np.float64),
    )


def test_experiment_unloading_curve_requires_successful_unloading() -> None:
    experiment = Experiment.from_measurements(
        stem="synthetic",
        timestamp=datetime(2026, 3, 4, 13, 56, 27),
        time=np.arange(3, dtype=np.float64),
        displacement=np.array([0.0, 1.0, 0.5], dtype=np.float64),
        force=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="successful unloading"):
        experiment.unloading_curve()


def test_load_experiment_reports_file_path_on_parse_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "broken.hld"
    file_path.write_text(
        "\n".join(
            [
                "File Version: Demo",
                "Test Type: Indentation",
                "Time Stamp: Wed Mar 04 13:56:27 2026",
                "Test Data Points: 1",
                "Time_s\tDisp_nm\tForce_µN",
                "0.0\t0.0",
            ]
        ),
        encoding="iso-8859-1",
    )

    with pytest.raises(ValueError, match="broken.hld"):
        load_experiment(file_path)


def test_load_folder_discovers_siblings_and_sorts_experiments() -> None:
    study = load_folder(DATA_DIR)

    assert len(study.experiments) == 4
    assert [experiment.stem for experiment in study.experiments] == [
        "experiment_a",
        "experiment_b",
        "experiment_c",
        "experiment_d",
    ]
    assert [
        experiment.timestamp for experiment in study.experiments
    ] == sorted(experiment.timestamp for experiment in study.experiments)
    assert all(
        experiment.paths.tdm_path is None for experiment in study.experiments
    )
    assert all(
        experiment.paths.tdx_path is None for experiment in study.experiments
    )


def test_load_folder_ignores_subdirectories_by_default(
    tmp_path: Path,
) -> None:
    _write_simple_hld(tmp_path / "top_level.hld")
    _write_simple_hld(
        tmp_path / "nested" / "nested_only.hld",
        timestamp="Wed Mar 04 14:56:27 2026",
    )

    study = load_folder(tmp_path)

    assert [experiment.stem for experiment in study.experiments] == [
        "top_level"
    ]


def test_load_folder_can_scan_subdirectories_recursively(
    tmp_path: Path,
) -> None:
    _write_simple_hld(tmp_path / "top_level.hld")
    _write_simple_hld(
        tmp_path / "nested" / "nested_only.hld",
        timestamp="Wed Mar 04 14:56:27 2026",
    )

    study = load_folder(tmp_path, recursive=True)

    assert [experiment.stem for experiment in study.experiments] == [
        "top_level",
        "nested_only",
    ]
