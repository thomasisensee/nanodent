from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nanodent import load_experiment, load_folder
from nanodent.io import _normalize_column_name
from nanodent.models import Experiment

DATA_DIR = Path(__file__).parent / "data"


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
    assert experiment.onset is None
    assert experiment.force_peaks is None
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
