from datetime import datetime
from pathlib import Path

import pytest

from nanodent import load_experiment, load_folder
from nanodent.io import _normalize_column_name

DATA_DIR = Path(__file__).parent / "data"


def test_load_experiment_parses_sections_and_metadata() -> None:
    experiment = load_experiment(
        DATA_DIR / "Tritium_Retention_Study_04.03.2026_0000.hld"
    )

    assert experiment.stem == "Tritium_Retention_Study_04.03.2026_0000"
    assert experiment.timestamp == datetime(2026, 3, 4, 13, 56, 27)
    assert experiment.temperature_c == pytest.approx(23.0)
    assert experiment.humidity_percent == pytest.approx(50.0)
    assert experiment.paths.tdm_path is not None
    assert experiment.paths.tdx_path is not None
    assert len(experiment.approach) == 2453
    assert len(experiment.drift) == 2181
    assert len(experiment.test) == 2800
    assert experiment.test.column_names[:3] == (
        "time_s",
        "disp_nm",
        "force_uN",
    )
    assert experiment.test["disp_nm"].dtype.name == "float64"
    assert experiment.test["force_uN"].dtype.name == "float64"
    assert len(experiment.segment_definitions) == 5
    assert experiment.segment_definitions[2].points == 997


def test_normalize_column_name_handles_micro_variants() -> None:
    assert _normalize_column_name("Force_µN") == "force_uN"
    assert _normalize_column_name("Force_�N") == "force_uN"
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
        "Tritium_Retention_Study_04.03.2026_0000",
        "Tritium_Retention_Study_04.03.2026_0001",
        "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0059",
        "Tritium_Retention_Study_11.03.2026_WED_oneweekafter_0060",
    ]
    assert all(
        experiment.paths.tdm_path is not None
        for experiment in study.experiments
    )
    assert all(
        experiment.paths.tdx_path is not None
        for experiment in study.experiments
    )


def test_load_folder_keeps_hld_only_experiments(tmp_path: Path) -> None:
    source = DATA_DIR / "Tritium_Retention_Study_04.03.2026_0000.hld"
    target = tmp_path / source.name
    target.write_text(
        source.read_text(encoding="iso-8859-1"), encoding="iso-8859-1"
    )

    study = load_folder(tmp_path)

    assert len(study.experiments) == 1
    experiment = study.experiments[0]
    assert experiment.paths.tdm_path is None
    assert experiment.paths.tdx_path is None
