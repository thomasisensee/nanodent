from pathlib import Path

import pytest

from nanodent import load_folder
from nanodent.study import Study

DATA_DIR = Path(__file__).parent / "data"
BASE_STEMS = {
    "experiment_a",
    "experiment_b",
    "experiment_c",
    "experiment_d",
}


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def base_study():
    study = load_folder(DATA_DIR)
    return Study(
        experiments=tuple(
            experiment
            for experiment in study.experiments
            if experiment.stem in BASE_STEMS
        )
    )
