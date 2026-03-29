from pathlib import Path

import pytest

from nanodent import load_folder

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def base_study():
    return load_folder(DATA_DIR)
