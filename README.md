# nanodent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/thomasisensee/nanodent/ci.yml?branch=main)](https://github.com/thomasisensee/nanodent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/thomasisensee/nanodent/branch/main/graph/badge.svg)](https://codecov.io/gh/thomasisensee/nanodent)

## Installation

The Python package `nanodent` can be installed from PyPI:

```
python -m pip install nanodent
```

## Development installation

If you want to contribute to the development of `nanodent`, we recommend the
following editable installation from this repository:

```
git clone https://github.com/thomasisensee/nanodent
cd nanodent
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Quick Start

`nanodent` loads `.hld` as the canonical source in v1, while keeping sibling
`.tdm` and `.tdx` file paths attached to each experiment for future extension.

```python
from nanodent import load_folder, plot_groups

study = load_folder("path/to/experiment-folder")
groups = study.group_by_time_gap()
summaries = study.describe_groups()

fig, ax = plot_groups(
    groups,
    x="disp_nm",
    y="force_uN",
    cmap="viridis",
    alignment={"method": "force_threshold", "force_threshold": 10.0},
)
```

The public API also exposes:

- `load_experiment(path) -> Experiment`
- `load_folder(path) -> Study`
- `Study.group_by_time_gap(...) -> list[ExperimentGroup]`
- `Study.describe_groups(...) -> list[dict[str, Any]]`
- `plot_groups(...) -> tuple[Figure, Axes]`

Signal processing stays NumPy-first and separate from the data objects:
`nanodent.savgol`, `nanodent.gradient`, `nanodent.align_curve`, and
`nanodent.curve_fit_model`.

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
