# nanodent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/thomasisensee/nanodent/ci.yml?branch=main)](https://github.com/thomasisensee/nanodent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/thomasisensee/nanodent/graph/badge.svg?token=DRJB38CIZI)](https://codecov.io/github/thomasisensee/nanodent)
![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)

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
from datetime import timedelta

from nanodent import (
    load_folder,
    plot_group_timeline,
    plot_groups,
    save_experiment_plots,
)

study = load_folder("path/to/experiment-folder")
filtered_study = study.classify_quality()
summaries = filtered_study.describe_groups()

timeline_fig, timeline_ax = plot_group_timeline(
    filtered_study,
    max_gap=timedelta(minutes=30),
)

groups = filtered_study.group_by_time_gap()
fig, axes = plot_groups(
    groups,
    x="disp_nm",
    y="force_uN",
    cmap="viridis",
    alignment={"method": "force_threshold", "force_threshold": 10.0},
    show_slope=True,
    xlim=(0.0, 1500.0),
)

saved = save_experiment_plots(filtered_study, "plots/")
```

`Study.classify_quality()` keeps all experiments loaded but marks
heuristically bad runs as `enabled=False` with a short `disabled_reason`,
currently including `gradual_onset`, `flat_force`, and local-jump outliers
such as `outlier_disp` or `outlier_force`.
Grouping and plotting ignore disabled experiments by default; group summaries
include them by default so quality decisions stay visible. Pass
`include_disabled=True` when you want disabled runs included in plots or
grouping output.

The public API also exposes:

- `load_experiment(path) -> Experiment`
- `load_folder(path) -> Study`
- `Study.group_by_time_gap(...) -> list[ExperimentGroup]`
- `Study.describe_groups(...) -> list[dict[str, Any]]`
- `plot_group_timeline(...) -> tuple[Figure, Axes]`
- `plot_groups(...) -> tuple[Figure, Axes | ndarray]`
- `save_experiment_plots(...) -> list[Path]`

Signal processing stays NumPy-first and separate from the data objects:
`nanodent.savgol`, `nanodent.gradient`, `nanodent.align_curve`, and
`nanodent.curve_fit_model`.

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
