# nanodent

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/thomasisensee/nanodent/ci.yml?branch=main)](https://github.com/thomasisensee/nanodent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/thomasisensee/nanodent/graph/badge.svg?token=DRJB38CIZI)](https://codecov.io/github/thomasisensee/nanodent)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)

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
python -m pip install --editable .[dev,docs,lint,tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Quick Start

`nanodent` loads `.hld` as the canonical source.

```python
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from nanodent import (
    load_folder,
    plot_experiments,
    plot_group_timeline,
    save_experiment_plots,
)

study = load_folder("path/to/experiment-folder")
filtered_study = study.classify_quality()
filtered_study = filtered_study.analyze_oliver_pharr()
hardness_rows = filtered_study.scalar_series("hardness")
pop_in_rows = filtered_study.detect_force_peaks().scalar_series("pop_in_load")
manual_groups = filtered_study.group_by_datetime_ranges(
    [
        (
            datetime(2026, 3, 4, 13, 0, 0),
            datetime(2026, 3, 4, 15, 0, 0),
        ),
    ]
)

timeline_fig, timeline_ax = plot_group_timeline(
    filtered_study,
    max_gap=timedelta(minutes=30),
)

fig, ax = plt.subplots()
plot_experiments(
    ax,
    filtered_study,
    fit_kwargs={"color": "gray", "linestyle": "solid", "linewidth": 2},
    zero_onset=False,
    cmap="rainbow",
)

ax.set_xlabel("Displacement h / nm")
ax.set_ylabel("Force P / μN")

saved = save_experiment_plots(
    filtered_study, "plots/", zero_onset=False
)

filtered_study.save_session("analysis-session.pkl")
resumed_study = load_folder("path/to/experiment-folder").load_session(
    "analysis-session.pkl"
)
```

`Study.classify_quality()` keeps all experiments loaded but marks
heuristically bad runs as `enabled=False` with a short `disabled_reason`,
currently including `gradual_onset`, `flat_force`, and local-jump outliers
such as `outlier_disp` or `outlier_force`.
Grouping and plotting ignore disabled experiments by default; group summaries
include them by default so quality decisions stay visible. Pass
`include_disabled=True` when you want disabled runs included in plots or
grouping output. `plot_experiments(...)` and `save_experiment_plots(...)`
always visualize the test force-displacement curve.

The public API also exposes:

- `load_experiment(path) -> Experiment`
- `load_folder(path) -> Study`
- `Study.analyze_oliver_pharr(...) -> Study`
- `Study.scalar_series(...) -> list[dict[str, Any]]`
- `Study.save_session(path) -> Path`
- `Study.load_session(path) -> Study`
- `Study.group_by_datetime_ranges(...) -> list[ExperimentGroup]`
- `Study.group_by_time_gap(...) -> list[ExperimentGroup]`
- `plot_group_timeline(...) -> tuple[Figure, Axes]`
- `plot_experiments(...) -> Axes`
- `save_experiment_plots(...) -> list[Path]`


## Demo notebook using methods provided by py4dgeo
Use the [demo notebook](https://github.com/thomasisensee/nanodent/blob/main/notebooks/demo.ipynb) to test the functionality of `nanodent` and see how it can be used to analyze and visualize nanoindentation experiments.

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
