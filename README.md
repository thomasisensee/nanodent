# Welcome to nanodent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/thomasisensee/nanodent/ci.yml?branch=main)](https://github.com/thomasisensee/nanodent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/thomasisensee/nanodent/branch/main/graph/badge.svg)](https://codecov.io/gh/thomasisensee/nanodent)

## Installation

The Python package `nanodent` can be installed from PyPI:

```
python -m pip install nanodent
```

## Development installation

If you want to contribute to the development of `nanodent`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/thomasisensee/nanodent
cd nanodent
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
