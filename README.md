# tuesday
A collection of lightcone postprocessing tools such as conversion to power spectrum and plotting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/tuesday/)](https://pypi.org/project/tuesday)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tuesday.svg)](https://pypi.python.org/pypi/tuesday/)
[![Documentation Status](https://readthedocs.org/projects/tuesday/badge/?version=latest)](http://tuesday.readthedocs.io/?badge=latest)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://codecov.io/gh/21cmfast/tuesday/branch/main/graph/badge.svg)](https://app.codecov.io/gh/21cmfast/tuesday)

## Installation

`tuesday` is available on PyPI and can be installed with standard
tools like `pip` or `uv`:

```bash
pip install tuesday
```

or

```bash
uv pip install tuesday
```

If you are developing `tuesday`, we recommend using a virtual environment.
You can create a new environment with `uv`:
```bash
uv sync
source .venv/bin/activate
```


## Documentation

Documentation at https://tuesday.readthedocs.io/en/latest/

## Development

If you are developing `tuesday`, here are some basic steps to follow to get setup.

First create a development environment with `uv`:

```bash
uv sync --all-extras --dev
```

Then install `pre-commit` in your repo so that style checks can be done on the fly:

```bash
pre-commit install
```

Make changes in a branch:

```bash
git checkout -b my-new-feature
```

Make sure to run the tests:

```bash
uv run pytest
```

If you add new dependencies, use `uv` to manage this:

```bash
uv add my-new-dependency
```

If it is a development dependency, use the `--dev` flag:

```bash
uv add my-new-dev-dependency --dev
```

When you are ready to submit your changes, open a pull request on GitHub.
