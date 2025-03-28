====================================================
tuesday - The Ultimate EoR Simulation Data AnalYser
====================================================

|PyPI| |Status| |Python| |License| |RTD| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/tuesday-eor.svg
   :target: https://pypi.org/project/tuesday-eor/
.. |Status| image:: https://img.shields.io/pypi/status/tuesday-eor.svg
   :target: https://pypi.org/project/tuesday-eor/
.. |Python| image:: https://img.shields.io/pypi/pyversions/tuesday-eor.svg

.. |License| image:: https://img.shields.io/pypi/l/tuesday-eor.svg
    :target: https://github.com/21cmfast/tuesday/blob/main/LICENSE
.. |Tests| image:: https://github.com/21cmfast/tuesday/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/21cmfast/tuesday/actions/workflows/tests.yml
.. |Codecov| image:: https://codecov.io/gh/21cmfast/tuesday/branch/main/graph/badge.svg?token=yUOqyTlZ3z
    :target: https://codecov.io/gh/21cmfast/tuesday
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
.. |RTD| image:: https://readthedocs.org/projects/tuesday/badge/?version=latest
    :target: https://tuesday.readthedocs.io/en/latest/
    :alt: Documentation Status
    
A collection of lightcone postprocessing tools such as conversion to power spectrum and plotting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/tuesday-eor/)](https://pypi.org/project/tuesday-eor)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tuesday-eor.svg)](https://pypi.python.org/pypi/tuesday-eor/)
[![Documentation Status](https://readthedocs.org/projects/tuesday/badge/?version=latest)](http://tuesday.readthedocs.io/?badge=latest)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://codecov.io/gh/21cmfast/tuesday/branch/main/graph/badge.svg)](https://app.codecov.io/gh/21cmfast/tuesday)

Installation
============

`tuesday` is available on PyPI and can be installed with standard
tools like `pip` or `uv`:

.. code-block:: console

    $ pip install tuesday-eor

or

.. code-block:: console

    $ uv pip install tuesday-eor


If you are developing `tuesday`, we recommend using a virtual environment.
You can create a new environment with `uv`:
.. code-block:: console

    $ uv sync
    $ source .venv/bin/activate


Documentation
=============

Documentation at https://tuesday.readthedocs.io/en/latest/

## Development

If you are developing `tuesday`, here are some basic steps to follow to get setup.

First create a development environment with `uv`:

.. code-block:: console

    $ uv sync --all-extras --dev

Then install `pre-commit` in your repo so that style checks can be done on the fly:

.. code-block:: console

    $ pre-commit install

Make changes in a branch:

.. code-block:: console

    $ git checkout -b my-new-feature

Make sure to run the tests:

.. code-block:: console

    $ uv run pytest

If you add new dependencies, use `uv` to manage this:

.. code-block:: console

    $ uv add my-new-dependency


If it is a development dependency, use the `--dev` flag:

.. code-block:: console

    $ uv add my-new-dev-dependency --dev

When you are ready to submit your changes, open a pull request on GitHub.
