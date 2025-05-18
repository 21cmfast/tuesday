===================================================
tuesday - The Ultimate EoR Simulation Data AnalYser
===================================================
A collection of lightcone postprocessing tools such as calculating power spectra, plotting, and making line intensity maps.

|PyPI| |Status| |License| |Version| |Python Version| |Docs| |Code Style| |Codecov|

.. |PyPI| image:: https://badgen.net/pypi/v/tuesday-eor/
   :target: https://pypi.org/project/tuesday-eor
   :alt: PyPI version
.. |Status| image:: https://badgen.net/github/status/tuesday.svg
    :target: https://pypi.org/project/tuesday-eor
    :alt: Status
.. |License| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License
.. |Version| image:: https://badgen.net/pypi/v/tuesday-eor/
    :target: https://pypi.org/project/tuesday-eor
    :alt: Version
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/tuesday-eor.svg
    :target: https://pypi.python.org/pypi/tuesday-eor/
    :alt: Python Version
.. |Docs| image:: https://readthedocs.org/projects/tuesday/badge/?version=latest
    :target: http://tuesday.readthedocs.io/?badge=latest
    :alt: Documentation Status
.. |Code Style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |Codecov| image:: https://codecov.io/gh/21cmfast/tuesday/branch/main/graph/badge.svg
    :target: https://app.codecov.io/gh/21cmfast/tuesday
    :alt: Code Coverage

Installation
============

``tuesday`` is available on PyPI and can be installed with standard
tools like ``pip`` or ``uv``::

    $ pip install tuesday-eor

or::

    $ uv pip install tuesday-eor

If you are developing ``tuesday``, we recommend using a virtual environment.
You can create a new environment with ``uv``::

    $ uv sync
    $ source .venv/bin/activate

Contribute
==========

``tuesday`` is meant to be the collection of all these useful functionalities of which everyone has their own implementation, such as the power spectrum calculation.
To contribute to ``tuesday``, first find where your code belongs:
if your code can be written in a simulator-independent manner (preferred), it goes into ``core``.
On the other hand, if it requires a simulator-dependent input, then it goes into ``simulators/your_simulator``.

To contribute, open a `pull request <https://github.com/21cmFAST/21cmEMU/pulls>`_ with your code including tests for all lines and docstrings for everything you add.
Please also add a notebook with a tutorial demonstrating the uses of your code as part of the documentation.

Documentation
=============

See the `documentation <https://tuesday.readthedocs.io/en/latest/>`_ for more information on how to use ``tuesday``.

Development
===========

If you are developing ``tuesday``, here are some basic steps to follow to get setup.

First create a development environment with ``uv``::

    $ uv sync --all-extras --dev


Then install ``pre-commit`` in your repo so that style checks can be done on the fly::

    $ pre-commit install


Make changes in a branch::

    $ git checkout -b my-new-feature

Make sure to run the tests::

    $ uv run pytest


If you add new dependencies, use ``uv`` to manage this::

    $ uv add my-new-dependency

If it is a development dependency, use the ``--dev`` flag::

    $ uv add my-new-dev-dependency --dev

When you are ready to submit your changes, open a pull request on GitHub.
