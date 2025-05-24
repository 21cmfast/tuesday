===================================================
tuesday - The Ultimate EoR Simulation Data AnalYser
===================================================
A collection of lightcone postprocessing tools such as calculating power spectra, plotting, and making line intensity maps.

|PyPI| |Status| |License| |Version| |Python Version| |Docs| |Code Style| |Codecov|

.. |PyPI| image:: https://badgen.net/pypi/v/tuesday-eor/
   :target: https://pypi.org/project/tuesday-eor
   :alt: PyPI version
.. |Status| image:: https://img.shields.io/pypi/status/tuesday.svg
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
.. |Code Style| image:: https://img.shields.io/badge/code%20style-ruff-red.svg
    :target: https://github.com/astral-sh/ruff
.. |Codecov| image:: https://codecov.io/gh/21cmfast/tuesday/branch/main/graph/badge.svg
    :target: https://app.codecov.io/gh/21cmfast/tuesday
    :alt: Code Coverage

Installation
------------
``tuesday`` is available on PyPI and can be installed with standard
tools like ``pip`` or ``uv``::

    $ pip install tuesday-eor

or::

    $ uv pip install tuesday-eor

If you are developing ``tuesday``, we recommend using a virtual environment.
You can create a new environment with ``uv``::

    $ uv venv --python=3.12 [env-name]
    $ source .venv/bin/activate
    $ uv pip install -e ".[dev]"

Then use::
    $ deactivate

to exit the virtual environment.

Documentation
-------------

See the `documentation <https://tuesday.readthedocs.io/en/latest/>`_ for more information on how to use ``tuesday``.

Contribute
----------

Contributions are very welcome.
To learn more, see the `Contributor Guide <https://github.com/21cmfast/tuesday/blob/main/CONTRIBUTING.rst>`_.
