Contributor Guide
=================

Thank you for your interest in improving this project.
This project is open-source under the `MIT license <https://opensource.org/licenses/MIT>`_  and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

``tuesday`` is a collection of useful functionality for 21cm simulations, bringing together many otherwise separate implementations, such as power spectrum calculations.
To contribute to ``tuesday``, first find where your code belongs:
if your code can be written in a simulator-independent manner (preferred), it goes into ``core``.
On the other hand, if it requires a simulator-dependent input, then it goes into ``simulators/your_simulator``.

In some cases, multiple implementations of similar functionality might exist -- one general-purpose tool that can act on raw arrays,
and other higher-level implementations that act on bespoke simulator outputs, that call the more general methods under the hood.

To contribute, open a `pull request <https://github.com/21cmFAST/21cmEMU/pulls>`_ with your code including tests for all lines and docstrings for everything you add.
Please also add a notebook with a tutorial demonstrating the uses of your code as part of the documentation.

Here is a list of important resources for contributors:

- `Source Code <https://github.com/21cmfast/tuesday>`_
- `Documentation <https://tuesday.readthedocs.io/>`_
- `Issue Tracker <https://github.com/21cmfast/tuesday/issues>`_
- :ref:`CODE OF CONDUCT`


Development
-----------

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

How to report a bug
-------------------

Report bugs on the `issue tracker <https://github.com/21cmfast/tuesday/issues>`_.

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

How to request a feature
------------------------

Request features on the `issue tracker <https://github.com/21cmfast/tuesday/issues>`_.

How to set up your development environment
------------------------------------------


Set up the dev environment::

    $ pip install -e ".[dev]"


How to test the project
-----------------------


Unit tests are located in the _tests_ directory,
and are written using the `pytest <https://pytest.readthedocs.io/>`_ testing framework.

The full test suite can be ran with the command::

    $ pytest

in the root directory of the project.


How to submit changes
---------------------

Open a `pull request <https://github.com/21cmfast/tuesday/pulls>`_ to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though we can always iterate on this.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command::

    $ pre-commit intsall

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.
