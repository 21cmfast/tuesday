"""The Ultimate EoR Simulation Data AnalYser (TUESDAY) package."""

__all__ = ["__version__", "core", "simulators"]
from . import core, simulators
try:
    from ._version import version as __version__
except ImportError:
    # fallback for docs build or source tree without generated _version.py
    from setuptools_scm import get_version
    __version__ = get_version(root='.', relative_to=__file__)
