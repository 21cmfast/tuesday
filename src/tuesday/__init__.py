"""The Ultimate EoR Simulation Data AnalYser (TUESDAY) package."""

__all__ = ["__version__", "core", "simulators"]
from . import core, simulators

try:
    from ._version import version as __version__
except ImportError:
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='../..', relative_to=__file__)
    except Exception:
        __version__ = "0.0.0+unknown"

