"""A sub-package for adding simulator-specific functionality and convenience methods."""

try:
    from . import py21cmfast as py21cmfast

    HAVE_21CMFAST = True
except ImportError:
    HAVE_21CMFAST = False
