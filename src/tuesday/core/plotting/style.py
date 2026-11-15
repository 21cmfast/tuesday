"""The dark mode of tuesday's plots.

Every plotting function of tuesday takes ``dark_mode``. With ``dark_mode=True`` the
plot is drawn in matplotlib's "dark_background" style, and with ``False`` in the
current style (a white background unless it was changed). Left as ``None``, it follows
:func:`set_dark_mode`, which is off until it is set, so a single call makes all later
plots dark::

    from tuesday.core.plotting import set_dark_mode

    set_dark_mode(True)
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import rcParams

_SETTINGS = {"dark_mode": False}

# The choice of an enclosing plot, which plots drawn inside it follow.
_CURRENT: ContextVar[bool | None] = ContextVar("dark_mode", default=None)


def set_dark_mode(dark_mode: bool = True) -> None:
    """Draw the later plots of tuesday in dark mode, or not.

    Parameters
    ----------
    dark_mode : bool, optional
        Whether plots whose ``dark_mode`` is left as None are dark.
    """
    _SETTINGS["dark_mode"] = bool(dark_mode)


def get_dark_mode() -> bool:
    """Return whether plots whose ``dark_mode`` is left as None are dark."""
    return _SETTINGS["dark_mode"]


@contextmanager
def dark_mode_context(dark_mode: bool | None = None) -> Iterator[None]:
    """Draw in dark mode inside the context, if ``dark_mode`` says so.

    ``None`` follows the plot that encloses this one, if any, and otherwise
    :func:`set_dark_mode`. The style is restored afterwards, but not interactive mode
    if a backend turned it on while plotting (Jupyter's inline backend does, at the
    first figure of a session), so that later figures are still shown.

    Parameters
    ----------
    dark_mode : bool, optional
        Whether to draw in dark mode.
    """
    if dark_mode is None:
        enclosing = _CURRENT.get()
        dark_mode = get_dark_mode() if enclosing is None else enclosing
    token = _CURRENT.set(dark_mode)
    try:
        if not dark_mode:
            yield
            return
        with plt.style.context("dark_background"):
            try:
                yield
            finally:
                interactive = rcParams["interactive"]
        rcParams["interactive"] = interactive
    finally:
        _CURRENT.reset(token)


def with_dark_mode(func: Callable[..., Any]) -> Callable[..., Any]:
    """Add the keyword ``dark_mode`` to a plotting function.

    The function draws inside :func:`dark_mode_context`. Its docstring describes the
    keyword.
    """

    @functools.wraps(func)
    def wrapper(*args, dark_mode: bool | None = None, **kwargs):
        with dark_mode_context(dark_mode):
            return func(*args, **kwargs)

    signature = inspect.signature(func)
    keyword = inspect.Parameter(
        "dark_mode",
        inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=bool | None,
    )
    wrapper.__signature__ = signature.replace(
        parameters=[*signature.parameters.values(), keyword]
    )
    return wrapper
