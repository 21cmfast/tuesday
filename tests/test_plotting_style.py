"""Tests of the dark mode of the plots, in core/plotting/style.py."""

import inspect

import astropy.units as un
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import rcParams
from matplotlib.colors import to_rgba

from tuesday.core import (
    calculate_ps_coeval,
    get_dark_mode,
    plot_1d_power_spectrum_k,
    plot_coeval_slice,
    plot_lightcone_slice,
    plot_pdf,
    plot_power_spectrum,
    set_dark_mode,
)
from tuesday.core.plotting.style import dark_mode_context

BLACK = to_rgba("black")
WHITE = to_rgba("white")


@pytest.fixture(autouse=True)
def _reset():
    """Leave dark mode off and close the figures after each test."""
    yield
    set_dark_mode(False)
    plt.close("all")


@pytest.fixture(scope="module")
def box():
    return np.random.default_rng(3).random((20, 20, 30)) * un.mK


def _background(ax):
    return to_rgba(ax.figure.get_facecolor())


def test_default_is_light(box):
    """Plots have a white background unless asked otherwise."""
    assert not get_dark_mode()
    ax = plot_coeval_slice(box, 100 * un.Mpc)
    assert _background(ax) == WHITE


def test_dark_mode_keyword(box):
    """dark_mode=True gives one dark plot and restores the style."""
    facecolor = rcParams["figure.facecolor"]
    z = np.linspace(6, 8, 30)
    ax = plot_lightcone_slice(box, 100 * un.Mpc, z, dark_mode=True, cmap="viridis")
    assert _background(ax) == BLACK
    assert rcParams["figure.facecolor"] == facecolor
    ax = plot_lightcone_slice(box, 100 * un.Mpc, z, cmap="viridis")
    assert _background(ax) == WHITE


def test_set_dark_mode(box):
    """set_dark_mode makes later plots dark, and the keyword overrides it."""
    set_dark_mode(True)
    assert get_dark_mode()
    ax = plot_pdf(box, hist_kwargs={"bins": 10})
    assert _background(ax) == BLACK
    ax = plot_pdf(box, hist_kwargs={"bins": 10}, dark_mode=False)
    assert _background(ax) == WHITE
    set_dark_mode(False)
    ax = plot_pdf(box, hist_kwargs={"bins": 10})
    assert _background(ax) == WHITE


@pytest.mark.filterwarnings("ignore:One or more radial bins had no cells")
def test_nested_plots_follow_the_enclosing_choice(box):
    """A plot drawn by another follows the choice made for the outer one."""
    ps1d, _ = calculate_ps_coeval(
        box=box.value * un.dimensionless_unscaled,
        box_length=100 * un.Mpc,
        calc_2d=False,
        calc_1d=True,
    )
    set_dark_mode(True)
    ax = plot_power_spectrum(ps1d, dark_mode=False)
    assert _background(ax) == WHITE
    set_dark_mode(False)
    ax = plot_power_spectrum(ps1d, dark_mode=True)
    assert _background(ax) == BLACK


def test_signature_and_docstring():
    """The keyword is in the signature and described in the docstring."""
    for func in (plot_lightcone_slice, plot_1d_power_spectrum_k, plot_pdf):
        parameter = inspect.signature(func).parameters["dark_mode"]
        assert parameter.default is None
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert "dark_mode : bool, optional" in func.__doc__


def test_context_keeps_interactive_mode():
    """The style is restored, but not interactive mode if a backend turned it on."""
    was_interactive = matplotlib.is_interactive()
    matplotlib.interactive(False)
    try:
        with dark_mode_context(True):
            assert rcParams["figure.facecolor"] == "black"
            matplotlib.interactive(True)
        assert matplotlib.is_interactive()
        assert rcParams["figure.facecolor"] == "white"
    finally:
        matplotlib.interactive(was_interactive)
