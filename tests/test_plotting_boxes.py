"""Test cases for the core/plotting/sliceplots.py module."""

import astropy.units as un
import numpy as np
import pytest

from tuesday.core import (
    coeval2slice_x,
    coeval2slice_y,
    coeval2slice_z,
    lc2slice_x,
    lc2slice_y,
    plot_coeval_slice,
    plot_pdf,
    plot_redshift_slice,
)
from tuesday.core.plotting.sliceplots import _plot_slice


@pytest.fixture(scope="session")
def test_coeval():
    """Fixture to create a random coeval box."""
    rng = np.random.default_rng()
    return rng.random((100, 100, 100)) * un.mK


@pytest.fixture(scope="session")
def test_lc():
    """Fixture to create a random lightcone."""
    rng = np.random.default_rng()
    return rng.random((100, 100, 300)) * un.mK


@pytest.fixture(scope="session")
def test_redshifts():
    """Fixture to create an array of redshifts for test_lc."""
    return np.logspace(np.log10(5), np.log10(30), 300)


def test_coeval_slice(test_coeval):
    """Test the plot_coeval_slice function."""
    box_len = 300 * un.cm
    ax = plot_coeval_slice(
        test_coeval, box_len, title="tiny box", transform2slice=coeval2slice_z(idx=5)
    )
    assert ax.get_xlabel() == f"Distance [{box_len.unit:latex_inline}]"
    assert ax.get_ylabel() == f"Distance [{box_len.unit:latex_inline}]"
    assert ax.get_title() == "tiny box"

    ax = plot_coeval_slice(
        test_coeval,
        300 * un.cm,
        vmin=-0.5,
        vmax=0.5,
        logc=False,
        logx=True,
        logy=True,
        transform2slice=coeval2slice_y(idx=5),
        v_x=test_coeval[15, ...] * un.m / un.s,
        v_y=test_coeval[15, ...] * un.m / un.s,
        quiver_decimate_factor=3,
        quiver_label="Velocity [m/s]",
        quiver_label_kwargs={"color": "blue"},
        quiver_kwargs={"color": "blue", "scale": 0.1},
    )

    ax = plot_coeval_slice(
        test_coeval.value * un.mK**2,
        300 * un.cm,
        ax=ax,
        transform2slice=coeval2slice_x(idx=5),
        smooth=True,
        v_x=test_coeval[15, ...] * un.m / un.s,
        v_y=test_coeval[15, ...] * un.m / un.s,
        quiver_label=True,
    )
    ax = plot_coeval_slice(
        test_coeval.value * un.dimensionless_unscaled, 300 * un.cm, smooth=True
    )


def test_lightcone_slice(test_lc, test_redshifts):
    """Test the plot_lightcone_slice function."""
    box_len = 300 * un.cm
    ax = plot_redshift_slice(
        test_lc.value * un.dimensionless_unscaled,
        box_len,
        test_redshifts,
        title="tiny lightcone",
        transform2slice=lc2slice_y(idx=5),
    )
    assert ax.get_ylabel() == f"Distance [{box_len.unit:latex_inline}]"
    assert ax.get_xlabel() == "Redshift"
    assert ax.get_title() == "tiny lightcone"

    ax = plot_redshift_slice(
        test_lc.value * un.mK**2,
        box_len,
        test_redshifts,
        vmin=-0.5,
        vmax=0.5,
        logc=False,
        logx=True,
        logy=True,
        transform2slice=lc2slice_x(idx=5),
    )

    ax = plot_redshift_slice(
        test_lc,
        box_len,
        test_redshifts,
        transform2slice=lc2slice_y(idx=5),
        smooth=True,
    )
    ax = plot_redshift_slice(
        test_lc,
        box_len,
        test_redshifts,
        smooth=True,
    )


def test_pdf(test_coeval):
    """Test the plot_slice function."""

    ax = plot_pdf(
        test_coeval,
        title="tiny box",
        hist_kwargs={"bins": 50, "density": True, "histtype": "step"},
        smooth=True,
    )
    ax = plot_pdf(
        test_coeval.value * un.mK**2,
        ax=ax,
        logx=True,
        hist_kwargs={"bins": 50, "density": True, "histtype": "step", "color": "red"},
    )
    ax = plot_pdf(
        test_coeval.value * un.dimensionless_unscaled,
        ax=ax,
        hist_kwargs={"bins": 50, "density": True, "histtype": "step", "color": "red"},
    )


def test_plot_slice(test_coeval):
    """Test the plot_slice function."""
    box_len = 300 * un.cm
    ax = _plot_slice(
        1.0 + abs(test_coeval[:, :, 1].value) * un.dimensionless_unscaled,
        np.linspace(0, box_len.value, test_coeval.shape[0]) * un.cm,
        np.linspace(0, box_len.value, test_coeval.shape[1]) * un.Mpc,
        title="tiny box",
        log=[True, True, True],
    )

    assert ax.get_title() == "tiny box"

    ax = _plot_slice(
        test_coeval[:, :, 1],
        np.linspace(0, box_len.value, test_coeval.shape[0]) * un.cm,
        np.linspace(0, box_len.value, test_coeval.shape[1]) * un.Mpc,
        vmin=-0.5,
        vmax=0.5,
        log=[True, True, False],
    )

    ax = _plot_slice(
        test_coeval[:, :, 1],
        np.linspace(0, box_len.value, test_coeval.shape[0]) * un.cm,
        np.linspace(0, box_len.value, test_coeval.shape[1]) * un.Mpc,
    )
