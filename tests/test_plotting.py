import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology.units import littleh

from tuesday.core import calculate_ps_lc, plot_power_spectrum


@pytest.fixture
def ps():
    """Fixture to create a random power spectrum."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0]

    ps = calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        lc_redshifts=test_redshifts,
        box_length=200 * un.Mpc,
        ps_redshifts=zs,
        calc_2d=True,
        calc_1d=True,
        interp=True,
    )
    return ps


@pytest.fixture
def ps2():
    """Fixture to create a random power spectrum."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0]

    ps = calculate_ps_lc(
        test_lc * un.mK,
        lc_redshifts=test_redshifts,
        box_length=200 * un.Mpc,
        ps_redshifts=zs,
        calc_2d=True,
        calc_1d=False,
    )
    return ps


def test_1d_ps_plot(ps):
    """Test the 1d power spectrum plot."""

    plot_power_spectrum(ps.k, ps.ps_1d, smooth=True)

    fig, ax = plt.subplots()
    plot_power_spectrum(
        ps.k,
        ps.ps_1d,
        fig=fig,
        ax=ax,
        title="Test Title",
        label="foo",
        log=[False, False],
    )
    plot_power_spectrum(
        ps.k,
        ps.ps_1d,
        fig=fig,
        title="Test Title",
        label=["z=6", "z=10", "z=27"],
    )


def test_1d_ps_units(ps):
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(ps.k, ps.ps_1d * un.mK**2 * un.Mpc**2)  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(ps.k / un.Mpc**4, ps.ps_1d)  # Wrong units on k


def test_2d_ps_plot(ps2):
    """Test the 2d power spectrum plot."""
    ps = ps2
    fig, ax = plt.subplots()
    mask = np.isnan(np.nanmean(ps.ps_2d, axis=1))
    plot_power_spectrum(
        [ps.kperp[~mask], ps.kpar],
        ps.ps_2d[~mask],
        fig=fig,
        ax=ax,
        log=False,
        label=["foo"],
    )
    plot_power_spectrum(
        [ps.kperp[~mask], ps.kpar],
        ps.ps_2d[~mask],
        smooth=True,
        title="Test Title",
        label="foo",
        log=[True, True],
    )


def test_2d_ps_units(ps):
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps.kperp, ps.kpar],
            ps.ps_2d.value * un.Mpc,
        )  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps.kperp.value * un.mK, ps.kpar],
            ps.ps_2d,
        )  # Wrong units on k
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps.kperp, ps.kpar.value * un.mK],
            ps.ps_2d,
        )  # Wrong units on k
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps.kperp, ps.kpar, ps.kpar],
            ps.ps_2d,
        )  # Wrong ks


@pytest.mark.parametrize("unit", [un.Mpc, un.Mpc / littleh])
def test_ps_plot_units(unit):
    """Test the 2d power spectrum plot."""

    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0]

    test_ps = calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        lc_redshifts=test_redshifts,
        box_length=200 * un.Mpc,
        ps_redshifts=zs,
        calc_2d=True,
        calc_1d=True,
        interp=True,
    )
    plot_power_spectrum(
        [test_ps.kperp.value / unit, test_ps.kpar.value / unit],
        test_ps.ps_2d,
        log=[True, True, False],
        label=["foo"],
    )

    plot_power_spectrum(
        test_ps.k.value / unit,
        test_ps.ps_1d,
        title=["z=6", "z=10", "z=27"],
        log=False,
    )
