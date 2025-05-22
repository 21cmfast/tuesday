import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology.units import littleh

from tuesday.core import (
    CylindricalPS,
    SphericalPS,
    calculate_ps_lc,
    plot_power_spectrum,
)


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
    return ps["ps_1d"]["z = 6.0"]


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
    return ps["ps_2d"]["z = 6.0"]


def test_1d_ps_plot(ps):
    """Test the 1d power spectrum plot."""

    plot_power_spectrum(ps, smooth=True)

    _, ax = plt.subplots()
    plot_power_spectrum(
        ps,
        ax=ax,
        title="Test Title",
        legend="foo",
        logx=False,
        logy=False,
        smooth=True,
    )
    plot_power_spectrum(
        ps,
        title="Test Title",
        legend="z=6",
    )
    with np.testing.assert_raises(ValueError):
        bad_ps = SphericalPS(
            np.append(ps.ps[None, ...], ps.ps[None, ...], axis=0), k=ps.k
        )
        plot_power_spectrum(bad_ps)  # Passing 2 1D PS


def test_bad_1d_ps_units(ps):
    with np.testing.assert_raises(ValueError):
        bad_ps = SphericalPS(ps.ps * un.mK**2 * un.Mpc**2, k=ps.k)
        plot_power_spectrum(bad_ps)  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        bad_ps = SphericalPS(ps.ps, k=ps.k / un.Mpc**4)
        plot_power_spectrum(bad_ps)  # Wrong units on k


@pytest.mark.parametrize("unit", [un.Mpc, un.Mpc / littleh])
def test_good_1d_ps_units(ps, unit):
    good_ps = SphericalPS(ps.ps.value * un.mK**2 * unit**3, k=ps.k, is_deltasq=False)
    plot_power_spectrum(good_ps)
    good_ps = SphericalPS(ps.ps.value * unit**3, k=ps.k, is_deltasq=False)
    plot_power_spectrum(good_ps)


def test_2d_ps_plot(ps2):
    """Test the 2d power spectrum plot."""
    fig, ax = plt.subplots()
    plot_power_spectrum(
        ps2,
        ax=ax,
        logx=False,
        legend=["foo"],
    )
    plot_power_spectrum(
        ps2,
        smooth=True,
        title="Test Title",
        legend="foo",
        logx=True,
    )


def test_2d_ps_units(ps2):
    with np.testing.assert_raises(ValueError):
        bad_ps = CylindricalPS(ps2.ps.value * un.Mpc, kperp=ps2.kperp, kpar=ps2.kpar)
        plot_power_spectrum(bad_ps)  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        bad_ps = CylindricalPS(ps2.ps, kperp=ps2.kperp * un.mK, kpar=ps2.kpar)
        plot_power_spectrum(bad_ps)  # Wrong units on k
    with np.testing.assert_raises(ValueError):
        bad_ps = CylindricalPS(ps2.ps, kperp=ps2.kperp, kpar=ps2.kpar * un.mK)
        plot_power_spectrum(bad_ps)  # Wrong units on k
