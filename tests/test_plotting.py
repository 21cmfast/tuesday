import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology.units import littleh

from tuesday.core import calculate_ps, plot_power_spectrum

@pytest.fixture
def ps():
    """Fixture to create a random power spectrum."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0]

    ps = calculate_ps(
        test_lc * un.dimensionless_unscaled,
        test_redshifts,
        box_length=200 * un.Mpc,
        zs=zs,
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

    ps = calculate_ps(
        test_lc * un.mK,
        test_redshifts,
        box_length=200 * un.Mpc,
        zs=zs,
        calc_2d=True,
        calc_1d=False,
    )
    return ps

def test_1d_ps_plot(ps):
    """Test the 1D power spectrum plot."""

    plot_power_spectrum(ps["k"], ps["ps_1D"], smooth=True)

    fig, ax = plt.subplots()
    plot_power_spectrum(
        ps["k"],
        ps["ps_1D"],
        fig=fig,
        ax=ax,
        title="Test Title",
        label="foo",
        log=[False, False],
    )
    plot_power_spectrum(
        ps["k"],
        ps["ps_1D"][0],
        fig=fig,
        title="Test Title",
        label=["z=6", "z=10", "z=27"],
    )

def test_1d_ps_units(ps):
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            ps["k"], ps["ps_1D"] * un.mK**2 * un.Mpc**2
        )  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(ps["k"] / un.Mpc**4, ps["ps_1D"])  # Wrong units on k

def test_2d_ps_plot(ps2):
    """Test the 2D power spectrum plot."""
    ps = ps2
    fig, ax = plt.subplots()
    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"][0],
        fig=fig,
        ax=ax,
        log=False,
        label=["foo"],
    )
    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"],
        smooth=True,
        title="Test Title",
        label="foo",
        log=[True, True],
    )


def test_2d_ps_units(ps):
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps["final_kperp"], ps["final_kpar"]],
            ps["final_ps_2D"].value * un.Mpc,
        )  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps["final_kperp"].value * un.mK, ps["final_kpar"]],
            ps["final_ps_2D"],
        )  # Wrong units on k
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps["final_kperp"], ps["final_kpar"].value * un.mK],
            ps["final_ps_2D"],
        )  # Wrong units on k
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps["final_kperp"], ps["final_kpar"], ps["final_kpar"]],
            ps["final_ps_2D"],
        )  # Wrong ks
        
@pytest.mark.parametrize("unit", [un.Mpc, un.Mpc / littleh])
@pytest.mark.parametrize("ps", [ps, ps2])
def test_ps_plot_units(unit, ps):
    """Test the 2D power spectrum plot."""

    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"],
        log=[True, True, False],
        labels=["foo"],
    )

    plot_power_spectrum(
        ps["k"],
        ps["ps_1D"],
        title=["z=6", "z=10", "z=27"],
        log=False,
    )
