import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology.units import littleh
import pytest
from tuesday.core import calculate_ps, plot_power_spectrum

def test_1d_ps_plot(unit):
    """Test the 1D power spectrum plot."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0, 10.0, 27.0]

    ps = calculate_ps(
        test_lc*un.dimensionless_unscaled,
        test_redshifts,
        box_length=200*un.Mpc,
        zs=zs,
        calc_2d=False,
        calc_1d=True,
    )

    plot_power_spectrum(
        ps["k"],
        ps["ps_1D"],
        log=[True, True],
        labels=["foo"],
    )  # This should not raise any exceptions
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            ps["k"], ps["ps_1D"] * un.mK**2 * un.Mpc**2
        )  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            ps["k"] / un.Mpc**4, ps["ps_1D"]
        )  # Wrong units on k
    plot_power_spectrum(ps["k"], ps["ps_1D"], smooth=True)
    fig, _ = plt.subplots()
    plot_power_spectrum(
        ps["k"],
        ps["ps_1D"],
        fig=fig,
        title="Test Title",
        labels="foo",
        log = [False, False],
    )
    plot_power_spectrum(
        ps["k"],
        ps["ps_1D"],
        fig=fig,
        title="Test Title",
        labels=["z=6", "z=10", "z=27"],
        log = True,
    )

def test_2d_ps_plot(unit):
    """Test the 2D power spectrum plot."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0, 10.0, 27.0]

    ps = calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200*unit,
        zs=zs,
        calc_2d=True,
        calc_1d=False,
        interp=True,
    )

    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"],
        log=[True, True, True],
        labels=["foo"],
    )  # This should not raise any exceptions
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps["final_kperp"], ps["final_kpar"]],
            ps["final_ps_2D"],
        )  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(
            [ps["final_kperp"], ps["final_kpar"]],
            ps["final_ps_2D"],
        )  # Wrong units on k
    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"],
        smooth=True,
    )
    fig, _ = plt.subplots()
    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"],
        fig=fig,
        title="Test Title",
        labels="foo",
        log = [True, True]
    )
    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"],
        fig=fig,
        title="Test Title",
        labels=["z=6", "z=10", "z=27"],
        log = False,
    )

@pytest.mark.parametrize("unit", [un.Mpc, un.Mpc/littleh])
def test_ps_plot_units(unit):
    """Test the 2D power spectrum plot."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0, 10.0, 27.0]

    ps = calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200*unit,
        zs=zs,
        calc_2d=True,
        calc_1d=False,
    )
    ps1 = calculate_ps(
        test_lc*un.mK,
        test_redshifts,
        box_length=200*unit,
        zs=zs,
        calc_2d=True,
        calc_1d=False,
    )

    plot_power_spectrum(
        [ps["final_kperp"], ps["final_kpar"]],
        ps["final_ps_2D"],
        log=[True, True, True],
        labels=["foo"],
    )

    plot_power_spectrum(
        ps["k"],
        ps["ps_1D"],
        title="Test Title",
        labels=["z=6", "z=10", "z=27"],
        log = True,
    )

    plot_power_spectrum(
        [ps1["final_kperp"], ps1["final_kpar"]],
        ps1["final_ps_2D"],
        log=[True, True, True],
        labels=["foo"],
    )

    plot_power_spectrum(
        ps1["k"],
        ps1["ps_1D"],
        title="Test Title",
        labels=["z=6", "z=10", "z=27"],
        log = True,
    )
