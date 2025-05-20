import numpy as np
import pytest
import astropy.units as un
from astropy.cosmology.units import littleh
import matplotlib.pyplot as plt

from tuesday.core import calculate_ps, plot_power_spectrum

def test_1d_ps_plot():
    """Test the 1D power spectrum plot."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0, 10.0, 27.0]

    ps = calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        box_side_shape=100,
        zs=zs,
        calc_2d=False,
        calc_1d=True,
    )

    plot_power_spectrum(ps["k"] * littleh / un.Mpc, ps["ps_1D"] * un.mK**2 / littleh**(-3), log=[True, True], labels = ['foo'])  # This should not raise any exceptions
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2 * un.Mpc**2) # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(ps["k"] / un.Mpc**4, ps["ps_1D"] * un.mK**2 * un.Mpc**3) # Wrong units on k
    plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2, smooth=True)
    fig, _ = plt.subplots()
    plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2 * un.Mpc**3, fig=fig, title='Test Title', labels = 'foo')
    plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2 * un.Mpc**3, fig=fig, title='Test Title', labels = ['z=6', 'z=10', 'z=27'])

def test_2d_ps_plot():
    """Test the 2D power spectrum plot."""
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0, 10.0, 27.0]

    ps = calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        box_side_shape=100,
        zs=zs,
        calc_2d=True,
        calc_1d=False,
        interp=True
    )

    plot_power_spectrum([ps['final_kperp'] * littleh / un.Mpc , ps['final_kpar'] * littleh / un.Mpc], 
                        ps['final_ps_2D'] * un.mK**2 / littleh**(-3), log=[True, True], labels = ['foo'])  # This should not raise any exceptions
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum([ps['final_kperp'] * littleh / un.Mpc , ps['final_kpar'] / un.Mpc], 
                            ps['final_ps_2D'] * un.mK**2 * un.Mpc**2) # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum([ps['final_kperp'] / un.Mpc**4 , ps['final_kpar'] / un.Mpc**4] , 
                            ps['final_ps_2D'] * un.mK**2 * un.Mpc**3) # Wrong units on k
    plot_power_spectrum([ps['final_kperp'] / un.Mpc , ps['final_kpar'] / un.Mpc], 
                        ps['final_ps_2D'] * un.mK**2, smooth=True)
    fig, _ = plt.subplots()
    plot_power_spectrum([ps['final_kperp'] / un.Mpc , ps['final_kpar'] / un.Mpc], 
                        ps['final_ps_2D'] * un.mK**2 * un.Mpc**3, fig=fig, title='Test Title', labels = 'foo')
    plot_power_spectrum([ps['final_kperp'] / un.Mpc , ps['final_kpar'] / un.Mpc], 
                        ps['final_ps_2D'] * un.mK**2 * un.Mpc**3, fig=fig, title='Test Title', labels = ['z=6', 'z=10', 'z=27'])
