import numpy as np
import pytest
import astropy.units as un
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
        calc_global=True,
    )

    plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2 * un.Mpc**3, log=[True, True], labels = ['foo'])  # This should not raise any exceptions
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2 * un.Mpc**2) # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        plot_power_spectrum(ps["k"] / un.Mpc**4, ps["ps_1D"] * un.mK**2 * un.Mpc**3) # Wrong units on k
    plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2 * un.Mpc**3, smooth=True)
    fig, _ = plt.subplots()
    plot_power_spectrum(ps["k"] / un.Mpc, ps["ps_1D"] * un.mK**2 * un.Mpc**3, fig=fig, title='Test Title', labels = 'foo')