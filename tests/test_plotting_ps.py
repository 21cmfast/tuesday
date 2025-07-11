import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.cosmology.units import littleh

from tuesday.core import (
    CylindricalPS,
    SphericalPS,
    calculate_ps_coeval,
    plot_1d_power_spectrum_k,
    plot_1d_power_spectrum_z,
    plot_2d_power_spectrum,
    plot_power_spectrum,
)


@pytest.fixture(scope="session")
def _psboth():
    """Fixture to create a random power spectrum."""
    rng = np.random.default_rng()
    box = rng.random((100, 100, 100))

    ps1d, ps2d = calculate_ps_coeval(
        box=box * un.dimensionless_unscaled,
        box_length=200 * un.Mpc,
        calc_2d=True,
        calc_1d=True,
        interp=True,
    )
    return ps1d, ps2d


@pytest.fixture(scope="session")
def ps1d(_psboth: tuple[SphericalPS, CylindricalPS]) -> SphericalPS:
    return _psboth[0]


@pytest.fixture(scope="session")
def ps2d(_psboth: tuple[SphericalPS, CylindricalPS]) -> CylindricalPS:
    return _psboth[1]


def test_1d_ps_plot(ps1d: SphericalPS):
    """Test the 1d power spectrum plot."""

    plot_power_spectrum(ps1d, smooth=True)

    _, ax = plt.subplots()
    plot_power_spectrum(
        ps1d,
        ax=ax,
        color="red",
        xlabel="k [1/Mpc]",
        ylabel="P(k) [mK^2 Mpc^3]",
        logx=False,
        logy=False,
        smooth=True,
    )
    plot_power_spectrum(
        ps1d,
        title="Test Title",
        legend="z=6",
    )
    plot_power_spectrum(
        [ps1d, ps1d],
        ax=ax,
        title="Test Title",
        legend="foo",
        logx=False,
        logy=False,
        smooth=False,
        at_k=1.0,
    )

    plot_power_spectrum(
        [ps1d, ps1d],
        ax=ax,
        title="Test Title",
        legend="foo",
        legend_kwargs={"frameon": False},
        logx=True,
        logy=True,
        smooth=True,
        at_k=1.0,
    )

    with pytest.raises(
        ValueError,
        match="power_spectrum must be a SphericalPS object or a list of "
        "SphericalPS objects,"
        " got <class 'numpy.ndarray'> instead.",
    ):
        plot_1d_power_spectrum_z(
            [ps1d, np.ones((10, 10))],
            1.0,
        )

    with pytest.raises(ValueError, match="power_spectrum must be a SphericalPS"):
        plot_1d_power_spectrum_k(np.linspace(0, 10, 10))  # Not a dataclass


def test_bad_1d_ps_units(ps1d):
    with pytest.raises(
        ValueError, match="Expected unit of PS to be temperature squared times volume"
    ):
        SphericalPS(ps1d.ps * un.mK**2 * un.Mpc**2, k=ps1d.k)

    with pytest.raises(ValueError, match="Unit of k must be a wavenumber"):
        SphericalPS(ps1d.ps, k=ps1d.k / un.Mpc**4)


@pytest.mark.parametrize("unit", [un.Mpc, un.Mpc / littleh])
def test_good_1d_ps_units(ps1d, unit):
    good_ps = SphericalPS(
        ps1d.ps.value * un.mK**2 * unit**3, k=ps1d.k, is_deltasq=False
    )
    plot_power_spectrum(good_ps)
    good_ps = SphericalPS(ps1d.ps.value * unit**3, k=ps1d.k, is_deltasq=False)
    plot_power_spectrum(good_ps)


def test_2d_ps_plot(ps2d):
    """Test the 2d power spectrum plot."""
    _, ax = plt.subplots()
    plot_power_spectrum(
        ps2d,
        ax=ax,
        logx=False,
        legend=["foo"],
    )
    plot_power_spectrum(
        ps2d,
        smooth=True,
        title="Test Title",
        legend="foo",
        logx=True,
        logc=True,
    )

    with pytest.raises(ValueError, match="power_spectrum must be a CylindricalPS"):
        plot_2d_power_spectrum(np.linspace(0, 10, 10))  # Not a dataclass
