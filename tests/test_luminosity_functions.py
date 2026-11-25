"""Tests of binned number densities and UV luminosity functions."""

import warnings

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology.units import littleh

from tuesday.core import CoevalCatalog
from tuesday.core.summaries.luminosity_functions import (
    KUV_SUN16,
    KUV_UNIT,
    UV_MAGNITUDE_EDGES,
    UV_MAGNITUDE_ZERO_POINT,
    BinnedNumberDensity,
    bin_number_density,
    sfr_to_uv_magnitude,
    uv_luminosity_function,
    uv_luminosity_to_magnitude,
)

VOLUME = (10.0 * un.Mpc) ** 3


def make_catalog(volume: un.Quantity | None = VOLUME, **features) -> CoevalCatalog:
    """Create a catalogue with the given features, all objects at the origin."""
    n = len(next(iter(features.values())))
    return CoevalCatalog(
        positions=np.zeros((n, 3)) * un.Mpc,
        features=features,
        volume=volume,
        redshift=8.0,
    )


def test_magnitude_scales_with_luminosity():
    # Ten times the luminosity is 2.5 mag brighter, in any unit of luminosity.
    lum = np.array([1e26, 1e27, 1e28]) * un.erg / un.s / un.Hz
    mag = uv_luminosity_to_magnitude(lum)
    assert mag.unit == un.mag
    np.testing.assert_allclose(np.diff(mag.value), -2.5, rtol=1e-12)
    np.testing.assert_allclose(
        uv_luminosity_to_magnitude(lum.to(un.W / un.Hz)), mag, rtol=1e-12
    )


def test_magnitude_is_an_absolute_ab_magnitude():
    # The AB magnitude (astropy) of the flux density at 10 pc. The zero point of
    # 21cmFAST differs from it by 0.035 mag (see uv_luminosity_to_magnitude).
    lum = np.logspace(26, 30, 5) * un.erg / un.s / un.Hz
    ab = (lum / (4 * np.pi * (10 * un.pc) ** 2)).to(un.ABmag).value
    mag = uv_luminosity_to_magnitude(lum).value
    assert np.all(np.abs(mag - ab) < 0.04)
    np.testing.assert_allclose(mag - ab, UV_MAGNITUDE_ZERO_POINT - 51.595, atol=1e-3)


def test_sfr_to_magnitude():
    # L_UV = SFR / K_UV: ten times the SFR is 2.5 mag brighter, twice K_UV is
    # 2.5 log10(2) mag fainter.
    sfr = np.array([0.1, 1.0, 10.0]) * un.Msun / un.yr
    mag = sfr_to_uv_magnitude(sfr)
    np.testing.assert_allclose(
        mag, uv_luminosity_to_magnitude(sfr / KUV_SUN16), rtol=1e-12
    )
    np.testing.assert_allclose(np.diff(mag.value), -2.5, rtol=1e-12)
    np.testing.assert_allclose(
        (sfr_to_uv_magnitude(sfr, kuv=2 * KUV_SUN16) - mag).value,
        2.5 * np.log10(2),
        rtol=1e-12,
    )


def test_zero_sfr_gives_infinite_magnitude_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mag = sfr_to_uv_magnitude(np.array([0.0, 1.0]) * un.Msun / un.yr)
    assert mag[0] == np.inf * un.mag
    assert np.isfinite(mag[1])


def test_sfr_needs_a_mass_rate():
    with pytest.raises(un.UnitConversionError):
        sfr_to_uv_magnitude(np.array([1.0]) * un.Msun)


def test_float32_sfr_is_converted_in_double_precision():
    sfr = np.array([1.0, 10.0], dtype=np.float32) * un.Msun / un.s
    np.testing.assert_allclose(
        sfr_to_uv_magnitude(sfr).value,
        sfr_to_uv_magnitude(sfr.astype(np.float64)).value,
        rtol=1e-15,
    )


def test_kuv_in_other_units():
    sfr = np.array([0.1, 1.0, 10.0]) * un.Msun / un.yr
    kuv = KUV_SUN16.to(un.kg / un.s / (un.W / un.Hz))
    np.testing.assert_allclose(
        sfr_to_uv_magnitude(sfr, kuv=kuv), sfr_to_uv_magnitude(sfr), rtol=1e-12
    )


@pytest.mark.parametrize("as_quantity", [False, True])
def test_kuv_function_gets_plain_arrays(as_quantity: bool):
    calls = []

    def kuv(sfr, stellar_mass, redshift):
        calls.append((sfr, stellar_mass, redshift))
        # A K_UV that grows with stellar mass.
        value = 1e-28 * (stellar_mass / 1e8)
        return value * KUV_UNIT if as_quantity else value

    sfr = np.array([1.0, 2.0]) * un.Msun / un.yr
    stellar_mass = np.array([1e8, 1e9]) * un.Msun
    mag = sfr_to_uv_magnitude(
        sfr.to(un.Msun / un.s), kuv=kuv, stellar_mass=stellar_mass, redshift=8.0
    )

    ((got_sfr, got_mass, got_z),) = calls
    assert not isinstance(got_sfr, un.Quantity)
    assert not isinstance(got_mass, un.Quantity)
    assert got_sfr.dtype == got_mass.dtype == np.float64
    np.testing.assert_allclose(got_sfr, [1.0, 2.0], rtol=1e-12)
    np.testing.assert_allclose(got_mass, [1e8, 1e9], rtol=1e-12)
    assert got_z == 8.0
    kuv_values = np.array([1e-28, 1e-27]) * KUV_UNIT
    np.testing.assert_allclose(
        mag, uv_luminosity_to_magnitude(sfr / kuv_values), rtol=1e-12
    )


def test_kuv_function_without_stellar_mass():
    calls = []

    def kuv(sfr, stellar_mass, redshift):
        calls.append((stellar_mass, redshift))
        return np.full_like(sfr, 1.15e-28)

    sfr = np.array([1.0]) * un.Msun / un.yr
    np.testing.assert_allclose(
        sfr_to_uv_magnitude(sfr, kuv=kuv), sfr_to_uv_magnitude(sfr), rtol=1e-12
    )
    assert calls == [(None, None)]


def test_bin_number_density():
    catalog = make_catalog(
        mag=np.array([-20.9, -20.1, -19.5, -19.4, -18.0, -17.0]) * un.mag
    )
    edges = np.array([-21.0, -20.0, -19.0, -17.0]) * un.mag
    result = bin_number_density(catalog, "mag", edges)

    assert isinstance(result, BinnedNumberDensity)
    assert result.feature == "mag"
    np.testing.assert_array_equal(result.bin_edges, edges)
    # The last bin includes its upper edge.
    np.testing.assert_array_equal(result.counts, [2, 2, 2])
    np.testing.assert_allclose(result.bin_widths.to_value(un.mag), [1.0, 1.0, 2.0])
    np.testing.assert_allclose(
        result.bin_centres.to_value(un.mag), [-20.5, -19.5, -18.0]
    )
    assert result.number_density.unit == 1 / (un.mag * un.Mpc**3)
    np.testing.assert_allclose(
        result.number_density.value, np.array([2, 2, 1]) / 1000.0
    )
    np.testing.assert_allclose(
        result.poisson_error.value,
        np.sqrt([2, 2, 2]) / np.array([1.0, 1.0, 2.0]) / 1000.0,
    )


def test_non_finite_values_are_not_counted():
    catalog = make_catalog(
        mag=np.array([-20.5, np.inf, np.nan, -np.inf, -19.5]) * un.mag
    )
    result = bin_number_density(catalog, "mag", 2)
    np.testing.assert_array_equal(result.counts, [1, 1])
    np.testing.assert_allclose(result.bin_edges.to_value(un.mag), [-20.5, -20.0, -19.5])


@pytest.mark.parametrize("values", [[], [np.inf, np.nan]], ids=["empty", "non-finite"])
def test_no_finite_values(values):
    catalog = make_catalog(mag=np.array(values, dtype=float) * un.mag)
    # With a number of bins, numpy.histogram puts them between 0 and 1.
    result = bin_number_density(catalog, "mag", 4)
    np.testing.assert_array_equal(result.counts, [0, 0, 0, 0])
    np.testing.assert_allclose(
        result.bin_edges.to_value(un.mag), [0, 0.25, 0.5, 0.75, 1]
    )
    np.testing.assert_array_equal(result.number_density.value, 0.0)
    result = bin_number_density(catalog, "mag", UV_MAGNITUDE_EDGES)
    np.testing.assert_array_equal(result.counts, 0)


def test_bin_masses():
    catalog = make_catalog(
        volume=1 * un.Gpc**3, mass=np.array([2e10, 5e10, 3e11]) * un.Msun
    )
    edges = np.array([1e10, 1e11, 1e12]) * un.Msun
    result = bin_number_density(catalog, "mass", edges)
    np.testing.assert_array_equal(result.counts, [2, 1])
    assert result.bin_centres.unit == un.Msun
    np.testing.assert_allclose(
        result.number_density.to_value(un.Mpc**-3 / un.Msun),
        np.array([2 / 9e10, 1 / 9e11]) / 1e9,
    )


def test_bin_edges_in_other_units():
    catalog = make_catalog(mass=np.array([2e10, 5e10, 3e11]) * un.Msun)
    edges = np.array([1e10, 1e11, 1e12]) * un.Msun
    np.testing.assert_array_equal(
        bin_number_density(catalog, "mass", edges.to(un.kg)).counts, [2, 1]
    )


def test_bin_edges_need_a_compatible_unit():
    catalog = make_catalog(mass=np.array([2e10, 5e10]) * un.Msun)
    with pytest.raises(un.UnitsError):
        bin_number_density(catalog, "mass", np.array([1e10, 1e11]))


def test_volume_with_littleh():
    catalog = make_catalog(
        volume=(10.0 * un.Mpc / littleh) ** 3,
        x=np.array([0.5]) * un.dimensionless_unscaled,
    )
    result = bin_number_density(catalog, "x", np.array([0.0, 1.0]))
    assert result.number_density.unit == (un.Mpc / littleh) ** -3


def test_catalog_needs_a_volume():
    catalog = make_catalog(volume=None, x=np.array([0.5]) * un.dimensionless_unscaled)
    with pytest.raises(ValueError, match="volume of the catalogue"):
        bin_number_density(catalog, "x", 2)


def test_features_with_more_axes_are_not_binned():
    catalog = make_catalog(velocity=np.zeros((2, 3)) * un.km / un.s)
    with pytest.raises(ValueError, match="one value per object"):
        bin_number_density(catalog, "velocity", 2)


def test_unknown_feature():
    catalog = make_catalog(mass=np.array([1e10]) * un.Msun)
    with pytest.raises(KeyError, match="no feature 'sfr'"):
        bin_number_density(catalog, "sfr", 2)


def schechter(magnitude: np.ndarray) -> np.ndarray:
    """A Schechter UV luminosity function (per Mpc^3 per mag) like those at z ~ 8."""
    phi_star, m_star, alpha = 1e-4, -20.5, -2.0
    x = 10 ** (0.4 * (m_star - magnitude))
    return 0.4 * np.log(10) * phi_star * x ** (alpha + 1) * np.exp(-x)


def draw(rng, pdf, lower, upper, n):
    """Draw n values between lower and upper from a pdf, by its inverse CDF."""
    grid = np.linspace(lower, upper, 100_001)
    cdf = np.concatenate([[0], np.cumsum((pdf(grid[1:]) + pdf(grid[:-1])) / 2)])
    return np.interp(rng.uniform(0, cdf[-1], n), cdf, grid), grid, cdf


def check_recovered(result, n_objects, grid, cdf, volume):
    """Compare a binned number density with the distribution it was drawn from.

    Each of the n_objects falls in bin i with probability p_i, so the expected
    number density is n_objects p_i / (width_i volume), with binomial errors.
    """
    edges = result.bin_edges.value
    p = np.diff(np.interp(edges, grid, cdf)) / cdf[-1]
    norm = result.bin_widths.value * volume.to_value(un.Mpc**3)
    expected = n_objects * p / norm
    sigma = np.sqrt(n_objects * p * (1 - p)) / norm
    observed = result.number_density.value
    populated = n_objects * p >= 10
    assert populated.sum() >= 10
    pulls = (observed - expected)[populated] / sigma[populated]
    assert np.all(np.abs(pulls) < 5)
    assert np.mean(pulls**2) < 2


def test_bin_number_density_recovers_a_known_distribution():
    # Halo masses with dn/dlnM proportional to M^-1 between 1e8 and 1e12 Msun:
    # binning ln(M) must give back that mass function.
    rng = np.random.default_rng(1)
    n_objects = 50_000
    ln_mass, grid, cdf = draw(
        rng, lambda x: np.exp(-x), np.log(1e8), np.log(1e12), n_objects
    )
    volume = (100 * un.Mpc) ** 3
    catalog = make_catalog(volume=volume, ln_mass=ln_mass * un.dimensionless_unscaled)
    edges = np.linspace(np.log(1e8), np.log(1e12), 21) * un.dimensionless_unscaled
    result = bin_number_density(catalog, "ln_mass", edges)
    assert result.number_density.unit == un.Mpc**-3
    check_recovered(result, n_objects, grid, cdf, volume)


def test_uv_luminosity_function_recovers_a_schechter_function():
    # Galaxies with UV magnitudes drawn from a Schechter function, and SFRs that give
    # those magnitudes: the UV LF must give back the Schechter function.
    rng = np.random.default_rng(2)
    volume = (100 * un.Mpc) ** 3
    grid = np.linspace(-24, -12, 1001)
    n_objects = round(np.trapezoid(schechter(grid), grid) * volume.to_value(un.Mpc**3))
    magnitude, grid, cdf = draw(rng, schechter, -24, -12, n_objects)
    luminosity = 10 ** (0.4 * (UV_MAGNITUDE_ZERO_POINT - magnitude))
    sfr = (luminosity * un.erg / un.s / un.Hz) * KUV_SUN16
    catalog = make_catalog(volume=volume, sfr=sfr.to(un.Msun / un.yr))

    result = uv_luminosity_function(catalog)
    assert result.feature == "uv_magnitude"
    np.testing.assert_array_equal(result.bin_edges, UV_MAGNITUDE_EDGES)
    assert result.number_density.unit == 1 / (un.mag * un.Mpc**3)
    check_recovered(result, n_objects, grid, cdf, volume)
    # The bright end falls off exponentially and the faint end as a power law, so
    # the LF decreases with increasing luminosity (in the bins with 10 galaxies or
    # more; the brightest bins are empty or nearly so).
    filled = result.counts >= 10
    assert filled.sum() >= 15
    assert np.all(np.diff(result.number_density.value[filled]) > 0)
    # The catalogue itself is not changed.
    assert "uv_magnitude" not in catalog.features


def test_uv_luminosity_function_bins_and_kuv():
    # M_UV = -18.22 and -20.72 with the default K_UV.
    catalog = make_catalog(sfr=np.array([1.0, 1.0, 10.0]) * un.Msun / un.yr)
    edges = np.array([-22.0, -20.0, -18.0]) * un.mag
    np.testing.assert_array_equal(
        uv_luminosity_function(catalog, bins=edges).counts, [1, 2]
    )
    # Twice the K_UV makes the galaxies 2.5 log10(2) = 0.75 mag fainter.
    result = uv_luminosity_function(catalog, bins=edges, kuv=2 * KUV_SUN16)
    np.testing.assert_array_equal(result.counts, [0, 1])


def test_uv_luminosity_function_passes_features_to_kuv():
    calls = []

    def kuv(sfr, stellar_mass, redshift):
        calls.append((sfr, stellar_mass, redshift))
        return np.full_like(sfr, 1.15e-28)

    catalog = make_catalog(
        star_formation=np.array([1.0, 10.0]) * un.Msun / un.yr,
        mstar=np.array([1e8, 1e9]) * un.Msun,
    )
    result = uv_luminosity_function(
        catalog, kuv=kuv, sfr="star_formation", stellar_mass="mstar"
    )
    assert result.counts.sum() == 2
    ((got_sfr, got_mass, got_z),) = calls
    np.testing.assert_allclose(got_sfr, [1.0, 10.0])
    np.testing.assert_allclose(got_mass, [1e8, 1e9])
    assert got_z == 8.0

    # Without a stellar-mass feature, the function gets None.
    uv_luminosity_function(catalog, kuv=kuv, sfr="star_formation", stellar_mass=None)
    assert calls[-1][1] is None


def test_stellar_mass_is_only_needed_by_a_kuv_function():
    catalog = make_catalog(sfr=np.array([1.0]) * un.Msun / un.yr)
    assert uv_luminosity_function(catalog).counts.sum() == 1
    with pytest.raises(KeyError, match="no feature 'stellar_mass'"):
        uv_luminosity_function(catalog, kuv=lambda sfr, mass, z: 1.15e-28)


def test_default_edges_are_read_only():
    with pytest.raises(ValueError, match="read-only"):
        UV_MAGNITUDE_EDGES[0] = -30.0 * un.mag
