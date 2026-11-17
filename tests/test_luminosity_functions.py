"""Tests of binned number densities and UV luminosity functions."""

import warnings

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology.units import littleh

from tuesday.core.summaries.luminosity_functions import (
    KUV_SUN16,
    KUV_UNIT,
    UV_MAGNITUDE_EDGES,
    BinnedNumberDensity,
    bin_number_density,
    sfr_to_uv_magnitude,
    uv_luminosity_function,
    uv_luminosity_to_magnitude,
)

VOLUME = (10.0 * un.Mpc) ** 3


def test_magnitude_of_one_msun_per_year():
    # L_UV = 1 / 1.15e-28 erg/s/Hz, so M_UV = 51.63 + 2.5 log10(1.15e-28).
    mag = sfr_to_uv_magnitude(np.array([1.0, 10.0]) * un.Msun / un.yr)
    np.testing.assert_allclose(mag, [-18.2182554, -20.7182554], atol=1e-6)


def test_luminosity_to_magnitude():
    lum = np.array([1e28, 1e29]) * un.erg / un.s / un.Hz
    np.testing.assert_allclose(uv_luminosity_to_magnitude(lum), [-18.37, -20.87])
    np.testing.assert_allclose(
        uv_luminosity_to_magnitude(lum.to(un.W / un.Hz)), [-18.37, -20.87]
    )


def test_zero_sfr_gives_infinite_magnitude_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mag = sfr_to_uv_magnitude(np.array([0.0, 1.0]) * un.Msun / un.yr)
    assert mag[0] == np.inf
    assert np.isfinite(mag[1])


def test_sfr_needs_a_mass_rate():
    with pytest.raises(un.UnitConversionError):
        sfr_to_uv_magnitude(np.array([1.0]) * un.Msun)


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
    np.testing.assert_allclose(got_sfr, [1.0, 2.0], rtol=1e-12)
    np.testing.assert_allclose(got_mass, [1e8, 1e9], rtol=1e-12)
    assert got_z == 8.0
    lum = np.array([1.0 / 1e-28, 2.0 / 1e-27])
    np.testing.assert_allclose(mag, 51.63 - 2.5 * np.log10(lum), rtol=1e-12)


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
    values = np.array([-20.9, -20.1, -19.5, -19.4, -18.0, -17.0])
    edges = np.array([-21.0, -20.0, -19.0, -17.0])
    result = bin_number_density(values, edges, VOLUME)

    assert isinstance(result, BinnedNumberDensity)
    np.testing.assert_array_equal(result.bin_edges, edges)
    # The last bin includes its upper edge.
    np.testing.assert_array_equal(result.counts, [2, 2, 2])
    np.testing.assert_allclose(result.bin_widths, [1.0, 1.0, 2.0])
    np.testing.assert_allclose(result.bin_centres, [-20.5, -19.5, -18.0])
    assert result.number_density.unit == un.Mpc**-3
    np.testing.assert_allclose(
        result.number_density.to_value(un.Mpc**-3), np.array([2, 2, 1]) / 1000.0
    )
    np.testing.assert_allclose(
        result.poisson_error.to_value(un.Mpc**-3),
        np.sqrt([2, 2, 2]) / np.array([1.0, 1.0, 2.0]) / 1000.0,
    )


def test_non_finite_values_are_not_counted():
    values = np.array([-20.5, np.inf, np.nan, -np.inf, -19.5])
    result = bin_number_density(values, 2, VOLUME)
    np.testing.assert_array_equal(result.counts, [1, 1])
    np.testing.assert_allclose(result.bin_edges, [-20.5, -20.0, -19.5])


def test_bin_quantities():
    masses = np.array([2e10, 5e10, 3e11]) * un.Msun
    edges = np.array([1e10, 1e11, 1e12]) * un.Msun
    result = bin_number_density(masses, edges, 1 * un.Gpc**3)
    np.testing.assert_array_equal(result.counts, [2, 1])
    assert result.bin_centres.unit == un.Msun
    np.testing.assert_allclose(
        result.number_density.to_value(un.Mpc**-3 / un.Msun),
        np.array([2 / 9e10, 1 / 9e11]) / 1e9,
    )


def test_bin_quantities_needs_bins_in_the_same_unit():
    masses = np.array([2e10, 5e10]) * un.Msun
    with pytest.raises(un.UnitsError):
        bin_number_density(masses, np.array([1e10, 1e11]), VOLUME)


def test_volume_with_littleh():
    volume = (10.0 * un.Mpc / littleh) ** 3
    result = bin_number_density(np.array([0.5]), np.array([0.0, 1.0]), volume)
    assert result.number_density.unit == (un.Mpc / littleh) ** -3


def test_volume_must_be_a_volume():
    with pytest.raises(ValueError, match="Expected a volume"):
        bin_number_density(np.array([0.5]), np.array([0.0, 1.0]), 10.0 * un.Mpc**2)


def test_uv_luminosity_function():
    rng = np.random.default_rng(0)
    sfr = 10 ** rng.uniform(-3, 2, 1000) * un.Msun / un.yr
    sfr[:10] = 0.0
    result = uv_luminosity_function(sfr, VOLUME)

    np.testing.assert_array_equal(result.bin_edges, UV_MAGNITUDE_EDGES)
    assert len(result.counts) == 24
    np.testing.assert_allclose(result.bin_widths, 0.5)
    mag = 51.63 - 2.5 * np.log10(sfr[10:].to_value(un.Msun / un.yr) / 1.15e-28)
    counts, _ = np.histogram(mag, UV_MAGNITUDE_EDGES)
    np.testing.assert_array_equal(result.counts, counts)
    np.testing.assert_allclose(
        result.number_density.to_value(un.Mpc**-3), counts / 0.5 / 1000.0
    )


def test_uv_luminosity_function_bins_and_kuv():
    # M_UV = -18.22 and -20.72 with the default K_UV.
    sfr = np.array([1.0, 1.0, 10.0]) * un.Msun / un.yr
    edges = np.array([-22.0, -20.0, -18.0])
    np.testing.assert_array_equal(
        uv_luminosity_function(sfr, VOLUME, bins=edges).counts, [1, 2]
    )
    # Twice the K_UV makes the galaxies 2.5 log10(2) = 0.75 mag fainter.
    result = uv_luminosity_function(sfr, VOLUME, bins=edges, kuv=2 * KUV_SUN16)
    np.testing.assert_array_equal(result.counts, [0, 1])


def test_default_edges_are_read_only():
    with pytest.raises(ValueError, match="read-only"):
        UV_MAGNITUDE_EDGES[0] = -30.0
