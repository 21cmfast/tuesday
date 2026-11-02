"""Tests of the [CII] luminosity, flux and surface-brightness calculations."""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import FlatLambdaCDM, Planck18, z_at_value

from tuesday.core.postprocessing.line_intensities import (
    CII_MODELS,
    FLUX_UNIT,
    NU_CII_REST,
    cii_flux,
    cii_luminosity,
    cii_surface_brightness,
    validate_cii_model,
)

REDSHIFT = 7.0
N_HALOS = 20_000
OTHER_COSMO = FlatLambdaCDM(H0=60.0, Om0=0.4)


def luminosity(model: str, seed: int = 0, sfr: un.Quantity | None = None) -> np.ndarray:
    """[CII] luminosity in Lsun of identical halos with SFR = 10 Msun/yr."""
    if sfr is None:
        sfr = np.full(N_HALOS, 10.0) * un.Msun / un.yr
    return cii_luminosity(
        sfr=sfr,
        metallicity=np.full(N_HALOS, 0.2),
        halo_mass=np.full(N_HALOS, 1e11) * un.Msun,
        redshift=REDSHIFT,
        model=model,
        seed=seed,
    ).to_value(un.Lsun)


# --- cii_luminosity ---------------------------------------------------------------


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown \\[CII\\] model"):
        validate_cii_model("Silva15")
    with pytest.raises(ValueError, match="Unknown \\[CII\\] model"):
        luminosity("Silva15")


@pytest.mark.parametrize("model", CII_MODELS)
def test_models_run(model: str):
    lum = luminosity(model, seed=1)

    assert lum.shape == (N_HALOS,)
    assert np.all(np.isfinite(lum))
    assert np.all(lum > 0)


@pytest.mark.parametrize("model", CII_MODELS)
def test_seed_is_reproducible(model: str):
    np.testing.assert_array_equal(luminosity(model, seed=7), luminosity(model, seed=7))
    assert not np.allclose(luminosity(model, seed=7), luminosity(model, seed=8))


def test_sfr_units_are_converted():
    sfr_per_s = np.full(N_HALOS, 10.0) * un.Msun / un.yr.to(un.s) / un.s

    np.testing.assert_allclose(
        luminosity("deLooze14", sfr=sfr_per_s), luminosity("deLooze14"), rtol=1e-10
    )


def test_delooze14_relation():
    log_lum = np.log10(luminosity("deLooze14"))

    # log L = 7.10 + 1.05 log SFR, with 0.4 dex Gaussian scatter.
    np.testing.assert_allclose(np.median(log_lum), 7.10 + 1.05, atol=0.02)
    np.testing.assert_allclose(log_lum.std(), 0.4, rtol=0.03)


def test_lagache18_relation():
    expected = (1.4 - 0.07 * REDSHIFT) + (7.1 - 0.07 * REDSHIFT)

    np.testing.assert_allclose(
        np.median(np.log10(luminosity("Lagache18"))), expected, atol=0.02
    )


# --- cii_flux ---------------------------------------------------------------------


@pytest.mark.parametrize("cosmo", [Planck18, OTHER_COSMO], ids=["Planck18", "other"])
def test_flux_from_luminosity(cosmo: FlatLambdaCDM):
    lum = np.array([1e7, 1e8, 1e9]) * un.Lsun
    flux = cii_flux(lum, REDSHIFT, cosmo)

    # L [Lsun] = 1.04e-3 S dv [Jy km/s] nu_obs [GHz] D_L^2 [Mpc^2]
    d_lum = cosmo.luminosity_distance(REDSHIFT).to_value(un.Mpc)
    nu_obs = NU_CII_REST.to_value(un.GHz) / (1 + REDSHIFT)
    assert flux.unit == FLUX_UNIT
    np.testing.assert_allclose(
        flux.value, lum.value / (1.04e-3 * d_lum**2 * nu_obs), rtol=1e-12
    )


# --- cii_surface_brightness -------------------------------------------------------

DISTANCES = np.linspace(8000.0, 8400.0, 6) * un.Mpc
CELL_SIZE = 10.0 * un.Mpc


def brightness(flux: np.ndarray, cosmo: FlatLambdaCDM = Planck18) -> np.ndarray:
    """Surface brightness in Jy/sr of a flux grid on the test distances."""
    return cii_surface_brightness(
        flux * FLUX_UNIT, distances=DISTANCES, cell_size=CELL_SIZE, cosmo=cosmo
    ).to_value(un.Jy / un.sr)


def reference_brightness(flux: np.ndarray, cosmo: FlatLambdaCDM) -> np.ndarray:
    """The original slice-by-slice implementation of the surface brightness."""
    nu_cii = NU_CII_REST.to_value(un.GHz)
    solid_angle = (CELL_SIZE / DISTANCES) ** 2
    # A tight tolerance: the default one gives errors of ~1e-6 in delta_nu.
    tol = {"ztol": 1e-12, "maxfun": 10_000}
    z = np.zeros(DISTANCES.shape)
    delta_nu = np.zeros(DISTANCES.shape)
    for i, dist in enumerate(DISTANCES):
        z[i] = z_at_value(cosmo.comoving_distance, dist, **tol).value
        z_front = z_at_value(cosmo.comoving_distance, dist - CELL_SIZE / 2, **tol)
        z_back = z_at_value(cosmo.comoving_distance, dist + CELL_SIZE / 2, **tol)
        delta_nu[i] = nu_cii / (1 + z_front.value) - nu_cii / (1 + z_back.value)
    return (
        nu_cii
        / (1 + z)
        / (const.c.to_value(un.km / un.s) * delta_nu * solid_angle.value)
        * flux
    )


@pytest.mark.parametrize("cosmo", [Planck18, OTHER_COSMO], ids=["Planck18", "other"])
def test_brightness_matches_reference(cosmo: FlatLambdaCDM):
    flux = np.random.default_rng(0).uniform(0, 1, size=(4, 4, 6))

    np.testing.assert_allclose(
        brightness(flux, cosmo), reference_brightness(flux, cosmo), rtol=1e-6
    )


def test_brightness_is_linear_in_flux():
    flux = np.random.default_rng(1).uniform(0, 1, size=(4, 4, 6))
    b1 = brightness(flux)

    np.testing.assert_allclose(brightness(3 * flux), 3 * b1, rtol=1e-12)
    assert np.all(b1 >= 0)


def test_brightness_depends_on_cosmology():
    flux = np.ones((4, 4, 6))

    assert not np.allclose(brightness(flux), brightness(flux, OTHER_COSMO), rtol=1e-3)


def test_brightness_any_transverse_shape():
    flux = np.random.default_rng(2).uniform(0, 1, size=(4, 4, 6))

    # Angular lightcones store fields as (pixel, slice).
    np.testing.assert_allclose(
        brightness(flux.reshape(16, 6)), brightness(flux).reshape(16, 6), rtol=1e-12
    )
