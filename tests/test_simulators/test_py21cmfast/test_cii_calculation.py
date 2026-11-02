"""Tests of the [CII] luminosity, intensity and surface-brightness calculations."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck18, z_at_value

from tuesday.simulators.py21cmfast.postprocessing import cii_calculation
from tuesday.simulators.py21cmfast.postprocessing.cii_calculation import (
    CII_MODELS,
    NU_CII_REST,
    calculate_bvoxel,
    calculate_cii,
)

REDSHIFT = 7.0


@pytest.fixture
def mock_halos(monkeypatch) -> SimpleNamespace:
    """Halo catalogue with a fixed SFR and metallicity, bypassing 21cmFAST."""
    n_halos = 20_000
    sfr_yr = np.full(n_halos, 10.0)  # Msun / yr
    props = {
        "halo_mass": np.full(n_halos, 1e11),
        "halo_stars": np.full(n_halos, 1e9),
        "halo_sfr": sfr_yr / un.yr.to(un.s),  # 21cmFAST returns Msun / s
        "metallicity": np.full(n_halos, 0.2),
    }
    monkeypatch.setattr(
        cii_calculation, "convert_halo_properties", lambda **kwargs: props
    )
    return SimpleNamespace(
        redshift=REDSHIFT, inputs=None, get=lambda key: np.zeros(n_halos)
    )


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown \\[CII\\] model"):
        calculate_cii(None, "Silva15")


@pytest.mark.parametrize("model", CII_MODELS)
def test_models_run(mock_halos, model: str):
    lum, intens = calculate_cii(mock_halos, model, seed=1)

    assert lum.shape == intens.shape == (20_000,)
    assert np.all(np.isfinite(lum))
    assert np.all(lum > 0)


@pytest.mark.parametrize("model", CII_MODELS)
def test_seed_is_reproducible(mock_halos, model: str):
    lum1, _ = calculate_cii(mock_halos, model, seed=7)
    lum2, _ = calculate_cii(mock_halos, model, seed=7)
    lum3, _ = calculate_cii(mock_halos, model, seed=8)

    np.testing.assert_array_equal(lum1, lum2)
    assert not np.allclose(lum1, lum3)


def test_delooze14_relation(mock_halos):
    lum, _ = calculate_cii(mock_halos, "deLooze14", seed=0)
    log_lum = np.log10(lum)

    # log L = 7.10 + 1.05 log SFR, with 0.4 dex Gaussian scatter.
    np.testing.assert_allclose(np.median(log_lum), 7.10 + 1.05, atol=0.02)
    np.testing.assert_allclose(log_lum.std(), 0.4, rtol=0.03)


def test_lagache18_relation(mock_halos):
    lum, _ = calculate_cii(mock_halos, "Lagache18", seed=0)

    expected = (1.4 - 0.07 * REDSHIFT) + (7.1 - 0.07 * REDSHIFT)
    np.testing.assert_allclose(np.median(np.log10(lum)), expected, atol=0.02)


def test_intensity_from_luminosity(mock_halos):
    lum, intens = calculate_cii(mock_halos, "deLooze14", seed=0)

    # L [Lsun] = 1.04e-3 S dv [Jy km/s] nu_obs [GHz] D_L^2 [Mpc^2]
    d_lum = Planck18.luminosity_distance(REDSHIFT).to_value(un.Mpc)
    nu_obs = NU_CII_REST.to_value(un.GHz) / (1 + REDSHIFT)
    np.testing.assert_allclose(intens, lum / (1.04e-3 * d_lum**2 * nu_obs), rtol=1e-12)


def make_intensity_lightcone(intensity: np.ndarray) -> SimpleNamespace:
    """Create a minimal stand-in for a LightCone carrying a [CII] intensity grid."""
    distances = np.linspace(8000.0, 8400.0, intensity.shape[-1]) * un.Mpc
    redshifts = z_at_value(Planck18.comoving_distance, distances).value
    return SimpleNamespace(
        lightcone_redshifts=redshifts,
        lightcone_distances=distances,
        lightcones={"CII_IntensityBox": intensity},
        simulation_options=SimpleNamespace(BOX_LEN=40.0, HII_DIM=4),
    )


def reference_bvoxel(lightcone: SimpleNamespace) -> np.ndarray:
    """The original slice-by-slice implementation of the surface brightness."""
    z = lightcone.lightcone_redshifts
    nu_cii = NU_CII_REST.to_value(un.GHz)
    cell_size = 10.0 * un.Mpc
    dcomv = lightcone.lightcone_distances
    solid_angle = (cell_size / dcomv) ** 2
    delta_nu = np.zeros(dcomv.shape)
    for i in range(len(dcomv)):
        z_front = z_at_value(Planck18.comoving_distance, dcomv[i] - cell_size / 2)
        z_back = z_at_value(Planck18.comoving_distance, dcomv[i] + cell_size / 2)
        delta_nu[i] = nu_cii / (1 + z_front.value) - nu_cii / (1 + z_back.value)
    intensity = lightcone.lightcones["CII_IntensityBox"]
    return (
        nu_cii
        / (1 + z)
        / (const.c.to_value(un.km / un.s) * delta_nu * solid_angle.value)
        * intensity
    )


def test_bvoxel_matches_reference():
    rng = np.random.default_rng(0)
    lightcone = make_intensity_lightcone(rng.uniform(0, 1, size=(4, 4, 6)))

    np.testing.assert_allclose(
        calculate_bvoxel(lightcone), reference_bvoxel(lightcone), rtol=1e-6
    )


def test_bvoxel_is_linear_in_intensity():
    rng = np.random.default_rng(1)
    intensity = rng.uniform(0, 1, size=(4, 4, 6))
    b1 = calculate_bvoxel(make_intensity_lightcone(intensity))
    b3 = calculate_bvoxel(make_intensity_lightcone(3 * intensity))

    np.testing.assert_allclose(b3, 3 * b1, rtol=1e-12)
    assert np.all(b1 >= 0)
