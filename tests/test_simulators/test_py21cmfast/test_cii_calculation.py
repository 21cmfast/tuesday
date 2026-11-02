"""Tests of the 21cmFAST wrappers for the [CII] calculations."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import FlatLambdaCDM, Planck18, z_at_value

from tuesday.core.postprocessing.line_intensities import (
    FLUX_UNIT,
    cii_flux,
    cii_luminosity,
    cii_surface_brightness,
)
from tuesday.simulators.py21cmfast.postprocessing import cii_calculation
from tuesday.simulators.py21cmfast.postprocessing.cii_calculation import (
    calculate_bvoxel,
    calculate_cii,
)

REDSHIFT = 7.0
N_HALOS = 1_000
OTHER_COSMO = FlatLambdaCDM(H0=60.0, Om0=0.4)


def make_halos(n_padded: int = N_HALOS, use_ts_fluct: bool = True) -> SimpleNamespace:
    """Create a stand-in for a HaloCatalog with ``N_HALOS`` real halos."""
    return SimpleNamespace(
        redshift=REDSHIFT,
        inputs=SimpleNamespace(
            astro_options=SimpleNamespace(USE_TS_FLUCT=use_ts_fluct)
        ),
        cosmo_params=SimpleNamespace(cosmo=OTHER_COSMO),
        n_halos=N_HALOS,
        get=lambda key: np.zeros((n_padded, 3) if key == "halo_coords" else n_padded),
    )


@pytest.fixture
def received(monkeypatch) -> dict:
    """Replace the 21cmFAST halo properties: SFR = 10 Msun/yr for every halo.

    The arguments passed to ``convert_halo_properties`` are recorded.
    """
    received = {}

    def fake_convert(**kwargs):
        received.update(kwargs)
        n = len(kwargs["halo_masses"])
        return {
            "halo_mass": np.full(n, 1e11),
            "halo_stars": np.full(n, 1e9),
            # 21cmFAST returns Msun / s.
            "halo_sfr": np.full(n, 10.0 / un.yr.to(un.s)),
            "metallicity": np.full(n, 0.2),
        }

    monkeypatch.setattr(cii_calculation, "convert_halo_properties", fake_convert)
    return received


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown \\[CII\\] model"):
        calculate_cii(None, "Silva15")


@pytest.mark.parametrize("model", ["Vallini15", "Horlaville24"])
def test_metallicity_models_need_ts_fluct(received, model: str):
    # 21cmFAST only computes halo metallicities when USE_TS_FLUCT is on.
    with pytest.raises(ValueError, match="USE_TS_FLUCT"):
        calculate_cii(make_halos(use_ts_fluct=False), model, seed=1)
    assert not received


@pytest.mark.parametrize("model", ["deLooze14", "Lagache18", "Casavecchia24"])
def test_sfr_models_without_ts_fluct(received, model: str):
    lum, _ = calculate_cii(make_halos(use_ts_fluct=False), model, seed=1)

    assert np.all(lum > 0)


def test_padding_is_trimmed(received):
    # 21cmFAST pads halo arrays with zeros beyond the n_halos real halos.
    lum, intens = calculate_cii(make_halos(n_padded=1_500), "Vallini15", seed=0)

    for key in ("halo_masses", "star_rng", "sfr_rng", "xray_rng"):
        assert len(received[key]) == N_HALOS
    assert received["halo_coords"].shape == (N_HALOS, 3)
    assert lum.shape == intens.shape == (N_HALOS,)
    assert np.all(np.isfinite(intens))


@pytest.mark.parametrize("model", ["Horlaville24", "deLooze14"])
def test_matches_core(received, model: str):
    lum, intens = calculate_cii(make_halos(), model, seed=3)

    # The SFR is converted to Msun/yr and the simulation cosmology is used.
    expected = cii_luminosity(
        sfr=np.full(N_HALOS, 10.0) * un.Msun / un.yr,
        metallicity=np.full(N_HALOS, 0.2),
        halo_mass=np.full(N_HALOS, 1e11) * un.Msun,
        redshift=REDSHIFT,
        model=model,
        seed=3,
    )
    np.testing.assert_allclose(lum, expected.to_value(un.Lsun), rtol=1e-10)
    np.testing.assert_allclose(
        intens,
        cii_flux(expected, REDSHIFT, OTHER_COSMO).to_value(FLUX_UNIT),
        rtol=1e-10,
    )


@pytest.mark.parametrize("shape", [(4, 4, 6), (16, 6)], ids=["rectilinear", "angular"])
def test_bvoxel_matches_core(shape: tuple[int, ...]):
    distances = np.linspace(8000.0, 8400.0, 6) * un.Mpc
    intensity = np.random.default_rng(0).uniform(0, 1, size=shape)
    lightcone = SimpleNamespace(
        lightcone_redshifts=z_at_value(Planck18.comoving_distance, distances).value,
        lightcone_distances=distances,
        lightcones={"CII_IntensityBox": intensity},
        simulation_options=SimpleNamespace(BOX_LEN=40.0, HII_DIM=4),
        cosmo_params=SimpleNamespace(cosmo=OTHER_COSMO),
    )
    expected = cii_surface_brightness(
        intensity * FLUX_UNIT, distances, cell_size=10.0 * un.Mpc, cosmo=OTHER_COSMO
    )

    np.testing.assert_allclose(
        calculate_bvoxel(lightcone), expected.to_value(un.Jy / un.sr), rtol=1e-12
    )
