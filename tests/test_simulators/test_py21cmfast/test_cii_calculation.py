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
    halo_metallicity,
)
from tuesday.simulators.py21cmfast.postprocessing.cii_calculation import (
    calculate_bvoxel,
    calculate_cii,
)

REDSHIFT = 7.0
N_HALOS = 1_000
OTHER_COSMO = FlatLambdaCDM(H0=60.0, Om0=0.4)


SFR_MSUN_PER_YR = 10.0
STELLAR_MASS = 1e9
HALO_MASS = 1e11


def make_halos(
    n_padded: int = N_HALOS, use_ts_fluct: bool = True, use_metallicity: bool = True
) -> SimpleNamespace:
    """Create a stand-in for a PerturbedHaloCatalog with ``N_HALOS`` real halos.

    Every halo has an SFR of 10 Msun/yr, a stellar mass of 1e9 Msun and a halo mass
    of 1e11 Msun. The padding beyond ``N_HALOS`` is NaN, so any use of it shows.
    """
    values = {
        # 21cmFAST gives the SFR in Msun/s.
        "sfr_acg": SFR_MSUN_PER_YR / un.yr.to(un.s),
        "stellar_masses_acg": STELLAR_MASS,
        "halo_masses": HALO_MASS,
    }

    def get(key):
        out = np.full(n_padded, np.nan, dtype=np.float32)
        out[:N_HALOS] = values[key]
        return out

    return SimpleNamespace(
        redshift=REDSHIFT,
        inputs=SimpleNamespace(
            astro_options=SimpleNamespace(
                USE_TS_FLUCT=use_ts_fluct, USE_METALLICITY=use_metallicity
            )
        ),
        cosmo_params=SimpleNamespace(cosmo=OTHER_COSMO),
        n_halos=N_HALOS,
        get=get,
    )


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown \\[CII\\] model"):
        calculate_cii(None, "Silva15")


@pytest.mark.parametrize("model", ["Vallini15", "Horlaville24"])
@pytest.mark.parametrize(
    ("use_ts_fluct", "use_metallicity"), [(False, True), (True, False)]
)
def test_metallicity_models_need_metallicities(
    model: str, use_ts_fluct: bool, use_metallicity: bool
):
    # 21cmFAST only models halo metallicities with USE_TS_FLUCT and USE_METALLICITY.
    halos = make_halos(use_ts_fluct=use_ts_fluct, use_metallicity=use_metallicity)
    with pytest.raises(ValueError, match="USE_METALLICITY"):
        calculate_cii(halos, model, seed=1)


@pytest.mark.parametrize("model", ["deLooze14", "Lagache18", "Casavecchia24"])
def test_sfr_models_without_ts_fluct(model: str):
    lum, _ = calculate_cii(make_halos(use_ts_fluct=False), model, seed=1)

    assert np.all(lum > 0)


def test_padding_is_trimmed():
    # 21cmFAST pads halo arrays beyond the n_halos real halos.
    lum, intens = calculate_cii(make_halos(n_padded=1_500), "Vallini15", seed=0)

    assert lum.shape == intens.shape == (N_HALOS,)
    assert np.all(np.isfinite(intens))


@pytest.mark.parametrize("model", ["Vallini15", "Horlaville24", "deLooze14"])
def test_matches_core(model: str):
    lum, intens = calculate_cii(make_halos(), model, seed=3)

    # The SFR is converted to Msun/yr, the metallicity follows from the SFR and
    # stellar mass, and the simulation cosmology is used.
    sfr = np.full(N_HALOS, SFR_MSUN_PER_YR) * un.Msun / un.yr
    expected = cii_luminosity(
        sfr=sfr,
        metallicity=halo_metallicity(
            sfr, np.full(N_HALOS, STELLAR_MASS) * un.Msun, REDSHIFT
        ),
        halo_mass=np.full(N_HALOS, HALO_MASS) * un.Msun,
        redshift=REDSHIFT,
        model=model,
        seed=3,
    )
    # The stand-in stores the properties in float32, as 21cmFAST does.
    np.testing.assert_allclose(lum, expected.to_value(un.Lsun), rtol=1e-6)
    np.testing.assert_allclose(
        intens,
        cii_flux(expected, REDSHIFT, OTHER_COSMO).to_value(FLUX_UNIT),
        rtol=1e-6,
    )


def test_halo_metallicity_matches_21cmfast():
    # The core relation must stay what 21cmFAST computes from a halo's SFR and stellar
    # mass, since calculate_cii uses it for the metallicity that 21cmFAST does not
    # store. convert_halo_properties returns the SFR, the stellar mass and the
    # metallicity it derives from them.
    p21c = pytest.importorskip("py21cmfast")
    from py21cmfast.wrapper.cfuncs import convert_halo_properties

    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "size-tiny"], random_seed=1
    )
    assert inputs.astro_options.USE_METALLICITY
    assert not inputs.astro_options.USE_MCGS
    rng = np.random.default_rng(0)
    n = 500
    props = convert_halo_properties(
        redshift=REDSHIFT,
        inputs=inputs,
        halo_masses=10 ** rng.uniform(8.5, 12.5, n),
        star_rng=rng.normal(size=n),
        sfr_rng=rng.normal(size=n),
        xray_rng=rng.normal(size=n),
    )
    assert np.all(props["metallicity_acg"] > 0)

    # 21cmFAST computes the metallicity in double precision before storing all three
    # properties in float32, and its year is 2e-5 shorter than astropy's.
    np.testing.assert_allclose(
        halo_metallicity(
            props["sfr_acg"] * un.Msun / un.s,
            props["stellar_mass_acg"] * un.Msun,
            REDSHIFT,
        ),
        props["metallicity_acg"],
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", [(4, 4, 6), (16, 6)], ids=["rectilinear", "angular"])
def test_bvoxel_matches_core(shape: tuple[int, ...]):
    distances = np.linspace(8000.0, 8400.0, 6) * un.Mpc
    intensity = np.random.default_rng(0).uniform(0, 1, size=shape)
    lightcone = SimpleNamespace(
        lightcone_redshifts=z_at_value(Planck18.comoving_distance, distances).value,
        lightcone_distances=distances,
        lightcones={"cii_intensity": intensity},
        simulation_options=SimpleNamespace(BOX_LEN=40.0, HII_DIM=4),
        cosmo_params=SimpleNamespace(cosmo=OTHER_COSMO),
    )
    expected = cii_surface_brightness(
        intensity * FLUX_UNIT, distances, cell_size=10.0 * un.Mpc, cosmo=OTHER_COSMO
    )

    np.testing.assert_allclose(
        calculate_bvoxel(lightcone), expected.to_value(un.Jy / un.sr), rtol=1e-12
    )
