"""Tests of adding LIM noise to a 21cmFAST [CII] lightcone."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck18

from tuesday.core.postprocessing.lim_instruments import LIMInstrument
from tuesday.core.postprocessing.lim_noise import observe_lim_lightcone
from tuesday.simulators.py21cmfast.postprocessing.add_noise_cii import add_cii_noise

KEYS = ("noise_ccatp", "Bcii_ccatp", "Bcii_ccatp_noise")


def make_lightcone(n_cells: int = 32) -> SimpleNamespace:
    """Create a minimal stand-in for a 21cmFAST LightCone with a [CII] field."""
    # nu_obs ~ 237-272 GHz: inside the CCAT-prime band.
    redshifts = np.linspace(6.0, 7.0, 40)
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        lightcone_redshifts=redshifts,
        lightcones={"CII_BrightnessDensity": rng.uniform(1, 2, (n_cells, n_cells, 40))},
        cosmo_params=SimpleNamespace(cosmo=Planck18),
        simulation_options=SimpleNamespace(BOX_LEN=2.0 * n_cells, HII_DIM=n_cells),
    )


def test_add_cii_noise_matches_core():
    lc = make_lightcone()
    out = add_cii_noise(lc, "ccatp", seed=4)
    # The lightcone geometry and cosmology are passed on: 2 Mpc cells.
    signal, noise = observe_lim_lightcone(
        lc.lightcones["CII_BrightnessDensity"] * un.Jy / un.sr,
        redshifts=lc.lightcone_redshifts,
        cell_size=2.0 * un.Mpc,
        cosmo=Planck18,
        noise_template="ccatp",
        seed=4,
    )

    for key in KEYS:
        assert out.lightcones[key].shape == lc.lightcones["CII_BrightnessDensity"].shape
    np.testing.assert_allclose(out.lightcones["Bcii_ccatp"], signal.value, rtol=1e-12)
    np.testing.assert_allclose(out.lightcones["noise_ccatp"], noise.value, rtol=1e-12)
    np.testing.assert_allclose(
        out.lightcones["Bcii_ccatp_noise"], signal.value + noise.value, rtol=1e-12
    )


def test_add_cii_noise_accepts_instrument():
    inst = LIMInstrument.from_template("ccatp", n_feeds=240)
    lc = add_cii_noise(make_lightcone(), inst, seed=1)

    assert set(KEYS) <= set(lc.lightcones)


def test_add_cii_noise_bad_template():
    with pytest.raises(ValueError, match="Input a correct instrument template"):
        add_cii_noise(make_lightcone(), "ccat", seed=1)


def test_add_cii_noise_does_not_mutate_input():
    lc = make_lightcone()
    original = lc.lightcones["CII_BrightnessDensity"]
    out = add_cii_noise(lc, "ccatp", seed=1)

    assert set(lc.lightcones) == {"CII_BrightnessDensity"}
    assert out is not lc
    assert out.lightcones["CII_BrightnessDensity"] is original


def test_add_cii_noise_rejects_angular_lightcone():
    lc = make_lightcone()
    lc.lightcones["CII_BrightnessDensity"] = lc.lightcones["CII_BrightnessDensity"][0]
    with pytest.raises(ValueError, match="rectilinear"):
        add_cii_noise(lc, "ccatp", seed=1)
