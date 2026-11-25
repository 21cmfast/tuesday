"""Tests of adding LIM noise to a 21cmFAST [CII] lightcone."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck18

from tuesday.core.postprocessing.lim_instruments import LIMInstrument
from tuesday.core.postprocessing.lim_noise import observe_lim_lightcone
from tuesday.simulators.py21cmfast.postprocessing.add_noise_cii import add_cii_noise

KEYS = ("cii_noise_ccatp", "cii_signal_ccatp", "cii_observed_ccatp")


def make_lightcone(n_cells: int = 32) -> SimpleNamespace:
    """Create a minimal stand-in for a 21cmFAST LightCone with a [CII] field."""
    # nu_obs ~ 237-272 GHz: inside the CCAT-prime band.
    redshifts = np.linspace(6.0, 7.0, 40)
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        lightcone_redshifts=redshifts,
        lightcones={"cii_brightness": rng.uniform(1, 2, (n_cells, n_cells, 40))},
        cosmo_params=SimpleNamespace(cosmo=Planck18),
        simulation_options=SimpleNamespace(BOX_LEN=2.0 * n_cells, HII_DIM=n_cells),
    )


def test_add_cii_noise_matches_core():
    lc = make_lightcone()
    out = add_cii_noise(lc, "ccatp", seed=4)
    # The lightcone geometry and cosmology are passed on: 2 Mpc cells.
    signal, noise = observe_lim_lightcone(
        lc.lightcones["cii_brightness"] * un.Jy / un.sr,
        redshifts=lc.lightcone_redshifts,
        cell_size=2.0 * un.Mpc,
        cosmo=Planck18,
        noise_template="ccatp",
        seed=4,
    )

    for key in KEYS:
        assert out.lightcones[key].shape == lc.lightcones["cii_brightness"].shape
    np.testing.assert_allclose(
        out.lightcones["cii_signal_ccatp"], signal.value, rtol=1e-12
    )
    np.testing.assert_allclose(
        out.lightcones["cii_noise_ccatp"], noise.value, rtol=1e-12
    )
    np.testing.assert_allclose(
        out.lightcones["cii_observed_ccatp"], signal.value + noise.value, rtol=1e-12
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
    original = lc.lightcones["cii_brightness"]
    out = add_cii_noise(lc, "ccatp", seed=1)

    assert set(lc.lightcones) == {"cii_brightness"}
    assert out is not lc
    assert out.lightcones["cii_brightness"] is original


def test_add_cii_noise_rejects_angular_lightcone():
    lc = make_lightcone()
    lc.lightcones["cii_brightness"] = lc.lightcones["cii_brightness"][0]
    with pytest.raises(ValueError, match="rectilinear"):
        add_cii_noise(lc, "ccatp", seed=1)


def test_add_cii_noise_other_brightness_field():
    # E.g. [CII] plus CO foregrounds: the output fields are named after the input.
    lc = make_lightcone()
    lc.lightcones["cii_co_brightness"] = 2 * lc.lightcones["cii_brightness"]
    out = add_cii_noise(lc, "ccatp", seed=4, brightness_field="cii_co_brightness")
    default = add_cii_noise(lc, "ccatp", seed=4)

    for key in ("cii_co_noise_ccatp", "cii_co_signal_ccatp", "cii_co_observed_ccatp"):
        assert key in out.lightcones
    np.testing.assert_allclose(
        out.lightcones["cii_co_signal_ccatp"],
        2 * default.lightcones["cii_signal_ccatp"],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        out.lightcones["cii_co_noise_ccatp"], default.lightcones["cii_noise_ccatp"]
    )


def test_add_cii_noise_bad_brightness_field():
    with pytest.raises(ValueError, match="_brightness"):
        add_cii_noise(make_lightcone(), "ccatp", seed=1, brightness_field="density")
