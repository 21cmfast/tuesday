"""Tests of the LIM noise model and mock observations."""

import importlib.util

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck18

from tuesday.core.postprocessing.lim_instruments import LIMInstrument
from tuesday.core.postprocessing.lim_noise import (
    calculate_noise,
    calculate_tpix,
    observe_lim_lightcone,
)
from tuesday.core.postprocessing.line_intensities import NU_CII_REST

BRIGHTNESS_UNIT = un.Jy / un.sr


def observe(
    brightness: float = 1.0,
    n_cells: int = 64,
    redshifts: np.ndarray | None = None,
    noise_template: LIMInstrument | str = "ccatp",
    **kwargs,
) -> tuple[un.Quantity, un.Quantity]:
    """Mock-observe a uniform lightcone with 1 Mpc cells."""
    if redshifts is None:
        # nu_obs ~ 237-272 GHz: inside the CCAT-prime and TIME bands.
        redshifts = np.linspace(6.0, 7.0, 40)
    shape = (n_cells, n_cells, redshifts.size)
    return observe_lim_lightcone(
        np.full(shape, brightness) * BRIGHTNESS_UNIT,
        redshifts=redshifts,
        cell_size=1.0 * un.Mpc,
        cosmo=Planck18,
        noise_template=noise_template,
        **kwargs,
    )


# --- calculate_tpix -------------------------------------------------------------


def test_calculate_tpix_formula():
    nu = 250 * un.GHz
    inst = LIMInstrument.from_template("ccatp")
    t_pix = calculate_tpix(
        nu,
        diameter=inst.dish_diameter,
        survey_area=inst.survey_area,
        survey_time=inst.survey_time,
        n_feeds=inst.n_feeds,
    )
    # Time per beam-sized pixel: t_surv * N_feed * Omega_beam / Omega_surv.
    expected = (
        inst.survey_time * inst.n_feeds * inst.beam_solid_angle(nu) / inst.survey_area
    )

    assert t_pix.unit == un.s
    np.testing.assert_allclose(t_pix.value, expected.to_value(un.s), rtol=1e-12)


def test_calculate_tpix_scalings():
    kw = {
        "diameter": 6 * un.m,
        "survey_area": 4 * un.deg**2,
        "survey_time": 2000 * un.hour,
        "n_feeds": 120,
    }
    t0 = calculate_tpix(250 * un.GHz, **kw)

    np.testing.assert_allclose(
        calculate_tpix(250 * un.GHz, **(kw | {"n_feeds": 240})) / t0, 2.0, rtol=1e-12
    )
    np.testing.assert_allclose(
        calculate_tpix(250 * un.GHz, **(kw | {"survey_area": 8 * un.deg**2})) / t0,
        0.5,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        calculate_tpix(500 * un.GHz, **kw) / t0, 0.25, rtol=1e-12
    )


# --- calculate_noise ------------------------------------------------------------


def test_calculate_noise_nefd():
    nu = 250 * un.GHz
    inst = LIMInstrument.from_template("ccatp")
    info = calculate_noise(nu, inst)

    expected = inst.sensitivity / inst.beam_solid_angle(nu) / np.sqrt(info["t_pix"])
    assert info["RMS_sigma"].unit == un.Jy / un.sr
    np.testing.assert_allclose(
        info["RMS_sigma"].value, expected.to_value(un.Jy / un.sr), rtol=1e-12
    )
    np.testing.assert_allclose(info["bw_sens"].to_value(un.GHz), 2.5, rtol=1e-12)


def test_calculate_noise_nei():
    nu = 250 * un.GHz
    inst = LIMInstrument.from_template("time")
    info = calculate_noise(nu, inst)

    expected = (11 * un.MJy * un.s**0.5 / un.sr) / np.sqrt(info["t_pix"])
    np.testing.assert_allclose(
        info["RMS_sigma"].value, expected.to_value(un.Jy / un.sr), rtol=1e-12
    )


def test_calculate_noise_rms_per_beam():
    nu = 250 * un.GHz
    inst = LIMInstrument.from_template("ccatp", sensitivity=1 * un.mJy)
    info = calculate_noise(nu, inst)

    # An rms per beam does not integrate down with time.
    expected = 1 * un.mJy / inst.beam_solid_angle(nu)
    np.testing.assert_allclose(
        info["RMS_sigma"].value, expected.to_value(un.Jy / un.sr), rtol=1e-12
    )


def test_calculate_noise_fixed_pixel_time():
    info = calculate_noise(250 * un.GHz, LIMInstrument.from_template("ccatp2"))

    np.testing.assert_allclose(info["t_pix"].to_value(un.hour), 38.0, rtol=1e-12)


def test_more_time_less_noise():
    nu = 250 * un.GHz
    rms1 = calculate_noise(nu, LIMInstrument.from_template("ccatp"))["RMS_sigma"]
    rms4 = calculate_noise(
        nu, LIMInstrument.from_template("ccatp", survey_time=8000 * un.hour)
    )["RMS_sigma"]

    np.testing.assert_allclose((rms1 / rms4).value, 2.0, rtol=1e-12)


def test_calculate_noise_redshift_range():
    info = calculate_noise(250 * un.GHz, LIMInstrument.from_template("ccatp"))

    np.testing.assert_allclose(
        info["z_min"], (NU_CII_REST / (420 * un.GHz)).value - 1, rtol=1e-12
    )
    np.testing.assert_allclose(
        info["z_max"], (NU_CII_REST / (210 * un.GHz)).value - 1, rtol=1e-12
    )


def test_calculate_noise_accepts_template_name():
    nu = 250 * un.GHz
    by_name = calculate_noise(nu, "ccatp")
    by_obj = calculate_noise(nu, LIMInstrument.from_template("ccatp"))

    assert by_name["RMS_sigma"] == by_obj["RMS_sigma"]


def test_calculate_noise_bad_template():
    with pytest.raises(ValueError, match="Input a correct instrument template"):
        calculate_noise(250 * un.GHz, "ccat")


@pytest.mark.skipif(
    importlib.util.find_spec("atlast_sc") is not None,
    reason="atlast_sc is installed",
)
def test_atlast_requires_atlast_sc():
    with pytest.raises(ImportError, match="atlast_sc"):
        calculate_noise(250 * un.GHz, "atlast")


def test_calculate_noise_rest_frequency():
    nu_co = 230.538 * un.GHz  # CO(2-1)
    info = calculate_noise(250 * un.GHz, "ccatp", rest_frequency=nu_co)

    np.testing.assert_allclose(
        info["z_max"], (nu_co / (210 * un.GHz)).value - 1, rtol=1e-12
    )


# --- observe_lim_lightcone --------------------------------------------------------


def test_observe_shapes_and_units():
    signal, noise = observe(n_cells=16, seed=1)

    assert signal.shape == noise.shape == (16, 16, 40)
    assert signal.unit == noise.unit == BRIGHTNESS_UNIT


def test_observe_seed_reproducible():
    _, noise1 = observe(n_cells=16, seed=42)
    _, noise2 = observe(n_cells=16, seed=42)
    _, noise3 = observe(n_cells=16, seed=43)
    _, noise4 = observe(n_cells=16)

    np.testing.assert_array_equal(noise1, noise2)
    assert not np.allclose(noise1, noise3)
    assert not np.allclose(noise1, noise4)


def test_observe_rms_matches_noise_model():
    redshifts = np.linspace(6.0, 7.0, 40)
    _, noise = observe(n_cells=128, redshifts=redshifts, seed=3)
    nu = NU_CII_REST / (1 + redshifts[0])
    rms = calculate_noise(nu, "ccatp")["RMS_sigma"].to_value(BRIGHTNESS_UNIT)

    noise = noise[..., 0].to_value(BRIGHTNESS_UNIT)
    # 128^2 samples: the standard deviation is known to better than 1 per cent.
    np.testing.assert_allclose(noise.std(), rms, rtol=0.03)
    np.testing.assert_allclose(noise.mean(), 0.0, atol=0.03 * rms)


@pytest.mark.parametrize("name", ["ccatp", "time"])
def test_observe_channel_is_averaged(name: str):
    # Each channel holds many lightcone slices: they must be averaged, not summed,
    # so a uniform surface brightness is unchanged by the channel integration and
    # the Gaussian beam.
    signal, _ = observe(brightness=2.0, noise_template=name, seed=0)

    np.testing.assert_allclose(signal.to_value(BRIGHTNESS_UNIT), 2.0, rtol=1e-10)


def test_observe_accepts_instrument():
    inst = LIMInstrument.from_template("ccatp", n_feeds=240)
    _, noise_obj = observe(n_cells=16, noise_template=inst, seed=1)
    _, noise_name = observe(n_cells=16, noise_template="ccatp", seed=1)

    # Twice the feeds: twice the time per pixel, so the noise is sqrt(2) lower.
    np.testing.assert_allclose(noise_name / noise_obj, np.sqrt(2), rtol=1e-10)


def test_observe_bad_template():
    with pytest.raises(ValueError, match="Input a correct instrument template"):
        observe(n_cells=4, noise_template="ccat", seed=1)


def test_observe_masks_out_of_band():
    # CCAT-prime covers 210-420 GHz, i.e. [CII] at z < 8.05.
    redshifts = np.linspace(7.8, 8.3, 30)
    with pytest.warns(UserWarning, match="outside the 210.0 GHz - 420.0 GHz band"):
        signal, noise = observe(n_cells=16, redshifts=redshifts, seed=1)

    nu = NU_CII_REST / (1 + redshifts)
    out_of_band = nu < 210 * un.GHz
    assert out_of_band.any()
    assert not out_of_band.all()
    for field in (signal, noise):
        assert np.all(np.isnan(field[..., out_of_band]))
        assert np.all(np.isfinite(field[..., ~out_of_band]))


def test_observe_rejects_angular_lightcone():
    with pytest.raises(ValueError, match="rectilinear"):
        observe_lim_lightcone(
            np.ones((16, 40)) * BRIGHTNESS_UNIT,
            redshifts=np.linspace(6.0, 7.0, 40),
            cell_size=1.0 * un.Mpc,
            cosmo=Planck18,
            noise_template="ccatp",
        )
