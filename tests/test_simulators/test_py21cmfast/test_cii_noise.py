"""Tests of the LIM instrument templates and the [CII] noise model."""

import importlib.util
from types import SimpleNamespace

import attrs
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import Planck18

from tuesday.simulators.py21cmfast.postprocessing.add_noise_cii import (
    add_cii_noise,
    calculate_noise,
    calculate_tpix,
)
from tuesday.simulators.py21cmfast.postprocessing.cii_calculation import NU_CII_REST
from tuesday.simulators.py21cmfast.postprocessing.lim_instruments import (
    LIMInstrument,
)

TEMPLATES = ["atlast", "ccatp", "ccatp2", "time", "time2"]
NEI_UNIT = un.Jy * un.s**0.5 / un.sr


def make_lightcone(
    brightness: float | np.ndarray = 1.0,
    n_cells: int = 64,
    redshifts: np.ndarray | None = None,
) -> SimpleNamespace:
    """Create a minimal stand-in for a 21cmFAST LightCone with a [CII] field."""
    if redshifts is None:
        # nu_obs ~ 237-272 GHz: inside the CCAT-prime and TIME bands.
        redshifts = np.linspace(6.0, 7.0, 40)
    shape = (n_cells, n_cells, redshifts.size)
    return SimpleNamespace(
        lightcone_redshifts=redshifts,
        lightcones={"CII_BrightnessDensity": np.broadcast_to(brightness, shape).copy()},
        cosmo_params=SimpleNamespace(cosmo=Planck18),
        simulation_options=SimpleNamespace(BOX_LEN=float(n_cells), HII_DIM=n_cells),
    )


# --- LIMInstrument ----------------------------------------------------------------


def test_available_templates():
    assert LIMInstrument.available_templates() == TEMPLATES


@pytest.mark.parametrize("name", TEMPLATES)
def test_from_template(name: str):
    inst = LIMInstrument.from_template(name)

    assert inst.name == name
    assert inst.spectral_resolution > 0
    assert inst.n_feeds >= 1
    assert inst.dish_diameter.unit == un.m
    assert inst.min_frequency < inst.max_frequency
    assert inst.survey_time.unit == un.hour
    assert inst.survey_area.unit.is_equivalent(un.sr)


def test_unknown_template_raises():
    with pytest.raises(ValueError, match="Input a correct instrument template"):
        LIMInstrument.from_template("not-a-telescope")


def test_template_overrides():
    inst = LIMInstrument.from_template("ccatp", n_feeds=240, survey_time=8000 * un.hour)

    assert inst.n_feeds == 240
    assert inst.survey_time == 8000 * un.hour
    assert inst.dish_diameter == 6 * un.m


def test_template_references():
    inst = LIMInstrument.from_template("ccatp")

    assert "2107.10364" in inst.references["survey_area"]
    assert "2309.15733" in inst.references["sensitivity"]


def test_wrong_units_raise():
    with pytest.raises(un.UnitConversionError):
        LIMInstrument.from_template("ccatp", dish_diameter=6 * un.GHz)
    with pytest.raises(un.UnitConversionError):
        LIMInstrument.from_template("ccatp", sensitivity=1 * un.K)


def test_bad_frequency_range_raises():
    with pytest.raises(ValueError, match="min_frequency"):
        LIMInstrument.from_template("ccatp", min_frequency=500 * un.GHz)


def test_instrument_is_frozen():
    inst = LIMInstrument.from_template("ccatp")
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        inst.n_feeds = 1


def test_from_yaml(tmp_path):
    path = tmp_path / "toy.yaml"
    path.write_text(
        """
name: toy
description: A toy instrument.
parameters:
  spectral_resolution: {value: 50}
  dish_diameter: {value: 10, unit: m, reference: Made up}
  min_frequency: {value: 200, unit: GHz}
  max_frequency: {value: 300, unit: GHz}
  n_feeds: {value: 10}
  survey_time: {value: 100, unit: h}
  survey_area: {value: 1, unit: deg2}
  sensitivity: {value: 10, unit: mJy s(1/2)}
"""
    )
    inst = LIMInstrument.from_yaml(path)

    assert inst.name == "toy"
    assert inst.dish_diameter == 10 * un.m
    assert inst.sensitivity == 10 * un.mJy * un.s**0.5
    assert inst.pixel_time is None
    assert inst.references == {"dish_diameter": "Made up"}


def test_sensitivity_break_frequency():
    inst = LIMInstrument.from_template("time")

    assert inst.sensitivity_at(250 * un.GHz) == 11 * un.MJy * un.s**0.5 / un.sr
    assert inst.sensitivity_at(230 * un.GHz) == 11 * un.MJy * un.s**0.5 / un.sr
    assert inst.sensitivity_at(200 * un.GHz) == 5.2 * un.MJy * un.s**0.5 / un.sr


def test_beam_scales_with_wavelength():
    inst = LIMInstrument.from_template("ccatp")
    nu = 250 * un.GHz

    expected_fwhm = 1.22 * (const.c / nu / inst.dish_diameter).to(
        un.dimensionless_unscaled
    )
    np.testing.assert_allclose(
        inst.beam_fwhm(nu).to_value(un.rad), expected_fwhm.value, rtol=1e-12
    )
    # Diffraction-limited beam: solid angle goes as lambda^2.
    np.testing.assert_allclose(
        (inst.beam_solid_angle(nu) / inst.beam_solid_angle(2 * nu)).value,
        4.0,
        rtol=1e-12,
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


# --- add_cii_noise --------------------------------------------------------------


def test_add_cii_noise_keys():
    lc = add_cii_noise(make_lightcone(), "ccatp", seed=1)

    for key in ("noise_ccatp", "Bcii_ccatp", "Bcii_ccatp_noise"):
        assert lc.lightcones[key].shape == lc.lightcones["CII_BrightnessDensity"].shape


def test_add_cii_noise_seed_reproducible():
    lc1 = add_cii_noise(make_lightcone(), "ccatp", seed=42)
    lc2 = add_cii_noise(make_lightcone(), "ccatp", seed=42)
    lc3 = add_cii_noise(make_lightcone(), "ccatp", seed=43)

    np.testing.assert_array_equal(
        lc1.lightcones["noise_ccatp"], lc2.lightcones["noise_ccatp"]
    )
    assert not np.allclose(lc1.lightcones["noise_ccatp"], lc3.lightcones["noise_ccatp"])


def test_add_cii_noise_without_seed():
    lc1 = add_cii_noise(make_lightcone(), "ccatp")
    lc2 = add_cii_noise(make_lightcone(), "ccatp")

    assert not np.allclose(lc1.lightcones["noise_ccatp"], lc2.lightcones["noise_ccatp"])


def test_add_cii_noise_rms_matches_noise_model():
    lc = add_cii_noise(make_lightcone(n_cells=128), "ccatp", seed=3)
    nu = NU_CII_REST / (1 + lc.lightcone_redshifts[0])
    rms = calculate_noise(nu, "ccatp")["RMS_sigma"].to_value(un.Jy / un.sr)

    noise = lc.lightcones["noise_ccatp"][..., 0]
    # 128^2 samples: the standard deviation is known to better than 1 per cent.
    np.testing.assert_allclose(noise.std(), rms, rtol=0.03)
    np.testing.assert_allclose(noise.mean(), 0.0, atol=0.03 * rms)


def test_add_cii_noise_is_signal_plus_noise():
    lc = add_cii_noise(make_lightcone(), "ccatp", seed=5)

    np.testing.assert_allclose(
        lc.lightcones["Bcii_ccatp_noise"],
        lc.lightcones["Bcii_ccatp"] + lc.lightcones["noise_ccatp"],
        rtol=1e-12,
    )


def test_add_cii_noise_beam_preserves_uniform_signal():
    lc = add_cii_noise(make_lightcone(brightness=2.0), "ccatp", seed=0)
    smoothed = lc.lightcones["Bcii_ccatp"]

    # A Gaussian beam leaves a uniform field unchanged within each slice.
    np.testing.assert_allclose(
        smoothed, np.broadcast_to(smoothed[:1, :1, :], smoothed.shape), rtol=1e-10
    )
    assert np.all(smoothed > 0)


@pytest.mark.parametrize("name", ["ccatp", "time"])
def test_add_cii_noise_channel_is_averaged(name: str):
    # Each channel holds many lightcone slices: they must be averaged, not summed,
    # so a uniform surface brightness is unchanged by the channel integration.
    lc = add_cii_noise(make_lightcone(brightness=2.0), name, seed=0)

    np.testing.assert_allclose(lc.lightcones[f"Bcii_{name}"], 2.0, rtol=1e-10)


def test_add_cii_noise_accepts_instrument():
    inst = LIMInstrument.from_template("ccatp", n_feeds=240)
    lc = add_cii_noise(make_lightcone(), inst, seed=1)

    assert "noise_ccatp" in lc.lightcones


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


def test_add_cii_noise_masks_out_of_band():
    # CCAT-prime covers 210-420 GHz, i.e. [CII] at z < 8.05.
    redshifts = np.linspace(7.8, 8.3, 30)
    with pytest.warns(UserWarning, match="outside the 210.0 GHz - 420.0 GHz band"):
        lc = add_cii_noise(make_lightcone(redshifts=redshifts), "ccatp", seed=1)

    nu = NU_CII_REST / (1 + redshifts)
    out_of_band = nu < 210 * un.GHz
    assert out_of_band.any()
    assert not out_of_band.all()
    for key in ("noise_ccatp", "Bcii_ccatp", "Bcii_ccatp_noise"):
        assert np.all(np.isnan(lc.lightcones[key][..., out_of_band]))
        assert np.all(np.isfinite(lc.lightcones[key][..., ~out_of_band]))


def test_add_cii_noise_rejects_angular_lightcone():
    lc = make_lightcone()
    lc.lightcones["CII_BrightnessDensity"] = lc.lightcones["CII_BrightnessDensity"][0]
    with pytest.raises(ValueError, match="rectilinear"):
        add_cii_noise(lc, "ccatp", seed=1)
