"""Tests of the LIM instrument templates."""

import attrs
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as un

from tuesday.core.postprocessing.lim_instruments import LIMInstrument

TEMPLATES = ["atlast", "ccatp", "ccatp2", "time", "time2"]


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
