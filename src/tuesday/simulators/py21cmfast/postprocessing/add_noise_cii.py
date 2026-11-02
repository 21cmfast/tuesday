"""Add instrumental noise to a [CII] surface-brightness lightcone.

The instrument and survey are specified by a
:class:`~.lim_instruments.LIMInstrument`, usually loaded from a template with
:meth:`~.lim_instruments.LIMInstrument.from_template`.
At each frequency of the lightcone, the signal is integrated over one spectral channel
of the instrument, smoothed with the Gaussian instrument beam, and white noise with the
per-pixel rms of the survey is added.
"""

import logging
from typing import Any

import numpy as np
from astropy import units as un
from astropy.stats import gaussian_fwhm_to_sigma
from py21cmfast import LightCone
from scipy.ndimage import gaussian_filter

from .cii_calculation import NU_CII_REST
from .lim_instruments import LIMInstrument, beam_solid_angle

logger = logging.getLogger(__name__)


def _as_instrument(noise_template: LIMInstrument | str) -> LIMInstrument:
    """Load an instrument template by name, or pass an instrument through."""
    if isinstance(noise_template, LIMInstrument):
        return noise_template
    return LIMInstrument.from_template(noise_template)


def calculate_tpix(
    nu_obs: un.Quantity[un.GHz],
    diameter: un.Quantity[un.m],
    survey_area: un.Quantity[un.deg**2],
    survey_time: un.Quantity[un.hour],
    n_feeds: int,
) -> un.Quantity[un.s]:
    """Calculate the integration time per beam-sized pixel of a survey.

    The survey time is shared evenly between the ``survey_area / beam_area`` pixels,
    and each of the ``n_feeds`` feeds observes a different pixel at the same time, so
    that ``t_pix = survey_time * n_feeds * beam_area / survey_area``.

    Parameters
    ----------
    nu_obs : un.Quantity[un.GHz]
        Observed frequency, which sets the beam size.
    diameter : un.Quantity[un.m]
        Diameter of the primary dish.
    survey_area : un.Quantity[un.deg**2]
        Solid angle covered by the survey.
    survey_time : un.Quantity[un.hour]
        Total on-sky integration time of the survey.
    n_feeds : int
        Number of independent spatial feeds observing at once.

    Returns
    -------
    un.Quantity[un.s]
        Integration time per pixel.
    """
    n_pixels = survey_area / beam_solid_angle(nu_obs, diameter)
    return (survey_time * n_feeds / n_pixels).to(un.s)


def _atlast_sensitivity(
    nu_obs: un.Quantity[un.GHz],
    bandwidth: un.Quantity[un.GHz],
    t_pix: un.Quantity[un.s],
) -> un.Quantity[un.Jy]:
    """Compute the rms flux density per beam with the AtLAST sensitivity calculator."""
    try:
        from atlast_sc.calculator import Calculator
    except ImportError as err:
        raise ImportError(
            "This instrument has no tabulated sensitivity. Computing it requires the "
            "AtLAST sensitivity calculator: install the 'atlast_sc' package, or set "
            "the 'sensitivity' parameter of the instrument."
        ) from err

    calculator = Calculator(
        {
            "t_int": {"value": t_pix.to_value(un.s), "unit": "s"},
            "bandwidth": {"value": bandwidth.to_value(un.GHz), "unit": "GHz"},
            "obs_freq": {"value": nu_obs.to_value(un.GHz), "unit": "GHz"},
        }
    )
    return un.Quantity(calculator.calculate_sensitivity()).to(un.Jy)


def calculate_noise(
    nu_obs: un.Quantity[un.GHz], noise_template: LIMInstrument | str
) -> dict[str, Any]:
    """Calculate the noise properties of an instrument at one observed frequency.

    The sensitivity of the instrument is converted to an rms surface brightness per
    beam-sized pixel. A noise-equivalent intensity (NEI) gives ``NEI / sqrt(t_pix)``,
    a noise-equivalent flux density (NEFD) gives ``NEFD / beam_area / sqrt(t_pix)``,
    and an rms flux density per beam gives ``rms / beam_area``.

    Parameters
    ----------
    nu_obs : un.Quantity[un.GHz]
        Observed frequency.
    noise_template : LIMInstrument or str
        The instrument, or the name of a built-in instrument template.

    Returns
    -------
    dict
        The noise properties. The rms surface brightness per pixel, in Jy/sr, is
        under ``"RMS_sigma"``; the other entries are the instrument parameters, the
        channel bandwidth ``"bw_sens"``, the redshift range of [CII] in the band
        (``"z_min"``, ``"z_max"``), the pixel integration time ``"t_pix"``, and the
        beam ``"fwhm"`` and ``"beam_area"``.

    Raises
    ------
    ValueError
        If ``noise_template`` is not the name of a built-in template.
    ImportError
        If the instrument has no tabulated sensitivity and ``atlast_sc`` is not
        installed.
    """
    instrument = _as_instrument(noise_template)

    bandwidth = nu_obs / instrument.spectral_resolution
    beam_area = instrument.beam_solid_angle(nu_obs)

    if instrument.pixel_time is None:
        t_pix = calculate_tpix(
            nu_obs,
            diameter=instrument.dish_diameter,
            survey_area=instrument.survey_area,
            survey_time=instrument.survey_time,
            n_feeds=instrument.n_feeds,
        )
    else:
        t_pix = instrument.pixel_time.to(un.s)

    sensitivity = instrument.sensitivity_at(nu_obs)
    if sensitivity is None:
        sensitivity = _atlast_sensitivity(nu_obs, bandwidth, t_pix)

    if sensitivity.unit.is_equivalent(un.Jy * un.s**0.5 / un.sr):
        rms = sensitivity / np.sqrt(t_pix)
    elif sensitivity.unit.is_equivalent(un.Jy * un.s**0.5):
        rms = sensitivity / beam_area / np.sqrt(t_pix)
    else:
        rms = sensitivity / beam_area

    return {
        "diameter": instrument.dish_diameter,
        "frequency resolution": instrument.spectral_resolution,
        "nu_sens": nu_obs,
        "bw_sens": bandwidth,
        "f_min": instrument.min_frequency,
        "f_max": instrument.max_frequency,
        "z_min": (NU_CII_REST / instrument.max_frequency).to_value("") - 1,
        "z_max": (NU_CII_REST / instrument.min_frequency).to_value("") - 1,
        "N_feed": instrument.n_feeds,
        "t_surv": instrument.survey_time,
        "surv_area": instrument.survey_area,
        "t_pix": t_pix,
        "fwhm": instrument.beam_fwhm(nu_obs),
        "beam_area": beam_area,
        "NEFD or NEI": sensitivity,
        "RMS_sigma": rms.to(un.Jy / un.sr),
    }


def add_cii_noise(
    lc_obj: LightCone,
    noise_template: LIMInstrument | str,
    seed: int | None = None,
) -> LightCone:
    """Mock-observe a [CII] lightcone with a line-intensity-mapping instrument.

    For each lightcone slice at observed frequency ``nu``, the [CII] surface
    brightness is summed over the slices within one spectral channel
    (``nu +/- nu / (2 R)``), smoothed with the Gaussian beam of the instrument, and
    Gaussian white noise with the rms from :func:`calculate_noise` is added.

    Parameters
    ----------
    lc_obj : LightCone
        Lightcone with a ``"CII_BrightnessDensity"`` field in Jy/sr, as produced by
        :func:`~.cii_lightcone.gen_lc`.
    noise_template : LIMInstrument or str
        The instrument, or the name of a built-in instrument template.
    seed : int, optional
        Seed for the noise realisation. If None, a random seed is used.

    Returns
    -------
    LightCone
        The input lightcone, with three new fields in Jy/sr, named after the
        instrument: ``"noise_<name>"`` (the noise),  ``"Bcii_<name>"`` (the
        beam-smoothed signal) and ``"Bcii_<name>_noise"`` (their sum).

    Raises
    ------
    ValueError
        If ``noise_template`` is not the name of a built-in template.
    """
    instrument = _as_instrument(noise_template)
    nu_lc = NU_CII_REST / (1 + lc_obj.lightcone_redshifts)
    b_cii = lc_obj.lightcones["CII_BrightnessDensity"]
    cell_size = (
        lc_obj.simulation_options.BOX_LEN / lc_obj.simulation_options.HII_DIM * un.Mpc
    )
    rng = np.random.default_rng(seed)

    noise = np.zeros(b_cii.shape)
    b_conv = np.zeros(b_cii.shape)

    logger.info("Adding %s noise (seed=%s)", instrument.name, seed)

    for i, nu_obs in enumerate(nu_lc):
        noise_info = calculate_noise(nu_obs, instrument)
        rms = noise_info["RMS_sigma"].to_value(un.Jy / un.sr)

        # Angle subtended by a (comoving) simulation cell at this redshift.
        z_obs = (NU_CII_REST / nu_obs).to_value("") - 1
        d_a = lc_obj.cosmo_params.cosmo.angular_diameter_distance(z_obs)
        cell_angle = (cell_size / (1 + z_obs) / d_a).to(
            un.rad, equivalencies=un.dimensionless_angles()
        )
        beam_sigma_cells = (
            noise_info["fwhm"] * gaussian_fwhm_to_sigma / cell_angle
        ).to_value("")

        half_bw = noise_info["bw_sens"] / 2
        in_channel = (nu_lc > nu_obs - half_bw) & (nu_lc < nu_obs + half_bw)
        b_channel = b_cii[..., in_channel].sum(axis=-1)

        noise[..., i] = rng.normal(loc=0.0, scale=1.0, size=b_channel.shape) * rms
        b_conv[..., i] = gaussian_filter(b_channel, sigma=beam_sigma_cells)

    lc_obj.lightcones[f"noise_{instrument.name}"] = noise
    lc_obj.lightcones[f"Bcii_{instrument.name}"] = b_conv
    lc_obj.lightcones[f"Bcii_{instrument.name}_noise"] = b_conv + noise

    return lc_obj
