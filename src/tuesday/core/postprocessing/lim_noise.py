"""Instrumental noise for line-intensity-mapping (LIM) surveys.

The instrument and survey are specified by a
:class:`~.lim_instruments.LIMInstrument`, usually loaded from a template with
:meth:`~.lim_instruments.LIMInstrument.from_template`.
:func:`calculate_noise` gives the rms surface brightness per pixel at one observed
frequency, and :func:`observe_lim_lightcone` mock-observes a surface-brightness
lightcone: at each frequency, the signal is averaged over one spectral channel of the
instrument and smoothed with the Gaussian instrument beam, and white noise with the
per-pixel rms of the survey is drawn.
"""

import logging
import warnings
from typing import Any

import numpy as np
from astropy import units as un
from astropy.cosmology import FLRW
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.ndimage import gaussian_filter

from .lim_instruments import LIMInstrument, beam_solid_angle
from .line_intensities import NU_CII_REST

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
    nu_obs: un.Quantity[un.GHz],
    noise_template: LIMInstrument | str,
    rest_frequency: un.Quantity[un.GHz] = NU_CII_REST,
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
        The instrument, or the name of a built-in instrument template (see
        ``lim_instrument_templates/`` for every value and its reference):

        - ``"ccatp"``: CCAT-prime / FYST EoR-Spec (CCAT-Prime Collaboration 2023,
          arXiv:2107.10364; Horlaville et al. 2024, arXiv:2309.15733).
        - ``"ccatp2"``: an extended CCAT-prime survey (Horlaville et al. 2024,
          arXiv:2309.15733).
        - ``"time"``: TIME (Chung et al. 2020, ApJ, 892, 51; Sun et al. 2021,
          arXiv:2012.09160).
        - ``"time2"``: an upgraded TIME survey (Sun et al. 2021, arXiv:2012.09160).
        - ``"atlast"``: AtLAST, with the sensitivity computed by ``atlast_sc``.
    rest_frequency : un.Quantity[un.GHz], optional
        Rest-frame frequency of the observed line, used for the redshift range of the
        band. Defaults to [CII].

    Returns
    -------
    dict
        The noise properties. The rms surface brightness per pixel, in Jy/sr, is
        under ``"RMS_sigma"``; the other entries are the instrument parameters, the
        channel bandwidth ``"bw_sens"``, the redshift range of the line in the band
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
        "z_min": (rest_frequency / instrument.max_frequency).to_value("") - 1,
        "z_max": (rest_frequency / instrument.min_frequency).to_value("") - 1,
        "N_feed": instrument.n_feeds,
        "t_surv": instrument.survey_time,
        "surv_area": instrument.survey_area,
        "t_pix": t_pix,
        "fwhm": instrument.beam_fwhm(nu_obs),
        "beam_area": beam_area,
        "NEFD or NEI": sensitivity,
        "RMS_sigma": rms.to(un.Jy / un.sr),
    }


def observe_lim_lightcone(
    brightness: un.Quantity[un.Jy / un.sr],
    redshifts: np.ndarray,
    cell_size: un.Quantity[un.Mpc],
    cosmo: FLRW,
    noise_template: LIMInstrument | str,
    rest_frequency: un.Quantity[un.GHz] = NU_CII_REST,
    seed: int | None = None,
) -> tuple[un.Quantity[un.Jy / un.sr], un.Quantity[un.Jy / un.sr]]:
    """Mock-observe a line surface-brightness lightcone with a LIM instrument.

    For each lightcone slice at observed frequency ``nu``, the surface brightness is
    averaged over the slices within one spectral channel (``nu +/- nu / (2 R)``) and
    smoothed with the Gaussian beam of the instrument, and Gaussian white noise with
    the rms from :func:`calculate_noise` is drawn. Slices outside the frequency range
    of the instrument are set to NaN.

    Parameters
    ----------
    brightness : un.Quantity[un.Jy / un.sr]
        Rectilinear lightcone of the line surface brightness, with shape
        ``(x, y, slice)``.
    redshifts : np.ndarray
        Redshift of each slice.
    cell_size : un.Quantity[un.Mpc]
        Comoving side length of a (cubic) cell.
    cosmo : FLRW
        Cosmology used for the angular size of a cell.
    noise_template : LIMInstrument or str
        The instrument, or the name of a built-in instrument template; see
        :func:`calculate_noise` for the built-in templates.
    rest_frequency : un.Quantity[un.GHz], optional
        Rest-frame frequency of the line. Defaults to [CII].
    seed : int, optional
        Seed for the noise realisation. If None, a random seed is used.

    Returns
    -------
    signal : un.Quantity[un.Jy / un.sr]
        The channel-averaged, beam-smoothed signal.
    noise : un.Quantity[un.Jy / un.sr]
        The noise realisation.

    Raises
    ------
    ValueError
        If ``noise_template`` is not the name of a built-in template, or
        ``brightness`` is not a 3D (rectilinear) lightcone.

    Warns
    -----
    UserWarning
        If some slices are outside the frequency range of the instrument.
    """
    instrument = _as_instrument(noise_template)
    if brightness.ndim != 3:
        raise ValueError(
            "A rectilinear lightcone with (x, y, slice) fields is needed; "
            f"got a field with shape {brightness.shape}."
        )
    b_line = brightness.to_value(un.Jy / un.sr)
    nu_lc = rest_frequency / (1 + np.asarray(redshifts))
    rng = np.random.default_rng(seed)

    noise = np.zeros(b_line.shape)
    signal = np.zeros(b_line.shape)

    in_band = (nu_lc >= instrument.min_frequency) & (nu_lc <= instrument.max_frequency)
    if not in_band.all():
        warnings.warn(
            f"{np.sum(~in_band)} of {nu_lc.size} lightcone slices are outside the "
            f"{instrument.min_frequency} - {instrument.max_frequency} band of "
            f"'{instrument.name}'; they are set to NaN.",
            stacklevel=2,
        )
    noise[..., ~in_band] = np.nan
    signal[..., ~in_band] = np.nan

    logger.info("Adding %s noise (seed=%s)", instrument.name, seed)

    for i in np.flatnonzero(in_band):
        nu_obs = nu_lc[i]
        noise_info = calculate_noise(nu_obs, instrument, rest_frequency)
        rms = noise_info["RMS_sigma"].to_value(un.Jy / un.sr)

        # Angle subtended by a (comoving) simulation cell at this redshift.
        z_obs = (rest_frequency / nu_obs).to_value("") - 1
        d_a = cosmo.angular_diameter_distance(z_obs)
        cell_angle = (cell_size / (1 + z_obs) / d_a).to(
            un.rad, equivalencies=un.dimensionless_angles()
        )
        beam_sigma_cells = (
            noise_info["fwhm"] * gaussian_fwhm_to_sigma / cell_angle
        ).to_value("")

        half_bw = noise_info["bw_sens"] / 2
        in_channel = (nu_lc > nu_obs - half_bw) & (nu_lc < nu_obs + half_bw)
        b_channel = b_line[..., in_channel].mean(axis=-1)

        noise[..., i] = rng.normal(loc=0.0, scale=1.0, size=b_channel.shape) * rms
        signal[..., i] = gaussian_filter(b_channel, sigma=beam_sigma_cells)

    return signal * un.Jy / un.sr, noise * un.Jy / un.sr
