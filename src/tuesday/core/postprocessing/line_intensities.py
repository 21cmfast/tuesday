"""Simulate line intensities from halo catalogues.

The [CII] 158 micron luminosity of each halo is computed from its star formation rate
(and, for some models, its metallicity or mass) with one of the empirical or
simulation-based relations in :data:`CII_MODELS`. The luminosity is converted to a
velocity-integrated flux density with :func:`cii_flux`, and a gridded flux density is
converted to a surface brightness with :func:`cii_surface_brightness`.

These functions work on plain arrays and do not depend on any simulation code; see
:mod:`tuesday.simulators.py21cmfast.postprocessing` for wrappers that read 21cmFAST
outputs.
"""

from collections.abc import Callable

import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import FLRW

from ..cosmology import comoving_distance_to_redshift

#: Rest-frame frequency of the [CII] 2P3/2 -> 2P1/2 fine-structure line.
NU_CII_REST = 1900.5369 * un.GHz

FLUX_UNIT = un.Jy * un.km / un.s


def _mean_preserving_lognormal(
    values: np.ndarray, sigma_dex: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw log-normal scatter of ``sigma_dex`` around ``values``, keeping the mean."""
    sigma_ln = sigma_dex * np.log(10)
    return values * np.exp(
        -(sigma_ln**2) / 2 + rng.standard_normal(len(values)) * sigma_ln
    )


def _log_lcii_vallini15(
    sfr: np.ndarray,
    metallicity: np.ndarray,
    halo_mass: np.ndarray,
    redshift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """[CII]-SFR-metallicity relation of Vallini et al. (2015), ApJ, 813, 36.

    Fitted to simulated galaxies at z = 6.6 with SFR = 0.1-100 Msun/yr and gas
    metallicity 0.05-1 Zsun. The metallicity has a 0.054 dex scatter, from the
    fundamental metallicity relation of Curti et al. (2020), MNRAS, 491, 944.
    """
    log_sfr = np.log10(sfr)
    log_z = np.log10(_mean_preserving_lognormal(metallicity, 0.054, rng))
    return (
        7.0 + 1.2 * log_sfr + 0.021 * log_z + 0.012 * log_sfr * log_z - 0.74 * log_z**2
    )


def _log_lcii_horlaville24(
    sfr: np.ndarray,
    metallicity: np.ndarray,
    halo_mass: np.ndarray,
    redshift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """[CII]-HI-metallicity relation of Horlaville et al. (2024), arXiv:2309.15733.

    ``L_CII = 0.024 M_HI Z``, valid for galaxies at z = 0-8 with stellar masses of
    1e5-1e11 Msun. The HI mass follows from the halo mass, and the metallicity has
    a 0.4 dex scatter.
    """
    m0_hi, m_min, alpha_hi = 1.9e9, 2e10, 0.74
    m_hi = (
        m0_hi * (halo_mass / m_min) ** alpha_hi * np.exp(-((m_min / halo_mass) ** 0.35))
    )
    z_sample = _mean_preserving_lognormal(metallicity, 0.4, rng)
    return np.log10(0.024) + np.log10(m_hi) + np.log10(z_sample)


def _log_lcii_delooze14(
    sfr: np.ndarray,
    metallicity: np.ndarray,
    halo_mass: np.ndarray,
    redshift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """[CII]-SFR relation of De Looze et al. (2014), A&A, 568, A62.

    Calibrated on local galaxies, with a 0.4 dex scatter.
    """
    return 7.10 + 1.05 * np.log10(sfr) + rng.standard_normal(len(sfr)) * 0.4


def _log_lcii_lagache18(
    sfr: np.ndarray,
    metallicity: np.ndarray,
    halo_mass: np.ndarray,
    redshift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Redshift-dependent [CII]-SFR relation of Lagache et al. (2018), A&A, 609, A130.

    Valid at z = 4-8 for L_CII > 1e7 Lsun and SFR < 1e3 Msun/yr, with a 0.5 dex
    scatter.
    """
    log_lum = np.log10(sfr) * (1.4 - 0.07 * redshift) + (7.1 - 0.07 * redshift)
    return log_lum + rng.standard_normal(len(sfr)) * 0.5


def _log_lcii_casavecchia24(
    sfr: np.ndarray,
    metallicity: np.ndarray,
    halo_mass: np.ndarray,
    redshift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """[CII]-SFR relation of Casavecchia et al. (2025), A&A, 693, A119.

    Valid at z = 6-12, with a 0.5 dex scatter.
    """
    log_lum = 1.19 * np.log10(sfr) - 0.12 * redshift + 7.58
    return log_lum + rng.standard_normal(len(sfr)) * 0.5


_LOG_LCII: dict[str, Callable[..., np.ndarray]] = {
    "Vallini15": _log_lcii_vallini15,
    "Horlaville24": _log_lcii_horlaville24,
    "deLooze14": _log_lcii_delooze14,
    "Lagache18": _log_lcii_lagache18,
    "Casavecchia24": _log_lcii_casavecchia24,
}

#: Names of the available [CII] luminosity models.
CII_MODELS = tuple(_LOG_LCII)

#: Models that depend on the halo metallicity.
METALLICITY_MODELS = ("Vallini15", "Horlaville24")


def validate_cii_model(model: str) -> None:
    """Check that ``model`` is one of :data:`CII_MODELS`.

    Parameters
    ----------
    model : str
        Name of a [CII] luminosity model.

    Raises
    ------
    ValueError
        If ``model`` is not one of :data:`CII_MODELS`.
    """
    if model not in _LOG_LCII:
        raise ValueError(
            f"Unknown [CII] model '{model}'. Available models are {list(CII_MODELS)}."
        )


def cii_luminosity(
    sfr: un.Quantity[un.Msun / un.yr],
    metallicity: np.ndarray,
    halo_mass: un.Quantity[un.Msun],
    redshift: float,
    model: str,
    seed: int | np.random.Generator | None = None,
) -> un.Quantity[un.Lsun]:
    """Calculate the [CII] luminosity of each halo.

    Parameters
    ----------
    sfr : un.Quantity[un.Msun / un.yr]
        Star formation rate of each halo.
    metallicity : np.ndarray
        Gas metallicity of each halo, in units of the solar metallicity. Only used by
        the models in :data:`METALLICITY_MODELS`.
    halo_mass : un.Quantity[un.Msun]
        Mass of each halo. Only used by ``"Horlaville24"``.
    redshift : float
        Redshift of the halos.
    model : str
        The [CII] luminosity model, one of :data:`CII_MODELS`:

        - ``"Vallini15"``: Vallini et al. (2015), ApJ, 813, 36.
        - ``"Horlaville24"``: Horlaville et al. (2024), arXiv:2309.15733.
        - ``"deLooze14"``: De Looze et al. (2014), A&A, 568, A62.
        - ``"Lagache18"``: Lagache et al. (2018), A&A, 609, A130.
        - ``"Casavecchia24"``: Casavecchia et al. (2025), A&A, 693, A119.
    seed : int or np.random.Generator, optional
        Seed or generator for the scatter of the model. If None, a random seed is
        used.

    Returns
    -------
    un.Quantity[un.Lsun]
        [CII] luminosity of each halo.

    Raises
    ------
    ValueError
        If ``model`` is not one of :data:`CII_MODELS`.
    """
    validate_cii_model(model)
    rng = np.random.default_rng(seed)
    log_lum = _LOG_LCII[model](
        sfr.to_value(un.Msun / un.yr),
        np.asarray(metallicity),
        halo_mass.to_value(un.Msun),
        redshift,
        rng,
    )
    return 10**log_lum * un.Lsun


def cii_flux(
    luminosity: un.Quantity[un.Lsun], redshift: float, cosmo: FLRW
) -> un.Quantity[FLUX_UNIT]:
    """Convert [CII] luminosities to velocity-integrated flux densities.

    Uses ``L [Lsun] = 1.04e-3 S dv [Jy km/s] nu_obs [GHz] D_L^2 [Mpc^2]``
    (Solomon & Vanden Bout 2005, ARA&A, 43, 677), with no lensing magnification.

    Parameters
    ----------
    luminosity : un.Quantity[un.Lsun]
        [CII] luminosity of each source.
    redshift : float
        Redshift of the sources.
    cosmo : FLRW
        Cosmology used for the luminosity distance.

    Returns
    -------
    un.Quantity[un.Jy * un.km / un.s]
        Velocity-integrated [CII] flux density of each source.
    """
    d_lum = cosmo.luminosity_distance(redshift).to_value(un.Mpc)
    nu_obs = NU_CII_REST.to_value(un.GHz) / (1 + redshift)
    return luminosity.to_value(un.Lsun) / 1.04e-3 / d_lum**2 / nu_obs * FLUX_UNIT


def cii_surface_brightness(
    flux: un.Quantity[FLUX_UNIT],
    distances: un.Quantity[un.Mpc],
    cell_size: un.Quantity[un.Mpc],
    cosmo: FLRW,
) -> un.Quantity[un.Jy / un.sr]:
    """Convert the [CII] flux density in each lightcone cell to a surface brightness.

    Each cell's velocity-integrated flux density ``S dv`` is spread over the solid
    angle and frequency width of the cell: ``B = S dv nu_obs / (c dnu Omega)``.

    Parameters
    ----------
    flux : un.Quantity[un.Jy * un.km / un.s]
        Velocity-integrated [CII] flux density in each cell. The last axis runs along
        the line of sight; the other axes can have any shape.
    distances : un.Quantity[un.Mpc]
        Comoving distance of each slice along the line of sight.
    cell_size : un.Quantity[un.Mpc]
        Comoving side length of a (cubic) cell.
    cosmo : FLRW
        Cosmology used for the distance-redshift relation.

    Returns
    -------
    un.Quantity[un.Jy / un.sr]
        [CII] surface brightness of each cell.
    """
    redshifts = comoving_distance_to_redshift(distances, cosmo)
    nu_obs = NU_CII_REST / (1 + redshifts)
    solid_angle = (cell_size / distances) ** 2 * un.sr
    z_front = comoving_distance_to_redshift(distances - cell_size / 2, cosmo)
    z_back = comoving_distance_to_redshift(distances + cell_size / 2, cosmo)
    delta_nu = NU_CII_REST / (1 + z_front) - NU_CII_REST / (1 + z_back)

    conversion = nu_obs / (const.c * delta_nu * solid_angle)
    return (flux * conversion).to(un.Jy / un.sr)
