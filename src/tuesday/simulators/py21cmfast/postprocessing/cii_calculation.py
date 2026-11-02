"""Calculate [CII] 158 micron emission from 21cmFAST halo catalogues.

The [CII] luminosity of each halo is computed from its star formation rate (and, for
some models, its metallicity or mass) using one of the empirical or simulation-based
relations in :data:`CII_MODELS`. The luminosity is converted to a velocity-integrated
flux density, which :func:`~.cii_lightcone.gen_coeval` grids into a box, and
:func:`calculate_bvoxel` turns into a surface brightness on the lightcone.
"""

from collections.abc import Callable

import numpy as np
from astropy import constants as const
from astropy import units as un
from astropy.cosmology import FLRW, z_at_value
from py21cmfast import LightCone
from py21cmfast.wrapper.cfuncs import convert_halo_properties
from py21cmfast.wrapper.outputs import HaloCatalog

#: Rest-frame frequency of the [CII] 2P3/2 -> 2P1/2 fine-structure line.
NU_CII_REST = 1900.5369 * un.GHz


def _comoving_distance_to_redshift(
    cosmo: FLRW, distances: un.Quantity[un.Mpc]
) -> np.ndarray:
    """Invert the comoving distance-redshift relation for an array of distances.

    The relation is tabulated on a fine redshift grid and interpolated, which is much
    faster than solving for each distance separately.
    """
    d_mpc = distances.to_value(un.Mpc)
    z_lo = z_at_value(cosmo.comoving_distance, d_mpc.min() * un.Mpc).value
    z_hi = z_at_value(cosmo.comoving_distance, d_mpc.max() * un.Mpc).value
    z_grid = np.linspace(max(z_lo - 0.01, 0.0), z_hi + 0.01, 10_000)
    d_grid = cosmo.comoving_distance(z_grid).to_value(un.Mpc)
    return np.interp(d_mpc, d_grid, z_grid)


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


def calculate_cii(
    halo_catalog: HaloCatalog,
    model: str,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the [CII] luminosity and integrated flux density of each halo.

    The velocity-integrated flux density follows from the luminosity as
    ``L [Lsun] = 1.04e-3 S dv [Jy km/s] nu_obs [GHz] D_L^2 [Mpc^2]``
    (Solomon & Vanden Bout 2005, ARA&A, 43, 677), with no lensing magnification.
    The luminosity distance uses the cosmology of the simulation.

    Parameters
    ----------
    halo_catalog : HaloCatalog
        Halo catalogue at a single redshift. Only its first ``n_halos`` entries
        (the real halos, not the zero padding) are used.
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
    luminosity : np.ndarray
        [CII] luminosity of each of the ``n_halos`` halos, in Lsun.
    intensity : np.ndarray
        Velocity-integrated [CII] flux density of each halo, in Jy km/s.

    Raises
    ------
    ValueError
        If ``model`` is not one of :data:`CII_MODELS`, or if it depends on the
        metallicity (:data:`METALLICITY_MODELS`) but the simulation was run without
        ``USE_TS_FLUCT``, in which case 21cmFAST does not compute halo
        metallicities and returns zero.
    """
    if model not in _LOG_LCII:
        raise ValueError(
            f"Unknown [CII] model '{model}'. Available models are {list(CII_MODELS)}."
        )
    if (
        model in METALLICITY_MODELS
        and not halo_catalog.inputs.astro_options.USE_TS_FLUCT
    ):
        raise ValueError(
            f"The [CII] model '{model}' needs halo metallicities, which 21cmFAST "
            "only computes when USE_TS_FLUCT=True. Rerun the simulation with "
            "USE_TS_FLUCT=True, or use one of the SFR-only models: "
            f"{[m for m in CII_MODELS if m not in METALLICITY_MODELS]}."
        )
    rng = np.random.default_rng(seed)

    redshift = halo_catalog.redshift
    # 21cmFAST pads the halo arrays with zeros beyond the n_halos real halos.
    n_halos = halo_catalog.n_halos
    halo_props = convert_halo_properties(
        redshift=redshift,
        inputs=halo_catalog.inputs,
        halo_masses=halo_catalog.get("halo_masses")[:n_halos],
        halo_coords=halo_catalog.get("halo_coords")[:n_halos],
        star_rng=halo_catalog.get("star_rng")[:n_halos],
        sfr_rng=halo_catalog.get("sfr_rng")[:n_halos],
        xray_rng=halo_catalog.get("xray_rng")[:n_halos],
    )
    # 21cmFAST gives the SFR in Msun/s.
    sfr_yr = halo_props["halo_sfr"] * un.yr.to(un.s)

    log_lum = _LOG_LCII[model](
        sfr_yr, halo_props["metallicity"], halo_props["halo_mass"], redshift, rng
    )
    luminosity = 10**log_lum

    cosmo = halo_catalog.cosmo_params.cosmo
    d_lum = cosmo.luminosity_distance(redshift).to_value(un.Mpc)
    nu_obs = NU_CII_REST.to_value(un.GHz) / (1 + redshift)
    intensity = luminosity / 1.04e-3 / d_lum**2 / nu_obs

    return luminosity, intensity


def calculate_bvoxel(lightcone: LightCone) -> np.ndarray:
    """Convert a [CII] flux-density lightcone to a surface brightness.

    Each cell's velocity-integrated flux density ``S dv`` is spread over the solid
    angle and frequency width of the cell: ``B = S dv nu_obs / (c dnu Omega)``.
    The frequency width uses the cosmology of the simulation. Both rectilinear
    (``(x, y, slice)``) and angular (``(pixel, slice)``) lightcones are supported.

    Parameters
    ----------
    lightcone : LightCone
        Lightcone with a ``"CII_IntensityBox"`` field in Jy km/s per cell.

    Returns
    -------
    np.ndarray
        [CII] surface brightness of each cell, in Jy/sr.
    """
    redshifts = lightcone.lightcone_redshifts
    nu_obs = NU_CII_REST / (1 + redshifts)
    cell_size = (
        lightcone.simulation_options.BOX_LEN / lightcone.simulation_options.HII_DIM
    ) * un.Mpc
    d_comv = lightcone.lightcone_distances

    solid_angle = (cell_size / d_comv) ** 2 * un.sr
    cosmo = lightcone.cosmo_params.cosmo
    z_front = _comoving_distance_to_redshift(cosmo, d_comv - cell_size / 2)
    z_back = _comoving_distance_to_redshift(cosmo, d_comv + cell_size / 2)
    delta_nu = NU_CII_REST / (1 + z_front) - NU_CII_REST / (1 + z_back)

    conversion = nu_obs / (const.c * delta_nu * solid_angle)
    flux = lightcone.lightcones["CII_IntensityBox"] * un.Jy * un.km / un.s
    return (flux * conversion).to_value(un.Jy / un.sr)
