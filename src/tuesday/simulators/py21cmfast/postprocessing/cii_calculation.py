"""Calculate [CII] 158 micron emission from 21cmFAST halo catalogues.

These are thin wrappers around :mod:`tuesday.core.postprocessing.line_intensities`.
:func:`calculate_cii` reads the halo properties from a 21cmFAST
:class:`~py21cmfast.wrapper.outputs.HaloCatalog` and returns the [CII] luminosity and
velocity-integrated flux density of each halo, which
:func:`~.cii_lightcone.gen_coeval` grids into a box. :func:`calculate_bvoxel` turns
the gridded flux density on a lightcone into a surface brightness.
"""

import numpy as np
from astropy import units as un
from py21cmfast import LightCone
from py21cmfast.wrapper.cfuncs import convert_halo_properties
from py21cmfast.wrapper.outputs import HaloCatalog

from tuesday.core.postprocessing.line_intensities import (
    CII_MODELS,
    FLUX_UNIT,
    METALLICITY_MODELS,
    cii_flux,
    cii_luminosity,
    cii_surface_brightness,
    validate_cii_model,
)


def calculate_cii(
    halo_catalog: HaloCatalog,
    model: str,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the [CII] luminosity and integrated flux density of each halo.

    The luminosity is computed with
    :func:`~tuesday.core.postprocessing.line_intensities.cii_luminosity` and the
    flux density with :func:`~tuesday.core.postprocessing.line_intensities.cii_flux`,
    using the cosmology of the simulation.

    Parameters
    ----------
    halo_catalog : HaloCatalog
        Halo catalogue at a single redshift. Only its first ``n_halos`` entries
        (the real halos, not the zero padding) are used.
    model : str
        The [CII] luminosity model, one of
        :data:`~tuesday.core.postprocessing.line_intensities.CII_MODELS`:

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
        If ``model`` is not one of the available models, or if it depends on the
        metallicity but the simulation was run without ``USE_TS_FLUCT``, in which
        case 21cmFAST does not compute halo metallicities and returns zero.
    """
    validate_cii_model(model)
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

    luminosity = cii_luminosity(
        # 21cmFAST gives the SFR in Msun/s.
        sfr=halo_props["halo_sfr"] * un.Msun / un.s,
        metallicity=halo_props["metallicity"],
        halo_mass=halo_props["halo_mass"] * un.Msun,
        redshift=redshift,
        model=model,
        seed=seed,
    )
    intensity = cii_flux(luminosity, redshift, halo_catalog.cosmo_params.cosmo)

    return luminosity.to_value(un.Lsun), intensity.to_value(FLUX_UNIT)


def calculate_bvoxel(lightcone: LightCone) -> np.ndarray:
    """Convert a [CII] flux-density lightcone to a surface brightness.

    See :func:`~tuesday.core.postprocessing.line_intensities.cii_surface_brightness`.
    The frequency width of the cells uses the cosmology of the simulation. Both
    rectilinear (``(x, y, slice)``) and angular (``(pixel, slice)``) lightcones are
    supported.

    Parameters
    ----------
    lightcone : LightCone
        Lightcone with a ``"CII_IntensityBox"`` field in Jy km/s per cell.

    Returns
    -------
    np.ndarray
        [CII] surface brightness of each cell, in Jy/sr.
    """
    cell_size = (
        lightcone.simulation_options.BOX_LEN / lightcone.simulation_options.HII_DIM
    ) * un.Mpc
    brightness = cii_surface_brightness(
        lightcone.lightcones["CII_IntensityBox"] * FLUX_UNIT,
        distances=lightcone.lightcone_distances,
        cell_size=cell_size,
        cosmo=lightcone.cosmo_params.cosmo,
    )
    return brightness.to_value(un.Jy / un.sr)
