"""Compute UV luminosity functions from 21cmFAST halo catalogues.

:func:`calculate_uvlf` turns a 21cmFAST halo catalogue at one redshift into a tuesday
catalogue with :func:`~tuesday.simulators.py21cmfast.catalogs.catalog_from_halo_catalog`
and returns the UV luminosity function of its galaxies in the simulation box, from
:func:`tuesday.core.summaries.luminosity_functions.uv_luminosity_function`.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from astropy import units as un
from py21cmfast import InitialConditions
from py21cmfast.wrapper.outputs import HaloCatalog, PerturbedHaloCatalog

from tuesday.core.summaries.luminosity_functions import (
    KUV_SUN16,
    BinnedNumberDensity,
    uv_luminosity_function,
)

from ..catalogs import catalog_from_halo_catalog


def calculate_uvlf(
    halo_catalog: HaloCatalog | PerturbedHaloCatalog,
    bins: int | un.Quantity | None = None,
    kuv: un.Quantity | Callable = KUV_SUN16,
    components: Literal["both", "acg", "mcg"] = "both",
    initial_conditions: InitialConditions | None = None,
) -> BinnedNumberDensity:
    """Calculate the UV luminosity function of the galaxies of a halo catalogue.

    The halos are turned into a tuesday catalogue with
    :func:`~tuesday.simulators.py21cmfast.catalogs.catalog_from_halo_catalog`, so the
    halo properties are those of the run for a ``PerturbedHaloCatalog`` and those of
    the galaxy model alone, without feedback, for a ``HaloCatalog``. The star
    formation rates and stellar masses of the chosen ``components`` are added up into
    the features ``"sfr"`` and ``"stellar_mass"``, and the luminosity function per
    Mpc^3 in the simulation box is computed with
    :func:`~tuesday.core.summaries.luminosity_functions.uv_luminosity_function`.

    Parameters
    ----------
    halo_catalog : HaloCatalog or PerturbedHaloCatalog
        Halo catalogue at a single redshift, e.g. from
        ``RunCache.get_output_struct_at_z("HaloCatalog", z)``. Only its first
        ``n_halos`` entries (the real halos, not the padding) are used.
    bins : int or un.Quantity, optional
        Edges of the magnitude bins (in ``un.mag``), or their number; see
        :func:`~tuesday.core.summaries.luminosity_functions.bin_number_density`. By
        default 0.5 mag wide from ``M_UV = -24`` to ``-12``.
    kuv : un.Quantity or callable, optional
        The conversion factor ``K_UV = SFR / L_UV``; by default the value 21cmFAST
        uses. See
        :func:`~tuesday.core.summaries.luminosity_functions.sfr_to_uv_magnitude`. A
        function is called with the SFR and stellar mass of the galaxies (of the
        same ``components``) and the redshift of the catalogue.
    components : {"both", "acg", "mcg"}, optional
        Which galaxies of each halo contribute: those of atomic-cooling galaxies
        (``"acg"``), of molecular-cooling galaxies in minihalos (``"mcg"``), or both,
        whose star formation rates (and stellar masses) are then summed. In runs
        without minihalos only ``"acg"`` contributes.
    initial_conditions : InitialConditions, optional
        Initial conditions of the run. Only needed for a ``HaloCatalog`` of a run with
        minihalos and ``V_CB_MODEL="FLUCTS"``; see
        :func:`~tuesday.simulators.py21cmfast.catalogs.get_halo_properties`.

    Returns
    -------
    BinnedNumberDensity
        The luminosity function, per Mpc^3 and per magnitude, with the counts and
        Poisson errors.

    Raises
    ------
    ValueError
        If ``components`` is not one of the options, or is ``"mcg"`` in a run
        without minihalos.
    TypeError
        If ``halo_catalog`` is neither a ``HaloCatalog`` nor a
        ``PerturbedHaloCatalog``.
    """
    if components not in ("both", "acg", "mcg"):
        raise ValueError(
            f"components must be 'both', 'acg' or 'mcg', got '{components}'."
        )
    if not isinstance(halo_catalog, HaloCatalog | PerturbedHaloCatalog):
        raise TypeError(
            "halo_catalog must be a HaloCatalog or a PerturbedHaloCatalog, got "
            f"{type(halo_catalog).__name__}."
        )
    use_mcgs = halo_catalog.inputs.astro_options.USE_MCGS
    if components == "mcg" and not use_mcgs:
        raise ValueError("components='mcg' needs a run with minihalos (USE_MCGS=True).")

    catalog = catalog_from_halo_catalog(halo_catalog, initial_conditions)
    kinds = {"acg": ["acg"], "mcg": ["mcg"], "both": ["acg", "mcg"]}[components]
    if not use_mcgs:
        kinds = ["acg"]
    # 21cmFAST stores the properties in float32; add them up in float64.
    catalog = catalog.with_features(
        sfr=sum(catalog[f"sfr_{k}"].astype(np.float64) for k in kinds),
        stellar_mass=sum(
            catalog[f"stellar_mass_{k}"].astype(np.float64) for k in kinds
        ),
    )
    return uv_luminosity_function(catalog, bins=bins, kuv=kuv)
