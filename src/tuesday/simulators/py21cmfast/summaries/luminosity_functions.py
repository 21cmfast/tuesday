"""Compute UV luminosity functions from 21cmFAST halo catalogues.

These are thin wrappers around :mod:`tuesday.core.summaries.luminosity_functions`.
:func:`calculate_uvlf` takes a 21cmFAST halo catalogue at one redshift and returns the
UV luminosity function of its galaxies in the simulation box.
:func:`get_halo_properties` returns the galaxy properties of the halos of a
``HaloCatalog`` without feedback, as 21cmFAST computes them.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from astropy import units as un
from py21cmfast import InitialConditions
from py21cmfast.wrapper.cfuncs import convert_halo_properties
from py21cmfast.wrapper.outputs import HaloCatalog, PerturbedHaloCatalog

from tuesday.core.summaries.luminosity_functions import (
    KUV_SUN16,
    BinnedNumberDensity,
    uv_luminosity_function,
)


def _real_halos(halo_catalog: HaloCatalog | PerturbedHaloCatalog, name: str):
    """Return one halo array of a catalogue, without the padding beyond n_halos."""
    return halo_catalog.get(name)[: halo_catalog.n_halos]


def get_halo_properties(
    halo_catalog: HaloCatalog,
    initial_conditions: InitialConditions | None = None,
) -> dict[str, np.ndarray]:
    """Calculate the galaxy properties of the halos of a catalogue, without feedback.

    21cmFAST computes the properties of each halo from its mass and the random
    numbers stored in the ``HaloCatalog``, with
    :func:`py21cmfast.wrapper.cfuncs.convert_halo_properties`. Here they are computed
    without Lyman-Werner or reionization feedback, i.e. for the galaxy model alone.
    The properties with the feedback of the run are those of the
    ``PerturbedHaloCatalog`` that :func:`py21cmfast.perturb_halo_catalog` returns
    when it is given the ``TsBox`` and ``IonizedBox`` of the previous node redshift.

    Parameters
    ----------
    halo_catalog : HaloCatalog
        Halo catalogue at a single redshift. Only its first ``n_halos`` entries (the
        real halos, not the padding) are used.
    initial_conditions : InitialConditions, optional
        Initial conditions of the run. Needed only for runs with minihalos
        (``USE_MCGS``) and ``V_CB_MODEL="FLUCTS"``, in which the minihalo turnover
        mass depends on the relative velocity of baryons and dark matter in the cell
        of each halo.

    Returns
    -------
    dict of np.ndarray
        The properties that :func:`~py21cmfast.wrapper.cfuncs.convert_halo_properties`
        returns, for each of the ``n_halos`` halos: ``"halo_mass"``,
        ``"stellar_mass_acg"`` and ``"stellar_mass_mcg"`` (Msun), ``"sfr_acg"`` and
        ``"sfr_mcg"`` (Msun/s), and the others. The minihalo (MCG) properties are zero
        in runs without minihalos.

    Raises
    ------
    ValueError
        If the run has minihalos and ``V_CB_MODEL="FLUCTS"`` but no
        ``initial_conditions`` are given.

    Notes
    -----
    21cmFAST needs grids of the feedback fields of size ``HII_DIM^3``, which are set
    to no feedback here: no Lyman-Werner background, no reionization (reionization
    redshift -1) and no photo-ionization.
    """
    inputs = halo_catalog.inputs
    astro_options = inputs.astro_options
    lowres_shape = (inputs.simulation_options.HII_DIM,) * 3
    vcb = np.zeros(lowres_shape, dtype=np.float32)
    if astro_options.USE_MCGS and inputs.matter_options.V_CB_MODEL == "FLUCTS":
        if initial_conditions is None:
            raise ValueError(
                "In runs with minihalos and V_CB_MODEL='FLUCTS' the halo properties "
                "depend on the relative velocities of the initial conditions: pass "
                "the initial_conditions of the run."
            )
        vcb = initial_conditions.get("lowres_vcb")

    return convert_halo_properties(
        redshift=halo_catalog.redshift,
        inputs=inputs,
        halo_masses=_real_halos(halo_catalog, "halo_masses"),
        star_rng=_real_halos(halo_catalog, "star_rng"),
        sfr_rng=_real_halos(halo_catalog, "sfr_rng"),
        xray_rng=_real_halos(halo_catalog, "xray_rng"),
        halo_coords=_real_halos(halo_catalog, "halo_coords"),
        vcb_grid=vcb,
        J_21_LW_grid=np.zeros(lowres_shape, dtype=np.float32),
        z_re_grid=np.full(lowres_shape, -1.0, dtype=np.float32),
        Gamma12_grid=np.zeros(lowres_shape, dtype=np.float32),
    )


def calculate_uvlf(
    halo_catalog: HaloCatalog | PerturbedHaloCatalog,
    bins: int | np.ndarray | None = None,
    kuv: un.Quantity | Callable = KUV_SUN16,
    components: Literal["both", "acg", "mcg"] = "both",
    initial_conditions: InitialConditions | None = None,
) -> BinnedNumberDensity:
    """Calculate the UV luminosity function of the galaxies of a halo catalogue.

    The UV magnitude of each galaxy follows from its star formation rate (see
    :func:`~tuesday.core.summaries.luminosity_functions.sfr_to_uv_magnitude`), and
    the magnitudes are binned into a number density per magnitude in the simulation
    box with
    :func:`~tuesday.core.summaries.luminosity_functions.uv_luminosity_function`.

    The halo properties are either those of a ``PerturbedHaloCatalog``, which
    21cmFAST computes for the run (with the minihalo and reionization feedback of the
    run if :func:`py21cmfast.perturb_halo_catalog` was given the previous node's
    ``TsBox`` and ``IonizedBox``), or, for a ``HaloCatalog``, those of the galaxy
    model alone, without feedback (see :func:`get_halo_properties`).

    Parameters
    ----------
    halo_catalog : HaloCatalog or PerturbedHaloCatalog
        Halo catalogue at a single redshift, e.g. from
        ``RunCache.get_output_struct_at_z("HaloCatalog", z)``. Only its first
        ``n_halos`` entries (the real halos, not the padding) are used.
    bins : int or np.ndarray, optional
        Edges of the magnitude bins, or their number; see
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
        minihalos and ``V_CB_MODEL="FLUCTS"``; see :func:`get_halo_properties`.

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
    use_mcgs = halo_catalog.inputs.astro_options.USE_MCGS
    if components == "mcg" and not use_mcgs:
        raise ValueError("components='mcg' needs a run with minihalos (USE_MCGS=True).")

    kinds = {"acg": ["acg"], "mcg": ["mcg"], "both": ["acg", "mcg"]}[components]
    if not use_mcgs:
        kinds = ["acg"]
    if isinstance(halo_catalog, PerturbedHaloCatalog):
        sfr = sum(_real_halos(halo_catalog, f"sfr_{k}") for k in kinds)
        stellar_mass = sum(
            _real_halos(halo_catalog, f"stellar_masses_{k}") for k in kinds
        )
    elif isinstance(halo_catalog, HaloCatalog):
        properties = get_halo_properties(halo_catalog, initial_conditions)
        sfr = sum(properties[f"sfr_{k}"] for k in kinds)
        stellar_mass = sum(properties[f"stellar_mass_{k}"] for k in kinds)
    else:
        raise TypeError(
            "halo_catalog must be a HaloCatalog or a PerturbedHaloCatalog, got "
            f"{type(halo_catalog).__name__}."
        )

    box_len = halo_catalog.simulation_options.BOX_LEN * un.Mpc
    # 21cmFAST gives the SFR in Msun/s, in float32.
    return uv_luminosity_function(
        sfr.astype(np.float64) * un.Msun / un.s,
        volume=box_len**3,
        bins=bins,
        kuv=kuv,
        stellar_mass=stellar_mass.astype(np.float64) * un.Msun,
        redshift=halo_catalog.redshift,
    )
