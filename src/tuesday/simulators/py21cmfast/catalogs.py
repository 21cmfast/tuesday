"""Make tuesday catalogues from 21cmFAST halo catalogues.

:func:`catalog_from_halo_catalog` turns a 21cmFAST ``HaloCatalog`` or
``PerturbedHaloCatalog`` at one redshift into a
:class:`tuesday.core.catalogs.CoevalCatalog`, with the halo positions, the galaxy
properties of the halos as features, the volume of the simulation box and the
redshift. :func:`get_halo_properties` computes the galaxy
properties of the halos of a ``HaloCatalog`` without feedback, as 21cmFAST does.
"""

import numpy as np
from astropy import units as un
from py21cmfast import InitialConditions
from py21cmfast.wrapper.cfuncs import convert_halo_properties
from py21cmfast.wrapper.outputs import HaloCatalog, PerturbedHaloCatalog

from tuesday.core.catalogs import CoevalCatalog

#: Units of the halo properties that 21cmFAST computes, by feature name.
HALO_PROPERTY_UNITS = {
    "halo_mass": un.Msun,
    "stellar_mass_acg": un.Msun,
    "sfr_acg": un.Msun / un.s,
    "stellar_mass_mcg": un.Msun,
    "sfr_mcg": un.Msun / un.s,
}

# Names of the halo properties in a PerturbedHaloCatalog, by feature name.
_PERTURBED_NAMES = {
    "halo_mass": "halo_masses",
    "stellar_mass_acg": "stellar_masses_acg",
    "sfr_acg": "sfr_acg",
    "stellar_mass_mcg": "stellar_masses_mcg",
    "sfr_mcg": "sfr_mcg",
}


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


def catalog_from_halo_catalog(
    halo_catalog: HaloCatalog | PerturbedHaloCatalog,
    initial_conditions: InitialConditions | None = None,
) -> CoevalCatalog:
    """Make a tuesday catalogue from a 21cmFAST halo catalogue.

    The features are the halo mass, and the stellar mass and star formation rate of
    the atomic-cooling galaxies (``"stellar_mass_acg"``, ``"sfr_acg"``) and, in runs
    with minihalos (``USE_MCGS``), of the molecular-cooling galaxies
    (``"stellar_mass_mcg"``, ``"sfr_mcg"``); see :data:`HALO_PROPERTY_UNITS`. For a
    ``PerturbedHaloCatalog`` they are its stored properties, i.e. those of the run,
    with the minihalo and reionization feedback of the run if
    :func:`py21cmfast.perturb_halo_catalog` was given the previous node's ``TsBox``
    and ``IonizedBox``. For a ``HaloCatalog`` they are those of the galaxy model
    alone, without feedback (see :func:`get_halo_properties`).

    Parameters
    ----------
    halo_catalog : HaloCatalog or PerturbedHaloCatalog
        Halo catalogue at a single redshift, e.g. from
        ``RunCache.get_output_struct_at_z("HaloCatalog", z)``. Only its first
        ``n_halos`` entries (the real halos, not the padding) are used.
    initial_conditions : InitialConditions, optional
        Initial conditions of the run. Only needed for a ``HaloCatalog`` of a run with
        minihalos and ``V_CB_MODEL="FLUCTS"``; see :func:`get_halo_properties`.

    Returns
    -------
    CoevalCatalog
        The halos, with positions in comoving Mpc (Lagrangian for a ``HaloCatalog``,
        Eulerian for a ``PerturbedHaloCatalog``), the features above in the units
        21cmFAST gives them (SFRs in Msun/s), the volume of the simulation box and the
        redshift of the catalogue. The arrays of a ``PerturbedHaloCatalog`` are not
        copied.

    Raises
    ------
    TypeError
        If ``halo_catalog`` is neither a ``HaloCatalog`` nor a
        ``PerturbedHaloCatalog``.
    """
    if not isinstance(halo_catalog, HaloCatalog | PerturbedHaloCatalog):
        raise TypeError(
            "halo_catalog must be a HaloCatalog or a PerturbedHaloCatalog, got "
            f"{type(halo_catalog).__name__}."
        )
    names = ["halo_mass", "stellar_mass_acg", "sfr_acg"]
    if halo_catalog.inputs.astro_options.USE_MCGS:
        names += ["stellar_mass_mcg", "sfr_mcg"]

    if isinstance(halo_catalog, PerturbedHaloCatalog):
        values = {
            name: _real_halos(halo_catalog, _PERTURBED_NAMES[name]) for name in names
        }
    else:
        properties = get_halo_properties(halo_catalog, initial_conditions)
        values = {name: properties[name] for name in names}

    box_len = halo_catalog.simulation_options.BOX_LEN * un.Mpc
    return CoevalCatalog(
        positions=_real_halos(halo_catalog, "halo_coords") << un.Mpc,
        features={
            name: value << HALO_PROPERTY_UNITS[name] for name, value in values.items()
        },
        volume=box_len**3,
        redshift=halo_catalog.redshift,
    )
