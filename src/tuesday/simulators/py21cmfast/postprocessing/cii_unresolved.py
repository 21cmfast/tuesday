r"""[CII] emission of the halos below the 21cmFAST sampler mass.

With discrete halos (``SOURCE_MODEL='CHMF-SAMPLER'``), 21cmFAST samples halos only
above ``SAMPLER_MIN_MASS``. The halos below it are not in the ``HaloCatalog``, but
21cmFAST still adds their emission to the ``EmissivityFields``, and hence to the 21-cm
signal, by integrating the conditional halo mass function (CHMF) of each cell
(``EmissivityFields.c: add_integral_contribution``,
``map_mass.c: move_integral_emissivities``):

1. On the Lagrangian low-resolution grid, the linear density of each cell,
   ``delta = lowres_density * D(z)``, sets the CHMF of a region of the cell mass.
2. The CHMF is integrated from the minimum source mass to ``SAMPLER_MIN_MASS``,
   weighted by the mean galaxy property of a halo of mass ``M``.
3. Each Lagrangian cell is moved to its Eulerian position with the velocities of
   the initial conditions (Zel'dovich and 2LPT), and its integral is assigned to
   the grid with cloud-in-cell interpolation.

This module does the same with the mean [CII] luminosity of a halo,
``<L_CII>(M)``, in place of its mean SFR, so that the [CII] emission of these
halos is consistent with the 21-cm signal of the run. Because ``L_CII`` is a
nonlinear function of the SFR, with scatter, it cannot be computed from the gridded
SFR of the ``EmissivityFields``; it is averaged halo by halo, inside the integral:

.. math::

    L_{\rm cell} = \int_{M_{\rm min}}^{M_{\rm samp}}
    \langle L_{\rm CII}\rangle(M) \,
    \frac{dN}{dM}(M \mid \delta, M_{\rm cell}) \, dM .

``<L_CII>(M)`` is averaged over the 21cmFAST scatter of the stellar mass and SFR
(the same ``star_rng``, ``sfr_rng`` and ``xray_rng`` deviates as for sampled halos)
and over the scatter of the [CII] model, with the same 21cmFAST relations as for the
sampled halos (:func:`py21cmfast.wrapper.cfuncs.convert_halo_properties`), except
for the stellar fraction (see :func:`integral_stellar_ratio`).

Only runs without minihalos (``USE_MCGS``), without reionization feedback on the
turnover mass (``USE_REIONIZATION_PHOTOHEATING_FEEDBACK``), with ``M_MIN_in_Mass``,
without ``PERTURB_ON_HIGH_RES``, with cubic boxes and with mean (not median) scaling
relations are supported. (21cmFAST itself allows only the Sheth-Tormen and
Press-Schechter mass functions for discrete halos, for which it does not rescale the
integrated emission to a global mean.)
"""

from collections.abc import Sequence

import numpy as np
import py21cmfast as p21c
from astropy import units as un
from numpy.polynomial.hermite_e import hermegauss
from py21cmfast.c_21cmfast import lib
from py21cmfast.drivers._global_initialization import c_state
from py21cmfast.wrapper.cfuncs import (
    convert_halo_properties,
    integrate_chmf_interval,
)

from tuesday.core.postprocessing.line_intensities import (
    CII_MODELS,
    METALLICITY_MODELS,
    cii_flux,
    cii_luminosity,
    halo_metallicity,
    validate_cii_model,
)


def _check_supported(inputs: p21c.InputParameters) -> None:
    """Raise if the run uses options that are not implemented here."""
    so, mo, ao = inputs.simulation_options, inputs.matter_options, inputs.astro_options
    unsupported = {
        "SOURCE_MODEL must be 'CHMF-SAMPLER'": mo.SOURCE_MODEL != "CHMF-SAMPLER",
        "USE_MCGS (minihalos) is not supported": ao.USE_MCGS,
        # The turnover mass of each cell would then depend on the IonizedBox of the
        # previous node redshift.
        "USE_REIONIZATION_PHOTOHEATING_FEEDBACK is not supported": (
            ao.USE_REIONIZATION_PHOTOHEATING_FEEDBACK
        ),
        "M_MIN_in_Mass=False is not supported": not ao.M_MIN_in_Mass,
        "PERTURB_ON_HIGH_RES is not supported": mo.PERTURB_ON_HIGH_RES,
        "NON_CUBIC_FACTOR != 1 is not supported": so.NON_CUBIC_FACTOR != 1,
        "HALO_SCALING_RELATIONS_MEDIAN is not supported": (
            ao.HALO_SCALING_RELATIONS_MEDIAN
        ),
    }
    problems = [message for message, bad in unsupported.items() if bad]
    if problems:
        raise NotImplementedError(
            "The [CII] of halos below the sampler mass is only implemented for "
            "discrete-halo runs without minihalos or reionization feedback: "
            f"{'; '.join(problems)}."
        )


def _validate_models(inputs: p21c.InputParameters, models: Sequence[str]) -> None:
    """Check the [CII] model names, and that the run models halo metallicities."""
    ao = inputs.astro_options
    for model in models:
        validate_cii_model(model)
        if model in METALLICITY_MODELS and not (ao.USE_TS_FLUCT and ao.USE_METALLICITY):
            raise ValueError(
                f"The [CII] model '{model}' needs halo metallicities, which 21cmFAST "
                "only models when USE_TS_FLUCT=True and USE_METALLICITY=True. Use one "
                "of the SFR-only models: "
                f"{[m for m in CII_MODELS if m not in METALLICITY_MODELS]}."
            )


def unresolved_mass_range(inputs: p21c.InputParameters) -> tuple[float, float]:
    """Mass range of the halos that 21cmFAST integrates instead of sampling.

    The lower limit is ``minimum_source_mass`` of 21cmFAST (``hmf.c``), which is
    not exposed to Python: ``M_TURN_STELLAR_FEEDBACK / 50`` for discrete halos
    without minihalos and with ``M_MIN_in_Mass``. The upper limit is
    ``SAMPLER_MIN_MASS``.

    Parameters
    ----------
    inputs : py21cmfast.InputParameters
        Input parameters of the run.

    Returns
    -------
    m_min, m_max : float
        Mass range, in Msun. ``m_min >= m_max`` if no halos are integrated.

    Raises
    ------
    NotImplementedError
        If the run uses options that are not supported (see the module notes).
    """
    _check_supported(inputs)
    m_min = inputs.astro_params.cdict["M_TURN_STELLAR_FEEDBACK"] / 50.0
    return m_min, inputs.simulation_options.SAMPLER_MIN_MASS


def integral_stellar_ratio(
    inputs: p21c.InputParameters, masses: np.ndarray
) -> np.ndarray:
    """Ratio of the stellar fraction of the 21cmFAST integrals to that of halos.

    With ``USE_UPPER_STELLAR_TURNOVER``, the stellar fraction of sampled halos
    (``get_halo_stellar_mass``) is the double power law
    ``F_STAR10_ACG * r / ((M / Mp)^-ALPHA_STAR_ACG + (M / Mp)^-ALPHA_UPPER)``, with
    ``r = (Mp / 1e10)^ALPHA_STAR_ACG + (Mp / 1e10)^ALPHA_UPPER``, but the CHMF
    integrals of the ``EmissivityFields`` (``hmf.c``) use the single power law
    ``F_STAR10_ACG * (M / 1e10)^ALPHA_STAR_ACG``. Below the pivot ``Mp`` the double
    power law is higher by ``1 + (Mp / 1e10)^(ALPHA_UPPER - ALPHA_STAR_ACG)``, about
    2.5% for the default parameters. Multiplying the stellar mass (and hence the SFR)
    of a halo by this ratio reproduces the integrals.

    Parameters
    ----------
    inputs : py21cmfast.InputParameters
        Input parameters of the run.
    masses : np.ndarray
        Halo masses, in Msun.

    Returns
    -------
    np.ndarray
        The ratio at each mass; one if the single power law is used for halos too.
    """
    ap = inputs.astro_params.cdict
    alpha, alpha_upper = ap["ALPHA_STAR_ACG"], ap["UPPER_STELLAR_TURNOVER_INDEX"]
    masses = np.asarray(masses, dtype=float)
    if not inputs.astro_options.USE_UPPER_STELLAR_TURNOVER or alpha <= alpha_upper:
        return np.ones_like(masses)
    pivot = ap["UPPER_STELLAR_TURNOVER_MASS"]
    norm_ratio = (pivot / 1e10) ** alpha + (pivot / 1e10) ** alpha_upper
    double = norm_ratio / (
        (masses / pivot) ** -alpha + (masses / pivot) ** -alpha_upper
    )
    return (masses / 1e10) ** alpha / double


class _QuadratureNormals(np.random.Generator):
    """A generator whose ``standard_normal`` returns given quadrature nodes.

    Every [CII] model draws its scatter with a single call of
    ``rng.standard_normal(n)``, one deviate per halo. Passing this generator as the
    seed of :func:`~tuesday.core.postprocessing.line_intensities.cii_luminosity`
    evaluates the model at the nodes of a Gauss-Hermite rule instead.
    """

    def __init__(self, nodes: np.ndarray):
        super().__init__(np.random.PCG64(0))
        self._nodes = nodes

    def standard_normal(self, size=None, dtype=np.float64, out=None) -> np.ndarray:
        """Return the nodes; ``size`` must be their number."""
        if size != len(self._nodes) or out is not None:
            raise RuntimeError(
                "A [CII] model drew its scatter other than with one "
                "standard_normal(n_halos) call, which the quadrature assumes."
            )
        return self._nodes.astype(dtype)


def _gauss_hermite(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalised weights of the Gauss-Hermite rule for a unit normal."""
    nodes, weights = hermegauss(n_nodes)
    return nodes, weights / weights.sum()


def mean_halo_properties(
    inputs: p21c.InputParameters,
    redshift: float,
    masses: np.ndarray,
    models: Sequence[str],
    n_nodes: int = 20,
) -> dict[str, np.ndarray]:
    """Mean SFR and [CII] luminosity of halos of given masses.

    The 21cmFAST scatter of the stellar mass and of the SFR of a halo, and the
    scatter of the [CII] model, are each driven by a unit normal deviate (the
    ``star_rng`` and ``sfr_rng`` of a halo catalogue, and the deviate drawn inside
    the model). The mean over the three is computed with a Gauss-Hermite rule of
    ``n_nodes`` nodes in each. At each node, the stellar mass and SFR are those of
    :func:`py21cmfast.wrapper.cfuncs.convert_halo_properties`, as for the sampled
    halos, rescaled to the relation of the 21cmFAST integrals
    (:func:`integral_stellar_ratio`); the metallicity follows from them with
    :func:`~tuesday.core.postprocessing.line_intensities.halo_metallicity`, the
    relation of 21cmFAST, and the [CII] luminosity from
    :func:`~tuesday.core.postprocessing.line_intensities.cii_luminosity`. The rule
    is exact for the mean SFR, whose scatter is log-normal, and accurate to well
    below a percent for log-normal scatters of up to ~1 dex.

    Parameters
    ----------
    inputs : py21cmfast.InputParameters
        Input parameters of the run.
    redshift : float
        Redshift of the halos.
    masses : np.ndarray
        Halo masses, in Msun.
    models : sequence of str
        [CII] luminosity models; see
        :data:`~tuesday.core.postprocessing.line_intensities.CII_MODELS`.
    n_nodes : int, optional
        Number of Gauss-Hermite nodes for each of the three deviates.

    Returns
    -------
    dict of str to np.ndarray
        ``"sfr"``: mean SFR in Msun/s; one entry per model: mean [CII] luminosity
        in Lsun.

    Raises
    ------
    ValueError
        If a model is unknown, or needs metallicities and the run was done without
        ``USE_TS_FLUCT`` and ``USE_METALLICITY``.
    """
    _validate_models(inputs, models)
    masses = np.asarray(masses, dtype=float)
    nodes, weights = _gauss_hermite(n_nodes)
    star_rng, sfr_rng = (grid.ravel() for grid in np.meshgrid(nodes, nodes))
    halo_weights = np.outer(weights, weights).ravel()
    n_halos = len(star_rng)

    # Without minihalos or reionization feedback, the halo properties do not depend
    # on the position of the halo. The X-ray deviate does not enter the SFR.
    props = convert_halo_properties(
        redshift=redshift,
        inputs=inputs,
        halo_masses=np.repeat(masses, n_halos),
        star_rng=np.tile(star_rng, len(masses)),
        sfr_rng=np.tile(sfr_rng, len(masses)),
        xray_rng=np.zeros(len(masses) * n_halos),
    )
    star_ratio = np.repeat(integral_stellar_ratio(inputs, masses), n_halos)
    # 21cmFAST gives the SFR in Msun/s.
    sfr = props["sfr_acg"].astype(np.float64) * star_ratio
    stellar_mass = props["stellar_mass_acg"].astype(np.float64) * star_ratio

    out = {"sfr": sfr.reshape(len(masses), n_halos) @ halo_weights}
    # Far below the turnover mass, the float32 SFR of 21cmFAST underflows to zero:
    # such halos do not emit.
    forming = np.flatnonzero(sfr > 0)
    mass_index = np.repeat(np.arange(len(masses)), n_halos)[forming]
    # Each forming halo, at each node of the model scatter.
    expand = np.repeat(forming, n_nodes)
    point_weights = np.repeat(np.tile(halo_weights, len(masses))[forming], n_nodes)
    point_weights *= np.tile(weights, len(forming))
    metallicity = halo_metallicity(
        sfr[expand] * un.Msun / un.s, stellar_mass[expand] * un.Msun, redshift
    )
    for model in models:
        luminosity = cii_luminosity(
            sfr=sfr[expand] * un.Msun / un.s,
            metallicity=metallicity,
            halo_mass=props["halo_mass"].astype(np.float64)[expand] * un.Msun,
            redshift=redshift,
            model=model,
            seed=_QuadratureNormals(np.tile(nodes, len(forming))),
        )
        out[model] = np.bincount(
            np.repeat(mass_index, n_nodes),
            weights=luminosity.to_value(un.Lsun) * point_weights,
            minlength=len(masses),
        )
    return out


def unresolved_cell_totals(
    initial_conditions: p21c.InitialConditions,
    redshift: float,
    models: Sequence[str],
    inputs: p21c.InputParameters | None = None,
    n_bins: int = 80,
    n_delta: int = 400,
    n_nodes: int = 20,
) -> dict[str, np.ndarray]:
    """Totals of the halos below the sampler mass in each Lagrangian cell.

    The expected number of halos of each of ``n_bins`` logarithmic mass bins in a
    low-resolution cell of linear density ``delta`` is given by the sampler's CHMF
    (:func:`py21cmfast.wrapper.cfuncs.integrate_chmf_interval`), and multiplied by
    the mean properties of :func:`mean_halo_properties` at the bin centre. The sums
    are tabulated on ``n_delta`` densities and interpolated to each cell.

    Parameters
    ----------
    initial_conditions : py21cmfast.InitialConditions
        Initial conditions of the run.
    redshift : float
        Redshift of the halos.
    models : sequence of str
        [CII] luminosity models.
    inputs : py21cmfast.InputParameters, optional
        Input parameters of the run. If None, those of ``initial_conditions`` are
        used; pass the run's own when the initial conditions may have been made for
        a run with other astrophysical parameters.
    n_bins : int, optional
        Number of logarithmic halo-mass bins.
    n_delta : int, optional
        Number of densities in the interpolation table.
    n_nodes : int, optional
        Number of quadrature nodes per deviate; see :func:`mean_halo_properties`.

    Returns
    -------
    dict of str to np.ndarray
        Arrays of the low-resolution shape, on the Lagrangian grid: ``"halo_mass"``
        (Msun), ``"sfr"`` (Msun/s), and one [CII] luminosity (Lsun) per model.

    Raises
    ------
    ValueError
        If a model is unknown, or needs metallicities and the run was done without
        ``USE_TS_FLUCT`` and ``USE_METALLICITY``.
    NotImplementedError
        If the run uses options that are not supported (see the module notes).
    """
    inputs = initial_conditions.inputs if inputs is None else inputs
    _validate_models(inputs, models)
    m_min, m_max = unresolved_mass_range(inputs)
    # One backend scope for all the calls, so that no other thread sets up the C
    # globals for other inputs in between.
    with c_state(inputs, broadcast_inputs=True, sigma=True):
        delta = initial_conditions.get("lowres_density").astype(float)
        delta *= lib.dicke(redshift)
        if m_min >= m_max:
            return {key: np.zeros_like(delta) for key in ("halo_mass", "sfr", *models)}

        edges = np.linspace(np.log(m_min), np.log(m_max), n_bins + 1)
        centres = np.exp(0.5 * (edges[1:] + edges[:-1]))
        # The table range is widened slightly, as in add_integral_contribution.
        deltas = np.linspace(delta.min() * 1.001, delta.max() * 1.001, n_delta)
        counts = integrate_chmf_interval(
            inputs=inputs,
            redshift=redshift,
            lnm_lower=edges[:-1],
            lnm_upper=edges[1:],
            cond_values=deltas,
        )
        means = mean_halo_properties(inputs, redshift, centres, models, n_nodes=n_nodes)
    tables = {"halo_mass": counts @ centres}
    tables.update({key: counts @ values for key, values in means.items()})
    return {
        key: np.interp(delta, deltas, table).reshape(delta.shape)
        for key, table in tables.items()
    }


def lagrangian_to_eulerian(
    initial_conditions: p21c.InitialConditions,
    redshift: float,
    inputs: p21c.InputParameters | None = None,
) -> np.ndarray:
    """Eulerian positions of the centres of the Lagrangian low-resolution cells.

    Follows ``move_integral_emissivities`` in ``map_mass.c``: the low-resolution
    velocities of the initial conditions are scaled by the growth since the initial
    redshift, with the 2LPT correction if ``PERTURB_ALGORITHM='2LPT'``.

    Parameters
    ----------
    initial_conditions : py21cmfast.InitialConditions
        Initial conditions of the run.
    redshift : float
        Redshift of the positions.
    inputs : py21cmfast.InputParameters, optional
        Input parameters of the run; if None, those of ``initial_conditions``.

    Returns
    -------
    np.ndarray
        Positions of shape ``(HII_DIM**3, 3)``, in units of the low-resolution cell
        size, in the C order of the low-resolution grid.
    """
    inputs = initial_conditions.inputs if inputs is None else inputs
    so = inputs.simulation_options
    with c_state(inputs, broadcast_inputs=True):
        growth = lib.dicke(redshift)
        growth_init = lib.dicke(so.INITIAL_REDSHIFT)
    n = so.HII_DIM
    factor = (growth - growth_init) / so.BOX_LEN * n
    factor_2lpt = -(3.0 / 7.0) * (growth**2 - growth_init**2) / so.BOX_LEN * n

    cells = np.indices((n, n, n), dtype=float)
    positions = np.empty((n**3, 3))
    for axis, name in enumerate("xyz"):
        pos = cells[axis] + initial_conditions.get(f"lowres_v{name}") * factor
        if inputs.matter_options.PERTURB_ALGORITHM == "2LPT":
            pos -= initial_conditions.get(f"lowres_v{name}_2LPT") * factor_2lpt
        positions[:, axis] = pos.ravel()
    return positions


def grid_unresolved_halos(
    initial_conditions: p21c.InitialConditions,
    redshift: float,
    models: Sequence[str],
    inputs: p21c.InputParameters | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Grid the halos below the sampler mass as 21cmFAST does for EmissivityFields.

    The totals of each Lagrangian cell (:func:`unresolved_cell_totals`) are moved
    to the Eulerian positions of :func:`lagrangian_to_eulerian` and assigned to the
    low-resolution grid with cloud-in-cell interpolation, so that the ``"sfr"`` grid
    equals what 21cmFAST adds to ``EmissivityFields.sfrd_acg`` (times the cell
    volume).

    Parameters
    ----------
    initial_conditions : py21cmfast.InitialConditions
        Initial conditions of the run.
    redshift : float
        Redshift of the halos.
    models : sequence of str
        [CII] luminosity models.
    inputs : py21cmfast.InputParameters, optional
        Input parameters of the run; if None, those of ``initial_conditions``.
    **kwargs
        Passed to :func:`unresolved_cell_totals`.

    Returns
    -------
    dict of str to np.ndarray
        Eulerian grids of the low-resolution shape: ``"halo_mass"`` (Msun),
        ``"sfr"`` (Msun/s), and, for each model, the velocity-integrated [CII]
        flux density in Jy km/s.

    Raises
    ------
    ValueError
        If a model is unknown, or needs metallicities and the run was done without
        ``USE_TS_FLUCT`` and ``USE_METALLICITY``.
    NotImplementedError
        If the run uses options that are not supported (see the module notes).
    """
    # Imported here: cii_lightcone imports this module.
    from .cii_lightcone import _cic_grid

    inputs = initial_conditions.inputs if inputs is None else inputs
    totals = unresolved_cell_totals(
        initial_conditions, redshift, models, inputs=inputs, **kwargs
    )
    n = inputs.simulation_options.HII_DIM
    positions = lagrangian_to_eulerian(initial_conditions, redshift, inputs=inputs)
    cosmo = inputs.cosmo_params.cosmo
    grids = {}
    for key, per_cell in totals.items():
        weights = per_cell.ravel()
        if key in models:
            weights = cii_flux(weights * un.Lsun, redshift, cosmo).value
        grids[key] = _cic_grid(positions, weights, n)
    return grids
