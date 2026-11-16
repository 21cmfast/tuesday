r"""[CII] emission of the halos below the 21cmFAST sampler mass.

With discrete halos (``SOURCE_MODEL='CHMF-SAMPLER'``), 21cmFAST samples halos only
above ``SAMPLER_MIN_MASS``. The halos below it are not in the ``HaloCatalog``, but
21cmFAST still adds their star formation to the ``HaloBox``, and hence to the 21-cm
signal, by integrating the conditional halo mass function (CHMF) of each cell
(``HaloBox.c: set_fixed_grids``, ``map_mass.c: move_grid_galprops``):

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
SFR of the ``HaloBox``; it is averaged halo by halo, inside the integral:

.. math::

    L_{\rm cell} = \int_{M_{\rm min}}^{M_{\rm samp}}
    \langle L_{\rm CII}\rangle(M) \,
    \frac{dN}{dM}(M \mid \delta, M_{\rm cell}) \, dM .

``<L_CII>(M)`` is averaged over the 21cmFAST scatter of the stellar mass and SFR
(the same ``star_rng``, ``sfr_rng`` and ``xray_rng`` deviates as for sampled halos)
and over the scatter of the [CII] model, with the same 21cmFAST relations as for the
sampled halos (:func:`py21cmfast.wrapper.cfuncs.convert_halo_properties`), except
for the stellar fraction (see :func:`integral_stellar_ratio`).

Only runs without minihalos, with ``M_MIN_in_Mass``, without
``PERTURB_ON_HIGH_RES`` and with mean (not median) scaling relations are supported.
"""

from collections.abc import Sequence

import attrs
import numpy as np
import py21cmfast as p21c
from astropy import units as un
from py21cmfast.c_21cmfast import lib
from py21cmfast.wrapper.cfuncs import (
    broadcast_input_struct,
    convert_halo_properties,
    integrate_chmf_interval,
)
from scipy.stats import norm

from tuesday.core.postprocessing.line_intensities import (
    cii_flux,
    cii_luminosity,
    validate_cii_model,
)


def fix_cached_inputs(inputs: p21c.InputParameters) -> p21c.InputParameters:
    """Work around a 21cmFAST bug in input parameters read from an HDF5 cache.

    In 21cmFAST 4.1, ``cosmo_tables.USE_SIGMA_8`` of inputs read back from a cache
    file is a ``numpy.bool_``. ``CosmoTables.cstruct`` only copies ``float`` and
    ``bool`` values to the C struct, so the flag is left False and the power
    spectrum is normalised with ``ps_norm`` (sigma_8) as if it were A_s: sigma(M),
    and every CHMF integral, are then nearly zero.

    Parameters
    ----------
    inputs : py21cmfast.InputParameters
        Input parameters, possibly read from a cache file.

    Returns
    -------
    py21cmfast.InputParameters
        The same parameters, with ``cosmo_tables.USE_SIGMA_8`` a Python ``bool``
        and ``ps_norm`` a Python ``float``.
    """
    tables = inputs.cosmo_tables
    if tables.USE_SIGMA_8 is None:
        return inputs
    return attrs.evolve(
        inputs,
        cosmo_tables=attrs.evolve(
            tables,
            ps_norm=float(tables.ps_norm),
            USE_SIGMA_8=bool(tables.USE_SIGMA_8),
        ),
    )


def _check_supported(inputs: p21c.InputParameters) -> None:
    """Raise if the run uses options that are not implemented here."""
    mo, ao = inputs.matter_options, inputs.astro_options
    unsupported = {
        "SOURCE_MODEL must be 'CHMF-SAMPLER'": mo.SOURCE_MODEL != "CHMF-SAMPLER",
        "USE_MINI_HALOS is not supported": ao.USE_MINI_HALOS,
        "M_MIN_in_Mass=False is not supported": not ao.M_MIN_in_Mass,
        "PERTURB_ON_HIGH_RES is not supported": mo.PERTURB_ON_HIGH_RES,
        "HALO_SCALING_RELATIONS_MEDIAN is not supported": (
            ao.HALO_SCALING_RELATIONS_MEDIAN
        ),
        # 21cmFAST rescales the integrated grids to the global mean for these.
        "HMF 'WATSON' and 'WATSON-Z' are not supported": mo.HMF
        in ("WATSON", "WATSON-Z"),
    }
    problems = [message for message, bad in unsupported.items() if bad]
    if problems:
        raise NotImplementedError(
            "The [CII] of halos below the sampler mass is only implemented for "
            f"discrete-halo runs without minihalos: {'; '.join(problems)}."
        )


def unresolved_mass_range(inputs: p21c.InputParameters) -> tuple[float, float]:
    """Mass range of the halos that 21cmFAST integrates instead of sampling.

    The lower limit is ``minimum_source_mass`` of 21cmFAST (``hmf.c``), which is
    not exposed to Python: ``M_TURN / 50`` for discrete halos without minihalos and
    with ``M_MIN_in_Mass``. The upper limit is ``SAMPLER_MIN_MASS``.

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
    m_min = 10**inputs.astro_params.M_TURN / 50.0
    return m_min, inputs.simulation_options.SAMPLER_MIN_MASS


def integral_stellar_ratio(
    inputs: p21c.InputParameters, masses: np.ndarray
) -> np.ndarray:
    """Ratio of the stellar fraction of the 21cmFAST integrals to that of halos.

    With ``USE_UPPER_STELLAR_TURNOVER``, the stellar fraction of sampled halos
    (``get_halo_stellarmass``) is the double power law
    ``F_STAR10 * r / ((M / Mp)^-ALPHA_STAR + (M / Mp)^-ALPHA_UPPER)``, with
    ``r = (Mp / 1e10)^ALPHA_STAR + (Mp / 1e10)^ALPHA_UPPER``, but the CHMF integrals
    of the ``HaloBox`` (``hmf.c``) use the single power law
    ``F_STAR10 * (M / 1e10)^ALPHA_STAR``. Below the pivot ``Mp`` the double power
    law is higher by ``1 + (Mp / 1e10)^(ALPHA_UPPER - ALPHA_STAR)``, about 2.5% for
    the default parameters. Multiplying the stellar mass (and hence the SFR) of a
    halo by this ratio reproduces the integrals.

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
    ap = inputs.astro_params
    alpha, alpha_upper = ap.ALPHA_STAR, ap.UPPER_STELLAR_TURNOVER_INDEX
    masses = np.asarray(masses, dtype=float)
    if not inputs.astro_options.USE_UPPER_STELLAR_TURNOVER or alpha <= alpha_upper:
        return np.ones_like(masses)
    pivot = 10**ap.UPPER_STELLAR_TURNOVER_MASS
    norm_ratio = (pivot / 1e10) ** alpha + (pivot / 1e10) ** alpha_upper
    double = norm_ratio / (
        (masses / pivot) ** -alpha + (masses / pivot) ** -alpha_upper
    )
    return (masses / 1e10) ** alpha / double


def mean_halo_properties(
    inputs: p21c.InputParameters,
    redshift: float,
    masses: np.ndarray,
    models: Sequence[str],
    n_draws: int = 100_000,
    seed: int | np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Mean SFR and [CII] luminosity of halos of given masses.

    For each mass, ``n_draws`` halos are given the 21cmFAST scatter of the stellar
    mass, SFR and X-ray luminosity. Their properties are computed with
    :func:`py21cmfast.wrapper.cfuncs.convert_halo_properties`, as for the sampled
    halos, with the stellar mass and SFR rescaled to the relation of the 21cmFAST
    integrals (:func:`integral_stellar_ratio`), and their [CII] luminosity with
    :func:`~tuesday.core.postprocessing.line_intensities.cii_luminosity`. The
    deviates are stratified (every quantile ``(i + 0.5) / n_draws`` is used once by
    each of the three), and the same deviates are used at every mass.

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
    n_draws : int, optional
        Number of halos averaged at each mass.
    seed : int or np.random.Generator, optional
        Seed or generator for the deviates and the [CII] scatter.

    Returns
    -------
    dict of str to np.ndarray
        ``"sfr"``: mean SFR in Msun/s; one entry per model: mean [CII] luminosity
        in Lsun.

    Notes
    -----
    The metallicity, used by the metallicity-dependent models, is that of the
    sampled-halo relation and is not recomputed for the rescaled SFR; the change is
    below a percent.
    """
    for model in models:
        validate_cii_model(model)
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(n_draws) + 0.5) / n_draws
    star_rng, sfr_rng, xray_rng = (
        norm.ppf(rng.permutation(quantiles)) for _ in range(3)
    )
    cii_seed = rng.integers(2**32)
    masses = np.asarray(masses, dtype=float)
    star_ratio = integral_stellar_ratio(inputs, masses)

    out = {key: np.empty(len(masses)) for key in ("sfr", *models)}
    for i, mass in enumerate(masses):
        props = convert_halo_properties(
            redshift=redshift,
            inputs=inputs,
            halo_masses=np.full(n_draws, mass),
            # Positions only matter for the turnover masses of minihalos.
            halo_coords=np.zeros((n_draws, 3)),
            star_rng=star_rng,
            sfr_rng=sfr_rng,
            xray_rng=xray_rng,
        )
        sfr = props["halo_sfr"] * star_ratio[i]
        out["sfr"][i] = sfr.mean()
        for model in models:
            luminosity = cii_luminosity(
                sfr=sfr * un.Msun / un.s,
                metallicity=props["metallicity"],
                halo_mass=props["halo_mass"] * un.Msun,
                redshift=redshift,
                model=model,
                seed=cii_seed,
            )
            out[model][i] = luminosity.to_value(un.Lsun).mean()
    return out


def _linear_density(
    initial_conditions: p21c.InitialConditions, redshift: float
) -> np.ndarray:
    """Linear density of the Lagrangian low-resolution cells at ``redshift``."""
    broadcast_input_struct(fix_cached_inputs(initial_conditions.inputs))
    density = initial_conditions.get("lowres_density").astype(float)
    return density * lib.dicke(redshift)


def unresolved_cell_totals(
    initial_conditions: p21c.InitialConditions,
    redshift: float,
    models: Sequence[str],
    n_bins: int = 80,
    n_delta: int = 400,
    n_draws: int = 100_000,
    seed: int | np.random.Generator | None = None,
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
    n_bins : int, optional
        Number of logarithmic halo-mass bins.
    n_delta : int, optional
        Number of densities in the interpolation table.
    n_draws : int, optional
        Number of halos averaged at each mass; see :func:`mean_halo_properties`.
    seed : int or np.random.Generator, optional
        Seed or generator for :func:`mean_halo_properties`.

    Returns
    -------
    dict of str to np.ndarray
        Arrays of the low-resolution shape, on the Lagrangian grid: ``"halo_mass"``
        (Msun), ``"sfr"`` (Msun/s), and one [CII] luminosity (Lsun) per model.

    Raises
    ------
    NotImplementedError
        If the run uses options that are not supported (see the module notes).
    """
    inputs = fix_cached_inputs(initial_conditions.inputs)
    m_min, m_max = unresolved_mass_range(inputs)
    delta = _linear_density(initial_conditions, redshift)
    if m_min >= m_max:
        return {key: np.zeros_like(delta) for key in ("halo_mass", "sfr", *models)}

    edges = np.linspace(np.log(m_min), np.log(m_max), n_bins + 1)
    centres = np.exp(0.5 * (edges[1:] + edges[:-1]))
    # The table range is widened slightly, as in set_fixed_grids.
    deltas = np.linspace(delta.min() * 1.001, delta.max() * 1.001, n_delta)
    counts = integrate_chmf_interval(
        inputs=inputs,
        redshift=redshift,
        lnm_lower=edges[:-1],
        lnm_upper=edges[1:],
        cond_values=deltas,
    )
    means = mean_halo_properties(
        inputs, redshift, centres, models, n_draws=n_draws, seed=seed
    )
    tables = {"halo_mass": counts @ centres}
    tables.update({key: counts @ values for key, values in means.items()})
    return {
        key: np.interp(delta, deltas, table).reshape(delta.shape)
        for key, table in tables.items()
    }


def lagrangian_to_eulerian(
    initial_conditions: p21c.InitialConditions, redshift: float
) -> np.ndarray:
    """Eulerian positions of the centres of the Lagrangian low-resolution cells.

    Follows ``move_grid_galprops`` in ``map_mass.c``: the low-resolution velocities
    of the initial conditions are scaled by the growth since the initial redshift,
    with the 2LPT correction if ``PERTURB_ALGORITHM='2LPT'``.

    Parameters
    ----------
    initial_conditions : py21cmfast.InitialConditions
        Initial conditions of the run.
    redshift : float
        Redshift of the positions.

    Returns
    -------
    np.ndarray
        Positions of shape ``(HII_DIM**3, 3)``, in units of the low-resolution cell
        size, in the C order of the low-resolution grid.
    """
    inputs = fix_cached_inputs(initial_conditions.inputs)
    so = inputs.simulation_options
    broadcast_input_struct(inputs)
    n = so.HII_DIM
    growth = lib.dicke(redshift)
    growth_init = lib.dicke(so.INITIAL_REDSHIFT)
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
    seed: int | np.random.Generator | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Grid the halos below the sampler mass as 21cmFAST does for its HaloBox.

    The totals of each Lagrangian cell (:func:`unresolved_cell_totals`) are moved
    to the Eulerian positions of :func:`lagrangian_to_eulerian` and assigned to the
    low-resolution grid with cloud-in-cell interpolation, so that the ``"sfr"`` grid
    equals what 21cmFAST adds to ``HaloBox.halo_sfr`` (times the cell volume).

    Parameters
    ----------
    initial_conditions : py21cmfast.InitialConditions
        Initial conditions of the run.
    redshift : float
        Redshift of the halos.
    models : sequence of str
        [CII] luminosity models.
    seed : int or np.random.Generator, optional
        Seed or generator for :func:`mean_halo_properties`.
    **kwargs
        Passed to :func:`unresolved_cell_totals`.

    Returns
    -------
    dict of str to np.ndarray
        Eulerian grids of the low-resolution shape: ``"halo_mass"`` (Msun),
        ``"sfr"`` (Msun/s), and, for each model, the velocity-integrated [CII]
        flux density in Jy km/s.
    """
    # Imported here: cii_lightcone imports this module.
    from .cii_lightcone import _cic_grid

    totals = unresolved_cell_totals(
        initial_conditions, redshift, models, seed=seed, **kwargs
    )
    inputs = fix_cached_inputs(initial_conditions.inputs)
    n = inputs.simulation_options.HII_DIM
    positions = lagrangian_to_eulerian(initial_conditions, redshift)
    cosmo = inputs.cosmo_params.cosmo
    grids = {}
    for key, per_cell in totals.items():
        weights = per_cell.ravel()
        if key in models:
            weights = cii_flux(weights * un.Lsun, redshift, cosmo).value
        grids[key] = _cic_grid(positions, weights, n)
    return grids
