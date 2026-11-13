"""Build a [CII] line-intensity lightcone from cached 21cmFAST halo catalogues.

For each node redshift of a 21cmFAST lightcone, the halo catalogue is read from the
output cache and moved to the Eulerian halo positions with
:func:`py21cmfast.perturb_halo_catalog`, which also computes the halo properties of
the run. The [CII] emission of each halo is computed from them with
:func:`~.cii_calculation.calculate_cii` and gridded into a coeval box, in the same way
as 21cmFAST grids the halo properties of its ``EmissivityFields``. The boxes are then
interpolated onto the lightcone with the same lightconer as the 21cmFAST run, and
converted to a surface brightness with :func:`~.cii_calculation.calculate_bvoxel`.

The ``HaloCatalog`` and ``InitialConditions`` of each node redshift must be in the
cache. Below ``Z_HEAT_MAX``, except at the first node redshift, the halo properties
also depend on boxes of the previous node redshift, from which 21cmFAST takes the
feedback: runs with minihalos (``USE_MCGS``) need its ``TsBox`` and ``IonizedBox``,
and runs with reionization feedback (``USE_REIONIZATION_PHOTOHEATING_FEEDBACK``) but
no minihalos need its ``IonizedBox``. The other 21-cm outputs of the run
(``PerturbedField``, ``BrightnessTemp``) are not used.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import attrs
import numpy as np
import py21cmfast as p21c
from attrs import define, field
from py21cmfast.io.caching import OutputCache, RunCache
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.lightconers import Lightconer

from ..lightcones import construct_lightcone_from_coevals
from .cii_calculation import calculate_bvoxel, calculate_cii

logger = logging.getLogger(__name__)


@define
class CIICoeval:
    """A coeval box of the gridded [CII] flux density at one redshift.

    It has the attributes that
    :meth:`~py21cmfast.lightconers.Lightconer.make_lightcone_slices` needs
    (``redshift``, ``simulation_options``, ``cosmo_params`` and the field), so it can
    be interpolated onto a lightcone like a :class:`py21cmfast.Coeval`, but it does
    not need the 21-cm outputs of the run.

    Parameters
    ----------
    redshift : float
        Redshift of the box.
    inputs : py21cmfast.InputParameters
        Input parameters of the 21cmFAST run.
    cii_intensity : np.ndarray
        Velocity-integrated [CII] flux density in each cell, in Jy km/s.
    initial_conditions : py21cmfast.InitialConditions, optional
        Initial conditions of the run.
    halobox : py21cmfast.EmissivityFields, optional
        The 21cmFAST gridded halo properties at this redshift, if cached (the
        ``EmissivityFields``, called ``HaloBox`` before 21cmFAST 4.3).
    """

    redshift: float = field(converter=float)
    inputs: p21c.InputParameters
    cii_intensity: np.ndarray
    initial_conditions: p21c.InitialConditions | None = None
    halobox: p21c.EmissivityFields | None = None

    @property
    def simulation_options(self) -> p21c.SimulationOptions:
        """Simulation options of the run."""
        return self.inputs.simulation_options

    @property
    def cosmo_params(self) -> p21c.CosmoParams:
        """Cosmological parameters of the run."""
        return self.inputs.cosmo_params


def _cic_grid(positions: np.ndarray, weights: np.ndarray, n_cells: int) -> np.ndarray:
    """Assign weights to a periodic cubic grid with cloud-in-cell interpolation.

    Follows the convention of 21cmFAST (``map_mass.c``), in which cell ``i`` is
    centred at position ``i``.

    Parameters
    ----------
    positions : np.ndarray
        Positions of shape ``(n, 3)``, in units of the cell size.
    weights : np.ndarray
        Weight of each position.
    n_cells : int
        Number of cells along each side of the grid.

    Returns
    -------
    np.ndarray
        The grid of shape ``(n_cells, n_cells, n_cells)``, holding the sum of the
        weights assigned to each cell.
    """
    lower = np.floor(positions).astype(int)
    frac = positions - lower
    grid = np.zeros(n_cells**3)
    for corner in np.ndindex(2, 2, 2):
        offset = np.array(corner)
        index = (lower + offset) % n_cells
        flat = (index[:, 0] * n_cells + index[:, 1]) * n_cells + index[:, 2]
        share = np.prod(np.where(offset, frac, 1 - frac), axis=1)
        grid += np.bincount(flat, weights=weights * share, minlength=n_cells**3)
    return grid.reshape((n_cells,) * 3)


def _run_cache(
    z: float, directory_path: str | Path, inputs: p21c.InputParameters | None
) -> RunCache:
    """Return the cached files of the run whose halo catalogue at ``z`` is used.

    Without ``inputs``, the run is the one whose ``HaloCatalog`` at ``z`` is in the
    cache directory; it must be the only one.
    """
    output_cache = OutputCache(direc=directory_path)
    if inputs is None:
        found = output_cache.list_datasets(kind="HaloCatalog", redshift=z)
        if not found:
            raise ValueError(
                f"The [CII] box at z={z} needs the HaloCatalog of the run, which is "
                f"not in the cache at {directory_path}."
            )
        if len(found) > 1:
            raise ValueError(
                f"The cache at {directory_path} holds the halo catalogues of "
                f"{len(found)} runs at z={z}: pass the inputs of the run to use."
            )
        inputs = read_output_struct(found[0]).inputs
    return RunCache.from_inputs(inputs, output_cache)


def _cached_path(run: RunCache, kind: str, z: float) -> Path | None:
    """Return the cached file of one output of the run (None if it is not there)."""
    paths = getattr(run, kind)
    if paths is None:
        # The run does not make this output (e.g. a TsBox without USE_TS_FLUCT).
        return None
    if isinstance(paths, Path):
        path = paths
    else:
        # The node redshift closest to z, as RunCache.get_output_struct_at_z does.
        closest = min(paths, key=lambda node: abs(node - z), default=None)
        path = (
            paths[closest] if closest is not None and abs(closest - z) < 1e-6 else None
        )
    return path if path is not None and path.exists() else None


def _cached_struct(run: RunCache, kind: str, z: float, directory_path: str | Path):
    """Read one cached output of the run, with an error that names what is missing."""
    path = _cached_path(run, kind, z)
    if path is None:
        raise ValueError(
            f"The [CII] box at z={z} needs the {kind} of the run"
            + ("" if kind == "InitialConditions" else f" at z={z}")
            + f", which is not in the cache at {directory_path}."
        )
    return read_output_struct(path)


def gen_coeval(
    z: float,
    model: str,
    directory_path: str | Path,
    seed: int | np.random.Generator | None = None,
    inputs: p21c.InputParameters | None = None,
) -> CIICoeval:
    """Build a coeval box with the gridded [CII] emission at one redshift.

    Parameters
    ----------
    z : float
        Redshift of the cached outputs to read, one of the node redshifts of the run.
    model : str
        The [CII] luminosity model; see :func:`~.cii_calculation.calculate_cii`.
    directory_path : str or Path
        Directory of the 21cmFAST output cache.
    seed : int or np.random.Generator, optional
        Seed or generator for the scatter of the [CII] model. If None, a random
        seed is used.
    inputs : py21cmfast.InputParameters, optional
        Input parameters of the run. Needed only if the cache directory holds more
        than one run; otherwise they are read from the cached halo catalogue.

    Returns
    -------
    CIICoeval
        The coeval box, with ``cii_intensity`` in Jy km/s per cell.

    Raises
    ------
    ValueError
        If the cache misses an output the box needs: the ``HaloCatalog`` at ``z``,
        the ``InitialConditions`` and, below ``Z_HEAT_MAX`` and after the first node
        redshift, the ``TsBox`` and ``IonizedBox`` of the previous node redshift for
        runs with minihalos, or its ``IonizedBox`` for runs with reionization
        feedback and no minihalos.

    Notes
    -----
    The halos are moved from the Lagrangian positions of the ``HaloCatalog`` to
    their Eulerian positions with :func:`py21cmfast.perturb_halo_catalog`, which
    also computes their properties as 21cmFAST does for the run: with the minihalo
    and reionization feedback of the previous node's ``TsBox`` (Lyman-Werner
    background) and ``IonizedBox`` (reionization redshift and photo-ionization
    rate), and the relative velocities of the initial conditions. The [CII] flux of
    each halo is then assigned to the grid with cloud-in-cell interpolation, as
    21cmFAST grids its ``EmissivityFields``, so the [CII] emission lies on the same
    cells as the other 21cmFAST fields.

    Only the halos of the catalogue contribute. 21cmFAST also adds the emission of
    halos below ``SAMPLER_MIN_MASS`` to its ``EmissivityFields`` (from the
    conditional mass function), which has no [CII] counterpart here.
    """
    run = _run_cache(z, directory_path, inputs)
    inputs = run.inputs
    halo_catalog = _cached_struct(run, "HaloCatalog", z, directory_path)
    initial_conditions = _cached_struct(run, "InitialConditions", z, directory_path)

    # The feedback on the halo properties comes from the boxes of the previous node
    # redshift, as in 21cmFAST's redshift loop: the Lyman-Werner background of the
    # TsBox for minihalos, and the reionization redshift and photo-ionization rate of
    # the IonizedBox for reionization feedback. perturb_halo_catalog asks for both
    # boxes in runs with minihalos. Above Z_HEAT_MAX, or at the first node, there is
    # no feedback.
    astro_options = inputs.astro_options
    previous = {}
    node_redshifts = sorted(inputs.node_redshifts, reverse=True)
    i_node = int(np.argmin(np.abs(np.array(node_redshifts) - z)))
    if z < inputs.simulation_options.Z_HEAT_MAX and i_node > 0:
        z_previous = node_redshifts[i_node - 1]
        if astro_options.USE_MCGS:
            previous["previous_spin_temp"] = _cached_struct(
                run, "TsBox", z_previous, directory_path
            )
        if (
            astro_options.USE_MCGS
            or astro_options.USE_REIONIZATION_PHOTOHEATING_FEEDBACK
        ):
            previous["previous_ionize_box"] = _cached_struct(
                run, "IonizedBox", z_previous, directory_path
            )

    # perturb_halo_catalog reads every halo array but only loads the coordinates and
    # masses from disk itself, and crashes if the others are not in memory, so load
    # them all.
    halo_catalog.load_all()
    perturbed = p21c.perturb_halo_catalog(
        initial_conditions=initial_conditions,
        inputs=inputs,
        halo_catalog=halo_catalog,
        **previous,
    )
    if perturbed.n_halos != halo_catalog.n_halos:
        raise ValueError(
            f"Perturbing the {halo_catalog.n_halos} halos at z={z} returned "
            f"{perturbed.n_halos} halos."
        )
    _, intensity = calculate_cii(perturbed, model, seed=seed)

    resolution = inputs.simulation_options.HII_DIM
    cell_size = inputs.simulation_options.BOX_LEN / resolution
    # Halo coordinates are in Mpc. Only the first n_halos entries are real halos,
    # matching the output of calculate_cii.
    positions = perturbed.get("halo_coords")[: perturbed.n_halos] / cell_size
    cii_intensity = _cic_grid(positions, intensity, resolution)

    emissivity_fields = _cached_path(run, "EmissivityFields", z)
    return CIICoeval(
        redshift=z,
        inputs=inputs,
        cii_intensity=cii_intensity,
        initial_conditions=initial_conditions,
        halobox=read_output_struct(emissivity_fields) if emissivity_fields else None,
    )


def gen_coeval_list(
    lc: p21c.LightCone,
    model: str,
    directory_path: str | Path,
    seed: int | np.random.Generator | None = None,
) -> list[CIICoeval]:
    """Build [CII] coeval boxes at all node redshifts of a lightcone.

    Parameters
    ----------
    lc : py21cmfast.LightCone
        The 21cmFAST lightcone whose node redshifts are used.
    model : str
        The [CII] luminosity model; see :func:`~.cii_calculation.calculate_cii`.
    directory_path : str or Path
        Directory of the 21cmFAST output cache.
    seed : int or np.random.Generator, optional
        Seed or generator for the scatter of the [CII] model. A single generator is
        shared by all redshifts, so their scatter is independent. If None, a random
        seed is used.

    Returns
    -------
    list of CIICoeval
        One coeval box per node redshift.
    """
    rng = np.random.default_rng(seed)
    coeval_list = []
    for z in lc.inputs.node_redshifts:
        logger.info("Building the [CII] coeval box at z=%.3f", z)
        coeval_list.append(
            gen_coeval(z, model, directory_path, seed=rng, inputs=lc.inputs)
        )
    return coeval_list


def gen_lc(
    lc: p21c.LightCone,
    lcn: Lightconer,
    model: str,
    directory_path: str | Path,
    seed: int | np.random.Generator | None = None,
) -> p21c.LightCone:
    """Build a [CII] surface-brightness lightcone from a 21cmFAST run.

    Neither ``lc`` nor ``lcn`` is modified: the [CII] fields are added to a new
    lightcone, which shares the other fields with ``lc``.

    The [CII] coeval boxes are built with :func:`gen_coeval` one node redshift at a
    time and interpolated onto the lightcone with
    :func:`~tuesday.simulators.py21cmfast.lightcones.construct_lightcone_from_coevals`
    as soon as they are built, so that only two boxes are in memory at any time.
    The result is the same as interpolating the boxes of :func:`gen_coeval_list`:
    a single generator seeded with ``seed`` is shared by all redshifts.

    Parameters
    ----------
    lc : py21cmfast.LightCone
        The 21cmFAST lightcone of the run.
    lcn : Lightconer
        The lightconer used for the run, which sets the lightcone geometry.
    model : str
        The [CII] luminosity model; see :func:`~.cii_calculation.calculate_cii`.
    directory_path : str or Path
        Directory of the 21cmFAST output cache.
    seed : int or np.random.Generator, optional
        Seed or generator for the scatter of the [CII] model, shared by all
        redshifts. If None, a random seed is used.

    Returns
    -------
    py21cmfast.LightCone
        A copy of ``lc`` with two new fields in ``lightcones``:
        ``"cii_intensity"`` (Jy km/s per cell) and ``"cii_brightness"``
        (Jy/sr).

    Raises
    ------
    ValueError
        If ``lcn`` does not have the same number of slices as ``lc``, or extends
        beyond the node redshifts of the run.
    """
    shape = lcn.get_shape(lc.inputs.simulation_options)
    if shape[-1] != len(lc.lightcone_distances):
        raise ValueError(
            f"The lightconer has {shape[-1]} slices but the lightcone has "
            f"{len(lc.lightcone_distances)}: pass the lightconer used for the run."
        )
    rng = np.random.default_rng(seed)

    def cii_coevals() -> Iterator[CIICoeval]:
        for z in lc.inputs.node_redshifts:
            logger.info("Building the [CII] coeval box at z=%.3f", z)
            coeval = gen_coeval(z, model, directory_path, seed=rng, inputs=lc.inputs)
            # Only the gridded [CII] field is interpolated; drop the initial
            # conditions and halo box read from the cache so that they can be freed.
            coeval.initial_conditions = None
            coeval.halobox = None
            yield coeval

    intensity = construct_lightcone_from_coevals(
        cii_coevals(),
        attrs.evolve(lcn, quantities=("cii_intensity",)),
        lc.inputs,
    ).lightcones["cii_intensity"]

    cii_lc = attrs.evolve(lc, lightcones={**lc.lightcones, "cii_intensity": intensity})
    logger.info("Calculating the [CII] surface brightness")
    cii_lc.lightcones["cii_brightness"] = calculate_bvoxel(cii_lc)

    return cii_lc
