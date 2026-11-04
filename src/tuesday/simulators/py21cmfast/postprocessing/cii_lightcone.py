"""Build a [CII] line-intensity lightcone from cached 21cmFAST halo catalogues.

For each node redshift of a 21cmFAST lightcone, the halo catalogue is read from the
output cache, the [CII] emission of each halo is computed with
:func:`~.cii_calculation.calculate_cii` and gridded into a coeval box at the Eulerian
halo positions, in the same way as 21cmFAST grids the halo properties of its
``HaloBox``. The boxes are then interpolated onto the lightcone with the same
lightconer as the 21cmFAST run, and converted to a surface brightness with
:func:`~.cii_calculation.calculate_bvoxel`.

Only the ``HaloCatalog`` and ``InitialConditions`` of each node redshift need to be in
the cache; the 21-cm outputs of the run (``PerturbedField``, ``IonizedBox``,
``BrightnessTemp``) are not used.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import attrs
import numpy as np
import py21cmfast as p21c
from attrs import define, field
from py21cmfast.io.caching import OutputCache
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
    CII_IntensityBox : np.ndarray
        Velocity-integrated [CII] flux density in each cell, in Jy km/s.
    initial_conditions : py21cmfast.InitialConditions, optional
        Initial conditions of the run.
    halobox : py21cmfast.HaloBox, optional
        The 21cmFAST gridded halo properties at this redshift, if cached.
    """

    redshift: float = field(converter=float)
    inputs: p21c.InputParameters
    CII_IntensityBox: np.ndarray
    initial_conditions: p21c.InitialConditions | None = None
    halobox: p21c.HaloBox | None = None

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


def gen_coeval(
    z: float,
    model: str,
    directory_path: str | Path,
    seed: int | np.random.Generator | None = None,
) -> CIICoeval:
    """Build a coeval box with the gridded [CII] emission at one redshift.

    Parameters
    ----------
    z : float
        Redshift of the cached outputs to read.
    model : str
        The [CII] luminosity model; see :func:`~.cii_calculation.calculate_cii`.
    directory_path : str or Path
        Directory of the 21cmFAST output cache.
    seed : int or np.random.Generator, optional
        Seed or generator for the scatter of the [CII] model. If None, a random
        seed is used.

    Returns
    -------
    CIICoeval
        The coeval box, with ``CII_IntensityBox`` in Jy km/s per cell.

    Raises
    ------
    ValueError
        If the cache has no ``HaloCatalog`` or ``InitialConditions`` at ``z``.

    Notes
    -----
    The halos are moved from the Lagrangian positions of the ``HaloCatalog`` to
    their Eulerian positions with :func:`py21cmfast.perturb_halo_catalog`, and their
    flux is assigned to the grid with cloud-in-cell interpolation. This is how
    21cmFAST grids the halo properties of its ``HaloBox`` (``map_mass.c``), so the
    [CII] emission lies on the same cells as the other 21cmFAST fields.
    """
    output_cache = OutputCache(direc=directory_path)
    cache_obj = {
        cache_file.stem: read_output_struct(cache_file)
        for cache_file in output_cache.list_datasets(redshift=z)
    }

    missing = [
        name for name in ("HaloCatalog", "InitialConditions") if name not in cache_obj
    ]
    if missing:
        raise ValueError(
            f"The [CII] box at z={z} needs the {' and '.join(missing)} of the run, "
            f"which are not in the cache at {directory_path} (found "
            f"{sorted(cache_obj) or 'nothing'})."
        )
    halo_catalog = cache_obj["HaloCatalog"]
    initial_conditions = cache_obj["InitialConditions"]
    _, intensity = calculate_cii(halo_catalog, model, seed=seed)

    # The HaloCatalog holds the Lagrangian halo positions. Move the halos to their
    # Eulerian positions, as 21cmFAST does for the HaloBox. perturb_halo_catalog
    # reads every halo array but only loads the coordinates and masses from disk
    # itself, and crashes if the others are not in memory, so load them all.
    halo_catalog.load_all()
    perturbed = p21c.perturb_halo_catalog(
        initial_conditions=initial_conditions,
        inputs=halo_catalog.inputs,
        halo_catalog=halo_catalog,
    )
    if perturbed.n_halos != halo_catalog.n_halos:
        raise ValueError(
            f"Perturbing the {halo_catalog.n_halos} halos at z={z} returned "
            f"{perturbed.n_halos} halos."
        )
    resolution = halo_catalog.simulation_options.HII_DIM
    cell_size = halo_catalog.simulation_options.BOX_LEN / resolution
    # Halo coordinates are in Mpc. Only the first n_halos entries are real halos,
    # matching the output of calculate_cii.
    positions = perturbed.get("halo_coords")[: perturbed.n_halos] / cell_size
    cii_intensity = _cic_grid(positions, intensity, resolution)

    return CIICoeval(
        redshift=z,
        inputs=halo_catalog.inputs,
        CII_IntensityBox=cii_intensity,
        initial_conditions=initial_conditions,
        halobox=cache_obj.get("HaloBox"),
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
        coeval_list.append(gen_coeval(z, model, directory_path, seed=rng))
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
        ``"CII_IntensityBox"`` (Jy km/s per cell) and ``"CII_BrightnessDensity"``
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
            coeval = gen_coeval(z, model, directory_path, seed=rng)
            # Only the gridded [CII] field is interpolated; drop the initial
            # conditions and halo box read from the cache so that they can be freed.
            coeval.initial_conditions = None
            coeval.halobox = None
            yield coeval

    intensity = construct_lightcone_from_coevals(
        cii_coevals(),
        attrs.evolve(lcn, quantities=("CII_IntensityBox",)),
        lc.inputs,
    ).lightcones["CII_IntensityBox"]

    cii_lc = attrs.evolve(
        lc, lightcones={**lc.lightcones, "CII_IntensityBox": intensity}
    )
    logger.info("Calculating the [CII] surface brightness")
    cii_lc.lightcones["CII_BrightnessDensity"] = calculate_bvoxel(cii_lc)

    return cii_lc
