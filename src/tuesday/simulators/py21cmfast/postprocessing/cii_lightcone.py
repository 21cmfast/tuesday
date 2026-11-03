"""Build a [CII] line-intensity lightcone from cached 21cmFAST halo catalogues.

For each node redshift of a 21cmFAST lightcone, the halo catalogue is read from the
output cache, the [CII] emission of each halo is computed with
:func:`~.cii_calculation.calculate_cii` and gridded into a coeval box at the Eulerian
halo positions, in the same way as 21cmFAST grids the halo properties of its
``HaloBox``. The boxes are
then interpolated onto the lightcone with the same lightconer as the 21cmFAST run,
and converted to a surface brightness with :func:`~.cii_calculation.calculate_bvoxel`.
"""

import logging
from itertools import pairwise
from pathlib import Path

import attrs
import numpy as np
import py21cmfast as p21c
from attrs import define, field
from py21cmfast.io.caching import OutputCache
from py21cmfast.io.h5 import read_output_struct
from py21cmfast.lightconers import Lightconer

from .cii_calculation import calculate_bvoxel, calculate_cii

logger = logging.getLogger(__name__)


@define(slots=False)
class CIICoeval(p21c.Coeval):
    """A 21cmFAST coeval box that also holds a gridded [CII] flux density.

    Parameters
    ----------
    CII_IntensityBox : np.ndarray
        Velocity-integrated [CII] flux density in each cell, in Jy km/s.
    **kwargs
        Arguments of :class:`py21cmfast.Coeval`.
    """

    CII_IntensityBox: np.ndarray = field(default=None)


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

    halo_catalog = cache_obj["HaloCatalog"]
    _, intensity = calculate_cii(halo_catalog, model, seed=seed)

    # The HaloCatalog holds the Lagrangian halo positions. Move the halos to their
    # Eulerian positions, as 21cmFAST does for the HaloBox. perturb_halo_catalog
    # reads every halo array but only loads the coordinates and masses from disk
    # itself, and crashes if the others are not in memory, so load them all.
    halo_catalog.load_all()
    perturbed = p21c.perturb_halo_catalog(
        initial_conditions=cache_obj["InitialConditions"],
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
        initial_conditions=cache_obj["InitialConditions"],
        perturbed_field=cache_obj["PerturbedField"],
        ionized_box=cache_obj["IonizedBox"],
        brightness_temperature=cache_obj["BrightnessTemp"],
        halobox=cache_obj["HaloBox"],
        CII_IntensityBox=cii_intensity,
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


def coevals_to_lightcone(
    coeval_list: list[p21c.Coeval],
    lcn: Lightconer,
    quantity: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Interpolate one field of a sequence of coeval boxes onto a lightcone.

    Each pair of consecutive boxes is interpolated with
    :meth:`~py21cmfast.lightconers.Lightconer.make_lightcone_slices`, as in
    :func:`py21cmfast.run_lightcone`.

    Parameters
    ----------
    coeval_list : list of py21cmfast.Coeval
        Coeval boxes at consecutive node redshifts, each with the field ``quantity``.
    lcn : Lightconer
        The lightconer that sets the lightcone geometry. Its ``quantities`` are
        ignored.
    quantity : str
        Name of the coeval field to interpolate.
    shape : tuple of int
        Shape of the lightcone field.

    Returns
    -------
    np.ndarray
        The field on the lightcone.
    """
    field_lcn = attrs.evolve(lcn, quantities=(quantity,))
    lightcone_field = np.zeros(shape)
    for coeval1, coeval2 in pairwise(coeval_list):
        for scalar_field, lcidx, slice_data in field_lcn.make_lightcone_slices(
            c1=coeval1, c2=coeval2
        ):
            # The lightconer can return no slice, or slices beyond the lightcone.
            if scalar_field is not None and lcidx < shape[-1]:
                lightcone_field[..., lcidx] = slice_data
    return lightcone_field


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
        Seed or generator for the scatter of the [CII] model. If None, a random
        seed is used.

    Returns
    -------
    py21cmfast.LightCone
        A copy of ``lc`` with two new fields in ``lightcones``:
        ``"CII_IntensityBox"`` (Jy km/s per cell) and ``"CII_BrightnessDensity"``
        (Jy/sr).
    """
    coeval_list = gen_coeval_list(lc, model, directory_path, seed=seed)
    intensity = coevals_to_lightcone(
        coeval_list,
        lcn,
        quantity="CII_IntensityBox",
        shape=lc.lightcones["brightness_temp"].shape,
    )

    cii_lc = attrs.evolve(
        lc, lightcones={**lc.lightcones, "CII_IntensityBox": intensity}
    )
    logger.info("Calculating the [CII] surface brightness")
    cii_lc.lightcones["CII_BrightnessDensity"] = calculate_bvoxel(cii_lc)

    return cii_lc
