"""Build a [CII] line-intensity lightcone from cached 21cmFAST halo catalogues.

For each node redshift of a 21cmFAST lightcone, the halo catalogue is read from the
output cache, the [CII] emission of each halo is computed with
:func:`~.cii_calculation.calculate_cii` and gridded into a coeval box. The boxes are
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
    """
    output_cache = OutputCache(direc=directory_path)
    cache_obj = {
        cache_file.stem: read_output_struct(cache_file)
        for cache_file in output_cache.list_datasets(redshift=z)
    }

    halo_catalog = cache_obj["HaloCatalog"]
    _, intensity = calculate_cii(halo_catalog, model, seed=seed)

    box_len = halo_catalog.simulation_options.BOX_LEN
    resolution = halo_catalog.simulation_options.HII_DIM
    # Halo coordinates are in Mpc; convert them to cell units. Only the first
    # n_halos entries are real halos, matching the output of calculate_cii.
    coordinates = halo_catalog.get("halo_coords")[: halo_catalog.n_halos] / (
        box_len / resolution
    )
    cii_intensity, _ = np.histogramdd(
        coordinates,
        bins=resolution,
        weights=intensity,
        range=[[0, resolution], [0, resolution], [0, resolution]],
    )

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
    cii_lcn = attrs.evolve(lcn, quantities=("CII_IntensityBox",))
    intensity = np.zeros(lc.lightcones["brightness_temp"].shape)

    coeval_list = gen_coeval_list(lc, model, directory_path, seed=seed)
    for coeval1, coeval2 in pairwise(coeval_list):
        for scalar_field, lcidx, slice_data in cii_lcn.make_lightcone_slices(
            c1=coeval1, c2=coeval2
        ):
            # The lightconer can return no slice, or slices beyond the lightcone.
            if scalar_field is not None and lcidx < intensity.shape[-1]:
                intensity[..., lcidx] = slice_data

    cii_lc = attrs.evolve(
        lc, lightcones={**lc.lightcones, "CII_IntensityBox": intensity}
    )
    logger.info("Calculating the [CII] surface brightness")
    cii_lc.lightcones["CII_BrightnessDensity"] = calculate_bvoxel(cii_lc)

    return cii_lc
