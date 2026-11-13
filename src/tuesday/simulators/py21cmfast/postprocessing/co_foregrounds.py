"""Add the CO interloper foreground to a 21cmFAST [CII] lightcone.

The CO lines in a [CII] map come from redshifts of about 0-3, far below those of the
[CII] lightcone, so their emitters trace a separate density lightcone. It is built
with :func:`build_density_lightcone` from its own 21cmFAST initial conditions, and
:func:`add_co_foregrounds` draws the CO foreground with
:func:`tuesday.core.postprocessing.co_foregrounds.co_foreground_brightness`.
"""

import logging

import attrs
import numpy as np
import py21cmfast as p21c
from astropy import units as un
from py21cmfast.io.caching import OutputCache

from tuesday.core.cosmology import comoving_distance_to_redshift
from tuesday.core.postprocessing.co_foregrounds import (
    CO_TRANSITIONS,
    LOG_L_RATIO,
    DensityLightcone,
    co_foreground_brightness,
)

logger = logging.getLogger(__name__)


def _orient(box: np.ndarray, index: int) -> np.ndarray:
    """Rotate or flip the box in the transverse plane, with a period of four boxes.

    Consecutive boxes along the line of sight repeat the same initial conditions;
    changing their orientation hides the repetition.
    """
    match index % 4:
        case 1:
            return np.rot90(box, axes=(0, 1))
        case 2:
            return np.flip(box, axis=0)
        case 3:
            return np.rot90(np.flip(box, axis=1), axes=(0, 1))
    return box


def build_density_lightcone(
    inputs: p21c.InputParameters,
    max_redshift: float,
    cache: OutputCache | None = None,
) -> DensityLightcone:
    """Build a matter-density lightcone from the observer out to ``max_redshift``.

    The slices are one cell apart, the first half a cell from the observer. The
    lightcone is cut into consecutive boxes of ``HII_DIM`` slices. Each box is the
    density of the same initial conditions, evolved with
    :func:`py21cmfast.perturb_field` to the redshift of the middle slice of the box,
    and rotated or flipped in the transverse plane with a period of four boxes so
    that the repeated structures do not line up.

    Parameters
    ----------
    inputs : py21cmfast.InputParameters
        Parameters of the density simulation; its cell size and number of cells set
        the lightcone resolution. To use the lightcone with
        :func:`add_co_foregrounds`, they must equal those of the [CII] lightcone,
        whose pixels the density cells are mapped to one to one.
    max_redshift : float
        Redshift up to which the lightcone extends. To cover the CO interlopers of
        a [CII] lightcone, use the largest CO redshift, ``nu_co / nu_obs - 1`` at
        the lowest observed frequency, e.g. about 3 for CO J=6-5 at 170 GHz.
    cache : OutputCache, optional
        Cache to read and write the 21cmFAST outputs. If None, nothing is cached.

    Returns
    -------
    DensityLightcone
        The density contrast lightcone.
    """
    simulation = inputs.simulation_options
    n_cells = simulation.HII_DIM
    cell = simulation.cell_size.to(un.Mpc)
    cosmo = inputs.cosmo_params.cosmo
    max_distance = cosmo.comoving_distance(max_redshift).to_value(un.Mpc)
    distances = np.arange(cell.value / 2, max_distance, cell.value) * un.Mpc
    redshifts = comoving_distance_to_redshift(distances, cosmo)
    cache_kwargs = {} if cache is None else {"cache": cache, "write": True}

    initial_conditions = p21c.compute_initial_conditions(inputs=inputs, **cache_kwargs)
    density = np.empty((n_cells, n_cells, distances.size), dtype=np.float32)
    for index, start in enumerate(range(0, distances.size, n_cells)):
        stop = min(start + n_cells, distances.size)
        redshift = float(redshifts[min(start + n_cells // 2, distances.size - 1)])
        logger.info("Density box %d at z=%.3f", index, redshift)
        perturbed = p21c.perturb_field(
            redshift=redshift,
            inputs=inputs,
            initial_conditions=initial_conditions,
            **cache_kwargs,
        )
        box = _orient(perturbed.get("density"), index)
        density[..., start:stop] = box[..., : stop - start]

    return DensityLightcone(density=density, distances=distances, cell_size=cell)


def add_co_foregrounds(
    lc: p21c.LightCone,
    density_lightcone: DensityLightcone,
    transitions: tuple[str, ...] = CO_TRANSITIONS,
    log_l_ratio: np.ndarray = LOG_L_RATIO,
    poisson: bool = True,
    seed: int | np.random.Generator | None = None,
) -> p21c.LightCone:
    """Add the CO interloper foreground to a [CII] lightcone.

    The foreground is drawn with
    :func:`~tuesday.core.postprocessing.co_foregrounds.co_foreground_brightness`,
    with the geometry and cosmology of ``lc``.

    Parameters
    ----------
    lc : py21cmfast.LightCone
        Rectilinear [CII] lightcone with a ``"cii_brightness"`` field in Jy/sr, as
        produced by :func:`~.cii_lightcone.gen_lc`. It is not modified.
    density_lightcone : DensityLightcone
        Matter density covering the CO redshifts, e.g. from
        :func:`build_density_lightcone`, with the same number of cells across and
        cell size as ``lc``.
    transitions : tuple of str, optional
        The CO transitions to include; all of ``CO J=1-0`` to ``CO J=6-5`` by
        default.
    log_l_ratio : np.ndarray, optional
        Uniform grid of ``log10(L / L*)`` of the luminosity bins.
    poisson : bool, optional
        If False, use the expected brightness of each pixel instead of a Poisson
        realisation.
    seed : int or np.random.Generator, optional
        Seed or generator for the Poisson draws. If None, a random seed is used.

    Returns
    -------
    py21cmfast.LightCone
        A copy of ``lc`` with two new fields in Jy/sr: ``"co_brightness"`` (the CO
        foreground) and ``"cii_co_brightness"`` ([CII] plus CO). The other fields
        are shared with ``lc``.

    Raises
    ------
    KeyError
        If ``lc`` has no ``"cii_brightness"`` field.
    ValueError
        If the density lightcone does not cover the CO redshifts, or its cells
        differ from those of ``lc``.
    """
    if "cii_brightness" not in lc.lightcones:
        raise KeyError(
            "The lightcone has no 'cii_brightness' field; build it with gen_lc."
        )
    simulation = lc.simulation_options
    co = co_foreground_brightness(
        lc.lightcone_distances,
        simulation.HII_DIM,
        simulation.BOX_LEN / simulation.HII_DIM * un.Mpc,
        lc.cosmo_params.cosmo,
        density_lightcone,
        transitions=transitions,
        log_l_ratio=log_l_ratio,
        poisson=poisson,
        seed=seed,
    ).to_value(un.Jy / un.sr)
    return attrs.evolve(
        lc,
        lightcones={
            **lc.lightcones,
            "co_brightness": co,
            "cii_co_brightness": lc.lightcones["cii_brightness"] + co,
        },
    )
