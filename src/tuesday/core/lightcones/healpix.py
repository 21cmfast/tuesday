r"""HEALPix pixels for angular lightcones.

An angular lightcone on a HEALPix grid holds, for each shell, a map of the field on
the pixels of the grid. These functions choose the grid's resolution from the cells
of the simulation box, give the directions at which to sample each pixel, select a
patch of the sky, and average maps to a coarser grid.

A HEALPix pixel of a grid with :math:`N_{\rm side}` has a side (the root of its area)
of :math:`\sqrt{\pi / 3} / N_{\rm side}` radians, and a cell of size :math:`\Delta`
subtends :math:`\Delta / \chi` radians at comoving distance :math:`\chi`. Pixels much
larger than the cells alias the field unless each pixel averages it over sub-pixels:
the :math:`4^k` pixels of :math:`N_{\rm side} 2^k` that tile it exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import astropy.units as un
import numpy as np
from astropy.coordinates import angular_separation
from astropy_healpix import HEALPix

_ORDERS = ("ring", "nested")


def _check_order(order: str) -> None:
    if order not in _ORDERS:
        raise ValueError(f"order must be 'ring' or 'nested', not {order!r}")


def pixel_side(nside: int) -> un.Quantity[un.rad]:
    """Return the side of a HEALPix pixel, the square root of its area.

    Parameters
    ----------
    nside
        The Nside of the grid.

    Returns
    -------
    un.Quantity[un.rad]
        The side of each pixel.
    """
    if nside < 1:
        raise ValueError("nside must be positive")
    return np.sqrt(np.pi / 3) / nside * un.rad


def nside_for_cell_size(
    cell_size: un.Quantity[un.Mpc], distance: un.Quantity[un.Mpc]
) -> int:
    """Return the smallest HEALPix Nside whose pixels are no larger than a cell.

    A cell subtends a larger angle on near shells than on far ones. At the farthest
    distance of a lightcone, the pixels resolve the cells on every shell. At a nearer
    distance, the pixels are larger than the cells further out, where each pixel
    should then average sub-pixels (see :func:`subsample_level_for_cell_size`).

    Parameters
    ----------
    cell_size
        The size of the cells of the simulation box.
    distance
        The comoving distance at which pixels and cells should match.

    Returns
    -------
    int
        The Nside, a power of 2.

    Examples
    --------
    For 1.5 Mpc cells out to z = 35 (11773 Mpc with Planck18):

    >>> import astropy.units as un
    >>> nside_for_cell_size(1.5 * un.Mpc, 11773 * un.Mpc)
    8192
    """
    if not (cell_size > 0 and distance > 0):
        raise ValueError("cell_size and distance must be positive")
    cells_per_radian = (distance / cell_size).to_value(un.dimensionless_unscaled)
    return 2 ** max(
        0, int(np.ceil(np.log2(pixel_side(1).to_value(un.rad) * cells_per_radian)))
    )


def subsample_level_for_cell_size(
    nside: int, cell_size: un.Quantity[un.Mpc], distance: un.Quantity[un.Mpc]
) -> int:
    """Return the smallest sub-sample level whose sub-pixels are no larger than a cell.

    With sub-sample level ``k``, each pixel is the mean over its ``4**k``
    sub-pixels, the pixels of ``Nside = nside * 2**k`` that tile it (see
    :func:`subpixel_lonlat`). This returns the smallest ``k`` for which the
    sub-pixels are no larger than a cell at ``distance``
    (:func:`nside_for_cell_size`), so that the mean takes in every cell inside the
    pixel. It is 0 if the pixels themselves are no larger than the cells. Each level
    costs four times as many samples.

    Parameters
    ----------
    nside
        The Nside of the maps.
    cell_size
        The size of the cells of the simulation box.
    distance
        The comoving distance at which sub-pixels and cells should match, usually the
        farthest distance of the lightcone.

    Returns
    -------
    int
        The sub-sample level.

    Examples
    --------
    Maps of Nside 2048 from a box of 1.5 Mpc cells out to z = 35 (11773 Mpc with
    Planck18) need 16 sub-pixels per pixel:

    >>> import astropy.units as un
    >>> subsample_level_for_cell_size(2048, 1.5 * un.Mpc, 11773 * un.Mpc)
    2
    """
    if nside < 1:
        raise ValueError("nside must be positive")
    matching = nside_for_cell_size(cell_size, distance)
    return max(0, int(np.ceil(np.log2(matching / nside))))


def _check_pixels(pixels: Sequence[int] | np.ndarray, npix: int) -> np.ndarray:
    pixels = np.asarray(pixels)
    if (
        pixels.ndim != 1
        or pixels.size == 0
        or not np.issubdtype(pixels.dtype, np.integer)
        or pixels.min() < 0
        or pixels.max() >= npix
        or np.unique(pixels).size != pixels.size
    ):
        raise ValueError(
            "pixels must be a non-empty 1D array of distinct pixel indices "
            f"between 0 and {npix - 1}"
        )
    return pixels


def subpixel_lonlat(
    nside: int,
    *,
    order: Literal["ring", "nested"] = "ring",
    subsample_level: int = 0,
    pixels: Sequence[int] | np.ndarray | None = None,
) -> tuple[un.Quantity[un.rad], un.Quantity[un.rad]]:
    """Return the centres of the sub-pixels of HEALPix pixels.

    With ``subsample_level = 0`` these are the centres of the pixels. With
    ``subsample_level = k``, they are the centres of the ``4**k`` pixels of
    ``Nside = nside * 2**k`` that tile each pixel exactly. The sub-pixels have equal
    areas, so the unweighted mean of a field over them estimates the mean of the field
    over the pixel.

    Parameters
    ----------
    nside
        The Nside of the grid, a power of 2 (astropy-healpix supports no other).
    order
        The ordering of the pixels, "ring" or "nested".
    subsample_level
        How many times the pixels are divided into four.
    pixels
        The distinct indices (in ``order``) of the pixels, for part of the sky. All
        ``12 * nside**2`` pixels by default.

    Returns
    -------
    longitude, latitude
        The longitude in [0, 2pi) and the latitude of each sub-pixel centre, of shape
        ``(4**subsample_level, n_pixels)``.
    """
    _check_order(order)
    if subsample_level < 0:
        raise ValueError("subsample_level must be non-negative")
    grid = HEALPix(nside=nside, order=order)
    pixels = (
        np.arange(grid.npix) if pixels is None else _check_pixels(pixels, grid.npix)
    )

    # The sub-pixels of a pixel are contiguous in nested ordering: sub-pixel m of
    # pixel p is p * 4**k + m.
    nested = grid.ring_to_nested(pixels) if order == "ring" else pixels
    n_sub = 4**subsample_level
    subgrid = HEALPix(nside=nside * 2**subsample_level, order="nested")
    lon, lat = subgrid.healpix_to_lonlat(
        (nested[None, :] * n_sub + np.arange(n_sub)[:, None]).ravel()
    )
    shape = (n_sub, pixels.size)
    return lon.to(un.rad).reshape(shape), lat.to(un.rad).reshape(shape)


def disc_pixels(
    nside: int,
    longitude: un.Quantity[un.rad],
    latitude: un.Quantity[un.rad],
    radius: un.Quantity[un.rad],
    *,
    order: Literal["ring", "nested"] = "ring",
) -> np.ndarray:
    """Return the pixels whose centres are within a radius of a point on the sky.

    This selects a round patch of the sky, e.g. the size of a telescope's beam.

    Parameters
    ----------
    nside
        The Nside of the grid, a power of 2 (astropy-healpix supports no other).
    longitude, latitude
        The centre of the patch.
    radius
        The angular radius of the patch.
    order
        The ordering of the pixels, "ring" or "nested".

    Returns
    -------
    np.ndarray
        The sorted indices (in ``order``) of the pixels.
    """
    _check_order(order)
    if not radius > 0:
        raise ValueError("radius must be positive")
    grid = HEALPix(nside=nside, order=order)
    # cone_search_lonlat also returns the pixels that only overlap the disc.
    candidates = grid.cone_search_lonlat(longitude, latitude, radius)
    lon, lat = grid.healpix_to_lonlat(candidates)
    separation = angular_separation(lon, lat, longitude, latitude)
    return np.sort(candidates[separation <= radius])


def degrade(
    maps: np.ndarray, nside_out: int, *, order: Literal["ring", "nested"] = "ring"
) -> np.ndarray:
    """Average full-sky HEALPix maps to a coarser grid.

    Each pixel of the coarser grid is the mean of the pixels of the finer grid that
    tile it.

    Parameters
    ----------
    maps
        Full-sky maps, of shape ``(12 * nside**2, ...)``: the pixels on the first axis,
        e.g. the ``(n_pixels, n_slices)`` of an angular lightcone.
    nside_out
        The Nside of the coarser grid. ``nside / nside_out`` must be a power of 2.
    order
        The ordering of the pixels of both grids, "ring" or "nested".

    Returns
    -------
    np.ndarray
        The averaged maps, of shape ``(12 * nside_out**2, ...)``.
    """
    _check_order(order)
    maps = np.asarray(maps)
    npix = maps.shape[0]
    nside = round(float(np.sqrt(npix / 12)))
    if 12 * nside**2 != npix:
        raise ValueError(f"{npix} pixels is not a full-sky HEALPix map")
    ratio = nside // nside_out if nside_out >= 1 else 0
    if ratio < 1 or ratio * nside_out != nside or ratio & (ratio - 1):
        raise ValueError(
            f"nside_out must be the map's Nside ({nside}) divided by a power of 2"
        )
    if ratio == 1:
        return maps.copy()

    fine = HEALPix(nside=nside, order=order)
    coarse = HEALPix(nside=nside_out, order=order)
    if order == "ring":
        # reorder to nested, where the children of a pixel are contiguous
        maps = maps[fine.nested_to_ring(np.arange(npix))]
    out = maps.reshape(coarse.npix, ratio**2, *maps.shape[1:]).mean(axis=1)
    if order == "ring":
        out = out[coarse.ring_to_nested(np.arange(coarse.npix))]
    return out
