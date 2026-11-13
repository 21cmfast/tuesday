"""Make one slice of an angular lightcone from the two neighbouring coevals.

A slice of an angular lightcone lies at a distance between those of two simulation
boxes (coevals). It is the field on a spherical shell of that radius, interpolated in
redshift between the two boxes. The boxes are sampled on the shell by a slice
interpolator of cosmotile (:func:`cosmotile.make_lightcone_slice_interpolator`),
which tiles the box periodically and can average each pixel over several directions.
A slice can be made in two ways that give the same result, to rounding:

* :func:`interpolate_then_sample` interpolates the two boxes, cell by cell, and
  samples the interpolated box on the shell. Its cost
  grows with the number of cells.
* :func:`sample_then_interpolate` samples each box on the shell and interpolates the
  samples. Its cost grows with the number of points sampled.

:func:`shell_slice` chooses the cheaper of the two (:func:`samples_first`), unless it
is told which to use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from astropy import units

#: Roughly how many cells of a box can be interpolated in redshift in the time it
#: takes to sample a box at one point (trilinear ``map_coordinates``). The two ways of
#: making a slice cost the same at about 9-20 cells per point (measured on one core
#: for 128^3 and 256^3 boxes and 10^3-10^6 directions, for 21cmFAST's
#: ``perf/angular-lightcone-sample-first``).
CELLS_PER_SAMPLE = 16

Interpolation = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _plain(values: Any) -> np.ndarray:
    """Return an array or a Quantity (of pixels, or dimensionless) as a plain array."""
    if isinstance(values, units.Quantity):
        if values.unit.is_equivalent(units.dimensionless_unscaled):
            return values.to_value(units.dimensionless_unscaled)
        return values.value
    return np.asarray(values)


def n_samples(interpolator: Any) -> int:
    """Return the number of points at which a slice interpolator samples a box.

    Parameters
    ----------
    interpolator
        A slice interpolator of cosmotile
        (:func:`cosmotile.make_lightcone_slice_interpolator`).

    Returns
    -------
    int
        The number of points, sub-samples included.
    """
    return np.shape(interpolator.keywords["coordinates"])[-1]


def samples_first(n_cells: int, n_samples: int, *, order: int = 1) -> bool:
    """Whether sampling the boxes first is cheaper than interpolating them first.

    Interpolating two boxes costs about one operation per cell, and sampling costs
    about :data:`CELLS_PER_SAMPLE` (16) operations per point sampled. Sampling first is
    cheaper when the boxes have more than :data:`CELLS_PER_SAMPLE` cells per point.
    For example, a 1500^3 box with a ring of 2^15 directions samples first, while a
    228^3 box with a HEALPix map of Nside 1024 (1.3e7 directions) interpolates first.

    Parameters
    ----------
    n_cells
        The number of cells of each box.
    n_samples
        The number of points sampled, sub-samples included (:func:`n_samples`).
    order
        The order of the interpolation on the shell. For splines (orders above 1),
        sampling first is never cheaper: each box is pre-filtered as a whole before it
        is sampled, so sampling two boxes first pre-filters twice instead of once.

    Returns
    -------
    bool
        Whether to sample first. Always False for ``order`` above 1.
    """
    if order > 1:
        return False
    return n_samples * CELLS_PER_SAMPLE < n_cells


def sample_cellwise(
    rule: Interpolation,
    boxes: Sequence[np.ndarray],
    coordinates: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Sample, trilinearly, a rule applied cell by cell to two periodic boxes.

    This gives what sampling ``rule(*boxes)`` with ``map_coordinates`` (order 1,
    ``mode="grid-wrap"``, as cosmotile samples) gives, but the rule is only evaluated
    at the eight cells around each point. It is how a rule that is not linear in the
    boxes, such as 21cmFAST's "mean_max" interpolation of the reionization redshift,
    can be sampled first.

    Parameters
    ----------
    rule
        A function of the values of both boxes at the same cells (arrays of equal
        shape), returning a value per cell.
    boxes
        The two 3D boxes, of the same shape.
    coordinates
        The coordinates of the points, in units of the cell size, of shape ``(3, n)``,
        e.g. ``interpolator.keywords["coordinates"]`` of a cosmotile slice
        interpolator.
    weights
        The weights of the sub-samples of each pixel, if the points are sub-samples
        laid out sub-sample by sub-sample (``interpolator.keywords["weights"]``).

    Returns
    -------
    np.ndarray
        The value at each point, or each pixel's weighted mean over its sub-samples.
    """
    box_a, box_b = (np.asarray(b) for b in boxes)
    if box_a.shape != box_b.shape or box_a.ndim != 3:
        raise ValueError("the boxes must be 3D and of the same shape")
    coordinates = _plain(coordinates).astype(float, copy=False)
    shape = np.array(box_a.shape)[:, None]
    floor = np.floor(coordinates)
    frac = coordinates - floor
    lower = floor.astype(np.int64) % shape
    upper = (lower + 1) % shape

    out = np.zeros(coordinates.shape[1])
    for corner in np.ndindex(2, 2, 2):
        index = tuple(upper[ax] if c else lower[ax] for ax, c in enumerate(corner))
        weight = np.prod(
            [frac[ax] if c else 1 - frac[ax] for ax, c in enumerate(corner)], axis=0
        )
        out += weight * _plain(rule(box_a[index], box_b[index]))
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        out = (weights[:, None] * out.reshape(weights.size, -1)).sum(axis=0)
    return out


def interpolate_then_sample(
    interpolate: Interpolation,
    box_a: np.ndarray,
    box_b: np.ndarray,
    interpolator: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Interpolate two boxes cell by cell, then sample the result on a shell.

    Parameters
    ----------
    interpolate
        The interpolation between the boxes: a function of the values of both boxes
        at the same cells (arrays of equal shape), returning the interpolated values,
        e.g. a linear interpolation in comoving distance.
    box_a, box_b
        The two 3D boxes around the slice, of the same shape.
    interpolator
        The slice interpolator of cosmotile for the shell
        (:func:`cosmotile.make_lightcone_slice_interpolator`).

    Returns
    -------
    np.ndarray
        The slice, one value per pixel.
    """
    return _plain(interpolator(_plain(interpolate(box_a, box_b))))


def sample_then_interpolate(
    interpolate: Interpolation,
    box_a: np.ndarray,
    box_b: np.ndarray,
    interpolator: Callable[[np.ndarray], np.ndarray],
    *,
    linear: bool,
) -> np.ndarray:
    """Sample two boxes on a shell, then interpolate the samples.

    This gives what :func:`interpolate_then_sample` gives. Sampling is linear in the
    box at any order (the spline pre-filter of orders above 1 is linear too), and so is
    the mean of each pixel over its sub-samples, so an interpolation that is linear in
    the values of the boxes can be applied to the samples. One that is not
    (``linear=False``, e.g. a rule that switches between the boxes cell by cell) is
    applied at the cells around each point before they are weighted
    (:func:`sample_cellwise`). That needs order 0 or 1: a spline sample depends on all
    cells of the box, so for orders above 1 such a rule is applied to the boxes first
    (:func:`interpolate_then_sample`).

    Parameters
    ----------
    interpolate
        The interpolation between the boxes (see :func:`interpolate_then_sample`).
    box_a, box_b
        The two 3D boxes around the slice, of the same shape.
    interpolator
        The slice interpolator of cosmotile for the shell
        (:func:`cosmotile.make_lightcone_slice_interpolator`).
    linear
        Whether ``interpolate`` is linear in the values of the boxes.

    Returns
    -------
    np.ndarray
        The slice, one value per pixel.
    """
    order = interpolator.keywords["order"]
    weights = interpolator.keywords.get("weights")
    if linear or (order == 0 and weights is None):
        # order 0 takes one cell per point, so any rule commutes with it
        return _plain(interpolate(interpolator(box_a), interpolator(box_b)))
    if order > 1:
        return interpolate_then_sample(interpolate, box_a, box_b, interpolator)
    coordinates = _plain(interpolator.keywords["coordinates"])
    if order == 0:
        # the cell around each point, as map_coordinates rounds with order 0
        coordinates = np.floor(coordinates + 0.5)
    return sample_cellwise(interpolate, (box_a, box_b), coordinates, weights=weights)


def shell_slice(
    interpolate: Interpolation,
    box_a: np.ndarray,
    box_b: np.ndarray,
    interpolator: Callable[[np.ndarray], np.ndarray],
    *,
    linear: bool,
    sample_first: bool | None = None,
) -> np.ndarray:
    """Make one slice of an angular lightcone from the two boxes around it.

    The slice is the interpolation of the two boxes, sampled on a shell. By default
    this samples the boxes first if that is cheaper (:func:`samples_first`, with
    :func:`sample_then_interpolate`), and otherwise interpolates them first
    (:func:`interpolate_then_sample`). Both give the same result, to rounding.

    Parameters
    ----------
    interpolate
        The interpolation between the boxes: a function of the values of both boxes
        at the same cells, returning the interpolated values.
    box_a, box_b
        The two 3D boxes around the slice, of the same shape.
    interpolator
        The slice interpolator of cosmotile for the shell
        (:func:`cosmotile.make_lightcone_slice_interpolator`).
    linear
        Whether ``interpolate`` is linear in the values of the boxes.
    sample_first
        True to sample first, False to interpolate first, None (the default) to
        choose the cheaper (:func:`samples_first`).

    Returns
    -------
    np.ndarray
        The slice, one value per pixel.
    """
    order = interpolator.keywords["order"]
    if sample_first is None:
        sample_first = samples_first(
            np.size(box_a), n_samples(interpolator), order=order
        )
    if sample_first:
        return sample_then_interpolate(
            interpolate, box_a, box_b, interpolator, linear=linear
        )
    return interpolate_then_sample(interpolate, box_a, box_b, interpolator)
