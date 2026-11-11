"""Make one slice of an angular lightcone from the two boxes around it.

A slice of an angular lightcone lies at a distance between those of two simulation
boxes (coevals). It is the field on a spherical shell of that radius, interpolated
in redshift between the two boxes. This can be done in two ways that give the same
result, to rounding:

* :func:`interpolate_then_sample` interpolates the two boxes, cell by cell, and
  samples the interpolated box on the shell. This is what 21cmFAST 4.3 does. Its cost
  grows with the number of cells.
* :func:`sample_then_interpolate` samples each box on the shell and interpolates the
  samples. Its cost grows with the number of directions.

:func:`shell_slice` chooses the cheaper of the two (:func:`samples_first`), unless it
is told which to use.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .sampling import sample_shell, sample_shell_cellwise

#: Roughly how many cells of a box can be interpolated in redshift in the time it
#: takes to sample a box in one direction (trilinear ``map_coordinates``). The two ways
#: of making a slice cost the same at about 9-20 cells per direction (measured on one
#: core for 128^3 and 256^3 boxes and 10^3-10^6 directions, for 21cmFAST's
#: ``perf/angular-lightcone-sample-first``).
CELLS_PER_SAMPLE = 16


def samples_first(n_cells: int, n_directions: int, *, order: int = 1) -> bool:
    """Whether sampling the boxes first is cheaper than interpolating them first.

    Interpolating two boxes costs about one operation per cell, and sampling costs
    about :data:`CELLS_PER_SAMPLE` (16) operations per direction. Sampling first is
    cheaper when the boxes have more than :data:`CELLS_PER_SAMPLE` cells per
    direction. For example, a 1500^3 box with a ring of 2^15 directions samples
    first, while a 228^3 box with a HEALPix map of Nside 1024 (1.3e7 directions)
    interpolates first.

    Parameters
    ----------
    n_cells
        The number of cells of each box.
    n_directions
        The number of directions sampled (sub-samples included).
    order
        The order of the interpolation on the shell. Only orders 0 and 1 can sample
        first.

    Returns
    -------
    bool
        Whether to sample first. Always False for ``order`` above 1.
    """
    if order > 1:
        return False
    return n_directions * CELLS_PER_SAMPLE < n_cells


def interpolate_then_sample(
    interpolate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    box_a: np.ndarray,
    box_b: np.ndarray,
    directions: np.ndarray,
    radius: float,
    *,
    origin: Sequence[float] | np.ndarray | None = None,
    order: int = 1,
    threads: int = 1,
    chunk_size: int = 2**20,
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
    directions
        Unit vectors of the lines of sight, of shape ``(3, n)``.
    radius
        The radius of the shell, in units of the cell size.
    origin
        The position of the observer, in units of the cell size. The corner of the
        box's (0, 0, 0) cell by default.
    order
        The order of the interpolation on the shell, 0 or 1.
    threads
        The number of threads that sample the box.
    chunk_size
        The number of directions sampled at a time.

    Returns
    -------
    np.ndarray
        The slice, of shape ``(n,)``.
    """
    box = np.asarray(interpolate(box_a, box_b), dtype=float)
    return sample_shell(
        (box,),
        directions,
        radius,
        origin=origin,
        order=order,
        threads=threads,
        chunk_size=chunk_size,
    )[0]


def sample_then_interpolate(
    interpolate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    box_a: np.ndarray,
    box_b: np.ndarray,
    directions: np.ndarray,
    radius: float,
    *,
    linear: bool,
    origin: Sequence[float] | np.ndarray | None = None,
    order: int = 1,
    threads: int = 1,
    chunk_size: int = 2**20,
) -> np.ndarray:
    """Sample two boxes on a shell, then interpolate the samples.

    This gives what :func:`interpolate_then_sample` gives. Sampling is linear in the
    box, so for an interpolation that is linear in the values of the boxes it can be
    applied to the samples. For one that is not (``linear=False``, e.g. a rule that
    switches between the boxes cell by cell), it is applied at the eight cells around
    each point before they are weighted (:func:`sample_shell_cellwise`). With
    ``order=0`` each point takes one cell, so any interpolation can be applied to the
    samples.

    Parameters
    ----------
    interpolate
        The interpolation between the boxes (see :func:`interpolate_then_sample`).
    box_a, box_b
        The two 3D boxes around the slice, of the same shape.
    directions
        Unit vectors of the lines of sight, of shape ``(3, n)``.
    radius
        The radius of the shell, in units of the cell size.
    origin
        The position of the observer, in units of the cell size. The corner of the
        box's (0, 0, 0) cell by default.
    order
        The order of the interpolation on the shell, 0 or 1.
    linear
        Whether ``interpolate`` is linear in the values of the boxes. It has no
        default: sampling first with an interpolation wrongly declared linear gives a
        different slice.
    threads
        The number of threads that sample the boxes.
    chunk_size
        The number of directions sampled at a time.

    Returns
    -------
    np.ndarray
        The slice, of shape ``(n,)``.
    """
    kw = {"origin": origin, "threads": threads, "chunk_size": chunk_size}
    if order == 1 and not linear:
        return sample_shell_cellwise(
            interpolate, (box_a, box_b), directions, radius, **kw
        )
    samples_a, samples_b = sample_shell(
        (box_a, box_b), directions, radius, order=order, **kw
    )
    return np.asarray(interpolate(samples_a, samples_b), dtype=float)


def shell_slice(
    interpolate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    box_a: np.ndarray,
    box_b: np.ndarray,
    directions: np.ndarray,
    radius: float,
    *,
    linear: bool,
    origin: Sequence[float] | np.ndarray | None = None,
    order: int = 1,
    sample_first: bool | None = None,
    threads: int = 1,
    chunk_size: int = 2**20,
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
    directions
        Unit vectors of the lines of sight, of shape ``(3, n)``.
    radius
        The radius of the shell, in units of the cell size.
    origin
        The position of the observer, in units of the cell size. The corner of the
        box's (0, 0, 0) cell by default.
    order
        The order of the interpolation on the shell, 0 or 1.
    linear
        Whether ``interpolate`` is linear in the values of the boxes. It has no
        default: sampling first with an interpolation wrongly declared linear gives a
        different slice.
    sample_first
        True to sample first, False to interpolate first, None (the default) to
        choose the cheaper.
    threads
        The number of threads that sample the boxes.
    chunk_size
        The number of directions sampled at a time.

    Returns
    -------
    np.ndarray
        The slice, of shape ``(n,)``.
    """
    if sample_first is None:
        sample_first = samples_first(
            np.size(box_a), np.shape(directions)[-1], order=order
        )
    kw = {
        "origin": origin,
        "order": order,
        "threads": threads,
        "chunk_size": chunk_size,
    }
    if sample_first:
        return sample_then_interpolate(
            interpolate, box_a, box_b, directions, radius, linear=linear, **kw
        )
    return interpolate_then_sample(interpolate, box_a, box_b, directions, radius, **kw)
