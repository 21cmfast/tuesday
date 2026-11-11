"""Sample periodic boxes on spherical shells, for angular lightcones.

An angular lightcone holds a field on spherical shells around an observer. A shell is
made from a simulation box tiled periodically in all directions: the field is sampled
at the points where the shell's lines of sight cross it. These functions sample such
a box at points given in units of the cell size, with the box's (0, 0, 0) cell at the
origin of the coordinates.

Sampling is linear in the box. Sampling two boxes and interpolating the samples
therefore gives the same result as interpolating the boxes and sampling the result,
at a cost set by the number of points rather than the number of cells.
:func:`sample_cellwise` extends this to rules that are applied cell by cell and are
not linear, such as the interpolation of the reionization redshift.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import map_coordinates


def shell_coordinates(
    directions: np.ndarray,
    radius: float,
    origin: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return the coordinates of points at a given distance along lines of sight.

    Parameters
    ----------
    directions
        Unit vectors of the lines of sight, of shape ``(3, n)``.
    radius
        The distance of the points from ``origin``, in units of the cell size.
    origin
        The position of the observer, in units of the cell size. The corner of the
        box's (0, 0, 0) cell by default.

    Returns
    -------
    np.ndarray
        The coordinates of the points, of shape ``(3, n)``.
    """
    coordinates = radius * np.asarray(directions, dtype=float)
    if origin is not None:
        coordinates += np.asarray(origin, dtype=float).reshape(3, 1)
    return coordinates


def sample_periodic(
    box: np.ndarray,
    coordinates: np.ndarray,
    *,
    order: int = 1,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a periodic box at the given coordinates.

    The box is tiled periodically, so coordinates outside it wrap around. Order 0
    takes the value of the nearest cell, and order 1 interpolates linearly between the
    eight cells around each point (trilinear interpolation). This is
    :func:`scipy.ndimage.map_coordinates` with ``mode="grid-wrap"``, as used by
    cosmotile and 21cmFAST for angular lightcones.

    Parameters
    ----------
    box
        The 3D box.
    coordinates
        The coordinates of the points, in units of the cell size, of shape ``(3, n)``.
    order
        The order of the interpolation, 0 or 1. Higher orders need the whole box to
        be pre-filtered, which is what sampling a few points avoids.
    output
        An array of shape ``(n,)`` to write the samples into. A new float64 array by
        default.

    Returns
    -------
    np.ndarray
        The samples, of shape ``(n,)``.
    """
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, not {order}")
    box = np.asarray(box)
    if box.ndim != 3:
        raise ValueError(f"box must be 3D, but has shape {box.shape}")
    if output is None:
        output = np.empty(np.shape(coordinates)[-1])
    map_coordinates(
        box, coordinates, order=order, mode="grid-wrap", prefilter=False, output=output
    )
    return output


def sample_cellwise(
    rule: Callable[..., np.ndarray],
    boxes: Sequence[np.ndarray],
    coordinates: np.ndarray,
    *,
    output: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a cell-by-cell combination of boxes trilinearly at the given coordinates.

    This gives what :func:`sample_periodic` with ``order=1`` gives for the box
    ``rule(*boxes)``, but evaluates ``rule`` only at the eight cells around each point
    instead of at every cell of the boxes. It is meant for rules that are not linear,
    so that the boxes cannot be sampled first and combined afterwards.

    Parameters
    ----------
    rule
        A function of the values of each box at the same cells (arrays of equal
        shape), returning the combined values at those cells.
    boxes
        The 3D boxes, all of the same shape.
    coordinates
        The coordinates of the points, in units of the cell size, of shape ``(3, n)``.
    output
        An array of shape ``(n,)`` to write the samples into. A new float64 array by
        default.

    Returns
    -------
    np.ndarray
        The samples, of shape ``(n,)``.
    """
    boxes = [np.asarray(box) for box in boxes]
    shape = boxes[0].shape
    if len(shape) != 3 or any(box.shape != shape for box in boxes):
        raise ValueError("boxes must be 3D and all of the same shape")

    coordinates = np.asarray(coordinates, dtype=float)
    floor = np.floor(coordinates)
    fraction = coordinates - floor
    size = np.array(shape).reshape(3, 1)
    lower = floor.astype(np.int64) % size
    upper = (lower + 1) % size

    if output is None:
        output = np.empty(coordinates.shape[-1])
    output[...] = 0.0
    for corner in np.ndindex(2, 2, 2):
        index = tuple(
            upper[axis] if c else lower[axis] for axis, c in enumerate(corner)
        )
        weight = np.prod(
            [
                fraction[axis] if c else 1 - fraction[axis]
                for axis, c in enumerate(corner)
            ],
            axis=0,
        )
        output += weight * np.asarray(rule(*(box[index] for box in boxes)), dtype=float)
    return output


def line_of_sight_component(vectors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return the line-of-sight component of vectors, positive towards the observer.

    This is the convention of cosmotile and 21cmFAST for the line-of-sight velocity.

    Parameters
    ----------
    vectors
        The vectors at each point, of shape ``(3, n)``.
    directions
        The lines of sight from the observer to each point, of shape ``(3, n)``. They
        are normalised here.

    Returns
    -------
    np.ndarray
        The component of each vector along the line of sight, towards the observer,
        of shape ``(n,)``.
    """
    directions = np.asarray(directions, dtype=float)
    norm = np.sqrt(np.sum(np.square(directions), axis=0))
    return -np.sum(np.asarray(vectors) * directions, axis=0) / norm


def average_subsamples(values: np.ndarray, n_subsamples: int) -> np.ndarray:
    """Average the sub-samples of each pixel.

    A pixel larger than the cells of a box is better represented by the mean of the
    field over several points inside it than by the value at its centre. The values
    are ordered sub-sample by sub-sample: the first ``n_pixels`` values are the first
    sub-sample of each pixel, and so on.

    Parameters
    ----------
    values
        The values at every sub-sample, of shape ``(n_subsamples * n_pixels,)``.
    n_subsamples
        The number of sub-samples per pixel.

    Returns
    -------
    np.ndarray
        The mean over the sub-samples of each pixel, of shape ``(n_pixels,)``. The
        values themselves if ``n_subsamples`` is 1.
    """
    values = np.asarray(values)
    if n_subsamples == 1:
        return values
    if n_subsamples < 1 or values.shape[0] % n_subsamples:
        raise ValueError(
            f"{values.shape[0]} values cannot be split into {n_subsamples} sub-samples"
        )
    return values.reshape(n_subsamples, -1, *values.shape[1:]).mean(axis=0)


def _in_chunks(
    work: Callable[[slice], None], n: int, *, threads: int, chunk_size: int
) -> None:
    """Call ``work`` on consecutive slices of ``range(n)``, in threads if asked."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    chunks = [slice(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
    if threads > 1 and len(chunks) > 1:
        # map_coordinates releases the GIL, so threads sample chunks in parallel.
        with ThreadPoolExecutor(threads) as executor:
            list(executor.map(work, chunks))
    else:
        for chunk in chunks:
            work(chunk)


def sample_shell(
    boxes: Sequence[np.ndarray],
    directions: np.ndarray,
    radius: float,
    *,
    origin: Sequence[float] | np.ndarray | None = None,
    order: int = 1,
    threads: int = 1,
    chunk_size: int = 2**20,
) -> np.ndarray:
    """Sample periodic boxes on a spherical shell.

    Each box is sampled with :func:`sample_periodic` at the points at ``radius`` along
    each line of sight (:func:`shell_coordinates`). The points are handled
    ``chunk_size`` at a time, so that their coordinates never take more memory than
    one chunk, and the chunks are shared between ``threads`` threads.

    Parameters
    ----------
    boxes
        The 3D boxes, all of the same shape.
    directions
        Unit vectors of the lines of sight, of shape ``(3, n)``.
    radius
        The radius of the shell, in units of the cell size.
    origin
        The position of the observer, in units of the cell size. The corner of the
        box's (0, 0, 0) cell by default.
    order
        The order of the interpolation, 0 or 1 (see :func:`sample_periodic`).
    threads
        The number of threads.
    chunk_size
        The number of points handled at a time.

    Returns
    -------
    np.ndarray
        The samples of each box, of shape ``(len(boxes), n)``.
    """
    directions = np.asarray(directions)
    out = np.empty((len(boxes), directions.shape[-1]))

    def work(chunk: slice) -> None:
        coordinates = shell_coordinates(directions[:, chunk], radius, origin)
        for box, samples in zip(boxes, out, strict=True):
            sample_periodic(box, coordinates, order=order, output=samples[chunk])

    _in_chunks(work, directions.shape[-1], threads=threads, chunk_size=chunk_size)
    return out


def sample_shell_cellwise(
    rule: Callable[..., np.ndarray],
    boxes: Sequence[np.ndarray],
    directions: np.ndarray,
    radius: float,
    *,
    origin: Sequence[float] | np.ndarray | None = None,
    threads: int = 1,
    chunk_size: int = 2**20,
) -> np.ndarray:
    """Sample a cell-by-cell combination of boxes on a spherical shell, trilinearly.

    This is :func:`sample_cellwise` at the points of :func:`sample_shell`.

    Parameters
    ----------
    rule
        A function of the values of each box at the same cells, returning the
        combined values (see :func:`sample_cellwise`).
    boxes
        The 3D boxes, all of the same shape.
    directions
        Unit vectors of the lines of sight, of shape ``(3, n)``.
    radius
        The radius of the shell, in units of the cell size.
    origin
        The position of the observer, in units of the cell size. The corner of the
        box's (0, 0, 0) cell by default.
    threads
        The number of threads.
    chunk_size
        The number of points handled at a time.

    Returns
    -------
    np.ndarray
        The samples, of shape ``(n,)``.
    """
    directions = np.asarray(directions)
    out = np.empty(directions.shape[-1])

    def work(chunk: slice) -> None:
        coordinates = shell_coordinates(directions[:, chunk], radius, origin)
        sample_cellwise(rule, boxes, coordinates, output=out[chunk])

    _in_chunks(work, directions.shape[-1], threads=threads, chunk_size=chunk_size)
    return out
