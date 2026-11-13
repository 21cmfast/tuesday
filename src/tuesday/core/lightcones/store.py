"""On-disk lightcone storage, with support for incremental checkpointing.

A lightcone made from the coeval boxes of a simulation is filled interval by interval:
the slices between two consecutive node redshifts come from the boxes at those two
redshifts. A :class:`LightconeStore` holds the lightcone of one quantity in an HDF5
file, together with a record of the intervals that are filled. A job that stops can
then resume where it stopped, and a job that runs while the simulation is still
running can fill the intervals whose boxes exist and leave the others for later.

The file holds these datasets:

``lightcone``
    The lightcone, with the slices on the last axis. Slices that are not filled yet
    are NaN.
``lightcone_distances``
    The comoving distance of each slice, in Mpc.
``lightcone_redshifts``
    The redshift of each slice.
``node_redshifts``
    The redshifts of the boxes that the lightcone is made from, in the order of the
    intervals (usually decreasing).
``global_mean``
    The mean of the quantity's box at each node redshift, NaN where it was not read.
``interval_done``
    Whether the slices between ``node_redshifts[i]`` and ``node_redshifts[i + 1]``
    are filled, for each ``i``.

The attributes of the file hold the quantity, whether the lightcone is complete, and
a ``metadata`` group that describes how the lightcone is made. A store is only
resumed if its layout and metadata are those requested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import astropy.units as un
import h5py
import numpy as np

#: The format of the files, recorded in their attributes.
FORMAT = "tuesday lightcone store 1"

#: About how many values one chunk of the ``lightcone`` dataset holds (1 MiB of
#: float32).
CHUNK_VALUES = 2**18

#: About how many values the lines of sight of the pixels of one chunk hold, over all
#: the slices (64 MiB of float32).
LINE_OF_SIGHT_VALUES = 2**24


def default_chunks(shape: Sequence[int]) -> tuple[int, ...]:
    """Return the chunks of a lightcone dataset of a given shape.

    The lightcone is written a few slices at a time, and read (e.g. for the dv/dr
    correction) a block of lines of sight at a time. A chunk therefore holds all the
    pixels of the other axes for a block of the first axis, and a block of slices.
    The block of the first axis is such that its lines of sight over all the slices
    hold about :data:`LINE_OF_SIGHT_VALUES` values. The block of slices is such that
    the chunk holds about :data:`CHUNK_VALUES` values. A chunk holds at least one
    pixel of the first axis and one slice.

    Parameters
    ----------
    shape
        The shape of the lightcone, with the slices on the last axis.

    Returns
    -------
    tuple[int, ...]
        The shape of a chunk.

    Examples
    --------
    >>> default_chunks((1500, 1500, 2500))
    (4, 1500, 43)
    >>> default_chunks((12 * 2048**2, 2500))
    (6710, 39)
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) < 2 or min(shape) < 1:
        raise ValueError("a lightcone has at least one slice and one pixel axis")
    first, *middle, n_slices = shape
    per_row = int(np.prod(middle, dtype=int))
    rows = min(
        first,
        LINE_OF_SIGHT_VALUES // (per_row * n_slices),
        CHUNK_VALUES // per_row,
    )
    rows = max(1, rows)
    slices = min(n_slices, max(1, CHUNK_VALUES // (rows * per_row)))
    return (rows, *middle, slices)


def _same(a: Any, b: Any) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-10, atol=0))
    return a.shape == b.shape and bool(np.all(a == b))


class LightconeStore:
    """The lightcone of one quantity in an HDF5 file, filled one interval at a time.

    Use :meth:`open_or_create` to start or resume a lightcone, and :meth:`open` to
    read one. The store keeps the file open until :meth:`close` is called, or until
    the end of a ``with`` block.

    Parameters
    ----------
    file
        The open HDF5 file of the store.
    """

    def __init__(self, file: h5py.File):
        if file.attrs.get("format") != FORMAT:
            name = file.filename
            file.close()
            raise ValueError(f"{name} is not a lightcone store")
        self._file = file

    @classmethod
    def open(cls, path: str | Path, mode: str = "r") -> LightconeStore:
        """Open an existing store.

        Parameters
        ----------
        path
            The file of the store.
        mode
            "r" to read the store, "r+" to fill it.

        Returns
        -------
        LightconeStore
            The store.
        """
        if mode not in ("r", "r+"):
            raise ValueError("mode must be 'r' or 'r+'")
        try:
            file = h5py.File(path, mode)
        except OSError as e:
            if not Path(path).exists():
                raise
            raise OSError(
                f"{path} cannot be opened ({e}). A job stopped while writing the file "
                "can leave it damaged; if so, create it again."
            ) from e
        return cls(file)

    @classmethod
    def open_or_create(
        cls,
        path: str | Path,
        *,
        quantity: str,
        shape: Sequence[int],
        distances: un.Quantity[un.Mpc],
        redshifts: Sequence[float] | np.ndarray,
        node_redshifts: Sequence[float] | np.ndarray,
        metadata: Mapping[str, Any] | None = None,
        chunks: Sequence[int] | None = None,
        clobber: bool = False,
    ) -> LightconeStore:
        """Open a store to fill it, creating it if it does not exist.

        An existing store is resumed if it has the same quantity, shape, distances,
        redshifts, node redshifts and metadata. Otherwise this raises an error, unless
        ``clobber`` is True, in which case the store is created again.

        Parameters
        ----------
        path
            The file of the store. Its directory is created if it does not exist.
        quantity
            The name of the quantity.
        shape
            The shape of the lightcone, with the slices on the last axis.
        distances
            The comoving distance of each slice.
        redshifts
            The redshift of each slice.
        node_redshifts
            The redshifts of the boxes that the lightcone is made from, in the order
            of the intervals.
        metadata
            A description of how the lightcone is made, as values that HDF5 can store
            as attributes (numbers, strings, and arrays of them).
        chunks
            The chunks of the ``lightcone`` dataset (:func:`default_chunks` by
            default).
        clobber
            Whether to create the store again if it exists.

        Returns
        -------
        LightconeStore
            The store, open for filling.
        """
        path = Path(path)
        shape = tuple(int(n) for n in shape)
        distances = un.Quantity(distances, un.Mpc).to_value(un.Mpc)
        redshifts = np.asarray(redshifts, dtype=float)
        node_redshifts = np.asarray(node_redshifts, dtype=float)
        metadata = dict(metadata or {})
        if not (distances.shape == redshifts.shape == (shape[-1],)):
            raise ValueError("there must be one distance and one redshift per slice")
        if node_redshifts.ndim != 1 or node_redshifts.size < 2:
            raise ValueError("there must be at least two node redshifts")

        if path.exists() and not clobber:
            store = cls.open(path, "r+")
            stored = {
                "quantity": (store.quantity, quantity),
                "shape": (store.shape, shape),
                "lightcone_distances": (store.distances.to_value(un.Mpc), distances),
                "lightcone_redshifts": (store.redshifts, redshifts),
                "node_redshifts": (store.node_redshifts, node_redshifts),
            }
            have = store.metadata
            for key in sorted(set(have) | set(metadata)):
                stored[f"metadata {key!r}"] = (have.get(key), metadata.get(key))
            for key, (a, b) in stored.items():
                if a is None or b is None or not _same(a, b):
                    store.close()
                    raise ValueError(
                        f"{path} was made with another {key}; create it again "
                        "(clobber=True) to change it"
                    )
            return store

        path.parent.mkdir(parents=True, exist_ok=True)
        file = h5py.File(path, "w")
        file.attrs["format"] = FORMAT
        file.attrs["quantity"] = quantity
        file.attrs["complete"] = False
        file.create_dataset(
            "lightcone",
            shape=shape,
            dtype=np.float32,
            chunks=tuple(chunks) if chunks is not None else default_chunks(shape),
            fillvalue=np.nan,
        )
        file["lightcone_distances"] = distances
        file["lightcone_distances"].attrs["unit"] = "Mpc"
        file["lightcone_redshifts"] = redshifts
        file["node_redshifts"] = node_redshifts
        file["global_mean"] = np.full(node_redshifts.size, np.nan)
        file["interval_done"] = np.zeros(node_redshifts.size - 1, dtype=bool)
        group = file.create_group("metadata")
        for key, value in metadata.items():
            group.attrs[key] = value
        file.flush()
        return cls(file)

    @property
    def path(self) -> Path:
        """The file of the store."""
        return Path(self._file.filename)

    @property
    def quantity(self) -> str:
        """The name of the quantity."""
        return str(self._file.attrs["quantity"])

    @property
    def lightcone(self) -> h5py.Dataset:
        """The lightcone dataset, with the slices on the last axis."""
        return self._file["lightcone"]

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the lightcone."""
        return tuple(self.lightcone.shape)

    @property
    def distances(self) -> un.Quantity[un.Mpc]:
        """The comoving distance of each slice."""
        return self._file["lightcone_distances"][()] * un.Mpc

    @property
    def redshifts(self) -> np.ndarray:
        """The redshift of each slice."""
        return self._file["lightcone_redshifts"][()]

    @property
    def node_redshifts(self) -> np.ndarray:
        """The redshifts of the boxes, in the order of the intervals."""
        return self._file["node_redshifts"][()]

    @property
    def global_mean(self) -> np.ndarray:
        """The mean of the quantity's box at each node redshift (NaN if not read)."""
        return self._file["global_mean"][()]

    @property
    def done(self) -> np.ndarray:
        """Whether the slices of each interval between node redshifts are filled."""
        return self._file["interval_done"][()]

    @property
    def complete(self) -> bool:
        """Whether every interval is filled."""
        return bool(self._file.attrs["complete"])

    @property
    def metadata(self) -> dict[str, Any]:
        """The description of how the lightcone is made."""
        return dict(self._file["metadata"].attrs)

    def write(self, index: slice | Sequence[int], values: np.ndarray) -> None:
        """Write slices of the lightcone.

        Parameters
        ----------
        index
            The indices of the slices on the last axis, as a slice or a sorted
            sequence.
        values
            The slices, with the slices on the last axis.
        """
        self.lightcone[..., index] = values

    def set_global_mean(self, node: int, value: float) -> None:
        """Record the mean of the quantity's box at one node redshift.

        Parameters
        ----------
        node
            The index of the node redshift.
        value
            The mean.
        """
        self._file["global_mean"][node] = value

    def mark_done(self, intervals: int | Sequence[int]) -> None:
        """Record that the slices of some intervals are filled, and write to disk.

        The store is complete once every interval is filled.

        Parameters
        ----------
        intervals
            The indices of the intervals: interval ``i`` lies between
            ``node_redshifts[i]`` and ``node_redshifts[i + 1]``.
        """
        done = self.done
        done[np.asarray(intervals, dtype=int)] = True
        self._file["interval_done"][...] = done
        self._file.attrs["complete"] = bool(done.all())
        self._file.flush()

    def close(self) -> None:
        """Close the file of the store."""
        self._file.close()

    def __enter__(self) -> Self:
        """Return the store."""
        return self

    def __exit__(self, *exc) -> None:
        """Close the file of the store."""
        self.close()
