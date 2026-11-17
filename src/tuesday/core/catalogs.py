"""Catalogues of objects, such as halos or galaxies, with positions and features.

A :class:`Catalog` holds the 3D position of each object and any number of features
(e.g. halo mass, stellar mass or star formation rate), each an astropy Quantity with
one value per object. It also holds the volume that contains the objects, which number
densities need, and their redshift. Functions that work on catalogues take a
:class:`Catalog` and the name of the feature to use, e.g.
:func:`tuesday.core.summaries.bin_number_density`.

The data are kept in an astropy :class:`~astropy.table.QTable`, which
:meth:`Catalog.to_table` returns and :meth:`Catalog.from_table` reads, so a catalogue
can be written to and read from the file formats of astropy tables that keep units and
metadata, such as HDF5 and ECSV, with :meth:`Catalog.write` and :meth:`Catalog.read`.

A catalogue does not depend on any simulation code; see
:mod:`tuesday.simulators.py21cmfast.catalogs` for making one from a 21cmFAST halo
catalogue.
"""

from collections.abc import Mapping
from os import PathLike
from types import MappingProxyType

import numpy as np
from astropy import units as un
from astropy.table import QTable, Table

from .units import without_littleh

#: Name of the column of the positions in the table of a catalogue.
POSITION_COLUMN = "position"

#: Path of the table in HDF5 files written by :meth:`Catalog.write`.
HDF5_PATH = "catalog"


def _is_hdf5(filename: str | PathLike, kwargs: dict) -> bool:
    """Whether a table is read from or written to HDF5."""
    if "format" in kwargs:
        return kwargs["format"] == "hdf5"
    return str(filename).endswith((".h5", ".hdf5", ".he5"))


class Catalog:
    """A catalogue of objects with 3D positions and features.

    The data are stored in an astropy :class:`~astropy.table.QTable`, with the
    positions in the column :data:`POSITION_COLUMN`, one column per feature, and the
    volume and redshift in its ``meta``. The catalogue is immutable:
    :meth:`with_features` and :meth:`select` return new catalogues. The arrays are not
    copied, so they share memory with the arrays given.

    Parameters
    ----------
    positions : un.Quantity
        Position of each object, a length of shape ``(n, 3)`` (e.g. comoving Mpc).
    features : Mapping[str, un.Quantity], optional
        Features of the objects by name, each a Quantity whose first axis has length
        ``n``. Features without a unit are dimensionless Quantities (e.g.
        ``un.dimensionless_unscaled``), or magnitudes in ``un.mag``. No feature may be
        called :data:`POSITION_COLUMN`.
    volume : un.Quantity, optional
        The volume that contains the objects, e.g. that of the simulation box: a single
        finite, positive value. Needed for number densities.
    redshift : float, optional
        Redshift of the objects, if they are all at one redshift (e.g. a coeval box).

    Raises
    ------
    ValueError
        If the positions are not a length of shape ``(n, 3)``, a feature does not
        have one value per object or is called :data:`POSITION_COLUMN`, or the volume
        is not a single finite, positive volume.
    TypeError
        If a feature is not a Quantity.
    """

    __slots__ = ("_table",)

    def __init__(
        self,
        positions: un.Quantity,
        features: Mapping[str, un.Quantity] | None = None,
        volume: un.Quantity | None = None,
        redshift: float | None = None,
    ):
        features = dict(features or {})
        if (
            not isinstance(positions, un.Quantity)
            or without_littleh(positions.unit).physical_type != "length"
        ):
            raise ValueError("positions must be a Quantity with a unit of length.")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (n, 3), got {positions.shape}."
            )
        n_objects = positions.shape[0]
        if POSITION_COLUMN in features:
            raise ValueError(
                f"No feature may be called '{POSITION_COLUMN}', the column of the "
                "positions."
            )
        for name, value in features.items():
            if not isinstance(value, un.Quantity):
                raise TypeError(
                    f"Feature '{name}' must be an astropy Quantity; use "
                    "un.dimensionless_unscaled for a feature without a unit."
                )
            if value.ndim == 0 or value.shape[0] != n_objects:
                raise ValueError(
                    f"Feature '{name}' has shape {value.shape}, but the catalogue "
                    f"has {n_objects} objects."
                )
        if volume is not None:
            if not isinstance(volume, un.Quantity):
                raise ValueError(
                    f"volume must be a Quantity with a unit of volume, got {volume!r}."
                )
            if without_littleh(volume.unit).physical_type != "volume":
                raise ValueError(
                    f"volume must be a volume, got {volume.unit.physical_type}."
                )
            if volume.ndim != 0 or not np.isfinite(volume) or volume <= 0:
                raise ValueError(
                    f"volume must be a single finite, positive value, got {volume}."
                )
        self._table = QTable(
            {POSITION_COLUMN: positions, **features},
            meta={"volume": volume, "redshift": redshift},
            copy=False,
        )

    @property
    def positions(self) -> un.Quantity:
        """Position of each object, of shape ``(n, 3)``."""
        return self._table[POSITION_COLUMN]

    @property
    def features(self) -> Mapping[str, un.Quantity]:
        """Features of the objects by name (read-only)."""
        return MappingProxyType(
            {
                name: self._table[name]
                for name in self._table.colnames
                if name != POSITION_COLUMN
            }
        )

    @property
    def volume(self) -> un.Quantity | None:
        """The volume that contains the objects."""
        return self._table.meta["volume"]

    @property
    def redshift(self) -> float | None:
        """Redshift of the objects."""
        return self._table.meta["redshift"]

    def __len__(self) -> int:
        """Return the number of objects."""
        return len(self._table)

    def __getitem__(self, name: str) -> un.Quantity:
        """Return one feature of the objects."""
        if name == POSITION_COLUMN or name not in self._table.colnames:
            raise KeyError(
                f"The catalogue has no feature '{name}'; its features are "
                f"{list(self.features)}."
            )
        return self._table[name]

    def __repr__(self) -> str:
        """Return a short description of the catalogue."""
        return (
            f"<Catalog of {len(self)} objects, features {list(self.features)}, "
            f"volume {self.volume}, redshift {self.redshift}>"
        )

    def with_features(self, **features: un.Quantity) -> "Catalog":
        """Return a catalogue with features added or replaced.

        Parameters
        ----------
        **features : un.Quantity
            The features to add, by name. A feature with the name of an existing one
            replaces it.

        Returns
        -------
        Catalog
            A new catalogue with the same positions, volume and redshift.
        """
        return Catalog(
            self.positions,
            {**self.features, **features},
            volume=self.volume,
            redshift=self.redshift,
        )

    def select(self, mask: np.ndarray, volume: un.Quantity | None = None) -> "Catalog":
        """Return a catalogue of some of the objects.

        Parameters
        ----------
        mask : np.ndarray
            Boolean array with one value per object, or the indices of the objects to
            keep.
        volume : un.Quantity, optional
            The volume that contains the selected objects. By default the volume of
            this catalogue, which is right for a selection by a feature (e.g. all
            halos above a mass). For a selection in space (e.g. a slab of the box),
            pass the volume of the selected region.

        Returns
        -------
        Catalog
            A new catalogue with the selected objects and the same redshift.
        """
        return Catalog(
            self.positions[mask],
            {name: value[mask] for name, value in self.features.items()},
            volume=self.volume if volume is None else volume,
            redshift=self.redshift,
        )

    def to_table(self) -> QTable:
        """Return the catalogue as an astropy QTable.

        Returns
        -------
        QTable
            A table with the positions in the column :data:`POSITION_COLUMN`, one
            column per feature, and ``meta["volume"]`` and ``meta["redshift"]``. It is
            a new table, but its columns share memory with the catalogue.
        """
        return self._table.copy(copy_data=False)

    @classmethod
    def from_table(cls, table: Table) -> "Catalog":
        """Make a catalogue from an astropy table.

        Parameters
        ----------
        table : Table or QTable
            A table with a column :data:`POSITION_COLUMN` of shape ``(n, 3)`` with a
            unit of length, and a column with a unit for each feature. The volume and
            redshift are read from ``meta["volume"]`` and ``meta["redshift"]``, if
            there.

        Returns
        -------
        Catalog
            The catalogue. Its columns share memory with the table.

        Raises
        ------
        ValueError
            If the table has no column :data:`POSITION_COLUMN`, or as for
            :class:`Catalog`.
        TypeError
            If a column has no unit.
        """
        table = QTable(table, copy=False)
        if POSITION_COLUMN not in table.colnames:
            raise ValueError(
                f"The table has no column '{POSITION_COLUMN}' with the positions."
            )
        return cls(
            table[POSITION_COLUMN],
            {name: table[name] for name in table.colnames if name != POSITION_COLUMN},
            volume=table.meta.get("volume"),
            redshift=table.meta.get("redshift"),
        )

    def write(self, filename: str | PathLike, **kwargs) -> None:
        """Write the catalogue to a file.

        The table of :meth:`to_table` is written with
        :meth:`astropy.table.QTable.write`. Use a format that keeps units and
        metadata, such as HDF5 (``.h5``, ``.hdf5``) or ECSV (``.ecsv``). For HDF5 the
        table is written at the path :data:`HDF5_PATH` and its metadata are
        serialized, unless ``path`` or ``serialize_meta`` are given.

        Parameters
        ----------
        filename : str or PathLike
            The file to write.
        **kwargs
            Passed on to :meth:`astropy.table.QTable.write`, e.g. ``overwrite=True``.
        """
        if _is_hdf5(filename, kwargs):
            kwargs.setdefault("path", HDF5_PATH)
            kwargs.setdefault("serialize_meta", True)
        self.to_table().write(filename, **kwargs)

    @classmethod
    def read(cls, filename: str | PathLike, **kwargs) -> "Catalog":
        """Read a catalogue written by :meth:`write`.

        Parameters
        ----------
        filename : str or PathLike
            The file to read.
        **kwargs
            Passed on to :meth:`astropy.table.QTable.read`. For HDF5 the path is
            :data:`HDF5_PATH` unless ``path`` is given.

        Returns
        -------
        Catalog
            The catalogue.
        """
        if _is_hdf5(filename, kwargs):
            kwargs.setdefault("path", HDF5_PATH)
        return cls.from_table(QTable.read(filename, **kwargs))
