"""Catalogues of objects, such as halos or galaxies, with positions and features.

A :class:`CoevalCatalog` holds the 3D position of each object and any number of
features (e.g. halo mass, stellar mass or star formation rate), each an astropy
Quantity with one value per object. It also holds the volume that contains the objects,
which number densities need, and their redshift, which is the same for all objects (as
in a coeval box of a simulation). Functions that work on catalogues take a
:class:`CoevalCatalog` and the name of the feature to use, e.g.
:func:`tuesday.core.summaries.bin_number_density`.

A catalogue converts to and from an astropy :class:`~astropy.table.QTable` with
:meth:`CoevalCatalog.to_table` and :meth:`CoevalCatalog.from_table`, so it can be
written to and read from the file formats of astropy tables that keep units and
metadata, such as HDF5 and ECSV, with :meth:`CoevalCatalog.write` and
:meth:`CoevalCatalog.read`.

A catalogue does not depend on any simulation code; see
:mod:`tuesday.simulators.py21cmfast.catalogs` for making one from a 21cmFAST halo
catalogue.
"""

from collections.abc import Mapping
from os import PathLike
from types import MappingProxyType

import attrs
import numpy as np
from astropy import units as un
from astropy.table import QTable, Table

from .units import without_littleh

#: Name of the column of the positions in the table of a catalogue.
POSITION_COLUMN = "position"

#: Path of the table in HDF5 files written by :meth:`CoevalCatalog.write`.
HDF5_PATH = "catalog"


def _is_hdf5(filename: str | PathLike, kwargs: dict) -> bool:
    """Whether a table is read from or written to HDF5."""
    if "format" in kwargs:
        return kwargs["format"] == "hdf5"
    return str(filename).endswith((".h5", ".hdf5", ".he5"))


def _read_only_features(
    features: Mapping[str, un.Quantity] | None,
) -> Mapping[str, un.Quantity]:
    """Store the features in a read-only mapping (the arrays are not copied)."""
    return MappingProxyType(dict(features or {}))


@attrs.define(frozen=True, eq=False, repr=False)
class CoevalCatalog:
    """A catalogue of objects at one redshift, with 3D positions and features.

    The catalogue is immutable: :meth:`with_features` and :meth:`select` return new
    catalogues. The arrays are not copied, so they share memory with the arrays
    given. For objects at different redshifts (e.g. on a lightcone), the redshift
    would be one of the features of a catalogue instead; this class does not cover
    that case.

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
        Redshift of the objects, e.g. that of the coeval box they come from.

    Raises
    ------
    ValueError
        If the positions are not a length of shape ``(n, 3)``, a feature does not
        have one value per object or is called :data:`POSITION_COLUMN`, or the volume
        is not a single finite, positive volume.
    TypeError
        If a feature is not a Quantity.
    """

    positions: un.Quantity = attrs.field()
    features: Mapping[str, un.Quantity] = attrs.field(
        factory=dict, converter=_read_only_features
    )
    volume: un.Quantity | None = attrs.field(default=None)
    redshift: float | None = attrs.field(default=None)

    @positions.validator
    def _check_positions(self, attribute: attrs.Attribute, value: un.Quantity):
        if (
            not isinstance(value, un.Quantity)
            or without_littleh(value.unit).physical_type != "length"
        ):
            raise ValueError("positions must be a Quantity with a unit of length.")
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f"positions must have shape (n, 3), got {value.shape}.")

    @features.validator
    def _check_features(
        self, attribute: attrs.Attribute, value: Mapping[str, un.Quantity]
    ):
        if POSITION_COLUMN in value:
            raise ValueError(
                f"No feature may be called '{POSITION_COLUMN}', the column of the "
                "positions."
            )
        n_objects = self.positions.shape[0]
        for name, feature in value.items():
            if not isinstance(feature, un.Quantity):
                raise TypeError(
                    f"Feature '{name}' must be an astropy Quantity; use "
                    "un.dimensionless_unscaled for a feature without a unit."
                )
            if feature.ndim == 0 or feature.shape[0] != n_objects:
                raise ValueError(
                    f"Feature '{name}' has shape {feature.shape}, but the catalogue "
                    f"has {n_objects} objects."
                )

    @volume.validator
    def _check_volume(self, attribute: attrs.Attribute, value: un.Quantity | None):
        if value is None:
            return
        if (
            not isinstance(value, un.Quantity)
            or without_littleh(value.unit).physical_type != "volume"
        ):
            raise ValueError(
                f"volume must be a Quantity with a unit of volume, got {value!r}."
            )
        if value.ndim != 0 or not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"volume must be a single finite, positive value, got {value}."
            )

    def __len__(self) -> int:
        """Return the number of objects."""
        return self.positions.shape[0]

    def __getitem__(self, name: str) -> un.Quantity:
        """Return one feature of the objects."""
        try:
            return self.features[name]
        except KeyError:
            raise KeyError(
                f"The catalogue has no feature '{name}'; its features are "
                f"{list(self.features)}."
            ) from None

    def __repr__(self) -> str:
        """Return a short description of the catalogue."""
        return (
            f"<CoevalCatalog of {len(self)} objects, features "
            f"{list(self.features)}, volume {self.volume}, redshift {self.redshift}>"
        )

    def with_features(self, **features: un.Quantity) -> "CoevalCatalog":
        """Return a catalogue with features added or replaced.

        Parameters
        ----------
        **features : un.Quantity
            The features to add, by name. A feature with the name of an existing one
            replaces it.

        Returns
        -------
        CoevalCatalog
            A new catalogue with the same positions, volume and redshift.
        """
        return attrs.evolve(self, features={**self.features, **features})

    def select(
        self, mask: np.ndarray, volume: un.Quantity | None = None
    ) -> "CoevalCatalog":
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
        CoevalCatalog
            A new catalogue with the selected objects and the same redshift.
        """
        return attrs.evolve(
            self,
            positions=self.positions[mask],
            features={name: value[mask] for name, value in self.features.items()},
            volume=self.volume if volume is None else volume,
        )

    def to_table(self) -> QTable:
        """Return the catalogue as an astropy QTable.

        Returns
        -------
        QTable
            A table with the positions in the column :data:`POSITION_COLUMN`, one
            column per feature, and ``meta["volume"]`` and ``meta["redshift"]``. Its
            columns share memory with the catalogue.
        """
        return QTable(
            {POSITION_COLUMN: self.positions, **self.features},
            meta={"volume": self.volume, "redshift": self.redshift},
            copy=False,
        )

    @classmethod
    def from_table(cls, table: Table) -> "CoevalCatalog":
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
        CoevalCatalog
            The catalogue. Its columns share memory with the table.

        Raises
        ------
        ValueError
            If the table has no column :data:`POSITION_COLUMN`, or as for
            :class:`CoevalCatalog`.
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
    def read(cls, filename: str | PathLike, **kwargs) -> "CoevalCatalog":
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
        CoevalCatalog
            The catalogue.
        """
        if _is_hdf5(filename, kwargs):
            kwargs.setdefault("path", HDF5_PATH)
        return cls.from_table(QTable.read(filename, **kwargs))
