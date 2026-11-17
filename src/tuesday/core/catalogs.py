"""Catalogues of objects, such as halos or galaxies, with positions and features.

A :class:`Catalog` holds the 3D position of each object and any number of features
(e.g. halo mass, stellar mass or star formation rate), each an astropy Quantity with
one value per object. It also holds the volume that contains the objects, which number
densities need, and their redshift. Functions that work on catalogues take a
:class:`Catalog` and the name of the feature to use, e.g.
:func:`tuesday.core.summaries.bin_number_density`.

A catalogue does not depend on any simulation code; see
:mod:`tuesday.simulators.py21cmfast.catalogs` for making one from a 21cmFAST halo
catalogue.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np
from astropy import units as un

from .units import without_littleh


@dataclass(frozen=True, eq=False)
class Catalog:
    """A catalogue of objects with 3D positions and features.

    The catalogue is immutable: :meth:`with_features` and :meth:`select` return new
    catalogues. The arrays are not copied, so they share memory with the arrays
    given.

    Attributes
    ----------
    positions : un.Quantity
        Position of each object, a length of shape ``(n, 3)`` (e.g. comoving Mpc).
    features : Mapping[str, un.Quantity]
        Features of the objects by name, each a Quantity whose first axis has length
        ``n``. Features without a unit are dimensionless Quantities (e.g.
        ``un.dimensionless_unscaled``), or magnitudes in ``un.mag``.
    volume : un.Quantity, optional
        The volume that contains the objects, e.g. that of the simulation box: a single
        finite, positive value. Needed for number densities.
    redshift : float, optional
        Redshift of the objects, if they are all at one redshift (e.g. a coeval box).
    """

    positions: un.Quantity
    features: Mapping[str, un.Quantity] = field(default_factory=dict)
    volume: un.Quantity | None = None
    redshift: float | None = None

    def __post_init__(self):
        """Validate the positions, features and volume."""
        positions = self.positions
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
        for name, value in self.features.items():
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
        volume = self.volume
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
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

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
        return replace(self, features={**self.features, **features})

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
        return replace(
            self,
            positions=self.positions[mask],
            features={name: value[mask] for name, value in self.features.items()},
            volume=self.volume if volume is None else volume,
        )
