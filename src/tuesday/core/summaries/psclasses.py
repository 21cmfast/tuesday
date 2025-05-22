from dataclasses import dataclass

import astropy.units as un
import numpy as np


@dataclass(frozen=True)
class SphericalPS:
    """Class to hold the 1D power spectrum data."""

    ps: un.Quantity
    k: un.Quantity
    redshift: np.ndarray | None = None
    Nmodes: np.ndarray | None = None
    var: un.Quantity | None = None
    delta: bool = False


@dataclass(frozen=True)
class CylindricalPS:
    """Class to hold the 2D power spectrum data."""

    ps: un.Quantity
    kperp: un.Quantity
    kpar: un.Quantity
    redshift: np.ndarray | None = None
    Nmodes: np.ndarray | None = None
    var: un.Quantity | None = None
    delta: bool = False
