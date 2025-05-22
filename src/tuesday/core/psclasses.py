"""Classes to hold the power spectrum data."""

from dataclasses import dataclass

import astropy.units as un
import numpy as np


@dataclass(frozen=True)
class SphericalPS:
    """Class to hold the 1D power spectrum data."""

    ps: un.Quantity
    k: un.Quantity
    redshift: np.ndarray | None = None
    n_modes: np.ndarray | None = None
    variance: un.Quantity | None = None
    is_deltasq: bool = False

    def __post_init__(self):
        if self.ps.ndim != 1:
            raise ValueError("The ps array must be 1D for a SphericalPS.")
        if self.n_modes is not None and self.n_modes.shape != self.ps.shape:
            raise ValueError("n_modes must have same shape as ps.")
        if self.k.shape[0] != self.ps.shape[0] and self.k.shape[0] != self.ps.shape[0]+1:
            raise ValueError("k must either be the same shape as the k-"\
                             "axis of the ps or larger by one if k is the bin edges.")


@dataclass(frozen=True)
class CylindricalPS:
    """Class to hold the 2D power spectrum data."""

    ps: un.Quantity
    kperp: un.Quantity
    kpar: un.Quantity
    redshift: np.ndarray | None = None
    n_modes: np.ndarray | None = None
    variance: un.Quantity | None = None
    is_deltasq: bool = False

    def __post_init__(self):
        if self.ps.ndim != 2:
            raise ValueError("The ps array must be 2D for a CylindricalPS.")
        if self.n_modes is not None and (self.n_modes.shape != self.ps.shape and self.n_modes.shape[0] != self.ps.shape[0]):
            raise ValueError("n_modes must have same shape as ps. Instead got"\
                             f"{self.n_modes.shape} and {self.ps.shape}.")
        if self.kperp.shape[0] != self.ps.shape[0] and self.kperp.shape[0] != self.ps.shape[0]+1:
            raise ValueError("kperp must either be the same shape as the kperp "\
                             "axis of the ps or larger by one if kperp is the bin edges.")
        if self.kpar.shape[0] != self.ps.shape[1] and self.kpar.shape[0] != self.ps.shape[1]+1:
            raise ValueError("kpar must either be the same shape as the kpar "\
                             "axis of the ps or larger by one if kpar is "\
                             f"the bin edges. Instead got {self.kpar.shape[0]} and {self.ps.shape[1]}")

