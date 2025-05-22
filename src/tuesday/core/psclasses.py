"""Classes to hold the power spectrum data."""

from dataclasses import dataclass

import astropy.units as un
import numpy as np
from astropy.cosmology.units import littleh


@dataclass(frozen=True)
class SphericalPS:
    r"""Class to hold the 1D power spectrum data."""

    ps: un.Quantity
    k: un.Quantity
    redshift: np.ndarray | None = None
    n_modes: np.ndarray | None = None
    variance: un.Quantity | None = None
    is_deltasq: bool = False

    def __post_init__(self):
        r"""Validate 1D PS array shapes."""
        if self.ps.ndim != 1:
            raise ValueError("The ps array must be 1D for a SphericalPS.")
        if self.n_modes is not None and self.n_modes.shape != self.ps.shape:
            raise ValueError("n_modes must have same shape as ps.")
        if (
            self.k.shape[0] != self.ps.shape[0]
            and self.k.shape[0] != self.ps.shape[0] + 1
        ):
            raise ValueError(
                "k must either be the same shape as the k-"
                "axis of the ps or larger by one if k is the bin edges."
            )
        if (
            self.k.unit.physical_type != un.get_physical_type("wavenumber")
            and self.k.unit.physical_type
            != un.get_physical_type("wavenumber") * littleh
        ):
            raise ValueError(
                f"Unit of k must be a wavenumber, got {self.k.unit.physical_type}."
            )
        if self.is_deltasq:
            if (
                self.ps.unit.physical_type != un.get_physical_type("temperature") ** 2
                and self.ps.unit.physical_type != "dimensionless"
            ):
                raise ValueError(
                    "Expected unit of delta PS to be temperature squared or"
                    f" dimensionless, but got {self.ps.unit.physical_type}."
                )
        else:
            if "littleh" in self.ps.unit.to_string():
                temp2xvol = (
                    un.get_physical_type("temperature") ** 2
                    * un.get_physical_type("volume")
                    / littleh**3
                )
                vol = un.get_physical_type("volume") / littleh**3

            else:
                temp2xvol = un.get_physical_type(
                    "temperature"
                ) ** 2 * un.get_physical_type("volume")
                vol = un.get_physical_type("volume")
            if (
                self.ps.unit.physical_type != temp2xvol
                and self.ps.unit.physical_type != vol
                and self.ps.unit.physical_type != "dimensionless"
            ):
                raise ValueError(
                    "Expected unit of PS to be temperature squared times volume, "
                    f"or volume but got {self.ps.unit.physical_type}."
                )


@dataclass(frozen=True)
class CylindricalPS:
    r"""Class to hold the 2D power spectrum data."""

    ps: un.Quantity
    kperp: un.Quantity
    kpar: un.Quantity
    redshift: np.ndarray | None = None
    n_modes: np.ndarray | None = None
    variance: un.Quantity | None = None
    is_deltasq: bool = False

    def __post_init__(self):
        r"""Validate 2D PS array shapes."""
        if self.ps.ndim != 2:
            raise ValueError("The ps array must be 2D for a CylindricalPS.")
        if self.n_modes is not None and (
            self.n_modes.shape != self.ps.shape
            and self.n_modes.shape[0] != self.ps.shape[0]
        ):
            raise ValueError(
                "n_modes must have same shape as ps. Instead got"
                f"{self.n_modes.shape} and {self.ps.shape}."
            )
        if (
            self.kperp.shape[0] != self.ps.shape[0]
            and self.kperp.shape[0] != self.ps.shape[0] + 1
        ):
            raise ValueError(
                "kperp must either be the same shape as the kperp "
                "axis of the ps or larger by one if kperp is the bin edges."
            )
        if (
            self.kpar.shape[0] != self.ps.shape[1]
            and self.kpar.shape[0] != self.ps.shape[1] + 1
        ):
            raise ValueError(
                "kpar must either be the same shape as the kpar "
                "axis of the ps or larger by one if kpar is the "
                f"bin edges. Instead got {self.kpar.shape[0]} and {self.ps.shape[1]}"
            )
        if (
            self.kperp.unit.physical_type != un.get_physical_type("wavenumber")
            and self.kperp.unit.physical_type
            != un.get_physical_type("wavenumber") * littleh
        ):
            raise ValueError(
                "Unit of kperp must be a wavenumber, "
                "got {self.kperp.unit.physical_type}."
            )
        if (
            self.kpar.unit.physical_type != un.get_physical_type("wavenumber")
            and self.kpar.unit.physical_type
            != un.get_physical_type("wavenumber") * littleh
        ):
            raise ValueError(
                "Unit of kpar must be a wavenumber, got {self.kpar.unit.physical_type}."
            )
        if self.is_deltasq:
            if (
                self.ps.unit.physical_type != un.get_physical_type("temperature") ** 2
                and self.ps.unit.physical_type != "dimensionless"
            ):
                raise ValueError(
                    "Expected unit of delta PS to be temperature squared or"
                    f" dimensionless, but got {self.ps.unit.physical_type}."
                )
        else:
            if "littleh" in self.ps.unit.to_string():
                temp2xvol = (
                    un.get_physical_type("temperature") ** 2
                    * un.get_physical_type("volume")
                    / littleh**3
                )
                vol = un.get_physical_type("volume") / littleh**3

            else:
                temp2xvol = un.get_physical_type(
                    "temperature"
                ) ** 2 * un.get_physical_type("volume")
                vol = un.get_physical_type("volume")
            if (
                self.ps.unit.physical_type != temp2xvol
                and self.ps.unit.physical_type != vol
                and self.ps.unit.physical_type != "dimensionless"
            ):
                raise ValueError(
                    "Expected unit of PS to be temperature squared times volume, "
                    f"or volume but got {self.ps.unit.physical_type}."
                )
