"""Classes to hold power spectrum data."""

from dataclasses import dataclass

import astropy.units as un
import numpy as np

from ..units import littleh_power, without_littleh


def _validate_ps_units(
    ps: un.Quantity, is_deltasq: bool, **wavenumbers: un.Quantity
) -> None:
    """Validate the units of a power spectrum and its wavenumbers.

    Little-h is allowed in the units, but must be consistent: all wavenumbers must
    carry the same power of little-h, and (unless the power spectrum is dimensionless
    or delta-squared) the power spectrum must carry the corresponding inverse-cubed
    power, e.g. ``k`` in ``littleh/Mpc`` requires ``ps`` in ``mK^2 Mpc^3/littleh^3``.
    """
    for name, k in wavenumbers.items():
        if without_littleh(k.unit).physical_type != "wavenumber":
            raise ValueError(
                f"Unit of {name} must be a wavenumber, got {k.unit.physical_type}."
            )

    k_hpowers = {name: littleh_power(k.unit) for name, k in wavenumbers.items()}
    if len(set(k_hpowers.values())) > 1:
        raise ValueError(
            f"All wavenumbers must have the same power of littleh, got {k_hpowers}."
        )

    ps_type = without_littleh(ps.unit).physical_type
    if is_deltasq:
        if ps_type not in [un.get_physical_type("temperature") ** 2, "dimensionless"]:
            raise ValueError(
                "Expected unit of delta PS to be temperature squared or"
                f" dimensionless, but got {ps.unit.physical_type}."
            )
        expected_hpower = 0
    else:
        vol = un.get_physical_type("volume")
        temp2xvol = un.get_physical_type("temperature") ** 2 * vol
        if ps_type not in [temp2xvol, vol, "dimensionless"]:
            raise ValueError(
                "Expected unit of PS to be temperature squared times volume, "
                f"or volume but got {ps.unit.physical_type}."
            )
        expected_hpower = (
            0 if ps_type == "dimensionless" else -3 * next(iter(k_hpowers.values()))
        )

    if littleh_power(ps.unit) != expected_hpower:
        raise ValueError(
            f"The PS unit ({ps.unit}) has inconsistent littleh with the wavenumbers "
            f"({k_hpowers}): expected littleh**{expected_hpower}."
        )


@dataclass(frozen=True)
class SphericalPS:
    r"""Class to hold the 1D power spectrum data.

    Attributes
    ----------
    ps : un.Quantity
        Power spectrum data, whose units depend on whether the power spectrum is
        is 'dimensionless' (i.e. the variance per log-k, commonly called delta-squared),
        and the units of `k`. A 1D array of shape (n_k,).
    k : un.Quantity
        Wavenumber data, in inverse distance units. These may be bin 'centers' (under
        some definition for 'center') or bin edges, in which case the length of the
        array is one larger than `ps`.
    redshift : float, optional
        The redshift at which the power spectrum was measured.
    n_modes : np.ndarray, optional
        The number of k-modes averaged into each ps value. Must be the same
        shape as `ps`. This may be used to calculate the variance of the power spectrum.
    variance : un.Quantity, optional
        The variance of the power spectrum, in the same units as `ps` (but squared).
        Covariances are not supported yet.
    is_deltasq : bool, optional
        Whether the power spectrum is in delta-squared units. If True, the units of `ps`
        are expected to be temperature squared or dimensionless. If False, the units of
        `ps` are expected to be temperature squared by volume.
    """

    ps: un.Quantity
    k: un.Quantity
    redshift: float | None = None
    n_modes: np.ndarray | None = None
    variance: un.Quantity | None = None
    is_deltasq: bool = False

    def __post_init__(self):
        r"""Validate 1D PS array shapes."""
        if self.ps.ndim != 1:
            raise ValueError("The ps array must be 1D for a SphericalPS.")
        if self.n_modes is not None and self.n_modes.shape != self.ps.shape:
            raise ValueError("n_modes must have same shape as ps.")
        if self.k.shape[0] not in [self.ps.shape[0], self.ps.shape[0] + 1]:
            raise ValueError(
                "k must either be the same shape as the k-"
                "axis of the ps or larger by one if k is the bin edges."
            )
        _validate_ps_units(self.ps, self.is_deltasq, k=self.k)

        if self.variance is not None and self.variance.shape != self.ps.shape:
            raise ValueError(
                "Variance must have same shape as ps. Instead got"
                f"{self.variance.shape} and {self.ps.shape}."
            )

        if self.variance is not None and self.variance.unit != self.ps.unit**2:
            raise ValueError(
                "Variance must have the same unit as ps squared. Instead got"
                f"{self.variance.unit} and {self.ps.unit**2}."
            )

    @property
    def nk(self) -> int:
        """Return the number of k bins."""
        return self.ps.shape[0]

    @property
    def kcenters(self) -> un.Quantity:
        """Return the centers of the k bins."""
        if len(self.k) == self.nk:
            return self.k
        if np.allclose(np.diff(self.k), np.diff(self.k)[0]):
            return (self.k[:-1] + self.k[1:]) / 2.0
        return (
            np.exp((np.log(self.k.value[1:]) + np.log(self.k.value[:-1])) / 2)
            * self.k.unit
        )


@dataclass(frozen=True)
class CylindricalPS:
    r"""Class to hold the 2D power spectrum data.

    Attributes
    ----------
    ps : un.Quantity
        Power spectrum data, whose units depend on whether the power spectrum is
        is 'dimensionless' (i.e. the variance per log-k, commonly called delta-squared),
        and the units of `k`. A 2D array of shape (n_kperp, n_kpar).
    kperp : un.Quantity
        Perpendicular wavenumbers, in inverse distance units.
        These may be bin 'centers' (under some definition for 'center') or bin edges,
        in which case the length of the array is one larger than the first dimension
        of `ps`.
    kpar : un.Quantity
        Parallel wavenumbers, in inverse distance units.
        These may be bin 'centers' (under some definition for 'center') or bin edges,
        in which case the length of the array is one larger than the second dimension
        of `ps`.
    redshift : float, optional
        The redshift at which the power spectrum was measured.
    n_modes : np.ndarray, optional
        The number of k-modes averaged into each ps value. Must be the same
        shape as `ps`. This may be used to calculate the variance of the power spectrum.
    variance : un.Quantity, optional
        The variance of the power spectrum, in the same units as `ps` (but squared).
        Covariances are not supported yet.
    is_deltasq : bool, optional
        Whether the power spectrum is in delta-squared units. If True, the units of `ps`
        are expected to be temperature squared or dimensionless. If False, the units of
        `ps` are expected to be temperature squared by volume.
    """

    ps: un.Quantity
    kperp: un.Quantity
    kpar: un.Quantity
    redshift: float | None = None
    n_modes: np.ndarray | None = None
    variance: un.Quantity | None = None
    is_deltasq: bool = False

    def __post_init__(self):
        r"""Validate 2D PS array shapes."""
        if self.ps.ndim != 2:
            raise ValueError("The ps array must be 2D for a CylindricalPS.")
        if self.n_modes is not None and self.n_modes.shape not in (
            self.ps.shape,
            self.ps.shape[:1],
        ):
            raise ValueError(
                "n_modes must have same shape as ps. Instead got"
                f"{self.n_modes.shape} and {self.ps.shape}."
            )
        if self.kperp.shape[0] not in [self.ps.shape[0], self.ps.shape[0] + 1]:
            raise ValueError(
                "kperp must either be the same shape as the kperp "
                "axis of the ps or larger by one if kperp is the bin edges."
            )
        if self.kpar.shape[0] not in [self.ps.shape[1], self.ps.shape[1] + 1]:
            raise ValueError(
                "kpar must either be the same shape as the kpar "
                "axis of the ps or larger by one if kpar is the "
                f"bin edges. Instead got {self.kpar.shape[0]} and {self.ps.shape[1]}"
            )
        _validate_ps_units(self.ps, self.is_deltasq, kperp=self.kperp, kpar=self.kpar)

        if self.variance is not None and self.variance.shape != self.ps.shape:
            raise ValueError(
                "Variance must have same shape as ps. Instead got"
                f"{self.variance.shape} and {self.ps.shape}."
            )

        if self.variance is not None and self.variance.unit != self.ps.unit**2:
            raise ValueError(
                "Variance must have the same unit as ps squared. Instead got"
                f"{self.variance.unit} and {self.ps.unit**2}."
            )

    @property
    def nkperp(self) -> int:
        """Return the number of kperp bins."""
        return self.ps.shape[0]

    @property
    def nkpar(self) -> int:
        """Return the number of kpar bins."""
        return self.ps.shape[1]

    @property
    def kperp_centers(self) -> un.Quantity:
        """Return the centers of the kperp bins."""
        return (
            self.kperp
            if len(self.kperp) == self.nkperp
            else (self.kperp[:-1] + self.kperp[1:]) / 2.0
        )

    @property
    def kpar_centers(self) -> un.Quantity:
        """Return the centers of the kpar bins."""
        return (
            self.kpar
            if len(self.kpar) == self.nkpar
            else (self.kpar[:-1] + self.kpar[1:]) / 2.0
        )
