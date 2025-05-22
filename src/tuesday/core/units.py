"""Validating the units."""

import astropy.units as un
from astropy.cosmology.units import littleh

from .psclasses import CylindricalPS, SphericalPS


def validate_ps(power_spectrum: SphericalPS | CylindricalPS) -> None:
    """
    Validate the unit of a given quantity.

    Parameters
    ----------
    qt : un.Quantity
        The quantity to validate.
    unit : str
        The expected physical type string.

    Raises
    ------
    ValueError
        If the unit of the quantity does not match the expected unit.
    """
    if (
        isinstance(power_spectrum, SphericalPS)
        and power_spectrum.k.unit.physical_type != "wavenumber"
    ):
        raise ValueError(
            f"Expected unit wavenumber, but got {power_spectrum.k.unit.physical_type}."
        )
    if isinstance(power_spectrum, CylindricalPS):
        if power_spectrum.kperp.unit.physical_type != "wavenumber":
            raise ValueError(
                "Expected unit of kperp to be wavenumber, but got"
                f" {power_spectrum.kperp.unit.physical_type}."
            )
        if power_spectrum.kpar.unit.physical_type != "wavenumber":
            raise ValueError(
                "Expected unit of kperp to be wavenumber, but got"
                f" {power_spectrum.kpar.unit.physical_type}."
            )
    if power_spectrum.is_deltasq:
        if (
            power_spectrum.ps.unit.physical_type
            != un.get_physical_type("temperature") ** 2
            and power_spectrum.ps.unit.physical_type != "dimensionless"
        ):
            raise ValueError(
                "Expected unit of delta PS to be temperature squared or"
                f" dimensionless, but got {power_spectrum.ps.unit.physical_type}."
            )
    else:
        if "littleh" in power_spectrum.ps.unit.to_string():
            temp2xvol = (
                un.get_physical_type("temperature") ** 2
                * un.get_physical_type("volume")
                / littleh**3
            )
            vol = un.get_physical_type("volume") / littleh**3

        else:
            temp2xvol = un.get_physical_type("temperature") ** 2 * un.get_physical_type(
                "volume"
            )
            vol = un.get_physical_type("volume")
        if (
            power_spectrum.ps.unit.physical_type != temp2xvol
            and power_spectrum.ps.unit.physical_type != vol
            and power_spectrum.ps.unit.physical_type != "dimensionless"
        ):
            raise ValueError(
                "Expected unit of PS to be temperature squared times volume, "
                f"or volume but got {power_spectrum.ps.unit.physical_type}."
            )


def validate(qt: un.Quantity, unit: str) -> None:
    """Validate the unit of a given quantity.

    Parameters
    ----------
    qt : un.Quantity
        The quantity to validate.
    unit : str
        The expected physical type string.

    Raises
    ------
    ValueError
        If the unit of the quantity does not match the expected unit.
    """
    if qt.unit.physical_type != unit:
        if unit == "temperature" and qt.unit.physical_type == "dimensionless":
            pass
        else:
            raise ValueError(f"Expected unit {unit}, but got {qt.unit.physical_type}.")
