"""Validating the units."""

import astropy.units as un


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
