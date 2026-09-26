"""Validating the units."""

import astropy.units as un
from astropy.cosmology.units import littleh


def littleh_power(unit: un.UnitBase) -> float:
    """Get the power to which little-h appears in a unit.

    Little-h is an irreducible unit, so it survives decomposition, and its power can be
    read off directly (e.g. ``Mpc/littleh`` gives -1, and ``littleh/Mpc`` gives 1).

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The unit to inspect.

    Returns
    -------
    float
        The power of little-h in the unit (zero if it doesn't appear).
    """
    decomposed = un.Unit(unit).decompose()
    return dict(zip(decomposed.bases, decomposed.powers, strict=True)).get(littleh, 0)


def without_littleh(unit: un.UnitBase) -> un.UnitBase:
    """Remove any factors of little-h from a unit.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The unit from which to remove little-h.

    Returns
    -------
    astropy.units.UnitBase
        The unit, with all factors of little-h divided out.
    """
    return un.Unit(unit) / littleh ** littleh_power(unit)


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
