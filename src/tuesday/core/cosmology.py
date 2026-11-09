"""Cosmological helper functions."""

from collections.abc import Hashable
from typing import Any

import numpy as np
from astropy import units as un
from astropy.cosmology import FLRW
from scipy.interpolate import CubicSpline


def _cosmology_key(cosmo: FLRW) -> tuple:
    """Build a hashable key from the class and all parameters of a cosmology.

    Astropy cosmologies are not hashable, so they cannot be memoized directly. Two
    cosmologies with the same class and parameters have the same key.
    """

    def as_hashable(value: Any) -> Hashable:
        if isinstance(value, un.Quantity):
            return (tuple(np.atleast_1d(value.value).tolist()), value.unit.to_string())
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        return value

    mapping = cosmo.to_format("mapping")
    return tuple(
        (name, as_hashable(value))
        for name, value in mapping.items()
        if name not in ("name", "meta")
    )


#: Interpolators built by :func:`redshift_interpolator`, by cosmology and table.
_REDSHIFT_INTERPOLATORS: dict[tuple, CubicSpline] = {}


def redshift_interpolator(
    cosmo: FLRW, z_max: float = 100.0, n_grid: int = 2000
) -> CubicSpline:
    """Get an interpolator of the redshift as a function of comoving distance.

    The comoving distance of ``cosmo`` is tabulated on ``n_grid`` redshifts between 0
    and ``z_max``, evenly spaced in ``log(1 + z)``, and its inverse is interpolated
    with a cubic spline. The interpolator is memoized: it is built once for each
    cosmology (identified by its class and parameters), ``z_max`` and ``n_grid``,
    and reused by later calls.

    Parameters
    ----------
    cosmo : FLRW
        Cosmology of the distance-redshift relation.
    z_max : float, optional
        Highest redshift of the table.
    n_grid : int, optional
        Number of redshifts in the table. The default gives a relative error below
        1e-8 for all redshifts up to ``z_max = 100``.

    Returns
    -------
    scipy.interpolate.CubicSpline
        A function that takes comoving distances in Mpc, as an array, and returns
        their redshifts. It extrapolates beyond ``z_max`` without warning; use
        :func:`comoving_distance_to_redshift` for checked conversions.
    """
    key = (_cosmology_key(cosmo), float(z_max), int(n_grid))
    if key not in _REDSHIFT_INTERPOLATORS:
        redshifts = np.expm1(np.linspace(0.0, np.log1p(z_max), n_grid))
        distances = cosmo.comoving_distance(redshifts).to_value(un.Mpc)
        _REDSHIFT_INTERPOLATORS[key] = CubicSpline(distances, redshifts)
    return _REDSHIFT_INTERPOLATORS[key]


def comoving_distance_to_redshift(
    distances: un.Quantity[un.Mpc],
    cosmo: FLRW,
    z_max: float = 100.0,
    n_grid: int = 2000,
) -> np.ndarray:
    """Get the redshift at each of many comoving distances.

    This inverts :meth:`astropy.cosmology.FLRW.comoving_distance`, like
    :func:`astropy.cosmology.z_at_value`, but much faster for many distances or
    repeated calls, by evaluating the memoized interpolator from
    :func:`redshift_interpolator`.

    Parameters
    ----------
    distances : un.Quantity[un.Mpc]
        Comoving distances, of any shape.
    cosmo : FLRW
        Cosmology of the distance-redshift relation.
    z_max : float, optional
        Highest redshift of the interpolation table; see :func:`redshift_interpolator`.
    n_grid : int, optional
        Number of redshifts in the interpolation table; see
        :func:`redshift_interpolator`.

    Returns
    -------
    np.ndarray
        Redshift at each distance, with the shape of ``distances``.

    Raises
    ------
    ValueError
        If a distance is negative or beyond the comoving distance to ``z_max``.
    """
    d_mpc = np.asarray(distances.to_value(un.Mpc), dtype=float)
    if np.any(d_mpc < 0):
        raise ValueError("Comoving distances cannot be negative.")
    interpolator = redshift_interpolator(cosmo, z_max, n_grid)
    d_max = interpolator.x[-1]
    if np.any(d_mpc > d_max):
        raise ValueError(
            f"Some distances are beyond the comoving distance to z_max={z_max} "
            f"({d_max:.1f} Mpc): pass a larger z_max."
        )
    return interpolator(d_mpc)
