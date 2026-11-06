"""Cosmological helper functions."""

import numpy as np
from astropy import units as un
from astropy.cosmology import FLRW, z_at_value
from scipy.interpolate import CubicSpline


def comoving_distance_to_redshift(
    distances: un.Quantity[un.Mpc], cosmo: FLRW, n_grid: int = 3000
) -> np.ndarray:
    """Get the redshift at each of many comoving distances.

    This inverts :meth:`astropy.cosmology.FLRW.comoving_distance`, like
    :func:`astropy.cosmology.z_at_value`, but much faster for many distances: the
    comoving distance is tabulated on a grid of redshifts covering the distances, and
    the inverse relation is interpolated with a cubic spline.

    Parameters
    ----------
    distances : un.Quantity[un.Mpc]
        Comoving distances, of any shape.
    cosmo : FLRW
        Cosmology of the distance-redshift relation.
    n_grid : int, optional
        Number of redshifts in the grid. The default gives a relative error below
        1e-8 between redshift 0.01 and 30.

    Returns
    -------
    np.ndarray
        Redshift at each distance, with the shape of ``distances``.
    """
    d_mpc = np.asarray(distances.to_value(un.Mpc), dtype=float)
    z_lo = z_at_value(cosmo.comoving_distance, d_mpc.min() * un.Mpc).value
    z_hi = z_at_value(cosmo.comoving_distance, d_mpc.max() * un.Mpc).value
    z_grid = np.linspace(max(z_lo - 0.01, 0.0), z_hi + 0.01, n_grid)
    d_grid = cosmo.comoving_distance(z_grid).to_value(un.Mpc)
    return CubicSpline(d_grid, z_grid)(d_mpc)
