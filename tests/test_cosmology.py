"""Tests of the cosmological helper functions."""

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import FlatLambdaCDM, Planck18, z_at_value

from tuesday.core.cosmology import comoving_distance_to_redshift

TOL = {"ztol": 1e-12, "maxfun": 1000}


@pytest.mark.parametrize("cosmo", [Planck18, FlatLambdaCDM(H0=70, Om0=0.3)])
def test_matches_z_at_value(cosmo):
    # From low redshift to cosmic dawn, as for a wide 21-cm frequency range.
    distances = np.linspace(1500, 11_000, 60) * un.Mpc
    expected = np.array(
        [z_at_value(cosmo.comoving_distance, d, **TOL).value for d in distances]
    )

    np.testing.assert_allclose(
        comoving_distance_to_redshift(distances, cosmo), expected, rtol=1e-6
    )


def test_redshift_differences_are_accurate():
    # The redshift extent of a thin slice is a difference of two nearby redshifts, so
    # the interpolation error must be much smaller than the slice depth.
    centres = np.linspace(6000, 10_000, 20) * un.Mpc
    half_depth = 0.5 * un.Mpc
    dz = comoving_distance_to_redshift(
        centres + half_depth, Planck18
    ) - comoving_distance_to_redshift(centres - half_depth, Planck18)
    expected = np.array(
        [
            z_at_value(Planck18.comoving_distance, d + half_depth, **TOL).value
            - z_at_value(Planck18.comoving_distance, d - half_depth, **TOL).value
            for d in centres
        ]
    )

    np.testing.assert_allclose(dz, expected, rtol=1e-6)


def test_scalar_and_shape():
    z = comoving_distance_to_redshift(8000 * un.Mpc, Planck18)
    assert np.ndim(z) == 0
    np.testing.assert_allclose(
        z, z_at_value(Planck18.comoving_distance, 8000 * un.Mpc, **TOL).value, rtol=1e-6
    )

    distances = np.full((2, 3), 8000.0) * un.Mpc
    assert comoving_distance_to_redshift(distances, Planck18).shape == (2, 3)


def test_converts_units():
    np.testing.assert_allclose(
        comoving_distance_to_redshift(8 * un.Gpc, Planck18),
        comoving_distance_to_redshift(8000 * un.Mpc, Planck18),
        rtol=1e-12,
    )
