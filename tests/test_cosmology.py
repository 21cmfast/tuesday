"""Tests of the cosmological helper functions."""

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import FlatLambdaCDM, Planck18, z_at_value

from tuesday.core.cosmology import (
    comoving_distance_to_redshift,
    redshift_interpolator,
)

TOL = {"ztol": 1e-12, "maxfun": 1000}


@pytest.mark.parametrize("cosmo", [Planck18, FlatLambdaCDM(H0=70, Om0=0.3)])
def test_matches_z_at_value(cosmo):
    # From the local universe to beyond cosmic dawn.
    distances = cosmo.comoving_distance(np.geomspace(0.01, 99.0, 60))
    expected = np.array(
        [z_at_value(cosmo.comoving_distance, d, **TOL).value for d in distances]
    )

    np.testing.assert_allclose(
        comoving_distance_to_redshift(distances, cosmo), expected, rtol=1e-8
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


def test_interpolator_is_memoized():
    # Repeated calls with the same cosmology reuse the same interpolator, even for
    # different but equal cosmology instances.
    assert redshift_interpolator(Planck18) is redshift_interpolator(Planck18)
    assert redshift_interpolator(
        FlatLambdaCDM(H0=70, Om0=0.3)
    ) is redshift_interpolator(FlatLambdaCDM(H0=70, Om0=0.3))
    assert redshift_interpolator(
        FlatLambdaCDM(H0=70, Om0=0.3)
    ) is not redshift_interpolator(FlatLambdaCDM(H0=70, Om0=0.31))
    assert redshift_interpolator(Planck18) is not redshift_interpolator(
        Planck18, z_max=50.0
    )


def test_interpolator_matches_function():
    distances = np.linspace(6000, 10_000, 20) * un.Mpc
    np.testing.assert_array_equal(
        redshift_interpolator(Planck18)(distances.to_value(un.Mpc)),
        comoving_distance_to_redshift(distances, Planck18),
    )


def test_z_max():
    distance = Planck18.comoving_distance(150.0)
    with pytest.raises(ValueError, match="z_max"):
        comoving_distance_to_redshift(distance, Planck18)
    np.testing.assert_allclose(
        comoving_distance_to_redshift(distance, Planck18, z_max=200.0),
        150.0,
        rtol=1e-8,
    )


def test_negative_distance():
    with pytest.raises(ValueError, match="negative"):
        comoving_distance_to_redshift([-1.0, 100.0] * un.Mpc, Planck18)
