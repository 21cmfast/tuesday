import astropy.units as un
import numpy as np
import pytest
from astropy.cosmology.units import littleh

from tuesday.core import (
    CylindricalPS,
    SphericalPS,
)


@pytest.fixture
def ps2():
    """Fixture to create a random power spectrum."""
    return CylindricalPS(
        np.linspace(0, 10, 100).reshape((10, 10)) * un.mK**2,
        kperp=np.linspace(0, 10, 10) / un.m,
        kpar=np.linspace(0, 10, 10) / un.m,
        is_deltasq=True,
    )


@pytest.fixture
def ps():
    """Fixture to create a random power spectrum."""
    return SphericalPS(
        np.linspace(0, 10, 10) * un.mK**2,
        k=np.linspace(0, 10, 10) / un.m,
        is_deltasq=True,
    )


def test_one_ps_per_obj(ps, ps2):
    with np.testing.assert_raises(ValueError):
        SphericalPS(np.append(ps.ps[None, ...], ps.ps[None, ...], axis=0), k=ps.k)
    with np.testing.assert_raises(ValueError):
        CylindricalPS(
            np.append(ps2.ps[None, ...], ps2.ps[None, ...], axis=0),
            kperp=ps2.kperp,
            kpar=ps2.kpar,
            is_deltasq=ps2.is_deltasq,
        )


@pytest.mark.parametrize("unit", [un.Mpc, un.Mpc / littleh])
@pytest.mark.parametrize("delta", [True, False])
def test_ps_correct_units(unit, delta):
    CylindricalPS(
        np.linspace(0, 10, 100).reshape((10, 10)) * un.dimensionless_unscaled
        if delta
        else np.linspace(0, 10, 100).reshape((10, 10))
        * un.dimensionless_unscaled
        * unit**3,
        kperp=np.linspace(0, 10, 10) / unit,
        kpar=np.linspace(0, 10, 10) / un.m,
        is_deltasq=delta,
    )
    CylindricalPS(
        np.linspace(0, 10, 100).reshape((10, 10)) * un.K**2
        if delta
        else np.linspace(0, 10, 100).reshape((10, 10)) * un.K**2 * unit**3,
        kperp=np.linspace(0, 10, 10) / unit,
        kpar=np.linspace(0, 10, 10) / unit,
        is_deltasq=delta,
    )
    SphericalPS(
        np.linspace(0, 10, 10) * un.dimensionless_unscaled
        if delta
        else np.linspace(0, 10, 10) * un.dimensionless_unscaled * unit**3,
        k=np.linspace(0, 10, 10) / un.m,
        is_deltasq=delta,
    )
    SphericalPS(
        np.linspace(0, 10, 10) * un.mK**2
        if delta
        else np.linspace(0, 10, 10) * un.K**2 * unit**3,
        k=np.linspace(0, 10, 10) / unit,
        is_deltasq=delta,
    )


def test_2d_ps_wrong_units(ps2):
    with np.testing.assert_raises(ValueError):
        CylindricalPS(
            ps2.ps.value * un.Mpc, kperp=ps2.kperp, kpar=ps2.kpar
        )  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        CylindricalPS(
            ps2.ps, kperp=ps2.kperp * un.mK, kpar=ps2.kpar
        )  # Wrong units on k
    with np.testing.assert_raises(ValueError):
        CylindricalPS(
            ps2.ps, kperp=ps2.kperp, kpar=ps2.kpar * un.mK
        )  # Wrong units on k
    with np.testing.assert_raises(ValueError):
        CylindricalPS(
            ps2.ps, kperp=ps2.kperp, kpar=ps2.kpar, is_deltasq=False
        )  # correct units but inconsistent with is_deltasq


def test_1d_ps_wrong_units(ps):
    with np.testing.assert_raises(ValueError):
        SphericalPS(ps.ps.value * un.Mpc, k=ps.k)  # Wrong units on PS
    with np.testing.assert_raises(ValueError):
        SphericalPS(ps.ps.value, k=ps.k * un.mK)
    with np.testing.assert_raises(ValueError):
        SphericalPS(
            ps.ps.value * un.m**3, k=ps.k, is_deltasq=True
        )  # correct units but inconsistent with is_deltasq
