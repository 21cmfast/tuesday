"""Test cases for the core/summaries/powerspectra.py module."""

import astropy.units as un
import numpy as np
import pytest

from tuesday.core import calculate_ps_coeval, calculate_ps_lc, cylindrical_to_spherical, bin_kpar, calculate_ps


@pytest.fixture
def test_coeval():
    """Fixture to create a random power spectrum."""
    rng = np.random.default_rng()
    return rng.random((100, 100, 100))


@pytest.fixture
def test_lc():
    rng = np.random.default_rng()
    return rng.random((100, 100, 300))


@pytest.fixture
def test_redshifts():
    return np.logspace(np.log10(5), np.log10(30), 300)

def test_calculate_ps_errors():
    with np.testing.assert_raises(ValueError):
        calculate_ps(
            test_lc, # No unit
            200 * un.Mpc,
            test_redshifts,
            ps_redshifts=3.0,
            calc_1d=True,
        )
    with np.testing.assert_raises(ValueError):
        calculate_ps(
            test_lc * un.mK, 
            200, # No unit
            test_redshifts,
            ps_redshifts=3.0,
            calc_1d=True,
        )
    with np.testing.assert_raises(ValueError):
        calculate_ps(
            test_lc * un.mK, 
            200 * un.Mpc,
            test_redshifts,
            ps_redshifts=3.0,
            calc_1d=False,
            calc_2d=False,
        )
    def prefactor(freq):
        return freq
    calculate_ps(
        test_lc * un.mK, 
        200 * un.Mpc,
        test_redshifts,
        ps_redshifts=3.0,
        calc_1d=False,
        calc_2d=True,
        prefactor_fnc=prefactor,
    )

@pytest.mark.parametrize("log_bins", [True, False])
def test_calculate_ps_lc(log_bins, test_lc, test_redshifts):
    calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        200 * un.Mpc,
        test_redshifts,
        ps_redshifts=[6.0],
        calc_2d=False,
        log_bins=log_bins,
    )

    calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        200 * un.Mpc,
        test_redshifts,
        ps_redshifts=6.8,
        calc_1d=False,
        interp=True,
        mu_min=0.5,
        log_bins=log_bins,
        transform_ps2d=bin_kpar(bins_kpar=10, log_kpar=True, interp_kpar=True),
    )

    def transform1d(ps):
        return ps
    calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        200 * un.Mpc,
        test_redshifts,
        mu_min=0.5,
        log_bins=log_bins,
        get_variance=True,
        transform_ps2d=bin_kpar(bins_kpar=None, 
                                log_kpar=False, 
                                interp_kpar=False, 
                                crop_kpar=(0,3), 
                                crop_kperp=(0,8)),
        transform_ps1d=transform1d,
    )


def test_calculate_ps_coeval(test_coeval):
    with np.testing.assert_raises(TypeError):
        calculate_ps_coeval(
            test_coeval * un.dimensionless_unscaled,
            box_length=200 * un.Mpc,
            ps_redshifts=6.8,
            calc_1d=False,
            interp=True,
            mu_min=0.5,
        )
    def transform1d(ps):
        return ps
    calculate_ps_coeval(
        test_coeval * un.dimensionless_unscaled,
        box_length=200 * un.Mpc,
        calc_1d=False,
        interp=True,
        mu_min=0.5,
        transform_ps2d=bin_kpar(bins_kpar=np.array([0.1,0.5,1])/un.Mpc, 
                                log_kpar=True, 
                                interp_kpar=True, 
                                crop_kpar=(0,3), 
                                crop_kperp=(0,8)),
        transform_ps1d=transform1d,
    )

    calculate_ps_coeval(
        test_coeval * un.dimensionless_unscaled,
        box_length=200 * un.Mpc,
        mu_min=0.5,
    )


def test_calculate_ps_corner_cases(test_lc, test_redshifts):
    with np.testing.assert_raises(ValueError):
        calculate_ps_lc(
            test_lc * un.dimensionless_unscaled,
            200 * un.Mpc,
            test_redshifts,
            ps_redshifts=3.0,
            calc_1d=True,
        )
    calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        200 * un.Mpc,
        test_redshifts,
        calc_1d=True,
        interp=True,
        mu_min=0.5,
        delta=True,
    )

    calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        200 * un.Mpc,
        test_redshifts,
        calc_1d=True,
        interp=True,
        mu_min=0.5,
        delta=False,
    )

    with np.testing.assert_raises(ValueError):
        calculate_ps_lc(
            test_lc * un.dimensionless_unscaled,
            200 * un.Mpc,
            test_redshifts,
            ps_redshifts=[50.0],  # outside test_redshifts
            calc_1d=True,
            get_variance=True,
        )


def test_calculate_ps_w_var(test_lc, test_redshifts):
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0]

    out = calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        200 * un.Mpc,
        test_redshifts,
        ps_redshifts=zs,
        calc_2d=False,
        calc_1d=True,
        get_variance=True,
    )
    assert out["ps_1d"]["z = 6.0"].variance is not None
    out = calculate_ps_lc(
        test_lc * un.dimensionless_unscaled,
        200 * un.Mpc,
        test_redshifts,
        ps_redshifts=zs,
        calc_2d=True,
        calc_1d=True,
        get_variance=True,
    )
    assert out["ps_2d"]["z = 6.0"].variance is not None
    assert out["ps_1d"]["z = 6.0"].variance is not None

    with np.testing.assert_raises(NotImplementedError):
        calculate_ps_lc(
            test_lc * un.dimensionless_unscaled,
            200 * un.Mpc,
            test_redshifts,
            ps_redshifts=zs,
            calc_1d=True,
            get_variance=True,
            interp="linear",
        )


def test_ps_avg():
    rng = np.random.default_rng()
    ps_2d = rng.random((32, 32))
    x = np.linspace(0, 1, 32)
    ps, k, sws = cylindrical_to_spherical(ps_2d, x, x, nbins=16)
    assert ps.shape == (16,)
    assert k.shape == (16,)
    assert sws.shape == (16,)
    kpar_mesh, kperp_mesh = np.meshgrid(x, x)
    theta = np.arctan(kperp_mesh / kpar_mesh)
    mu_mesh = np.cos(theta)
    mask = mu_mesh >= 0.9
    ps_2d[mask] = 1000
    ps, k, sws = cylindrical_to_spherical(
        ps_2d, x, x, nbins=32, interp=True, mu_min=0.98
    )
    assert np.nanmean(ps[-20:]) == 1000.0
